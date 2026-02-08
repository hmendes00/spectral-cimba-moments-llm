"""
Fejér Mixture Moment Attention (FMMA) — core attention module.

s_i(t) = Σ_m φ_m(k_i) · p_m(θ_i − τ | q_t)      (nonneg scores)
w_i(t) = s_i(t) / Σ_j s_j(t)                      (normalised)
y_t    = Σ_i w_i(t) · v_i                          (output)

Memory: O(r · S · d) constant in sequence length.
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import FMMACache
from .features import NonnegativeKeyFeatures, QueryMomentNetwork


class FMMAAttention(nn.Module):
    """
    Fejér Mixture Moment Attention (per FMMA paper Theorem 1).

    Implementation note — addresses ALL items in PLAN.md:
      0.1  Causality: cached inference streams token-by-token (update then query each t).
      0.2  Phase mapping: always θ = 2π·pos / max_seq_len (training AND inference).
      0.3  Δ sign: training uses θ_i − τ (matches paper / cache path).
      1.2  PSD enforcement: autocorrelation parameterization (hard PSD Toeplitz).
      1.3  Imaginary diagnostics: debug assertion on imaginary magnitude.
      2.1  HF DynamicCache: update() returns zero-sized tensors; get_seq_length() correct.
      2.2  No phase↔position round-trip: integer position_ids passed directly to cache.
      2.3  Padding/attention_mask applied: masked tokens excluded from moments & scores.
      3.2  Linear normalisation with guaranteed ≥0 scores (PSD + nonneg features).
      3.3  Density nonnegativity: PSD autocorrelation hard-enforces nonneg densities.
           No clamp (train/inference parity). Monitored via _last_min_density.
      4.1  Module replacement targets known Qwen/Llama/Mistral attention classes.
      4.2  GQA uses repeat_interleave for correct head mapping.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        num_channels: int = 4,
        bandwidth: int = 64,
        max_seq_len: int = 8192,
        layer_idx: int = 0,
        rope_theta: float = 10000.0,
        num_kv_heads: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Fix 4.2: store for GQA
        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx

        # Q, K, V, O projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # FMMA-specific components
        self.key_features = NonnegativeKeyFeatures(self.head_dim, num_channels)
        self.query_moments = QueryMomentNetwork(
            self.head_dim, num_channels, bandwidth, enforce_psd=True
        )

        # Precompute Fejér weights & k indices as buffers
        k_indices = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
        self.register_buffer(
            "fejer_weights", 1.0 - torch.abs(k_indices) / (bandwidth + 1)
        )
        self.register_buffer("k_indices", k_indices)

        # Gradient checkpointing flag (set externally to save memory)
        self.gradient_checkpointing: bool = False

        # Diagnostics (not saved, no grad)
        self._last_min_density: Optional[float] = None
        self._last_query_moments: Optional[torch.Tensor] = None

    # Fix 0.2: Always use max_seq_len for phase mapping (training AND inference)
    def _get_phases(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert integer position_ids to circular phases.
        θ = 2π · position / max_seq_len   (ALWAYS, per PLAN.md 0.2)
        """
        return 2 * math.pi * position_ids.float() / self.max_seq_len

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV tensors from num_kv_heads to num_heads (GQA).
        Fix 4.2: uses repeat_interleave for correct head mapping.
        """
        if self.num_kv_heads == self.num_heads:
            return x
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads}) for GQA expansion"
        )
        repeat_factor = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat_factor, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[FMMACache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """Forward pass with optional FMMA caching."""
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Project Q, K, V
        queries = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [B, H, T, D]

        keys = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [B, Hkv, T, D]

        values = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [B, Hkv, T, D]

        # Expand KV heads for GQA (Fix 4.2)
        keys = self._expand_kv_heads(keys)      # [B, H, T, D]
        values = self._expand_kv_heads(values)   # [B, H, T, D]

        # Get positions
        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
            else:
                seq_offset = (
                    past_key_value.get_seq_length(self.layer_idx)
                    if past_key_value
                    else 0
                )
                position_ids = torch.arange(
                    seq_offset, seq_offset + seq_len, device=device
                ).unsqueeze(0).expand(batch_size, -1)

        # Build a simple 1/0 mask from HF attention_mask for FMMA paths.
        # HF attention_mask can be:
        #   - [B, T]       int/long with 1=valid, 0=pad  (from tokenizer)
        #   - [B, 1, T, T] bool with True=attend, False=mask (HF causal pipeline)
        #   - [B, 1, T, T] float with 0=attend, -inf=mask   (older HF versions)
        simple_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                simple_mask = attention_mask  # [B, T] 1/0
            elif attention_mask.dim() == 4:
                # Extract per-key validity from the last query row.
                last_row = attention_mask[:, 0, -1, :]  # [B, T]
                if last_row.dtype == torch.bool:
                    simple_mask = last_row.long()
                else:
                    # Float mask: 0 = attend, large negative = mask
                    simple_mask = (last_row > -1e4).long()

        # Fix 0.2: single canonical phase mapping
        phases = self._get_phases(position_ids)  # [B, T]

        if past_key_value is not None and isinstance(past_key_value, FMMACache):
            output = self._fmma_forward(
                queries, keys, values, position_ids, phases,
                past_key_value, use_cache, simple_mask,
            )
        else:
            output = self._fmma_training_forward(
                queries, keys, values, phases, simple_mask,
            )

        # Output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, None

    def _fmma_forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,  # [B, T] integer  (Fix 2.2)
        phases: torch.Tensor,  # [B, T]
        cache: FMMACache,
        use_cache: bool,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        FMMA cached inference.

        Fix 0.1 (PLAN.md): Streams token-by-token — update cache with token t,
        then compute output for token t — so causality is respected.

        Fix 2.2: Passes integer position_ids to cache (no phase↔position
        round-trip).

        Fix 2.3: Forwards attention_mask to cache so padding tokens are
        excluded from moments.
        """
        batch_size, num_heads, seq_len, head_dim = queries.shape
        device = queries.device

        key_feats = self.key_features(keys)  # [B, H, T, r]
        q_moments = self.query_moments(queries)  # [B, H, T, r, 2S+1]

        # Exact causal loop: for each token t, update cache (so cache
        # contains tokens 0..t), then query from that state.  This
        # guarantees output_t depends only on tokens ≤ t.
        #
        # For autoregressive generation seq_len is typically 1 (one new
        # token), so the loop body executes once.  For prefill the
        # training path (_fmma_training_forward) handles efficiency.
        outputs = []
        for t in range(seq_len):
            # --- update cache with token t ---
            kf_t = key_feats[:, :, t : t + 1, :]  # [B, H, 1, r]
            v_t = values[:, :, t : t + 1, :]       # [B, H, 1, D]
            pos_t = position_ids[:, t : t + 1]      # [B, 1]

            mask_t = None
            if attention_mask is not None:
                mask_t = attention_mask[:, t : t + 1]  # [B, 1]

            cache.update_moments(
                layer_idx=self.layer_idx,
                key_features=kf_t,
                values=v_t,
                position_ids=pos_t,
                attention_mask=mask_t,
            )

            # --- query from cache (sees tokens 0..t) ---
            q_mom_t = q_moments[:, :, t]  # [B, H, r, 2S+1]
            phase_t = phases[:, t].unsqueeze(1).expand(-1, num_heads)

            out_t = cache.compute_attention(
                layer_idx=self.layer_idx,
                query_moments=q_mom_t,
                query_phases=phase_t,
            )
            outputs.append(out_t)

        return torch.stack(outputs, dim=2)  # [B, H, T, D]

    def _fmma_training_forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        phases: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        FMMA forward for training (no cache, explicit attention computation).

        Fixes applied:
          0.2  Phase mapping: phases already computed with max_seq_len.
          0.3  Δ sign: delta = θ_i − τ  (matches paper §3.4 and cache path).
          2.3  Padding mask: zeros attention scores for padded positions.
          3.3  Density nonnegativity: PSD autocorrelation hard-enforces nonneg;
               no clamp (train/inference parity). Monitored via _last_min_density.

        Memory optimizations:
          - Query moments computed per chunk (avoid 339 MB full tensor in autograd)
          - cos/sin basis reuse via del to free eagerly
          - Optional gradient checkpointing per chunk
        """
        batch_size, num_heads, seq_len, head_dim = queries.shape
        device = queries.device
        dtype = queries.dtype
        S = self.bandwidth
        r = self.num_channels

        # Nonneg key features: φ_m(k_i) ≥ 0
        key_feats = self.key_features(keys)  # [B, H, L, r]

        # Fejér weights
        c_k = self.fejer_weights.to(device)  # [2S+1]
        k_vals = torch.arange(1, S + 1, device=device, dtype=dtype)

        chunk_size = min(32, seq_len)
        outputs = []

        # Store moments for regularization — but only subsample to keep memory low
        # We'll collect from the first chunk only (representative)
        self._last_query_moments = None

        for q_start in range(0, seq_len, chunk_size):
            q_end = min(q_start + chunk_size, seq_len)

            if self.gradient_checkpointing and torch.is_grad_enabled():
                output_chunk = torch.utils.checkpoint.checkpoint(
                    self._training_chunk,
                    queries, keys, values, key_feats, phases,
                    attention_mask, c_k, k_vals,
                    q_start, q_end, seq_len,
                    use_reentrant=False,
                )
            else:
                output_chunk = self._training_chunk(
                    queries, keys, values, key_feats, phases,
                    attention_mask, c_k, k_vals,
                    q_start, q_end, seq_len,
                )
            outputs.append(output_chunk)

        return torch.cat(outputs, dim=2).to(dtype)

    def _training_chunk(
        self,
        queries: torch.Tensor,      # [B, H, T, D]
        keys: torch.Tensor,          # [B, H, T, D] (unused here, needed for checkpoint signature)
        values: torch.Tensor,        # [B, H, T, D]
        key_feats: torch.Tensor,     # [B, H, T, r]
        phases: torch.Tensor,        # [B, T]
        attention_mask: Optional[torch.Tensor],
        c_k: torch.Tensor,           # [2S+1]
        k_vals: torch.Tensor,        # [S]
        q_start: int,
        q_end: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Process one query chunk — isolated for gradient checkpointing."""
        batch_size, num_heads = queries.shape[:2]
        device = queries.device
        S = self.bandwidth

        # Compute query moments ONLY for this chunk
        q_chunk = queries[:, :, q_start:q_end]    # [B, H, chunk, D]
        q_moments_chunk = self.query_moments(q_chunk)  # [B, H, chunk, r, 2S+1]

        # Store moments from first chunk for PSD/bandwidth regularization
        # Keep gradients attached so penalty losses can backprop
        if q_start == 0:
            self._last_query_moments = q_moments_chunk

        # Fejér-weight and split into DC + harmonics
        weighted = q_moments_chunk * c_k.view(1, 1, 1, 1, -1)
        del q_moments_chunk  # free full complex tensor

        mu_0_real = weighted[..., S].real              # [B, H, chunk, r]
        mu_pos_real = weighted[..., S + 1:].real       # [B, H, chunk, r, S]
        mu_pos_imag = weighted[..., S + 1:].imag       # [B, H, chunk, r, S]
        del weighted

        phases_q = phases[:, q_start:q_end]            # [B, chunk]

        # Fix 0.3: delta = θ_i − τ  (paper convention, matches cache)
        delta = phases.unsqueeze(1) - phases_q.unsqueeze(2)  # [B, chunk, L]

        k_delta = delta.unsqueeze(-1) * k_vals         # [B, chunk, L, S]
        del delta
        cos_basis = torch.cos(k_delta)
        sin_basis = torch.sin(k_delta)
        del k_delta

        # p_m(Δ|q) = μ_0 + 2 Σ_k [Re(μ_k)cos(kΔ) - Im(μ_k)sin(kΔ)]
        cos_contrib = torch.einsum("bhqrs,bqks->bhqkr", mu_pos_real, cos_basis)
        sin_contrib = torch.einsum("bhqrs,bqks->bhqkr", mu_pos_imag, sin_basis)
        del cos_basis, sin_basis, mu_pos_real, mu_pos_imag

        densities = mu_0_real.unsqueeze(3) + 2.0 * (cos_contrib - sin_contrib)
        del cos_contrib, sin_contrib
        # [B, H, chunk, L, r]

        # Monitor density nonnegativity.  PSD autocorrelation guarantees
        # p_m(Δ) ≥ 0, so negatives indicate a numerical problem.  We log
        # rather than clamp so training and inference see the *same*
        # operator (no silent distribution shift).
        #
        # Only run the monitoring kernel during eval/generate — during
        # training the result is never read, so the reduction is wasted work.
        if not torch.is_grad_enabled():
            with torch.no_grad():
                chunk_min_val = densities.min().item()
                if self._last_min_density is None:
                    self._last_min_density = chunk_min_val
                else:
                    self._last_min_density = min(self._last_min_density, chunk_min_val)
                if chunk_min_val < -1e-4:
                    warnings.warn(
                        f"FMMA density went negative (min={chunk_min_val:.2e}). "
                        f"PSD autocorrelation may have numerical issues."
                    )

        # Attention scores: Σ_m φ_m(k_i) · p_m(Δ|q)
        attn_scores = (key_feats.unsqueeze(2) * densities).sum(dim=-1)
        del densities
        # [B, H, chunk, L]

        # Causal mask
        q_positions = torch.arange(q_start, q_end, device=device)
        k_positions = torch.arange(seq_len, device=device)
        causal_mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), 0.0
        )

        # Fix 2.3: Apply padding mask
        if attention_mask is not None:
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(pad_mask, 0.0)

        # Normalise (linear normalisation — valid because scores ≥ 0)
        attn_weights = attn_scores / (
            attn_scores.sum(dim=-1, keepdim=True) + 1e-8
        )

        return torch.matmul(attn_weights, values.float())

    def _standard_forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard softmax attention fallback (for debugging / comparison)."""
        scale = 1.0 / math.sqrt(self.head_dim)

        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        seq_len = queries.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=queries.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, values)
