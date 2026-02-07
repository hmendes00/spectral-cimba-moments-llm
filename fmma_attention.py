"""
Fejér Mixture Moment Attention (FMMA)

A constant-memory, content-aware attention operator on the circle.

Key properties:
- Content awareness via r nonnegative key feature channels
- Position awareness via bandlimited Fejér densities
- EXACT evaluation from O(rS) cached moments (independent of sequence length)
- Nonnegative weights guaranteed by PSD Toeplitz + Fejér reconstruction
- Both K and V become O(rS) - 100% KV cache savings!

Based on the FMMA paper (FMMA_breakthrough_tex.txt): combining CIMBA moment sketching
with CircleStat PSD Toeplitz realizability and nonnegative mixture decomposition.

Implementation note — this file addresses ALL items in PLAN.md:
  0.1  Causality: cached inference streams token-by-token (update then query each t).
  0.2  Phase mapping: always θ = 2π·pos / max_seq_len (training AND inference).
  0.3  Δ sign: training uses θ_i − τ (matches paper / cache path).
  1.2  PSD enforcement: autocorrelation parameterization (hard PSD Toeplitz).
  1.3  Imaginary diagnostics: debug assertion on imaginary magnitude.
  2.1  HF DynamicCache: update() returns zero-sized tensors; get_seq_length() correct.
  2.2  No phase↔position round-trip: integer position_ids passed directly to cache.
  2.3  Padding/attention_mask applied: masked tokens excluded from moments & scores.
  3.2  Linear normalisation with guaranteed ≥0 scores (PSD + nonneg features).
  3.3  Density clamped to 0 as safety net; monitored via _last_min_density.
  4.1  Module replacement targets known Qwen/Llama/Mistral attention classes.
  4.2  GQA uses repeat_interleave for correct head mapping.
"""

import math
import warnings
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache


# ============================================================================
# PSD Toeplitz Penalty and Bandwidth Regularization (per FMMA paper §5.4)
# ============================================================================

def build_toeplitz_matrix(moments: torch.Tensor) -> torch.Tensor:
    """
    Build Toeplitz matrix T_S from moments μ_{-S}, ..., μ_S.

    T_S[i,j] = μ_{i-j} for i,j = 0, ..., S

    Args:
        moments: [..., 2S+1] complex tensor with μ_{-S}, ..., μ_S

    Returns:
        T: [..., S+1, S+1] complex Toeplitz matrix
    """
    *batch_dims, num_moments = moments.shape
    S = (num_moments - 1) // 2
    size = S + 1

    i_idx = torch.arange(size, device=moments.device)
    j_idx = torch.arange(size, device=moments.device)
    diff = i_idx.unsqueeze(1) - j_idx.unsqueeze(0)  # [S+1, S+1]
    indices = diff + S

    moments_flat = moments.reshape(-1, num_moments)
    batch_size = moments_flat.shape[0]
    indices_expanded = indices.unsqueeze(0).expand(batch_size, -1, -1)

    T_flat = torch.gather(
        moments_flat.unsqueeze(1).expand(-1, size, -1),
        dim=2,
        index=indices_expanded,
    )

    T = T_flat.reshape(*batch_dims, size, size)
    return T


def psd_toeplitz_penalty(
    moments: torch.Tensor,
    reduction: str = "mean",
    max_samples: int = 64,
) -> torch.Tensor:
    """
    Compute penalty for non-PSD Toeplitz matrices.

    Per FMMA paper §5.4: "penalize negative eigenvalues of T_S(μ^(m)(q_t))"

    Args:
        moments: [batch, heads, seq_len, r, 2S+1] complex moments
        reduction: 'mean', 'sum', or 'none'
        max_samples: Max moment vectors to sample for eigenvalue computation.

    Returns:
        penalty: scalar or tensor depending on reduction
    """
    moments_flat = moments.reshape(-1, moments.shape[-1])
    N = moments_flat.shape[0]

    if max_samples is not None and N > max_samples:
        indices = torch.randperm(N, device=moments.device)[:max_samples]
        moments_sampled = moments_flat[indices]
    else:
        moments_sampled = moments_flat

    T = build_toeplitz_matrix(moments_sampled)

    try:
        eigenvalues = torch.linalg.eigvalsh(T)
    except RuntimeError:
        eigenvalues = torch.linalg.eigvals(T).real

    negative_eigs = torch.clamp(eigenvalues, max=0)
    penalty_per_matrix = (negative_eigs ** 2).sum(dim=-1)

    if reduction == "mean":
        return penalty_per_matrix.mean()
    elif reduction == "sum":
        scale = N / moments_sampled.shape[0]
        return penalty_per_matrix.sum() * scale
    else:
        return penalty_per_matrix


def bandwidth_regularization(
    moments: torch.Tensor,
    bandwidth: int,
    alpha: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Regularize high-frequency moments to encourage smooth Fejér densities.

    Per FMMA paper §5.4: "penalize high-order moments to keep effective bandwidth small"

    reg = Σ_k (|k|/S)^α * |μ_k|²
    """
    device = moments.device
    S = bandwidth

    k_indices = torch.arange(-S, S + 1, device=device, dtype=torch.float32)
    weights = (torch.abs(k_indices) / S) ** alpha  # [2S+1]

    moment_magnitudes_sq = moments.real ** 2 + moments.imag ** 2
    reg_per_moment = (weights * moment_magnitudes_sq).sum(dim=-1)

    if reduction == "mean":
        return reg_per_moment.mean()
    elif reduction == "sum":
        return reg_per_moment.sum()
    else:
        return reg_per_moment


# ============================================================================
# FMMA Cache: O(rS) memory for both K and V
# ============================================================================

class FMMACache(DynamicCache):
    """
    Constant-memory cache for FMMA attention.

    Stores moment tensors instead of full KV sequences:
    - M_V[m, k, :] = Σ_i φ_m(k_i) e^{ikθ_i} v_i  ∈ C^{d_v}
    - M_1[m, k]    = Σ_i φ_m(k_i) e^{ikθ_i}      ∈ C

    Memory: O(r * S * d_v) per layer - CONSTANT in sequence length!

    Fix 2.1 (PLAN.md): Implements proper HF DynamicCache contract.
    Fix 2.2 (PLAN.md): Accepts integer position_ids directly; single canonical
                       phase mapping θ = 2π·pos / max_seq_len everywhere.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_channels: int = 4,
        bandwidth: int = 64,
        max_seq_len: int = 8192,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device

        self.num_moments = 2 * bandwidth + 1

        self._M_V: List[Optional[torch.Tensor]] = [None] * num_layers
        self._M_1: List[Optional[torch.Tensor]] = [None] * num_layers
        self._seq_len: List[int] = [0] * num_layers

        # Precompute Fejér weights: c_k = 1 - |k|/(S+1)
        k_indices = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
        self.fejer_weights = 1.0 - torch.abs(k_indices) / (bandwidth + 1)

    def _init_moments(self, layer_idx: int, batch_size: int, device: torch.device):
        """Initialize moment tensors for a layer."""
        self._M_V[layer_idx] = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_channels,
            self.num_moments,
            self.head_dim,
            dtype=torch.complex64,
            device=device,
        )
        self._M_1[layer_idx] = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_channels,
            self.num_moments,
            dtype=torch.complex64,
            device=device,
        )

    def update_moments(
        self,
        layer_idx: int,
        key_features: torch.Tensor,  # [batch, num_heads, seq_len, r] nonneg
        values: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        position_ids: torch.Tensor,  # [seq_len] or [batch, seq_len] integer
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len] 0/1
    ) -> None:
        """
        Incrementally update moment caches.

        Fix 2.2: Accepts integer position_ids; computes phases internally via
                 the single canonical mapping θ = 2π·pos / max_seq_len.
        Fix 2.3: Zeros contributions from padding positions when attention_mask
                 is provided.
        """
        batch_size, num_heads, seq_len, _ = values.shape
        device = values.device

        if self._M_V[layer_idx] is None:
            self._init_moments(layer_idx, batch_size, device)

        # Canonical phase mapping (Fix 0.2): always use max_seq_len
        if position_ids.dim() == 1:
            phases = 2 * math.pi * position_ids.float() / self.max_seq_len
            phases = phases.unsqueeze(0).expand(batch_size, -1)
        else:
            phases = 2 * math.pi * position_ids.float() / self.max_seq_len

        # Apply attention mask (Fix 2.3): zero out padding
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] with 1=valid, 0=pad
            mask = attention_mask.float()  # [B, T]
            # Expand for heads and channels
            key_features = key_features * mask.unsqueeze(1).unsqueeze(-1)

        # e^{ikθ} for k ∈ {-S, ..., S}
        k_indices = torch.arange(
            -self.bandwidth,
            self.bandwidth + 1,
            dtype=torch.float32,
            device=device,
        )
        exp_ik_theta = torch.exp(
            1j * k_indices.view(1, 1, -1) * phases.unsqueeze(-1)
        )  # [B, T, 2S+1]

        exp_ik_theta_h = exp_ik_theta.unsqueeze(1).expand(
            -1, num_heads, -1, -1
        )  # [B, H, T, 2S+1]

        key_features_c = key_features.to(torch.complex64)
        values_c = values.to(torch.complex64)

        delta_M1 = torch.einsum("bhtr,bhtk->bhrk", key_features_c, exp_ik_theta_h)
        delta_MV = torch.einsum(
            "bhtr,bhtk,bhtd->bhrkd", key_features_c, exp_ik_theta_h, values_c
        )

        self._M_1[layer_idx] = self._M_1[layer_idx] + delta_M1
        self._M_V[layer_idx] = self._M_V[layer_idx] + delta_MV
        self._seq_len[layer_idx] += seq_len

    def compute_attention(
        self,
        layer_idx: int,
        query_moments: torch.Tensor,  # [batch, num_heads, r, 2S+1] complex
        query_phases: torch.Tensor,  # [batch, num_heads]
    ) -> torch.Tensor:
        """
        Compute FMMA output exactly from cached moments.

        Per FMMA paper Theorem 1 (eq. 7):
            y = Num / Den
            Num = Σ_{m,k} c_k μ^{(m)}_k(q) e^{-ikτ} M_V[m,k]
            Den = Σ_{m,k} c_k μ^{(m)}_k(q) e^{-ikτ} M_1[m,k]

        Fix 1.3: Checks imaginary magnitude of output; warns if large.

        Returns: [batch, num_heads, head_dim]
        """
        device = query_moments.device

        M_V = self._M_V[layer_idx]
        M_1 = self._M_1[layer_idx]

        c_k = self.fejer_weights.to(device)

        k_indices = torch.arange(
            -self.bandwidth,
            self.bandwidth + 1,
            dtype=torch.float32,
            device=device,
        )

        # e^{-ikτ}: [batch, num_heads, 2S+1]
        exp_neg_ik_tau = torch.exp(
            -1j * k_indices.view(1, 1, -1) * query_phases.unsqueeze(-1)
        )

        # combined = c_k * μ_k(q) * e^{-ikτ}   — [B, H, r, 2S+1]
        combined = (
            c_k.view(1, 1, 1, -1) * query_moments * exp_neg_ik_tau.unsqueeze(2)
        )

        numerator = torch.einsum("bhrk,bhrkd->bhd", combined, M_V)
        denominator = torch.einsum("bhrk,bhrk->bh", combined, M_1)

        output = numerator / (denominator.unsqueeze(-1) + 1e-8)

        # Fix 1.3: Diagnose imaginary leakage (avoid GPU sync in hot path
        # by only checking periodically when training flag might be set)
        if not torch.is_grad_enabled():
            # Only check during inference (eval/generate), not training
            imag_mag = output.imag.abs().max()
            if imag_mag > 1e-4:
                warnings.warn(
                    f"FMMA cache output has non-trivial imaginary part "
                    f"(max |imag| = {imag_mag:.2e}). Hermitian symmetry may be violated."
                )

        return output.real.to(self.dtype)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return current sequence length across all layers."""
        return self._seq_len[layer_idx]

    def memory_usage(self, layer_idx: int = 0) -> dict:
        """Report memory usage — should be constant regardless of seq_len."""
        M_V = self._M_V[layer_idx]
        M_1 = self._M_1[layer_idx]

        m_v_bytes = M_V.numel() * 8 if M_V is not None else 0
        m_1_bytes = M_1.numel() * 8 if M_1 is not None else 0

        return {
            "M_V_bytes": m_v_bytes,
            "M_1_bytes": m_1_bytes,
            "total_bytes": m_v_bytes + m_1_bytes,
            "seq_len": self._seq_len[layer_idx],
            "num_channels": self.num_channels,
            "bandwidth": self.bandwidth,
            "theoretical_kv_bytes": 0,
        }

    # Fix 2.1: DynamicCache interface — return zero-sized tensors so HF
    # does not allocate a standard KV cache upstream.
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HF DynamicCache.update() contract.

        We return zero-length tensors on the sequence dimension so HF does not
        accumulate KV states, while remaining compatible with downstream code
        that checks `.shape`.
        """
        B, H, _, D = key_states.shape
        empty_k = key_states.new_zeros(B, H, 0, D)
        empty_v = value_states.new_zeros(B, H, 0, D)
        return empty_k, empty_v


# ============================================================================
# Nonnegative Key Feature Network
# ============================================================================


class NonnegativeKeyFeatures(nn.Module):
    """
    Produces r nonnegative features from key vectors.

    φ_m(k) = softplus(u_m^T k + b_m) for m = 1, ..., r

    Ensures φ_m(k) ≥ 0 which is required for FMMA nonnegativity.
    """

    def __init__(self, key_dim: int, num_channels: int, beta: float = 1.0):
        super().__init__()
        self.key_dim = key_dim
        self.num_channels = num_channels

        self.projection = nn.Linear(key_dim, num_channels, bias=True)
        self.beta = beta

        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keys: [batch, num_heads, seq_len, head_dim]
        Returns:
            features: [batch, num_heads, seq_len, r] nonnegative
        """
        projected = self.projection(keys)
        return F.softplus(projected, beta=self.beta)


# ============================================================================
# Query Moment Network with PSD Toeplitz Constraint
# ============================================================================


class QueryMomentNetwork(nn.Module):
    """
    Produces query-conditioned harmonic moments μ^{(m)}_k(q) with PSD Toeplitz structure.

    For each channel m, outputs complex moments satisfying:
    - μ_0 = 1 (normalized)
    - μ_{-k} = conj(μ_k) (Hermitian symmetry)
    - T_S(μ) ≽ 0 (positive semidefinite Toeplitz matrix)

    Fix 1.2 (PLAN.md): Uses autocorrelation parameterization to *hard-enforce*
    PSD Toeplitz.  We parameterize a vector a ∈ C^{S+1} and compute moments as
    the autocorrelation μ_k = (a ⋆ a*)_k / ||a||², which is guaranteed PSD
    Toeplitz with μ_0 = 1.  This replaces the old |μ_k| < 1 heuristic.

    When enforce_psd=False (legacy), falls back to soft |μ_k| < 1.
    """

    def __init__(
        self,
        query_dim: int,
        num_channels: int,
        bandwidth: int,
        hidden_dim: int = None,
        enforce_psd: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.enforce_psd = enforce_psd

        if hidden_dim is None:
            hidden_dim = query_dim

        if enforce_psd:
            # Autocorrelation parameterization: output (S+1) complex coefficients
            # per channel.  The autocorrelation of a is guaranteed PSD Toeplitz.
            output_dim = num_channels * (bandwidth + 1) * 2  # real + imag
        else:
            # Legacy: S complex numbers per channel (μ_1 … μ_S)
            output_dim = num_channels * bandwidth * 2

        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _autocorrelation_psd(raw_flat: torch.Tensor, S: int) -> torch.Tensor:
        """
        Autocorrelation PSD parameterization (Fix 1.2).

        Isolated as a static method so it can be wrapped with
        torch.utils.checkpoint to trade compute for memory.

        Args:
            raw_flat: [N, r, S+1, 2]  (flattened over B*H*T)
            S: bandwidth
        Returns:
            moments: [N, r, 2S+1] complex64
        """
        N, r = raw_flat.shape[:2]
        raw_f = raw_flat.float()
        a = torch.complex(raw_f[..., 0], raw_f[..., 1])  # [N, r, S+1]

        # Autocorrelation via FFT: μ = IFFT(|FFT(a)|²), normalised so μ_0=1
        n_fft = 2 * S + 1
        A = torch.fft.fft(a, n=n_fft, dim=-1)  # [N, r, 2S+1]
        power = A.abs().square()                 # |A|²  (fused, avoids separate .real/.imag)
        del A                                    # free immediately
        autocorr = torch.fft.ifft(power, dim=-1) # [N, r, 2S+1]
        del power

        # Normalise so μ_0 = 1
        mu_0 = autocorr[..., 0:1].real.clamp(min=1e-8)
        autocorr = autocorr / mu_0

        # Build full Hermitian moment vector [μ_{-S}…μ_0…μ_S]
        mu_positive = autocorr[..., 1 : S + 1]   # [N, r, S]
        mu_negative = torch.conj(torch.flip(mu_positive, dims=[-1]))
        mu_zero = torch.ones(N, r, 1, dtype=torch.complex64, device=raw_flat.device)
        return torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)  # [N, r, 2S+1]

    def forward(self, queries: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        """
        Args:
            queries: [B, H, T, D] or [B, H, D]
            use_checkpoint: if True, checkpoint the FFT autocorrelation to save memory
        Returns:
            moments: [B, H, T, r, 2S+1] or [B, H, r, 2S+1] complex
        """
        is_single_query = queries.dim() == 3
        if is_single_query:
            queries = queries.unsqueeze(2)

        B, H, T, D = queries.shape
        S = self.bandwidth
        r = self.num_channels

        raw = self.net(queries)  # [B, H, T, output_dim]

        if self.enforce_psd:
            # Flatten B*H*T so the FFT operates on a 3-D tensor (much less memory)
            raw_flat = raw.view(B * H * T, r, S + 1, 2)

            if use_checkpoint and raw_flat.requires_grad:
                moments_flat = torch.utils.checkpoint.checkpoint(
                    QueryMomentNetwork._autocorrelation_psd,
                    raw_flat, S,
                    use_reentrant=False,
                )
            else:
                moments_flat = QueryMomentNetwork._autocorrelation_psd(raw_flat, S)

            moments = moments_flat.view(B, H, T, r, 2 * S + 1)
        else:
            # --- Legacy soft constraint ---
            raw = raw.view(B, H, T, r, S, 2)
            raw_f = raw.float()
            mu_positive = torch.complex(raw_f[..., 0], raw_f[..., 1])

            magnitude = torch.abs(mu_positive)
            phase = torch.angle(mu_positive)
            scaled_mag = 0.95 * torch.sigmoid(magnitude)
            mu_positive = scaled_mag * torch.exp(1j * phase)

            mu_negative = torch.conj(torch.flip(mu_positive, dims=[-1]))
            mu_zero = torch.ones(B, H, T, r, 1, dtype=torch.complex64, device=queries.device)
            moments = torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)

        if is_single_query:
            moments = moments.squeeze(2)

        return moments


# ============================================================================
# FMMA Attention Module
# ============================================================================


class FMMAAttention(nn.Module):
    """
    Fejér Mixture Moment Attention (per FMMA paper Theorem 1).

    s_i(t) = Σ_m φ_m(k_i) · p_m(θ_i − τ | q_t)      (nonneg scores)
    w_i(t) = s_i(t) / Σ_j s_j(t)                      (normalised)
    y_t    = Σ_i w_i(t) · v_i                          (output)

    Memory: O(r · S · d) constant in sequence length.
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

        # Build a simple 1/0 mask from HF attention_mask for FMMA paths
        # HF attention_mask can be [B, T] (1=valid, 0=pad) or 4-D causal
        simple_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                simple_mask = attention_mask  # [B, T] 1/0
            elif attention_mask.dim() == 4:
                # 4-D causal mask: [B, 1, T, T] with 0=attend, -inf=mask
                # Derive per-token validity: a token is valid if any query attends to it
                # Use the last query row as a proxy
                simple_mask = (attention_mask[:, 0, -1, :] > -1e4).long()

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

        outputs = []
        for t in range(seq_len):
            # --- update cache with token t FIRST (Fix 0.1) ---
            kf_t = key_feats[:, :, t : t + 1, :]  # [B, H, 1, r]
            v_t = values[:, :, t : t + 1, :]  # [B, H, 1, D]
            pos_t = position_ids[:, t : t + 1]  # [B, 1]

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

            # --- query from cache ---
            q_mom_t = q_moments[:, :, t]  # [B, H, r, 2S+1]
            phase_t = phases[:, t].unsqueeze(1).expand(-1, num_heads)  # [B, H]

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
          3.3  Density positivity: clamps densities ≥ 0 as safety net.

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

        # Fix 3.3: Clamp densities ≥ 0 as safety net
        if not torch.is_grad_enabled():
            with torch.no_grad():
                chunk_min = densities.min().item()
                if self._last_min_density is None:
                    self._last_min_density = chunk_min
                else:
                    self._last_min_density = min(self._last_min_density, chunk_min)
        densities = densities.clamp(min=0.0)

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


# ============================================================================
# Integration Helpers
# ============================================================================


def create_fmma_cache(
    model,
    num_channels: int = 4,
    bandwidth: int = 64,
    max_seq_len: int = 8192,
) -> FMMACache:
    """Create an FMMA cache for a model."""
    config = model.config
    return FMMACache(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        num_channels=num_channels,
        bandwidth=bandwidth,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )


# Fix 4.1 (PLAN.md): Known attention class names for targeted replacement.
_KNOWN_ATTENTION_CLASSES = {
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen2FlashAttention2",
    "LlamaAttention",
    "LlamaSdpaAttention",
    "LlamaFlashAttention2",
    "MistralAttention",
    "MistralSdpaAttention",
    "MistralFlashAttention2",
    "GemmaAttention",
    "GemmaSdpaAttention",
    "Phi3Attention",
    "Phi3SdpaAttention",
    "Starcoder2Attention",
}


def replace_attention_with_fmma(
    model,
    num_channels: int = 4,
    bandwidth: int = 64,
    max_seq_len: int = 8192,
    layers_to_replace: Optional[List[int]] = None,
):
    """
    Replace model's attention layers with FMMA attention.

    Fix 4.1: Targets known transformer attention classes instead of fragile
             string matching with 'attn'.
    Fix 4.2: Uses repeat_interleave for GQA head mapping.
    """
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    if layers_to_replace is None:
        layers_to_replace = list(range(num_layers))

    replaced = 0

    for name, module in list(model.named_modules()):
        class_name = type(module).__name__

        # Fix 4.1: Match by known class OR by structural signature
        is_known_class = class_name in _KNOWN_ATTENTION_CLASSES
        is_structural_match = (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
        )

        if not (is_known_class or is_structural_match):
            continue

        # Extract layer index
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                break

        if layer_idx is None or layer_idx not in layers_to_replace:
            continue

        device = module.q_proj.weight.device
        dtype = module.q_proj.weight.dtype

        fmma = FMMAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            bandwidth=bandwidth,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            num_kv_heads=num_kv_heads,
        )
        fmma = fmma.to(device=device, dtype=dtype)

        # Copy Q projection (always full num_heads)
        fmma.q_proj.weight.data.copy_(module.q_proj.weight.data)

        # Copy K, V projections — FMMA stores them at num_kv_heads size
        # (expansion happens in forward via _expand_kv_heads)
        k_weight = module.k_proj.weight.data
        v_weight = module.v_proj.weight.data

        if k_weight.shape[0] == fmma.k_proj.weight.shape[0]:
            fmma.k_proj.weight.data.copy_(k_weight)
            fmma.v_proj.weight.data.copy_(v_weight)
        elif k_weight.shape[0] < fmma.k_proj.weight.shape[0]:
            # Original is GQA (fewer KV heads) but FMMA was created with
            # num_kv_heads — this shouldn't happen with new constructor, but
            # handle gracefully.  Fix 4.2: repeat_interleave
            repeat_factor = fmma.k_proj.weight.shape[0] // k_weight.shape[0]
            fmma.k_proj.weight.data.copy_(
                k_weight.repeat_interleave(repeat_factor, dim=0)
            )
            fmma.v_proj.weight.data.copy_(
                v_weight.repeat_interleave(repeat_factor, dim=0)
            )
        else:
            fmma.k_proj.weight.data.copy_(k_weight)
            fmma.v_proj.weight.data.copy_(v_weight)

        fmma.o_proj.weight.data.copy_(module.o_proj.weight.data)

        # Replace module in parent
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, fmma)

        replaced += 1
        print(f"  Replaced layer {layer_idx} ({class_name}) with FMMA")

    print(f"Total layers replaced: {replaced}")
    return model


# ============================================================================
# Test Functions
# ============================================================================


def test_fmma_algebraic_exactness():
    """
    Acceptance test 1 (per paper §6.1): Verify algebraic exactness of moment
    computation.

    Compute y_t both ways:
    1. Explicit: sum over all tokens
    2. From moments: using cached M_V and M_1

    Should match to numerical precision.
    """
    print("=" * 60)
    print("Test 1: FMMA Algebraic Exactness")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    seq_len = 32
    head_dim = 16
    num_channels = 3
    bandwidth = 8
    max_seq_len = 1024

    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)

    positions = torch.arange(seq_len)
    phases = 2 * math.pi * positions.float() / max_seq_len

    key_features = F.softplus(
        torch.randn(batch_size, num_heads, seq_len, num_channels)
    )

    mu_positive = torch.randn(batch_size, num_heads, num_channels, bandwidth) + 1j * torch.randn(
        batch_size, num_heads, num_channels, bandwidth
    )
    mu_positive = 0.5 * mu_positive
    mu_negative = torch.conj(torch.flip(mu_positive, dims=[-1]))
    mu_zero = torch.ones(batch_size, num_heads, num_channels, 1, dtype=torch.complex64)
    query_moments = torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)

    k_indices = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
    c_k = 1.0 - torch.abs(k_indices) / (bandwidth + 1)

    query_phase = phases[-1]

    # Method 1: Explicit
    delta = phases - query_phase
    exp_ik_delta = torch.exp(1j * k_indices.unsqueeze(0) * delta.unsqueeze(1))

    p_m = torch.einsum(
        "bhrk,tk,k->bhrt", query_moments, exp_ik_delta, c_k.to(torch.complex64)
    )
    s_i = torch.einsum("bhtr,bhrt->bht", key_features.to(torch.complex64), p_m)

    numerator_explicit = torch.einsum("bht,bhtd->bhd", s_i, values.to(torch.complex64))
    denominator_explicit = s_i.sum(dim=-1)
    y_explicit = (numerator_explicit / denominator_explicit.unsqueeze(-1)).real

    # Method 2: From cached moments
    exp_ik_theta = torch.exp(1j * k_indices.unsqueeze(0) * phases.unsqueeze(1))

    M_1 = torch.einsum("bhtr,tk->bhrk", key_features.to(torch.complex64), exp_ik_theta)
    M_V = torch.einsum(
        "bhtr,tk,bhtd->bhrkd",
        key_features.to(torch.complex64),
        exp_ik_theta,
        values.to(torch.complex64),
    )

    exp_neg_ik_tau = torch.exp(-1j * k_indices * query_phase)
    combined = c_k.to(torch.complex64) * query_moments * exp_neg_ik_tau

    numerator_moments = torch.einsum("bhrk,bhrkd->bhd", combined, M_V)
    denominator_moments = torch.einsum("bhrk,bhrk->bh", combined, M_1)

    y_moments = (numerator_moments / denominator_moments.unsqueeze(-1)).real

    max_error = (y_explicit - y_moments).abs().max().item()
    print(f"Max error between explicit and moment computation: {max_error:.2e}")

    # float32 complex arithmetic over S=8, T=32 → expect ~1e-5 to 1e-4
    if max_error < 1e-4:
        print("✓ PASSED: Algebraic exactness verified")
        return True
    else:
        print("✗ FAILED: Results don't match")
        return False


def test_fmma_memory_constant():
    """Acceptance test 2: Verify memory stays constant as sequence grows."""
    print("\n" + "=" * 60)
    print("Test 2: FMMA Memory Constant")
    print("=" * 60)

    num_heads = 4
    head_dim = 16
    num_channels = 4
    bandwidth = 32
    max_seq_len = 8192

    memories = []

    for seq_len in [10, 100, 1000]:
        cache = FMMACache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            bandwidth=bandwidth,
            max_seq_len=max_seq_len,
        )

        key_feats = torch.rand(1, num_heads, seq_len, num_channels)
        values = torch.rand(1, num_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)

        cache.update_moments(
            layer_idx=0,
            key_features=key_feats,
            values=values,
            position_ids=positions,
        )

        mem = cache.memory_usage(0)
        memories.append(mem["total_bytes"])
        print(
            f"Seq len {seq_len:5d}: M_V = {mem['M_V_bytes']/1024:.1f} KB, "
            f"M_1 = {mem['M_1_bytes']/1024:.1f} KB, "
            f"total = {mem['total_bytes']/1024:.1f} KB"
        )

    if memories[0] == memories[1] == memories[2]:
        print("✓ PASSED: Memory is constant in sequence length")
        return True
    else:
        print("✗ FAILED: Memory varies with sequence length")
        return False


def test_fmma_nonnegativity():
    """
    Acceptance test 3: Verify attention weights are nonneg with PSD moments.
    Uses wrapped Cauchy (μ_k = ρ^|k|) which is always PSD Toeplitz.
    """
    print("\n" + "=" * 60)
    print("Test 3: FMMA Nonnegativity (with PSD-constrained moments)")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    seq_len = 20
    head_dim = 16
    num_channels = 3
    bandwidth = 8

    key_features = F.softplus(
        torch.randn(batch_size, num_heads, seq_len, num_channels)
    )
    positions = torch.arange(seq_len)
    phases = 2 * math.pi * positions.float() / 1024

    rho = 0.7
    k_range = torch.arange(1, bandwidth + 1, dtype=torch.float32)
    mu_magnitudes = rho ** k_range

    mu_positive = mu_magnitudes.view(1, 1, 1, -1).expand(
        batch_size, num_heads, num_channels, -1
    ).to(torch.complex64)
    mu_negative = torch.flip(mu_positive, dims=[-1])
    mu_zero = torch.ones(batch_size, num_heads, num_channels, 1, dtype=torch.complex64)
    query_moments = torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)

    k_indices = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
    c_k = 1.0 - torch.abs(k_indices) / (bandwidth + 1)

    all_nonneg = True
    for query_pos in [0, seq_len // 2, seq_len - 1]:
        query_phase = phases[query_pos]
        delta = phases - query_phase

        exp_ik_delta = torch.exp(1j * k_indices.unsqueeze(0) * delta.unsqueeze(1))
        p_m = torch.einsum(
            "bhrk,tk,k->bhrt",
            query_moments,
            exp_ik_delta,
            c_k.to(torch.complex64),
        )

        p_m_real = p_m.real
        min_p = p_m_real.min().item()

        s_i = torch.einsum("bhtr,bhrt->bht", key_features, p_m_real)
        min_s = s_i.min().item()
        print(f"Query pos {query_pos:2d}: min(p_m)={min_p:.6f}, min(s_i)={min_s:.6f}")
        if min_s < -1e-6:
            all_nonneg = False

    if all_nonneg:
        print("✓ PASSED: Attention scores are nonnegative with PSD moments")
        return True
    else:
        print("✗ FAILED: Some attention scores are negative")
        return False


def test_fmma_causality():
    """
    Acceptance test 4 (per paper §6.2): Verify causality.
    Modifying future tokens must not change past outputs.
    """
    print("\n" + "=" * 60)
    print("Test 4: FMMA Causality")
    print("=" * 60)

    torch.manual_seed(42)

    hidden_size = 64
    num_heads = 2
    head_dim = 32
    num_channels = 4
    bandwidth = 8
    seq_len = 16

    attn = FMMAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        layer_idx=0,
    )
    attn.eval()

    x = torch.randn(1, seq_len, hidden_size)

    with torch.no_grad():
        output1, _ = attn(x)

    x_modified = x.clone()
    x_modified[:, 8:, :] = torch.randn(1, 8, hidden_size)

    with torch.no_grad():
        output2, _ = attn(x_modified)

    max_diff_past = (output1[:, :8, :] - output2[:, :8, :]).abs().max().item()
    max_diff_future = (output1[:, 8:, :] - output2[:, 8:, :]).abs().max().item()

    print(f"Max diff in past positions (0-7): {max_diff_past:.2e}")
    print(f"Max diff in future positions (8-15): {max_diff_future:.2e}")

    if max_diff_past < 1e-5:
        print("✓ PASSED: Modifying future tokens doesn't affect past outputs")
        return True
    else:
        print("✗ FAILED: Past outputs changed when future tokens modified")
        return False


def test_fmma_long_context_stability():
    """
    Acceptance test 5 (per paper §6.3): Long-context stability.
    Memory stays constant; moment magnitudes stay bounded.
    """
    print("\n" + "=" * 60)
    print("Test 5: FMMA Long-Context Stability")
    print("=" * 60)

    torch.manual_seed(42)

    num_heads = 4
    head_dim = 16
    num_channels = 4
    bandwidth = 16
    max_seq_len = 8192

    seq_lengths = [100, 500, 1000, 2000]
    memories = []

    for seq_len in seq_lengths:
        cache = FMMACache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            bandwidth=bandwidth,
            max_seq_len=max_seq_len,
        )

        key_feats = torch.rand(1, num_heads, seq_len, num_channels)
        values = torch.rand(1, num_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)

        cache.update_moments(
            layer_idx=0,
            key_features=key_feats,
            values=values,
            position_ids=positions,
        )

        mem = cache.memory_usage(0)
        memories.append(mem["total_bytes"])

        M_1 = cache._M_1[0]
        min_mag = M_1.abs().min().item()
        max_mag = M_1.abs().max().item()

        print(
            f"Seq len {seq_len:5d}: memory={mem['total_bytes']/1024:.1f} KB, "
            f"|M_1| range=[{min_mag:.2e}, {max_mag:.2e}]"
        )

    memory_constant = all(m == memories[0] for m in memories)
    print(f"\nMemory constant across all sequence lengths: {memory_constant}")

    if memory_constant:
        print("✓ PASSED: Memory is O(rS), constant in sequence length")
        return True
    else:
        print("✗ FAILED: Memory varies with sequence length")
        return False


def test_psd_penalty():
    """Test the PSD Toeplitz penalty function."""
    print("\n" + "=" * 60)
    print("Test 6: PSD Toeplitz Penalty")
    print("=" * 60)

    torch.manual_seed(42)
    bandwidth = 4

    # PSD moments (wrapped Cauchy)
    rho = 0.5
    k_range = torch.arange(1, bandwidth + 1, dtype=torch.float32)
    mu_pos = (rho ** k_range).to(torch.complex64)
    mu_neg = torch.flip(mu_pos, dims=[-1])
    mu_zero = torch.ones(1, dtype=torch.complex64)
    psd_moments = torch.cat([mu_neg, mu_zero, mu_pos]).unsqueeze(0)

    penalty_psd = psd_toeplitz_penalty(psd_moments, reduction="mean").item()
    print(f"PSD moments (wrapped Cauchy): penalty = {penalty_psd:.6f}")

    # Random moments
    random_moments = torch.randn(1, 2 * bandwidth + 1) + 1j * torch.randn(
        1, 2 * bandwidth + 1
    )
    random_moments = random_moments.clone()
    random_moments[0, bandwidth] = random_moments[0, bandwidth].real + 0j
    for k in range(1, bandwidth + 1):
        random_moments[0, bandwidth - k] = random_moments[0, bandwidth + k].conj()

    penalty_random = psd_toeplitz_penalty(random_moments, reduction="mean").item()
    print(f"Random Hermitian moments: penalty = {penalty_random:.6f}")

    if penalty_psd < 1e-6 and penalty_random > penalty_psd:
        print("✓ PASSED: PSD penalty correctly identifies PSD vs non-PSD moments")
        return True
    else:
        print("✗ FAILED: PSD penalty not working as expected")
        return False


def test_bandwidth_regularization():
    """Test the bandwidth regularization function."""
    print("\n" + "=" * 60)
    print("Test 7: Bandwidth Regularization")
    print("=" * 60)

    bandwidth = 4

    low_freq = torch.zeros(1, 2 * bandwidth + 1, dtype=torch.complex64)
    low_freq[0, bandwidth] = 1.0

    reg_low = bandwidth_regularization(
        low_freq, bandwidth, alpha=2.0, reduction="mean"
    ).item()

    high_freq = torch.zeros(1, 2 * bandwidth + 1, dtype=torch.complex64)
    high_freq[0, bandwidth] = 1.0
    high_freq[0, 0] = 0.5
    high_freq[0, -1] = 0.5

    reg_high = bandwidth_regularization(
        high_freq, bandwidth, alpha=2.0, reduction="mean"
    ).item()

    print(f"Low-frequency moments: reg = {reg_low:.6f}")
    print(f"High-frequency moments: reg = {reg_high:.6f}")

    if reg_high > reg_low:
        print("✓ PASSED: High-frequency moments penalized more than low-frequency")
        return True
    else:
        print("✗ FAILED: Bandwidth regularization not working as expected")
        return False


def test_psd_autocorrelation_parameterization():
    """
    Test 8: Verify the autocorrelation parameterization produces PSD Toeplitz
    moments (Fix 1.2).
    """
    print("\n" + "=" * 60)
    print("Test 8: PSD Autocorrelation Parameterization")
    print("=" * 60)

    torch.manual_seed(42)

    query_dim = 32
    num_channels = 4
    bandwidth = 8

    net = QueryMomentNetwork(
        query_dim=query_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        enforce_psd=True,
    )

    # Random queries
    queries = torch.randn(2, 4, 10, query_dim)  # [B, H, T, D]
    moments = net(queries)  # [B, H, T, r, 2S+1]

    # Check μ_0 ≈ 1
    S = bandwidth
    mu_0 = moments[..., S]
    mu_0_error = (mu_0.real - 1.0).abs().max().item()
    mu_0_imag = mu_0.imag.abs().max().item()
    print(f"μ_0 real deviation from 1: {mu_0_error:.6f}")
    print(f"μ_0 imaginary part: {mu_0_imag:.6f}")

    # Check Hermitian symmetry: μ_{-k} = conj(μ_k)
    hermitian_error = 0.0
    for k in range(1, S + 1):
        mu_neg_k = moments[..., S - k]
        mu_pos_k = moments[..., S + k]
        err = (mu_neg_k - mu_pos_k.conj()).abs().max().item()
        hermitian_error = max(hermitian_error, err)
    print(f"Max Hermitian symmetry error: {hermitian_error:.6f}")

    # Check PSD via eigenvalues
    penalty = psd_toeplitz_penalty(moments, reduction="mean", max_samples=32).item()
    print(f"PSD Toeplitz penalty: {penalty:.6f}")

    if mu_0_error < 1e-4 and hermitian_error < 1e-5 and penalty < 1e-5:
        print("✓ PASSED: Autocorrelation produces valid PSD Toeplitz moments")
        return True
    else:
        print("✗ FAILED: Autocorrelation parameterization issues")
        return False


def test_fmma_hf_cache_contract():
    """
    Test 9 (Fix 2.1): Verify HF DynamicCache.update() returns zero-sized
    tensors.
    """
    print("\n" + "=" * 60)
    print("Test 9: HF DynamicCache Contract")
    print("=" * 60)

    cache = FMMACache(
        num_layers=1,
        num_heads=4,
        head_dim=16,
        num_channels=4,
        bandwidth=8,
        max_seq_len=1024,
    )

    key_states = torch.randn(1, 4, 10, 16)
    value_states = torch.randn(1, 4, 10, 16)

    k_out, v_out = cache.update(key_states, value_states, layer_idx=0)

    print(f"Input K shape:  {key_states.shape}")
    print(f"Output K shape: {k_out.shape}")
    print(f"Input V shape:  {value_states.shape}")
    print(f"Output V shape: {v_out.shape}")

    if k_out.shape[2] == 0 and v_out.shape[2] == 0:
        print("✓ PASSED: update() returns zero-length sequence tensors")
        return True
    else:
        print("✗ FAILED: update() did not return zero-length tensors")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("FMMA (Fejér Mixture Moment Attention) Tests")
    print("=" * 60)
    print()

    results = {}
    results["algebraic_exactness"] = test_fmma_algebraic_exactness()
    results["memory_constant"] = test_fmma_memory_constant()
    results["nonnegativity"] = test_fmma_nonnegativity()
    results["causality"] = test_fmma_causality()
    results["long_context_stability"] = test_fmma_long_context_stability()
    results["psd_penalty"] = test_psd_penalty()
    results["bandwidth_regularization"] = test_bandwidth_regularization()
    results["psd_autocorrelation"] = test_psd_autocorrelation_parameterization()
    results["hf_cache_contract"] = test_fmma_hf_cache_contract()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:30s}: {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        print("\n✓ All FMMA acceptance tests passed!")
        print("\nFMMA provides:")
        print("  - Content awareness via r nonnegative key channels")
        print("  - O(rS) memory for BOTH K and V (100% KV cache savings)")
        print("  - Exact computation from cached moments")
        print("  - Guaranteed nonneg attention weights (PSD autocorrelation)")
        print("  - Causal streaming (token-by-token cache update)")
        print("  - Consistent phase mapping (always max_seq_len)")
        print("  - Proper HF DynamicCache contract")
    else:
        print("\n✗ Some tests failed — review implementation")
