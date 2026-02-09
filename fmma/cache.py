"""
FMMA Cache: O(rS) memory for both K and V.

Stores moment tensors instead of full KV sequences:
- M_V[m, k, :] = Σ_i φ_m(k_i) e^{ikθ_i} v_i  ∈ C^{d_v}
- M_1[m, k]    = Σ_i φ_m(k_i) e^{ikθ_i}      ∈ C

Memory: O(r * S * d_v) per layer — CONSTANT in sequence length!
"""

import math
import warnings
from typing import Optional, Tuple, List

import torch
from transformers import DynamicCache


class FMMACache(DynamicCache):
    """
    Constant-memory cache for FMMA attention.

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

        # Fix 11 (PLAN.md): Cache k_indices as a persistent tensor instead of
        # re-allocating every update_moments call (minor speed/memory win for
        # long generation where update is called once per token).
        self._cached_k_indices = k_indices  # [2S+1]

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

        # Mask sanity assertion (PLAN pre-training check):
        # Verify padded positions contribute zero to moments.
        if attention_mask is not None and __debug__:
            pad_positions = (attention_mask == 0)  # [B, T]
            if pad_positions.any():
                # key_features already zeroed above; confirm it
                kf_at_pad = key_features[:, :, pad_positions[0]].abs().sum()
                assert kf_at_pad == 0, (
                    f"Mask sanity failed: padded positions have nonzero "
                    f"key_features (sum={kf_at_pad.item():.2e})"
                )

        # e^{ikθ} for k ∈ {-S, ..., S}
        # Fix 11 (PLAN.md): Use cached k_indices instead of allocating anew.
        k_indices = self._cached_k_indices.to(device=device)
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

        # Division-by-zero guard for the degenerate case (e.g. empty cache).
        # PSD autocorrelation guarantees nonneg densities, so the
        # denominator is naturally positive during normal operation.
        output = numerator / (denominator.unsqueeze(-1) + 1e-8)

        # Fix 1.3: Diagnose imaginary leakage.
        # Only check during inference (eval/generate) — this path is not hot
        # and the .item() sync is acceptable here.
        if not torch.is_grad_enabled():
            imag_mag = output.imag.abs().max().item()
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
