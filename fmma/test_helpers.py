"""
Shared test helpers for FMMA comprehensive tests.

Provides:
  - Deterministic seeding
  - Factory for small FMMA modules
  - explicit_fmma_output()  — slow-path reference from the definition
  - cached_fmma_output()    — fast-path via FMMACache moments
  - streaming_cache_runner() — per-token update-then-query loop
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .attention import FMMAAttention
from .cache import FMMACache
from .features import NonnegativeKeyFeatures, QueryMomentNetwork

DEVICE = "cpu"
DTYPE = torch.float32
SEED = 1234

# Tolerances (fp32)
ATOL = 1e-5
RTOL = 1e-4


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_attention(
    hidden_size: int = 32,
    num_heads: int = 4,
    head_dim: int = 8,
    num_channels: int = 4,
    bandwidth: int = 4,
    max_seq_len: int = 64,
    layer_idx: int = 0,
    **kwargs,
) -> FMMAAttention:
    """Build a small FMMA module for tests."""
    attn = FMMAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        max_seq_len=max_seq_len,
        layer_idx=layer_idx,
        **kwargs,
    )
    return attn.to(DEVICE)


def make_cache(
    num_heads: int = 4,
    head_dim: int = 8,
    num_channels: int = 4,
    bandwidth: int = 4,
    max_seq_len: int = 64,
    num_layers: int = 1,
) -> FMMACache:
    """Build a matching FMMA cache."""
    return FMMACache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        max_seq_len=max_seq_len,
        dtype=DTYPE,
        device=DEVICE,
    )


# ─────────────────────────────────────────────────────────────
# Helper: explicit FMMA output from the definition (slow path)
# ─────────────────────────────────────────────────────────────
def explicit_fmma_output(
    key_features: torch.Tensor,    # [B, H, L, r]  nonneg
    values: torch.Tensor,          # [B, H, L, D]
    query_moments: torch.Tensor,   # [B, H, r, 2S+1] complex  (for ONE query)
    phases: torch.Tensor,          # [B, L]    key phases
    query_phase: torch.Tensor,     # [B]       query phase
    bandwidth: int,
    causal_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Reference implementation: compute y_t from the sum-over-tokens definition.

    s_i = Σ_m φ_m(k_i) · p_m(θ_i − τ | q_t)
    y_t = Σ_i s_i v_i / Σ_i s_i

    If causal_len is given, only tokens 0..causal_len-1 contribute.

    Returns: [B, H, D]
    """
    S = bandwidth
    k_indices = torch.arange(-S, S + 1, dtype=torch.float32, device=key_features.device)
    c_k = 1.0 - torch.abs(k_indices) / (S + 1)

    # delta = θ_i − τ  (paper convention)
    delta = phases - query_phase.unsqueeze(-1)  # [B, L]

    # e^{ik·Δ}  →  [B, L, 2S+1]
    exp_ik_delta = torch.exp(
        1j * k_indices.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1)
    )

    # p_m(Δ|q) = Σ_k c_k μ_k e^{ikΔ}   →  [B, H, r, L]
    p_m = torch.einsum(
        "bhrk,blk,k->bhrl",
        query_moments,
        exp_ik_delta.to(torch.complex64),
        c_k.to(torch.complex64),
    )  # [B, H, r, L]

    # s_i = Σ_m φ_m(k_i) p_m(Δ_i|q)  →  [B, H, L]
    s_i = torch.einsum(
        "bhlr,bhrl->bhl",
        key_features.to(torch.complex64),
        p_m,
    )

    # Apply causal mask if needed
    if causal_len is not None:
        L = s_i.shape[-1]
        mask = torch.zeros(L, device=s_i.device)
        mask[causal_len:] = 1.0
        s_i = s_i.masked_fill(mask.bool().unsqueeze(0).unsqueeze(0), 0.0)

    num = torch.einsum("bhl,bhld->bhd", s_i, values.to(torch.complex64))
    den = s_i.sum(dim=-1)  # [B, H]
    y = (num / (den.unsqueeze(-1) + 1e-8)).real
    return y


# ─────────────────────────────────────────────────────────────
# Helper: cached theorem output (fast path via moments)
# ─────────────────────────────────────────────────────────────
def cached_fmma_output(
    key_features: torch.Tensor,    # [B, H, L, r]
    values: torch.Tensor,          # [B, H, L, D]
    query_moments: torch.Tensor,   # [B, H, r, 2S+1] complex
    phases: torch.Tensor,          # [B, L]
    query_phase: torch.Tensor,     # [B]
    bandwidth: int,
    max_seq_len: int,
    position_ids: torch.Tensor,   # [L] integer
    causal_len: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build FMMACache, feed tokens, compute attention via theorem.

    If causal_len is given, only tokens 0..causal_len-1 are fed.

    Returns: [B, H, D]
    """
    B, H, L, r = key_features.shape
    D = values.shape[-1]
    S = bandwidth

    feed_len = causal_len if causal_len is not None else L

    cache = FMMACache(
        num_layers=1,
        num_heads=H,
        head_dim=D,
        num_channels=r,
        bandwidth=S,
        max_seq_len=max_seq_len,
    )

    cache.update_moments(
        layer_idx=0,
        key_features=key_features[:, :, :feed_len],
        values=values[:, :, :feed_len],
        position_ids=position_ids[:feed_len],
        attention_mask=attention_mask[:, :feed_len] if attention_mask is not None else None,
    )

    q_phases = query_phase.unsqueeze(1).expand(-1, H)  # [B, H]
    return cache.compute_attention(0, query_moments, q_phases)


# ─────────────────────────────────────────────────────────────
# Helper: streaming cache runner (per-token update-then-query)
# ─────────────────────────────────────────────────────────────
def streaming_cache_runner(
    key_features: torch.Tensor,    # [B, H, L, r]
    values: torch.Tensor,          # [B, H, L, D]
    query_moments_all: torch.Tensor,  # [B, H, L, r, 2S+1] complex
    phases: torch.Tensor,          # [B, L]
    bandwidth: int,
    max_seq_len: int,
    position_ids: torch.Tensor,   # [L] integer
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Stream tokens one-by-one: for each t, update cache with token t then
    query.  Returns all outputs [B, H, L, D].
    """
    B, H, L, r = key_features.shape
    D = values.shape[-1]
    S = bandwidth

    cache = FMMACache(
        num_layers=1,
        num_heads=H,
        head_dim=D,
        num_channels=r,
        bandwidth=S,
        max_seq_len=max_seq_len,
    )

    outputs = []
    for t in range(L):
        kf_t = key_features[:, :, t:t+1, :]
        v_t = values[:, :, t:t+1, :]
        pos_t = position_ids[t:t+1]

        mask_t = None
        if attention_mask is not None:
            mask_t = attention_mask[:, t:t+1]

        cache.update_moments(0, kf_t, v_t, pos_t, mask_t)

        q_mom_t = query_moments_all[:, :, t]  # [B, H, r, 2S+1]
        phase_t = phases[:, t].unsqueeze(1).expand(-1, H)

        out_t = cache.compute_attention(0, q_mom_t, phase_t)
        outputs.append(out_t)

    return torch.stack(outputs, dim=2)  # [B, H, L, D]
