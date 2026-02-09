"""
Moment Distillation: Softmax → FMMA Normalization Bridge

Implements the practical moment-matching loss from the KL-projection paper:
    m_k(t) = Σ_{i≤t} w^soft_{t,i} · e^{ik(θ_i − θ_t)},  |k| ≤ S

Instead of solving the inner KL optimization, we directly match the teacher's
low-order circular moments — which Theorem 2 (moment-matching characterization)
proves is the correct target for the KL-optimal bandlimited proxy.

Usage:
    from fmma.moment_distill import (
        compute_teacher_phase_moments,
        compute_moment_loss,
        select_query_indices,
        select_layers_for_moments,
    )
"""

import math
import random
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# §3.1  Canonical phase mapping (must match FMMAAttention._get_phases)
# ---------------------------------------------------------------------------

def phases_from_positions(
    position_ids: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Convert integer position_ids → circular phases.

    θ = 2π · pos / max_seq_len   (canonical, matches FMMAAttention._get_phases)

    Args:
        position_ids: [B, T] or [T] integer positions
        max_seq_len: max sequence length used by FMMA

    Returns:
        phases: same shape as position_ids, float
    """
    return 2 * math.pi * position_ids.float() / max_seq_len


# ---------------------------------------------------------------------------
# §7  Layer / head / query subsampling
# ---------------------------------------------------------------------------

def select_layers_for_moments(
    fmma_layer_indices: List[int],
    mom_layers: Optional[List[int]],
    mom_max_layers_per_step: int,
) -> List[int]:
    """
    Choose which FMMA layers to supervise this step.

    Args:
        fmma_layer_indices: all layer indices that have FMMA attention
        mom_layers: explicit list (from config), or None → sample randomly
        mom_max_layers_per_step: max layers to pick per step

    Returns:
        List of layer indices to compute moment loss on
    """
    if mom_layers is not None:
        # Intersect with available FMMA layers
        candidates = [l for l in mom_layers if l in fmma_layer_indices]
    else:
        candidates = list(fmma_layer_indices)

    if len(candidates) <= mom_max_layers_per_step:
        return candidates

    return random.sample(candidates, mom_max_layers_per_step)


def select_query_indices(
    seq_len: int,
    mom_query_stride: int,
    mom_max_queries: int,
) -> torch.Tensor:
    """
    Choose query positions for moment supervision.

    Strategy: stride through the second half of the sequence (the first few
    positions have trivially concentrated attention), then cap to max queries.

    Args:
        seq_len: total sequence length T
        mom_query_stride: take every N-th position
        mom_max_queries: cap number of positions

    Returns:
        t_idx: 1-D long tensor of query position indices
    """
    start = max(seq_len // 4, 1)  # skip the very first positions
    t_idx = torch.arange(start, seq_len, mom_query_stride)

    if len(t_idx) > mom_max_queries:
        # Subsample uniformly
        perm = torch.randperm(len(t_idx))[:mom_max_queries]
        t_idx = t_idx[perm].sort().values

    return t_idx


def select_heads(
    num_heads: int,
    mom_head_stride: int,
) -> torch.Tensor:
    """
    Choose heads for moment supervision.

    Args:
        num_heads: total number of attention heads
        mom_head_stride: take every N-th head

    Returns:
        h_idx: 1-D long tensor of head indices
    """
    return torch.arange(0, num_heads, mom_head_stride)


# ---------------------------------------------------------------------------
# §3.2  Compute teacher phase moments from attention weights
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_phase_moments(
    attn_weights: torch.Tensor,       # [B, H, T, T]  teacher attention probs
    phases: torch.Tensor,             # [B, T]         circular phases
    t_idx: torch.Tensor,             # [Q]            query position indices
    bandwidth: int,                   # S
    h_idx: Optional[torch.Tensor] = None,  # [Hsel] head indices
    use_fejer_weights: bool = True,
    detach: bool = True,
) -> torch.Tensor:
    """
    Compute teacher's softmax phase moments for selected query positions.

    m_k(t) = Σ_{i≤t} w^soft_{t,i} · e^{ik(θ_i − θ_t)},  k = −S…S

    Args:
        attn_weights: [B, H, T, T] softmax attention probabilities from teacher
        phases: [B, T] circular phases (same mapping as FMMA)
        t_idx: [Q] query positions to compute moments for
        bandwidth: S, harmonic bandwidth
        h_idx: [Hsel] optional head subset; None → all heads
        use_fejer_weights: multiply moments by Fejér weights c_k
        detach: detach result (should be True since teacher is frozen)

    Returns:
        m_teacher: [B, Hsel, Q, 2S+1] complex64  teacher moments
    """
    device = attn_weights.device
    S = bandwidth

    # Subsample heads
    if h_idx is not None:
        h_idx = h_idx.to(device)
        A = attn_weights[:, h_idx]  # [B, Hsel, T, T]
    else:
        A = attn_weights  # [B, H, T, T]

    # Subsample query positions: A_q[b, h, q, i] = attn_weights[b, h, t_idx[q], i]
    t_idx_dev = t_idx.to(device)
    A_q = A[:, :, t_idx_dev, :]  # [B, Hsel, Q, T]

    # §8 Sanity: rows should sum to ~1
    # Use pure-GPU check to avoid .item() sync on every moment-distill step.
    row_sums = A_q.sum(dim=-1)  # [B, Hsel, Q]
    max_dev = (row_sums - 1.0).abs().max()
    if max_dev > 0.05:
        # Only call .item() on the rare failure path
        warnings.warn(
            f"Teacher attention rows don't sum to 1 "
            f"(max deviation: {max_dev.item():.4f}). "
            f"Check attention_mask handling."
        )

    # Phase differences: δ = θ_i − θ_t  for each query t
    theta_q = phases[:, t_idx_dev]                    # [B, Q]
    delta = phases.unsqueeze(1) - theta_q.unsqueeze(2)  # [B, Q, T]

    # Build complex exponentials for k = 1…S and compute moments
    k_vals = torch.arange(1, S + 1, device=device, dtype=torch.float32)  # [S]

    # k * delta: [B, Q, T, S]
    k_delta = delta.unsqueeze(-1) * k_vals  # [B, Q, T, S]

    # E_k = e^{ikδ}: [B, Q, T, S]
    E_k = torch.exp(1j * k_delta.to(torch.float32))

    # m_k = Σ_i A_q[b,h,q,i] · E_k[b,q,i,k]  → [B, Hsel, Q, S]
    # A_q is [B, Hsel, Q, T], E_k is [B, Q, T, S]
    m_pos = torch.einsum("bhqi,bqis->bhqs", A_q.to(torch.complex64), E_k)

    # m_0 = Σ_i A_q = 1 (by normalization)
    m_zero = torch.ones(
        m_pos.shape[0], m_pos.shape[1], m_pos.shape[2], 1,
        dtype=torch.complex64, device=device,
    )

    # m_{-k} = conj(m_k) (Hermitian symmetry)
    m_neg = torch.conj(torch.flip(m_pos, dims=[-1]))

    # Pack: [B, Hsel, Q, 2S+1] with ordering μ_{-S}…μ_0…μ_S
    m_teacher = torch.cat([m_neg, m_zero, m_pos], dim=-1)

    # §8 Sanity: |m_k| ≤ 1 (moments of a probability measure)
    # Use pure-GPU check to avoid .item() sync on every moment-distill step.
    max_mag = m_pos.abs().max()
    if max_mag > 1.05:
        # Only call .item() on the rare failure path
        warnings.warn(
            f"Teacher moment magnitudes exceed 1 (max={max_mag.item():.4f}). "
            f"This suggests numerical issues."
        )

    # Optionally apply Fejér weights to match FMMA's density reconstruction basis
    if use_fejer_weights:
        k_indices = torch.arange(-S, S + 1, dtype=torch.float32, device=device)
        c_k = 1.0 - torch.abs(k_indices) / (S + 1)  # [2S+1]
        m_teacher = m_teacher * c_k.view(1, 1, 1, -1)

    if detach:
        m_teacher = m_teacher.detach()

    return m_teacher


# ---------------------------------------------------------------------------
# §5  Moment loss computation
# ---------------------------------------------------------------------------

def compute_moment_loss(
    mu_student: torch.Tensor,     # [B, H, Q, r, 2S+1] complex — student QueryMomentNetwork output
    m_teacher: torch.Tensor,      # [B, H, Q, 2S+1] complex — teacher phase moments
    bandwidth: int,               # S
    use_fejer_weights: bool = True,
) -> torch.Tensor:
    """
    Compute L2 moment-matching loss between student and teacher.

    §5.1: Average student moments over channels r for distribution-shape supervision.
    §5.2: Compare only k=1…S (skip k=0 which is fixed at 1).

    If teacher moments already have Fejér weights applied, student moments
    should also be weighted before comparison.

    Args:
        mu_student: [B, H, Q, r, 2S+1] complex — raw student moments
        m_teacher: [B, H, Q, 2S+1] complex — teacher target moments
        bandwidth: S
        use_fejer_weights: if True, apply Fejér weights to student moments

    Returns:
        loss: scalar, mean L2 over (batch, heads, queries, harmonics)
    """
    S = bandwidth

    # §5.1: Reduce channels — average over r for distribution-shape supervision
    mu_eff = mu_student.mean(dim=3)  # [B, H, Q, 2S+1]

    # Apply Fejér weights to student if teacher already has them
    if use_fejer_weights:
        device = mu_eff.device
        k_indices = torch.arange(-S, S + 1, dtype=torch.float32, device=device)
        c_k = 1.0 - torch.abs(k_indices) / (S + 1)
        mu_eff = mu_eff * c_k.view(1, 1, 1, -1)

    # §5.2: Compare only positive harmonics k=1…S (skip k=0)
    student_pos = mu_eff[..., S + 1:]   # [B, H, Q, S]
    teacher_pos = m_teacher[..., S + 1:]  # [B, H, Q, S]

    diff = student_pos - teacher_pos
    loss = (diff.real ** 2 + diff.imag ** 2).mean()

    return loss
