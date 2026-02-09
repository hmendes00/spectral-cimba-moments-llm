"""
PSD Toeplitz Penalty and Bandwidth Regularization (per FMMA paper §5.4).
"""

import torch


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

    # Guard: filter out rows with NaN/Inf (no useful gradient from them)
    finite_mask = torch.isfinite(moments_sampled.real).all(dim=-1) & \
                  torch.isfinite(moments_sampled.imag).all(dim=-1)
    if not finite_mask.any():
        return torch.tensor(0.0, device=moments.device, requires_grad=True)
    moments_sampled = moments_sampled[finite_mask]

    T = build_toeplitz_matrix(moments_sampled)

    # Guard: check Toeplitz matrix is finite
    T_real_finite = torch.isfinite(T.real).all()
    T_imag_finite = torch.isfinite(T.imag).all()
    if not (T_real_finite and T_imag_finite):
        return torch.tensor(0.0, device=moments.device, requires_grad=True)

    try:
        eigenvalues = torch.linalg.eigvalsh(T)
    except RuntimeError:
        # eigvalsh can fail on ill-conditioned matrices; skip penalty
        return torch.tensor(0.0, device=moments.device, requires_grad=True)

    negative_eigs = torch.clamp(eigenvalues, max=0)
    penalty_per_matrix = (negative_eigs ** 2).sum(dim=-1)

    if reduction == "mean":
        return penalty_per_matrix.mean()
    elif reduction == "sum":
        # Fix C2 (PLAN.md): Scale by the ratio of total to *effective* sample
        # size (after finite_mask filtering), not the pre-filtered size.
        effective_samples = moments_sampled.shape[0]
        scale = N / effective_samples if effective_samples > 0 else 1.0
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

    # Fix C3 (PLAN.md): Guard against S=0 which would cause division by zero.
    if S == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

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
