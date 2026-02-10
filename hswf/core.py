"""
HSWF Core Algorithm — Harmonic Structured Weight Factorization

Implements the deterministic, post-training compression from the HSWF paper:
    W ≈ Ŵ = C_S + UV^T

where C_S is a bandlimited circulant stored by S Fourier coefficients and
UV^T is an optimal rank-r correction (Eckart-Young-Mirsky).

Key theorems verified:
  Lemma 1:   Best circulant = wrap-diagonal averaging
  Theorem 2: Bandlimit error = Fourier tail energy
  Theorem 3: Pythagorean decomposition (orthogonal projection)
  Theorem 4: Total error identity
"""

import math
from typing import Optional, List, Dict

import torch
import torch.fft


# ============================================================================
# Step 1: Best circulant approximation (Lemma 1)
# ============================================================================

def wrap_diagonal_average(W: torch.Tensor) -> torch.Tensor:
    """
    Compute the best circulant approximation via wrap-diagonal averaging.

    Lemma 1 (paper):  c*_k = (1/d) Σ_i W[i, (i-k) mod d]

    This is the orthogonal projection of W onto the circulant subspace
    under the Frobenius inner product.

    Args:
        W: [d, d] square matrix
    Returns:
        c: [d] first column of the best circulant approximation C*
    """
    d = W.shape[0]
    assert W.shape == (d, d), f"Matrix must be square, got {W.shape}"

    # Vectorized: gathered[k, i] = W[i, (i-k) mod d]
    indices = torch.arange(d, device=W.device)
    i_idx = indices.unsqueeze(0).expand(d, d)   # [d, d]: row index = i
    k_idx = indices.unsqueeze(1).expand(d, d)   # [d, d]: diagonal index = k
    col_idx = (i_idx - k_idx) % d               # [d, d]: (i-k) mod d

    gathered = W[i_idx, col_idx]                 # [d, d]
    return gathered.mean(dim=1)                  # [d]


# ============================================================================
# Circulant utilities
# ============================================================================

def circulant_from_column(c: torch.Tensor) -> torch.Tensor:
    """
    Build the full d×d circulant matrix from its first column.

    C[i, j] = c[(i - j) mod d]

    NOTE: Only use for analysis / testing.  For inference, use circulant_matvec
    (FFT-based, O(d log d) instead of O(d²)).
    """
    d = c.shape[0]
    indices = torch.arange(d, device=c.device)
    idx = (indices.unsqueeze(1) - indices.unsqueeze(0)) % d
    return c[idx]


def circulant_spectrum(c: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of circ(c).

    λ_k = DFT(c)_k = Σ_j c_j e^{-2πijk/d}

    Uses PyTorch's default FFT normalization (unnormalized forward, 1/d inverse).
    Under this convention, C = (1/d) F^* diag(λ) F, and
    ||C||_F² = Σ_k |λ_k|²  (verified by test_spectrum_norm).

    Note: this is NOT the unitary DFT (which would use 1/√d both ways).
    """
    return torch.fft.fft(c.float())


def circulant_matvec(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply circulant matrix to input via FFT: y = circ(c) @ x.

    For nn.Linear convention (output = input @ W^T), use:
        output = circulant_matvec_left(c, input)

    This function computes the RIGHT multiplication: C @ x.

    Cost: O(d log d) instead of O(d²).
    """
    c_fft = torch.fft.fft(c.float())
    x_fft = torch.fft.fft(x.float(), dim=-1)
    return torch.fft.ifft(c_fft * x_fft, dim=-1).real


def circulant_matvec_left(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Left-multiply: y = x @ C^T  (nn.Linear convention).

    x @ C^T is circular convolution of x and c:
        (x @ C^T)_j = Σ_i x_i c_{(j-i) mod d} = (x * c)_j

    FFT property: FFT(x * c) = FFT(x) · FFT(c).
    """
    c_fft = torch.fft.fft(c.float())
    x_fft = torch.fft.fft(x.float(), dim=-1)
    return torch.fft.ifft(c_fft * x_fft, dim=-1).real


# ============================================================================
# Step 2: Bandlimiting (Theorem 2)
# ============================================================================

def bandlimit_spectrum(
    spectrum: torch.Tensor,
    S: int,
    strategy: str = "energy",
) -> tuple:
    """
    Bandlimit: keep only S Fourier modes, zero the rest.

    Args:
        spectrum: [d] complex Fourier coefficients (eigenvalues of C*)
        S: number of modes to retain
        strategy:
            "energy"  — keep the S modes with largest |λ_k|² (optimal error)
            "lowpass" — keep the S lowest-frequency modes (DC-centered)

    Returns:
        (bandlimited_spectrum [d], kept_indices [S])
    """
    d = spectrum.shape[0]
    S = min(S, d)

    if strategy == "energy":
        magnitudes = spectrum.abs()
        _, kept_idx = torch.topk(magnitudes, S)
        kept_idx = kept_idx.sort().values
    elif strategy == "lowpass":
        if S >= d:
            kept_idx = torch.arange(d, device=spectrum.device)
        else:
            # DC at index 0, then positive freqs 1..d//2, negative freqs
            n_pos = (S + 1) // 2   # includes DC
            n_neg = S - n_pos
            pos_idx = torch.arange(n_pos, device=spectrum.device)
            neg_idx = torch.arange(d - n_neg, d, device=spectrum.device)
            kept_idx = torch.cat([pos_idx, neg_idx]).sort().values
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    bandlimited = torch.zeros_like(spectrum)
    bandlimited[kept_idx] = spectrum[kept_idx]

    return bandlimited, kept_idx


# ============================================================================
# Full HSWF decomposition (Theorem 4)
# ============================================================================

def hswf_decompose(
    W: torch.Tensor,
    S: int,
    r: int,
    bandlimit_strategy: str = "energy",
) -> Dict:
    """
    Full HSWF decomposition of a square matrix.

    Algorithm:
        1. C* = best circulant approximation (wrap-diagonal average)  [Lemma 1]
        2. C_S = bandlimit C* to S Fourier modes                     [Theorem 2]
        3. R = W - C_S  (residual)
        4. R_r = best rank-r approx of R via SVD                     [Lemma 3]
        5. Ŵ = C_S + R_r                                             [Theorem 4]

    Args:
        W: [d, d] weight matrix (float32)
        S: number of Fourier modes to keep (circulant bandwidth)
        r: rank of low-rank correction
        bandlimit_strategy: "energy" (optimal) or "lowpass"

    Returns:
        dict with components, error metrics, and compression stats
    """
    W = W.float()
    d = W.shape[0]
    assert W.shape == (d, d), f"Matrix must be square, got {W.shape}"

    # Step 1: Best circulant
    c_star = wrap_diagonal_average(W)
    spectrum_full = circulant_spectrum(c_star)   # [d] complex

    # Non-circulant error: ||W - C*||_F
    C_star = circulant_from_column(c_star)
    non_circ_error_sq = (W - C_star).square().sum()
    W_norm_sq = W.square().sum()

    # Step 2: Bandlimit
    S = min(S, d)
    spectrum_S, kept_indices = bandlimit_spectrum(
        spectrum_full, S, strategy=bandlimit_strategy
    )
    c_S = torch.fft.ifft(spectrum_S).real        # [d]

    # Fourier tail energy (Theorem 2): ||C* - C_S||² = Σ_{k∉Ω} |λ*_k|²
    tail_mask = torch.ones(d, dtype=torch.bool, device=W.device)
    tail_mask[kept_indices] = False
    fourier_tail_energy_sq = spectrum_full[tail_mask].abs().square().sum()

    # Step 3: Residual
    C_S_mat = circulant_from_column(c_S)
    R = W - C_S_mat

    # Step 4: Rank-r SVD of residual
    r = min(r, d)
    U_full, sigma, Vt_full = torch.linalg.svd(R, full_matrices=False)
    sigma_r = sigma[:r]
    U_r = U_full[:, :r]
    V_r = Vt_full[:r, :].T       # [d, r]

    # R_r for error computation
    R_r = U_r @ torch.diag(sigma_r) @ V_r.T

    # Step 5: Ŵ = C_S + R_r
    W_hat = C_S_mat + R_r

    # Error metrics
    total_error_sq = (W - W_hat).square().sum()
    residual_tail_sq = sigma[r:].square().sum() if r < len(sigma) else torch.tensor(0.0)

    W_norm = W_norm_sq.sqrt().item()
    total_error = total_error_sq.sqrt().item()

    # Compression stats
    dense_params = d * d
    # S complex Fourier coefficients = 2S reals; U,V each d×r reals; sigma r reals
    # But we store U√σ and V√σ directly, so just 2dr reals
    hswf_params = 2 * S + 2 * r * d
    compression_ratio = dense_params / max(hswf_params, 1)

    return {
        # Components (for HSWFLinear)
        "c_S": c_S,                               # [d] real
        "spectrum_S": spectrum_S,                  # [d] complex (sparse)
        "kept_indices": kept_indices,              # [S] long
        "U_r": U_r,                                # [d, r]
        "sigma_r": sigma_r,                        # [r]
        "V_r": V_r,                                # [d, r]
        "all_singular_values": sigma,              # [min(d,d)]

        # Full spectrum for analysis
        "spectrum_full": spectrum_full,            # [d] complex

        # Error metrics
        "W_norm": W_norm,
        "non_circ_error": non_circ_error_sq.sqrt().item(),
        "non_circ_relative": (non_circ_error_sq.sqrt() / max(W_norm, 1e-10)).item(),
        "fourier_tail_energy": fourier_tail_energy_sq.sqrt().item(),
        "total_error": total_error,
        "total_relative_error": total_error / max(W_norm, 1e-10),

        # Theorem verification (both sqrt and squared for rigorous checking)
        "residual_tail_energy": residual_tail_sq.sqrt().item(),
        "pythagorean_lhs": (W - C_S_mat).square().sum().sqrt().item(),
        "pythagorean_rhs": (non_circ_error_sq + fourier_tail_energy_sq).sqrt().item(),
        "pythagorean_lhs_sq": (W - C_S_mat).square().sum().item(),
        "pythagorean_rhs_sq": (non_circ_error_sq + fourier_tail_energy_sq).item(),

        # Compression
        "dense_params": dense_params,
        "hswf_params": hswf_params,
        "compression_ratio": compression_ratio,
        "S": S,
        "r": r,
        "d": d,
    }


# ============================================================================
# Compressibility analysis
# ============================================================================

def analyze_compressibility(W: torch.Tensor, name: str = "") -> Dict:
    """
    Analyze how compressible a square matrix is under HSWF.

    Returns structural metrics WITHOUT committing to specific (S, r):
    - Non-circulant error (how far from circulant)
    - Fourier energy distribution (determines optimal S)
    - Sorted singular values of residual at various S (determines r)
    """
    W = W.float()
    d = W.shape[0]
    assert W.shape == (d, d), f"Matrix must be square, got {W.shape}"

    # Best circulant
    c_star = wrap_diagonal_average(W)
    spectrum_full = circulant_spectrum(c_star)
    C_star = circulant_from_column(c_star)

    W_norm = W.norm().item()
    non_circ_error = (W - C_star).norm().item()

    # Fourier energy distribution (sorted by magnitude)
    magnitudes = spectrum_full.abs()
    sorted_mags, sorted_indices = magnitudes.sort(descending=True)
    cumulative_energy = sorted_mags.square().cumsum(dim=0)
    total_fourier_energy = cumulative_energy[-1].item()

    # How many modes needed for X% of Fourier energy
    if total_fourier_energy > 0:
        cum_frac = cumulative_energy / total_fourier_energy
        s_50 = (cum_frac >= 0.50).nonzero(as_tuple=True)[0][0].item() + 1
        s_90 = (cum_frac >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1
        s_95 = (cum_frac >= 0.95).nonzero(as_tuple=True)[0][0].item() + 1
        s_99 = (cum_frac >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1
    else:
        s_50 = s_90 = s_95 = s_99 = d

    # Full SVD of W (for pure-SVD baseline comparison)
    _, sigma_W, _ = torch.linalg.svd(W, full_matrices=False)

    return {
        "name": name,
        "d": d,
        "W_norm": W_norm,
        "non_circ_error": non_circ_error,
        "non_circ_relative": non_circ_error / max(W_norm, 1e-10),
        "total_fourier_energy": total_fourier_energy,
        "fourier_s50": s_50,
        "fourier_s90": s_90,
        "fourier_s95": s_95,
        "fourier_s99": s_99,
        "sorted_fourier_magnitudes": sorted_mags.cpu(),
        "singular_values_W": sigma_W.cpu(),
        "spectrum_full": spectrum_full.cpu(),
    }


def compressibility_sweep(
    W: torch.Tensor,
    S_values: Optional[List[int]] = None,
    r_values: Optional[List[int]] = None,
    bandlimit_strategy: str = "energy",
) -> List[Dict]:
    """
    Sweep (S, r) and compute error/compression tradeoff for each combination.

    Also computes the pure-SVD baseline (no circulant) for comparison.
    """
    W = W.float()
    d = W.shape[0]
    assert W.shape == (d, d), f"Matrix must be square, got {W.shape}"

    if S_values is None:
        S_values = [d // 32, d // 16, d // 8, d // 4, d // 2, 3 * d // 4]
    if r_values is None:
        r_values = [0, 1, 2, 4, 8, 16, 32, 64]

    S_values = [min(max(1, s), d) for s in S_values]
    r_values = [min(max(0, r), d) for r in r_values]

    W_norm = W.norm().item()

    # Pre-compute circulant
    c_star = wrap_diagonal_average(W)
    spectrum_full = circulant_spectrum(c_star)

    # Pure SVD baseline (no circulant component)
    _, sigma_W, _ = torch.linalg.svd(W, full_matrices=False)
    svd_results = []
    for r in r_values:
        if r == 0:
            continue
        r_c = min(r, len(sigma_W))
        svd_error = sigma_W[r_c:].square().sum().sqrt().item()
        svd_params = 2 * r_c * d  # U and V only
        svd_results.append({
            "method": "SVD-only",
            "S": 0,
            "r": r_c,
            "total_error": svd_error,
            "relative_error": svd_error / max(W_norm, 1e-10),
            "hswf_params": svd_params,
            "dense_params": d * d,
            "compression_ratio": (d * d) / max(svd_params, 1),
        })

    # HSWF sweep
    hswf_results = []
    for S in S_values:
        spectrum_S, kept_idx = bandlimit_spectrum(
            spectrum_full, S, strategy=bandlimit_strategy
        )
        c_S = torch.fft.ifft(spectrum_S).real
        C_S = circulant_from_column(c_S)
        R = W - C_S

        # SVD of residual
        _, sigma_R, _ = torch.linalg.svd(R, full_matrices=False)

        for r in r_values:
            r_c = min(r, len(sigma_R))
            if r_c > 0:
                residual_error = sigma_R[r_c:].square().sum().sqrt().item()
            else:
                residual_error = R.norm().item()

            hswf_params = 2 * S + 2 * r_c * d
            hswf_results.append({
                "method": "HSWF",
                "S": S,
                "r": r_c,
                "total_error": residual_error,
                "relative_error": residual_error / max(W_norm, 1e-10),
                "hswf_params": hswf_params,
                "dense_params": d * d,
                "compression_ratio": (d * d) / max(hswf_params, 1),
            })

    return svd_results + hswf_results


def analyze_rectangular_svd(W: torch.Tensor, name: str = "") -> Dict:
    """
    Analyze compressibility of a rectangular matrix via SVD.

    Circulant structure requires square matrices; for rectangular weights
    we report singular value decay (how amenable to low-rank approximation).
    """
    W = W.float()
    m, n = W.shape
    W_norm = W.norm().item()

    _, sigma, _ = torch.linalg.svd(W, full_matrices=False)

    # Energy concentration
    cumulative = sigma.square().cumsum(dim=0)
    total_energy = cumulative[-1].item()

    if total_energy > 0:
        cum_frac = cumulative / total_energy
        r_50 = (cum_frac >= 0.50).nonzero(as_tuple=True)[0][0].item() + 1
        r_90 = (cum_frac >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1
        r_95 = (cum_frac >= 0.95).nonzero(as_tuple=True)[0][0].item() + 1
        r_99 = (cum_frac >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1
    else:
        r_50 = r_90 = r_95 = r_99 = min(m, n)

    return {
        "name": name,
        "shape": (m, n),
        "W_norm": W_norm,
        "singular_values": sigma.cpu(),
        "rank_50": r_50,
        "rank_90": r_90,
        "rank_95": r_95,
        "rank_99": r_99,
        "is_square": m == n,
    }
