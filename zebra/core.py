"""
ZEBRA Core Algorithm — Zero-Training Eigen-Bandlimited Residual Approximation

Implements the post-training weight compression from the ZEBRA paper:
    W ≈ A^T · P_K(AWB^T) · B  +  R_r

where A ∈ F_m, B ∈ F_n are from a fast orthogonal family,
P_K keeps the K largest-magnitude entries (transform-domain sparsification),
and R_r is an optimal rank-r correction (Eckart-Young-Mirsky).

Key results:
  Lemma (Tail energy):       ||W - W_{A,B,K}||_F² = Σ_{(i,j)∉Ω_K} |W̃_ij|²
  Lemma (Top-K optimality):  P_K is the best K-sparse approx under fixed (A,B)
  Lemma (EYM):               ||R - R_r||_F² = Σ_{j>r} σ_j(R)²
  Theorem (Decomposition):   Two certificate curves per layer
"""

import math
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# Fast orthogonal transform families
# ============================================================================

def hadamard_matrix(d: int, device=None) -> torch.Tensor:
    """
    Construct a d×d normalized Hadamard matrix (d must be a power of 2).

    Uses the Sylvester/Walsh construction: H_1 = [1], H_{2n} = [[H_n, H_n], [H_n, -H_n]].
    Normalized so that H^T H = I.

    Cost: O(d²) to construct, but Hx can be applied in O(d log d) via butterfly.
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d must be a power of 2, got {d}"
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(d)


def fast_hadamard_transform(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply the normalized Hadamard transform along the given dimension.

    Uses the butterfly (recursive) algorithm: O(d log d) per vector.
    The transform is its own inverse (up to normalization).

    Args:
        x: input tensor, size along `dim` must be a power of 2
        dim: dimension to transform along

    Returns:
        Hx / sqrt(d)  (orthonormal convention)
    """
    d = x.shape[dim]
    assert d > 0 and (d & (d - 1)) == 0, f"dim size must be power of 2, got {d}"

    # Move target dim to last for easier indexing
    x = x.movedim(dim, -1).contiguous()
    shape = x.shape
    x = x.reshape(-1, d).float()

    # Butterfly stages
    h = 1
    while h < d:
        # Split into pairs of size h
        x = x.view(-1, d // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.view(-1, d)
        h *= 2

    x = x / math.sqrt(d)
    x = x.reshape(shape)
    return x.movedim(-1, dim)


def random_sign_diagonal(d: int, seed: int, device=None) -> torch.Tensor:
    """
    Generate a random ±1 diagonal vector from a seed.

    The diagonal matrix D = diag(signs) is orthogonal (D^T D = I)
    and can be applied in O(d) by element-wise multiplication.
    """
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=gen, device='cpu').float() * 2 - 1
    return signs.to(device) if device is not None else signs


def random_permutation(d: int, seed: int, device=None) -> torch.Tensor:
    """
    Generate a random permutation from a seed.

    Returns the permutation indices (not the matrix).
    P^T P = I. Cost: O(d) to apply.
    """
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    perm = torch.randperm(d, generator=gen, device='cpu')
    return perm.to(device) if device is not None else perm


def _decompose_into_powers_of_2(d: int) -> List[int]:
    """
    Decompose d into an ordered list of powers of 2 (greedy/binary).

    E.g. 896 → [512, 256, 128], since 512+256+128 = 896.
    Every positive integer has a unique binary decomposition.
    """
    blocks = []
    remaining = d
    while remaining > 0:
        # Largest power of 2 ≤ remaining
        p = 1
        while p * 2 <= remaining:
            p *= 2
        blocks.append(p)
        remaining -= p
    return blocks


def build_fast_orthogonal(
    d: int,
    seed: int = 0,
    device=None,
    n_factors: int = 3,
) -> dict:
    """
    Build a fast orthogonal transform Q as a block-diagonal product.

    For dimension d, we decompose d into a sum of powers of 2:
        d = 2^{a_1} + 2^{a_2} + ... + 2^{a_k}

    Then Q is block-diagonal, with each block being a scrambled
    Hadamard of size 2^{a_i}:
        Q = diag(Q_1, Q_2, ..., Q_k)
        Q_i = D_n P_n ... D_1 P_1 H   (within block i)

    Before and after the block-diagonal transform, a global permutation
    and sign diagonal are applied to mix across blocks:
        Full Q = D_post · P_post · BlockDiag(Q_i) · D_pre · P_pre

    This is truly orthogonal on R^d for ANY d (no padding/truncation).

    Args:
        d: dimension (any positive integer)
        seed: random seed for reproducibility
        n_factors: number of scrambling factors per block

    Returns:
        dict with transform descriptor (for implicit storage)
    """
    assert d > 0, f"d must be positive, got {d}"

    blocks = _decompose_into_powers_of_2(d)

    # Build per-block scrambled Hadamard
    block_descriptors = []
    for bi, block_size in enumerate(blocks):
        factors = []
        for i in range(n_factors):
            perm = random_permutation(block_size, seed=seed * 10000 + bi * 1000 + i * 2, device=device)
            signs = random_sign_diagonal(block_size, seed=seed * 10000 + bi * 1000 + i * 2 + 1, device=device)
            # Precompute inverse permutation so transpose path avoids per-call allocation
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(block_size, device=perm.device if device is None else device)
            factors.append({"perm": perm, "signs": signs, "inv_perm": inv_perm})
        block_descriptors.append({
            "size": block_size,
            "factors": factors,
        })

    # Global pre/post permutation + signs to mix across blocks
    pre_perm = random_permutation(d, seed=seed * 10000 + 9000, device=device)
    pre_signs = random_sign_diagonal(d, seed=seed * 10000 + 9001, device=device)
    post_perm = random_permutation(d, seed=seed * 10000 + 9002, device=device)
    post_signs = random_sign_diagonal(d, seed=seed * 10000 + 9003, device=device)

    # Precompute inverse permutations (avoids per-call allocation in transpose path)
    _dev = pre_perm.device
    inv_pre_perm = torch.empty_like(pre_perm)
    inv_pre_perm[pre_perm] = torch.arange(d, device=_dev)
    inv_post_perm = torch.empty_like(post_perm)
    inv_post_perm[post_perm] = torch.arange(d, device=_dev)

    return {
        "d": d,
        "blocks": blocks,
        "block_descriptors": block_descriptors,
        "n_factors": n_factors,
        "pre_perm": pre_perm,
        "pre_signs": pre_signs,
        "post_perm": post_perm,
        "post_signs": post_signs,
        "inv_pre_perm": inv_pre_perm,
        "inv_post_perm": inv_post_perm,
        "seed": seed,
    }


def _apply_block_scrambled_hadamard(
    x: torch.Tensor,
    block_desc: dict,
    transpose: bool,
) -> torch.Tensor:
    """
    Apply a single-block scrambled Hadamard (power-of-2 size) along last dim.

    Forward:  D_n P_n ... D_1 P_1 H x
    Transpose: H P_1^{-1} D_1 ... P_n^{-1} D_n x
    """
    factors = block_desc["factors"]
    bs = block_desc["size"]

    if not transpose:
        x = fast_hadamard_transform(x, dim=-1)
        for factor in factors:
            perm = factor["perm"]
            signs = factor["signs"]
            x = x[..., perm]
            x = x * signs
    else:
        for factor in reversed(factors):
            signs = factor["signs"]
            inv_perm = factor["inv_perm"]
            x = x * signs
            x = x[..., inv_perm]
        x = fast_hadamard_transform(x, dim=-1)
    return x


def apply_fast_orthogonal(
    x: torch.Tensor,
    transform: dict,
    transpose: bool = False,
    dim: int = -1,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Apply a fast orthogonal transform Q (or Q^T) to x along `dim`.

    Q = D_post · P_post · BlockDiag(Q_i) · D_pre · P_pre

    Each block Q_i is a scrambled Hadamard of power-of-2 size.
    The blocks partition the dimension, so the full transform is
    truly orthogonal on R^d for any d.

    Args:
        compute_dtype: dtype for internal computation (default: float32).
            Set to x.dtype to match inference pipeline exactly.

    Cost: O(d log d_max) per vector, where d_max is the largest block.
    """
    d = transform["d"]
    blocks = transform["blocks"]
    block_descriptors = transform["block_descriptors"]
    pre_perm = transform["pre_perm"]
    pre_signs = transform["pre_signs"]
    post_perm = transform["post_perm"]
    post_signs = transform["post_signs"]

    # Move target dim to last
    x = x.movedim(dim, -1)
    orig_shape = x.shape
    orig_dtype = x.dtype
    cdtype = compute_dtype if compute_dtype is not None else torch.float32
    x = x.to(cdtype)

    if not transpose:
        # Forward: pre_signs, pre_perm, block transforms, post_perm, post_signs
        x = x * pre_signs
        x = x[..., pre_perm]

        # Apply block-diagonal transform
        chunks = x.split(blocks, dim=-1)
        out_chunks = []
        for chunk, bdesc in zip(chunks, block_descriptors):
            out_chunks.append(_apply_block_scrambled_hadamard(chunk, bdesc, transpose=False))
        x = torch.cat(out_chunks, dim=-1)

        x = x[..., post_perm]
        x = x * post_signs
    else:
        # Transpose: reverse everything
        # Q^T = P_pre^{-1} D_pre BlockDiag(Q_i^T) P_post^{-1} D_post
        x = x * post_signs
        x = x[..., transform["inv_post_perm"]]

        # Block-diagonal transpose
        chunks = x.split(blocks, dim=-1)
        out_chunks = []
        for chunk, bdesc in zip(chunks, block_descriptors):
            out_chunks.append(_apply_block_scrambled_hadamard(chunk, bdesc, transpose=True))
        x = torch.cat(out_chunks, dim=-1)

        x = x[..., transform["inv_pre_perm"]]
        x = x * pre_signs

    x = x.reshape(orig_shape)
    return x.movedim(-1, dim).to(orig_dtype)


def build_orthogonal_matrix(transform: dict, device=None) -> torch.Tensor:
    """
    Materialize the full d×d orthogonal matrix Q from a transform descriptor.

    Only for analysis/testing — inference should use apply_fast_orthogonal.
    O(d²) cost — do NOT call for d > 512.
    """
    d = transform["d"]
    assert d <= 512, (
        f"build_orthogonal_matrix is O(d²) test-only; got d={d}. "
        f"Use apply_fast_orthogonal for real workloads."
    )
    I = torch.eye(d, device=device)
    Q = torch.zeros(d, d, device=device)
    for i in range(d):
        e_i = I[i]
        Q[i] = apply_fast_orthogonal(e_i.unsqueeze(0), transform, dim=-1).squeeze(0)
    return Q


def compute_W_tilde(
    W: torch.Tensor,
    transform_A: dict,
    transform_B: dict,
) -> torch.Tensor:
    """
    Compute W̃ = A W B^T without materializing dense A or B matrices.

    Convention: apply(v, T, forward) = v @ Q,  apply(v, T, transpose) = v @ Q^T.

    Cost: O(m*n*log(m) + m*n*log(n)) instead of O(m²n + mn²) for dense products.
    """
    m, n = W.shape

    # Step 1: W1 = Q_A @ W
    # Column j: W1[:,j] = Q_A @ W[:,j].
    # In row-vector form: W[:,j]^T @ Q_A^T → apply_transpose on each row of W.T.
    # Then transpose back.
    W1 = apply_fast_orthogonal(W.T, transform_A, transpose=True, dim=-1).T  # [m, n]

    # Step 2: W̃ = W1 @ Q_B^T
    # Each row: v_i → v_i @ Q_B^T → apply_transpose on each row of W1.
    W_tilde = apply_fast_orthogonal(W1, transform_B, transpose=True, dim=-1)  # [m, n]

    return W_tilde


def reconstruct_from_sparse(
    W_tilde_K: torch.Tensor,
    transform_A: dict,
    transform_B: dict,
) -> torch.Tensor:
    """
    Compute W_{A,B,K} = A^T @ W̃_K @ B without materializing A or B.

    Inverse of compute_W_tilde applied to the sparsified coefficients.
    """
    m, n = W_tilde_K.shape

    # Step 1: temp = W̃_K @ Q_B
    # Each row: v_i → v_i @ Q_B → apply_forward on each row.
    temp = apply_fast_orthogonal(W_tilde_K, transform_B, transpose=False, dim=-1)  # [m, n]

    # Step 2: result = Q_A^T @ temp
    # Column j: result[:,j] = Q_A^T @ temp[:,j].
    # In row-vector form: temp[:,j]^T @ Q_A → apply_forward on each row of temp.T.
    # Then transpose back.
    result = apply_fast_orthogonal(temp.T, transform_A, transpose=False, dim=-1).T  # [m, n]

    return result


# ============================================================================
# Top-K operator (transform-domain sparsification)
# ============================================================================

def top_k_sparsify(
    X: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Top-K sparsification: keep the K entries with largest |X_ij|, zero the rest.

    This is P_K(X) from the paper.

    Args:
        X: [m, n] matrix
        K: number of entries to keep

    Returns:
        (sparse_X, indices, values):
            sparse_X: [m, n] matrix with only K nonzeros
            indices:  [K, 2] kept (row, col) indices
            values:   [K] kept values
    """
    m, n = X.shape
    K = min(K, m * n)

    if K <= 0:
        sparse_X = torch.zeros_like(X)
        indices = torch.zeros(0, 2, dtype=torch.long, device=X.device)
        values = torch.zeros(0, dtype=X.dtype, device=X.device)
        return sparse_X, indices, values

    flat = X.reshape(-1)
    magnitudes = flat.abs()

    _, top_idx = torch.topk(magnitudes, K)

    # Convert flat indices to (row, col)
    rows = top_idx // n
    cols = top_idx % n
    indices = torch.stack([rows, cols], dim=1)  # [K, 2]
    values = flat[top_idx]

    # Build sparse matrix
    sparse_X = torch.zeros_like(X)
    sparse_X[rows, cols] = values

    return sparse_X, indices, values


def top_k_indices_only(
    X: torch.Tensor,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-K sparsification returning only (indices, values), no dense matrix.

    Use this in sweeps to avoid allocating an m×n dense zeros tensor per K.

    Args:
        X: [m, n] matrix
        K: number of entries to keep

    Returns:
        (indices, values):
            indices: [K, 2] kept (row, col) indices
            values:  [K] kept values
    """
    m, n = X.shape
    K = min(K, m * n)

    if K <= 0:
        return (torch.zeros(0, 2, dtype=torch.long, device=X.device),
                torch.zeros(0, dtype=X.dtype, device=X.device))

    flat = X.reshape(-1)
    _, top_idx = torch.topk(flat.abs(), K)
    rows = top_idx // n
    cols = top_idx % n
    return torch.stack([rows, cols], dim=1), flat[top_idx]


def reconstruct_from_indices(
    indices: torch.Tensor,
    values: torch.Tensor,
    m: int,
    n: int,
    transform_A: dict,
    transform_B: dict,
) -> torch.Tensor:
    """
    Reconstruct W_{A,B,K} = A^T @ W̃_K @ B from sparse (indices, values).

    Avoids materializing a full m×n dense W̃_K — builds it just-in-time.

    Args:
        indices: [K, 2] row/col indices of kept coefficients
        values:  [K] values of kept coefficients
        m, n:    shape of W̃_K
        transform_A, transform_B: fast orthogonal transform descriptors

    Returns:
        [m, n] reconstructed weight matrix
    """
    W_tilde_K = torch.zeros(m, n, device=values.device, dtype=values.dtype)
    if len(values) > 0:
        W_tilde_K[indices[:, 0], indices[:, 1]] = values
    return reconstruct_from_sparse(W_tilde_K, transform_A, transform_B)


def top_k_tail_energy(X: torch.Tensor, K: int) -> float:
    """
    Compute the tail energy: Σ_{(i,j)∉Ω_K} |X_ij|².

    This is the squared Frobenius error of the top-K approximation.
    """
    m, n = X.shape
    K = min(K, m * n)

    if K <= 0:
        return X.square().sum().item()

    flat = X.reshape(-1)
    magnitudes = flat.abs()

    if K >= m * n:
        return 0.0

    # Sort and sum the smallest (mn - K) squared magnitudes
    sorted_mags, _ = magnitudes.sort(descending=True)
    tail = sorted_mags[K:]
    return tail.square().sum().item()


# ============================================================================
# Sparse matvec
# ============================================================================

def sparse_matmul_left(
    x: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    m: int,
    n: int,
) -> torch.Tensor:
    """
    Compute y = x @ S^T where S is a sparse [m, n] matrix.

    x: [..., m]  →  y: [..., n]

    Using the sparse representation (indices [K,2], values [K]).
    """
    batch_shape = x.shape[:-1]
    x_flat = x.reshape(-1, m).float()

    rows = indices[:, 0]  # S rows → x columns
    cols = indices[:, 1]  # S cols → output columns

    # y[b, col] += x[b, row] * value
    # Build sparse tensor for efficient batched matmul
    S_sparse = torch.sparse_coo_tensor(
        indices.T.long(),
        values.float(),
        size=(m, n),
    ).coalesce()

    y = torch.sparse.mm(S_sparse.T, x_flat.T).T  # [batch, n]
    return y.reshape(*batch_shape, n)


# ============================================================================
# ZEBRA decomposition
# ============================================================================

def zebra_decompose(
    W: torch.Tensor,
    K: int,
    r: int,
    seed_A: int = 0,
    seed_B: int = 1,
    n_factors: int = 3,
    n_candidates: int = 1,
) -> Dict:
    """
    Full ZEBRA decomposition of a weight matrix (can be rectangular).

    Algorithm:
        1. Build fast orthogonal transforms A ∈ F_m, B ∈ F_n
        2. Compute transform coefficients: W̃ = A W B^T  (via fast transform)
        3. Sparsify: W̃_K = P_K(W̃)  (keep K largest entries)
        4. Reconstruct: W_{A,B,K} = A^T W̃_K B  (via fast transform)
        5. Residual: R = W - W_{A,B,K}
        6. Rank-r correction: R_r via SVD
        7. Final: Ŵ = W_{A,B,K} + R_r

    Uses apply_fast_orthogonal throughout — never materializes dense Q.

    Args:
        W: [m, n] weight matrix (float32)
        K: number of transform-domain coefficients to keep
        r: rank of low-rank residual correction
        seed_A: seed for left transform A
        seed_B: seed for right transform B
        n_factors: number of scrambling factors per transform
        n_candidates: if >1, try multiple seeds and pick the best

    Returns:
        dict with components, error metrics, and compression stats
    """
    W = W.float()
    m, n = W.shape
    mn = m * n
    K = min(K, mn)
    r = min(r, min(m, n))

    W_norm_sq = W.square().sum()
    W_norm = W_norm_sq.sqrt().item()

    best_result = None
    best_error = float("inf")

    for cand in range(n_candidates):
        sa = seed_A + cand * 100
        sb = seed_B + cand * 100

        # Step 1: Build transforms
        transform_A = build_fast_orthogonal(m, seed=sa, device=W.device, n_factors=n_factors)
        transform_B = build_fast_orthogonal(n, seed=sb, device=W.device, n_factors=n_factors)

        # Step 2: W̃ = A W B^T via fast transform (no dense matrix materialization)
        W_tilde = compute_W_tilde(W, transform_A, transform_B)

        # Step 3: Top-K sparsification
        W_tilde_K, indices, values = top_k_sparsify(W_tilde, K)

        # Tail energy (Lemma: orthogonal invariance)
        tail_energy_sq = (W_tilde - W_tilde_K).square().sum().item()

        if tail_energy_sq < best_error:
            best_error = tail_energy_sq
            best_result = {
                "transform_A": transform_A,
                "transform_B": transform_B,
                "W_tilde": W_tilde,
                "W_tilde_K": W_tilde_K,
                "sparse_indices": indices,
                "sparse_values": values,
                "tail_energy_sq": tail_energy_sq,
                "seed_A": sa,
                "seed_B": sb,
            }

    # Unpack best candidate
    transform_A = best_result["transform_A"]
    transform_B = best_result["transform_B"]
    W_tilde = best_result["W_tilde"]
    W_tilde_K = best_result["W_tilde_K"]
    indices = best_result["sparse_indices"]
    values = best_result["sparse_values"]
    tail_energy_sq = best_result["tail_energy_sq"]

    A_mat = None  # not materialized
    B_mat = None

    # Step 4: Reconstruct W_{A,B,K} = A^T W̃_K B via fast transform
    W_ABK = reconstruct_from_sparse(W_tilde_K, transform_A, transform_B)

    # Verify orthogonal invariance (the tail energy identity)
    recon_error_sq = (W - W_ABK).square().sum().item()
    _tail_mismatch = abs(recon_error_sq - tail_energy_sq) / max(1.0, tail_energy_sq)
    if _tail_mismatch > 1e-3:
        import warnings
        warnings.warn(
            f"Tail identity mismatch: ||W-W_ABK||²={recon_error_sq:.6f} vs "
            f"tail_energy²={tail_energy_sq:.6f} (rel diff {_tail_mismatch:.2e}). "
            f"Check transform conventions / dtype."
        )

    # Step 5: Residual
    R = W - W_ABK

    # Step 6: Rank-r SVD of residual
    U_full, sigma, Vt_full = torch.linalg.svd(R, full_matrices=False)
    r = min(r, len(sigma))
    sigma_r = sigma[:r]

    if r > 0:
        sqrt_sigma = sigma_r.sqrt()
        U_r = U_full[:, :r] * sqrt_sigma.unsqueeze(0)     # [m, r]
        V_r = Vt_full[:r, :].T * sqrt_sigma.unsqueeze(0)  # [n, r]
        R_r = U_r @ V_r.T
    else:
        U_r = torch.zeros(m, 0, device=W.device)
        V_r = torch.zeros(n, 0, device=W.device)
        R_r = torch.zeros_like(W)

    # Step 7: Final Ŵ = W_{A,B,K} + R_r
    W_hat = W_ABK + R_r
    total_error_sq = (W - W_hat).square().sum().item()
    total_error = math.sqrt(max(total_error_sq, 0))
    residual_tail_sq = sigma[r:].square().sum().item() if r < len(sigma) else 0.0

    # Compression stats
    dense_params = mn
    # Values-only count (optimistic, assumes indices are free)
    sparse_params_values = K
    # Realistic count: each sparse entry needs row+col indices.
    # 2 int32 indices = 2 float32 equivalents, plus 1 value = 3 float-equivalents per entry.
    sparse_params_with_idx = K * 3  # K values + 2K index floats
    lr_params = r * (m + n)  # U [m,r] + V [n,r]
    zebra_params = sparse_params_with_idx + lr_params
    zebra_params_values_only = sparse_params_values + lr_params
    compression_ratio = dense_params / max(zebra_params, 1)

    return {
        # Transform descriptors
        "transform_A": transform_A,
        "transform_B": transform_B,
        "seed_A": best_result["seed_A"],
        "seed_B": best_result["seed_B"],

        # Sparse component
        "sparse_indices": indices,           # [K, 2]
        "sparse_values": values,             # [K]
        "W_tilde_K": W_tilde_K,              # [m, n] (dense for analysis)

        # Low-rank component
        "U_r": U_r,                          # [m, r]
        "V_r": V_r,                          # [n, r]
        "all_singular_values": sigma,        # [min(m,n)]

        # Error certificates
        "W_norm": W_norm,
        "tail_energy_sq": tail_energy_sq,
        "tail_energy": math.sqrt(max(tail_energy_sq, 0)),
        "recon_error_sq": recon_error_sq,    # should match tail_energy_sq
        "total_error_sq": total_error_sq,
        "total_error": total_error,
        "total_relative_error": total_error / max(W_norm, 1e-10),
        "residual_tail_sq": residual_tail_sq,
        "residual_tail": math.sqrt(max(residual_tail_sq, 0)),

        # Certificate curves data
        "sparse_error_sq": tail_energy_sq,   # error after sparse approx
        "lr_error_sq": residual_tail_sq,     # error after low-rank correction

        # Compression (with realistic index accounting)
        "dense_params": dense_params,
        "zebra_params": zebra_params,                     # 2K + r(m+n)
        "zebra_params_values_only": zebra_params_values_only,  # K + r(m+n)
        "sparse_params": sparse_params_with_idx,
        "sparse_params_values_only": sparse_params_values,
        "lr_params": lr_params,
        "compression_ratio": compression_ratio,
        "K": K,
        "r": r,
        "m": m,
        "n": n,
    }


# ============================================================================
# Certificate curves
# ============================================================================

def certificate_curve_sparse(
    W: torch.Tensor,
    K_values: Optional[List[int]] = None,
    seed_A: int = 0,
    seed_B: int = 1,
    n_factors: int = 3,
) -> List[Dict]:
    """
    Compute the transform-tail energy certificate curve: error vs K.

    For fixed (A, B), computes ||W - W_{A,B,K}||_F² for various K values.
    """
    W = W.float()
    m, n = W.shape
    mn = m * n

    if K_values is None:
        K_values = [mn // 64, mn // 32, mn // 16, mn // 8,
                    mn // 4, mn // 2, 3 * mn // 4]
        K_values = [max(1, k) for k in K_values]

    W_norm = W.norm().item()

    # Build transforms and compute W̃ once (no dense matrix materialization)
    transform_A = build_fast_orthogonal(m, seed=seed_A, device=W.device, n_factors=n_factors)
    transform_B = build_fast_orthogonal(n, seed=seed_B, device=W.device, n_factors=n_factors)
    W_tilde = compute_W_tilde(W, transform_A, transform_B)

    # Sort all entries by magnitude (descending) once
    flat = W_tilde.reshape(-1)
    sorted_sq, _ = flat.abs().square().sort(descending=True)
    total_energy = sorted_sq.sum().item()
    cumulative = sorted_sq.cumsum(dim=0)

    results = []
    for K in K_values:
        K = min(K, mn)
        if K >= mn:
            tail_energy_sq = 0.0
        else:
            tail_energy_sq = total_energy - cumulative[K - 1].item() if K > 0 else total_energy

        results.append({
            "K": K,
            "tail_energy_sq": tail_energy_sq,
            "tail_energy": math.sqrt(max(tail_energy_sq, 0)),
            "relative_error": math.sqrt(max(tail_energy_sq, 0)) / max(W_norm, 1e-10),
            "sparsity_ratio": K / mn,
        })

    return results


def certificate_curve_residual(
    R: torch.Tensor,
    r_values: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Compute the residual singular-tail certificate curve: error vs r.

    For residual R, computes ||R - R_r||_F² = Σ_{j>r} σ_j(R)² for various r.
    """
    R = R.float()
    m, n = R.shape
    R_norm = R.norm().item()

    _, sigma, _ = torch.linalg.svd(R, full_matrices=False)

    if r_values is None:
        max_r = min(m, n)
        r_values = [0, 1, 2, 4, 8, 16, 32, 64]
        r_values = [r for r in r_values if r <= max_r]

    results = []
    for r in r_values:
        r = min(r, len(sigma))
        tail_sq = sigma[r:].square().sum().item() if r < len(sigma) else 0.0

        results.append({
            "r": r,
            "tail_energy_sq": tail_sq,
            "tail_energy": math.sqrt(max(tail_sq, 0)),
            "relative_error": math.sqrt(max(tail_sq, 0)) / max(R_norm, 1e-10),
        })

    return results


# ============================================================================
# Transform selection (Section 5 of paper)
# ============================================================================

def select_transforms(
    W: torch.Tensor,
    K: int,
    n_candidates: int = 8,
    n_factors: int = 3,
    base_seed: int = 42,
) -> Tuple[int, int, float]:
    """
    Multi-seed sampling: select (A, B) from the fast orthogonal family
    that minimizes tail energy at budget K.

    This is the "compression objective" from Section 5.1:
        (A*, B*) = argmin_{A,B} E_K(A,B;W)

    Args:
        W: [m, n] weight matrix
        K: sparsity budget
        n_candidates: number of (A,B) pairs to try
        n_factors: scrambling depth per transform
        base_seed: starting seed

    Returns:
        (best_seed_A, best_seed_B, best_tail_energy_sq)
    """
    W = W.float()
    m, n = W.shape
    mn = m * n
    K = min(K, mn)

    best_sa, best_sb = 0, 1
    best_energy = float("inf")

    for cand in range(n_candidates):
        sa = base_seed + cand * 7
        sb = base_seed + cand * 7 + 1000

        transform_A = build_fast_orthogonal(m, seed=sa, device=W.device, n_factors=n_factors)
        transform_B = build_fast_orthogonal(n, seed=sb, device=W.device, n_factors=n_factors)

        W_tilde = compute_W_tilde(W, transform_A, transform_B)

        energy = top_k_tail_energy(W_tilde, K)

        if energy < best_energy:
            best_energy = energy
            best_sa = sa
            best_sb = sb

    return best_sa, best_sb, best_energy


# ============================================================================
# Compressibility analysis
# ============================================================================

def analyze_compressibility(
    W: torch.Tensor,
    name: str = "",
    n_candidates: int = 4,
    n_factors: int = 3,
) -> Dict:
    """
    Analyze how compressible a weight matrix is under ZEBRA.

    Works for both square and rectangular matrices.

    Returns structural metrics WITHOUT committing to specific (K, r):
    - Transform-domain coefficient decay (determines optimal K)
    - Singular value decay (determines optimal r for residual)
    - Best transforms found via multi-seed sampling
    """
    W = W.float()
    m, n = W.shape
    mn = m * n
    W_norm = W.norm().item()

    # Try several transform candidates
    K_test = mn // 4  # Use 25% sparsity for selection
    best_sa, best_sb, _ = select_transforms(
        W, K=K_test, n_candidates=n_candidates, n_factors=n_factors,
    )

    # Compute W̃ with best transforms (no dense matrix materialization)
    transform_A = build_fast_orthogonal(m, seed=best_sa, device=W.device, n_factors=n_factors)
    transform_B = build_fast_orthogonal(n, seed=best_sb, device=W.device, n_factors=n_factors)
    W_tilde = compute_W_tilde(W, transform_A, transform_B)

    # Coefficient magnitude distribution
    flat_mags = W_tilde.reshape(-1).abs()
    sorted_mags, _ = flat_mags.sort(descending=True)
    cumulative = sorted_mags.square().cumsum(dim=0)
    total_energy = cumulative[-1].item()

    # How many coefficients needed for X% of energy
    if total_energy > 0:
        cum_frac = cumulative / total_energy
        k_50 = (cum_frac >= 0.50).nonzero(as_tuple=True)[0][0].item() + 1
        k_90 = (cum_frac >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1
        k_95 = (cum_frac >= 0.95).nonzero(as_tuple=True)[0][0].item() + 1
        k_99 = (cum_frac >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1
    else:
        k_50 = k_90 = k_95 = k_99 = mn

    # SVD singular values only (no U,V) for pure low-rank baseline
    sigma_W = torch.linalg.svdvals(W)

    return {
        "name": name,
        "m": m,
        "n": n,
        "mn": mn,
        "W_norm": W_norm,
        "total_transform_energy": total_energy,
        "coeff_k50": k_50,
        "coeff_k90": k_90,
        "coeff_k95": k_95,
        "coeff_k99": k_99,
        "coeff_k50_frac": k_50 / mn,
        "coeff_k90_frac": k_90 / mn,
        "coeff_k95_frac": k_95 / mn,
        "coeff_k99_frac": k_99 / mn,
        "sorted_coeff_magnitudes": sorted_mags.cpu(),
        "singular_values_W": sigma_W.cpu(),
        "best_seed_A": best_sa,
        "best_seed_B": best_sb,
    }


def compressibility_sweep(
    W: torch.Tensor,
    K_values: Optional[List[int]] = None,
    r_values: Optional[List[int]] = None,
    seed_A: Optional[int] = None,
    seed_B: Optional[int] = None,
    n_factors: int = 3,
    n_candidates: int = 1,
) -> List[Dict]:
    """
    Sweep (K, r) and compute error/compression tradeoff for each combination.

    Also computes the pure-SVD baseline (no transform sparsification) for comparison.

    If seed_A/seed_B are None, runs transform search with n_candidates.
    """
    W = W.float()
    m, n = W.shape
    mn = m * n

    if K_values is None:
        K_values = [mn // 32, mn // 16, mn // 8, mn // 4, mn // 2,
                    3 * mn // 4, 7 * mn // 8, 15 * mn // 16]
    if r_values is None:
        min_dim = min(m, n)
        # Dense r-grid: avoid jumping straight to max r
        r_values = sorted(set(
            list(range(0, min(65, min_dim + 1), 8))         # 0,8,16,...,64
            + list(range(64, min(257, min_dim + 1), 32))    # 64,96,...,256
            + list(range(256, min(513, min_dim + 1), 64))   # 256,320,...,512
            + list(range(512, min_dim + 1, 128))            # 512,640,...
        ))
        r_values = [r for r in r_values if 0 <= r <= min_dim]

    K_values = [min(max(1, k), mn) for k in K_values]
    r_values = [min(max(0, r), min(m, n)) for r in r_values]

    W_norm = W.norm().item()

    # Select transforms: use provided seeds or auto-search
    if seed_A is None or seed_B is None:
        K_test = max(1, mn // 4)
        seed_A, seed_B, _ = select_transforms(
            W, K=K_test, n_candidates=max(n_candidates, 4), n_factors=n_factors,
        )

    # Build transform and compute W̃ once (no dense matrix materialization)
    transform_A = build_fast_orthogonal(m, seed=seed_A, device=W.device, n_factors=n_factors)
    transform_B = build_fast_orthogonal(n, seed=seed_B, device=W.device, n_factors=n_factors)
    W_tilde = compute_W_tilde(W, transform_A, transform_B)

    # Pure SVD baseline — svdvals only (faster, no U/V)
    sigma_W = torch.linalg.svdvals(W)
    svd_results = []
    for r in r_values:
        if r == 0:
            continue
        r_c = min(r, len(sigma_W))
        svd_error = sigma_W[r_c:].square().sum().sqrt().item()
        svd_params = r_c * (m + n)
        svd_results.append({
            "method": "SVD-only",
            "K": 0,
            "r": r_c,
            "total_error": svd_error,
            "relative_error": svd_error / max(W_norm, 1e-10),
            "zebra_params": svd_params,
            "dense_params": mn,
            "compression_ratio": mn / max(svd_params, 1),
        })

    # Tail identity sanity check (1.3): verify on first K value
    if K_values:
        _K_check = K_values[0]
        _idx_check, _val_check = top_k_indices_only(W_tilde, _K_check)
        # Build sparse version for tail energy
        _Wk_check = torch.zeros_like(W_tilde)
        if len(_val_check) > 0:
            _Wk_check[_idx_check[:, 0], _idx_check[:, 1]] = _val_check
        _tail_sq = (W_tilde - _Wk_check).square().sum().item()
        _recon_check = reconstruct_from_indices(
            _idx_check, _val_check, m, n, transform_A, transform_B
        )
        _recon_sq = (W - _recon_check).square().sum().item()
        _mismatch = abs(_recon_sq - _tail_sq) / max(1.0, _tail_sq)
        if _mismatch > 1e-3:
            import warnings
            warnings.warn(
                f"compressibility_sweep: tail identity mismatch "
                f"(rel diff {_mismatch:.2e}). Transform convention may be broken."
            )

    # ZEBRA sweep
    # Validation: spot-check one (K, r) config with full reconstruction
    _validated = False

    zebra_results = []
    for ki, K in enumerate(K_values):
        # Reconstruct W_{A,B,K} from sparse indices — avoids dense W̃_K allocation
        _indices, _values = top_k_indices_only(W_tilde, K)
        W_ABK = reconstruct_from_indices(
            _indices, _values, m, n, transform_A, transform_B
        )
        R = W - W_ABK
        R_norm_sq = R.square().sum().item()

        # Always use exact svdvals for analysis correctness
        sigma_R = torch.linalg.svdvals(R)
        sigma_sq_cumsum = sigma_R.square().cumsum(dim=0)

        # Spot-check: on the first K, verify svdvals-based error matches
        # actual reconstruction error for a mid-range r.
        if not _validated and ki == 0 and len(sigma_R) > 1:
            # Pick a safe r strictly below full rank to avoid tail=0 edge case
            _check_r = min(max(1, len(sigma_R) // 2), len(sigma_R) - 1)
            _U, _S, _Vh = torch.linalg.svd(R, full_matrices=False)
            _Rr = (_U[:, :_check_r] * _S[:_check_r]) @ _Vh[:_check_r, :]
            _true_err = (W - W_ABK - _Rr).norm().item()
            _est_sq = max(R_norm_sq - sigma_sq_cumsum[_check_r - 1].item(), 0.0)
            _est_err = math.sqrt(_est_sq)
            # Normalize by W_norm (stable even when est_err ≈ 0)
            _check_mismatch = abs(_true_err - _est_err) / max(W_norm, 1e-10)
            if _check_mismatch > 1e-3:
                import warnings
                warnings.warn(
                    f"compressibility_sweep: reconstruction spot-check mismatch: "
                    f"true_err={_true_err:.6f} vs estimated={_est_err:.6f} "
                    f"(rel diff vs W_norm: {_check_mismatch:.2e}). "
                    f"Low-rank correction may not match svdvals estimate."
                )
            _validated = True

        for r in r_values:
            r_c = min(r, len(sigma_R))
            if r_c > 0:
                # tail = total - sum of top r_c squared singular values
                residual_sq = max(R_norm_sq - sigma_sq_cumsum[r_c - 1].item(), 0.0)
                residual_error = math.sqrt(residual_sq)
            else:
                residual_error = math.sqrt(max(R_norm_sq, 0.0))

            zebra_params = K * 3 + r_c * (m + n)  # 3K (value+2 idx) + low-rank
            zebra_params_values_only = K + r_c * (m + n)
            zebra_results.append({
                "method": "ZEBRA",
                "K": K,
                "r": r_c,
                "total_error": residual_error,
                "relative_error": residual_error / max(W_norm, 1e-10),
                "zebra_params": zebra_params,
                "zebra_params_values_only": zebra_params_values_only,
                "dense_params": mn,
                "compression_ratio": mn / max(zebra_params, 1),
            })

    return svd_results + zebra_results
