"""
Tests for ZEBRA — verify paper theorems on actual matrices.

Tests:
  1. Hadamard basics: orthogonality, self-inverse, fast transform
  2. Fast orthogonal family: Q^T Q = I, roundtrip
  3. Top-K optimality: P_K is best K-sparse under fixed (A,B)
  4. Lemma (Tail energy): ||W - W_{A,B,K}||² = Σ_{∉Ω_K} |W̃_ij|²
  5. Lemma (EYM): ||R - R_r||² = Σ_{j>r} σ_j(R)²
  6. Theorem (ZEBRA decomposition): full error identity
  7. ZEBRALinear matches dense: from_linear(L).forward(x) ≈ L(x)
  8. ZEBRA vs pure SVD: comparison at matched params
  9. Rectangular matrices: ZEBRA works for m ≠ n
  10. Edge cases: identity, zero, small dimensions
"""

import math
import pytest
import torch
import torch.nn as nn

from zebra.core import (
    hadamard_matrix,
    fast_hadamard_transform,
    random_sign_diagonal,
    random_permutation,
    build_fast_orthogonal,
    apply_fast_orthogonal,
    build_orthogonal_matrix,
    compute_W_tilde,
    reconstruct_from_sparse,
    top_k_sparsify,
    top_k_tail_energy,
    zebra_decompose,
    certificate_curve_sparse,
    certificate_curve_residual,
    select_transforms,
    analyze_compressibility,
    compressibility_sweep,
)
from zebra.layer import ZEBRALinear


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-4  # float32 tolerance


# ============================================================================
# Test 1: Hadamard basics
# ============================================================================

class TestHadamardBasics:
    def test_hadamard_orthogonal(self):
        """H^T H = I for normalized Hadamard."""
        for d in [2, 4, 8, 16, 64]:
            H = hadamard_matrix(d, device=DEVICE)
            I_approx = H.T @ H
            I_exact = torch.eye(d, device=DEVICE)
            assert torch.allclose(I_approx, I_exact, atol=TOL), (
                f"H^T H ≠ I for d={d}, max diff = {(I_approx - I_exact).abs().max():.2e}"
            )

    def test_hadamard_symmetric(self):
        """Normalized Hadamard is symmetric: H = H^T."""
        for d in [2, 4, 8, 32]:
            H = hadamard_matrix(d, device=DEVICE)
            assert torch.allclose(H, H.T, atol=TOL), f"H ≠ H^T for d={d}"

    def test_hadamard_self_inverse(self):
        """H @ H = I (up to sign, since H is orthogonal and symmetric)."""
        for d in [2, 4, 16]:
            H = hadamard_matrix(d, device=DEVICE)
            HH = H @ H
            I_exact = torch.eye(d, device=DEVICE)
            assert torch.allclose(HH, I_exact, atol=TOL), f"H² ≠ I for d={d}"

    def test_fast_hadamard_matches_dense(self):
        """Fast (butterfly) Hadamard matches dense matrix product."""
        torch.manual_seed(42)
        for d in [4, 16, 64, 128]:
            H = hadamard_matrix(d, device=DEVICE)
            x = torch.randn(8, d, device=DEVICE)

            y_dense = x @ H.T  # H is symmetric, so H.T = H
            y_fast = fast_hadamard_transform(x, dim=-1)

            assert torch.allclose(y_dense, y_fast, atol=TOL), (
                f"Fast Hadamard mismatch for d={d}, "
                f"max diff = {(y_dense - y_fast).abs().max():.2e}"
            )

    def test_fast_hadamard_batched(self):
        """Fast Hadamard works with batched inputs."""
        torch.manual_seed(42)
        d = 32
        x = torch.randn(4, 8, d, device=DEVICE)
        H = hadamard_matrix(d, device=DEVICE)

        y_dense = x @ H
        y_fast = fast_hadamard_transform(x, dim=-1)

        assert torch.allclose(y_dense, y_fast, atol=TOL)

    def test_fast_hadamard_roundtrip(self):
        """Two applications of normalized Hadamard = identity."""
        torch.manual_seed(42)
        d = 64
        x = torch.randn(4, d, device=DEVICE)
        y = fast_hadamard_transform(x, dim=-1)
        x_back = fast_hadamard_transform(y, dim=-1)
        assert torch.allclose(x, x_back, atol=TOL)

    def test_hadamard_power_of_2_required(self):
        """Non-power-of-2 dimensions raise an error."""
        with pytest.raises(AssertionError):
            hadamard_matrix(3)
        with pytest.raises(AssertionError):
            hadamard_matrix(7)


# ============================================================================
# Test 2: Fast orthogonal family
# ============================================================================

class TestFastOrthogonalFamily:
    def test_orthogonality(self):
        """Q^T Q = I for scrambled Hadamard transforms."""
        for d in [4, 16, 32, 64]:
            transform = build_fast_orthogonal(d, seed=42, device=DEVICE)
            Q = build_orthogonal_matrix(transform, device=DEVICE)
            I_approx = Q.T @ Q
            I_exact = torch.eye(d, device=DEVICE)
            assert torch.allclose(I_approx, I_exact, atol=TOL), (
                f"Q^T Q ≠ I for d={d}, max diff = {(I_approx - I_exact).abs().max():.2e}"
            )

    def test_roundtrip(self):
        """Q^T (Qx) = x (apply forward then inverse)."""
        torch.manual_seed(42)
        for d in [8, 32, 64]:
            transform = build_fast_orthogonal(d, seed=42, device=DEVICE)
            x = torch.randn(4, d, device=DEVICE)

            Qx = apply_fast_orthogonal(x, transform, transpose=False, dim=-1)
            x_back = apply_fast_orthogonal(Qx, transform, transpose=True, dim=-1)

            assert torch.allclose(x, x_back, atol=1e-3), (
                f"Roundtrip failed for d={d}, "
                f"max diff = {(x - x_back).abs().max():.2e}"
            )

    def test_apply_matches_matrix(self):
        """apply_fast_orthogonal matches explicit matrix multiplication."""
        torch.manual_seed(42)
        for d in [8, 16, 32]:
            transform = build_fast_orthogonal(d, seed=123, device=DEVICE)
            Q = build_orthogonal_matrix(transform, device=DEVICE)
            x = torch.randn(4, d, device=DEVICE)

            y_mat = x @ Q.T  # Q acts on rows, so x @ Q^T = (Qx^T)^T
            y_fast = apply_fast_orthogonal(x, transform, transpose=True, dim=-1)

            # The matrix Q is built column-by-column via forward transform
            # Q[i] = apply_forward(e_i), so Q @ e_i = Q[i]
            # Thus Qx <=> apply_forward(x) and Q^Tx <=> apply_transpose(x)

            # Q is built so Q[i] = forward(e_i).
            # By linearity, forward(x) along last dim = x @ Q.
            # And transpose(x) = x @ Q^T.
            y_forward = apply_fast_orthogonal(x, transform, transpose=False, dim=-1)
            y_mat_forward = x @ Q
            assert torch.allclose(y_mat_forward, y_forward, atol=1e-3)

            y_transpose = apply_fast_orthogonal(x, transform, transpose=True, dim=-1)
            y_mat_transpose = x @ Q.T
            assert torch.allclose(y_mat_transpose, y_transpose, atol=1e-3)

    def test_different_seeds_give_different_transforms(self):
        """Different seeds produce different orthogonal matrices."""
        d = 32
        Q1 = build_orthogonal_matrix(
            build_fast_orthogonal(d, seed=1, device=DEVICE), device=DEVICE
        )
        Q2 = build_orthogonal_matrix(
            build_fast_orthogonal(d, seed=2, device=DEVICE), device=DEVICE
        )
        assert not torch.allclose(Q1, Q2, atol=0.1), (
            "Different seeds should produce different transforms"
        )

    def test_sign_diagonal_properties(self):
        """Random sign diagonal has entries in {-1, +1}."""
        signs = random_sign_diagonal(64, seed=42, device=DEVICE)
        assert torch.all((signs == 1.0) | (signs == -1.0))
        # D^T D = I since D is diagonal with |entries| = 1
        D = torch.diag(signs)
        I_approx = D.T @ D
        assert torch.allclose(I_approx, torch.eye(64, device=DEVICE), atol=TOL)

    def test_permutation_properties(self):
        """Random permutation is a valid permutation (bijection)."""
        perm = random_permutation(64, seed=42, device=DEVICE)
        assert len(perm.unique()) == 64, "Permutation must be a bijection"

    def test_non_power_of_2_orthogonality(self):
        """C1: Q^T Q = I for non-power-of-2 dimensions (block-diagonal)."""
        for d in [3, 5, 7, 12, 48, 96, 100, 200]:
            transform = build_fast_orthogonal(d, seed=42, device=DEVICE)
            Q = build_orthogonal_matrix(transform, device=DEVICE)
            I_approx = Q.T @ Q
            I_exact = torch.eye(d, device=DEVICE)
            assert torch.allclose(I_approx, I_exact, atol=1e-3), (
                f"Q^T Q ≠ I for d={d}, max diff = {(I_approx - I_exact).abs().max():.2e}"
            )

    def test_non_power_of_2_roundtrip(self):
        """C1: Q^T(Qx) = x for non-power-of-2 dimensions."""
        torch.manual_seed(42)
        for d in [3, 7, 48, 96, 100]:
            transform = build_fast_orthogonal(d, seed=42, device=DEVICE)
            x = torch.randn(4, d, device=DEVICE)

            Qx = apply_fast_orthogonal(x, transform, transpose=False, dim=-1)
            x_back = apply_fast_orthogonal(Qx, transform, transpose=True, dim=-1)

            assert torch.allclose(x, x_back, atol=1e-3), (
                f"Roundtrip failed for d={d}, "
                f"max diff = {(x - x_back).abs().max():.2e}"
            )

    def test_non_power_of_2_norm_preservation(self):
        """C1: ||Qx|| = ||x|| for non-power-of-2 dimensions."""
        torch.manual_seed(42)
        for d in [3, 7, 48, 96]:
            transform = build_fast_orthogonal(d, seed=42, device=DEVICE)
            x = torch.randn(8, d, device=DEVICE)

            Qx = apply_fast_orthogonal(x, transform, transpose=False, dim=-1)

            for i in range(x.shape[0]):
                x_norm = x[i].norm().item()
                Qx_norm = Qx[i].norm().item()
                assert abs(x_norm - Qx_norm) < 1e-3 * x_norm, (
                    f"d={d}: ||x|| = {x_norm:.6f}, ||Qx|| = {Qx_norm:.6f}"
                )

    def test_compute_W_tilde_matches_dense(self):
        """compute_W_tilde matches explicit A @ W @ B^T for any dim."""
        torch.manual_seed(42)
        for m, n in [(8, 8), (16, 32), (48, 96), (7, 13)]:
            W = torch.randn(m, n, device=DEVICE)
            tA = build_fast_orthogonal(m, seed=10, device=DEVICE)
            tB = build_fast_orthogonal(n, seed=20, device=DEVICE)

            # Fast path
            W_tilde_fast = compute_W_tilde(W, tA, tB)

            # Dense path
            A_mat = build_orthogonal_matrix(tA, device=DEVICE)
            B_mat = build_orthogonal_matrix(tB, device=DEVICE)
            W_tilde_dense = A_mat @ W @ B_mat.T

            assert torch.allclose(W_tilde_fast, W_tilde_dense, atol=1e-3), (
                f"({m},{n}): compute_W_tilde mismatch, "
                f"max diff = {(W_tilde_fast - W_tilde_dense).abs().max():.2e}"
            )

    def test_reconstruct_from_sparse_matches_dense(self):
        """reconstruct_from_sparse matches explicit A^T @ S @ B for any dim."""
        torch.manual_seed(42)
        for m, n in [(8, 8), (16, 32), (48, 96), (7, 13)]:
            S = torch.randn(m, n, device=DEVICE)
            tA = build_fast_orthogonal(m, seed=10, device=DEVICE)
            tB = build_fast_orthogonal(n, seed=20, device=DEVICE)

            # Fast path
            W_fast = reconstruct_from_sparse(S, tA, tB)

            # Dense path
            A_mat = build_orthogonal_matrix(tA, device=DEVICE)
            B_mat = build_orthogonal_matrix(tB, device=DEVICE)
            W_dense = A_mat.T @ S @ B_mat

            assert torch.allclose(W_fast, W_dense, atol=1e-3), (
                f"({m},{n}): reconstruct mismatch, "
                f"max diff = {(W_fast - W_dense).abs().max():.2e}"
            )

    def test_convention_roundtrip_13x11(self):
        """1.1: Deterministic convention check at m=13, n=11 (non-power-of-2).

        Verifies:
        - compute_W_tilde(W, A, B) == A_mat @ W @ B_mat.T
        - reconstruct_from_sparse(W_tilde_K, A, B) == A_mat.T @ W_tilde_K @ B_mat
        - Full roundtrip: reconstruct(compute(W)) == W
        """
        torch.manual_seed(99)
        m, n = 13, 11
        W = torch.randn(m, n, device=DEVICE)
        tA = build_fast_orthogonal(m, seed=7, device=DEVICE)
        tB = build_fast_orthogonal(n, seed=13, device=DEVICE)

        A_mat = build_orthogonal_matrix(tA, device=DEVICE)
        B_mat = build_orthogonal_matrix(tB, device=DEVICE)

        # Forward: W̃ = A W B^T
        W_tilde_fast = compute_W_tilde(W, tA, tB)
        W_tilde_dense = A_mat @ W @ B_mat.T
        fwd_fro_err = (W_tilde_fast - W_tilde_dense).norm() / W_tilde_dense.norm()
        assert fwd_fro_err < 1e-4, f"compute_W_tilde Fro error: {fwd_fro_err:.2e}"

        # Inverse: A^T W̃ B
        W_back_fast = reconstruct_from_sparse(W_tilde_fast, tA, tB)
        W_back_dense = A_mat.T @ W_tilde_fast @ B_mat
        inv_fro_err = (W_back_fast - W_back_dense).norm() / W_back_dense.norm()
        assert inv_fro_err < 1e-4, f"reconstruct Fro error: {inv_fro_err:.2e}"

        # Full roundtrip: A^T (A W B^T) B == W
        roundtrip_err = (W_back_fast - W).norm() / W.norm()
        assert roundtrip_err < 1e-4, f"Roundtrip Fro error: {roundtrip_err:.2e}"


# ============================================================================
# Test 3: Top-K optimality (Lemma: optimality of top-K under fixed transforms)
# ============================================================================

class TestTopKOptimality:
    def test_top_k_keeps_largest(self):
        """P_K keeps exactly the K entries with largest absolute value.

        C2 fix: checks that the kept index set matches the true top-K indices.
        """
        torch.manual_seed(42)
        X = torch.randn(8, 8, device=DEVICE)
        K = 10

        sparse_X, indices, values = top_k_sparsify(X, K)

        # Check that exactly K entries are nonzero
        assert (sparse_X != 0).sum().item() == K

        # C2: Compare the actual kept index set against ground truth
        flat = X.reshape(-1)
        _, expected_flat_idx = flat.abs().topk(K)
        expected_rows = expected_flat_idx // X.shape[1]
        expected_cols = expected_flat_idx % X.shape[1]
        expected_set = set(zip(expected_rows.tolist(), expected_cols.tolist()))

        actual_set = set(zip(indices[:, 0].tolist(), indices[:, 1].tolist()))
        assert actual_set == expected_set, (
            f"Top-K index sets differ: got {len(actual_set & expected_set)}/{K} overlap"
        )

        # Also verify values match
        for r_idx, c_idx in actual_set:
            assert torch.isclose(sparse_X[r_idx, c_idx], X[r_idx, c_idx], atol=TOL)

    def test_top_k_is_optimal_sparse(self):
        """P_K(W̃) minimizes ||W̃ - X||_F over all K-sparse X."""
        torch.manual_seed(42)
        X = torch.randn(16, 16, device=DEVICE)
        K = 30

        sparse_X, _, _ = top_k_sparsify(X, K)
        opt_error = (X - sparse_X).square().sum().item()

        # Compare against random K-sparse matrices (they should be worse)
        for trial in range(10):
            gen = torch.Generator(device='cpu')
            gen.manual_seed(trial)
            random_idx = torch.randperm(256, generator=gen)[:K]
            random_sparse = torch.zeros_like(X).reshape(-1)
            random_sparse[random_idx] = X.reshape(-1)[random_idx]
            random_sparse = random_sparse.reshape(16, 16)
            random_error = (X - random_sparse).square().sum().item()
            assert opt_error <= random_error + TOL, (
                f"Top-K not optimal: {opt_error:.4f} > {random_error:.4f}"
            )

    def test_tail_energy_matches_dropped(self):
        """Tail energy = sum of squared dropped entries."""
        torch.manual_seed(42)
        X = torch.randn(16, 16, device=DEVICE)
        K = 50

        sparse_X, _, _ = top_k_sparsify(X, K)
        tail_energy = top_k_tail_energy(X, K)
        direct_error = (X - sparse_X).square().sum().item()

        assert abs(tail_energy - direct_error) < TOL * max(tail_energy, 1e-8)

    def test_full_K_zero_error(self):
        """K = mn should give zero tail energy."""
        X = torch.randn(8, 8, device=DEVICE)
        assert top_k_tail_energy(X, 64) < TOL

    def test_top_k_zero(self):
        """K=0 returns all-zero sparse_X and empty indices/values."""
        X = torch.randn(8, 8, device=DEVICE)
        sparse_X, indices, values = top_k_sparsify(X, 0)
        assert sparse_X.abs().max() == 0
        assert indices.shape == (0, 2)
        assert values.shape == (0,)

    def test_tail_energy_K_zero(self):
        """K=0 tail energy equals full ||X||²_F."""
        X = torch.randn(8, 8, device=DEVICE)
        tail = top_k_tail_energy(X, 0)
        expected = X.square().sum().item()
        assert abs(tail - expected) < TOL * max(expected, 1e-8)


# ============================================================================
# Test 4: Lemma (Orthogonal invariance / exact tail energy)
# ============================================================================

class TestTailEnergyLemma:
    def test_orthogonal_invariance(self):
        """||W - W_{A,B,K}||² = ||W̃ - W̃_K||² (orthogonal invariance)."""
        torch.manual_seed(42)
        d = 32

        for K in [50, 200, 500]:
            W = torch.randn(d, d, device=DEVICE)

            transform_A = build_fast_orthogonal(d, seed=0, device=DEVICE)
            transform_B = build_fast_orthogonal(d, seed=1, device=DEVICE)
            A_mat = build_orthogonal_matrix(transform_A, device=DEVICE)
            B_mat = build_orthogonal_matrix(transform_B, device=DEVICE)

            W_tilde = A_mat @ W @ B_mat.T
            W_tilde_K, _, _ = top_k_sparsify(W_tilde, K)

            # Transform-domain error
            transform_error_sq = (W_tilde - W_tilde_K).square().sum().item()

            # Weight-space error
            W_ABK = A_mat.T @ W_tilde_K @ B_mat
            weight_error_sq = (W - W_ABK).square().sum().item()

            assert abs(transform_error_sq - weight_error_sq) < TOL * max(transform_error_sq, 1e-8), (
                f"K={K}: transform error² = {transform_error_sq:.6f}, "
                f"weight error² = {weight_error_sq:.6f}"
            )

    def test_tail_energy_equals_dropped_entries(self):
        """||W - W_{A,B,K}||² = Σ_{∉Ω_K} |W̃_ij|² (exact tail energy)."""
        torch.manual_seed(42)
        d = 32
        W = torch.randn(d, d, device=DEVICE)

        transform_A = build_fast_orthogonal(d, seed=0, device=DEVICE)
        transform_B = build_fast_orthogonal(d, seed=1, device=DEVICE)
        A_mat = build_orthogonal_matrix(transform_A, device=DEVICE)
        B_mat = build_orthogonal_matrix(transform_B, device=DEVICE)

        W_tilde = A_mat @ W @ B_mat.T

        for K in [100, 300, 800]:
            W_tilde_K, indices, _ = top_k_sparsify(W_tilde, K)
            W_ABK = A_mat.T @ W_tilde_K @ B_mat

            # LHS: weight-space error
            lhs = (W - W_ABK).square().sum().item()

            # RHS: sum of squared dropped entries
            rhs = top_k_tail_energy(W_tilde, K)

            assert abs(lhs - rhs) < TOL * max(lhs, 1e-8), (
                f"K={K}: ||W-W_ABK||² = {lhs:.6f}, tail energy = {rhs:.6f}"
            )


# ============================================================================
# Test 5: Eckart-Young-Mirsky for residual correction
# ============================================================================

class TestEckartYoungMirsky:
    def test_eym_identity(self):
        """||R - R_r||² = Σ_{j>r} σ_j(R)² for best rank-r approx."""
        torch.manual_seed(42)
        R = torch.randn(32, 32, device=DEVICE)

        U, sigma, Vt = torch.linalg.svd(R, full_matrices=False)

        for r in [2, 4, 8, 16]:
            R_r = U[:, :r] @ torch.diag(sigma[:r]) @ Vt[:r, :]
            actual_error_sq = (R - R_r).square().sum().item()
            expected_error_sq = sigma[r:].square().sum().item()

            assert abs(actual_error_sq - expected_error_sq) < TOL * max(actual_error_sq, 1e-8), (
                f"r={r}: actual = {actual_error_sq:.6f}, expected = {expected_error_sq:.6f}"
            )

    def test_eym_rectangular(self):
        """EYM also works for rectangular matrices."""
        torch.manual_seed(42)
        R = torch.randn(64, 32, device=DEVICE)

        U, sigma, Vt = torch.linalg.svd(R, full_matrices=False)

        for r in [2, 8, 16]:
            R_r = U[:, :r] @ torch.diag(sigma[:r]) @ Vt[:r, :]
            actual_error_sq = (R - R_r).square().sum().item()
            expected_error_sq = sigma[r:].square().sum().item()

            assert abs(actual_error_sq - expected_error_sq) < TOL * max(actual_error_sq, 1e-8)


# ============================================================================
# Test 6: Theorem (ZEBRA error decomposition with certificates)
# ============================================================================

class TestZEBRADecomposition:
    def test_error_decomposition(self):
        """Full ZEBRA error = residual SVD tail (Theorem 1)."""
        torch.manual_seed(42)
        d = 32

        W = torch.randn(d, d, device=DEVICE)

        for K, r in [(100, 4), (200, 8), (50, 16)]:
            result = zebra_decompose(W, K=K, r=r)

            # Total error should match residual SVD tail
            total_err_sq = result["total_error_sq"]
            residual_tail_sq = result["residual_tail_sq"]

            assert abs(total_err_sq - residual_tail_sq) / max(total_err_sq, 1e-8) < 5e-3, (
                f"K={K}, r={r}: total² = {total_err_sq:.6f}, "
                f"residual tail² = {residual_tail_sq:.6f}"
            )

    def test_sparse_error_identity(self):
        """Sparse part: ||W - W_{A,B,K}||² = tail energy (first certificate)."""
        torch.manual_seed(42)
        W = torch.randn(32, 32, device=DEVICE)

        result = zebra_decompose(W, K=200, r=4)
        assert abs(result["tail_energy_sq"] - result["recon_error_sq"]) < TOL * max(result["tail_energy_sq"], 1e-8)

    def test_certificate_curves(self):
        """Certificate curve functions return valid, monotonically decreasing errors."""
        torch.manual_seed(42)
        W = torch.randn(32, 32, device=DEVICE)

        # Sparse certificate
        K_values = [50, 100, 200, 500, 800]
        sparse_curve = certificate_curve_sparse(W, K_values=K_values)

        # Error should decrease with K
        for i in range(len(sparse_curve) - 1):
            assert sparse_curve[i]["tail_energy_sq"] >= sparse_curve[i + 1]["tail_energy_sq"] - TOL

        # Residual certificate (using some residual)
        result = zebra_decompose(W, K=200, r=0)
        A_mat = build_orthogonal_matrix(result["transform_A"], device=DEVICE)
        B_mat = build_orthogonal_matrix(result["transform_B"], device=DEVICE)
        W_ABK = A_mat.T @ result["W_tilde_K"] @ B_mat
        R = W - W_ABK

        r_values = [0, 1, 2, 4, 8, 16]
        residual_curve = certificate_curve_residual(R, r_values=r_values)

        for i in range(len(residual_curve) - 1):
            assert residual_curve[i]["tail_energy_sq"] >= residual_curve[i + 1]["tail_energy_sq"] - TOL

    def test_rectangular_decomposition(self):
        """ZEBRA works for rectangular matrices."""
        torch.manual_seed(42)

        for m, n in [(64, 32), (32, 64), (128, 48)]:
            W = torch.randn(m, n, device=DEVICE)
            mn = m * n

            result = zebra_decompose(W, K=mn // 4, r=4)

            assert result["m"] == m
            assert result["n"] == n
            assert result["total_error"] >= 0
            assert result["total_relative_error"] < 1.0  # Should be < 100% error

    def test_non_power_of_2_certificates(self):
        """C3: Certificate curves work for non-power-of-2 dimensions."""
        torch.manual_seed(42)
        for m, n in [(48, 96), (7, 13), (100, 50)]:
            W = torch.randn(m, n, device=DEVICE)
            mn = m * n

            # Full decomposition
            result = zebra_decompose(W, K=mn // 4, r=4)
            assert result["total_error"] >= 0
            assert result["total_relative_error"] < 1.0

            # Sparse certificate curve
            K_vals = [mn // 8, mn // 4, mn // 2]
            sparse_curve = certificate_curve_sparse(W, K_values=K_vals)
            assert len(sparse_curve) == 3
            # Monotonically decreasing
            for i in range(len(sparse_curve) - 1):
                assert sparse_curve[i]["tail_energy_sq"] >= sparse_curve[i + 1]["tail_energy_sq"] - TOL

    def test_non_power_of_2_error_decomposition(self):
        """C3: Full ZEBRA error identity holds for non-power-of-2 dims."""
        torch.manual_seed(42)
        for m, n in [(48, 96), (100, 50)]:
            W = torch.randn(m, n, device=DEVICE)
            mn = m * n
            result = zebra_decompose(W, K=mn // 4, r=4)

            total_err_sq = result["total_error_sq"]
            residual_tail_sq = result["residual_tail_sq"]

            assert abs(total_err_sq - residual_tail_sq) / max(total_err_sq, 1e-8) < 5e-3, (
                f"({m},{n}): total² = {total_err_sq:.6f}, "
                f"residual tail² = {residual_tail_sq:.6f}"
            )


# ============================================================================
# Test 7: ZEBRALinear matches dense
# ============================================================================

class TestZEBRALinear:
    def test_from_linear_roundtrip(self):
        """ZEBRALinear.forward(x) ≈ Linear.forward(x) within compression error."""
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=True).to(DEVICE)
        nn.init.xavier_uniform_(linear.weight)

        # Heavy compression (K=d*d, r=d) should be near-exact
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=d * d, r=d, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        x = torch.randn(4, d, device=DEVICE)
        y_dense = linear(x)
        y_zebra = zebra_layer(x)

        max_diff = (y_dense - y_zebra).abs().max().item()
        assert max_diff < 1e-2, f"Max diff with full K, full r: {max_diff:.6e}"

    def test_sparse_only_forward(self):
        """With r=0, ZEBRALinear uses only the sparse transform component."""
        torch.manual_seed(0)
        d = 32
        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        nn.init.normal_(linear.weight)

        zebra_layer = ZEBRALinear.from_linear(
            linear, K=d * d, r=0, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        x = torch.randn(3, d, device=DEVICE)
        y = zebra_layer(x)

        # Should match the sparse-only reconstruction
        W_hat = zebra_layer._reconstruct_weight()
        y_expected = x @ W_hat.T
        assert torch.allclose(y, y_expected, atol=1e-3), (
            f"Sparse-only max diff: {(y - y_expected).abs().max().item():.2e}"
        )

    def test_compressed_forward(self):
        """ZEBRALinear with K<mn, r<min(m,n) produces correct shape."""
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=True).to(DEVICE)
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=d * d // 4, r=8, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        x = torch.randn(4, 8, d, device=DEVICE)
        y = zebra_layer(x)
        assert y.shape == x.shape

    def test_rectangular_forward(self):
        """ZEBRALinear works for rectangular weight matrices."""
        torch.manual_seed(42)

        # out > in
        linear1 = nn.Linear(32, 64, bias=True).to(DEVICE)
        zebra1 = ZEBRALinear.from_linear(
            linear1, K=512, r=4, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)
        x1 = torch.randn(4, 32, device=DEVICE)
        y1 = zebra1(x1)
        assert y1.shape == (4, 64)

        # in > out
        linear2 = nn.Linear(64, 32, bias=True).to(DEVICE)
        zebra2 = ZEBRALinear.from_linear(
            linear2, K=512, r=4, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)
        x2 = torch.randn(4, 64, device=DEVICE)
        y2 = zebra2(x2)
        assert y2.shape == (4, 32)

    def test_compression_info(self):
        """compression_info reports correct stats."""
        d = 64
        K = 256
        r = 8

        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=K, r=r, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        info = zebra_layer.compression_info()
        assert info["out_features"] == d
        assert info["in_features"] == d
        assert info["K"] == K
        assert info["r"] == r
        assert info["dense_params"] == d * d
        # 3K (value + 2 idx) + r(m+n) + descriptor overhead 2*(m+n)
        expected = K * 3 + 2 * r * d + 2 * (d + d)
        assert info["zebra_params"] == expected
        assert "descriptor_overhead" in info
        assert "params_ratio" in info
        assert info["compression_ratio"] > 0

    def test_reconstruction_error(self):
        """reconstruction_error matches actual weight difference."""
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=d * d // 4, r=8, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        err = zebra_layer.reconstruction_error(linear)
        assert err >= 0
        assert err < 1.0  # Below 100% error

    def test_no_bias(self):
        """Works without bias."""
        d = 32
        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=128, r=4, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        x = torch.randn(2, d, device=DEVICE)
        y = zebra_layer(x)
        assert y.shape == (2, d)
        assert zebra_layer.bias is None

    def test_non_power_of_2_forward(self):
        """C1: ZEBRALinear works for non-power-of-2 dimensions."""
        torch.manual_seed(42)
        for m, n in [(48, 96), (100, 50), (7, 13)]:
            linear = nn.Linear(n, m, bias=True).to(DEVICE)
            nn.init.xavier_uniform_(linear.weight)

            zebra_layer = ZEBRALinear.from_linear(
                linear, K=m * n, r=min(m, n),
                n_candidates=1, seed_A=0, seed_B=1,
            ).to(DEVICE)

            x = torch.randn(4, n, device=DEVICE)
            y_dense = linear(x)
            y_zebra = zebra_layer(x)

            max_diff = (y_dense - y_zebra).abs().max().item()
            assert max_diff < 0.1, (
                f"Non-pow2 ({m},{n}): max diff = {max_diff:.6e}"
            )

    def test_sparse_coo_cache_is_sparse(self):
        """E1 validation: internal cache is truly sparse, not dense."""
        d = 32
        K = 64
        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=K, r=4, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        cache = zebra_layer._sparse_coo_cache
        assert cache is not None
        assert cache.is_sparse, "Cache should be a sparse tensor"
        assert cache.shape == (d, d)
        assert cache._nnz() == K

    def test_forward_bf16(self):
        """Forward works with bf16 inputs (on GPU)."""
        if DEVICE == "cpu":
            pytest.skip("bf16 matmul needs GPU")
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=True).to(DEVICE)
        zebra_layer = ZEBRALinear.from_linear(
            linear, K=d * d // 4, r=8, n_candidates=1, seed_A=0, seed_B=1,
        ).to(DEVICE)

        x_f32 = torch.randn(4, d, device=DEVICE)
        x_bf16 = x_f32.bfloat16()

        y_f32 = zebra_layer(x_f32)
        y_bf16 = zebra_layer(x_bf16)

        assert y_bf16.dtype == torch.bfloat16
        max_diff = (y_f32 - y_bf16.float()).abs().max().item()
        assert max_diff < 0.1, f"bf16 vs f32 max diff: {max_diff:.4f}"


# ============================================================================
# Test 8: ZEBRA vs pure SVD
# ============================================================================

class TestZEBRAvsSVD:
    def test_compressibility_sweep_returns_results(self):
        """compressibility_sweep produces valid results for both methods."""
        torch.manual_seed(42)
        d = 32

        W = torch.randn(d, d, device=DEVICE)
        results = compressibility_sweep(
            W,
            K_values=[64, 128, 256, 512],
            r_values=[0, 2, 4, 8],
        )

        assert len(results) > 0
        methods = set(r["method"] for r in results)
        assert "ZEBRA" in methods
        assert "SVD-only" in methods

        for r in results:
            assert r["total_error"] >= 0
            assert r["compression_ratio"] > 0

    def test_transform_selection_improves(self):
        """More candidates should find equal or better transforms."""
        torch.manual_seed(42)
        d = 32
        W = torch.randn(d, d, device=DEVICE)
        K = d * d // 4

        _, _, energy_1 = select_transforms(W, K=K, n_candidates=1)
        _, _, energy_4 = select_transforms(W, K=K, n_candidates=4)
        _, _, energy_16 = select_transforms(W, K=K, n_candidates=16)

        # More candidates can only help (or tie)
        assert energy_4 <= energy_1 + TOL
        assert energy_16 <= energy_4 + TOL


# ============================================================================
# Test 9: Edge cases
# ============================================================================

class TestEdgeCases:
    def test_identity_matrix(self):
        """ZEBRA of identity has bounded error."""
        d = 32
        W = torch.eye(d, device=DEVICE)
        result = zebra_decompose(W, K=d * d, r=d)
        # Full budget → near-zero error
        assert result["total_error"] < TOL

    def test_zero_matrix(self):
        """ZEBRA of zero matrix gives zero error."""
        d = 16
        W = torch.zeros(d, d, device=DEVICE)
        result = zebra_decompose(W, K=16, r=2)
        assert result["total_error"] < TOL
        assert result["tail_energy"] < TOL

    def test_small_d(self):
        """ZEBRA works for very small matrices (d=2, 4)."""
        for d in [2, 4]:
            W = torch.randn(d, d, device=DEVICE)
            result = zebra_decompose(W, K=d * d, r=d)
            assert result["total_error"] < 1e-3, (
                f"d={d}: error = {result['total_error']:.4f}"
            )

    def test_rank_1_matrix(self):
        """Rank-1 matrix should compress well with r=1."""
        torch.manual_seed(42)
        d = 32
        u = torch.randn(d, 1, device=DEVICE)
        v = torch.randn(d, 1, device=DEVICE)
        W = u @ v.T  # rank 1

        # Even with K=0, r=1 should capture almost everything
        result = zebra_decompose(W, K=0, r=1)
        # The low-rank component should capture the rank-1 structure
        # But K=0 means the sparse part contributes nothing,
        # so residual = W, and rank-1 SVD of rank-1 matrix is exact
        assert result["total_relative_error"] < 1e-3

    def test_frobenius_preservation(self):
        """||W||_F = ||W̃||_F (orthogonal transforms preserve Frobenius norm)."""
        torch.manual_seed(42)
        d = 32
        W = torch.randn(d, d, device=DEVICE)

        transform_A = build_fast_orthogonal(d, seed=0, device=DEVICE)
        transform_B = build_fast_orthogonal(d, seed=1, device=DEVICE)
        A_mat = build_orthogonal_matrix(transform_A, device=DEVICE)
        B_mat = build_orthogonal_matrix(transform_B, device=DEVICE)

        W_tilde = A_mat @ W @ B_mat.T

        W_norm = W.norm().item()
        W_tilde_norm = W_tilde.norm().item()

        assert abs(W_norm - W_tilde_norm) < TOL * W_norm, (
            f"||W|| = {W_norm:.6f}, ||W̃|| = {W_tilde_norm:.6f}"
        )


# ============================================================================
# Test 10: Analyze compressibility
# ============================================================================

class TestAnalyzeCompressibility:
    def test_returns_valid_fields(self):
        """analyze_compressibility returns required fields."""
        torch.manual_seed(42)
        W = torch.randn(32, 32, device=DEVICE)
        result = analyze_compressibility(W, name="test")

        assert "W_norm" in result
        assert "coeff_k50" in result
        assert "coeff_k90" in result
        assert "coeff_k95" in result
        assert "coeff_k99" in result
        assert "singular_values_W" in result
        assert result["W_norm"] > 0
        assert result["coeff_k50"] <= result["coeff_k90"] <= result["coeff_k95"] <= result["coeff_k99"]

    def test_rectangular_compressibility(self):
        """analyze_compressibility works for rectangular matrices."""
        torch.manual_seed(42)
        W = torch.randn(64, 32, device=DEVICE)
        result = analyze_compressibility(W, name="rect_test")

        assert result["m"] == 64
        assert result["n"] == 32
        assert result["mn"] == 2048
        assert result["W_norm"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
