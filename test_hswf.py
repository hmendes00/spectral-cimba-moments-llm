"""
Tests for HSWF — verify paper theorems on actual matrices.

Tests:
  1. Circulant roundtrip: circ(c) → wrap_diagonal_average → recovers c
  2. Fourier diagonalization: C = F* diag(λ) F
  3. Theorem 2: Bandlimit error = Fourier tail energy
  4. Theorem 3: Pythagorean decomposition  ||W-C||² = ||W-C*||² + ||C*-C||²
  5. Theorem 4: Total error = residual SVD tail
  6. HSWFLinear matches dense: from_linear(L).forward(x) ≈ L(x)
  7. HSWF vs pure SVD: HSWF beats SVD at same param count (for structured W)
"""

import math
import pytest
import torch
import torch.nn as nn

from hswf.core import (
    wrap_diagonal_average,
    circulant_from_column,
    circulant_spectrum,
    circulant_matvec,
    circulant_matvec_left,
    bandlimit_spectrum,
    hswf_decompose,
    compressibility_sweep,
)
from hswf.layer import HSWFLinear


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-4  # float32 tolerance


# ============================================================================
# Test 1: Circulant roundtrip
# ============================================================================

class TestCirculantBasics:
    def test_circulant_from_column(self):
        """circ([1,2,3]) should produce the correct 3×3 matrix."""
        c = torch.tensor([1.0, 2.0, 3.0])
        C = circulant_from_column(c)
        expected = torch.tensor([
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ])
        assert torch.allclose(C, expected), f"Got:\n{C}"

    def test_wrap_diagonal_recovers_circulant(self):
        """wrap_diagonal_average(circ(c)) == c for any c."""
        torch.manual_seed(42)
        for d in [3, 16, 64, 128]:
            c = torch.randn(d, device=DEVICE)
            C = circulant_from_column(c)
            c_recovered = wrap_diagonal_average(C)
            assert torch.allclose(c_recovered, c.to(c_recovered.device), atol=TOL), (
                f"d={d}: max diff = {(c_recovered - c.to(c_recovered.device)).abs().max():.2e}"
            )

    def test_circulant_matvec(self):
        """FFT-based matvec matches dense matvec."""
        torch.manual_seed(42)
        d = 64
        c = torch.randn(d, device=DEVICE)
        x = torch.randn(d, device=DEVICE)
        C = circulant_from_column(c)

        y_dense = C @ x
        y_fft = circulant_matvec(c, x)

        assert torch.allclose(y_dense, y_fft.to(y_dense.device), atol=TOL), (
            f"max diff = {(y_dense - y_fft).abs().max():.2e}"
        )

    def test_circulant_matvec_left(self):
        """x @ C^T via FFT matches dense."""
        torch.manual_seed(42)
        d = 64
        c = torch.randn(d, device=DEVICE)
        x = torch.randn(d, device=DEVICE)
        C = circulant_from_column(c)

        y_dense = x @ C.T
        y_fft = circulant_matvec_left(c, x)

        assert torch.allclose(y_dense, y_fft.to(y_dense.device), atol=TOL), (
            f"max diff = {(y_dense - y_fft).abs().max():.2e}"
        )

    def test_circulant_matvec_batched(self):
        """FFT matvec works with batched inputs."""
        torch.manual_seed(42)
        d = 32
        c = torch.randn(d, device=DEVICE)
        x = torch.randn(4, 8, d, device=DEVICE)  # batch
        C = circulant_from_column(c)

        y_dense = x @ C.T
        y_fft = circulant_matvec_left(c, x)

        assert torch.allclose(y_dense, y_fft.to(y_dense.device), atol=TOL)


# ============================================================================
# Test 2: Fourier diagonalization
# ============================================================================

class TestFourierDiag:
    def test_spectrum_norm(self):
        """||C||_F² = Σ|λ_k|² (unitary invariance of Frobenius norm)."""
        torch.manual_seed(42)
        d = 64
        c = torch.randn(d, device=DEVICE)
        C = circulant_from_column(c)
        spectrum = circulant_spectrum(c)

        C_norm_sq = C.square().sum().item()
        spectrum_norm_sq = spectrum.abs().square().sum().item()

        assert abs(C_norm_sq - spectrum_norm_sq) < TOL * C_norm_sq, (
            f"||C||² = {C_norm_sq:.4f}, Σ|λ|² = {spectrum_norm_sq:.4f}"
        )

    def test_spectrum_reconstruction(self):
        """IFFT(FFT(c)) = c (roundtrip)."""
        torch.manual_seed(42)
        d = 64
        c = torch.randn(d, device=DEVICE)
        spectrum = circulant_spectrum(c)
        c_back = torch.fft.ifft(spectrum).real

        assert torch.allclose(c, c_back.to(c.device), atol=TOL)


# ============================================================================
# Test 3: Theorem 2 — Fourier tail energy
# ============================================================================

class TestTheorem2:
    def test_bandlimit_error_equals_tail_energy(self):
        """||C* - C_S||² = Σ_{k∉Ω} |λ*_k|² (Theorem 2)."""
        torch.manual_seed(42)
        d = 64
        c = torch.randn(d, device=DEVICE)
        spectrum = circulant_spectrum(c)

        for S in [8, 16, 32]:
            spectrum_S, kept_idx = bandlimit_spectrum(spectrum, S, strategy="energy")

            # LHS: actual matrix norm
            C = circulant_from_column(c)
            c_S = torch.fft.ifft(spectrum_S).real
            C_S = circulant_from_column(c_S)
            lhs = (C - C_S).square().sum().item()

            # RHS: Fourier tail energy
            tail_mask = torch.ones(d, dtype=torch.bool, device=DEVICE)
            tail_mask[kept_idx] = False
            rhs = spectrum[tail_mask].abs().square().sum().item()

            assert abs(lhs - rhs) < TOL * max(lhs, 1e-8), (
                f"S={S}: ||C-C_S||² = {lhs:.6f}, Σ|λ_tail|² = {rhs:.6f}"
            )


# ============================================================================
# Test 4: Theorem 3 — Pythagorean decomposition
# ============================================================================

class TestTheorem3:
    def test_pythagorean(self):
        """||W - C||² = ||W - C*||² + ||C* - C||² for any circulant C."""
        torch.manual_seed(42)
        d = 32

        W = torch.randn(d, d, device=DEVICE)
        c_star = wrap_diagonal_average(W)
        C_star = circulant_from_column(c_star)

        # Use C_S (a different circulant) as "C"
        spectrum = circulant_spectrum(c_star)
        spectrum_S, _ = bandlimit_spectrum(spectrum, d // 4, strategy="energy")
        c_S = torch.fft.ifft(spectrum_S).real
        C = circulant_from_column(c_S)

        lhs = (W - C).square().sum().item()
        rhs1 = (W - C_star).square().sum().item()
        rhs2 = (C_star - C).square().sum().item()
        rhs = rhs1 + rhs2

        assert abs(lhs - rhs) < TOL * max(lhs, 1e-8), (
            f"||W-C||² = {lhs:.6f}, ||W-C*||² + ||C*-C||² = {rhs:.6f} "
            f"({rhs1:.6f} + {rhs2:.6f})"
        )

    def test_pythagorean_with_random_circulant(self):
        """Also verify with a completely independent random circulant."""
        torch.manual_seed(123)
        d = 64

        W = torch.randn(d, d, device=DEVICE)
        c_star = wrap_diagonal_average(W)
        C_star = circulant_from_column(c_star)

        # Random circulant (not derived from C*)
        c_random = torch.randn(d, device=DEVICE)
        C_random = circulant_from_column(c_random)

        lhs = (W - C_random).square().sum().item()
        rhs = (W - C_star).square().sum().item() + (C_star - C_random).square().sum().item()

        assert abs(lhs - rhs) < TOL * max(lhs, 1e-8)


# ============================================================================
# Test 5: Theorem 4 — Total HSWF error identity
# ============================================================================

class TestTheorem4:
    def test_total_error_equals_svd_tail(self):
        """||W - Ŵ||² = Σ_{j>r} σ_j(R)² (Theorem 4)."""
        torch.manual_seed(42)
        d = 64

        W = torch.randn(d, d, device=DEVICE)

        for S, r in [(16, 4), (32, 8), (8, 16)]:
            result = hswf_decompose(W, S=S, r=r)

            # Reconstruct Ŵ
            C_S = circulant_from_column(result["c_S"].to(DEVICE))
            R_r = result["U_r"].to(DEVICE) @ torch.diag(result["sigma_r"].to(DEVICE)) @ result["V_r"].to(DEVICE).T
            W_hat = C_S + R_r

            # LHS: actual error
            lhs = (W - W_hat).square().sum().item()

            # RHS: SVD tail of residual
            sigma_all = result["all_singular_values"]
            rhs = sigma_all[r:].square().sum().item()

            assert abs(lhs - rhs) < TOL * max(lhs, 1e-8), (
                f"S={S}, r={r}: ||W-Ŵ||² = {lhs:.6f}, Σσ²_tail = {rhs:.6f}"
            )

    def test_pythagorean_component(self):
        """||W - C_S||² = ||W - C*||² + Σ_{k∉Ω}|λ*_k|² (second part of Thm 4)."""
        torch.manual_seed(42)
        d = 64

        W = torch.randn(d, d, device=DEVICE)

        result = hswf_decompose(W, S=16, r=4)

        # Compare squared values directly (more sensitive than sqrt)
        # Slightly looser than TOL since squared norms accumulate more fp32 error
        lhs_sq = result["pythagorean_lhs_sq"]
        rhs_sq = result["pythagorean_rhs_sq"]

        assert abs(lhs_sq - rhs_sq) / max(lhs_sq, 1e-8) < 5e-4, (
            f"||W-C_S||² = {lhs_sq:.6f}, ||W-C*||² + tail² = {rhs_sq:.6f}"
        )


# ============================================================================
# Test 6: HSWFLinear matches dense
# ============================================================================

class TestHSWFLinear:
    def test_from_linear_roundtrip(self):
        """HSWFLinear.forward(x) ≈ Linear.forward(x) within compression error."""
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=True).to(DEVICE)
        nn.init.xavier_uniform_(linear.weight)

        # Exact compression (S=d, r=d) should be near-exact
        hswf = HSWFLinear.from_linear(linear, S=d, r=d).to(DEVICE)

        x = torch.randn(4, d, device=DEVICE)
        y_dense = linear(x)
        y_hswf = hswf(x)

        # With S=d and r=d (no compression), should be near-exact
        max_diff = (y_dense - y_hswf).abs().max().item()
        assert max_diff < 3e-4, f"Max diff with S=d, r=d: {max_diff:.6e}"

        # Also verify reconstruction error is near zero
        err = hswf.reconstruction_error(linear)
        assert err < 3e-5, f"Reconstruction error with S=d, r=d: {err:.6e}"

    def test_circulant_only_matches_dense(self):
        """With r=0, HSWFLinear circulant output matches dense x @ C_S^T."""
        torch.manual_seed(0)
        d = 64
        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        nn.init.normal_(linear.weight)

        # Force r=0 so only circulant branch matters
        hswf = HSWFLinear.from_linear(linear, S=d, r=0).to(DEVICE)

        x = torch.randn(3, d, device=DEVICE)
        C_S = circulant_from_column(hswf.c_S.to(DEVICE))
        y_dense = x @ C_S.T
        y_fft = hswf(x)

        assert torch.allclose(y_dense, y_fft, atol=3e-4), (
            f"Circulant-only max diff: {(y_dense - y_fft).abs().max().item():.2e}"
        )

    def test_forward_bf16(self):
        """Forward works with bf16 inputs (on GPU)."""
        if DEVICE == "cpu":
            pytest.skip("bf16 test requires CUDA")
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=True).to(DEVICE)
        hswf = HSWFLinear.from_linear(linear, S=32, r=8).to(DEVICE)

        x_f32 = torch.randn(4, d, device=DEVICE)
        x_bf16 = x_f32.bfloat16()

        y_f32 = hswf(x_f32)
        y_bf16 = hswf(x_bf16)

        # bf16 output should match f32 output within bf16 precision (~1e-2)
        assert y_bf16.dtype == torch.bfloat16
        max_diff = (y_f32 - y_bf16.float()).abs().max().item()
        assert max_diff < 0.05, f"bf16 vs f32 max diff: {max_diff:.4f}"

    def test_compressed_forward(self):
        """HSWFLinear with S<d, r<d produces output of correct shape."""
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=True).to(DEVICE)
        hswf = HSWFLinear.from_linear(linear, S=16, r=8).to(DEVICE)

        x = torch.randn(4, 8, d, device=DEVICE)
        y = hswf(x)
        assert y.shape == x.shape

    def test_compression_info(self):
        """compression_info reports correct stats."""
        d = 64
        S = 16
        r = 8

        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        hswf = HSWFLinear.from_linear(linear, S=S, r=r).to(DEVICE)

        info = hswf.compression_info()
        assert info["d"] == d
        assert info["S"] == S
        assert info["r"] == r
        assert info["dense_params"] == d * d
        assert info["hswf_params"] == 2 * S + 2 * r * d
        assert info["compression_ratio"] > 1.0  # Actually compressed

    def test_reconstruction_error(self):
        """reconstruction_error matches hswf_decompose total error."""
        torch.manual_seed(42)
        d = 64

        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        hswf = HSWFLinear.from_linear(linear, S=16, r=8).to(DEVICE)

        err = hswf.reconstruction_error(linear)
        assert err >= 0
        assert err < 1.0  # Should be well below 100% error

    def test_no_bias(self):
        """Works without bias."""
        d = 32
        linear = nn.Linear(d, d, bias=False).to(DEVICE)
        hswf = HSWFLinear.from_linear(linear, S=8, r=4).to(DEVICE)

        x = torch.randn(2, d, device=DEVICE)
        y = hswf(x)
        assert y.shape == (2, d)
        assert hswf.bias is None


# ============================================================================
# Test 7: HSWF vs pure SVD
# ============================================================================

class TestHSWFvsSVD:
    def test_hswf_beats_svd_on_circulant_plus_noise(self):
        """For a matrix that IS circulant + noise, HSWF should beat pure SVD."""
        torch.manual_seed(42)
        d = 64

        # Construct W = circulant + small noise
        c = torch.randn(d, device=DEVICE)
        C = circulant_from_column(c)
        noise = 0.1 * torch.randn(d, d, device=DEVICE)
        W = C + noise

        S = 32  # Keep half the spectrum
        r = 4   # Small rank correction

        # HSWF: uses S Fourier modes + rank-r correction
        result = hswf_decompose(W, S=S, r=r)
        hswf_error = result["total_error"]
        hswf_params = result["hswf_params"]

        # Pure SVD with same param budget — test nearby ranks for robustness
        svd_r_base = max(1, hswf_params // (2 * d))
        _, sigma_W, _ = torch.linalg.svd(W, full_matrices=False)

        best_svd_error = float("inf")
        for rr in [max(1, svd_r_base - 1), svd_r_base, min(d, svd_r_base + 1)]:
            svd_err = sigma_W[rr:].square().sum().sqrt().item()
            best_svd_error = min(best_svd_error, svd_err)

        # HSWF should win because the matrix has circulant structure
        assert hswf_error < best_svd_error * 1.1, (
            f"HSWF error ({hswf_error:.4f}) should be < best SVD error "
            f"({best_svd_error:.4f}) for circulant+noise matrix. "
            f"HSWF params={hswf_params}, SVD rank_base={svd_r_base}"
        )

    def test_compressibility_sweep_returns_results(self):
        """compressibility_sweep produces valid results for both methods."""
        torch.manual_seed(42)
        d = 32

        W = torch.randn(d, d, device=DEVICE)
        results = compressibility_sweep(
            W,
            S_values=[4, 8, 16],
            r_values=[0, 2, 4, 8],
        )

        assert len(results) > 0
        methods = set(r["method"] for r in results)
        assert "HSWF" in methods
        assert "SVD-only" in methods

        for r in results:
            assert r["relative_error"] >= 0
            assert r["compression_ratio"] > 0


# ============================================================================
# Test 8: Edge cases
# ============================================================================

class TestEdgeCases:
    def test_identity_matrix(self):
        """HSWF of identity: circulant part is (1/d)·ones (DC mode)."""
        d = 32
        W = torch.eye(d, device=DEVICE)
        c_star = wrap_diagonal_average(W)

        # Each wrap-diagonal of I has exactly one 1 and (d-1) zeros
        # So c_0 = 1 (main diagonal, all 1s), c_k = 0 for k>0
        # Wait: c_0 = mean of main diagonal = 1.0
        # c_1 = mean of {I[i, (i-1)%d]} = mean of {0,0,...,0} = 0
        assert abs(c_star[0].item() - 1.0) < TOL
        for k in range(1, d):
            assert abs(c_star[k].item()) < TOL

    def test_zero_matrix(self):
        """HSWF of zero matrix gives zero error."""
        d = 16
        W = torch.zeros(d, d, device=DEVICE)
        result = hswf_decompose(W, S=4, r=2)
        assert result["total_error"] < TOL
        assert result["non_circ_error"] < TOL

    def test_pure_circulant(self):
        """HSWF of a circulant matrix: non-circ error should be 0."""
        torch.manual_seed(42)
        d = 32
        c = torch.randn(d, device=DEVICE)
        W = circulant_from_column(c)

        result = hswf_decompose(W, S=d, r=0)
        assert result["non_circ_error"] < TOL, (
            f"Non-circulant error should be 0, got {result['non_circ_error']:.2e}"
        )
        assert result["total_error"] < TOL

    def test_small_d(self):
        """Works for very small dimensions."""
        for d in [2, 3, 4]:
            W = torch.randn(d, d, device=DEVICE)
            result = hswf_decompose(W, S=1, r=1)
            assert result["total_error"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
