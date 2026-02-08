"""
Tests for FMMA Moment Distillation (Softmax → FMMA normalization bridge).

Validates all components from PLAN.md §0–§8:
  - Phase mapping matches FMMAAttention
  - Teacher moment computation (magnitudes ≤ 1, m_0 ≈ 1, Hermitian symmetry)
  - Moment loss computation (shapes, gradients flow)
  - Position-aware moment storage in FMMAAttention
  - Layer/head/query subsampling
  - End-to-end integration test
"""

import math
import torch
import pytest

# Make sure fmma package is importable
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fmma.attention import FMMAAttention
from fmma.moment_distill import (
    phases_from_positions,
    compute_teacher_phase_moments,
    compute_moment_loss,
    select_query_indices,
    select_layers_for_moments,
    select_heads,
)


# ============================================================================
# §3.1 Phase mapping
# ============================================================================

class TestPhaseMapping:
    """Phase mapping must match FMMAAttention._get_phases exactly."""

    def test_matches_fmma_attention(self):
        max_seq_len = 8192
        fmma = FMMAAttention(hidden_size=64, num_heads=4, head_dim=16,
                             max_seq_len=max_seq_len)
        pos = torch.arange(0, 128)
        expected = fmma._get_phases(pos)
        actual = phases_from_positions(pos, max_seq_len)
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Phase mismatch: max diff = {(expected - actual).abs().max()}"

    def test_range_zero_to_2pi(self):
        phases = phases_from_positions(torch.arange(0, 100), 100)
        assert phases.min() >= 0
        assert phases.max() < 2 * math.pi + 1e-6

    def test_monotonic(self):
        phases = phases_from_positions(torch.arange(0, 50), 1000)
        diffs = phases[1:] - phases[:-1]
        assert (diffs > 0).all(), "Phases should be monotonically increasing"


# ============================================================================
# §3.2 Teacher phase moments
# ============================================================================

class TestTeacherPhaseMoments:
    """Teacher moments from softmax attention weights."""

    def _make_uniform_attn(self, B, H, T):
        """Uniform causal attention (lower triangular, row-normalized)."""
        mask = torch.tril(torch.ones(T, T))
        attn = mask / mask.sum(dim=-1, keepdim=True)
        return attn.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

    def test_basic_shape(self):
        B, H, T, S = 2, 4, 32, 8
        attn = self._make_uniform_attn(B, H, T)
        phases = phases_from_positions(torch.arange(T), 1000).unsqueeze(0).expand(B, -1)
        t_idx = torch.tensor([8, 16, 24])

        m = compute_teacher_phase_moments(attn, phases, t_idx, S)
        assert m.shape == (B, H, 3, 2 * S + 1), f"Expected (2, 4, 3, 17), got {m.shape}"
        assert m.dtype == torch.complex64

    def test_hermitian_symmetry(self):
        """m_{-k} = conj(m_k)"""
        B, H, T, S = 1, 2, 16, 4
        # Random attention (still valid causal)
        mask = torch.tril(torch.ones(T, T))
        raw = torch.rand(B, H, T, T) * mask
        attn = raw / raw.sum(dim=-1, keepdim=True)
        phases = phases_from_positions(torch.arange(T), 512).unsqueeze(0)
        t_idx = torch.tensor([4, 8, 12])

        m = compute_teacher_phase_moments(attn, phases, t_idx, S, use_fejer_weights=False)

        for k in range(1, S + 1):
            m_pos = m[..., S + k]
            m_neg = m[..., S - k]
            assert torch.allclose(m_neg, torch.conj(m_pos), atol=1e-5), \
                f"Hermitian symmetry violated at k={k}"

    def test_m0_is_one(self):
        """m_0 = Σ_i w_i = 1 (for normalized attention)."""
        B, H, T, S = 1, 2, 20, 6
        attn = self._make_uniform_attn(B, H, T)
        phases = phases_from_positions(torch.arange(T), 512).unsqueeze(0)
        t_idx = torch.tensor([5, 10, 15])

        m = compute_teacher_phase_moments(attn, phases, t_idx, S, use_fejer_weights=False)
        m_0 = m[..., S]  # DC component
        assert torch.allclose(m_0.real, torch.ones_like(m_0.real), atol=1e-5)
        assert torch.allclose(m_0.imag, torch.zeros_like(m_0.imag), atol=1e-5)

    def test_magnitudes_bounded(self):
        """|m_k| ≤ 1 for a probability measure."""
        B, H, T, S = 1, 4, 32, 12
        mask = torch.tril(torch.ones(T, T))
        raw = torch.rand(B, H, T, T) * mask
        attn = raw / raw.sum(dim=-1, keepdim=True)
        phases = phases_from_positions(torch.arange(T), 1024).unsqueeze(0)
        t_idx = torch.tensor([8, 16, 24])

        m = compute_teacher_phase_moments(attn, phases, t_idx, S, use_fejer_weights=False)
        mags = m.abs()
        assert mags.max() <= 1.05, f"Moment magnitude > 1: {mags.max()}"

    def test_head_subsampling(self):
        B, H, T, S = 1, 8, 16, 4
        attn = self._make_uniform_attn(B, H, T)
        phases = phases_from_positions(torch.arange(T), 512).unsqueeze(0)
        t_idx = torch.tensor([4, 8])
        h_idx = torch.tensor([0, 2, 4, 6])

        m = compute_teacher_phase_moments(attn, phases, t_idx, S, h_idx=h_idx)
        assert m.shape == (B, 4, 2, 2 * S + 1)

    def test_fejer_weights(self):
        """Fejér-weighted moments should have smaller magnitude at high k."""
        B, H, T, S = 1, 2, 20, 8
        attn = self._make_uniform_attn(B, H, T)
        phases = phases_from_positions(torch.arange(T), 512).unsqueeze(0)
        t_idx = torch.tensor([10])

        m_no_fejer = compute_teacher_phase_moments(attn, phases, t_idx, S, use_fejer_weights=False)
        m_fejer = compute_teacher_phase_moments(attn, phases, t_idx, S, use_fejer_weights=True)

        # Fejér weights taper to 0, so high-k moments should be smaller
        assert m_fejer[..., -1].abs().max() < m_no_fejer[..., -1].abs().max() + 1e-8


# ============================================================================
# §5 Moment loss
# ============================================================================

class TestMomentLoss:
    """Moment loss function."""

    def test_zero_when_equal(self):
        B, H, Q, r, S = 2, 4, 8, 3, 6
        mu = torch.randn(B, H, Q, r, 2 * S + 1) + 1j * torch.randn(B, H, Q, r, 2 * S + 1)
        m_teacher = mu.mean(dim=3)  # identical effective moments

        # Apply Fejér to teacher to match what compute_moment_loss will do
        k_indices = torch.arange(-S, S + 1, dtype=torch.float32)
        c_k = 1.0 - torch.abs(k_indices) / (S + 1)
        m_teacher = m_teacher * c_k.view(1, 1, 1, -1)

        loss = compute_moment_loss(mu, m_teacher, S, use_fejer_weights=True)
        assert loss.item() < 1e-8, f"Loss should be ~0 for identical moments, got {loss.item()}"

    def test_positive_when_different(self):
        B, H, Q, r, S = 1, 2, 4, 3, 4
        mu_student = torch.randn(B, H, Q, r, 2 * S + 1, dtype=torch.complex64)
        m_teacher = torch.randn(B, H, Q, 2 * S + 1, dtype=torch.complex64)

        loss = compute_moment_loss(mu_student, m_teacher, S)
        assert loss.item() > 0, "Loss should be positive for different moments"

    def test_gradient_flows(self):
        """Gradients should flow through moment loss to student parameters."""
        B, H, Q, r, S = 1, 2, 4, 3, 4
        mu_student = torch.randn(B, H, Q, r, 2 * S + 1, dtype=torch.complex64, requires_grad=True)
        m_teacher = torch.randn(B, H, Q, 2 * S + 1, dtype=torch.complex64)

        loss = compute_moment_loss(mu_student, m_teacher, S)
        loss.backward()
        assert mu_student.grad is not None, "Gradient should flow to student moments"
        assert mu_student.grad.abs().sum() > 0, "Gradient should be non-zero"


# ============================================================================
# §7 Subsampling
# ============================================================================

class TestSubsampling:
    """Layer/head/query subsampling strategies."""

    def test_select_query_indices(self):
        t_idx = select_query_indices(seq_len=128, mom_query_stride=8, mom_max_queries=16)
        assert len(t_idx) <= 16
        assert t_idx.min() >= 128 // 4  # should skip first quarter
        assert t_idx.max() < 128

    def test_select_query_indices_short_seq(self):
        t_idx = select_query_indices(seq_len=16, mom_query_stride=2, mom_max_queries=100)
        assert len(t_idx) > 0
        assert t_idx.max() < 16

    def test_select_layers(self):
        fmma_indices = [0, 1, 2, 3, 4, 5]
        selected = select_layers_for_moments(fmma_indices, None, 2)
        assert len(selected) <= 2
        assert all(l in fmma_indices for l in selected)

    def test_select_layers_explicit(self):
        fmma_indices = [0, 2, 4, 6]
        selected = select_layers_for_moments(fmma_indices, [0, 4], 10)
        assert set(selected) == {0, 4}

    def test_select_heads(self):
        h_idx = select_heads(16, mom_head_stride=4)
        expected = torch.tensor([0, 4, 8, 12])
        assert torch.equal(h_idx, expected)


# ============================================================================
# §4 Position-aware moment storage in FMMAAttention
# ============================================================================

class TestPositionAwareMomentStorage:
    """FMMAAttention stores queries; caller computes moments lazily."""

    def test_store_queries_for_lazy_moments(self):
        B, T, D, H = 1, 64, 64, 4
        r, S = 4, 8

        fmma = FMMAAttention(
            hidden_size=D, num_heads=H, head_dim=D // H,
            num_channels=r, bandwidth=S, max_seq_len=1024,
        )

        # Forward
        x = torch.randn(B, T, D)
        with torch.no_grad():
            fmma(x)

        # Check stored queries
        stored_q = fmma._last_queries
        assert stored_q is not None, "Should have stored projected queries"
        assert stored_q.shape == (B, H, T, D // H), \
            f"Expected shape (1, 4, 64, 16), got {stored_q.shape}"

        # Can lazily compute moments at specific positions
        req_positions = torch.tensor([8, 16, 32, 48])
        mu = fmma.query_moments(stored_q[:, :, req_positions])
        assert mu.shape == (B, H, len(req_positions), r, 2 * S + 1)

    def test_first_chunk_still_stored(self):
        """_last_query_moments (for PSD/BW reg) should still work."""
        B, T, D, H = 1, 64, 64, 4

        fmma = FMMAAttention(
            hidden_size=D, num_heads=H, head_dim=D // H,
            num_channels=4, bandwidth=8, max_seq_len=1024,
        )

        x = torch.randn(B, T, D)
        with torch.no_grad():
            fmma(x)

        assert fmma._last_query_moments is not None, \
            "First-chunk moments should still be stored for PSD/BW regularization"


# ============================================================================
# End-to-end: teacher moments + student moments + loss
# ============================================================================

class TestEndToEnd:
    """Full pipeline: teacher attn → moments → loss with student moments."""

    def test_full_pipeline(self):
        B, T, D, H = 1, 32, 64, 4
        r, S = 4, 8
        max_seq_len = 1024

        # 1. Create FMMA module
        fmma = FMMAAttention(
            hidden_size=D, num_heads=H, head_dim=D // H,
            num_channels=r, bandwidth=S, max_seq_len=max_seq_len,
        )

        # 2. Prepare query indices
        t_idx = select_query_indices(T, mom_query_stride=4, mom_max_queries=8)

        # 3. Student forward
        x = torch.randn(B, T, D, requires_grad=True)
        output, _ = fmma(x)

        # 4. Simulate teacher attention (random causal)
        mask = torch.tril(torch.ones(T, T))
        raw = torch.rand(B, H, T, T) * mask
        teacher_attn = raw / raw.sum(dim=-1, keepdim=True)

        # 5. Compute teacher moments
        phases = phases_from_positions(torch.arange(T), max_seq_len).unsqueeze(0)
        h_idx = select_heads(H, mom_head_stride=1)  # all heads

        m_teacher = compute_teacher_phase_moments(
            teacher_attn, phases, t_idx, S, h_idx=h_idx, use_fejer_weights=True,
        )

        # 6. Get student moments (compute lazily from stored queries)
        stored_q = fmma._last_queries
        assert stored_q is not None, "Student queries should be stored"
        mu_student = fmma.query_moments(stored_q[:, :, t_idx])  # lazy

        mu_student = mu_student[:, h_idx]

        # Align lengths
        min_q = min(mu_student.shape[2], m_teacher.shape[2])
        mu_student = mu_student[:, :, :min_q]
        m_teacher = m_teacher[:, :, :min_q]

        # 7. Compute loss
        loss = compute_moment_loss(mu_student, m_teacher, S, use_fejer_weights=True)
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should have gradients"

        # 8. Check gradient flows back to input
        loss.backward()
        # Gradients should reach the FMMA parameters
        has_grad = False
        for p in fmma.query_moments.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should flow to QueryMomentNetwork parameters"


class TestGradientCheckpointing:
    """Gradient checkpointing must not cause tensor-count mismatch."""

    def test_checkpoint_backward_no_crash(self):
        """Reproduce the CheckpointError: forward + backward with grad checkpointing."""
        B, T, D, H = 1, 64, 64, 4
        r, S = 4, 8

        fmma = FMMAAttention(
            hidden_size=D, num_heads=H, head_dim=D // H,
            num_channels=r, bandwidth=S, max_seq_len=1024,
        )
        fmma.gradient_checkpointing = True  # Enable checkpointing
        fmma.train()

        x = torch.randn(B, T, D, requires_grad=True)
        output, _ = fmma(x)

        # This backward() call was crashing with CheckpointError
        loss = output.sum()
        loss.backward()  # Should NOT raise CheckpointError

        # Also verify moments were stored correctly
        assert fmma._last_query_moments is not None
        assert fmma._last_queries is not None

    def test_checkpoint_backward_without_moment_distill(self):
        """Grad checkpointing should work fine without moment distillation too."""
        B, T, D, H = 1, 64, 64, 4

        fmma = FMMAAttention(
            hidden_size=D, num_heads=H, head_dim=D // H,
            num_channels=4, bandwidth=8, max_seq_len=1024,
        )
        fmma.gradient_checkpointing = True
        fmma.train()

        x = torch.randn(B, T, D, requires_grad=True)
        output, _ = fmma(x)
        loss = output.sum()
        loss.backward()  # Should NOT raise CheckpointError

    def test_checkpoint_psd_reg_still_works(self):
        """PSD/BW regularization should still work with checkpointing."""
        B, T, D, H = 1, 64, 64, 4

        fmma = FMMAAttention(
            hidden_size=D, num_heads=H, head_dim=D // H,
            num_channels=4, bandwidth=8, max_seq_len=1024,
        )
        fmma.gradient_checkpointing = True
        fmma.train()

        x = torch.randn(B, T, D, requires_grad=True)
        output, _ = fmma(x)

        # _last_query_moments should be set for PSD/BW regularization
        assert fmma._last_query_moments is not None

        # Build a combined loss (output + PSD reg)
        from fmma.regularization import psd_toeplitz_penalty
        psd_loss = psd_toeplitz_penalty(fmma._last_query_moments)
        total_loss = output.sum() + 0.1 * psd_loss
        total_loss.backward()  # Should NOT raise CheckpointError


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
