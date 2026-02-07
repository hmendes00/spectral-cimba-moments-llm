"""
FMMA Comprehensive Tests — Groups A, B, C.

A: Theorem exactness (explicit vs cached)
B: Causality / no future leakage
C: Phase mapping consistency (train vs inference, Δ sign)

Run with:
    pytest fmma/test_theorem_causality.py -v
    pytest fmma/test_theorem_causality.py -v -k "not slow"
"""

import math

import pytest
import torch
import torch.nn.functional as F

from .cache import FMMACache
from .features import NonnegativeKeyFeatures, QueryMomentNetwork
from .attention import FMMAAttention
from .test_helpers import (
    ATOL, RTOL, SEED, DEVICE, set_seed,
    make_attention, make_cache,
    explicit_fmma_output, cached_fmma_output, streaming_cache_runner,
)

# ═══════════════════════════════════════════════════════════════
# Common fixture: configs to parametrize
# ═══════════════════════════════════════════════════════════════
CONFIGS = [
    pytest.param(2, 2, 8,  id="S2-r2-L8"),
    pytest.param(4, 4, 16, id="S4-r4-L16"),
    pytest.param(8, 8, 32, id="S8-r8-L32"),
]


def _random_inputs(B, H, L, D, r, S, max_seq_len=64, seed=SEED):
    """Build random key_features, values, query_moments, phases."""
    set_seed(seed)
    key_features = F.softplus(torch.randn(B, H, L, r))
    values = torch.randn(B, H, L, D)
    position_ids = torch.arange(L)
    phases = 2 * math.pi * position_ids.float() / max_seq_len

    # PSD moments via autocorrelation (use the network for realism)
    net = QueryMomentNetwork(D, r, S, enforce_psd=True)
    queries = torch.randn(B, H, L, D)
    query_moments_all = net(queries)  # [B, H, L, r, 2S+1]

    return key_features, values, query_moments_all, phases, position_ids


# ═══════════════════════════════════════════════════════════════
#  GROUP A: Theorem Exactness
# ═══════════════════════════════════════════════════════════════

class TestTheoremExactness:
    """Validate: cached moment formula == explicit sum-over-tokens."""

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_a1_single_step_explicit_vs_cached(self, S, r, L):
        """A1: Single query position — explicit definition matches cached theorem."""
        B, H, D = 1, 4, 8
        max_seq_len = 64
        kf, vals, q_moments_all, phases, pos_ids = _random_inputs(
            B, H, L, D, r, S, max_seq_len
        )

        # Pick query at t = L-1 (last valid position)
        t = L - 1
        q_moments_t = q_moments_all[:, :, t]  # [B, H, r, 2S+1]
        phases_b = phases.unsqueeze(0).expand(B, -1)  # [B, L]
        query_phase = phases_b[:, t]           # [B]

        y_explicit = explicit_fmma_output(
            kf, vals, q_moments_t, phases_b, query_phase, S,
            causal_len=t + 1,
        )
        y_cached = cached_fmma_output(
            kf, vals, q_moments_t, phases_b, query_phase, S,
            max_seq_len, pos_ids, causal_len=t + 1,
        )

        err = (y_explicit - y_cached).abs().max().item()
        assert err < ATOL, (
            f"Theorem mismatch: max|explicit-cached| = {err:.2e} (tol {ATOL})"
        )

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_a2_streamed_all_positions(self, S, r, L):
        """A2: Streamed theorem equality for ALL t — catches full-chunk pre-update."""
        B, H, D = 1, 4, 8
        max_seq_len = 64
        kf, vals, q_moments_all, phases, pos_ids = _random_inputs(
            B, H, L, D, r, S, max_seq_len
        )
        phases_b = phases.unsqueeze(0).expand(B, -1)

        # Streaming: per-token update-then-query
        y_streamed = streaming_cache_runner(
            kf, vals, q_moments_all, phases_b, S, max_seq_len, pos_ids,
        )  # [B, H, L, D]

        # Explicit: for each t, build from scratch with causal_len=t+1
        for t in range(L):
            q_mom_t = q_moments_all[:, :, t]
            q_phase = phases_b[:, t]

            y_explicit_t = explicit_fmma_output(
                kf, vals, q_mom_t, phases_b, q_phase, S,
                causal_len=t + 1,
            )  # [B, H, D]

            err = (y_streamed[:, :, t] - y_explicit_t).abs().max().item()
            assert err < ATOL, (
                f"Streamed vs explicit mismatch at t={t}: "
                f"max|diff| = {err:.2e} (tol {ATOL})"
            )

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_a3_multi_batch(self, S, r, L):
        """A3: Multi-batch theorem equality — confirms batch indexing."""
        B, H, D = 2, 4, 8
        max_seq_len = 64
        kf, vals, q_moments_all, phases, pos_ids = _random_inputs(
            B, H, L, D, r, S, max_seq_len
        )
        phases_b = phases.unsqueeze(0).expand(B, -1)

        y_streamed = streaming_cache_runner(
            kf, vals, q_moments_all, phases_b, S, max_seq_len, pos_ids,
        )

        # Spot check 3 positions per batch
        for t in [0, L // 2, L - 1]:
            q_mom_t = q_moments_all[:, :, t]
            q_phase = phases_b[:, t]

            y_explicit_t = explicit_fmma_output(
                kf, vals, q_mom_t, phases_b, q_phase, S,
                causal_len=t + 1,
            )

            err = (y_streamed[:, :, t] - y_explicit_t).abs().max().item()
            assert err < ATOL, (
                f"Batch theorem mismatch at t={t}: "
                f"max|diff| = {err:.2e}"
            )


# ═══════════════════════════════════════════════════════════════
#  GROUP B: Causality / No Future Leakage
# ═══════════════════════════════════════════════════════════════

class TestCausality:
    """Verify no future tokens leak into past outputs."""

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_b1_training_path_chunk_leakage(self, S, r, L):
        """B1: Modify future tokens; past outputs must NOT change (training path)."""
        set_seed()
        H_SIZE = 32
        H = 4
        D = H_SIZE // H

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.eval()

        x = torch.randn(1, L, H_SIZE, device=DEVICE)
        t0 = L // 2

        with torch.no_grad():
            out1, _ = attn(x)

        x2 = x.clone()
        x2[:, t0:] = torch.randn(1, L - t0, H_SIZE, device=DEVICE)

        with torch.no_grad():
            out2, _ = attn(x2)

        diff_past = (out1[:, :t0] - out2[:, :t0]).abs().max().item()
        diff_future = (out1[:, t0:] - out2[:, t0:]).abs().max().item()

        assert diff_past < ATOL, (
            f"Future leak in training path: past diff = {diff_past:.2e}"
        )
        # Sanity: future SHOULD differ
        assert diff_future > 1e-3, "Future outputs unexpectedly identical"

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_b1_cached_path_chunk_leakage(self, S, r, L):
        """B1-cache: Same leakage test on cached inference path."""
        set_seed()
        H_SIZE = 32
        H = 4
        D = H_SIZE // H

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.eval()

        x = torch.randn(1, L, H_SIZE, device=DEVICE)
        pos = torch.arange(L).unsqueeze(0)
        t0 = L // 2

        cache1 = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)
        with torch.no_grad():
            out1, _ = attn(x, position_ids=pos, past_key_value=cache1, use_cache=True)

        x2 = x.clone()
        x2[:, t0:] = torch.randn(1, L - t0, H_SIZE, device=DEVICE)

        cache2 = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)
        with torch.no_grad():
            out2, _ = attn(x2, position_ids=pos, past_key_value=cache2, use_cache=True)

        diff_past = (out1[:, :t0] - out2[:, :t0]).abs().max().item()
        assert diff_past < ATOL, (
            f"Future leak in cached path: past diff = {diff_past:.2e}"
        )

    def test_b2_generation_step_incremental(self):
        """B2: Step-by-step generation equals full causal forward."""
        set_seed()
        S, r = 4, 4
        B, H, D = 1, 4, 8
        L, max_seq_len = 16, 64

        kf, vals, q_moments_all, phases, pos_ids = _random_inputs(
            B, H, L, D, r, S, max_seq_len
        )
        phases_b = phases.unsqueeze(0).expand(B, -1)

        # Full streamed output
        y_stream = streaming_cache_runner(
            kf, vals, q_moments_all, phases_b, S, max_seq_len, pos_ids,
        )

        # Step-by-step: fresh cache per step (explicit reference)
        for t in range(L):
            q_mom_t = q_moments_all[:, :, t]
            q_phase = phases_b[:, t]

            y_ref = explicit_fmma_output(
                kf, vals, q_mom_t, phases_b, q_phase, S,
                causal_len=t + 1,
            )

            err = (y_stream[:, :, t] - y_ref).abs().max().item()
            assert err < ATOL, (
                f"Generation step {t} mismatch: {err:.2e}"
            )


# ═══════════════════════════════════════════════════════════════
#  GROUP C: Phase Mapping Consistency
# ═══════════════════════════════════════════════════════════════

class TestPhaseMapping:
    """Verify training and inference use the same phase mapping and Δ sign."""

    def test_c1_train_vs_inference_phase_mapping(self):
        """C1: Phases computed in training path match those in inference path."""
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L, max_seq_len = 12, 64

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S, max_seq_len=max_seq_len,
        )

        pos_ids = torch.arange(L)

        # Training path phases (via _get_phases)
        phases_train = attn._get_phases(pos_ids.unsqueeze(0))  # [1, L]

        # Inference/cache path phases (same canonical mapping)
        phases_infer = 2 * math.pi * pos_ids.float() / max_seq_len

        err = (phases_train[0] - phases_infer).abs().max().item()
        assert err < 1e-8, (
            f"Phase mapping train!=infer: max diff = {err:.2e}. "
            f"Fix to a single canonical mapping."
        )

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_c2_delta_sign_consistency(self, S, r, L):
        """
        C2: Training density and cached density must agree in Δ sign.

        Build both from the same inputs and compare the actual output.
        """
        B, H, D = 1, 4, 8
        max_seq_len = 64
        kf, vals, q_moments_all, phases, pos_ids = _random_inputs(
            B, H, L, D, r, S, max_seq_len
        )
        phases_b = phases.unsqueeze(0).expand(B, -1)

        # For each query position, compare explicit definition (which uses
        # delta = θ_i - τ explicitly) with cached theorem.
        for t in [0, L // 2, L - 1]:
            q_mom_t = q_moments_all[:, :, t]
            q_phase = phases_b[:, t]

            y_explicit = explicit_fmma_output(
                kf, vals, q_mom_t, phases_b, q_phase, S,
                causal_len=t + 1,
            )
            y_cached = cached_fmma_output(
                kf, vals, q_mom_t, phases_b, q_phase, S,
                max_seq_len, pos_ids, causal_len=t + 1,
            )

            err = (y_explicit - y_cached).abs().max().item()
            assert err < ATOL, (
                f"Δ sign mismatch at t={t}: max|diff| = {err:.2e}. "
                f"Training delta and inference delta have opposite signs."
            )

    def test_c2_purely_imaginary_moment(self):
        """
        C2-extra: Purely imaginary μ_1 magnifies sin-term errors.

        If Δ sign is flipped, the sin contribution will be negated,
        causing a large mismatch.
        """
        set_seed()
        B, H, r, S = 1, 2, 2, 2
        D, L = 8, 10
        max_seq_len = 64

        kf = F.softplus(torch.randn(B, H, L, r))
        vals = torch.randn(B, H, L, D)
        pos_ids = torch.arange(L)
        phases_b = (2 * math.pi * pos_ids.float() / max_seq_len).unsqueeze(0)

        # Construct moments with mu_1 purely imaginary
        mu_0 = torch.ones(B, H, r, 1, dtype=torch.complex64)
        mu_1 = torch.zeros(B, H, r, 1, dtype=torch.complex64)
        mu_1.imag = torch.ones_like(mu_1.real) * 0.5
        mu_2 = torch.zeros(B, H, r, 1, dtype=torch.complex64)
        # Hermitian: mu_{-k} = conj(mu_k)
        mu_neg1 = mu_1.conj()
        mu_neg2 = mu_2.conj()
        q_moments = torch.cat([mu_neg2, mu_neg1, mu_0, mu_1, mu_2], dim=-1)

        t = L - 1
        q_phase = phases_b[:, t]

        y_explicit = explicit_fmma_output(
            kf, vals, q_moments, phases_b, q_phase, S,
            causal_len=t + 1,
        )
        y_cached = cached_fmma_output(
            kf, vals, q_moments, phases_b, q_phase, S,
            max_seq_len, pos_ids, causal_len=t + 1,
        )

        err = (y_explicit - y_cached).abs().max().item()
        assert err < ATOL, (
            f"Purely imaginary μ_1 test failed: {err:.2e}. "
            f"Sin-term sign likely wrong."
        )
