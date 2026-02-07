"""
FMMA Comprehensive Tests — Groups D, E, F, G, H.

D: Nonnegativity, normalization, PSD / Hermitian
E: Masking (padding + causal)
F: HF cache integration
G: Numerical stability
H: Gradient flow

Run with:
    pytest fmma/test_properties.py -v
    pytest fmma/test_properties.py -v -k "not slow"
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

CONFIGS = [
    pytest.param(2, 2, 8,  id="S2-r2-L8"),
    pytest.param(4, 4, 16, id="S4-r4-L16"),
    pytest.param(8, 8, 32, id="S8-r8-L32"),
]


# ═══════════════════════════════════════════════════════════════
#  GROUP D: Nonnegativity, Normalization, PSD Constraints
# ═══════════════════════════════════════════════════════════════

class TestNonnegativity:
    """Verify PSD densities, nonneg scores, normalization, Hermitian output."""

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_d1_density_nonnegativity(self, S, r, L):
        """
        D1: PSD autocorrelation → Fejér density ≥ 0 on fine grid.

        Evaluate p_m(Δ) on 256 Δ values; all must be ≥ -1e-6.
        """
        set_seed()
        B, H, D = 1, 4, 8

        net = QueryMomentNetwork(D, r, S, enforce_psd=True)
        queries = torch.randn(B, H, 1, D)
        moments = net(queries)  # [B, H, 1, r, 2S+1]
        mu = moments[:, :, 0]  # [B, H, r, 2S+1]

        k_indices = torch.arange(-S, S + 1, dtype=torch.float32)
        c_k = 1.0 - torch.abs(k_indices) / (S + 1)

        # Fine grid of delta values
        delta_grid = torch.linspace(0, 2 * math.pi, 256)
        exp_ik_delta = torch.exp(
            1j * k_indices.unsqueeze(0) * delta_grid.unsqueeze(1)
        )  # [256, 2S+1]

        # p_m(Δ) = Σ_k c_k μ_k e^{ikΔ}   →  [B, H, r, 256]
        p_m = torch.einsum(
            "bhrk,dk,k->bhrd",
            mu,
            exp_ik_delta.to(torch.complex64),
            c_k.to(torch.complex64),
        )

        min_p = p_m.real.min().item()
        assert min_p >= -1e-6, (
            f"Density went negative: min(p_m) = {min_p:.2e}. "
            f"PSD enforcement may be broken."
        )

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_d2_score_nonnegativity_and_normalization(self, S, r, L):
        """
        D2: Attention scores ≥ 0 and weights sum to 1.

        Uses the full module forward (training path) and inspects
        scores/weights via a hook.
        """
        set_seed()
        H_SIZE, H = 32, 4
        D = H_SIZE // H

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.eval()

        x = torch.randn(1, L, H_SIZE, device=DEVICE)

        # We can't directly hook into attn_weights in training path
        # without modifying the module, so we replicate the score
        # computation from the internals.
        with torch.no_grad():
            queries = attn.q_proj(x).view(1, L, H, D).transpose(1, 2)
            keys = attn.k_proj(x).view(1, L, H, D).transpose(1, 2)
            values = attn.v_proj(x).view(1, L, H, D).transpose(1, 2)

            key_feats = attn.key_features(keys)  # [B, H, L, r]
            q_moments_all = attn.query_moments(queries)  # [B, H, L, r, 2S+1]

            pos_ids = torch.arange(L)
            phases = attn._get_phases(pos_ids.unsqueeze(0))  # [1, L]

            c_k = attn.fejer_weights
            k_vals = torch.arange(1, S + 1, dtype=torch.float32)

            # Compute scores for each query position
            for t in range(L):
                q_mom = q_moments_all[:, :, t]  # [B, H, r, 2S+1]
                weighted = q_mom * c_k.view(1, 1, 1, -1)

                mu_0_real = weighted[..., S].real
                mu_pos_real = weighted[..., S + 1:].real
                mu_pos_imag = weighted[..., S + 1:].imag

                phases_q = phases[:, t:t+1]
                delta = phases - phases_q  # [B, L]

                k_delta = delta.unsqueeze(-1) * k_vals  # [B, L, S]
                cos_b = torch.cos(k_delta)
                sin_b = torch.sin(k_delta)

                cos_c = torch.einsum("bhrs,bks->bhkr", mu_pos_real, cos_b)
                sin_c = torch.einsum("bhrs,bks->bhkr", mu_pos_imag, sin_b)

                densities = mu_0_real.unsqueeze(2) + 2.0 * (cos_c - sin_c)
                # [B, H, L, r]

                scores = (key_feats * densities).sum(dim=-1)  # [B, H, L]

                # Causal mask
                scores[:, :, t + 1:] = 0.0

                # Check nonnegativity
                min_score = scores[:, :, :t + 1].min().item()
                assert min_score >= -1e-6, (
                    f"Negative score at t={t}: min={min_score:.2e}"
                )

                # Check normalization
                denom = scores.sum(dim=-1)  # [B, H]
                weights = scores / (denom.unsqueeze(-1) + 1e-8)
                weight_sum = weights.sum(dim=-1)  # [B, H]
                sum_err = (weight_sum - 1.0).abs().max().item()
                assert sum_err < 1e-5, (
                    f"Weight normalization error at t={t}: "
                    f"|sum-1| = {sum_err:.2e}"
                )

    @pytest.mark.parametrize("S, r, L", CONFIGS)
    def test_d3_hermitian_symmetry_real_output(self, S, r, L):
        """
        D3: Cache output imaginary part must be negligible.

        Hermitian moments → real output. Large imag = broken symmetry.
        """
        set_seed()
        B, H, D = 1, 4, 8
        max_seq_len = 64

        net = QueryMomentNetwork(D, r, S, enforce_psd=True)
        queries = torch.randn(B, H, L, D)
        q_moments_all = net(queries)

        kf = F.softplus(torch.randn(B, H, L, r))
        vals = torch.randn(B, H, L, D)
        pos_ids = torch.arange(L)

        cache = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S,
                           max_seq_len=max_seq_len)

        cache.update_moments(0, kf, vals, pos_ids)

        for t in [0, L // 2, L - 1]:
            q_mom_t = q_moments_all[:, :, t]
            phase_t = (2 * math.pi * pos_ids[t].float() / max_seq_len)
            phase_t = phase_t.unsqueeze(0).unsqueeze(0).expand(B, H)

            # Compute raw (before .real) by calling cache internals
            device = q_mom_t.device
            M_V = cache._M_V[0]
            M_1 = cache._M_1[0]
            c_k = cache.fejer_weights.to(device)
            k_idx = torch.arange(-S, S + 1, dtype=torch.float32, device=device)

            exp_neg = torch.exp(-1j * k_idx.view(1, 1, -1) * phase_t.unsqueeze(-1))
            combined = c_k.view(1, 1, 1, -1) * q_mom_t * exp_neg.unsqueeze(2)

            num = torch.einsum("bhrk,bhrkd->bhd", combined, M_V)
            den = torch.einsum("bhrk,bhrk->bh", combined, M_1)
            raw = num / (den.unsqueeze(-1) + 1e-8)

            imag_max = raw.imag.abs().max().item()
            assert imag_max < 1e-4, (
                f"Imaginary leakage at t={t}: max|imag| = {imag_max:.2e}. "
                f"Hermitian symmetry violated."
            )


# ═══════════════════════════════════════════════════════════════
#  GROUP E: Masking (Padding + Causal)
# ═══════════════════════════════════════════════════════════════

class TestMasking:
    """Verify padding mask exclusion and causal mask correctness."""

    def test_e1_padding_exclusion(self):
        """
        E1: Padded tokens must not influence valid-token outputs.

        Compare: (1) full batch with mask vs (2) truncated batch (no padding).
        """
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L = 8
        valid_len = 5

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.eval()

        x = torch.randn(1, L, H_SIZE)
        pos = torch.arange(L).unsqueeze(0)

        mask = torch.ones(1, L, dtype=torch.long)
        mask[:, valid_len:] = 0

        with torch.no_grad():
            out_masked, _ = attn(x, attention_mask=mask, position_ids=pos)
            out_trunc, _ = attn(x[:, :valid_len], position_ids=pos[:, :valid_len])

        diff = (out_masked[:, :valid_len] - out_trunc).abs().max().item()
        assert diff < 1e-4, (
            f"Padding mask not working: valid-token diff = {diff:.2e}"
        )

    def test_e1_padding_exclusion_cached(self):
        """E1-cache: Same padding test on cached inference path."""
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L = 8
        valid_len = 5

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.eval()

        x = torch.randn(1, L, H_SIZE)
        pos = torch.arange(L).unsqueeze(0)

        mask = torch.ones(1, L, dtype=torch.long)
        mask[:, valid_len:] = 0

        cache1 = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)
        cache2 = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)

        with torch.no_grad():
            out_masked, _ = attn(
                x, attention_mask=mask, position_ids=pos,
                past_key_value=cache1, use_cache=True,
            )
            out_trunc, _ = attn(
                x[:, :valid_len], position_ids=pos[:, :valid_len],
                past_key_value=cache2, use_cache=True,
            )

        diff = (out_masked[:, :valid_len] - out_trunc).abs().max().item()
        assert diff < 1e-4, (
            f"Cached padding mask not working: diff = {diff:.2e}"
        )

    def test_e2_causal_mask_training_path(self):
        """
        E2: In training forward, token t must NOT attend to any i > t.

        We verify by checking that modifying future tokens doesn't change
        the output at t (stronger than just checking zero weights).
        """
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L = 16

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.eval()

        x = torch.randn(1, L, H_SIZE)

        # Check at multiple split points
        for t in [0, 4, 8, L - 2]:
            x2 = x.clone()
            x2[:, t + 1:] = torch.randn(1, L - t - 1, H_SIZE)

            with torch.no_grad():
                o1, _ = attn(x)
                o2, _ = attn(x2)

            diff = (o1[:, :t + 1] - o2[:, :t + 1]).abs().max().item()
            assert diff < ATOL, (
                f"Causal violation at t={t}: past diff = {diff:.2e}"
            )


# ═══════════════════════════════════════════════════════════════
#  GROUP F: HF Cache Integration
# ═══════════════════════════════════════════════════════════════

class TestHFCache:
    """Verify DynamicCache contract: seq length tracking, no KV growth."""

    def test_f1_seq_length_tracking(self):
        """F1: get_seq_length() increments correctly per token."""
        S, r = 4, 4
        H, D = 4, 8

        cache = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)

        assert cache.get_seq_length(0) == 0

        for step in range(1, 21):
            kf = F.softplus(torch.randn(1, H, 1, r))
            v = torch.randn(1, H, 1, D)
            pos = torch.tensor([step - 1])

            cache.update_moments(0, kf, v, pos)
            assert cache.get_seq_length(0) == step, (
                f"seq_length mismatch at step {step}: "
                f"got {cache.get_seq_length(0)}"
            )

    def test_f2_no_kv_growth(self):
        """F2: update() returns zero-length tensors; memory doesn't grow."""
        S, r = 4, 4
        H, D = 4, 8

        cache = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)

        key_states = torch.randn(1, H, 5, D)
        val_states = torch.randn(1, H, 5, D)

        k_out, v_out = cache.update(key_states, val_states, layer_idx=0)
        assert k_out.shape[2] == 0, f"K seq dim not 0: {k_out.shape}"
        assert v_out.shape[2] == 0, f"V seq dim not 0: {v_out.shape}"

        # Feed 100 tokens incrementally and check memory is constant
        kf = F.softplus(torch.randn(1, H, 1, r))
        v = torch.randn(1, H, 1, D)

        cache.update_moments(0, kf, v, torch.tensor([0]))
        mem_after_1 = cache.memory_usage(0)["total_bytes"]

        for i in range(1, 100):
            cache.update_moments(0, kf, v, torch.tensor([i]))

        mem_after_100 = cache.memory_usage(0)["total_bytes"]
        assert mem_after_1 == mem_after_100, (
            f"Memory grew: {mem_after_1} → {mem_after_100}"
        )


# ═══════════════════════════════════════════════════════════════
#  GROUP G: Numerical Stability
# ═══════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Verify no NaN/Inf/explosion under stress conditions."""

    @pytest.mark.slow
    def test_g1_long_stream_50k(self):
        """
        G1: Stream 50k tokens — no NaN/Inf, bounded output.

        Marked slow; skip with: pytest -m "not slow"
        """
        set_seed()
        S, r = 4, 4
        H, D = 2, 8
        N = 50_000
        max_seq_len = 100_000
        chunk = 500

        cache = make_cache(
            num_heads=H, head_dim=D, num_channels=r, bandwidth=S,
            max_seq_len=max_seq_len,
        )

        net = QueryMomentNetwork(D, r, S, enforce_psd=True)

        den_mins, den_maxs = [], []
        out_maxs = []

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            clen = end - start

            kf = F.softplus(torch.randn(1, H, clen, r))
            vals = torch.randn(1, H, clen, D)
            pos = torch.arange(start, end)

            cache.update_moments(0, kf, vals, pos)

            # Query from last token in chunk
            q = torch.randn(1, H, D)
            q_mom = net(q)  # [1, H, r, 2S+1]
            phase = (2 * math.pi * (end - 1) / max_seq_len)
            phase_t = torch.full((1, H), phase)

            out = cache.compute_attention(0, q_mom, phase_t)

            # Track stats
            assert torch.isfinite(out).all(), (
                f"NaN/Inf in output at tokens {start}-{end}"
            )
            out_maxs.append(out.abs().max().item())

        assert cache.get_seq_length(0) == N
        max_output = max(out_maxs)
        assert max_output < 1e3, (
            f"Output magnitude exploding: max = {max_output:.2e}"
        )

    def test_g2_extreme_inputs(self):
        """G2: Very large key/query magnitudes → no NaN/Inf."""
        set_seed()
        S, r = 4, 4
        H, D = 4, 8
        L = 16

        for scale in [100.0, 1000.0]:
            kf = F.softplus(torch.randn(1, H, L, r) * scale)
            vals = torch.randn(1, H, L, D) * scale
            pos = torch.arange(L)

            cache = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)
            cache.update_moments(0, kf, vals, pos)

            net = QueryMomentNetwork(D, r, S, enforce_psd=True)
            q = torch.randn(1, H, D) * scale
            q_mom = net(q)
            phase_t = torch.zeros(1, H)

            out = cache.compute_attention(0, q_mom, phase_t)

            assert torch.isfinite(out).all(), (
                f"NaN/Inf with scale={scale}: "
                f"max={out.abs().max().item():.2e}"
            )

    def test_g2_zero_inputs(self):
        """G2-zero: All-zero inputs must not cause NaN (division guard)."""
        S, r = 4, 4
        H, D = 4, 8
        L = 8

        kf = torch.zeros(1, H, L, r)  # all zero → zero denominator
        vals = torch.zeros(1, H, L, D)
        pos = torch.arange(L)

        cache = make_cache(num_heads=H, head_dim=D, num_channels=r, bandwidth=S)
        cache.update_moments(0, kf, vals, pos)

        net = QueryMomentNetwork(D, r, S, enforce_psd=True)
        q = torch.zeros(1, H, D)
        q_mom = net(q)
        phase_t = torch.zeros(1, H)

        out = cache.compute_attention(0, q_mom, phase_t)
        assert torch.isfinite(out).all(), "NaN/Inf on all-zero inputs"


# ═══════════════════════════════════════════════════════════════
#  GROUP H: Gradient Flow
# ═══════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Verify gradients reach QueryMomentNetwork and key features."""

    def test_h1_query_moment_network_grads(self):
        """H1: QueryMomentNetwork parameters get nonzero gradients."""
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L = 8

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.train()

        x = torch.randn(1, L, H_SIZE, requires_grad=False)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        # Check query_moments network
        has_grad = False
        for name, p in attn.query_moments.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if grad_norm > 0:
                    has_grad = True

        assert has_grad, (
            "No gradients reached QueryMomentNetwork. "
            "Check autograd graph."
        )

    def test_h2_feature_projection_grads(self):
        """H2: NonnegativeKeyFeatures parameters get nonzero gradients."""
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L = 8

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.train()

        x = torch.randn(1, L, H_SIZE, requires_grad=False)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        # Check key_features network
        has_grad = False
        for name, p in attn.key_features.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if grad_norm > 0:
                    has_grad = True

        assert has_grad, (
            "No gradients reached NonnegativeKeyFeatures. "
            "Check autograd graph."
        )

    def test_h_projections_grads(self):
        """H-extra: Q/K/V/O projections all get gradients."""
        set_seed()
        S, r = 4, 4
        H_SIZE, H, D = 32, 4, 8
        L = 8

        attn = make_attention(
            hidden_size=H_SIZE, num_heads=H, head_dim=D,
            num_channels=r, bandwidth=S,
        )
        attn.train()

        x = torch.randn(1, L, H_SIZE, requires_grad=False)
        out, _ = attn(x)
        loss = out.sum()
        loss.backward()

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(attn, proj_name)
            grad_norm = proj.weight.grad.norm().item() if proj.weight.grad is not None else 0.0
            assert grad_norm > 0, (
                f"No gradient for {proj_name}.weight"
            )
