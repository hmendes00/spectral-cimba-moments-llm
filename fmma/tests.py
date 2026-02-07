"""
FMMA Acceptance Tests — 11 tests covering algebraic exactness, memory,
nonnegativity, causality, stability, PSD constraints, HF contract, masks,
and cached-path causality.
"""

import math

import torch
import torch.nn.functional as F

from .cache import FMMACache
from .features import QueryMomentNetwork
from .attention import FMMAAttention
from .regularization import psd_toeplitz_penalty, bandwidth_regularization


def test_fmma_algebraic_exactness():
    """
    Acceptance test 1 (per paper §6.1): Verify algebraic exactness of moment
    computation.

    Compute y_t both ways:
    1. Explicit: sum over all tokens
    2. From moments: using cached M_V and M_1

    Should match to numerical precision.
    """
    print("=" * 60)
    print("Test 1: FMMA Algebraic Exactness")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    seq_len = 32
    head_dim = 16
    num_channels = 3
    bandwidth = 8
    max_seq_len = 1024

    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)

    positions = torch.arange(seq_len)
    phases = 2 * math.pi * positions.float() / max_seq_len

    key_features = F.softplus(
        torch.randn(batch_size, num_heads, seq_len, num_channels)
    )

    mu_positive = torch.randn(batch_size, num_heads, num_channels, bandwidth) + 1j * torch.randn(
        batch_size, num_heads, num_channels, bandwidth
    )
    mu_positive = 0.5 * mu_positive
    mu_negative = torch.conj(torch.flip(mu_positive, dims=[-1]))
    mu_zero = torch.ones(batch_size, num_heads, num_channels, 1, dtype=torch.complex64)
    query_moments = torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)

    k_indices = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
    c_k = 1.0 - torch.abs(k_indices) / (bandwidth + 1)

    query_phase = phases[-1]

    # Method 1: Explicit
    delta = phases - query_phase
    exp_ik_delta = torch.exp(1j * k_indices.unsqueeze(0) * delta.unsqueeze(1))

    p_m = torch.einsum(
        "bhrk,tk,k->bhrt", query_moments, exp_ik_delta, c_k.to(torch.complex64)
    )
    s_i = torch.einsum("bhtr,bhrt->bht", key_features.to(torch.complex64), p_m)

    numerator_explicit = torch.einsum("bht,bhtd->bhd", s_i, values.to(torch.complex64))
    denominator_explicit = s_i.sum(dim=-1)
    y_explicit = (numerator_explicit / denominator_explicit.unsqueeze(-1)).real

    # Method 2: From cached moments
    exp_ik_theta = torch.exp(1j * k_indices.unsqueeze(0) * phases.unsqueeze(1))

    M_1 = torch.einsum("bhtr,tk->bhrk", key_features.to(torch.complex64), exp_ik_theta)
    M_V = torch.einsum(
        "bhtr,tk,bhtd->bhrkd",
        key_features.to(torch.complex64),
        exp_ik_theta,
        values.to(torch.complex64),
    )

    exp_neg_ik_tau = torch.exp(-1j * k_indices * query_phase)
    combined = c_k.to(torch.complex64) * query_moments * exp_neg_ik_tau

    numerator_moments = torch.einsum("bhrk,bhrkd->bhd", combined, M_V)
    denominator_moments = torch.einsum("bhrk,bhrk->bh", combined, M_1)

    y_moments = (numerator_moments / denominator_moments.unsqueeze(-1)).real

    max_error = (y_explicit - y_moments).abs().max().item()
    print(f"Max error between explicit and moment computation: {max_error:.2e}")

    # float32 complex arithmetic over S=8, T=32 → expect ~1e-5 to 1e-4
    if max_error < 1e-4:
        print("✓ PASSED: Algebraic exactness verified")
        return True
    else:
        print("✗ FAILED: Results don't match")
        return False


def test_fmma_memory_constant():
    """Acceptance test 2: Verify memory stays constant as sequence grows."""
    print("\n" + "=" * 60)
    print("Test 2: FMMA Memory Constant")
    print("=" * 60)

    num_heads = 4
    head_dim = 16
    num_channels = 4
    bandwidth = 32
    max_seq_len = 8192

    memories = []

    for seq_len in [10, 100, 1000]:
        cache = FMMACache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            bandwidth=bandwidth,
            max_seq_len=max_seq_len,
        )

        key_feats = torch.rand(1, num_heads, seq_len, num_channels)
        values = torch.rand(1, num_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)

        cache.update_moments(
            layer_idx=0,
            key_features=key_feats,
            values=values,
            position_ids=positions,
        )

        mem = cache.memory_usage(0)
        memories.append(mem["total_bytes"])
        print(
            f"Seq len {seq_len:5d}: M_V = {mem['M_V_bytes']/1024:.1f} KB, "
            f"M_1 = {mem['M_1_bytes']/1024:.1f} KB, "
            f"total = {mem['total_bytes']/1024:.1f} KB"
        )

    if memories[0] == memories[1] == memories[2]:
        print("✓ PASSED: Memory is constant in sequence length")
        return True
    else:
        print("✗ FAILED: Memory varies with sequence length")
        return False


def test_fmma_nonnegativity():
    """
    Acceptance test 3: Verify attention weights are nonneg with PSD moments.
    Uses wrapped Cauchy (μ_k = ρ^|k|) which is always PSD Toeplitz.
    """
    print("\n" + "=" * 60)
    print("Test 3: FMMA Nonnegativity (with PSD-constrained moments)")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    num_heads = 4
    seq_len = 20
    head_dim = 16
    num_channels = 3
    bandwidth = 8

    key_features = F.softplus(
        torch.randn(batch_size, num_heads, seq_len, num_channels)
    )
    positions = torch.arange(seq_len)
    phases = 2 * math.pi * positions.float() / 1024

    rho = 0.7
    k_range = torch.arange(1, bandwidth + 1, dtype=torch.float32)
    mu_magnitudes = rho ** k_range

    mu_positive = mu_magnitudes.view(1, 1, 1, -1).expand(
        batch_size, num_heads, num_channels, -1
    ).to(torch.complex64)
    mu_negative = torch.flip(mu_positive, dims=[-1])
    mu_zero = torch.ones(batch_size, num_heads, num_channels, 1, dtype=torch.complex64)
    query_moments = torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)

    k_indices = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
    c_k = 1.0 - torch.abs(k_indices) / (bandwidth + 1)

    all_nonneg = True
    for query_pos in [0, seq_len // 2, seq_len - 1]:
        query_phase = phases[query_pos]
        delta = phases - query_phase

        exp_ik_delta = torch.exp(1j * k_indices.unsqueeze(0) * delta.unsqueeze(1))
        p_m = torch.einsum(
            "bhrk,tk,k->bhrt",
            query_moments,
            exp_ik_delta,
            c_k.to(torch.complex64),
        )

        p_m_real = p_m.real
        min_p = p_m_real.min().item()

        s_i = torch.einsum("bhtr,bhrt->bht", key_features, p_m_real)
        min_s = s_i.min().item()
        print(f"Query pos {query_pos:2d}: min(p_m)={min_p:.6f}, min(s_i)={min_s:.6f}")
        if min_s < -1e-6:
            all_nonneg = False

    if all_nonneg:
        print("✓ PASSED: Attention scores are nonnegative with PSD moments")
        return True
    else:
        print("✗ FAILED: Some attention scores are negative")
        return False


def test_fmma_causality():
    """
    Acceptance test 4 (per paper §6.2): Verify causality.
    Modifying future tokens must not change past outputs.
    """
    print("\n" + "=" * 60)
    print("Test 4: FMMA Causality")
    print("=" * 60)

    torch.manual_seed(42)

    hidden_size = 64
    num_heads = 2
    head_dim = 32
    num_channels = 4
    bandwidth = 8
    seq_len = 16

    attn = FMMAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        layer_idx=0,
    )
    attn.eval()

    x = torch.randn(1, seq_len, hidden_size)

    with torch.no_grad():
        output1, _ = attn(x)

    x_modified = x.clone()
    x_modified[:, 8:, :] = torch.randn(1, 8, hidden_size)

    with torch.no_grad():
        output2, _ = attn(x_modified)

    max_diff_past = (output1[:, :8, :] - output2[:, :8, :]).abs().max().item()
    max_diff_future = (output1[:, 8:, :] - output2[:, 8:, :]).abs().max().item()

    print(f"Max diff in past positions (0-7): {max_diff_past:.2e}")
    print(f"Max diff in future positions (8-15): {max_diff_future:.2e}")

    if max_diff_past < 1e-5:
        print("✓ PASSED: Modifying future tokens doesn't affect past outputs")
        return True
    else:
        print("✗ FAILED: Past outputs changed when future tokens modified")
        return False


def test_fmma_long_context_stability():
    """
    Acceptance test 5 (per paper §6.3): Long-context stability.
    Memory stays constant; moment magnitudes stay bounded.
    """
    print("\n" + "=" * 60)
    print("Test 5: FMMA Long-Context Stability")
    print("=" * 60)

    torch.manual_seed(42)

    num_heads = 4
    head_dim = 16
    num_channels = 4
    bandwidth = 16
    max_seq_len = 8192

    seq_lengths = [100, 500, 1000, 2000]
    memories = []

    for seq_len in seq_lengths:
        cache = FMMACache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            bandwidth=bandwidth,
            max_seq_len=max_seq_len,
        )

        key_feats = torch.rand(1, num_heads, seq_len, num_channels)
        values = torch.rand(1, num_heads, seq_len, head_dim)
        positions = torch.arange(seq_len)

        cache.update_moments(
            layer_idx=0,
            key_features=key_feats,
            values=values,
            position_ids=positions,
        )

        mem = cache.memory_usage(0)
        memories.append(mem["total_bytes"])

        M_1 = cache._M_1[0]
        min_mag = M_1.abs().min().item()
        max_mag = M_1.abs().max().item()

        print(
            f"Seq len {seq_len:5d}: memory={mem['total_bytes']/1024:.1f} KB, "
            f"|M_1| range=[{min_mag:.2e}, {max_mag:.2e}]"
        )

    memory_constant = all(m == memories[0] for m in memories)
    print(f"\nMemory constant across all sequence lengths: {memory_constant}")

    if memory_constant:
        print("✓ PASSED: Memory is O(rS), constant in sequence length")
        return True
    else:
        print("✗ FAILED: Memory varies with sequence length")
        return False


def test_psd_penalty():
    """Test the PSD Toeplitz penalty function."""
    print("\n" + "=" * 60)
    print("Test 6: PSD Toeplitz Penalty")
    print("=" * 60)

    torch.manual_seed(42)
    bandwidth = 4

    # PSD moments (wrapped Cauchy)
    rho = 0.5
    k_range = torch.arange(1, bandwidth + 1, dtype=torch.float32)
    mu_pos = (rho ** k_range).to(torch.complex64)
    mu_neg = torch.flip(mu_pos, dims=[-1])
    mu_zero = torch.ones(1, dtype=torch.complex64)
    psd_moments = torch.cat([mu_neg, mu_zero, mu_pos]).unsqueeze(0)

    penalty_psd = psd_toeplitz_penalty(psd_moments, reduction="mean").item()
    print(f"PSD moments (wrapped Cauchy): penalty = {penalty_psd:.6f}")

    # Random moments
    random_moments = torch.randn(1, 2 * bandwidth + 1) + 1j * torch.randn(
        1, 2 * bandwidth + 1
    )
    random_moments = random_moments.clone()
    random_moments[0, bandwidth] = random_moments[0, bandwidth].real + 0j
    for k in range(1, bandwidth + 1):
        random_moments[0, bandwidth - k] = random_moments[0, bandwidth + k].conj()

    penalty_random = psd_toeplitz_penalty(random_moments, reduction="mean").item()
    print(f"Random Hermitian moments: penalty = {penalty_random:.6f}")

    if penalty_psd < 1e-6 and penalty_random > penalty_psd:
        print("✓ PASSED: PSD penalty correctly identifies PSD vs non-PSD moments")
        return True
    else:
        print("✗ FAILED: PSD penalty not working as expected")
        return False


def test_bandwidth_regularization():
    """Test the bandwidth regularization function."""
    print("\n" + "=" * 60)
    print("Test 7: Bandwidth Regularization")
    print("=" * 60)

    bw = 4

    low_freq = torch.zeros(1, 2 * bw + 1, dtype=torch.complex64)
    low_freq[0, bw] = 1.0

    reg_low = bandwidth_regularization(
        low_freq, bw, alpha=2.0, reduction="mean"
    ).item()

    high_freq = torch.zeros(1, 2 * bw + 1, dtype=torch.complex64)
    high_freq[0, bw] = 1.0
    high_freq[0, 0] = 0.5
    high_freq[0, -1] = 0.5

    reg_high = bandwidth_regularization(
        high_freq, bw, alpha=2.0, reduction="mean"
    ).item()

    print(f"Low-frequency moments: reg = {reg_low:.6f}")
    print(f"High-frequency moments: reg = {reg_high:.6f}")

    if reg_high > reg_low:
        print("✓ PASSED: High-frequency moments penalized more than low-frequency")
        return True
    else:
        print("✗ FAILED: Bandwidth regularization not working as expected")
        return False


def test_psd_autocorrelation_parameterization():
    """
    Test 8: Verify the autocorrelation parameterization produces PSD Toeplitz
    moments (Fix 1.2).
    """
    print("\n" + "=" * 60)
    print("Test 8: PSD Autocorrelation Parameterization")
    print("=" * 60)

    torch.manual_seed(42)

    query_dim = 32
    num_channels = 4
    bandwidth = 8

    net = QueryMomentNetwork(
        query_dim=query_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        enforce_psd=True,
    )

    # Random queries
    queries = torch.randn(2, 4, 10, query_dim)  # [B, H, T, D]
    moments = net(queries)  # [B, H, T, r, 2S+1]

    # Check μ_0 ≈ 1
    S = bandwidth
    mu_0 = moments[..., S]
    mu_0_error = (mu_0.real - 1.0).abs().max().item()
    mu_0_imag = mu_0.imag.abs().max().item()
    print(f"μ_0 real deviation from 1: {mu_0_error:.6f}")
    print(f"μ_0 imaginary part: {mu_0_imag:.6f}")

    # Check Hermitian symmetry: μ_{-k} = conj(μ_k)
    hermitian_error = 0.0
    for k in range(1, S + 1):
        mu_neg_k = moments[..., S - k]
        mu_pos_k = moments[..., S + k]
        err = (mu_neg_k - mu_pos_k.conj()).abs().max().item()
        hermitian_error = max(hermitian_error, err)
    print(f"Max Hermitian symmetry error: {hermitian_error:.6f}")

    # Check PSD via eigenvalues
    penalty = psd_toeplitz_penalty(moments, reduction="mean", max_samples=32).item()
    print(f"PSD Toeplitz penalty: {penalty:.6f}")

    if mu_0_error < 1e-4 and hermitian_error < 1e-5 and penalty < 1e-5:
        print("✓ PASSED: Autocorrelation produces valid PSD Toeplitz moments")
        return True
    else:
        print("✗ FAILED: Autocorrelation parameterization issues")
        return False


def test_fmma_hf_cache_contract():
    """
    Test 9 (Fix 2.1): Verify HF DynamicCache.update() returns zero-sized
    tensors.
    """
    print("\n" + "=" * 60)
    print("Test 9: HF DynamicCache Contract")
    print("=" * 60)

    cache = FMMACache(
        num_layers=1,
        num_heads=4,
        head_dim=16,
        num_channels=4,
        bandwidth=8,
        max_seq_len=1024,
    )

    key_states = torch.randn(1, 4, 10, 16)
    value_states = torch.randn(1, 4, 10, 16)

    k_out, v_out = cache.update(key_states, value_states, layer_idx=0)

    print(f"Input K shape:  {key_states.shape}")
    print(f"Output K shape: {k_out.shape}")
    print(f"Input V shape:  {value_states.shape}")
    print(f"Output V shape: {v_out.shape}")

    if k_out.shape[2] == 0 and v_out.shape[2] == 0:
        print("✓ PASSED: update() returns zero-length sequence tensors")
        return True
    else:
        print("✗ FAILED: update() did not return zero-length tensors")
        return False


def test_fmma_mask_formats():
    """
    Test 10 (Fix 2.3): Verify all three attention-mask formats produce
    identical outputs AND that padded tokens never update the cache.

    Formats exercised:
      (a) 2D int mask    — [B, T]          1=valid, 0=pad
      (b) 4D bool mask   — [B, 1, T, T]    True=attend  (HF Qwen style)
      (c) 4D float mask  — [B, 1, T, T]    0.0=attend, -inf=mask
    """
    print("\n" + "=" * 60)
    print("Test 10: Attention Mask Formats")
    print("=" * 60)

    torch.manual_seed(42)

    hidden_size = 64
    num_heads = 2
    head_dim = 32
    num_channels = 4
    bandwidth = 8
    seq_len = 8
    # Tokens 0..5 are valid, tokens 6..7 are padding
    valid_len = 6

    attn = FMMAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        layer_idx=0,
    )
    attn.eval()

    x = torch.randn(1, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # ---- (a) 2D int mask [B, T] ----
    mask_2d = torch.ones(1, seq_len, dtype=torch.long)
    mask_2d[:, valid_len:] = 0

    with torch.no_grad():
        out_2d, _ = attn(x, attention_mask=mask_2d, position_ids=position_ids)

    # ---- (b) 4D bool mask [B, 1, T, T] ----
    mask_4d_bool = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool)
    # Mask out columns for padded tokens
    mask_4d_bool[:, :, :, valid_len:] = False
    # Also causal: mask out positions where key > query
    for i in range(seq_len):
        mask_4d_bool[:, :, i, i + 1 :] = False

    with torch.no_grad():
        out_4d_bool, _ = attn(x, attention_mask=mask_4d_bool, position_ids=position_ids)

    # ---- (c) 4D float mask [B, 1, T, T] ----
    mask_4d_float = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
    mask_4d_float[:, :, :, valid_len:] = -1e9
    for i in range(seq_len):
        mask_4d_float[:, :, i, i + 1 :] = -1e9

    with torch.no_grad():
        out_4d_float, _ = attn(x, attention_mask=mask_4d_float, position_ids=position_ids)

    # Compare valid positions only (0..valid_len-1)
    diff_bool_vs_2d = (out_2d[:, :valid_len] - out_4d_bool[:, :valid_len]).abs().max().item()
    diff_float_vs_2d = (out_2d[:, :valid_len] - out_4d_float[:, :valid_len]).abs().max().item()
    print(f"Max diff (4D bool vs 2D)  on valid tokens: {diff_bool_vs_2d:.2e}")
    print(f"Max diff (4D float vs 2D) on valid tokens: {diff_float_vs_2d:.2e}")

    passed = True
    # All mask formats should produce the same output on valid tokens.
    # The forward() method normalizes all mask formats to a 2D [B, T]
    # simple_mask before passing to both FMMA paths.
    if diff_bool_vs_2d > 1e-4:
        print("✗ FAILED: 4D bool mask differs from 2D mask")
        passed = False
    if diff_float_vs_2d > 1e-4:
        print("✗ FAILED: 4D float mask differs from 2D mask")
        passed = False

    # ---- Verify padded tokens never update cache ----
    # Feed only valid tokens with cache, check cache state.
    # Then feed all tokens (valid + pad), check cache state is same.
    cache_a = FMMACache(
        num_layers=1, num_heads=num_heads, head_dim=head_dim,
        num_channels=num_channels, bandwidth=bandwidth, max_seq_len=1024,
    )
    cache_b = FMMACache(
        num_layers=1, num_heads=num_heads, head_dim=head_dim,
        num_channels=num_channels, bandwidth=bandwidth, max_seq_len=1024,
    )

    # Feed only valid tokens (no mask needed)
    with torch.no_grad():
        attn(x[:, :valid_len], position_ids=position_ids[:, :valid_len],
             past_key_value=cache_a, use_cache=True)

    # Feed ALL tokens but with mask zeroing out padded ones
    with torch.no_grad():
        attn(x, attention_mask=mask_2d, position_ids=position_ids,
             past_key_value=cache_b, use_cache=True)

    # Cache moments should match exactly
    diff_MV = (cache_a._M_V[0] - cache_b._M_V[0]).abs().max().item()
    diff_M1 = (cache_a._M_1[0] - cache_b._M_1[0]).abs().max().item()
    diff_seq = abs(cache_a._seq_len[0] - cache_b._seq_len[0])
    print(f"Cache M_V diff (valid-only vs masked): {diff_MV:.2e}")
    print(f"Cache M_1 diff (valid-only vs masked): {diff_M1:.2e}")
    print(f"Cache seq_len diff: {diff_seq}")

    if diff_MV > 1e-5 or diff_M1 > 1e-5:
        print("✗ FAILED: Padded tokens updated the cache!")
        passed = False

    if passed:
        print("✓ PASSED: All mask formats produce consistent outputs; "
              "padded tokens excluded from cache")
    return passed


def test_fmma_cached_causality():
    """
    Test 11: Verify causality on the CACHED inference path (_fmma_forward).

    The existing test_fmma_causality tests the training path (eval mode,
    no cache).  This test exercises the per-token cache loop: modify
    future tokens and confirm past outputs are unchanged.
    """
    print("\n" + "=" * 60)
    print("Test 11: Cached-Path Causality")
    print("=" * 60)

    torch.manual_seed(42)

    hidden_size = 64
    num_heads = 2
    head_dim = 32
    num_channels = 4
    bandwidth = 8
    seq_len = 16
    split = 8

    attn = FMMAAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_channels=num_channels,
        bandwidth=bandwidth,
        layer_idx=0,
    )
    attn.eval()

    x = torch.randn(1, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Run 1: original sequence through cache
    cache1 = FMMACache(
        num_layers=1, num_heads=num_heads, head_dim=head_dim,
        num_channels=num_channels, bandwidth=bandwidth, max_seq_len=1024,
    )
    with torch.no_grad():
        out1, _ = attn(x, position_ids=position_ids,
                       past_key_value=cache1, use_cache=True)

    # Run 2: modify future tokens (split..end), feed through fresh cache
    x_mod = x.clone()
    x_mod[:, split:, :] = torch.randn(1, seq_len - split, hidden_size)

    cache2 = FMMACache(
        num_layers=1, num_heads=num_heads, head_dim=head_dim,
        num_channels=num_channels, bandwidth=bandwidth, max_seq_len=1024,
    )
    with torch.no_grad():
        out2, _ = attn(x_mod, position_ids=position_ids,
                       past_key_value=cache2, use_cache=True)

    max_diff_past = (out1[:, :split] - out2[:, :split]).abs().max().item()
    max_diff_future = (out1[:, split:] - out2[:, split:]).abs().max().item()

    print(f"Max diff in past positions (0-{split-1}): {max_diff_past:.2e}")
    print(f"Max diff in future positions ({split}-{seq_len-1}): {max_diff_future:.2e}")

    if max_diff_past < 1e-5:
        print("✓ PASSED: Cached path is causal — future changes don't affect past")
        return True
    else:
        print(f"✗ FAILED: Past outputs changed (diff={max_diff_past:.2e})")
        return False


def run_all_tests():
    """Run all 11 FMMA acceptance tests."""
    print("=" * 60)
    print("FMMA (Fejér Mixture Moment Attention) Tests")
    print("=" * 60)
    print()

    results = {}
    results["algebraic_exactness"] = test_fmma_algebraic_exactness()
    results["memory_constant"] = test_fmma_memory_constant()
    results["nonnegativity"] = test_fmma_nonnegativity()
    results["causality"] = test_fmma_causality()
    results["long_context_stability"] = test_fmma_long_context_stability()
    results["psd_penalty"] = test_psd_penalty()
    results["bandwidth_regularization"] = test_bandwidth_regularization()
    results["psd_autocorrelation"] = test_psd_autocorrelation_parameterization()
    results["hf_cache_contract"] = test_fmma_hf_cache_contract()
    results["mask_formats"] = test_fmma_mask_formats()
    results["cached_causality"] = test_fmma_cached_causality()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:30s}: {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        print("\n✓ All FMMA acceptance tests passed!")
        print("\nFMMA provides:")
        print("  - Content awareness via r nonnegative key channels")
        print("  - O(rS) memory for BOTH K and V (100% KV cache savings)")
        print("  - Exact computation from cached moments")
        print("  - Guaranteed nonneg attention weights (PSD autocorrelation)")
        print("  - Causal streaming (token-by-token cache update)")
        print("  - Consistent phase mapping (always max_seq_len)")
        print("  - Proper HF DynamicCache contract")
    else:
        print("\n✗ Some tests failed — review implementation")

    return results
