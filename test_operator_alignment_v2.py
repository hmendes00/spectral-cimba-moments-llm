"""
Phase 1: Operator Alignment Tests (REVISED v2)

Methodologically sound tests for analyzing CIMBA sketching of SpectralAttention.

BANDLIMIT_MODE controls mask behavior:
  - "bins": keep exactly S bins [0..S-1]
  - "inclusive": keep S+1 bins [0..S]

Run: python test_operator_alignment_v2.py
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# ============================================================================
# CONFIGURATION
# ============================================================================
BANDLIMIT_MODE = "inclusive"  # "bins" or "inclusive"


def get_bandlimit_cutoff(S: int) -> int:
    """Return the number of bins kept based on BANDLIMIT_MODE."""
    if BANDLIMIT_MODE == "bins":
        return S  # keep [0..S-1], exactly S bins
    else:  # inclusive
        return S + 1  # keep [0..S], exactly S+1 bins


def apply_hard_bandlimit(signal_fft: torch.Tensor, S: int) -> torch.Tensor:
    """Apply hard bandlimit mask to FFT signal."""
    rfft_len = signal_fft.shape[-1]
    cutoff = get_bandlimit_cutoff(S)
    cutoff = min(cutoff, rfft_len)
    
    mask = torch.zeros(rfft_len, device=signal_fft.device, dtype=torch.float32)
    mask[:cutoff] = 1.0
    
    return signal_fft * mask.to(signal_fft.dtype)


def compute_spectrum_stats(signal: torch.Tensor, S: int) -> Dict:
    """
    Compute spectrum statistics for a 1D signal.
    
    Returns:
        total_power: sum of |fft|^2
        power_beyond_cutoff: power in bins >= cutoff
        relative_power_beyond: power_beyond / total_power
        bins_for_99: number of bins needed for 99% power
    """
    rfft_len = signal.shape[-1] // 2 + 1
    cutoff = get_bandlimit_cutoff(S)
    
    signal_fft = torch.fft.rfft(signal.float())
    power = (signal_fft.abs() ** 2)
    total_power = power.sum().item()
    
    if total_power < 1e-20:
        return {
            'total_power': total_power,
            'power_beyond_cutoff': 0.0,
            'relative_power_beyond': 0.0,
            'bins_for_99': 0,
            'cutoff': cutoff,
            'rfft_len': rfft_len,
        }
    
    power_beyond = power[cutoff:].sum().item() if cutoff < rfft_len else 0.0
    relative_beyond = power_beyond / total_power
    
    # Bins for 99% power
    power_normalized = power / total_power
    cumsum = power_normalized.cumsum(dim=0)
    bins_for_99 = (cumsum < 0.99).sum().item() + 1
    
    return {
        'total_power': total_power,
        'power_beyond_cutoff': power_beyond,
        'relative_power_beyond': relative_beyond,
        'bins_for_99': bins_for_99,
        'cutoff': cutoff,
        'rfft_len': rfft_len,
    }


def spectral_attention_single_query(
    query: torch.Tensor,  # [batch, heads, 1, head_dim]
    key: torch.Tensor,    # [batch, heads, kv_len, head_dim]
    value: torch.Tensor,  # [batch, heads, kv_len, head_dim]
    S: int,
    apply_softmax: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SpectralAttention for a SINGLE query.
    
    Returns:
        output: [batch, heads, 1, head_dim]
        weights: [batch, heads, kv_len] (post-normalization)
        scores_filtered: [batch, heads, kv_len] (pre-normalization, bandlimited)
    """
    head_dim = query.shape[-1]
    kv_len = key.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    
    # Raw scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    scores = scores.squeeze(2)  # [batch, heads, kv_len]
    
    # FFT bandlimit
    scores_f32 = scores.float()
    scores_fft = torch.fft.rfft(scores_f32, dim=-1)
    scores_fft_filtered = apply_hard_bandlimit(scores_fft, S)
    scores_filtered = torch.fft.irfft(scores_fft_filtered, n=kv_len, dim=-1)
    scores_filtered = scores_filtered.to(scores.dtype)
    
    # Normalization
    if apply_softmax:
        weights = F.softmax(scores_filtered, dim=-1)
    else:
        # L1 normalization (no exp, just normalize to sum=1)
        # Shift to positive then normalize
        shifted = scores_filtered - scores_filtered.min(dim=-1, keepdim=True)[0] + 1e-6
        weights = shifted / shifted.sum(dim=-1, keepdim=True)
    
    # Output
    output = torch.matmul(weights.unsqueeze(2), value)
    
    return output, weights, scores_filtered


# ============================================================================
# TEST 1: SpectralAttention(identity) matches Standard attention
# ============================================================================

def test_identity_matches_standard():
    print("=" * 70)
    print("TEST 1: SpectralAttention(identity) vs Standard Attention")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    configs = [
        {'batch': 1, 'heads': 4, 'kv_len': 64, 'head_dim': 32},
        {'batch': 1, 'heads': 4, 'kv_len': 256, 'head_dim': 64},
    ]
    
    all_passed = True
    
    for cfg in configs:
        Q = torch.randn(cfg['batch'], cfg['heads'], 1, cfg['head_dim'], device=DEVICE, dtype=DTYPE)
        K = torch.randn(cfg['batch'], cfg['heads'], cfg['kv_len'], cfg['head_dim'], device=DEVICE, dtype=DTYPE)
        V = torch.randn(cfg['batch'], cfg['heads'], cfg['kv_len'], cfg['head_dim'], device=DEVICE, dtype=DTYPE)
        
        # Standard attention
        scale = 1.0 / math.sqrt(cfg['head_dim'])
        scores_std = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights_std = F.softmax(scores_std, dim=-1)
        out_std = torch.matmul(weights_std, V)
        
        # SpectralAttention with S large enough to be identity
        S_identity = cfg['kv_len']  # All bins
        out_spec, _, _ = spectral_attention_single_query(Q, K, V, S=S_identity, apply_softmax=True)
        
        max_err = (out_std - out_spec).abs().max().item()
        passed = max_err < 1e-5
        print(f"\nkv_len={cfg['kv_len']}: max_err={max_err:.2e} {'✓' if passed else '✗'}")
        if not passed:
            all_passed = False
    
    return all_passed


# ============================================================================
# TEST 2: Pre-softmax bandlimitedness + sanity check
# ============================================================================

def test_presoftmax_bandlimited():
    print("\n" + "=" * 70)
    print("TEST 2: Pre-Softmax Scores Are Bandlimited (with sanity check)")
    print(f"        BANDLIMIT_MODE = '{BANDLIMIT_MODE}'")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    kv_len = 256
    head_dim = 64
    S_values = [8, 16, 32, 64]
    
    Q = torch.randn(1, 1, 1, head_dim, device=DEVICE, dtype=DTYPE)
    K = torch.randn(1, 1, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
    V = torch.randn(1, 1, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    all_passed = True
    
    for S in S_values:
        _, _, scores_filtered = spectral_attention_single_query(Q, K, V, S=S, apply_softmax=True)
        scores_row = scores_filtered[0, 0]  # [kv_len]
        
        # Sanity check: re-FFT and verify power beyond cutoff
        stats = compute_spectrum_stats(scores_row, S)
        
        cutoff = stats['cutoff']
        passed = stats['relative_power_beyond'] < 1e-10
        
        print(f"\n  S={S} (keeping bins [0..{cutoff-1}], {cutoff} bins total):")
        print(f"    Total power:          {stats['total_power']:.4e}")
        print(f"    Power beyond cutoff:  {stats['power_beyond_cutoff']:.2e}")
        print(f"    Relative beyond:      {stats['relative_power_beyond']:.2e} {'✓' if passed else '✗ FAIL'}")
        
        if not passed:
            all_passed = False
    
    return all_passed


# ============================================================================
# TEST 3: Post-softmax spectrum analysis
# ============================================================================

def test_postsoftmax_spectrum():
    print("\n" + "=" * 70)
    print("TEST 3: Post-Softmax Attention Weight Spectrum")
    print(f"        BANDLIMIT_MODE = '{BANDLIMIT_MODE}'")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    kv_len = 256
    head_dim = 64
    S_values = [8, 16, 32, 64]
    
    Q = torch.randn(1, 1, 1, head_dim, device=DEVICE, dtype=DTYPE)
    K = torch.randn(1, 1, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
    V = torch.randn(1, 1, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    rfft_len = kv_len // 2 + 1
    
    for S in S_values:
        _, attn_weights, scores_filtered = spectral_attention_single_query(Q, K, V, S=S, apply_softmax=True)
        weights_row = attn_weights[0, 0]  # [kv_len]
        scores_row = scores_filtered[0, 0]  # [kv_len]
        
        cutoff = get_bandlimit_cutoff(S)
        
        # Pre-softmax stats
        pre_stats = compute_spectrum_stats(scores_row, S)
        
        # Post-softmax stats
        post_stats = compute_spectrum_stats(weights_row, S)
        
        print(f"\n  S={S} (cutoff={cutoff} bins, rfft_len={rfft_len}):")
        print(f"    PRE-SOFTMAX:")
        print(f"      Total power:        {pre_stats['total_power']:.4e}")
        print(f"      Power beyond S:     {pre_stats['relative_power_beyond']:.2e}")
        print(f"      Bins for 99%:       {pre_stats['bins_for_99']}")
        print(f"    POST-SOFTMAX:")
        print(f"      Total power:        {post_stats['total_power']:.4e}")
        print(f"      Power beyond S:     {post_stats['relative_power_beyond']:.4f} ({100*post_stats['relative_power_beyond']:.1f}%)")
        print(f"      Bins for 99%:       {post_stats['bins_for_99']}")


# ============================================================================
# TEST 4: Position-only moment sketch cannot match content-based attention
# ============================================================================

def test_position_vs_content():
    print("\n" + "=" * 70)
    print("TEST 4: Position-only moment sketch (Dirichlet-style)")
    print("        cannot match content-based SpectralAttention")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    kv_len = 128
    head_dim = 32
    S = 32
    max_seq_len = 4096
    
    Q = torch.randn(1, 1, 1, head_dim, device=DEVICE, dtype=DTYPE)
    K = torch.randn(1, 1, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
    V = torch.randn(1, 1, kv_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    V_flat = V.squeeze()  # [kv_len, head_dim]
    
    # Position setup
    theta = 2 * math.pi * torch.arange(kv_len, device=DEVICE).float() / max_seq_len
    k_vals = torch.arange(-S, S+1, device=DEVICE, dtype=torch.float32)
    query_theta = theta[-1]  # Query at last position
    
    # ========================================================================
    # TEST 4a: SpectralAttention WITH softmax vs Position-only CIMBA
    # ========================================================================
    print(f"\n  TEST 4a: SpectralAttention (softmax) vs Position-only CIMBA")
    print(f"           kv_len={kv_len}, S={S}")
    
    # SpectralAttention output (with softmax)
    out_spectral_softmax, weights_softmax, _ = spectral_attention_single_query(
        Q, K, V, S=S, apply_softmax=True
    )
    out_spectral_softmax = out_spectral_softmax.squeeze()  # [head_dim]
    
    # Position-only CIMBA reconstruction
    V_complex = V_flat.to(torch.complex64)
    phase_matrix = torch.exp(1j * k_vals.unsqueeze(1) * theta.unsqueeze(0))  # [2S+1, kv_len]
    moments = torch.einsum('kn,nd->kd', phase_matrix, V_complex)
    norm_moments = phase_matrix.sum(dim=1)
    
    query_phase = torch.exp(-1j * k_vals * query_theta)
    raw_output = (query_phase @ moments).real / (2*S + 1)
    norm_factor = (query_phase @ norm_moments).real / (2*S + 1)
    out_cimba = raw_output / (norm_factor + 1e-10)
    
    err_4a = (out_spectral_softmax - out_cimba).abs()
    max_err_4a = err_4a.max().item()
    mean_err_4a = err_4a.mean().item()
    rel_err_4a = mean_err_4a / (out_spectral_softmax.abs().mean().item() + 1e-10)
    
    print(f"    SpectralAttention(softmax) output norm: {out_spectral_softmax.norm().item():.4f}")
    print(f"    Position-CIMBA output norm:             {out_cimba.norm().item():.4f}")
    print(f"    Max abs error:  {max_err_4a:.4f}")
    print(f"    Mean abs error: {mean_err_4a:.4f}")
    print(f"    Relative error: {100*rel_err_4a:.1f}%")
    print(f"    Match? {'✓' if max_err_4a < 1e-4 else '✗ NO'}")
    
    # ========================================================================
    # TEST 4b: SpectralAttention WITHOUT softmax (L1 norm) vs Position-only CIMBA
    # This removes softmax vs Dirichlet-renorm as confound
    # ========================================================================
    print(f"\n  TEST 4b: SpectralAttention (L1 norm, no softmax) vs Position-only CIMBA")
    print(f"           Removes softmax-vs-renorm confound")
    
    # SpectralAttention output (without softmax, using L1 norm)
    out_spectral_l1, weights_l1, scores_filtered = spectral_attention_single_query(
        Q, K, V, S=S, apply_softmax=False
    )
    out_spectral_l1 = out_spectral_l1.squeeze()  # [head_dim]
    
    # Position-only CIMBA uses same Dirichlet-style renorm (already computed above)
    
    err_4b = (out_spectral_l1 - out_cimba).abs()
    max_err_4b = err_4b.max().item()
    mean_err_4b = err_4b.mean().item()
    rel_err_4b = mean_err_4b / (out_spectral_l1.abs().mean().item() + 1e-10)
    
    print(f"    SpectralAttention(L1 norm) output norm: {out_spectral_l1.norm().item():.4f}")
    print(f"    Position-CIMBA output norm:             {out_cimba.norm().item():.4f}")
    print(f"    Max abs error:  {max_err_4b:.4f}")
    print(f"    Mean abs error: {mean_err_4b:.4f}")
    print(f"    Relative error: {100*rel_err_4b:.1f}%")
    print(f"    Match? {'✓' if max_err_4b < 1e-4 else '✗ NO'}")
    
    # ========================================================================
    # Analysis: Compare actual attention weights
    # ========================================================================
    print(f"\n  ANALYSIS: Why does it fail?")
    
    # Compute Dirichlet kernel weights for comparison
    def dirichlet_kernel(delta, S):
        x = delta + 1e-10
        return torch.sin((S + 0.5) * x) / torch.sin(x / 2) / (2*S + 1)
    
    delta = query_theta - theta  # [kv_len]
    dirichlet_weights = dirichlet_kernel(delta, S)
    dirichlet_weights = dirichlet_weights / dirichlet_weights.sum()  # Normalize
    
    # SpectralAttention weights (softmax version)
    spectral_weights = weights_softmax.squeeze()  # [kv_len]
    
    weight_diff = (spectral_weights - dirichlet_weights).abs()
    print(f"    Attention weight difference (SpectralAttention vs Dirichlet):")
    print(f"      Max:  {weight_diff.max().item():.4f}")
    print(f"      Mean: {weight_diff.mean().item():.4f}")
    
    # Key insight: where do they differ most?
    max_diff_idx = weight_diff.argmax().item()
    print(f"    Max diff at position {max_diff_idx}:")
    print(f"      SpectralAttention weight: {spectral_weights[max_diff_idx].item():.6f}")
    print(f"      Dirichlet weight:         {dirichlet_weights[max_diff_idx].item():.6f}")
    
    print(f"\n  CONCLUSION:")
    print(f"    Position-only moments weight V by position: Σᵢ Vᵢ·f(θᵢ-θ_query)")
    print(f"    SpectralAttention weights V by content:     Σᵢ Vᵢ·g(Q,Kᵢ)")
    print(f"    These are fundamentally different - failure is due to CONTENT DEPENDENCE,")
    print(f"    not due to softmax-vs-renorm mismatch.")
    
    return max_err_4a, max_err_4b


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("OPERATOR ALIGNMENT TESTS v2")
    print(f"BANDLIMIT_MODE = '{BANDLIMIT_MODE}'")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    print("=" * 70)
    
    # Test 1
    test1_passed = test_identity_matches_standard()
    
    # Test 2
    test2_passed = test_presoftmax_bandlimited()
    
    # Test 3
    test_postsoftmax_spectrum()
    
    # Test 4
    err_4a, err_4b = test_position_vs_content()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (identity=standard):     {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"  Test 2 (pre-softmax bandlimit): {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"  Test 3: See spectrum analysis above")
    print(f"  Test 4a (position vs spectral+softmax): err={err_4a:.4f} {'✗ EXPECTED' if err_4a > 0.01 else '✓'}")
    print(f"  Test 4b (position vs spectral+L1norm):  err={err_4b:.4f} {'✗ EXPECTED' if err_4b > 0.01 else '✓'}")


if __name__ == "__main__":
    main()
