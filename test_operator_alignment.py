"""
Phase 1: Operator Alignment Tests

This module contains the acceptance tests for CIMBA sketching of SpectralAttention.
These tests MUST pass before we can claim correctness.

Tests:
1. SpectralAttention(identity) matches Standard attention
2. SpectralAttention(hard bandlimit S) reference outputs (for comparison with CIMBA)
3. [Future] CIMBA-cache reproduces SpectralAttention(hard bandlimit S)

Definitions:
- Standard attention: softmax(QK^T / sqrt(d)) @ V
- SpectralAttention(identity): filter = 1.0 everywhere -> should match Standard
- SpectralAttention(hard bandlimit S): frequency bins > S are exactly zero

Run: python test_operator_alignment.py
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import copy

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32  # Use float32 for precision tests


def standard_attention(
    query: torch.Tensor,  # [batch, heads, q_len, head_dim]
    key: torch.Tensor,    # [batch, heads, kv_len, head_dim]
    value: torch.Tensor,  # [batch, heads, kv_len, head_dim]
    causal: bool = True,
) -> torch.Tensor:
    """
    Standard attention: softmax(QK^T / sqrt(d)) @ V
    
    This is the ground truth for comparison.
    """
    head_dim = query.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    
    # Compute scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply causal mask
    if causal:
        q_len, kv_len = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=scores.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Attend
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights


def spectral_attention_manual(
    query: torch.Tensor,  # [batch, heads, q_len, head_dim]
    key: torch.Tensor,    # [batch, heads, kv_len, head_dim]
    value: torch.Tensor,  # [batch, heads, kv_len, head_dim]
    bandwidth: int,       # S: number of frequency bins to keep (0 = identity)
    causal: bool = True,
    return_intermediate: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    SpectralAttention operator (manual implementation for testing).
    
    Operator:
    1. scores = QK^T / sqrt(d)
    2. scores_fft = rfft(scores, dim=kv_len)
    3. filtered_fft = scores_fft * hard_bandlimit_filter(S)
    4. scores_filtered = irfft(filtered_fft, n=kv_len)
    5. Apply causal mask
    6. attn_weights = softmax(scores_filtered)
    7. output = attn_weights @ V
    
    Args:
        bandwidth: S. If 0 or >= kv_len//2+1, acts as identity (no filtering).
    """
    batch, heads, q_len, head_dim = query.shape
    kv_len = key.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    
    # Step 1: Compute raw scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Step 2-4: FFT filtering (if bandwidth > 0)
    rfft_len = kv_len // 2 + 1
    
    if bandwidth > 0 and bandwidth < rfft_len:
        # Convert to float32 for FFT stability
        scores_f32 = scores.float()
        
        # Step 2: FFT along kv dimension
        scores_fft = torch.fft.rfft(scores_f32, dim=-1)
        
        # Step 3: Hard bandlimit - zero out frequencies > S
        mask = torch.zeros(rfft_len, device=scores.device, dtype=torch.float32)
        mask[:bandwidth] = 1.0
        scores_fft_filtered = scores_fft * mask
        
        # Step 4: IFFT back
        scores_filtered = torch.fft.irfft(scores_fft_filtered, n=kv_len, dim=-1)
        scores_filtered = scores_filtered.to(scores.dtype)
    else:
        # Identity: no filtering
        scores_filtered = scores
    
    # Step 5: Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=scores.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1
        )
        scores_filtered = scores_filtered.masked_fill(causal_mask, float('-inf'))
    
    # Step 6: Softmax
    attn_weights = F.softmax(scores_filtered, dim=-1)
    
    # Step 7: Attend
    output = torch.matmul(attn_weights, value)
    
    if return_intermediate:
        return output, attn_weights, scores, scores_filtered
    return output, attn_weights


def analyze_attention_spectrum(attn_weights: torch.Tensor, label: str = ""):
    """
    Analyze the frequency content of attention weights.
    
    This helps us understand whether post-softmax weights are bandlimited.
    """
    # Take mean over batch and heads for analysis
    weights_mean = attn_weights.mean(dim=(0, 1))  # [q_len, kv_len]
    
    # FFT along kv dimension for each query position
    weights_fft = torch.fft.rfft(weights_mean.float(), dim=-1)
    power = (weights_fft.abs() ** 2).mean(dim=0)  # [rfft_len]
    
    # Normalize
    power = power / power.sum()
    
    # Cumulative power
    cumsum = power.cumsum(dim=0)
    
    # Find 90%, 95%, 99% bandwidth
    total_bins = power.shape[0]
    bw_90 = (cumsum < 0.90).sum().item() + 1
    bw_95 = (cumsum < 0.95).sum().item() + 1
    bw_99 = (cumsum < 0.99).sum().item() + 1
    
    print(f"\n{label} Attention Spectrum Analysis:")
    print(f"  Total frequency bins: {total_bins}")
    print(f"  90% power in first {bw_90} bins ({100*bw_90/total_bins:.1f}%)")
    print(f"  95% power in first {bw_95} bins ({100*bw_95/total_bins:.1f}%)")
    print(f"  99% power in first {bw_99} bins ({100*bw_99/total_bins:.1f}%)")
    
    return {
        'total_bins': total_bins,
        'bw_90': bw_90,
        'bw_95': bw_95,
        'bw_99': bw_99,
        'power': power.cpu(),
    }


# =============================================================================
# TEST 1: SpectralAttention(identity) matches Standard attention
# =============================================================================

def test_identity_matches_standard():
    """
    Verify that SpectralAttention with identity filter (bandwidth=0) 
    exactly matches standard attention.
    
    This is a NON-NEGOTIABLE requirement for drop-in compatibility.
    """
    print("=" * 70)
    print("TEST 1: SpectralAttention(identity) vs Standard Attention")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    test_configs = [
        {'batch': 1, 'heads': 4, 'q_len': 64, 'kv_len': 64, 'head_dim': 32},
        {'batch': 2, 'heads': 8, 'q_len': 128, 'kv_len': 128, 'head_dim': 64},
        {'batch': 1, 'heads': 4, 'q_len': 1, 'kv_len': 256, 'head_dim': 64},  # Generation case
    ]
    
    all_passed = True
    
    for cfg in test_configs:
        Q = torch.randn(cfg['batch'], cfg['heads'], cfg['q_len'], cfg['head_dim'], 
                       device=DEVICE, dtype=DTYPE)
        K = torch.randn(cfg['batch'], cfg['heads'], cfg['kv_len'], cfg['head_dim'], 
                       device=DEVICE, dtype=DTYPE)
        V = torch.randn(cfg['batch'], cfg['heads'], cfg['kv_len'], cfg['head_dim'], 
                       device=DEVICE, dtype=DTYPE)
        
        # Standard attention
        out_std, _ = standard_attention(Q, K, V, causal=True)
        
        # SpectralAttention(identity) - bandwidth=0 means no filtering
        out_spec, _ = spectral_attention_manual(Q, K, V, bandwidth=0, causal=True)
        
        max_err = (out_std - out_spec).abs().max().item()
        mean_err = (out_std - out_spec).abs().mean().item()
        
        passed = max_err < 1e-5
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"\nConfig: {cfg}")
        print(f"  Max error:  {max_err:.2e}")
        print(f"  Mean error: {mean_err:.2e}")
        print(f"  {status}")
        
        if not passed:
            all_passed = False
    
    return all_passed


# =============================================================================
# TEST 2: SpectralAttention(hard bandlimit S) reference + spectrum analysis
# =============================================================================

def test_hard_bandlimit_reference():
    """
    Generate reference outputs for SpectralAttention(hard bandlimit S).
    
    Also analyze whether the POST-SOFTMAX attention weights are bandlimited.
    This is critical: if they're not bandlimited, CIMBA sketching won't be exact.
    """
    print("\n" + "=" * 70)
    print("TEST 2: SpectralAttention(hard bandlimit S) Reference + Spectrum Analysis")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Test configuration
    batch, heads, seq_len, head_dim = 1, 4, 256, 64
    S_values = [8, 16, 32, 64]  # Different bandwidths to test
    
    Q = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    K = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    V = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    print(f"\nConfiguration: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")
    
    results = {}
    
    for S in S_values:
        print(f"\n--- Bandwidth S = {S} ---")
        
        # Get SpectralAttention output
        out_spec, attn_weights, scores_raw, scores_filt = spectral_attention_manual(
            Q, K, V, bandwidth=S, causal=True, return_intermediate=True
        )
        
        # Analyze PRE-softmax filtered scores spectrum
        # Only look at valid (non -inf) positions
        scores_valid = scores_filt.clone()
        scores_valid[scores_valid == float('-inf')] = 0  # Replace -inf with 0 for FFT
        scores_fft = torch.fft.rfft(scores_valid.float(), dim=-1)
        scores_power = (scores_fft.abs() ** 2).mean(dim=(0, 1, 2))
        total_power = scores_power.sum()
        if total_power > 0:
            scores_power_normalized = scores_power / total_power
            power_beyond_S = scores_power_normalized[S:].sum().item()
        else:
            power_beyond_S = 0.0
        print(f"  Pre-softmax power beyond S={S}: {power_beyond_S:.2e} (should be ~0)")
        
        # Analyze POST-softmax attention weights spectrum
        spectrum_info = analyze_attention_spectrum(attn_weights, f"S={S}")
        
        # KEY QUESTION: Is post-softmax bandlimited?
        # If 99% of power is NOT in first S bins, CIMBA sketching won't be exact
        is_bandlimited = spectrum_info['bw_99'] <= S * 1.5  # Allow some margin
        print(f"  Post-softmax bandlimited to S? {is_bandlimited} (99% power needs {spectrum_info['bw_99']} bins)")
        
        results[S] = {
            'output': out_spec,
            'attn_weights': attn_weights,
            'spectrum': spectrum_info,
            'pre_softmax_power_beyond_S': power_beyond_S,
            'is_post_softmax_bandlimited': is_bandlimited,
        }
    
    return results


# =============================================================================
# TEST 3: Mathematical barrier analysis
# =============================================================================

def test_softmax_destroys_bandlimit():
    """
    Demonstrate that softmax destroys bandlimitedness.
    
    Even if pre-softmax scores are perfectly bandlimited (only S frequency bins),
    the softmax operation introduces high-frequency content.
    
    This is the core challenge for CIMBA sketching SpectralAttention.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Does Softmax Preserve Bandlimitedness?")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    seq_len = 256
    S = 16  # Bandwidth
    
    # Create a perfectly bandlimited signal (only first S frequency components)
    rfft_len = seq_len // 2 + 1
    fft_coeffs = torch.zeros(rfft_len, dtype=torch.complex64, device=DEVICE)
    fft_coeffs[:S] = torch.randn(S, dtype=torch.complex64, device=DEVICE)
    
    # Convert to position domain
    bandlimited_signal = torch.fft.irfft(fft_coeffs, n=seq_len)
    
    # Verify it's bandlimited
    recon_fft = torch.fft.rfft(bandlimited_signal)
    power_beyond_S = (recon_fft[S:].abs() ** 2).sum() / (recon_fft.abs() ** 2).sum()
    print(f"\nOriginal signal power beyond S={S}: {power_beyond_S:.2e} (should be ~0)")
    
    # Apply softmax
    softmax_signal = F.softmax(bandlimited_signal, dim=-1)
    
    # Check if result is still bandlimited
    softmax_fft = torch.fft.rfft(softmax_signal)
    softmax_power = (softmax_fft.abs() ** 2)
    softmax_power_normalized = softmax_power / softmax_power.sum()
    
    power_in_S = softmax_power_normalized[:S].sum().item()
    power_beyond_S_softmax = softmax_power_normalized[S:].sum().item()
    
    print(f"\nAfter softmax:")
    print(f"  Power in first {S} bins: {power_in_S:.4f} ({100*power_in_S:.1f}%)")
    print(f"  Power beyond S={S}: {power_beyond_S_softmax:.4f} ({100*power_beyond_S_softmax:.1f}%)")
    
    # How many bins for 99% power?
    cumsum = softmax_power_normalized.cumsum(dim=0)
    bw_99 = (cumsum < 0.99).sum().item() + 1
    print(f"  99% power requires {bw_99} bins (out of {rfft_len})")
    
    conclusion = power_beyond_S_softmax < 0.01
    print(f"\n  CONCLUSION: Softmax preserves bandlimit? {conclusion}")
    print(f"  (If False, CIMBA cannot exactly sketch SpectralAttention)")
    
    return conclusion


# =============================================================================
# TEST 4: Placeholder for CIMBA vs SpectralAttention comparison
# =============================================================================

def test_cimba_vs_spectral_placeholder():
    """
    [PLACEHOLDER] This test will compare CIMBA-cache reconstruction 
    against SpectralAttention(hard bandlimit S).
    
    This is the ACCEPTANCE GATE. CIMBA implementation is correct only if:
    - max_abs_error(spectral_output, cimba_output) <= 2e-5 (float32)
    - mean_abs_error <= 1e-6
    
    Currently returns False because CIMBA for SpectralAttention is not implemented.
    """
    print("\n" + "=" * 70)
    print("TEST 4: CIMBA vs SpectralAttention(hard bandlimit S) [PLACEHOLDER]")
    print("=" * 70)
    
    print("\nCIMBA sketching for SpectralAttention is NOT YET IMPLEMENTED.")
    print("This test will be filled in once the implementation exists.")
    print("\nExpected acceptance criteria:")
    print("  - max_abs_error <= 2e-5 (float32)")
    print("  - mean_abs_error <= 1e-6")
    print("  - Must work for seq_len in [256, 1024, 4096]")
    
    return False  # Placeholder always fails


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("OPERATOR ALIGNMENT TESTS")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    
    results = {}
    
    # Test 1: Identity matches standard
    results['identity_matches_standard'] = test_identity_matches_standard()
    
    # Test 2: Hard bandlimit reference + spectrum analysis
    results['bandlimit_reference'] = test_hard_bandlimit_reference()
    
    # Test 3: Does softmax preserve bandlimit?
    results['softmax_preserves_bandlimit'] = test_softmax_destroys_bandlimit()
    
    # Test 4: CIMBA vs SpectralAttention (placeholder)
    results['cimba_vs_spectral'] = test_cimba_vs_spectral_placeholder()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n1. SpectralAttention(identity) matches Standard: "
          f"{'✓ PASS' if results['identity_matches_standard'] else '✗ FAIL'}")
    
    print(f"2. Hard bandlimit reference generated: ✓")
    
    print(f"3. Softmax preserves bandlimit: "
          f"{'✓ YES' if results['softmax_preserves_bandlimit'] else '✗ NO'}")
    
    print(f"4. CIMBA vs SpectralAttention: "
          f"{'✓ PASS' if results['cimba_vs_spectral'] else '⚠ NOT IMPLEMENTED'}")
    
    # Critical finding
    if not results['softmax_preserves_bandlimit']:
        print("\n" + "!" * 70)
        print("CRITICAL FINDING: Softmax destroys bandlimitedness!")
        print("!" * 70)
        print("""
This means the POST-SOFTMAX attention weights are NOT bandlimited,
even when pre-softmax scores are perfectly bandlimited.

Implications for CIMBA sketching:
1. Standard CIMBA moment sketching assumes bandlimited attention weights.
2. Since softmax introduces high-frequency content, exact reconstruction
   from O(S) moments is mathematically impossible for SpectralAttention.

Possible paths forward:
A) Accept approximation error (not exact, but maybe good enough)
B) Remove softmax entirely (use different normalization)
C) Sketch the pre-softmax filtered scores instead of V-weighted output
D) Use the existing Dirichlet-based CIMBAAttention (different operator)

This is a fundamental mathematical barrier that must be addressed.
""")
    
    return results


if __name__ == "__main__":
    main()
