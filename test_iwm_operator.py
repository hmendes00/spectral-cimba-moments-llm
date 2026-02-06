"""
IWM Content-Kernel Attention Operator Tests

Acceptance gates for IWMContentKernelAttention:
T1) Kernel bandlimit sanity: FFT of kernel shows power ~0 beyond S
T2) Nonnegativity sanity: kernel values >= -eps on a grid
T3) Moment vs explicit sum correctness: both paths give same output
T4) Constant memory: cache shapes remain fixed during long generation

Usage:
    python test_iwm_operator.py
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32  # Use float32 for test precision


def test_kernel_bandlimit():
    """
    T1: Verify kernel is bandlimited to |k| <= S.
    
    For Fejér-smoothed kernel with bandwidth S, FFT power should be ~0
    beyond the 2S+1 low-frequency bins.
    """
    print("\n" + "="*60)
    print("TEST 1: Kernel Bandlimit Sanity")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention
    
    for S in [8, 16, 32, 64]:
        print(f"\n--- S={S} ---")
        
        # Create minimal attention module
        attn = IWMContentKernelAttention(
            hidden_size=64,
            num_heads=1,
            head_dim=64,
            bandwidth=S,
        ).to(DEVICE)
        
        # Random coefficients (simulating query-predicted)
        batch, heads, seq = 1, 1, 1
        c_coeffs = torch.randn(batch, heads, seq, S + 1, device=DEVICE) * 0.3
        c_coeffs[..., 0] = 1.0  # Ensure c_0 is dominant (helps positivity)
        
        # Build kernel coefficients
        g_coeffs = attn._build_kernel_coeffs(c_coeffs)  # [1, 1, 1, 2S+1] complex
        
        # Evaluate kernel on fine grid
        num_points = 1024
        delta_theta = torch.linspace(-math.pi, math.pi, num_points, device=DEVICE)
        
        # Kernel: p(Δθ) = Σ_k g_k e^{ikΔθ}
        k_vals = attn.k_vals  # [-S, ..., S]
        phase = torch.exp(1j * k_vals.unsqueeze(1) * delta_theta.unsqueeze(0))  # [2S+1, num_points]
        
        g_flat = g_coeffs.squeeze()  # [2S+1]
        kernel_vals = torch.einsum('k,kp->p', g_flat, phase).real  # [num_points]
        
        # FFT of kernel
        kernel_fft = torch.fft.fft(kernel_vals)
        power = torch.abs(kernel_fft) ** 2
        power = power / power.sum()  # Normalize
        
        # Power in low-frequency bins
        # FFT bins: [0, 1, ..., N/2, -N/2+1, ..., -1]
        # For bandlimit S, should have power only in bins |k| <= S
        # With 1024 points, S << 512, so bins [0..S] and [1024-S..1023] should have power
        
        low_freq_mask = torch.zeros(num_points, dtype=torch.bool, device=DEVICE)
        low_freq_mask[:S+1] = True  # Bins 0..S
        low_freq_mask[-(S):] = True  # Bins -S..-1
        
        power_low = power[low_freq_mask].sum().item()
        power_high = power[~low_freq_mask].sum().item()
        
        print(f"  Power in |k|<={S}: {power_low:.6f}")
        print(f"  Power in |k|>{S}:  {power_high:.2e}")
        
        # Check: high-frequency power should be negligible
        # FFT numerical precision degrades slightly at larger S
        threshold = 1e-5  # Allow up to 0.001% power in high frequencies
        if power_high < threshold:
            print(f"  ✓ PASS: Kernel is bandlimited")
        else:
            print(f"  ✗ FAIL: Significant high-frequency power (>{threshold})")
            return False
    
    return True


def test_kernel_nonnegativity():
    """
    T2: Verify kernel is nonnegative (or nearly so) on a grid.
    
    Fejér smoothing with proper coefficients should give nonnegative kernel.
    We allow small negative values (< -1e-6) due to soft PSD constraint.
    """
    print("\n" + "="*60)
    print("TEST 2: Kernel Nonnegativity Sanity")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention
    
    S = 32
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=1,
        head_dim=64,
        bandwidth=S,
    ).to(DEVICE)
    
    # Test multiple random coefficient samples
    num_tests = 20
    max_negatives = []
    
    for i in range(num_tests):
        # Random coefficients with c_0 dominant
        c_coeffs = torch.randn(1, 1, 1, S + 1, device=DEVICE) * 0.2
        c_coeffs[..., 0] = 1.0  # Dominant DC term
        
        # Decay higher harmonics (helps PSD)
        decay = torch.exp(-0.1 * torch.arange(S + 1, device=DEVICE))
        c_coeffs = c_coeffs * decay
        
        # Build kernel
        g_coeffs = attn._build_kernel_coeffs(c_coeffs)
        
        # Evaluate on grid
        num_points = 512
        delta_theta = torch.linspace(-math.pi, math.pi, num_points, device=DEVICE)
        k_vals = attn.k_vals
        phase = torch.exp(1j * k_vals.unsqueeze(1) * delta_theta.unsqueeze(0))
        
        g_flat = g_coeffs.squeeze()
        kernel_vals = torch.einsum('k,kp->p', g_flat, phase).real
        
        min_val = kernel_vals.min().item()
        max_negatives.append(min_val)
    
    worst_negative = min(max_negatives)
    avg_min = sum(max_negatives) / len(max_negatives)
    
    print(f"\n  Tested {num_tests} random coefficient samples")
    print(f"  Worst minimum kernel value: {worst_negative:.6f}")
    print(f"  Average minimum: {avg_min:.6f}")
    
    # Allow small negatives due to soft PSD constraint
    tolerance = -1e-4
    if worst_negative >= tolerance:
        print(f"  ✓ PASS: Kernel is effectively nonnegative (>= {tolerance})")
        return True
    else:
        print(f"  ✗ FAIL: Significant negative kernel values")
        # Note: This may fail if coefficients don't satisfy PSD constraint
        # The training will use PSD loss to encourage positivity
        print(f"  (Note: PSD loss during training encourages positivity)")
        return False


def test_moment_vs_explicit():
    """
    T3: Verify moment reconstruction matches explicit computation.
    
    Both training (prefix moments with cumsum) and explicit (causal mask)
    should produce identical outputs.
    """
    print("\n" + "="*60)
    print("TEST 3: Moment vs Explicit Sum Correctness")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention
    
    S = 16
    head_dim = 64
    kv_len = 64
    
    attn = IWMContentKernelAttention(
        hidden_size=head_dim,
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        bandwidth=S,
        layer_idx=0,
    ).to(DEVICE)
    
    # Random input (batch=1 for simplicity)
    hidden_states = torch.randn(1, kv_len, head_dim, device=DEVICE)
    
    # Create dummy position embeddings
    cos = torch.ones(1, kv_len, head_dim, device=DEVICE)
    sin = torch.zeros(1, kv_len, head_dim, device=DEVICE)
    position_embeddings = (cos, sin)
    
    # Method (a): forward_with_cache in training mode (prefix moments + cumsum)
    # This is what training uses
    output_moment, psd_loss = attn.forward_with_cache(
        hidden_states, position_embeddings, 
        cache=None,  # No cache = training mode
        attention_mask=None
    )
    
    # Method (b): forward_explicit with causal mask
    output_explicit, attn_weights, _ = attn.forward_explicit(
        hidden_states, position_embeddings
    )
    
    # Compare outputs
    diff = (output_explicit - output_moment).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Per-position differences
    per_pos_max = diff.squeeze(0).max(dim=-1).values  # [kv_len]
    
    print(f"\n  Sequence length: {kv_len}")
    print(f"  Bandwidth S: {S}")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Per-position max diff (first 5): {per_pos_max[:5].tolist()}")
    print(f"  Per-position max diff (last 5): {per_pos_max[-5:].tolist()}")
    
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"  ✓ PASS: Moment and explicit outputs match (tol={tolerance})")
        return True
    else:
        print(f"  ✗ FAIL: Outputs differ by more than {tolerance}")
        return False


def test_causality():
    """
    T3b: Verify causality - position t should not see positions > t.
    
    Compare outputs when future tokens are changed vs not.
    """
    print("\n" + "="*60)
    print("TEST 3b: Causality Verification")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention
    
    S = 16
    head_dim = 64
    kv_len = 32
    
    attn = IWMContentKernelAttention(
        hidden_size=head_dim,
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        bandwidth=S,
        layer_idx=0,
    ).to(DEVICE)
    
    # Position embeddings
    cos = torch.ones(1, kv_len, head_dim, device=DEVICE)
    sin = torch.zeros(1, kv_len, head_dim, device=DEVICE)
    position_embeddings = (cos, sin)
    
    # Input A: random
    hidden_A = torch.randn(1, kv_len, head_dim, device=DEVICE)
    
    # Input B: same as A for first half, different for second half
    hidden_B = hidden_A.clone()
    hidden_B[:, kv_len//2:, :] = torch.randn(1, kv_len//2, head_dim, device=DEVICE)
    
    # Forward both
    output_A, _ = attn.forward_with_cache(hidden_A, position_embeddings, cache=None)
    output_B, _ = attn.forward_with_cache(hidden_B, position_embeddings, cache=None)
    
    # First half outputs should be identical (causality)
    first_half_diff = (output_A[:, :kv_len//2] - output_B[:, :kv_len//2]).abs().max().item()
    
    # Second half outputs should differ (since inputs differ)
    second_half_diff = (output_A[:, kv_len//2:] - output_B[:, kv_len//2:]).abs().max().item()
    
    print(f"\n  Sequence length: {kv_len}")
    print(f"  First half differs at position {kv_len//2}")
    print(f"  First half output diff (should be ~0): {first_half_diff:.2e}")
    print(f"  Second half output diff (should be >0): {second_half_diff:.2e}")
    
    if first_half_diff < 1e-6 and second_half_diff > 1e-3:
        print(f"  ✓ PASS: Causality maintained")
        return True
    else:
        print(f"  ✗ FAIL: Causality violated")
        return False


def test_padding_mask():
    """
    T3c: Verify padding mask is respected.
    
    Padded tokens should not contribute to attention outputs.
    """
    print("\n" + "="*60)
    print("TEST 3c: Padding Mask Verification")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention
    
    S = 16
    head_dim = 64
    seq_len = 32
    
    attn = IWMContentKernelAttention(
        hidden_size=head_dim,
        num_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        bandwidth=S,
        layer_idx=0,
    ).to(DEVICE)
    
    # Position embeddings
    cos = torch.ones(1, seq_len, head_dim, device=DEVICE)
    sin = torch.zeros(1, seq_len, head_dim, device=DEVICE)
    position_embeddings = (cos, sin)
    
    # Input with padding at the end
    hidden = torch.randn(1, seq_len, head_dim, device=DEVICE)
    real_len = 20  # Only first 20 tokens are real
    
    # Attention mask: 1 for real tokens, 0 for padding
    attention_mask = torch.zeros(1, seq_len, device=DEVICE)
    attention_mask[:, :real_len] = 1.0
    
    # Forward with mask
    output_masked, _ = attn.forward_with_cache(
        hidden, position_embeddings, cache=None, attention_mask=attention_mask
    )
    
    # Now change the padded tokens and verify output doesn't change
    hidden_modified = hidden.clone()
    hidden_modified[:, real_len:, :] = torch.randn(1, seq_len - real_len, head_dim, device=DEVICE) * 100
    
    output_modified, _ = attn.forward_with_cache(
        hidden_modified, position_embeddings, cache=None, attention_mask=attention_mask
    )
    
    # Outputs for real tokens should be identical
    diff = (output_masked[:, :real_len] - output_modified[:, :real_len]).abs().max().item()
    
    print(f"\n  Sequence length: {seq_len}, real length: {real_len}")
    print(f"  Max diff in real token outputs: {diff:.2e}")
    
    if diff < 1e-6:
        print(f"  ✓ PASS: Padding mask respected")
        return True
    else:
        print(f"  ✗ FAIL: Padded tokens affected outputs")
        return False


def test_constant_memory():
    """
    T4: Verify cache shapes remain constant during long generation.
    
    Run 10k token loop updating moments, ensure V_moments and Z_moments
    shapes never change (O(S) memory, not O(N)).
    """
    print("\n" + "="*60)
    print("TEST 4: Constant Memory During Generation")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention, create_iwm_cache
    
    S = 32
    head_dim = 64
    num_tokens = 10000
    num_kv_heads = 4
    num_heads = 4
    
    attn = IWMContentKernelAttention(
        hidden_size=head_dim * num_heads,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bandwidth=S,
        layer_idx=0,
    ).to(DEVICE)
    
    cache = create_iwm_cache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bandwidth=S,
        device=DEVICE,
    )
    
    # Record initial shapes
    initial_v_shape = cache.v_moments.shape
    initial_z_shape = cache.norm_moments.shape
    
    print(f"\n  Initial V_moments shape: {initial_v_shape}")
    print(f"  Initial Z_moments shape: {initial_z_shape}")
    print(f"  Expected: O(S) = O({S}), independent of sequence length")
    
    cos = torch.ones(1, 1, head_dim, device=DEVICE)
    sin = torch.zeros(1, 1, head_dim, device=DEVICE)
    position_embeddings = (cos, sin)
    
    # Simulate autoregressive generation (batch=1, seq_len=1)
    print(f"\n  Simulating {num_tokens} token generation...")
    
    shapes_constant = True
    for t in range(num_tokens):
        # Single token input (batch=1, seq_len=1)
        hidden_states = torch.randn(1, 1, head_dim * num_heads, device=DEVICE)
        cache_position = torch.tensor([t], device=DEVICE)
        
        # Forward pass updates cache (generation mode: seq_len=1 with cache)
        _, _ = attn.forward_with_cache(
            hidden_states, position_embeddings, cache, cache_position
        )
        
        # Check shapes haven't changed
        if cache.v_moments.shape != initial_v_shape:
            print(f"  ✗ V_moments shape changed at t={t}: {cache.v_moments.shape}")
            shapes_constant = False
            break
        if cache.norm_moments.shape != initial_z_shape:
            print(f"  ✗ Z_moments shape changed at t={t}: {cache.norm_moments.shape}")
            shapes_constant = False
            break
        
        # Progress update
        if (t + 1) % 2500 == 0:
            print(f"    t={t+1}: cache.seen_tokens={cache.seen_tokens}, shapes OK")
    
    if shapes_constant:
        print(f"\n  Final seen_tokens: {cache.seen_tokens}")
        print(f"  V_moments shape: {cache.v_moments.shape}")
        print(f"  Z_moments shape: {cache.norm_moments.shape}")
        print(f"  ✓ PASS: Cache shapes constant across {num_tokens} tokens")
        return True
    else:
        print(f"  ✗ FAIL: Cache shapes changed during generation")
        return False


def test_coefficient_prediction():
    """
    Bonus: Verify coefficient prediction from queries works correctly.
    """
    print("\n" + "="*60)
    print("TEST 5: Coefficient Prediction From Queries")
    print("="*60)
    
    from iwm_attention import IWMContentKernelAttention
    
    S = 16
    head_dim = 64
    batch = 2
    seq = 8
    heads = 4
    
    attn = IWMContentKernelAttention(
        hidden_size=head_dim * heads,
        num_heads=heads,
        head_dim=head_dim,
        bandwidth=S,
    ).to(DEVICE)
    
    # Random query
    query = torch.randn(batch, heads, seq, head_dim, device=DEVICE)
    
    # Predict coefficients
    coeffs = attn._predict_coefficients(query)
    
    print(f"\n  Query shape: {query.shape}")
    print(f"  Coefficient shape: {coeffs.shape}")
    print(f"  Expected: [batch={batch}, heads={heads}, seq={seq}, S+1={S+1}]")
    
    expected_shape = (batch, heads, seq, S + 1)
    if coeffs.shape == expected_shape:
        print(f"  ✓ PASS: Coefficient prediction shape correct")
        
        # Check initial coefficients (should be ~uniform: c_0=1, rest≈0)
        c0_mean = coeffs[..., 0].mean().item()
        c_rest_mean = coeffs[..., 1:].abs().mean().item()
        print(f"  c_0 mean: {c0_mean:.4f} (expected ~1.0)")
        print(f"  c_1..S abs mean: {c_rest_mean:.4f} (expected ~0)")
        
        return True
    else:
        print(f"  ✗ FAIL: Shape mismatch {coeffs.shape} vs {expected_shape}")
        return False


def run_all_tests():
    """Run all acceptance tests."""
    print("\n" + "="*70)
    print("IWM CONTENT-KERNEL ATTENTION OPERATOR TESTS")
    print("="*70)
    
    results = {}
    
    # T1: Bandlimit
    try:
        results['T1_bandlimit'] = test_kernel_bandlimit()
    except Exception as e:
        print(f"T1 EXCEPTION: {e}")
        results['T1_bandlimit'] = False
    
    # T2: Nonnegativity
    try:
        results['T2_nonnegativity'] = test_kernel_nonnegativity()
    except Exception as e:
        print(f"T2 EXCEPTION: {e}")
        results['T2_nonnegativity'] = False
    
    # T3: Moment vs explicit
    try:
        results['T3_moment_explicit'] = test_moment_vs_explicit()
    except Exception as e:
        print(f"T3 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results['T3_moment_explicit'] = False
    
    # T3b: Causality
    try:
        results['T3b_causality'] = test_causality()
    except Exception as e:
        print(f"T3b EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results['T3b_causality'] = False
    
    # T3c: Padding mask
    try:
        results['T3c_padding_mask'] = test_padding_mask()
    except Exception as e:
        print(f"T3c EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results['T3c_padding_mask'] = False
    
    # T4: Constant memory
    try:
        results['T4_constant_memory'] = test_constant_memory()
    except Exception as e:
        print(f"T4 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results['T4_constant_memory'] = False
    
    # T5: Coefficient prediction
    try:
        results['T5_coeff_prediction'] = test_coefficient_prediction()
    except Exception as e:
        print(f"T5 EXCEPTION: {e}")
        results['T5_coeff_prediction'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
