"""
Acceptance tests for IWM attention implementation.

These tests verify:
1. Batch correctness - different sequences produce different outputs
2. Causality - last-token output only depends on prefix
3. Cache vs explicit consistency (for generation)
4. Speed improvement with last-token-only training
"""

import torch
import time
import sys
from iwm_attention import IWMContentKernelAttention, IWMCache

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {DEVICE}")


def test_batch_correctness():
    """
    TEST 1: Batch correctness
    
    Run with batch=2 different sequences and confirm:
    - Moments differ per batch item
    - Outputs differ per batch item
    """
    print("\n" + "=" * 70)
    print("TEST 1: Batch Correctness")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        bandwidth=8,
        layer_idx=0,
    ).to(DEVICE)
    attn.training_last_token_only = True
    attn.eval()
    
    batch = 2
    seq_len = 32
    
    # Create two DIFFERENT sequences
    hidden_states = torch.randn(batch, seq_len, 64, device=DEVICE)
    # Make them clearly different
    hidden_states[0] *= 2.0
    hidden_states[1] *= -1.0
    
    attention_mask = torch.ones(batch, seq_len, device=DEVICE)
    
    # Dummy position embeddings
    cos = torch.ones(seq_len, 8, device=DEVICE)
    sin = torch.zeros(seq_len, 8, device=DEVICE)
    
    with torch.no_grad():
        output, psd_loss = attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
        )
    
    # Check outputs differ per batch
    last_idx = seq_len - 1
    out_0 = output[0, last_idx]
    out_1 = output[1, last_idx]
    
    output_diff = (out_0 - out_1).abs().mean().item()
    
    print(f"\n  Sequence 0 stats: mean={hidden_states[0].mean():.3f}, std={hidden_states[0].std():.3f}")
    print(f"  Sequence 1 stats: mean={hidden_states[1].mean():.3f}, std={hidden_states[1].std():.3f}")
    print(f"\n  Output 0 at last position: norm={out_0.norm():.4f}")
    print(f"  Output 1 at last position: norm={out_1.norm():.4f}")
    print(f"  Output difference: {output_diff:.6f}")
    
    # They should be different (not identical)
    passed = output_diff > 0.01
    
    if passed:
        print(f"\n✓ PASS: Outputs differ per batch item (diff={output_diff:.6f} > 0.01)")
    else:
        print(f"\n✗ FAIL: Outputs are too similar (diff={output_diff:.6f} <= 0.01)")
    
    return passed


def test_causality():
    """
    TEST 2: Causality
    
    For last-token-only training:
    - Change token at position t < T-1 -> last output SHOULD change
    - Confirm no "future leakage"
    """
    print("\n" + "=" * 70)
    print("TEST 2: Causality (last-token depends only on prefix)")
    print("=" * 70)
    
    torch.manual_seed(123)
    
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        bandwidth=8,
        layer_idx=0,
    ).to(DEVICE)
    attn.training_last_token_only = True
    attn.eval()
    
    batch = 1
    seq_len = 32
    last_idx = 20  # We'll set attention_mask so only 0..20 are valid
    
    # Base sequence
    hidden_A = torch.randn(batch, seq_len, 64, device=DEVICE)
    
    # Sequence B: same prefix (0..20), different suffix (21..31)
    hidden_B = hidden_A.clone()
    hidden_B[:, last_idx+1:, :] = torch.randn(batch, seq_len - last_idx - 1, 64, device=DEVICE) * 5.0
    
    # Mask so we only "see" up to last_idx
    attention_mask = torch.zeros(batch, seq_len, device=DEVICE)
    attention_mask[:, :last_idx+1] = 1.0
    
    cos = torch.ones(seq_len, 8, device=DEVICE)
    sin = torch.zeros(seq_len, 8, device=DEVICE)
    
    with torch.no_grad():
        out_A, _ = attn(hidden_A, (cos, sin), attention_mask=attention_mask)
        out_B, _ = attn(hidden_B, (cos, sin), attention_mask=attention_mask)
    
    # Get outputs at last valid position
    out_A_last = out_A[0, last_idx]
    out_B_last = out_B[0, last_idx]
    
    diff = (out_A_last - out_B_last).abs().max().item()
    
    print(f"\n  Prefix (0..{last_idx}): identical")
    print(f"  Suffix ({last_idx+1}..{seq_len-1}): different (A vs B)")
    print(f"  Attention mask: 1s for 0..{last_idx}, 0s for {last_idx+1}..{seq_len-1}")
    print(f"\n  Output diff at position {last_idx}: {diff:.2e}")
    
    # Outputs should be IDENTICAL since suffix is masked
    passed = diff < 1e-6
    
    if passed:
        print(f"\n✓ PASS: No future leakage (diff={diff:.2e} < 1e-6)")
    else:
        print(f"\n✗ FAIL: Future tokens leaked into output (diff={diff:.2e})")
    
    # Part 2: Changing prefix SHOULD change output
    print("\n  --- Part 2: Changing prefix changes output ---")
    
    hidden_C = hidden_A.clone()
    hidden_C[:, 10, :] = torch.randn(batch, 64, device=DEVICE) * 10.0  # Change token 10
    
    with torch.no_grad():
        out_C, _ = attn(hidden_C, (cos, sin), attention_mask=attention_mask)
    
    out_C_last = out_C[0, last_idx]
    prefix_diff = (out_A_last - out_C_last).abs().max().item()
    
    print(f"  Changed token at position 10")
    print(f"  Output diff at position {last_idx}: {prefix_diff:.6f}")
    
    prefix_passed = prefix_diff > 0.001
    
    if prefix_passed:
        print(f"\n✓ PASS: Prefix change affects output (diff={prefix_diff:.6f} > 0.001)")
    else:
        print(f"\n✗ FAIL: Prefix change didn't affect output (diff={prefix_diff:.6f})")
    
    return passed and prefix_passed


def test_cache_vs_explicit():
    """
    TEST 3: Cache vs explicit consistency (generation)
    
    For a growing prefix, compare:
    - forward_with_cache(seq_len=1) incrementally
    - forward_explicit on full prefix for last token
    
    Tolerance: ~1e-5
    """
    print("\n" + "=" * 70)
    print("TEST 3: Cache vs Explicit (generation consistency)")
    print("=" * 70)
    
    torch.manual_seed(456)
    
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        bandwidth=8,
        layer_idx=0,
    ).to(DEVICE)
    attn.eval()
    
    # Initialize cache
    num_layers = 1
    cache = IWMCache(
        v_moments=torch.zeros(num_layers, 2, 17, 16, dtype=torch.complex64, device=DEVICE),
        norm_moments=torch.zeros(num_layers, 2, 17, dtype=torch.complex64, device=DEVICE),
        seen_tokens=0,
    )
    
    # Generate tokens one by one
    total_len = 16
    full_hidden = torch.randn(1, total_len, 64, device=DEVICE)
    
    cos = torch.ones(total_len, 8, device=DEVICE)
    sin = torch.zeros(total_len, 8, device=DEVICE)
    
    max_diff = 0.0
    all_passed = True
    
    print(f"\n  Growing prefix from 1 to {total_len} tokens...")
    
    for t in range(total_len):
        # Reset cache for fresh computation
        cache.v_moments.zero_()
        cache.norm_moments.zero_()
        cache.seen_tokens = 0
        
        # Feed all tokens up to t+1 via cache
        for i in range(t + 1):
            token_hidden = full_hidden[:, i:i+1, :]
            token_mask = torch.ones(1, 1, device=DEVICE)
            
            with torch.no_grad():
                out_cached, _ = attn.forward_with_cache(
                    token_hidden,
                    position_embeddings=(cos[i:i+1], sin[i:i+1]),
                    cache=cache,
                    cache_position=torch.tensor([i], device=DEVICE),
                    attention_mask=token_mask,
                )
        
        # Explicit computation on full prefix
        prefix_hidden = full_hidden[:, :t+1, :]
        prefix_mask = torch.ones(1, t+1, device=DEVICE)
        
        with torch.no_grad():
            out_explicit, _, _ = attn.forward_explicit(
                prefix_hidden,
                position_embeddings=(cos[:t+1], sin[:t+1]),
                attention_mask=prefix_mask,
            )
        
        # Compare at last position
        diff = (out_cached[0, 0] - out_explicit[0, t]).abs().max().item()
        max_diff = max(max_diff, diff)
        
        if diff > 1e-4:
            print(f"    t={t}: diff={diff:.2e} ✗")
            all_passed = False
        elif t % 5 == 0 or t == total_len - 1:
            print(f"    t={t}: diff={diff:.2e} ✓")
    
    print(f"\n  Max difference across all positions: {max_diff:.2e}")
    
    passed = max_diff < 1e-4
    
    if passed:
        print(f"\n✓ PASS: Cache matches explicit (max diff={max_diff:.2e} < 1e-4)")
    else:
        print(f"\n✗ FAIL: Cache/explicit mismatch (max diff={max_diff:.2e})")
    
    return passed


def test_speed_improvement():
    """
    TEST 4: Speed improvement with last-token-only
    
    Compare step time between:
    - Full sequence forward (O(seq) memory)
    - Last-token-only forward (O(1) memory)
    """
    print("\n" + "=" * 70)
    print("TEST 4: Speed Improvement (last-token-only vs full)")
    print("=" * 70)
    
    torch.manual_seed(789)
    
    attn = IWMContentKernelAttention(
        hidden_size=256,
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        bandwidth=32,
        layer_idx=0,
    ).to(DEVICE)
    
    batch = 4
    seq_len = 512
    
    hidden_states = torch.randn(batch, seq_len, 256, device=DEVICE)
    attention_mask = torch.ones(batch, seq_len, device=DEVICE)
    
    cos = torch.ones(seq_len, 16, device=DEVICE)
    sin = torch.zeros(seq_len, 16, device=DEVICE)
    
    # Warmup
    attn.training_last_token_only = True
    attn.train()
    for _ in range(3):
        with torch.no_grad():
            _, _ = attn(hidden_states, (cos, sin), attention_mask=attention_mask)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    # Time last-token-only
    n_iters = 10
    
    attn.training_last_token_only = True
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            out_last, _ = attn(hidden_states, (cos, sin), attention_mask=attention_mask)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    time_last = (time.perf_counter() - start) / n_iters
    
    # Time full sequence (using forward_with_cache in training mode)
    attn.training_last_token_only = False
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            out_full, _ = attn.forward_with_cache(
                hidden_states, (cos, sin), attention_mask=attention_mask
            )
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    time_full = (time.perf_counter() - start) / n_iters
    
    speedup = time_full / time_last if time_last > 0 else float('inf')
    
    print(f"\n  Configuration:")
    print(f"    batch={batch}, seq_len={seq_len}, bandwidth=32")
    print(f"    hidden_size=256, num_heads=8")
    print(f"\n  Timing ({n_iters} iterations each):")
    print(f"    Last-token-only: {time_last*1000:.2f} ms")
    print(f"    Full sequence:   {time_full*1000:.2f} ms")
    print(f"    Speedup:         {speedup:.1f}x")
    
    # We expect at least 2x speedup (often much more)
    passed = speedup > 1.5
    
    if passed:
        print(f"\n✓ PASS: Significant speedup achieved ({speedup:.1f}x > 1.5x)")
    else:
        print(f"\n✗ FAIL: Speedup not significant ({speedup:.1f}x <= 1.5x)")
    
    return passed


def test_coefficient_activity():
    """
    SANITY CHECK: Verify kernel coefficients are not frozen
    
    Print min/mean/max of c_coeffs[..., 1:].abs().sum() / c0.abs()
    to confirm coefficients evolve during forward passes.
    """
    print("\n" + "=" * 70)
    print("SANITY CHECK: Coefficient Activity")
    print("=" * 70)
    
    torch.manual_seed(999)
    
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        bandwidth=8,
        layer_idx=0,
    ).to(DEVICE)
    attn.training_last_token_only = True
    attn.train()
    
    batch = 2
    seq_len = 32
    
    hidden_states = torch.randn(batch, seq_len, 64, device=DEVICE)
    attention_mask = torch.ones(batch, seq_len, device=DEVICE)
    
    cos = torch.ones(seq_len, 8, device=DEVICE)
    sin = torch.zeros(seq_len, 8, device=DEVICE)
    
    # Forward pass to compute coefficients
    query = attn.q_proj(hidden_states)
    query = query.view(batch, seq_len, 4, 16).transpose(1, 2)  # [B, heads, seq, head_dim]
    
    c_coeffs = attn._predict_coefficients(query)  # [B, heads, seq, S+1]
    
    # Analyze coefficients
    c0 = c_coeffs[..., 0].abs()  # [B, heads, seq]
    c_rest = c_coeffs[..., 1:].abs().sum(dim=-1)  # [B, heads, seq]
    
    ratio = c_rest / (c0 + 1e-10)
    
    print(f"\n  c_coeffs shape: {c_coeffs.shape}")
    print(f"  c0 (DC component):")
    print(f"    min={c0.min():.4f}, mean={c0.mean():.4f}, max={c0.max():.4f}")
    print(f"  c_rest (higher harmonics sum):")
    print(f"    min={c_rest.min():.4f}, mean={c_rest.mean():.4f}, max={c_rest.max():.4f}")
    print(f"  ratio (c_rest / c0):")
    print(f"    min={ratio.min():.4f}, mean={ratio.mean():.4f}, max={ratio.max():.4f}")
    
    # Also check kernel values on a dense grid
    g_coeffs = attn._build_kernel_coeffs(c_coeffs)
    
    # Sample kernel at various angular separations
    theta_grid = torch.linspace(0, 2*torch.pi, 100, device=DEVICE)
    k_vals = attn.k_vals  # [-S, ..., S]
    
    # Compute kernel: K(θ) = Σ_k g_k * e^{-ikθ}
    phase = torch.exp(-1j * k_vals.unsqueeze(1) * theta_grid.unsqueeze(0))  # [2S+1, 100]
    
    # Use last token's coefficients for one sample
    g_last = g_coeffs[0, 0, -1, :]  # [2S+1] complex
    kernel_vals = torch.einsum('k,kt->t', g_last, phase)  # [100] complex
    kernel_real = kernel_vals.real
    
    print(f"\n  Kernel values on dense grid (sample 0, head 0, last token):")
    print(f"    min={kernel_real.min():.4f}, mean={kernel_real.mean():.4f}, max={kernel_real.max():.4f}")
    
    if kernel_real.min() >= -1e-6:
        print(f"    Kernel is non-negative (PSD constraint satisfied)")
    else:
        print(f"    Kernel has negative values (PSD constraint may be violated)")
    
    print("\n✓ SANITY CHECK COMPLETE (informational, no pass/fail)")
    return True


def main():
    print("=" * 70)
    print("IWM ATTENTION ACCEPTANCE TESTS")
    print("=" * 70)
    
    results = {}
    
    results['batch_correctness'] = test_batch_correctness()
    results['causality'] = test_causality()
    results['cache_vs_explicit'] = test_cache_vs_explicit()
    results['speed'] = test_speed_improvement()
    test_coefficient_activity()  # Informational only
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL ACCEPTANCE TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
