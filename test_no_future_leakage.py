"""
Test: No future leakage in last-token computation.

This test verifies that the output at last_idx is identical whether or not
there are additional tokens after last_idx.
"""

import torch
from iwm_attention import IWMContentKernelAttention

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_no_future_leakage():
    """
    Create two sequences:
    - Sequence A: tokens up to last_idx only (mask=1 up to last_idx)
    - Sequence B: same tokens up to last_idx, but additional different tokens after
                  (mask=1 up to last_idx, then mask=0 after)
    
    The output at last_idx should be IDENTICAL, proving no future leakage.
    """
    print("="*70)
    print("TEST: No Future Leakage in Last-Token Computation")
    print("="*70)
    
    batch_size = 2
    seq_len = 32
    last_idx = 20  # Position we care about
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    bandwidth = 8
    
    print(f"\nTest configuration:")
    print(f"  seq_len: {seq_len}")
    print(f"  last_idx: {last_idx}")
    print(f"  Positions 0-{last_idx}: identical content")
    print(f"  Positions {last_idx+1}-{seq_len-1}: different content")
    
    # Create module
    torch.manual_seed(42)
    attn = IWMContentKernelAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bandwidth=bandwidth,
    ).to(DEVICE)
    attn.eval()
    
    # Create common content for positions 0..last_idx
    torch.manual_seed(123)
    common_content = torch.randn(batch_size, last_idx + 1, hidden_size, device=DEVICE)
    
    # Sequence A: common content, then zeros (padding)
    hidden_A = torch.zeros(batch_size, seq_len, hidden_size, device=DEVICE)
    hidden_A[:, :last_idx + 1, :] = common_content
    mask_A = torch.zeros(batch_size, seq_len, device=DEVICE)
    mask_A[:, :last_idx + 1] = 1.0  # Valid up to last_idx
    
    # Sequence B: common content, then DIFFERENT content (but masked)
    torch.manual_seed(456)  # Different seed for different content
    different_content = torch.randn(batch_size, seq_len - last_idx - 1, hidden_size, device=DEVICE)
    hidden_B = torch.zeros(batch_size, seq_len, hidden_size, device=DEVICE)
    hidden_B[:, :last_idx + 1, :] = common_content
    hidden_B[:, last_idx + 1:, :] = different_content  # Different content after last_idx
    mask_B = torch.zeros(batch_size, seq_len, device=DEVICE)
    mask_B[:, :last_idx + 1] = 1.0  # Valid only up to last_idx (same as A)
    
    print(f"\nSequence A: {hidden_A.shape}, valid positions: 0-{last_idx}")
    print(f"Sequence B: {hidden_B.shape}, valid positions: 0-{last_idx} (different content after)")
    
    position_embeddings = (None, None)
    
    # Compute output for A
    print("\nComputing output for Sequence A...")
    with torch.no_grad():
        output_A, last_indices_A, _ = attn.forward_train_last_token(
            hidden_A,
            position_embeddings,
            attention_mask=mask_A,
        )
    
    # Compute output for B
    print("Computing output for Sequence B...")
    with torch.no_grad():
        output_B, last_indices_B, _ = attn.forward_train_last_token(
            hidden_B,
            position_embeddings,
            attention_mask=mask_B,
        )
    
    # Verify last indices are the same
    assert torch.all(last_indices_A == last_idx), f"A last indices: {last_indices_A}"
    assert torch.all(last_indices_B == last_idx), f"B last indices: {last_indices_B}"
    print(f"  Last indices A: {last_indices_A.tolist()}")
    print(f"  Last indices B: {last_indices_B.tolist()}")
    
    # Extract output at last_idx
    batch_idx = torch.arange(batch_size, device=DEVICE)
    output_A_last = output_A[batch_idx, last_idx, :]
    output_B_last = output_B[batch_idx, last_idx, :]
    
    # Compare
    diff = (output_A_last - output_B_last).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nComparison at last_idx={last_idx}:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    # Verify that the hidden content after last_idx is actually different
    content_diff = (hidden_A[:, last_idx+1:, :] - hidden_B[:, last_idx+1:, :]).abs().max().item()
    print(f"  Content difference after last_idx: {content_diff:.2e} (should be > 0)")
    
    # Pass/fail
    threshold = 1e-10  # Should be essentially zero
    if max_diff <= threshold:
        print(f"\n✓ PASS: No future leakage (diff = {max_diff:.2e} <= {threshold})")
        return True
    else:
        print(f"\n✗ FAIL: Future leakage detected (diff = {max_diff:.2e} > {threshold})")
        return False


def test_no_leakage_with_padding_variations():
    """
    Test with multiple batch elements having different last_idx.
    """
    print("\n" + "="*70)
    print("TEST: No Leakage With Varying Last Indices")
    print("="*70)
    
    batch_size = 4
    seq_len = 40
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    bandwidth = 8
    
    # Different last indices for each sample
    last_indices_target = [15, 25, 10, 30]
    
    print(f"\nTarget last indices: {last_indices_target}")
    
    # Create module
    torch.manual_seed(42)
    attn = IWMContentKernelAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bandwidth=bandwidth,
    ).to(DEVICE)
    attn.eval()
    
    # Create common content
    torch.manual_seed(123)
    all_content = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)
    
    # Sequence A: content with mask up to last_idx
    hidden_A = all_content.clone()
    mask_A = torch.zeros(batch_size, seq_len, device=DEVICE)
    for b, last_idx in enumerate(last_indices_target):
        mask_A[b, :last_idx + 1] = 1.0
    
    # Sequence B: DIFFERENT content after last_idx (but masked)
    torch.manual_seed(789)
    noise = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE) * 10  # Large different values
    hidden_B = all_content.clone()
    mask_B = mask_A.clone()  # Same mask
    for b, last_idx in enumerate(last_indices_target):
        hidden_B[b, last_idx + 1:, :] = noise[b, last_idx + 1:, :]  # Replace with noise
    
    position_embeddings = (None, None)
    
    # Compute outputs
    print("\nComputing outputs...")
    with torch.no_grad():
        output_A, last_idx_A, _ = attn.forward_train_last_token(
            hidden_A, position_embeddings, attention_mask=mask_A
        )
        output_B, last_idx_B, _ = attn.forward_train_last_token(
            hidden_B, position_embeddings, attention_mask=mask_B
        )
    
    # Verify last indices
    expected_last_indices = torch.tensor(last_indices_target, device=DEVICE)
    assert torch.all(last_idx_A == expected_last_indices), f"Mismatch: {last_idx_A} vs {expected_last_indices}"
    assert torch.all(last_idx_B == expected_last_indices), f"Mismatch: {last_idx_B} vs {expected_last_indices}"
    print(f"  Last indices: {last_idx_A.tolist()}")
    
    # Compare at each sample's last position
    batch_idx = torch.arange(batch_size, device=DEVICE)
    output_A_last = output_A[batch_idx, last_idx_A, :]
    output_B_last = output_B[batch_idx, last_idx_B, :]
    
    diff = (output_A_last - output_B_last).abs()
    max_diff = diff.max().item()
    per_sample_max = diff.max(dim=1).values
    
    print(f"\nComparison:")
    print(f"  Per-sample max diff: {[f'{d:.2e}' for d in per_sample_max.tolist()]}")
    print(f"  Overall max diff: {max_diff:.2e}")
    
    threshold = 1e-10
    if max_diff <= threshold:
        print(f"\n✓ PASS: No future leakage across all samples")
        return True
    else:
        print(f"\n✗ FAIL: Future leakage detected")
        return False


if __name__ == "__main__":
    results = {}
    
    results['basic'] = test_no_future_leakage()
    results['varying_lengths'] = test_no_leakage_with_padding_variations()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
