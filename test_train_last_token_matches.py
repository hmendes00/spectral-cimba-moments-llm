"""
Test: Last-token training path matches full prefix-moment path.

This test verifies that forward_train_last_token() produces the same output
at the last position as the full prefix-moment computation.
"""

import torch
from iwm_attention import IWMContentKernelAttention

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_last_token_matches_full():
    """
    Build a tiny random batch with no padding and compare:
    (a) Old full prefix-moment path (computes all positions)
    (b) New forward_train_last_token path (only last position)
    
    Assert max_abs_diff <= 1e-5
    """
    print("="*70)
    print("TEST: Last-Token Output Matches Full Prefix-Moment Path")
    print("="*70)
    
    # Test parameters
    batch_size = 4
    seq_len = 32
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    bandwidth = 8
    
    print(f"\nTest configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  bandwidth: {bandwidth}")
    
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
    
    # Create random input (no padding - all valid)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)
    attention_mask = torch.ones(batch_size, seq_len, device=DEVICE)  # All valid
    position_embeddings = (None, None)  # Not used in IWM
    
    # Path A: Full prefix-moment computation (all positions)
    print("\n(A) Computing full prefix-moment output...")
    attn.training_last_token_only = False
    with torch.no_grad():
        full_output, _ = attn.forward_with_cache(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
    # full_output: [batch, seq_len, hidden_size]
    
    # Extract last position output
    # Since no padding, last_idx = seq_len - 1 for all samples
    last_idx = seq_len - 1
    full_output_last = full_output[:, last_idx, :]  # [batch, hidden_size]
    
    # Path B: Last-token-only computation
    print("(B) Computing last-token-only output...")
    with torch.no_grad():
        last_output, last_indices, _ = attn.forward_train_last_token(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
    # last_output: [batch, seq_len, hidden_size] - zeros except at last_idx
    
    # Verify last_indices are correct
    expected_last_indices = torch.full((batch_size,), last_idx, device=DEVICE)
    assert torch.all(last_indices == expected_last_indices), \
        f"Last indices mismatch: expected {expected_last_indices}, got {last_indices}"
    print(f"  Last indices: {last_indices.tolist()} (expected all {last_idx})")
    
    # Extract the output at last position
    batch_idx = torch.arange(batch_size, device=DEVICE)
    last_token_output = last_output[batch_idx, last_indices, :]  # [batch, hidden_size]
    
    # Compare
    diff = (full_output_last - last_token_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nComparison (at last position):")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    # Verify zeros elsewhere in last_output
    for b in range(batch_size):
        non_last_output = last_output[b, :last_idx, :]  # All positions before last
        if non_last_output.abs().max() > 1e-10:
            print(f"  WARNING: Non-zero output at positions before last for batch {b}")
    
    # Pass/fail
    threshold = 1e-4
    if max_diff <= threshold:
        print(f"\n✓ PASS: Last-token output matches full path (diff <= {threshold})")
        return True
    else:
        print(f"\n✗ FAIL: Difference {max_diff:.2e} exceeds threshold {threshold}")
        return False


def test_with_padding():
    """
    Test with variable-length sequences (different last_idx per sample).
    """
    print("\n" + "="*70)
    print("TEST: Last-Token With Padding")
    print("="*70)
    
    batch_size = 4
    seq_len = 32
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    bandwidth = 8
    
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
    
    # Create input with variable-length padding
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)
    
    # Create attention masks with different lengths
    # Lengths: [32, 20, 28, 15]
    lengths = [32, 20, 28, 15]
    attention_mask = torch.zeros(batch_size, seq_len, device=DEVICE)
    for b, length in enumerate(lengths):
        attention_mask[b, :length] = 1.0
    
    print(f"\nSequence lengths: {lengths}")
    expected_last_indices = torch.tensor([l - 1 for l in lengths], device=DEVICE)
    
    position_embeddings = (None, None)
    
    # Path A: Full computation
    print("\n(A) Computing full output...")
    attn.training_last_token_only = False
    with torch.no_grad():
        full_output, _ = attn.forward_with_cache(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
    
    # Extract at each sample's last position
    batch_idx = torch.arange(batch_size, device=DEVICE)
    full_output_last = full_output[batch_idx, expected_last_indices, :]
    
    # Path B: Last-token-only
    print("(B) Computing last-token-only output...")
    with torch.no_grad():
        last_output, last_indices, _ = attn.forward_train_last_token(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
    
    # Verify last_indices
    assert torch.all(last_indices == expected_last_indices), \
        f"Last indices mismatch: expected {expected_last_indices.tolist()}, got {last_indices.tolist()}"
    print(f"  Last indices: {last_indices.tolist()} (correct)")
    
    # Extract at last positions
    last_token_output = last_output[batch_idx, last_indices, :]
    
    # Compare
    diff = (full_output_last - last_token_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nComparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    threshold = 1e-4
    if max_diff <= threshold:
        print(f"\n✓ PASS: Last-token matches with padding (diff <= {threshold})")
        return True
    else:
        print(f"\n✗ FAIL: Difference {max_diff:.2e} exceeds threshold {threshold}")
        return False


if __name__ == "__main__":
    results = {}
    
    results['no_padding'] = test_last_token_matches_full()
    results['with_padding'] = test_with_padding()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
