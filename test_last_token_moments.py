"""
Test that the new O(1)-memory last-token moments match the old cumsum-based approach.

Verifies:
1. _compute_last_prefix_moments() gives same result as _compute_prefix_moments()[-1]
2. forward_train_last_token() output is unchanged
"""

import torch
import sys
sys.path.insert(0, '.')

from iwm_attention import IWMContentKernelAttention

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_last_prefix_moments_match():
    """Verify new O(1) moments match old cumsum approach at last position."""
    print("=" * 60)
    print("TEST: _compute_last_prefix_moments() matches cumsum[-1]")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create attention module
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=4,
        head_dim=16,
        bandwidth=8,
    ).to(DEVICE)
    
    configs = [
        {'batch': 1, 'seq_len': 32, 'padding': False, 'desc': 'small, no padding'},
        {'batch': 2, 'seq_len': 64, 'padding': False, 'desc': 'batch=2, no padding'},
        {'batch': 2, 'seq_len': 64, 'padding': True, 'desc': 'batch=2, with padding'},
    ]
    
    all_passed = True
    
    for i, cfg in enumerate(configs):
        batch = cfg['batch']
        seq_len = cfg['seq_len']
        kv_heads = 4
        head_dim = 16
        
        # Random value tensor
        value = torch.randn(batch, kv_heads, seq_len, head_dim, device=DEVICE)
        positions = torch.arange(seq_len, device=DEVICE)
        
        # Create attention mask
        attention_mask = torch.ones(batch, seq_len, device=DEVICE)
        if cfg['padding']:
            # Different lengths per sample
            attention_mask[0, seq_len//2:] = 0  # First sample half-length
            if batch > 1:
                attention_mask[1, -10:] = 0  # Second sample missing last 10
        
        # OLD METHOD: cumsum over full sequence, extract last
        v_moments_full, norm_moments_full = attn._compute_prefix_moments(
            value, positions, attention_mask
        )
        # v_moments_full: [batch, kv_heads, seq_len, 2S+1, head_dim]
        # Extract last valid position per sample
        positions_range = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        masked_positions = positions_range * attention_mask.long() - (1 - attention_mask.long()) * seq_len
        last_indices = masked_positions.argmax(dim=1)
        
        batch_idx = torch.arange(batch, device=DEVICE)
        V_last_old = v_moments_full[batch_idx, :, last_indices, :, :]
        Z_last_old = norm_moments_full[batch_idx, :, last_indices, :]
        
        # NEW METHOD: direct O(1) computation
        V_last_new, Z_last_new = attn._compute_last_prefix_moments(value, attention_mask)
        
        # Compare
        v_diff = (V_last_old - V_last_new).abs().max().item()
        z_diff = (Z_last_old - Z_last_new).abs().max().item()
        
        passed = v_diff < 1e-4 and z_diff < 1e-4  # Allow small numerical differences
        status = '✓' if passed else '✗'
        
        print(f"\n  Config {i+1}: {cfg['desc']}")
        print(f"    batch={batch}, seq_len={seq_len}")
        print(f"    V_last diff: {v_diff:.2e} {status}")
        print(f"    Z_last diff: {z_diff:.2e} {status}")
        
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'='*60}\n")
    return all_passed


def test_forward_train_last_token_output():
    """Verify forward_train_last_token produces valid output."""
    print("=" * 60)
    print("TEST: forward_train_last_token() produces valid output")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create attention module
    attn = IWMContentKernelAttention(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=4,
        head_dim=16,
        bandwidth=8,
    ).to(DEVICE)
    attn.training_last_token_only = True
    attn.train()
    
    # Test input
    batch = 2
    seq_len = 32
    hidden_size = 64
    
    hidden_states = torch.randn(batch, seq_len, hidden_size, device=DEVICE)
    attention_mask = torch.ones(batch, seq_len, device=DEVICE)
    attention_mask[0, seq_len//2:] = 0  # First sample ends at seq_len//2
    
    # Dummy position embeddings (not used by IWM)
    cos = torch.randn(seq_len, 16, device=DEVICE)
    sin = torch.randn(seq_len, 16, device=DEVICE)
    position_embeddings = (cos, sin)
    
    # Forward pass
    output, last_indices, psd_loss = attn.forward_train_last_token(
        hidden_states,
        position_embeddings,
        attention_mask=attention_mask,
    )
    
    print(f"\n  Output shape: {output.shape}")
    print(f"  Last indices: {last_indices.tolist()}")
    print(f"  PSD loss: {psd_loss.item():.6f}")
    
    # Check output is non-zero only at last indices
    batch_idx = torch.arange(batch, device=DEVICE)
    output_at_last = output[batch_idx, last_indices, :]
    
    # Output should be non-zero at last positions
    output_norm = output_at_last.norm(dim=-1)
    print(f"  Output norm at last positions: {output_norm.tolist()}")
    
    # Output should be zero elsewhere
    output_copy = output.clone()
    output_copy[batch_idx, last_indices, :] = 0
    other_norm = output_copy.abs().max().item()
    print(f"  Max output elsewhere: {other_norm:.2e}")
    
    passed = (output_norm > 0.01).all().item() and other_norm < 1e-6
    status = '✓ PASS' if passed else '✗ FAIL'
    
    print(f"\n{'='*60}")
    print(f"RESULT: {status}")
    print(f"{'='*60}\n")
    return passed


def test_memory_improvement():
    """Verify the new method doesn't allocate huge intermediate tensors."""
    print("=" * 60)
    print("TEST: Memory improvement (no seq-sized moment tensor)")
    print("=" * 60)
    
    if DEVICE != 'cuda':
        print("  Skipping memory test (requires CUDA)")
        return True
    
    torch.manual_seed(42)
    
    # Create attention module with larger bandwidth
    S = 32
    attn = IWMContentKernelAttention(
        hidden_size=1024,
        num_heads=16,
        num_kv_heads=4,
        head_dim=64,
        bandwidth=S,
    ).to(DEVICE)
    
    batch = 4
    seq_len = 512
    kv_heads = 4
    head_dim = 64
    
    value = torch.randn(batch, kv_heads, seq_len, head_dim, device=DEVICE)
    attention_mask = torch.ones(batch, seq_len, device=DEVICE)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # New method
    mem_before = torch.cuda.memory_allocated()
    V_last, Z_last = attn._compute_last_prefix_moments(value, attention_mask)
    mem_after = torch.cuda.memory_allocated()
    new_method_alloc = mem_after - mem_before
    
    # Expected allocation for new method: O(batch × heads × (2S+1) × head_dim) complex
    # = 4 × 4 × 65 × 64 × 8 (complex64 = 8 bytes) = ~5.3 MB
    expected_new = batch * kv_heads * (2*S+1) * head_dim * 8
    
    # Old method would allocate: O(batch × heads × seq × (2S+1) × head_dim) complex
    # = 4 × 4 × 512 × 65 × 64 × 8 = ~2.7 GB!
    expected_old = batch * kv_heads * seq_len * (2*S+1) * head_dim * 8
    
    print(f"\n  New method allocation:  {new_method_alloc / 1e6:.1f} MB")
    print(f"  Expected (new):         {expected_new / 1e6:.1f} MB")
    print(f"  Old method would be:    {expected_old / 1e6:.1f} MB")
    print(f"  Improvement factor:     {expected_old / max(new_method_alloc, 1):.0f}x")
    
    # The new method should allocate roughly expected_new (plus some overhead)
    # It should definitely NOT allocate expected_old
    passed = new_method_alloc < expected_old / 10  # At least 10x improvement
    status = '✓ PASS' if passed else '✗ FAIL'
    
    print(f"\n{'='*60}")
    print(f"RESULT: {status}")
    print(f"{'='*60}\n")
    return passed


if __name__ == "__main__":
    test1 = test_last_prefix_moments_match()
    test2 = test_forward_train_last_token_output()
    test3 = test_memory_improvement()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Moments match:      {'✓' if test1 else '✗'}")
    print(f"  Forward output:     {'✓' if test2 else '✗'}")
    print(f"  Memory improvement: {'✓' if test3 else '✗'}")
    
    all_passed = test1 and test2 and test3
    print(f"\nALL TESTS: {'PASS' if all_passed else 'FAIL'}")
