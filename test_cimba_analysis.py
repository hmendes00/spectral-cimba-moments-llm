"""
Deep Analysis: Can CIMBA Sketch SpectralAttention?

This script investigates the mathematical requirements for CIMBA sketching
of the SpectralAttention operator.

Key insight from CIMBA theory:
- If attention weights W[i,j] = f(θ_i - θ_j) for some bandlimited f with bandwidth S,
  then output[i] = Σ_j W[i,j] * V[j] can be reconstructed from O(S) V-moments.
  
Question: Does SpectralAttention produce weights of this form?

SpectralAttention does:
1. scores = Q @ K^T / sqrt(d)  -- content-dependent
2. scores_filt = IFFT(FFT(scores) * lowpass)  -- bandlimited scores
3. attn = softmax(scores_filt)  -- normalized
4. out = attn @ V

The issue: W[i,j] depends on (Q[i], K[j]) content, not just (θ_i - θ_j) positions.
This means W is NOT of the form f(θ_i - θ_j), so standard CIMBA doesn't apply directly.

BUT: What if we sketch something else?
- Option A: Sketch the pre-softmax filtered scores and recompute softmax
- Option B: Sketch the V output directly using the actual attention weights
- Option C: Accept that content-based attention can't be O(S) sketched

Let's test these empirically.
"""

import torch
import torch.nn.functional as F
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32


def test_cimba_on_position_only():
    """
    Test: CIMBA works perfectly for position-only (Dirichlet) attention.
    
    This confirms the theory is sound when W[i,j] = D_S(θ_i - θ_j).
    """
    print("=" * 70)
    print("TEST: CIMBA on Position-Only Dirichlet Attention")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    seq_len = 256
    head_dim = 64
    S = 32  # Bandwidth
    max_seq_len = 4096
    
    # Random values
    V = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    # Position angles
    theta = 2 * math.pi * torch.arange(seq_len, device=DEVICE).float() / max_seq_len
    k_vals = torch.arange(-S, S+1, device=DEVICE, dtype=torch.float32)
    
    # === Direct Dirichlet computation ===
    def dirichlet_kernel(delta, S):
        x = delta + 1e-10
        return torch.sin((S + 0.5) * x) / torch.sin(x / 2) / (2*S + 1)
    
    delta = theta.unsqueeze(0) - theta.unsqueeze(1)  # [q, k]
    W_dirichlet = dirichlet_kernel(delta, S)
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE))
    W_causal = W_dirichlet * causal_mask
    W_causal = W_causal / (W_causal.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Direct output
    output_direct = W_causal @ V  # [seq_len, head_dim]
    
    # === CIMBA moment reconstruction ===
    V_complex = V.to(torch.complex64)
    output_cimba = torch.zeros_like(V)
    
    # Accumulate moments incrementally (causal)
    moments = torch.zeros(2*S+1, head_dim, device=DEVICE, dtype=torch.complex64)
    norm_moments = torch.zeros(2*S+1, device=DEVICE, dtype=torch.complex64)
    
    for t in range(seq_len):
        # Update moments with new V
        phase_t = torch.exp(1j * k_vals * theta[t])  # [2S+1]
        moments += phase_t.unsqueeze(1) * V_complex[t].unsqueeze(0)  # [2S+1, d]
        norm_moments += phase_t
        
        # Reconstruct at position t
        query_phase = torch.exp(-1j * k_vals * theta[t])  # [2S+1]
        
        # Unnormalized output
        raw_output = (query_phase @ moments).real / (2*S + 1)
        
        # Normalization factor
        norm_factor = (query_phase @ norm_moments).real / (2*S + 1)
        
        output_cimba[t] = raw_output / (norm_factor + 1e-10)
    
    # Compare
    max_err = (output_direct - output_cimba).abs().max().item()
    mean_err = (output_direct - output_cimba).abs().mean().item()
    
    print(f"\nSeq length: {seq_len}, Bandwidth S: {S}")
    print(f"Max error:  {max_err:.2e}")
    print(f"Mean error: {mean_err:.2e}")
    print(f"Status: {'✓ EXACT' if max_err < 1e-4 else '✗ NOT EXACT'}")
    
    return max_err < 1e-4


def test_cimba_on_content_based():
    """
    Test: Can CIMBA sketch content-based attention (QK similarity)?
    
    This tests if we can use O(S) moments when attention depends on content.
    
    Approach: Try to use the ACTUAL attention weights (from QK) to weight V,
    but only store S moments. This won't work because the weights aren't
    of the form f(θ_i - θ_j).
    """
    print("\n" + "=" * 70)
    print("TEST: CIMBA on Content-Based Attention")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    seq_len = 64
    head_dim = 32
    S = 16
    max_seq_len = 4096
    
    # Random Q, K, V
    Q = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    K = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    V = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    # Standard content-based attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = Q @ K.T * scale  # [seq_len, seq_len]
    
    # Bandlimit the scores (SpectralAttention style)
    rfft_len = seq_len // 2 + 1
    scores_fft = torch.fft.rfft(scores.float(), dim=-1)
    mask = torch.zeros(rfft_len, device=DEVICE)
    mask[:S] = 1.0
    scores_fft_filtered = scores_fft * mask
    scores_filtered = torch.fft.irfft(scores_fft_filtered, n=seq_len, dim=-1).to(DTYPE)
    
    # Causal mask + softmax
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool), diagonal=1)
    scores_filtered = scores_filtered.masked_fill(causal_mask, float('-inf'))
    W_spectral = F.softmax(scores_filtered, dim=-1)
    
    # Direct output
    output_direct = W_spectral @ V  # [seq_len, head_dim]
    
    print(f"\nSeq length: {seq_len}, Bandwidth S: {S}")
    
    # === Attempt 1: Use position moments anyway (will fail) ===
    print("\nAttempt 1: Position-based CIMBA (ignoring content)")
    
    theta = 2 * math.pi * torch.arange(seq_len, device=DEVICE).float() / max_seq_len
    k_vals = torch.arange(-S, S+1, device=DEVICE, dtype=torch.float32)
    
    V_complex = V.to(torch.complex64)
    output_cimba_pos = torch.zeros_like(V)
    
    moments = torch.zeros(2*S+1, head_dim, device=DEVICE, dtype=torch.complex64)
    norm_moments = torch.zeros(2*S+1, device=DEVICE, dtype=torch.complex64)
    
    for t in range(seq_len):
        phase_t = torch.exp(1j * k_vals * theta[t])
        moments += phase_t.unsqueeze(1) * V_complex[t].unsqueeze(0)
        norm_moments += phase_t
        
        query_phase = torch.exp(-1j * k_vals * theta[t])
        raw_output = (query_phase @ moments).real / (2*S + 1)
        norm_factor = (query_phase @ norm_moments).real / (2*S + 1)
        output_cimba_pos[t] = raw_output / (norm_factor + 1e-10)
    
    max_err_pos = (output_direct - output_cimba_pos).abs().max().item()
    mean_err_pos = (output_direct - output_cimba_pos).abs().mean().item()
    print(f"  Max error:  {max_err_pos:.2e}")
    print(f"  Mean error: {mean_err_pos:.2e}")
    print(f"  Status: {'✓ EXACT' if max_err_pos < 1e-4 else '✗ NOT EXACT (expected)'}")
    
    # === Attempt 2: Sketch the pre-softmax filtered scores ===
    print("\nAttempt 2: Sketch pre-softmax scores (store score moments)")
    
    # Can we reconstruct scores_filtered[t, :t] from moments?
    # scores_filtered is [q_len, kv_len], we need the filtered row for each query
    # This is content-dependent, so we'd need to store moments for each query separately
    # That's O(N * S), not O(S)
    
    print("  Analysis: Pre-softmax scores are content-dependent (Q[t] @ K[:].T)")
    print("  Each query position t produces a different score pattern.")
    print("  Cannot reconstruct scores[t, :] from O(S) moments without storing Q[t].")
    print("  Status: ✗ NOT O(S) SKETCHABLE (requires O(N) or storing Q)")
    
    # === Attempt 3: Per-row approximation error analysis ===
    print("\nAttempt 3: Analyze per-row approximation error")
    
    errors_per_row = []
    for t in range(seq_len):
        err = (output_direct[t] - output_cimba_pos[t]).abs().mean().item()
        errors_per_row.append(err)
    
    print(f"  Error range across rows: [{min(errors_per_row):.4f}, {max(errors_per_row):.4f}]")
    print(f"  Mean error per row: {sum(errors_per_row)/len(errors_per_row):.4f}")
    
    return False  # Content-based cannot be exactly sketched


def test_spectral_vs_dirichlet_outputs():
    """
    Compare actual outputs from SpectralAttention vs Dirichlet attention.
    
    This shows how different the two operators are for the same input.
    """
    print("\n" + "=" * 70)
    print("TEST: SpectralAttention vs Dirichlet - Output Comparison")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    seq_len = 64
    head_dim = 32
    S = 16
    max_seq_len = 4096
    
    # Random Q, K, V
    Q = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    K = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    V = torch.randn(seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    
    # === SpectralAttention output ===
    scale = 1.0 / math.sqrt(head_dim)
    scores = Q @ K.T * scale
    
    rfft_len = seq_len // 2 + 1
    scores_fft = torch.fft.rfft(scores.float(), dim=-1)
    mask = torch.zeros(rfft_len, device=DEVICE)
    mask[:S] = 1.0
    scores_filtered = torch.fft.irfft(scores_fft * mask, n=seq_len, dim=-1).to(DTYPE)
    
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool), diagonal=1)
    scores_filtered = scores_filtered.masked_fill(causal_mask, float('-inf'))
    W_spectral = F.softmax(scores_filtered, dim=-1)
    output_spectral = W_spectral @ V
    
    # === Dirichlet attention output ===
    theta = 2 * math.pi * torch.arange(seq_len, device=DEVICE).float() / max_seq_len
    
    def dirichlet_kernel(delta, S):
        x = delta + 1e-10
        return torch.sin((S + 0.5) * x) / torch.sin(x / 2) / (2*S + 1)
    
    delta = theta.unsqueeze(0) - theta.unsqueeze(1)
    W_dirichlet = dirichlet_kernel(delta, S)
    
    causal_dirichlet = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE))
    W_dirichlet = W_dirichlet * causal_dirichlet
    W_dirichlet = W_dirichlet / (W_dirichlet.sum(dim=-1, keepdim=True) + 1e-10)
    output_dirichlet = W_dirichlet @ V
    
    # Compare
    diff = (output_spectral - output_dirichlet).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Also compare attention patterns
    W_diff = (W_spectral - W_dirichlet.to(W_spectral.dtype)).abs()
    
    print(f"\nSeq length: {seq_len}, Bandwidth S: {S}")
    print(f"\nOutput difference:")
    print(f"  Max:  {max_diff:.4f}")
    print(f"  Mean: {mean_diff:.4f}")
    print(f"\nAttention weight difference:")
    print(f"  Max:  {W_diff.max().item():.4f}")
    print(f"  Mean: {W_diff.mean().item():.4f}")
    
    print(f"\nConclusion: SpectralAttention and Dirichlet are DIFFERENT operators.")
    print(f"  Replacing one with the other changes model behavior.")


def main():
    print("\n" + "=" * 70)
    print("DEEP ANALYSIS: CAN CIMBA SKETCH SPECTRAL ATTENTION?")
    print("=" * 70)
    
    # Test 1: CIMBA works for position-only
    result1 = test_cimba_on_position_only()
    
    # Test 2: CIMBA fails for content-based
    result2 = test_cimba_on_content_based()
    
    # Test 3: Compare operators
    test_spectral_vs_dirichlet_outputs()
    
    # Summary
    print("\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
FINDING: CIMBA moment sketching CANNOT exactly reproduce SpectralAttention.

REASON:
- CIMBA requires attention weights of the form W[i,j] = f(θ_i - θ_j)
  (i.e., weights depend only on position difference)
- SpectralAttention produces W[i,j] that depends on (Q[i], K[j]) content
- Even with bandlimited pre-softmax scores, the weights are content-dependent

WHAT WORKS:
- CIMBA perfectly sketches Dirichlet/position-only attention (verified above)
- The existing CIMBAAttention (Dirichlet-based) in the repo is correct for its operator

PATHS FORWARD:

Option A: Use Dirichlet attention instead of SpectralAttention
  - Accept content-independence as a trade-off for O(S) memory
  - Current CIMBAAttention already implements this
  - Requires training the model to work without Q·K similarity
  
Option B: Hybrid approach - O(S) for V moments, O(N) for attention computation
  - Store V in CIMBA moments
  - Compute attention weights on-the-fly (still requires all K)
  - Saves ~50% memory (V is half of KV cache)
  - Still O(N) for attention, but uses O(S) V storage

Option C: Accept that SpectralAttention can't be O(S) sketched
  - Use SpectralAttention with standard KV cache
  - Benefits: content-based attention preserved
  - Drawback: no memory improvement from CIMBA

Option D: Research new sketching methods for content-based attention
  - Outside current CIMBA theory
  - May not be possible with O(S) memory

RECOMMENDATION: 
Be explicit about which operator each component uses:
- SpectralAttention = content-based (QK) + bandlimited filtering + softmax (NOT O(S) sketchable)
- CIMBAAttention/DirichletAttention = position-only (Dirichlet) (O(S) sketchable)
""")


if __name__ == "__main__":
    main()
