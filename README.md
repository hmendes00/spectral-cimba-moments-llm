# Spectral Attention

A drop-in replacement for transformer softmax attention that uses FFT-based spectral filtering. This project explores whether attention patterns can be made **bandlimited** (sparse in frequency domain) to enable more efficient computation.

## What is Spectral Attention?

Standard transformer attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Spectral Attention adds a **learnable frequency-domain filter** to the attention scores:
```
scores = QK^T / √d
filtered_scores = IFFT(FFT(scores) * spectral_filter)
Attention(Q, K, V) = softmax(filtered_scores) · V
```

The `spectral_filter` is a learnable parameter that can suppress or amplify specific frequency components in the attention pattern.

## Current Status: Proof of Concept ✓

### What Works
- **Drop-in replacement**: SpectralAttention can replace Qwen2Attention with zero quality loss
- **No finetuning required**: With filter initialized to 1.0 (identity), the model produces identical outputs
- **Full generation support**: KV cache, RoPE, GQA all work correctly
- **Minimal overhead**: Only 49K additional parameters (0.01% of model)

### Test Results (Qwen2.5-Coder-0.5B)
| Test | Baseline | Spectral Attention |
|------|----------|-------------------|
| Palindrome check | ✓ Correct | ✓ Identical |
| Bubble sort | ✓ Correct | ✓ Identical |
| Fibonacci | ✓ Correct | ✓ Identical |
| Binary search | ✓ Correct | ✓ Identical |
| Linked list reversal | ✓ Correct | ✓ Identical |

### What's NOT Yet Proven
- Whether the spectral filter can be trained to become **sparse** (bandlimited)
- Whether bandlimited attention enables O(K) memory via moment sketching (CIMBA)
- Whether quality is preserved after aggressive frequency filtering

## Theoretical Foundation

This project is inspired by unpublished research by **Hugo Leonardo Simas Mendes**:

### From CIMBA (Circular Moment-Based Attention)
- The idea that attention patterns may have exploitable structure in the frequency domain
- Moment-based sketching for O(K) memory KV cache (not yet implemented here)
- The insight that if attention is bandlimited, it can be exactly reconstructed from a finite number of moments

### From CircleStat (Circular Statistics for Sequence Modeling)
- Circular cross-correlation via FFT as an alternative to dot-product attention
- The Dirichlet kernel interpretation: bandlimited kernels that are "peaky" like softmax
- The theoretical framework for why spectral methods might work for attention

### Key Insight
The original goal was to **approximate** softmax attention with CIMBA moments. This failed (52% MSE) because softmax attention is NOT bandlimited—it has infinite frequency content.

The pivot: Instead of approximating softmax, **replace** it with a mechanism that is spectral by construction. Train the model to work with bandlimited attention, then apply CIMBA sketching.

## Project Structure

```
spectral-att/
├── README.md                      # This file
├── SPECTRAL_ATTENTION_SPEC.md     # Original specification document
├── spectral_attention.py          # Main implementation
│   ├── SpectralAttention          # Drop-in attention replacement
│   ├── PureSpectralAttention      # Experimental: no softmax variant
│   └── replace_attention_with_spectral()  # Helper function
├── test_spectral_comprehensive.py # Multi-prompt generation tests
├── test_spectral_generation.py    # Basic generation test
├── test_minimal_attention.py      # Debugging: minimal correct attention
├── test_single_forward.py         # Debugging: single forward pass comparison
└── papers/                        # Reference papers (unpublished)
    ├── CIMBA_tex.txt
    ├── CircleStat_tex.txt
    ├── MAST_tex.txt
    └── sbgp_tex.txt
```

## Key Files

### [spectral_attention.py](spectral_attention.py)
The main implementation. Contains:
- `SpectralAttention`: The production-ready attention replacement
- Proper handling of: RoPE, GQA, KV cache, causal masking, bfloat16
- Learnable `spectral_filter` and `spectral_sharpness` parameters

### [test_spectral_comprehensive.py](test_spectral_comprehensive.py)
Validates that SpectralAttention produces identical outputs to baseline across multiple coding tasks.

## Technical Details

### Why FFT?
The FFT transforms the attention pattern from position domain to frequency domain:
- **Low frequencies**: Global patterns (overall attention distribution)
- **High frequencies**: Local patterns (sharp peaks, fine details)

By filtering in frequency domain, we can control the "bandwidth" of attention—how much fine detail vs. smooth structure it captures.

### The Spectral Filter
```python
spectral_filter = nn.Parameter(torch.ones(max_position_embeddings // 2 + 1))
```
- Shape: `[seq_len // 2 + 1]` (rfft output size)
- Initialized to 1.0 (identity filter, no modification)
- Can be trained to learn which frequencies matter

### The Sharpness Parameter
```python
spectral_sharpness = nn.Parameter(torch.tensor(1.0))
```
- Controls how "peaky" the filter becomes: `filter ** softplus(sharpness)`
- Higher sharpness = more aggressive filtering

## Next Steps

1. **Train the spectral filter**: Finetune on coding tasks while encouraging sparsity in frequency domain
2. **Measure bandwidth**: After training, check if attention becomes bandlimited (most energy in low frequencies)
3. **Implement CIMBA sketching**: If bandlimited, replace full KV cache with O(K) moment representation
4. **Scale up**: Test on larger models (Qwen 7B, 72B) to validate memory savings

## Citation

If you use this work, please cite the theoretical foundations:

```
Hugo Leonardo Simas Mendes. "CIMBA: Circular Moment-Based Attention for 
O(K) Memory Transformer Inference." Unpublished manuscript, 2025.

Hugo Leonardo Simas Mendes. "CircleStat: Circular Statistics for Sequence
Modeling." Unpublished manuscript, 2025.
```

## License

Research code. Contact Hugo Leonardo Simas Mendes for licensing.
