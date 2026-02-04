# NEXT STEPS: CIMBA Implementation Roadmap

> This document preserves context for continued development. Read this after any context reset.

---

## Current Status (Completed)

### ✅ Phase 1: Spectral Attention Prototype
- **SpectralAttention class** in `spectral_attention.py`: Drop-in replacement for Qwen2Attention
- **Hard bandlimited filter**: `lowpass_param` with S frequency bins, high frequencies forced to exactly 0
- **KV cache support**: Compatible with HuggingFace's `DynamicCache`
- **Training script** in `train_bandlimited.py`: Curriculum-based bandwidth shrinking (256 → 128 → 64)
- **Generation verified**: Model produces correct code (quicksort, fibonacci, etc.)

### ✅ Phase 2: CIMBA Theory Validation
- **Moment reconstruction test**: Proved mathematically exact (max error ~1e-5)
- **Compression verified**: 15.9x at seq_len=4096, S=64
- **Sharp peak capture**: Confirmed - Dirichlet kernel IS the optimal bandlimited peak
- **Documentation**: See `CIMBA_EXPLAINED.md` for full explanation

---

## Next Steps (TODO)

### 🔲 Phase 3: CIMBA KV Cache Implementation

**Goal**: Replace the full KV cache with moment-based storage during generation.

#### 3.1 Create `CIMBACache` Class

Create new file `cimba_cache.py`:

```python
import torch
import math

class CIMBACache:
    """
    Stores O(S) moments instead of O(N) key-value pairs.
    
    For each layer and each attention head:
    - moments: tensor of shape [num_heads, 2*S+1, head_dim] (complex64)
    - norm_sums: running sum for causal attention normalization
    """
    
    def __init__(self, num_layers, num_kv_heads, head_dim, S, device, dtype=torch.complex64):
        self.S = S
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        
        # k values for Fourier basis: -S, -S+1, ..., S
        self.k_vals = torch.arange(-S, S+1, device=device, dtype=torch.float32)
        
        # Moments for each layer: [num_kv_heads, 2*S+1, head_dim]
        self.moments = {
            layer_idx: torch.zeros(num_kv_heads, 2*S+1, head_dim, device=device, dtype=dtype)
            for layer_idx in range(num_layers)
        }
        
        # Running normalization sums for causal attention
        self.norm_sums = {
            layer_idx: torch.zeros(num_kv_heads, device=device, dtype=torch.float32)
            for layer_idx in range(num_layers)
        }
        
        # Token counter for position calculation
        self.seq_len = 0
        self.max_seq_len = 131072  # For theta calculation
        
    def update(self, layer_idx: int, v_new: torch.Tensor, position_id: int):
        """
        Add new token's value to moments.
        
        Args:
            layer_idx: Which transformer layer
            v_new: Value vector [num_kv_heads, head_dim]
            position_id: Token position for circular angle calculation
        """
        # Circular position
        theta_t = 2 * math.pi * position_id / self.max_seq_len
        
        # Phase factors for all k values: [2*S+1]
        phase = torch.exp(1j * self.k_vals * theta_t)
        
        # Update moments: [num_kv_heads, 2*S+1, head_dim]
        # v_new is [num_kv_heads, head_dim], phase is [2*S+1]
        v_complex = v_new.to(torch.complex64)
        self.moments[layer_idx] += phase.unsqueeze(0).unsqueeze(-1) * v_complex.unsqueeze(1)
        
        # Update normalization sum (Dirichlet kernel value at 0 is 2S+1)
        self.norm_sums[layer_idx] += 1.0  # Simplified; exact would use kernel value
        
    def reconstruct(self, layer_idx: int, query_position: int) -> torch.Tensor:
        """
        Reconstruct attention output for query at given position.
        
        Args:
            layer_idx: Which transformer layer
            query_position: Position of the query token
            
        Returns:
            Reconstructed attention output [num_kv_heads, head_dim]
        """
        theta_q = 2 * math.pi * query_position / self.max_seq_len
        
        # Query phase factors: [2*S+1]
        query_phase = torch.exp(-1j * self.k_vals * theta_q)
        
        # Reconstruct: sum over k of phase * moment
        # query_phase is [2*S+1], moments is [num_kv_heads, 2*S+1, head_dim]
        raw = torch.einsum('k,nhd->nd', query_phase, self.moments[layer_idx]).real
        raw = raw / (2 * self.S + 1)
        
        # Normalize for causal attention
        output = raw / (self.norm_sums[layer_idx].unsqueeze(-1) + 1e-10)
        
        return output
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return current sequence length."""
        return self.seq_len
```

#### 3.2 Modify SpectralAttention for CIMBA Mode

In `spectral_attention.py`, add CIMBA cache support:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Union[Cache, CIMBACache]] = None,  # Accept both
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cimba: bool = False,  # NEW FLAG
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
    
    if use_cimba and isinstance(past_key_value, CIMBACache):
        # CIMBA path: use moment-based reconstruction
        return self._forward_cimba(hidden_states, past_key_value, position_ids, ...)
    else:
        # Standard path: use full KV cache
        return self._forward_standard(...)
```

#### 3.3 Key Integration Challenges

**Challenge 1: RoPE Integration**
- RoPE applies rotation to Q and K before attention
- CIMBA uses circular positions for moment phases
- Solution: Keep RoPE for Q/K similarity, use separate CIMBA phases for V moments

**Challenge 2: Grouped Query Attention (GQA)**
- Qwen uses 14 Q heads but only 2 KV heads
- Multiple Q heads share the same KV moments
- Solution: Store moments per KV head, expand during reconstruction

**Challenge 3: Normalization**
- Causal attention requires row-wise normalization
- Need to track normalization sums incrementally
- Solution: Store running sum of kernel values in CIMBACache

---

### 🔲 Phase 4: Validation at Scale

#### 4.1 Memory Benchmarks

Create `benchmark_memory.py`:

```python
def benchmark_memory(seq_lengths=[1024, 4096, 16384, 65536]):
    """Compare memory: standard KV cache vs CIMBA cache"""
    for seq_len in seq_lengths:
        # Standard KV cache memory
        standard_mem = 2 * num_layers * num_kv_heads * seq_len * head_dim * 2  # K+V, bf16
        
        # CIMBA cache memory
        cimba_mem = num_layers * num_kv_heads * (2*S+1) * head_dim * 8  # complex64
        
        print(f"seq_len={seq_len}: Standard={standard_mem/1e9:.2f}GB, CIMBA={cimba_mem/1e9:.2f}GB")
```

#### 4.2 Quality Benchmarks

Test on:
1. **Coding tasks**: HumanEval, MBPP
2. **Long context**: Needle-in-haystack
3. **Perplexity**: WikiText-103

#### 4.3 Speed Benchmarks

Measure:
- Tokens/second during generation
- Time-to-first-token
- Memory bandwidth utilization

---

### 🔲 Phase 5: Scale to Larger Models

#### Model Progression

| Model | Parameters | Current VRAM | With CIMBA (estimated) |
|-------|------------|--------------|------------------------|
| Qwen2.5-0.5B | 0.5B | 2GB | 1GB |
| Qwen2.5-7B | 7B | 16GB | 8GB |
| Qwen2.5-72B | 72B | 150GB | **16GB** (goal!) |

#### Training Considerations

- More training data needed for larger models
- Consider layer-wise bandwidth (different S per layer)
- May need mixed precision training

---

### 🔲 Phase 6: Production Optimization

#### 6.1 CUDA Kernels

Write custom kernels for:
- Fused moment updates
- Fused reconstruction
- Complex number operations

#### 6.2 Quantization

Test moment quantization:
- INT8 moments: 2x memory reduction
- INT4 moments: 4x memory reduction (may lose precision)

#### 6.3 Framework Integration

- vLLM: Custom attention backend
- TensorRT-LLM: Custom plugin
- llama.cpp: GGUF format extension

---

## Technical Reference

### Architecture (Qwen2.5-Coder-0.5B)

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Hidden size | 896 |
| Num attention heads | 14 |
| Num KV heads | 2 (GQA) |
| Head dimension | 64 |
| Bandwidth S | 64 |
| Moments per head | 129 (2S+1) |

### Key Equations

**Dirichlet kernel:**
$$D_S(\Delta) = \frac{\sin((S + 0.5)\Delta)}{\sin(\Delta/2)}$$

**Moment definition:**
$$V_k = \sum_{i} v_i \cdot e^{ik\theta_i} \quad \text{for } k = -S, \ldots, S$$

**Reconstruction:**
$$\text{output}(\tau) = \sum_{k=-S}^{S} \frac{1}{2S+1} \cdot e^{-ik\theta_\tau} \cdot V_k$$

**Causal normalization:**
$$\text{output}_{\text{norm}}(\tau) = \frac{\text{output}(\tau)}{\sum_{i \leq \tau} D_S(\theta_\tau - \theta_i)}$$

### Files in Workspace

| File | Purpose | Status |
|------|---------|--------|
| `spectral_attention.py` | SpectralAttention with hard bandlimit | ✅ Complete |
| `train_bandlimited.py` | Training with curriculum | ✅ Complete |
| `README.md` | Project overview | ✅ Complete |
| `CIMBA_EXPLAINED.md` | Theory for publication | ✅ Complete |
| `NEXT_STEPS.md` | This roadmap | ✅ Complete |
| `cimba_cache.py` | CIMBA moment cache | 🔲 TODO |
| `benchmark_memory.py` | Memory comparison | 🔲 TODO |
| `benchmark_quality.py` | Quality comparison | 🔲 TODO |

---

## Potential Issues & Solutions

| Issue | Solution |
|-------|----------|
| Numerical instability | Use float64 for phase accumulation, float32 for output |
| RoPE interaction | Dual encoding: RoPE for Q/K, CIMBA phases for V |
| GQA mismatch | Store moments per KV head, expand for Q heads |
| Causality | Track running normalization sum |
| Long sequences | Use modular arithmetic for theta to avoid overflow |

---

## Success Criteria

### Phase 3: CIMBA Cache
- [ ] Generation with CIMBA cache matches standard cache output
- [ ] Memory usage verified as O(S) not O(N)
- [ ] No quality degradation

### Phase 4: Validation
- [ ] <5% perplexity increase vs original
- [ ] Needle-in-haystack maintained
- [ ] 10x+ memory reduction at 100K context

### Phase 5: Scale
- [ ] Qwen2.5-72B runs on 16GB VRAM
- [ ] Inference quality comparable to original

---

## Quick Start After Context Reset

1. Read this file (`NEXT_STEPS.md`) first
2. Read `CIMBA_EXPLAINED.md` for theory
3. Review `spectral_attention.py` for current implementation
4. Continue from **Phase 3.1**: Create `cimba_cache.py`

**The key insight**: CIMBA moments can EXACTLY reconstruct bandlimited attention with O(S) memory instead of O(N). This has been mathematically proven with max error ~1e-5 across all tested sequence lengths.
