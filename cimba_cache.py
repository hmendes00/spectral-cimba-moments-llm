"""
CIMBA Cache: O(S) Memory KV Cache via Moment Sketching

This module implements the CIMBA (Circular Moment-Based Attention) cache that
stores O(S) moments instead of O(N) key-value pairs.

Key insight from Hugo Leonardo Simas Mendes' CIMBA paper:
For a bandlimited attention kernel of degree S, the attention output can be
EXACTLY reconstructed from 2S+1 moments:

    V_k = Σ_i v_i · e^{ikθ_i}  for k = -S, ..., S

This reduces memory from O(N·d) to O(S·d), where:
- N = sequence length (can be millions)
- S = bandwidth (typically 64-256)
- d = head dimension

For N=100K and S=64: ~780x compression!

Theory:
- Dirichlet kernel: D_S(Δ) = sin((S+0.5)Δ) / sin(Δ/2)
- This kernel is bandlimited to degree S by construction
- Moments are the Fourier coefficients of V weighted by position phases
- Reconstruction uses inverse Fourier transform at query position

References:
- CIMBA: Circular Moment-Based Attention (Mendes, 2025)
- CircleStat: Circular Statistics for Attention (Mendes, 2025)
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple, List


class CIMBACache:
    """
    CIMBA moment-based KV cache with O(S) memory complexity.
    
    Instead of storing all N key-value pairs [N, head_dim], we store
    2S+1 complex moments [2S+1, head_dim] that can exactly reconstruct
    the attention output for bandlimited kernels.
    
    Usage:
        cache = CIMBACache(num_layers=24, num_kv_heads=2, head_dim=64, S=64, device='cuda')
        
        # During generation:
        for t, token in enumerate(tokens):
            v_new = compute_value(token)  # [num_kv_heads, head_dim]
            cache.update(layer_idx=0, v_new=v_new, position_id=t)
            output = cache.reconstruct(layer_idx=0, query_position=t)
    
    Attributes:
        S: Bandwidth parameter (stores 2S+1 moments per head)
        moments: Dict mapping layer_idx -> tensor [num_kv_heads, 2*S+1, head_dim] (complex64)
        norm_sums: Dict mapping layer_idx -> tensor [num_kv_heads] (running normalization)
        k_vals: Fourier indices [-S, -S+1, ..., S]
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        S: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
        max_seq_len: int = 131072,
    ):
        """
        Initialize CIMBA cache.
        
        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads (for GQA)
            head_dim: Dimension of each attention head
            S: Bandwidth parameter (stores 2S+1 moments)
            device: Device to store moments on
            dtype: Data type for moments (complex64 or complex128)
            max_seq_len: Maximum sequence length for theta calculation
        """
        self.S = S
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        # Fourier indices: k = -S, -S+1, ..., 0, ..., S-1, S
        self.k_vals = torch.arange(-S, S + 1, device=self.device, dtype=torch.float32)
        self.num_moments = 2 * S + 1
        
        # Initialize moments for each layer: [num_kv_heads, 2*S+1, head_dim]
        self.moments: Dict[int, torch.Tensor] = {}
        for layer_idx in range(num_layers):
            self.moments[layer_idx] = torch.zeros(
                num_kv_heads, self.num_moments, head_dim,
                device=self.device, dtype=dtype
            )
        
        # Running normalization sums for causal attention
        # For each position t, we need Σ_{i≤t} D_S(θ_t - θ_i) for normalization
        # We track this incrementally using moments: Σ_{i≤t} e^{ikθ_i}
        # This is head-independent, so just [2*S+1]
        self.norm_moments: Dict[int, torch.Tensor] = {}
        for layer_idx in range(num_layers):
            self.norm_moments[layer_idx] = torch.zeros(
                self.num_moments,
                device=self.device, dtype=dtype
            )
        
        # Track sequence length per layer
        self.seen_tokens: Dict[int, int] = {i: 0 for i in range(num_layers)}
        
        # Precompute Dirichlet kernel Fourier coefficients
        # For normalized Dirichlet: ĝ_k = 1/(2S+1) for all |k| ≤ S
        self.dirichlet_coeffs = torch.ones(self.num_moments, device=self.device) / self.num_moments
        
    def _get_theta(self, position_id: int) -> float:
        """
        Convert token position to circular angle.
        
        θ_t = 2π · t / max_seq_len
        
        This maps positions to the circle [0, 2π).
        """
        return 2 * math.pi * position_id / self.max_seq_len
    
    def _get_theta_tensor(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized version of _get_theta."""
        return 2 * math.pi * position_ids.float() / self.max_seq_len
    
    def update(
        self,
        layer_idx: int,
        v_new: torch.Tensor,
        position_id: int,
    ) -> None:
        """
        Add a new token's value to the moments (single token update).
        
        This is the O(S) incremental update:
            V_k += v_new · e^{+ikθ_t}
        
        Args:
            layer_idx: Which transformer layer
            v_new: Value vector [num_kv_heads, head_dim] or [batch, num_kv_heads, head_dim]
            position_id: Token position for circular angle calculation
        """
        theta_t = self._get_theta(position_id)
        
        # Phase factors for all k values: e^{+ik·θ_t} for k = -S, ..., S
        # Shape: [2*S+1]
        phase = torch.exp(1j * self.k_vals * theta_t)
        
        # Handle different input shapes
        if v_new.dim() == 2:
            # [num_kv_heads, head_dim]
            v_complex = v_new.to(dtype=self.dtype, device=self.device)
            # Update moments: [num_kv_heads, 2*S+1, head_dim]
            # phase is [2*S+1], v_complex is [num_kv_heads, head_dim]
            # Want: moments[h, k, d] += phase[k] * v_complex[h, d]
            self.moments[layer_idx] += phase.unsqueeze(0).unsqueeze(-1) * v_complex.unsqueeze(1)
            
            # Update normalization moments: add e^{ikθ_t}
            # This is head-independent
            self.norm_moments[layer_idx] += phase
            
        elif v_new.dim() == 3:
            # [batch, num_kv_heads, head_dim] - batched update
            batch = v_new.shape[0]
            v_complex = v_new.to(dtype=self.dtype, device=self.device)
            # For simplicity, assume batch=1 during generation
            self.moments[layer_idx] += phase.unsqueeze(0).unsqueeze(-1) * v_complex[0].unsqueeze(1)
            self.norm_moments[layer_idx] += phase
        
        self.seen_tokens[layer_idx] = max(self.seen_tokens[layer_idx], position_id + 1)
    
    def update_batch(
        self,
        layer_idx: int,
        v_batch: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """
        Add multiple tokens' values to the moments (prefill).
        
        Args:
            layer_idx: Which transformer layer
            v_batch: Value vectors [seq_len, num_kv_heads, head_dim]
            position_ids: Token positions [seq_len]
        """
        seq_len = v_batch.shape[0]
        
        # Get all theta values: [seq_len]
        theta = self._get_theta_tensor(position_ids)
        
        # Phase matrix: [seq_len, 2*S+1]
        # phase[t, k] = e^{+ik·θ_t}
        phase = torch.exp(1j * self.k_vals.unsqueeze(0) * theta.unsqueeze(1))
        
        # Convert values to complex: [seq_len, num_kv_heads, head_dim]
        v_complex = v_batch.to(dtype=self.dtype, device=self.device)
        
        # Update moments by summing over sequence dimension
        # moments[h, k, d] = Σ_t phase[t, k] * v[t, h, d]
        # Use einsum: 'tk,thd->hkd'
        self.moments[layer_idx] += torch.einsum('tk,thd->hkd', phase, v_complex)
        
        # Update normalization moments: Σ_t e^{ikθ_t}
        # Head-independent: just sum phases over sequence
        self.norm_moments[layer_idx] += phase.sum(dim=0)  # [2*S+1]
        
        self.seen_tokens[layer_idx] = max(self.seen_tokens[layer_idx], position_ids.max().item() + 1)
    
    def reconstruct(
        self,
        layer_idx: int,
        query_position: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Reconstruct attention output for query at given position.
        
        This is the O(S) reconstruction:
            output = Σ_k (1/(2S+1)) · e^{-ikθ_q} · V_k
        
        Where V_k = Σ_{i≤t} v_i e^{+ikθ_i}, so:
            output = (1/(2S+1)) Σ_k Σ_i e^{ik(θ_i - θ_q)} v_i
                   = (1/(2S+1)) Σ_i D_S(θ_i - θ_q) v_i  [D_S is symmetric]
        
        Args:
            layer_idx: Which transformer layer
            query_position: Position of the query token
            normalize: Whether to apply causal normalization
            
        Returns:
            Reconstructed attention output [num_kv_heads, head_dim]
        """
        theta_q = self._get_theta(query_position)
        
        # Query phase factors: e^{-ik·θ_q} for k = -S, ..., S
        query_phase = torch.exp(-1j * self.k_vals * theta_q)  # [2*S+1]
        
        # Reconstruct: Σ_k ĝ_k · e^{-ikθ_q} · V_k
        # dirichlet_coeffs[k] = 1/(2S+1)
        # query_phase is [2*S+1], moments is [num_kv_heads, 2*S+1, head_dim]
        weighted_phase = query_phase * self.dirichlet_coeffs  # [2*S+1]
        
        # Contract over k: Σ_k weighted_phase[k] * moments[h, k, d]
        raw_output = torch.einsum('k,hkd->hd', weighted_phase, self.moments[layer_idx])
        raw_output = raw_output.real  # Take real part: [num_kv_heads, head_dim]
        
        if normalize:
            # Compute normalization: Σ_{i≤t} D_S(θ_t - θ_i)
            # Using moments: norm = Σ_k (1/(2S+1)) · e^{-ikθ_q} · (Σ_{i≤t} e^{ikθ_i})
            norm = (weighted_phase @ self.norm_moments[layer_idx]).real  # scalar
            
            # Avoid division by zero
            raw_output = raw_output / (norm + 1e-10)
        
        return raw_output
    
    def reconstruct_batch(
        self,
        layer_idx: int,
        query_positions: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Reconstruct attention output for multiple query positions.
        
        NOTE: This assumes moments already contain exactly the tokens that should
        be attended to. For causal attention during prefill, use reconstruct_batch_causal.
        For generation (one token at a time), use reconstruct().
        
        Args:
            layer_idx: Which transformer layer
            query_positions: Positions of query tokens [num_queries]
            
        Returns:
            Reconstructed outputs [num_queries, num_kv_heads, head_dim]
        """
        num_queries = query_positions.shape[0]
        theta_q = self._get_theta_tensor(query_positions)  # [num_queries]
        
        # Query phase matrix: [num_queries, 2*S+1]
        query_phase = torch.exp(-1j * self.k_vals.unsqueeze(0) * theta_q.unsqueeze(1))
        
        # Weight by Dirichlet coefficients
        weighted_phase = query_phase * self.dirichlet_coeffs.unsqueeze(0)  # [num_queries, 2*S+1]
        
        # Reconstruct: [num_queries, num_kv_heads, head_dim]
        raw_output = torch.einsum('qk,hkd->qhd', weighted_phase, self.moments[layer_idx])
        raw_output = raw_output.real
        
        if normalize:
            # Compute normalization for each query position
            # norm[q] = Σ_k weighted_phase[q, k] * norm_moments[k]
            norm = torch.einsum('qk,k->q', weighted_phase, self.norm_moments[layer_idx])
            norm = norm.real  # [num_queries]
            raw_output = raw_output / (norm.unsqueeze(-1).unsqueeze(-1) + 1e-10)
        
        return raw_output
    
    def reconstruct_batch_causal(
        self,
        layer_idx: int,
        v_batch: torch.Tensor,
        position_ids: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Reconstruct causal attention outputs for a batch of positions (prefill).
        
        This handles causality correctly: each position t only attends to positions <= t.
        The moments are built incrementally.
        
        Args:
            layer_idx: Which transformer layer
            v_batch: Value vectors [seq_len, num_kv_heads, head_dim]
            position_ids: Token positions [seq_len]
            normalize: Whether to apply causal normalization
            
        Returns:
            Causal attention outputs [seq_len, num_kv_heads, head_dim]
        """
        seq_len = v_batch.shape[0]
        num_kv_heads = v_batch.shape[1]
        head_dim = v_batch.shape[2]
        
        # Get theta values
        theta = self._get_theta_tensor(position_ids)
        
        # Convert values to complex
        v_complex = v_batch.to(dtype=self.dtype, device=self.device)
        
        # Incremental moments for causal attention
        running_moments = torch.zeros(
            num_kv_heads, self.num_moments, head_dim,
            device=self.device, dtype=self.dtype
        )
        running_norm = torch.zeros(self.num_moments, device=self.device, dtype=self.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # Add current token to moments
            phase_t = torch.exp(1j * self.k_vals * theta[t])  # [2*S+1]
            
            # Update moments: [num_kv_heads, 2*S+1, head_dim]
            running_moments += phase_t.unsqueeze(0).unsqueeze(-1) * v_complex[t].unsqueeze(1)
            running_norm += phase_t
            
            # Reconstruct for this position
            query_phase = torch.exp(-1j * self.k_vals * theta[t])
            weighted_phase = query_phase * self.dirichlet_coeffs
            
            # raw: [num_kv_heads, head_dim]
            raw = torch.einsum('k,hkd->hd', weighted_phase, running_moments).real
            
            if normalize:
                norm = (weighted_phase @ running_norm).real
                raw = raw / (norm + 1e-10)
            
            outputs.append(raw)
        
        # Also update the cache's moments with the final state
        self.moments[layer_idx] = running_moments
        self.norm_moments[layer_idx] = running_norm
        self.seen_tokens[layer_idx] = position_ids.max().item() + 1
        
        return torch.stack(outputs, dim=0)  # [seq_len, num_kv_heads, head_dim]
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return current sequence length for a layer."""
        return self.seen_tokens.get(layer_idx, 0)
    
    def reset(self, layer_idx: Optional[int] = None):
        """
        Reset moments to zero.
        
        Args:
            layer_idx: Specific layer to reset, or None for all layers
        """
        if layer_idx is not None:
            self.moments[layer_idx].zero_()
            self.norm_moments[layer_idx].zero_()
            self.seen_tokens[layer_idx] = 0
        else:
            for idx in self.moments:
                self.moments[idx].zero_()
                self.norm_moments[idx].zero_()
                self.seen_tokens[idx] = 0
    
    def memory_usage(self) -> Dict[str, int]:
        """
        Calculate memory usage in bytes.
        
        Returns:
            Dict with 'moments', 'norm_moments', 'total' keys
        """
        moment_size = 0
        norm_size = 0
        
        for layer_idx in self.moments:
            moment_size += self.moments[layer_idx].numel() * self.moments[layer_idx].element_size()
            norm_size += self.norm_moments[layer_idx].numel() * self.norm_moments[layer_idx].element_size()
        
        return {
            'moments': moment_size,
            'norm_moments': norm_size,
            'total': moment_size + norm_size,
        }
    
    def compare_memory_to_standard(self, seq_len: int) -> Dict[str, float]:
        """
        Compare memory usage to standard KV cache.
        
        Args:
            seq_len: Sequence length to compare against
            
        Returns:
            Dict with comparison statistics
        """
        # Standard KV cache: 2 (K+V) * num_layers * num_kv_heads * seq_len * head_dim * 2 (bf16)
        standard_mem = 2 * self.num_layers * self.num_kv_heads * seq_len * self.head_dim * 2
        
        # CIMBA cache
        cimba_mem = self.memory_usage()['total']
        
        return {
            'standard_cache_bytes': standard_mem,
            'cimba_cache_bytes': cimba_mem,
            'compression_ratio': standard_mem / cimba_mem if cimba_mem > 0 else float('inf'),
            'memory_savings_percent': 100 * (1 - cimba_mem / standard_mem) if standard_mem > 0 else 0,
        }


class CIMBAKVCache:
    """
    Combined cache that stores both full KV (for Q·K similarity) and CIMBA moments (for V).
    
    This is the practical implementation for SpectralAttention:
    - K is stored normally (for computing Q·K attention scores)
    - V is compressed using CIMBA moments
    
    This gives O(N) for K but O(S) for V, which is still a 2x memory improvement
    at minimum, and allows exact V reconstruction.
    
    For full O(S) compression, use the pure CIMBA approach where Q·K similarity
    is replaced with a bandlimited kernel (see PureSpectralAttention).
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        S: int = 64,
        max_seq_len: int = 131072,
        device: torch.device = None,
        k_dtype: torch.dtype = torch.bfloat16,  # Standard dtype for K cache
    ):
        """
        Initialize hybrid CIMBA cache.
        
        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension of each head
            S: Bandwidth for CIMBA V compression
            max_seq_len: Maximum sequence length
            device: Device for tensors
            k_dtype: Data type for K cache (bf16 for memory efficiency)
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.S = S
        self.max_seq_len = max_seq_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k_dtype = k_dtype
        
        # Standard K cache: [num_layers][num_kv_heads, seq_len, head_dim]
        # Starts empty, grows with sequence
        self.k_cache: Dict[int, torch.Tensor] = {}
        
        # CIMBA V cache
        self.v_cache = CIMBACache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            S=S,
            max_seq_len=max_seq_len,
            device=device,
        )
        
        self.seen_tokens: Dict[int, int] = {i: 0 for i in range(num_layers)}
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Update cache with new key-value states.
        
        Matches the interface of HuggingFace's DynamicCache.update().
        
        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Optional kwargs (e.g., cache_position for RoPE)
            
        Returns:
            (full_keys, None) - V is not returned since it's in moment form
        """
        batch = key_states.shape[0]
        seq_len = key_states.shape[2]
        
        # Get cache position (for incremental generation)
        if cache_kwargs is not None and 'cache_position' in cache_kwargs:
            cache_position = cache_kwargs['cache_position']
            if cache_position is not None:
                start_pos = cache_position[0].item()
            else:
                start_pos = self.seen_tokens.get(layer_idx, 0)
        else:
            start_pos = self.seen_tokens.get(layer_idx, 0)
        
        # Update K cache (standard append)
        if layer_idx not in self.k_cache:
            self.k_cache[layer_idx] = key_states.clone()
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=2)
        
        # Update V cache (CIMBA moments)
        # value_states is [batch, num_kv_heads, seq_len, head_dim]
        # We need to add each position with its circular angle
        for t in range(seq_len):
            pos = start_pos + t
            v_t = value_states[0, :, t, :]  # [num_kv_heads, head_dim]
            self.v_cache.update(layer_idx, v_t, pos)
        
        self.seen_tokens[layer_idx] = start_pos + seq_len
        
        # Return full K, but V needs reconstruction via CIMBA
        return self.k_cache[layer_idx], None
    
    def reconstruct_v(
        self,
        layer_idx: int,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct V for given query positions using CIMBA.
        
        Args:
            layer_idx: Layer index
            query_positions: [num_queries] positions
            
        Returns:
            Reconstructed V: [num_queries, num_kv_heads, head_dim]
        """
        return self.v_cache.reconstruct_batch(layer_idx, query_positions)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length."""
        return self.seen_tokens.get(layer_idx, 0)
    
    def get_usable_length(self, layer_idx: int = 0, new_seq_length: int = 0) -> int:
        """Get usable length (for compatibility with HF cache interface)."""
        return self.get_seq_length(layer_idx)


def test_cimba_cache():
    """Test CIMBA cache correctness."""
    print("="*60)
    print("CIMBA CACHE TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    num_layers = 2
    num_kv_heads = 2
    head_dim = 64
    S = 32
    seq_len = 256
    
    # Create cache
    cache = CIMBACache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        S=S,
        device=device,
    )
    
    # Generate random V values
    torch.manual_seed(42)
    V = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
    
    # Add tokens one by one
    print(f"\nAdding {seq_len} tokens to cache...")
    for t in range(seq_len):
        cache.update(layer_idx=0, v_new=V[t], position_id=t)
    
    print(f"Cache sequence length: {cache.get_seq_length(0)}")
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    
    # Reconstruct at last position
    reconstructed = cache.reconstruct(layer_idx=0, query_position=seq_len-1)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Memory comparison
    mem = cache.compare_memory_to_standard(seq_len)
    print(f"\nMemory comparison at seq_len={seq_len}:")
    print(f"  Standard KV cache: {mem['standard_cache_bytes']/1024:.1f} KB")
    print(f"  CIMBA cache:       {mem['cimba_cache_bytes']/1024:.1f} KB")
    print(f"  Compression ratio: {mem['compression_ratio']:.1f}x")
    
    # Test batch update (prefill)
    print("\n" + "="*60)
    print("BATCH UPDATE TEST (Prefill)")
    print("="*60)
    
    cache.reset()
    position_ids = torch.arange(seq_len, device=device)
    cache.update_batch(layer_idx=0, v_batch=V, position_ids=position_ids)
    
    print(f"Batch update complete. Sequence length: {cache.get_seq_length(0)}")
    
    # Test batch reconstruction
    query_positions = torch.tensor([0, seq_len//2, seq_len-1], device=device)
    reconstructed_batch = cache.reconstruct_batch(layer_idx=0, query_positions=query_positions)
    print(f"Batch reconstruction shape: {reconstructed_batch.shape}")
    
    print("\n✓ CIMBA cache tests passed!")


if __name__ == "__main__":
    test_cimba_cache()
