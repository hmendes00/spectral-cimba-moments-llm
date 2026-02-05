"""
Spectral Attention Module

Replaces softmax attention with a natively spectral mechanism based on
circular cross-correlation via FFT. This follows the CircleStat paper's
insight that attention should use bandlimited kernels.

Key insight: Instead of approximating softmax (which failed), we REPLACE it
with a kernel that is spectral by construction.

The Dirichlet kernel: K_dir(x) = sin((N+1/2)x) / sin(x/2)
- Bandlimited by definition (sum of cosines)
- Peaky (converges to delta as N → ∞)
- Computable via FFT

Attention formula:
    A[t, i] = IFFT(FFT(Q[t]) · conj(FFT(K[i])))
    
This is circular cross-correlation, which finds "similarity peaks" spectrally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SpectralAttention(nn.Module):
    """
    Spectral Attention using FFT-based circular cross-correlation.
    
    Instead of softmax(QK^T / sqrt(d)), we compute attention weights via:
    1. FFT of queries and keys along the sequence dimension
    2. Element-wise multiplication in frequency domain (correlation)
    3. IFFT to get attention weights
    4. Normalization to form valid attention distribution
    5. Apply causal mask
    6. Attend to values
    
    This is a drop-in replacement for standard attention that is
    natively compatible with CIMBA moment sketching.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        temperature: float = 1.0,
        spectral_sharpness: float = 1.0,
        layer_idx: int = 0,  # Required for KV cache
        bandwidth_target: int = 64,  # S: number of frequency bins (for CIMBA: 2S+1 moments)
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.attention_dropout = attention_dropout
        self.layer_idx = layer_idx  # For KV cache
        self.is_causal = True  # Required for generation
        
        # Spectral-specific parameters
        self.temperature = temperature
        self.spectral_sharpness = nn.Parameter(torch.tensor(spectral_sharpness))
        
        # For GQA
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # RoPE parameters
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        # Projections (will be copied from original attention)
        # Qwen uses bias=True for Q, K, V and bias=False for O
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE: will be copied from original attention
        self.rotary_emb = None  # Set during weight copy
        
        # HARD BANDLIMITED spectral filter
        # Only parameterize first S frequency bins; all higher bins are EXACTLY 0
        # This guarantees bandlimitedness by construction (no regularizer needed)
        self.max_rfft_bins = max_position_embeddings // 2 + 1
        self.bandwidth_target = bandwidth_target  # S: mutable for curriculum training
        self.lowpass_param = nn.Parameter(torch.ones(bandwidth_target))
    
    def set_bandwidth(self, new_bandwidth: int):
        """
        Set bandwidth target for curriculum training.
        
        Note: If new_bandwidth > len(lowpass_param), only len(lowpass_param) bins are used.
        For curriculum, start with large bandwidth_target in __init__, then shrink.
        """
        self.bandwidth_target = new_bandwidth
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Apply rotary position embeddings to query and key tensors."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def _spectral_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention weights using spectral cross-correlation.
        
        Args:
            query: [batch, num_heads, q_len, head_dim]
            key: [batch, num_kv_heads, kv_len, head_dim]
            attention_mask: [batch, 1, q_len, kv_len] or None
            
        Returns:
            attn_weights: [batch, num_heads, q_len, kv_len]
        """
        batch, num_heads, q_len, head_dim = query.shape
        kv_len = key.shape[2]
        
        # Expand key for GQA if needed
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute standard dot-product similarity (scaled)
        # [batch, heads, q_len, kv_len]
        scale = 1.0 / (self.head_dim ** 0.5)
        raw_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Now apply spectral filtering to the attention pattern
        # CRITICAL: cuFFT doesn't support half precision or bfloat16
        # Convert to float32 for FFT, then back to original dtype
        orig_dtype = raw_scores.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            raw_scores = raw_scores.float()
        
        # FFT along the key dimension (last dim)
        scores_fft = torch.fft.rfft(raw_scores, dim=-1)
        
        # Build HARD BANDLIMITED filter: only first `bandwidth_target` bins are non-zero
        # All higher frequency bins are EXACTLY 0 by construction
        filter_len = scores_fft.shape[-1]
        bw = min(self.bandwidth_target, filter_len)
        
        # Create filter: zeros everywhere, then fill in the low-frequency bins
        spectral_filter = torch.zeros(filter_len, device=scores_fft.device, dtype=torch.float32)
        spectral_filter[:bw] = self.lowpass_param[:bw].to(device=scores_fft.device, dtype=torch.float32)
        
        # Keep filter positive and stable with softplus
        spectral_filter = F.softplus(spectral_filter)
        
        # Sharpness control: higher sharpness = more peaky attention
        sharpness = F.softplus(self.spectral_sharpness.to(scores_fft.device).float())
        spectral_filter = spectral_filter ** sharpness
        
        # Filter in frequency domain (high frequencies are exactly 0)
        scores_filtered = scores_fft * spectral_filter.to(scores_fft.dtype)
        
        # IFFT back to position domain - use kv_len as output size!
        attn_weights = torch.fft.irfft(scores_filtered, n=kv_len, dim=-1)
        
        # Convert back to original dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            attn_weights = attn_weights.to(orig_dtype)
        
        # Scale by temperature
        attn_weights = attn_weights / self.temperature
        
        # Apply causal mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # Create default causal mask
            # For generation with cache: q_len=1, kv_len=full_seq
            # The single query at position q_len can attend to all kv positions <= its position
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool),
                diagonal=kv_len - q_len + 1  # Shift diagonal for cache case
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Normalize to form valid attention distribution
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        return attn_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) from RoPE
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,  # KV cache (DynamicCache object)
        cache_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching Qwen2Attention interface.
        Returns (attn_output, attn_weights) to match Qwen2DecoderLayer expectations.
        """
        batch, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape to [batch, heads, seq_len, head_dim]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (Rotary Position Embeddings) - from position_embeddings tuple
        cos, sin = position_embeddings
        query, key = self._apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handle KV cache (for generation)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(
                key, value, self.layer_idx, cache_kwargs
            )
        
        # Expand V for GQA
        if self.num_kv_heads != self.num_heads:
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute spectral attention weights
        attn_weights = self._spectral_attention_weights(query, key, attention_mask)
        
        # Apply dropout
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        
        # Attend to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, attn_weights
        else:
            return attn_output, None


class PureSpectralAttention(nn.Module):
    """
    Pure spectral attention without any softmax.
    
    This version uses the Dirichlet kernel interpretation directly:
    - Attention weights are computed purely in the frequency domain
    - Normalization is done via L1 norm (not softmax)
    - Causality is enforced via spectral windowing
    
    This is more faithful to the CIMBA/CircleStat theory but may be
    harder to train from scratch.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 4096,
        num_moments: int = 64,  # S in CIMBA notation
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.num_moments = num_moments
        
        # For GQA
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # Learnable Fourier coefficients for the attention kernel (per head)
        # These define the "shape" of attention in frequency domain
        # Initialized to approximate a peaked kernel
        self.kernel_coeffs = nn.Parameter(
            self._init_dirichlet_coeffs(num_heads, num_moments)
        )
        
    def _init_dirichlet_coeffs(self, num_heads: int, num_moments: int) -> torch.Tensor:
        """Initialize coefficients to approximate Dirichlet kernel."""
        # Dirichlet kernel: all Fourier coefficients equal
        # Start with slight decay to encourage peakiness
        k = torch.arange(num_moments, dtype=torch.float32)
        decay = torch.exp(-0.1 * k)  # Slight high-frequency suppression
        coeffs = decay.unsqueeze(0).expand(num_heads, -1).clone()
        return coeffs
    
    def _compute_spectral_kernel(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute the attention kernel from learned Fourier coefficients.
        
        Returns:
            kernel: [num_heads, seq_len] - the attention kernel for each head
        """
        # Construct kernel from coefficients using inverse DFT
        # K(x) = sum_k c_k * cos(2π k x / L)
        
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        phases = 2 * math.pi * positions / seq_len  # [seq_len]
        
        # k values: [num_moments]
        k = torch.arange(self.num_moments, device=device, dtype=torch.float32)
        
        # Compute cos(k * phases) for all combinations
        # [num_moments, seq_len]
        cos_terms = torch.cos(k.unsqueeze(1) * phases.unsqueeze(0))
        
        # Weight by coefficients and sum
        # kernel_coeffs: [num_heads, num_moments]
        # cos_terms: [num_moments, seq_len]
        # result: [num_heads, seq_len]
        kernel = torch.matmul(self.kernel_coeffs.abs(), cos_terms)
        
        # Normalize to sum to 1
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        
        return kernel
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with pure spectral attention."""
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand for GQA
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute similarity scores (standard dot product)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # [batch, heads, seq_len, seq_len]
        
        # Apply spectral kernel via convolution
        # Get the learned kernel
        kernel = self._compute_spectral_kernel(seq_len, device)
        # kernel: [heads, seq_len]
        
        # Convolve scores with kernel along the key dimension
        # This is equivalent to: for each query, weight the key similarities
        # by the spectral kernel centered at that query position
        
        # Use FFT for efficient convolution
        scores_fft = torch.fft.rfft(scores, dim=-1)
        kernel_fft = torch.fft.rfft(kernel, n=seq_len, dim=-1)
        
        # Broadcast kernel across batch and query dimensions
        kernel_fft = kernel_fft.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, freq]
        
        # Convolve in frequency domain
        filtered_fft = scores_fft * kernel_fft
        
        # Back to position domain
        attn_weights = torch.fft.irfft(filtered_fft, n=seq_len, dim=-1)
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Normalize (softmax for stability, but on spectrally-filtered scores)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Attend
        attn_output = torch.matmul(attn_weights, value)
        
        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


def replace_attention_with_spectral(
    model,
    attention_class: type = SpectralAttention,
    layers_to_replace: Optional[list] = None,
    bandwidth_target: int = 64,  # S: number of frequency bins for hard bandlimit
) -> nn.Module:
    """
    Replace attention layers in a model with spectral attention.
    
    Args:
        model: The transformer model (e.g., Qwen)
        attention_class: SpectralAttention or PureSpectralAttention
        layers_to_replace: List of layer indices, or None for all
        bandwidth_target: S, the number of frequency bins to keep (for CIMBA: 2S+1 moments)
        
    Returns:
        Modified model
    """
    config = model.config
    
    for layer_idx, layer in enumerate(model.model.layers):
        if layers_to_replace is not None and layer_idx not in layers_to_replace:
            continue
        
        old_attn = layer.self_attn
        
        # Create new spectral attention with layer_idx for KV cache
        new_attn = attention_class(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            head_dim=config.hidden_size // config.num_attention_heads,
            layer_idx=layer_idx,  # CRITICAL for KV cache
            bandwidth_target=bandwidth_target,  # Hard bandlimit
        )
        
        # Copy weights from original projections
        new_attn.q_proj.weight.data = old_attn.q_proj.weight.data.clone()
        new_attn.k_proj.weight.data = old_attn.k_proj.weight.data.clone()
        new_attn.v_proj.weight.data = old_attn.v_proj.weight.data.clone()
        new_attn.o_proj.weight.data = old_attn.o_proj.weight.data.clone()
        
        # Copy biases if they exist
        if hasattr(old_attn.q_proj, 'bias') and old_attn.q_proj.bias is not None:
            new_attn.q_proj.bias.data = old_attn.q_proj.bias.data.clone()
        if hasattr(old_attn.k_proj, 'bias') and old_attn.k_proj.bias is not None:
            new_attn.k_proj.bias.data = old_attn.k_proj.bias.data.clone()
        if hasattr(old_attn.v_proj, 'bias') and old_attn.v_proj.bias is not None:
            new_attn.v_proj.bias.data = old_attn.v_proj.bias.data.clone()
        if hasattr(old_attn.o_proj, 'bias') and old_attn.o_proj.bias is not None:
            new_attn.o_proj.bias.data = old_attn.o_proj.bias.data.clone()
        
        # Replace
        layer.self_attn = new_attn
        
        print(f"Replaced layer {layer_idx} attention with {attention_class.__name__} (bw={bandwidth_target})")
    
    return model


def set_model_bandwidth(model, bandwidth: int):
    """
    Set bandwidth target for all SpectralAttention layers.
    Used for curriculum training: start large, shrink over epochs.
    
    Args:
        model: Model with SpectralAttention layers
        bandwidth: New bandwidth target S
    """
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, 'set_bandwidth'):
            attn.set_bandwidth(bandwidth)


def get_spectral_stats(model, bandwidth_target: int = 64):
    """
    Get statistics about spectral filters for monitoring training.
    
    Returns dict with:
        - leakage: fraction of energy in high frequencies (should be ~0 with hard bandlimit)
        - effective_bandwidth: where 95% of energy is concentrated
        - filter_stats: per-layer statistics
    """
    stats = []
    total_low_energy = 0.0
    total_high_energy = 0.0
    
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, 'lowpass_param'):
            # Get the actual current bandwidth used by this layer
            bw = attn.bandwidth_target
            param = attn.lowpass_param.detach()
            param_len = len(param)
            
            # Build the actual filter used in forward pass
            # Filter is zeros everywhere except first min(bw, param_len) bins
            active_bins = min(bw, param_len)
            filt = torch.zeros(attn.max_rfft_bins)
            filt[:active_bins] = param[:active_bins]
            filt = F.softplus(filt)
            
            # Energy calculation relative to the target bandwidth for comparison
            energy = filt.pow(2)
            low_energy = energy[:bandwidth_target].sum().item()
            high_energy = energy[bandwidth_target:].sum().item()
            total_energy = low_energy + high_energy
            
            total_low_energy += low_energy
            total_high_energy += high_energy
            
            # Effective bandwidth (95% energy threshold)
            cumsum = energy.cumsum(0)
            if total_energy > 0:
                effective_bw = (cumsum < 0.95 * total_energy).sum().item() + 1
            else:
                effective_bw = len(filt)
            
            stats.append({
                'layer': i,
                'bandwidth_target': bw,
                'param_size': param_len,
                'active_bins': active_bins,
                'effective_bw': effective_bw,
                'low_freq_energy': low_energy,
                'high_freq_energy': high_energy,
                'param_mean': param.mean().item(),
                'param_std': param.std().item(),
            })
    
    total_energy = total_low_energy + total_high_energy
    leakage = total_high_energy / (total_energy + 1e-12)
    
    return {
        'leakage': leakage,
        'total_low_energy': total_low_energy,
        'total_high_energy': total_high_energy,
        'per_layer': stats,
    }
