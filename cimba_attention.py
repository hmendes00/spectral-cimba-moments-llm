"""
CIMBA-Integrated Spectral Attention

This module provides a SpectralAttention implementation that uses CIMBA's O(S) 
moment-based KV cache instead of storing full keys/values.

Key Design:
- During prefill: Compute attention normally, but store V in CIMBA moments
- During generation: Use CIMBA cache to reconstruct attention output directly

The CIMBA approach assumes attention weights follow a bandlimited Dirichlet kernel
based on position. This is exactly what our trained SpectralAttention learns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from cimba_cache import CIMBACache


class CIMBAAttention(nn.Module):
    """
    Spectral Attention with CIMBA O(S) cache integration.
    
    This attention module replaces the standard KV cache with CIMBA's
    moment-based cache. Key features:
    
    1. O(S) memory: Instead of O(N) for full KV cache
    2. Exact reconstruction: For bandlimited Dirichlet attention
    3. Fast generation: Only need to update S moments per token
    
    The trade-off: We use position-based attention (Dirichlet kernel)
    rather than content-based attention (Q·K). For tasks that benefit
    from spectral structure, this can be advantageous.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        layer_idx: int = 0,
        bandwidth: int = 64,  # S: number of frequency components
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.attention_dropout = attention_dropout
        self.layer_idx = layer_idx
        self.is_causal = True
        self.max_seq_len = max_position_embeddings
        
        # CIMBA bandwidth
        self.S = bandwidth
        self.num_moments = 2 * bandwidth + 1  # k ∈ [-S, S]
        
        # For GQA
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections (will be copied from original attention)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE: will be copied from original attention
        self.rotary_emb = None
        
        # Pre-compute theta scale for positions
        self.theta_scale = 2 * math.pi / max_position_embeddings
        
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Apply rotary position embeddings."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _dirichlet_weights(
        self,
        query_positions: torch.Tensor,  # [q_len]
        key_positions: torch.Tensor,  # [kv_len]
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Compute Dirichlet kernel attention weights.
        
        D_S(x) = sin((S + 0.5)x) / sin(x/2) / (2S+1)
        
        Returns: [q_len, kv_len] attention weight matrix
        """
        device = query_positions.device
        
        # Convert positions to angles
        theta_q = query_positions.float() * self.theta_scale  # [q_len]
        theta_k = key_positions.float() * self.theta_scale  # [kv_len]
        
        # Compute position differences
        delta = theta_q.unsqueeze(1) - theta_k.unsqueeze(0)  # [q_len, kv_len]
        
        # Dirichlet kernel with numerical stability
        x = delta + 1e-10
        weights = torch.sin((self.S + 0.5) * x) / torch.sin(x / 2)
        weights = weights / (2 * self.S + 1)
        
        # Apply causal mask - VECTORIZED (no Python loops!)
        if causal:
            # Create causal mask: position j can only attend to position i if i <= j
            # query_positions[i] >= key_positions[j] for valid attention
            causal_mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)
            weights = weights * causal_mask.float()
            
            # Renormalize rows
            row_sums = weights.sum(dim=-1, keepdim=True)
            weights = weights / (row_sums + 1e-10)
        
        return weights
    
    def forward_standard(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Standard forward pass with Dirichlet attention (no CIMBA cache).
        Uses full attention matrix computation.
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (we keep K projections for potential future use, but Dirichlet doesn't need them)
        cos, sin = position_embeddings
        query, key = self._apply_rotary_pos_emb(query, key, cos, sin)
        
        # Get positions from cache_position
        if cache_position is not None:
            positions = cache_position
        else:
            positions = torch.arange(seq_len, device=device)
        
        # Handle past_key_values (standard DynamicCache for comparison)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key, value = past_key_values.update(key, value, self.layer_idx, cache_kwargs)
            all_positions = torch.arange(key.shape[2], device=device)
        else:
            all_positions = positions
        
        kv_len = all_positions.shape[0]
        
        # Compute Dirichlet attention weights
        # [q_len, kv_len] - same for all heads (position-based)
        attn_weights = self._dirichlet_weights(positions, all_positions, causal=True)
        
        # Apply attention_mask to mask out padding tokens
        # This is CRITICAL - without this, model attends to padding during training!
        if attention_mask is not None:
            # attention_mask from HF can be:
            # - [batch, kv_len] with 1=valid, 0=pad
            # - [batch, 1, q_len, kv_len] additive mask with 0=valid, -inf=pad
            if attention_mask.dim() == 2:
                # [batch, kv_len] -> take first batch item, convert to bool
                key_padding_mask = attention_mask[0, :kv_len].bool()  # [kv_len]
            elif attention_mask.dim() == 4:
                # [batch, 1, q_len, kv_len] additive mask -> valid positions have 0
                # Take the mask for the first query position (all rows should be same for padding)
                key_padding_mask = (attention_mask[0, 0, 0, :kv_len] > -1e9)  # [kv_len]
            else:
                key_padding_mask = None
            
            if key_padding_mask is not None:
                # Mask out padding keys and renormalize
                attn_weights = attn_weights * key_padding_mask.unsqueeze(0).float()
                row_sums = attn_weights.sum(dim=-1, keepdim=True)
                attn_weights = attn_weights / (row_sums + 1e-10)
        
        # Expand for batch and heads
        attn_weights = attn_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
        attn_weights = attn_weights.expand(batch, self.num_heads, -1, -1)
        
        # Expand V for GQA
        if self.num_kv_heads != self.num_heads:
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Attend to values
        attn_output = torch.matmul(attn_weights.to(value.dtype), value)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None
    
    def forward_cimba(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cimba_cache: CIMBACache,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass using CIMBA O(S) cache.
        
        Key insight: For Dirichlet attention, the output at position t is:
            o_t = Σ_j D_S(θ_t - θ_j) v_j / Σ_j D_S(θ_t - θ_j)
            
        This can be computed directly from CIMBA's V moments without
        storing the full V sequence!
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Only batch=1 supported for CIMBA generation
        assert batch == 1, "CIMBA forward only supports batch_size=1"
        
        # Project to V (Q and K not needed for pure Dirichlet attention)
        value = self.v_proj(hidden_states)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = value.squeeze(0)  # [seq_len, num_kv_heads, head_dim]
        
        # Get positions
        if cache_position is not None:
            positions = cache_position
        else:
            positions = torch.arange(seq_len, device=device)
        
        # Determine if this is prefill or generation
        if seq_len > 1:
            # Prefill: use causal batch reconstruction
            attn_output = cimba_cache.reconstruct_batch_causal(
                layer_idx=self.layer_idx,
                v_batch=value.float(),
                position_ids=positions,
                normalize=True,
            )
        else:
            # Generation: single token
            # Update cache with new V
            pos_id = positions[0].item()
            cimba_cache.update(
                layer_idx=self.layer_idx,
                v_new=value[0].float(),
                position_id=pos_id,
            )
            
            # Reconstruct output for this position
            attn_output = cimba_cache.reconstruct(
                layer_idx=self.layer_idx,
                query_position=pos_id,
                normalize=True,
            )
            attn_output = attn_output.unsqueeze(0)  # [1, num_kv_heads, head_dim]
        
        # Expand for GQA: [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
        if self.num_kv_heads != self.num_heads:
            attn_output = attn_output.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Reshape: [seq_len, num_heads, head_dim] -> [batch, seq_len, hidden]
        attn_output = attn_output.view(1, seq_len, -1)
        
        # Project output
        attn_output = self.o_proj(attn_output.to(hidden_states.dtype))
        
        return attn_output, None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        cache_position: Optional[torch.Tensor] = None,
        cimba_cache: Optional[CIMBACache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass with automatic routing to standard or CIMBA mode.
        
        Args:
            hidden_states: Input tensor
            position_embeddings: (cos, sin) from RoPE
            attention_mask: Standard attention mask
            past_key_values: Standard KV cache (DynamicCache) - used if cimba_cache is None
            cache_position: Position indices
            cimba_cache: If provided, use CIMBA O(S) cache instead of standard cache
        """
        if cimba_cache is not None:
            return self.forward_cimba(
                hidden_states, position_embeddings, cimba_cache, cache_position, **kwargs
            )
        else:
            return self.forward_standard(
                hidden_states, position_embeddings, attention_mask, 
                past_key_values, cache_position, **kwargs
            )


def create_cimba_cache_for_model(
    model,
    bandwidth: int = 64,
    max_seq_len: int = 131072,
    device: str = 'cuda',
) -> CIMBACache:
    """
    Create a CIMBACache sized for a given model.
    
    Args:
        model: The model (needs to have model.config.num_hidden_layers, etc.)
        bandwidth: S parameter (controls quality vs memory)
        max_seq_len: Maximum sequence length
        device: Device for cache tensors
        
    Returns:
        CIMBACache configured for the model
    """
    config = model.config
    
    return CIMBACache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        S=bandwidth,
        device=device,
        max_seq_len=max_seq_len,
    )


def replace_attention_with_cimba(model, bandwidth: int = 64):
    """
    Replace all attention layers with CIMBAAttention.
    
    Args:
        model: The model to modify
        bandwidth: S parameter for CIMBA
        
    Returns:
        Modified model with CIMBAAttention layers
    """
    config = model.config
    
    # Get device from existing model parameters
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        new_attn = CIMBAAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            bandwidth=bandwidth,
        )
        
        # Copy weights
        new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
        new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
        new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
        new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
        
        # Copy rotary embedding if available (different attribute names across versions)
        if hasattr(old_attn, 'rotary_emb'):
            new_attn.rotary_emb = old_attn.rotary_emb
        elif hasattr(old_attn, 'rotary_fn'):
            new_attn.rotary_emb = old_attn.rotary_fn
        
        # Move to correct device and dtype
        new_attn = new_attn.to(device=device, dtype=dtype)
        
        layer.self_attn = new_attn
        print(f"Replaced layer {layer_idx} with CIMBAAttention (S={bandwidth})")
    
    return model


def test_cimba_attention():
    """Test CIMBAAttention matches direct computation."""
    print("Testing CIMBAAttention...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a small attention module
    attn = CIMBAAttention(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        max_position_embeddings=4096,
        layer_idx=0,
        bandwidth=32,
    ).to(device)
    
    # Create CIMBA cache
    cache = CIMBACache(
        num_layers=1,
        num_kv_heads=2,
        head_dim=32,
        S=32,
        device=device,
        max_seq_len=4096,
    )
    
    # Test input
    batch = 1
    seq_len = 16
    hidden_states = torch.randn(batch, seq_len, 128, device=device)
    
    # Mock position embeddings
    cos = torch.ones(batch, seq_len, 32, device=device)
    sin = torch.zeros(batch, seq_len, 32, device=device)
    position_embeddings = (cos, sin)
    cache_position = torch.arange(seq_len, device=device)
    
    # Run standard forward
    out_standard, _ = attn.forward_standard(
        hidden_states, position_embeddings, cache_position=cache_position
    )
    
    # Run CIMBA forward
    out_cimba, _ = attn.forward_cimba(
        hidden_states, position_embeddings, cache, cache_position=cache_position
    )
    
    # Compare
    max_err = (out_standard - out_cimba).abs().max().item()
    mean_err = (out_standard - out_cimba).abs().mean().item()
    
    print(f"Max error: {max_err:.2e}")
    print(f"Mean error: {mean_err:.2e}")
    
    if max_err < 1e-4:
        print("✓ CIMBAAttention passes!")
    else:
        print("✗ CIMBAAttention failed - error too high")
    
    return max_err < 1e-4


if __name__ == "__main__":
    test_cimba_attention()
