"""
Test: Does replacing attention module at all break the model?
Create a "passthrough" attention that mimics standard attention exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

class StandardAttentionReplacement(nn.Module):
    """
    Exact reimplementation of standard attention.
    If this works, the problem is in spectral filtering.
    If this breaks, the problem is in how we replace the module.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx
        
        # Projections - MUST match original (Q, K, V have bias, O doesn't)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE: will be copied from original
        self.rotary_emb = None
        
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings."""
        # cos, sin: [batch, seq_len, head_dim]
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape to [batch, heads, seq_len, head_dim]
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE using EXACT transformers function
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Handle KV cache (skip for now - just testing)
        # if past_key_values is not None:
        #     key, value = past_key_values.update(key, value, self.layer_idx, {"cache_position": cache_position})
        
        # Expand K, V for GQA using EXACT transformers function
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        
        # Use PyTorch's scaled_dot_product_attention with built-in causal masking
        # This is what SDPA implementation uses
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,  # Use causal mask if no explicit mask
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        # Return format: (output, attention_weights_or_None)
        return attn_output, None


def replace_with_standard(model):
    """Replace attention with our standard reimplementation."""
    config = model.config
    
    model_rotary_emb = getattr(model.model, 'rotary_emb', None)
    
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        new_attn = StandardAttentionReplacement(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            head_dim=config.hidden_size // config.num_attention_heads,
            layer_idx=layer_idx,
        )
        
        # Copy weights AND biases
        new_attn.q_proj.weight.data = old_attn.q_proj.weight.data.clone()
        new_attn.k_proj.weight.data = old_attn.k_proj.weight.data.clone()
        new_attn.v_proj.weight.data = old_attn.v_proj.weight.data.clone()
        new_attn.o_proj.weight.data = old_attn.o_proj.weight.data.clone()
        
        # Copy biases for Q, K, V (they have biases)
        if old_attn.q_proj.bias is not None:
            new_attn.q_proj.bias.data = old_attn.q_proj.bias.data.clone()
        if old_attn.k_proj.bias is not None:
            new_attn.k_proj.bias.data = old_attn.k_proj.bias.data.clone()
        if old_attn.v_proj.bias is not None:
            new_attn.v_proj.bias.data = old_attn.v_proj.bias.data.clone()
        
        # Copy RoPE
        if model_rotary_emb is not None:
            new_attn.rotary_emb = model_rotary_emb
        
        layer.self_attn = new_attn
    
    return model


# Test
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")

print("Replacing attention with StandardAttentionReplacement...")
model = replace_with_standard(model)

prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"\nPrompt: {prompt}\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated:\n{generated}")

print("\n" + "="*60)
if "return" in generated and ("==" in generated or "[::" in generated):
    print("✓ StandardAttentionReplacement WORKS!")
    print("  -> The problem is in spectral filtering, not in module replacement")
else:
    print("✗ StandardAttentionReplacement BROKEN!")
    print("  -> The problem is in how we replace the module")
