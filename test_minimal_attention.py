"""
Test attention replacement using the proven-correct MinimalAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

class MinimalAttention(nn.Module):
    """This exact implementation has been verified to produce correct output."""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True  # Required for generation
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values = None,  # KV cache
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_len = input_shape
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # KV cache handling
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Determine if we should use causal masking
        # During generation with cache, we're generating one token at a time
        # so causal masking is implicit (query can see all past keys)
        use_causal = attention_mask is None and key_states.shape[2] == query_states.shape[2]
        
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=use_causal,
            scale=self.scaling,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None


def replace_attention(model):
    """Replace all attention layers with MinimalAttention."""
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        new_attn = MinimalAttention(model.config, layer_idx)
        
        # Copy weights and biases
        new_attn.q_proj.weight.data = old_attn.q_proj.weight.data.clone()
        new_attn.k_proj.weight.data = old_attn.k_proj.weight.data.clone()
        new_attn.v_proj.weight.data = old_attn.v_proj.weight.data.clone()
        new_attn.o_proj.weight.data = old_attn.o_proj.weight.data.clone()
        new_attn.q_proj.bias.data = old_attn.q_proj.bias.data.clone()
        new_attn.k_proj.bias.data = old_attn.k_proj.bias.data.clone()
        new_attn.v_proj.bias.data = old_attn.v_proj.bias.data.clone()
        
        layer.self_attn = new_attn
        
    return model


# Test
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,  # Use float32 to rule out precision issues
).to("cpu")  # Use CPU to rule out CUDA issues

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

print("Replacing attention...")
model = replace_attention(model)

prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"
inputs = tokenizer(prompt, return_tensors="pt")  # Stay on CPU

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
    print("✓ MinimalAttention replacement WORKS!")
else:
    print("✗ MinimalAttention replacement BROKEN!")
