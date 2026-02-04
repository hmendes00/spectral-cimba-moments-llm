"""
Minimal test: Just replace one layer and check output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

class MinimalAttention(nn.Module):
    """Minimal attention that should produce identical output to Qwen2Attention."""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        # Create projections matching Qwen2
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # DEBUG: Check shapes before reshape
        print(f"  Q after proj: {query_states.shape}")
        print(f"  K after proj: {key_states.shape}")
        print(f"  V after proj: {value_states.shape}")
        
        input_shape = hidden_states.shape[:-1]
        # The hidden_shape calculation is WRONG for K, V which have fewer heads!
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)
        
        query_states = query_states.view(q_shape).transpose(1, 2)
        key_states = key_states.view(kv_shape).transpose(1, 2)
        value_states = value_states.view(kv_shape).transpose(1, 2)
        
        print(f"  Q after reshape: {query_states.shape}")
        print(f"  K after reshape: {key_states.shape}")
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Use SDPA with causal mask
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,
            scale=self.scaling,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None


# Test
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,  # Use float32 for precise comparison
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Get layer 0 attention
old_attn = model.model.layers[0].self_attn

# Create replacement
new_attn = MinimalAttention(model.config, layer_idx=0)

# Copy weights exactly
new_attn.q_proj.weight.data = old_attn.q_proj.weight.data.clone()
new_attn.k_proj.weight.data = old_attn.k_proj.weight.data.clone()
new_attn.v_proj.weight.data = old_attn.v_proj.weight.data.clone()
new_attn.o_proj.weight.data = old_attn.o_proj.weight.data.clone()
new_attn.q_proj.bias.data = old_attn.q_proj.bias.data.clone()
new_attn.k_proj.bias.data = old_attn.k_proj.bias.data.clone()
new_attn.v_proj.bias.data = old_attn.v_proj.bias.data.clone()

# Test on same input
inputs = tokenizer("def hello():", return_tensors="pt")
hidden = model.model.embed_tokens(inputs["input_ids"])
position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
position_embeddings = model.model.rotary_emb(hidden, position_ids)

print(f"\nInput shape: {hidden.shape}")

# Run original
with torch.no_grad():
    original_out, _ = old_attn(
        hidden_states=hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )
    print(f"Original output: {original_out[0, 0, :5]}")

# Run replacement
with torch.no_grad():
    new_out, _ = new_attn(
        hidden_states=hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )
    print(f"New output: {new_out[0, 0, :5]}")

# Compare
diff = (original_out - new_out).abs()
print(f"\nMax difference: {diff.max():.8f}")
print(f"Mean difference: {diff.mean():.8f}")

if diff.max() < 1e-5:
    print("✓ Outputs MATCH!")
else:
    print("✗ Outputs differ")
    print(f"\nFull comparison:")
    print(f"Original: {original_out[0, 0, :10]}")
    print(f"New: {new_out[0, 0, :10]}")
