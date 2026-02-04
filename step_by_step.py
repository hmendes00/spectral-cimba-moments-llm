"""
Step-by-step comparison of attention flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Get inputs
inputs = tokenizer("def hello():", return_tensors="pt")
hidden = model.model.embed_tokens(inputs["input_ids"])
position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
position_embeddings = model.model.rotary_emb(hidden, position_ids)
cos, sin = position_embeddings

# Original attention
attn = model.model.layers[0].self_attn
head_dim = attn.head_dim
num_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
num_groups = num_heads // num_kv_heads
scaling = head_dim ** -0.5

print(f"Config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
print(f"Hidden shape: {hidden.shape}")

# Create new layers exactly like MinimalAttention
class MinimalAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

new_attn = MinimalAttention(model.config, 0)
new_attn.q_proj.weight.data = attn.q_proj.weight.data.clone()
new_attn.k_proj.weight.data = attn.k_proj.weight.data.clone()
new_attn.v_proj.weight.data = attn.v_proj.weight.data.clone()
new_attn.o_proj.weight.data = attn.o_proj.weight.data.clone()
new_attn.q_proj.bias.data = attn.q_proj.bias.data.clone()
new_attn.k_proj.bias.data = attn.k_proj.bias.data.clone()
new_attn.v_proj.bias.data = attn.v_proj.bias.data.clone()

with torch.no_grad():
    print("\n" + "="*60)
    print("STEP 1: Projections")
    print("="*60)
    
    # Original
    q_orig = attn.q_proj(hidden)
    k_orig = attn.k_proj(hidden)
    v_orig = attn.v_proj(hidden)
    
    # New
    q_new = new_attn.q_proj(hidden)
    k_new = new_attn.k_proj(hidden)
    v_new = new_attn.v_proj(hidden)
    
    print(f"Q match: {torch.allclose(q_orig, q_new)} (max diff: {(q_orig-q_new).abs().max():.10f})")
    print(f"K match: {torch.allclose(k_orig, k_new)} (max diff: {(k_orig-k_new).abs().max():.10f})")
    print(f"V match: {torch.allclose(v_orig, v_new)} (max diff: {(v_orig-v_new).abs().max():.10f})")
    
    print("\n" + "="*60)
    print("STEP 2: Reshape")
    print("="*60)
    
    # Original's reshape
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    q_orig_r = q_orig.view(hidden_shape).transpose(1, 2)
    k_orig_r = k_orig.view(hidden_shape).transpose(1, 2)
    v_orig_r = v_orig.view(hidden_shape).transpose(1, 2)
    
    # New reshape (correct for GQA)
    q_shape = (*input_shape, num_heads, head_dim)
    kv_shape = (*input_shape, num_kv_heads, head_dim)
    q_new_r = q_new.view(q_shape).transpose(1, 2)
    k_new_r = k_new.view(kv_shape).transpose(1, 2)
    v_new_r = v_new.view(kv_shape).transpose(1, 2)
    
    print(f"Original Q shape after reshape: {q_orig_r.shape}")
    print(f"New Q shape after reshape: {q_new_r.shape}")
    print(f"Original K shape after reshape: {k_orig_r.shape}")
    print(f"New K shape after reshape: {k_new_r.shape}")
    
    print(f"\nQ reshape match: {torch.allclose(q_orig_r, q_new_r)}")
    print(f"K reshape match: {torch.allclose(k_orig_r, k_new_r)}")
    print(f"V reshape match: {torch.allclose(v_orig_r, v_new_r)}")
    
    print("\n" + "="*60)
    print("STEP 3: RoPE")
    print("="*60)
    
    q_orig_rope, k_orig_rope = apply_rotary_pos_emb(q_orig_r, k_orig_r, cos, sin)
    q_new_rope, k_new_rope = apply_rotary_pos_emb(q_new_r, k_new_r, cos, sin)
    
    print(f"Q RoPE match: {torch.allclose(q_orig_rope, q_new_rope)}")
    print(f"K RoPE match: {torch.allclose(k_orig_rope, k_new_rope)}")
    
    print("\n" + "="*60)
    print("STEP 4: GQA Expansion")
    print("="*60)
    
    k_orig_exp = repeat_kv(k_orig_rope, num_groups)
    v_orig_exp = repeat_kv(v_orig_r, num_groups)
    k_new_exp = repeat_kv(k_new_rope, num_groups)
    v_new_exp = repeat_kv(v_new_r, num_groups)
    
    print(f"K exp shape orig: {k_orig_exp.shape}, new: {k_new_exp.shape}")
    print(f"K expand match: {torch.allclose(k_orig_exp, k_new_exp)}")
    print(f"V expand match: {torch.allclose(v_orig_exp, v_new_exp)}")
    
    print("\n" + "="*60)
    print("STEP 5: SDPA")
    print("="*60)
    
    out_orig = F.scaled_dot_product_attention(
        q_orig_rope, k_orig_exp, v_orig_exp,
        attn_mask=None, dropout_p=0.0, is_causal=True, scale=scaling
    )
    out_new = F.scaled_dot_product_attention(
        q_new_rope, k_new_exp, v_new_exp,
        attn_mask=None, dropout_p=0.0, is_causal=True, scale=scaling
    )
    
    print(f"SDPA output shapes: orig={out_orig.shape}, new={out_new.shape}")
    print(f"SDPA match: {torch.allclose(out_orig, out_new)} (max diff: {(out_orig-out_new).abs().max():.10f})")
    
    print("\n" + "="*60)
    print("STEP 6: Reshape back & O projection")
    print("="*60)
    
    # Original reshape
    out_orig_r = out_orig.transpose(1, 2).contiguous()
    out_orig_r = out_orig_r.reshape(*input_shape, -1)
    
    # New reshape
    out_new_r = out_new.transpose(1, 2).contiguous()
    out_new_r = out_new_r.reshape(*input_shape, -1)
    
    print(f"Reshaped orig: {out_orig_r.shape}, new: {out_new_r.shape}")
    print(f"Reshape match: {torch.allclose(out_orig_r, out_new_r)}")
    
    # O projection
    final_orig = attn.o_proj(out_orig_r)
    final_new = new_attn.o_proj(out_new_r)
    
    print(f"\nFinal output orig: {final_orig[0,0,:5]}")
    print(f"Final output new: {final_new[0,0,:5]}")
    print(f"Final match: {torch.allclose(final_orig, final_new)} (max diff: {(final_orig-final_new).abs().max():.10f})")
    
    # Now compare to the ACTUAL original forward
    print("\n" + "="*60)
    print("COMPARISON WITH MinimalAttention FORWARD")
    print("="*60)
    
    # Now test MinimalAttention's forward method
    class MinimalAttention2(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.head_dim = config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_kv_heads
            self.scaling = self.head_dim ** -0.5
            
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
            
        def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            q_shape = (*input_shape, self.num_heads, self.head_dim)
            kv_shape = (*input_shape, self.num_kv_heads, self.head_dim)
            
            query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
            
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None,
                scale=self.scaling,
            )
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)
            
            return attn_output, None
    
    new_attn2 = MinimalAttention2(model.config, 0)
    new_attn2.q_proj.weight.data = attn.q_proj.weight.data.clone()
    new_attn2.k_proj.weight.data = attn.k_proj.weight.data.clone()
    new_attn2.v_proj.weight.data = attn.v_proj.weight.data.clone()
    new_attn2.o_proj.weight.data = attn.o_proj.weight.data.clone()
    new_attn2.q_proj.bias.data = attn.q_proj.bias.data.clone()
    new_attn2.k_proj.bias.data = attn.k_proj.bias.data.clone()
    new_attn2.v_proj.bias.data = attn.v_proj.bias.data.clone()
    
    minimal_out, _ = new_attn2(
        hidden_states=hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )
    
    print(f"MinimalAttention output: {minimal_out[0,0,:5]}")
    print(f"Expected output: {final_orig[0,0,:5]}")
    print(f"Match: {torch.allclose(final_orig, minimal_out)} (max diff: {(final_orig-minimal_out).abs().max():.10f})")
