"""
Debug SDPA vs manual attention.
"""

import torch
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

# Get attention params
attn = model.model.layers[0].self_attn
head_dim = attn.head_dim
num_heads = attn.config.num_attention_heads
num_kv_heads = attn.config.num_key_value_heads
num_groups = num_heads // num_kv_heads
scaling = head_dim ** -0.5

print(f"head_dim: {head_dim}, num_heads: {num_heads}, num_kv_heads: {num_kv_heads}")
print(f"scaling: {scaling}")

with torch.no_grad():
    # Project Q, K, V
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)
    
    q = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    k = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    
    print(f"\nAfter projection:")
    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    
    # Apply RoPE
    cos, sin = position_embeddings
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"\nAfter RoPE:")
    print(f"Q sample: {q_rope[0, 0, 0, :5]}")
    print(f"K sample: {k_rope[0, 0, 0, :5]}")
    
    # Expand for GQA
    k_expanded = repeat_kv(k_rope, num_groups)
    v_expanded = repeat_kv(v, num_groups)
    
    print(f"\nAfter GQA expand:")
    print(f"K shape: {k_expanded.shape}")
    print(f"V shape: {v_expanded.shape}")
    
    # Method 1: SDPA
    print("\n" + "="*60)
    print("SDPA OUTPUT")
    sdpa_out = F.scaled_dot_product_attention(
        q_rope, k_expanded, v_expanded,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scaling,
    )
    print(f"SDPA output shape: {sdpa_out.shape}")
    print(f"SDPA output sample: {sdpa_out[0, 0, 0, :5]}")
    
    # Method 2: Manual
    print("\n" + "="*60)
    print("MANUAL OUTPUT")
    scores = torch.matmul(q_rope, k_expanded.transpose(2, 3)) * scaling
    print(f"Scores shape: {scores.shape}")
    print(f"Scores sample: {scores[0, 0, 0, :]}")
    
    # Causal mask
    seq_len = q_rope.shape[2]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores_masked = scores.masked_fill(causal_mask, float('-inf'))
    print(f"Scores masked: {scores_masked[0, 0, 0, :]}")
    
    attn_weights = F.softmax(scores_masked, dim=-1, dtype=torch.float32)
    print(f"Attention weights: {attn_weights[0, 0, 0, :]}")
    
    manual_out = torch.matmul(attn_weights, v_expanded)
    print(f"Manual output sample: {manual_out[0, 0, 0, :5]}")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    diff = (sdpa_out - manual_out).abs()
    print(f"Max diff: {diff.max():.8f}")
    
    if diff.max() < 1e-5:
        print("✓ SDPA matches manual")
    else:
        print("✗ SDPA differs from manual!")
