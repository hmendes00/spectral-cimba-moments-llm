"""
Compare attention outputs between original and replacement.
This will help identify exactly where the discrepancy occurs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,  # Use float32 for precise comparison
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Simple test input
inputs = tokenizer("def hello():", return_tensors="pt")
input_ids = inputs["input_ids"]
print(f"Input: {tokenizer.decode(input_ids[0])}")
print(f"Shape: {input_ids.shape}")

# Get embeddings
with torch.no_grad():
    hidden = model.model.embed_tokens(input_ids)
    print(f"Hidden shape: {hidden.shape}")

# Get RoPE
position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
cos, sin = model.model.rotary_emb(hidden, position_ids)
position_embeddings = (cos, sin)
print(f"RoPE cos shape: {cos.shape}, sin shape: {sin.shape}")

# Test on layer 0
layer = model.model.layers[0]
attn = layer.self_attn

print("\n" + "="*60)
print("COMPARING ATTENTION OUTPUTS")
print("="*60)

# Run original attention
with torch.no_grad():
    original_output, _ = attn(
        hidden_states=hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )
    print(f"\nOriginal output shape: {original_output.shape}")
    print(f"Original output sample: {original_output[0, 0, :5]}")

# Now manually compute what the attention SHOULD do
with torch.no_grad():
    # Project
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    v = attn.v_proj(hidden)
    
    print(f"\nQ shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")
    
    # Reshape
    batch, seq_len = input_ids.shape
    head_dim = attn.head_dim
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    print(f"\nAfter reshape:")
    print(f"Q shape: {q.shape}  (batch, heads, seq, head_dim)")
    print(f"K shape: {k.shape}  (batch, kv_heads, seq, head_dim)")
    print(f"V shape: {v.shape}  (batch, kv_heads, seq, head_dim)")
    
    # Apply RoPE
    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"\nAfter RoPE:")
    print(f"Q_rope sample: {q_rope[0, 0, 0, :5]}")
    print(f"K_rope sample: {k_rope[0, 0, 0, :5]}")
    
    # Repeat KV
    k_repeated = repeat_kv(k_rope, num_heads // num_kv_heads)
    v_repeated = repeat_kv(v, num_heads // num_kv_heads)
    print(f"\nAfter GQA expansion:")
    print(f"K shape: {k_repeated.shape}")
    print(f"V shape: {v_repeated.shape}")
    
    # Compute attention
    scale = head_dim ** -0.5
    scores = torch.matmul(q_rope, k_repeated.transpose(2, 3)) * scale
    print(f"\nAttention scores shape: {scores.shape}")
    print(f"Scores sample (first row): {scores[0, 0, 0, :]}")
    
    # Softmax WITH causal mask (Qwen uses causal attention)
    seq_len_kv = k_repeated.shape[2]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len_kv, dtype=torch.bool), diagonal=1)
    scores_masked = scores.masked_fill(causal_mask, float('-inf'))
    attn_weights = torch.softmax(scores_masked, dim=-1, dtype=torch.float32)
    print(f"Attention weights sample (first row): {attn_weights[0, 0, 0, :]}")
    
    # Apply to values
    output = torch.matmul(attn_weights, v_repeated)
    print(f"\nOutput after attention shape: {output.shape}")
    
    # Reshape back
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch, seq_len, -1)
    
    # Project
    manual_output = attn.o_proj(output)
    print(f"Manual output shape: {manual_output.shape}")
    print(f"Manual output sample: {manual_output[0, 0, :5]}")
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    diff = (original_output - manual_output).abs()
    print(f"Max difference: {diff.max():.6f}")
    print(f"Mean difference: {diff.mean():.6f}")
    
    if diff.max() < 1e-4:
        print("✓ Outputs MATCH - manual implementation is correct")
    else:
        print("✗ Outputs DO NOT MATCH - there's a bug")
        
        # Find where the difference is
        print(f"\nOriginal first token output: {original_output[0, 0, :10]}")
        print(f"Manual first token output: {manual_output[0, 0, :10]}")
