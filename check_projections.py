"""
Check if projection outputs match between original and new.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
import torch.nn as nn

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Get inputs
inputs = tokenizer("def hello():", return_tensors="pt")
hidden = model.model.embed_tokens(inputs["input_ids"])

# Original attention
attn = model.model.layers[0].self_attn

# Create new projections and copy weights
q_proj_new = nn.Linear(model.config.hidden_size, model.config.num_attention_heads * 64, bias=True)
k_proj_new = nn.Linear(model.config.hidden_size, model.config.num_key_value_heads * 64, bias=True)

q_proj_new.weight.data = attn.q_proj.weight.data.clone()
q_proj_new.bias.data = attn.q_proj.bias.data.clone()
k_proj_new.weight.data = attn.k_proj.weight.data.clone()
k_proj_new.bias.data = attn.k_proj.bias.data.clone()

# Compare outputs
with torch.no_grad():
    q_orig = attn.q_proj(hidden)
    q_new = q_proj_new(hidden)
    
    k_orig = attn.k_proj(hidden)
    k_new = k_proj_new(hidden)
    
    print(f"Q projection output match: {torch.allclose(q_orig, q_new)}")
    print(f"  Max diff: {(q_orig - q_new).abs().max():.10f}")
    
    print(f"K projection output match: {torch.allclose(k_orig, k_new)}")
    print(f"  Max diff: {(k_orig - k_new).abs().max():.10f}")
    
    # Check shapes
    print(f"\nQ orig shape: {q_orig.shape}")
    print(f"K orig shape: {k_orig.shape}")
    
    # Check weight shapes
    print(f"\nOriginal q_proj weight shape: {attn.q_proj.weight.shape}")
    print(f"New q_proj weight shape: {q_proj_new.weight.shape}")
    print(f"Original k_proj weight shape: {attn.k_proj.weight.shape}")
    print(f"New k_proj weight shape: {k_proj_new.weight.shape}")
