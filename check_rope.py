"""Check how Qwen2 implements RoPE."""
import torch
from transformers import AutoModelForCausalLM

print("Loading model...")
m = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-Coder-0.5B-Instruct',
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

attn = m.model.layers[0].self_attn
print(f"\nAttention type: {type(attn).__name__}")

print(f"\nAttention has 'rotary_emb': {hasattr(attn, 'rotary_emb')}")
print(f"Model has 'rotary_emb': {hasattr(m.model, 'rotary_emb')}")

if hasattr(m.model, 'rotary_emb'):
    print(f"model.rotary_emb type: {type(m.model.rotary_emb).__name__}")
    
if hasattr(attn, 'rotary_emb'):
    print(f"attn.rotary_emb type: {type(attn.rotary_emb).__name__}")
