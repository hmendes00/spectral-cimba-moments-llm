"""Verify that RoPE is properly attached after attention replacement."""

import torch
from transformers import AutoModelForCausalLM
from spectral_attention import replace_attention_with_spectral, SpectralAttention

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cpu")

print("\nBefore replacement:")
print(f"  Layer 0 attention type: {type(model.model.layers[0].self_attn).__name__}")

print("\nReplacing attention layers...")
model = replace_attention_with_spectral(model, SpectralAttention)

print("\nAfter replacement:")
for i, layer in enumerate(model.model.layers[:3]):  # Check first 3 layers
    attn = layer.self_attn
    has_rope = attn.rotary_emb is not None
    rope_type = type(attn.rotary_emb).__name__ if has_rope else "None"
    print(f"  Layer {i}: {type(attn).__name__}, RoPE: {has_rope} ({rope_type})")

# Verify all layers share the same RoPE instance
rope_ids = [id(layer.self_attn.rotary_emb) for layer in model.model.layers]
all_same = len(set(rope_ids)) == 1 and rope_ids[0] != id(None)
print(f"\nAll layers share same RoPE instance: {all_same}")

if all_same:
    print("✓ RoPE fix successful!")
else:
    print("✗ RoPE fix FAILED - layers don't share RoPE")
