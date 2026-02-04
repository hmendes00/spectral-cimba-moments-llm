"""
Test single forward pass (not generation) with replaced attention.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from test_minimal_attention import MinimalAttention, replace_attention

print("Loading model...")
model1 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,
).to("cpu")

model2 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

print("Replacing attention in model2...")
model2 = replace_attention(model2)

# Test single forward pass
inputs = tokenizer("def hello():", return_tensors="pt")

print("\nRunning single forward pass...")

with torch.no_grad():
    # Get logits from both models
    out1 = model1(**inputs)
    out2 = model2(**inputs)
    
    logits1 = out1.logits
    logits2 = out2.logits
    
    print(f"Model1 logits shape: {logits1.shape}")
    print(f"Model2 logits shape: {logits2.shape}")
    
    # Compare last token logits (the prediction)
    last_logits1 = logits1[0, -1, :]
    last_logits2 = logits2[0, -1, :]
    
    # Get top predictions
    top1 = torch.topk(last_logits1, 5)
    top2 = torch.topk(last_logits2, 5)
    
    print(f"\nModel1 top 5 tokens: {[tokenizer.decode([t]) for t in top1.indices]}")
    print(f"Model2 top 5 tokens: {[tokenizer.decode([t]) for t in top2.indices]}")
    
    # Compare
    diff = (logits1 - logits2).abs()
    print(f"\nMax logit difference: {diff.max():.6f}")
    print(f"Mean logit difference: {diff.mean():.6f}")
    
    if diff.max() < 0.01:
        print("✓ Single forward pass MATCHES")
    else:
        print("✗ Single forward pass DIFFERS")
