"""Quick check: are gradients flowing to coeff_proj?"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float32,
    device_map=DEVICE,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Replace with IWM
from iwm_attention import replace_attention_with_iwm

print("Replacing attention with IWM...")
model = replace_attention_with_iwm(model, bandwidth=64)

# Enable last-token-only training
for layer in model.model.layers:
    layer.self_attn.training_last_token_only = True

# Freeze all except coeff_proj
for param in model.parameters():
    param.requires_grad = False
    
trainable = []
for name, param in model.named_parameters():
    if 'coeff_proj' in name:
        param.requires_grad = True
        trainable.append((name, param))
        
print(f"\nTrainable params: {len(trainable)}")

# Forward pass with loss
model.train()
text = "The capital of France is Paris."
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

outputs = model(**inputs, labels=inputs['input_ids'])
loss = outputs.loss

print(f"\nLoss: {loss.item():.4f}")

# Backward
loss.backward()

# Check gradients
print("\nGradient check:")
for name, param in trainable[:3]:  # First 3 layers
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_max = param.grad.abs().max().item()
        print(f"  {name}: grad_norm={grad_norm:.2e}, grad_max={grad_max:.2e}")
    else:
        print(f"  {name}: NO GRADIENT")

# Check weight values
print("\nWeight check (coeff_proj.bias for first 3 layers):")
for layer_idx in [0, 1, 2]:
    attn = model.model.layers[layer_idx].self_attn
    bias = attn.coeff_proj.bias.data
    print(f"  Layer {layer_idx}: c0={bias[0].item():.4f}, c1={bias[1].item():.4f}, c2={bias[2].item():.4f}")
