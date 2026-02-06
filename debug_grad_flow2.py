"""Debug: trace gradient flow through full IWM forward_train_last_token"""

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
    
for name, param in model.named_parameters():
    if 'coeff_proj' in name:
        param.requires_grad = True

# Get one layer for detailed tracing
attn = model.model.layers[0].self_attn

# Create test input
text = "The capital of France is Paris."
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Manual forward through first layer to trace gradients
model.train()

# Get hidden states after embedding
with torch.no_grad():
    embed = model.model.embed_tokens(input_ids)

# Now do the attention forward WITH gradients
hidden_states = embed.clone().detach().requires_grad_(True)

# Get position embeddings
batch, seq_len = input_ids.shape
position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

# Create dummy position embeddings (cos, sin)
cos = torch.ones(seq_len, attn.head_dim // 2, device=DEVICE)
sin = torch.zeros(seq_len, attn.head_dim // 2, device=DEVICE)

print(f"\nInput shape: {hidden_states.shape}")
print(f"attention_mask shape: {attention_mask.shape}")

# Forward through attention
output, last_indices, psd_loss = attn.forward_train_last_token(
    hidden_states,
    position_embeddings=(cos, sin),
    attention_mask=attention_mask.float(),
)

print(f"\nOutput shape: {output.shape}")
print(f"Last indices: {last_indices}")
print(f"Output at last position norm: {output[0, last_indices[0]].norm().item():.4f}")

# Compute simple loss on the output
loss = output[0, last_indices[0]].sum()
print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

# Check gradients
print("\nGradient check:")
print(f"  hidden_states.grad: {'exists' if hidden_states.grad is not None else 'NONE'}")
if hidden_states.grad is not None:
    print(f"    norm: {hidden_states.grad.norm().item():.2e}")

print(f"\n  coeff_proj.weight.grad: {'exists' if attn.coeff_proj.weight.grad is not None else 'NONE'}")
if attn.coeff_proj.weight.grad is not None:
    print(f"    norm: {attn.coeff_proj.weight.grad.norm().item():.2e}")
    print(f"    max: {attn.coeff_proj.weight.grad.abs().max().item():.2e}")

print(f"\n  coeff_proj.bias.grad: {'exists' if attn.coeff_proj.bias.grad is not None else 'NONE'}")
if attn.coeff_proj.bias.grad is not None:
    print(f"    norm: {attn.coeff_proj.bias.grad.norm().item():.2e}")
    print(f"    max: {attn.coeff_proj.bias.grad.abs().max().item():.2e}")

# Also check q_proj and v_proj (should be None since frozen)
print(f"\n  q_proj.weight.grad: {'exists' if attn.q_proj.weight.grad is not None else 'NONE (expected)'}")
print(f"  v_proj.weight.grad: {'exists' if attn.v_proj.weight.grad is not None else 'NONE (expected)'}")
