"""
Proper comparison of spectral attention quality.
This test properly measures whether spectral attention preserves model quality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_attention import replace_attention_with_spectral, SpectralAttention

print("="*60)
print("SPECTRAL ATTENTION QUALITY TEST")
print("="*60)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Test prompts
prompts = [
    "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):",
    "# Binary search implementation\ndef binary_search(arr, target):",
    "# Factorial function\ndef factorial(n):",
]

def generate(model, prompt, max_tokens=100):
    """Generate with temperature=0 for deterministic comparison."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,  # Greedy for reproducibility
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test 1: Baseline Qwen
print("\n" + "="*60)
print("1. BASELINE QWEN (unmodified)")
print("="*60)

model_baseline = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")

for prompt in prompts[:1]:
    print(f"\nPrompt: {prompt[:50]}...")
    output = generate(model_baseline, prompt)
    print(f"Output:\n{output}")

# Free memory
del model_baseline
torch.cuda.empty_cache()

# Test 2: Fresh spectral replacement (no finetuning)
print("\n" + "="*60)
print("2. SPECTRAL ATTENTION (fresh, no finetuning)")
print("="*60)

model_spectral = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")

model_spectral = replace_attention_with_spectral(model_spectral, SpectralAttention)

for prompt in prompts[:1]:
    print(f"\nPrompt: {prompt[:50]}...")
    output = generate(model_spectral, prompt)
    print(f"Output:\n{output}")

# Test 3: Load finetuned spectral model properly
# We need to:
# 1. Load the state_dict from checkpoint
# 2. Apply it to a model that has SpectralAttention already
print("\n" + "="*60)
print("3. SPECTRAL ATTENTION (loading finetuned weights)")
print("="*60)

try:
    # Load the saved state dict
    checkpoint_path = "outputs/spectral_finetune/final_model"
    
    # The checkpoint has spectral params, but they won't match standard Qwen
    # We need to create a spectral model and load weights carefully
    import os
    import json
    
    # Load weight file
    import safetensors.torch
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    state_dict = safetensors.torch.load_file(weights_path)
    
    # Show what params we have
    spectral_keys = [k for k in state_dict.keys() if 'spectral' in k]
    print(f"\nFound {len(spectral_keys)} spectral parameters in checkpoint:")
    for k in spectral_keys[:5]:
        print(f"  {k}: {state_dict[k].shape}")
    if len(spectral_keys) > 5:
        print(f"  ... and {len(spectral_keys)-5} more")
    
    # Load into our spectral model
    # First filter to only load matching keys
    model_dict = model_spectral.state_dict()
    
    # Find matching keys
    matched = 0
    unmatched_checkpoint = []
    unmatched_model = []
    
    for k in state_dict:
        if k in model_dict:
            if state_dict[k].shape == model_dict[k].shape:
                model_dict[k] = state_dict[k]
                matched += 1
            else:
                print(f"Shape mismatch: {k}: checkpoint {state_dict[k].shape} vs model {model_dict[k].shape}")
        else:
            unmatched_checkpoint.append(k)
    
    for k in model_dict:
        if k not in state_dict:
            unmatched_model.append(k)
    
    print(f"\nLoading weights:")
    print(f"  Matched: {matched}")
    print(f"  Checkpoint-only: {len(unmatched_checkpoint)}")
    print(f"  Model-only: {len(unmatched_model)}")
    
    # Load matched weights
    model_spectral.load_state_dict(model_dict, strict=False)
    
    print("\nGenerating with loaded weights:")
    for prompt in prompts[:1]:
        print(f"\nPrompt: {prompt[:50]}...")
        output = generate(model_spectral, prompt)
        print(f"Output:\n{output}")
        
except Exception as e:
    print(f"Could not load finetuned weights: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
