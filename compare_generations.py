"""Compare baseline vs spectral attention generation quality."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading baseline model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n" + "="*60)
print("BASELINE QWEN (no modification)")
print("="*60)
print(f"Prompt: {prompt}\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated:\n{generated}")

# Now load and test the finetuned spectral model
print("\n" + "="*60)
print("FINETUNED SPECTRAL ATTENTION")
print("="*60)

from spectral_attention import replace_attention_with_spectral, SpectralAttention

# Load fresh model and replace attention
model2 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")
model2 = replace_attention_with_spectral(model2, SpectralAttention)

# Load finetuned weights (just the spectral params changed)
# Actually, let's test the UNTRAINED spectral attention first
print("\nUNTRAINED spectral attention (fresh replacement):")
with torch.no_grad():
    outputs = model2.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated:\n{generated}")

# Now load the trained checkpoint
print("\n" + "-"*60)
print("Loading finetuned checkpoint...")
checkpoint_path = "outputs/spectral_finetune/final_model"
try:
    model3 = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    print("\nFINETUNED spectral attention:")
    with torch.no_grad():
        outputs = model3.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated:\n{generated}")
except Exception as e:
    print(f"Could not load checkpoint: {e}")
