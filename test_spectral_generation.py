"""
Test SpectralAttention generation with KV cache support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spectral_attention import SpectralAttention, replace_attention_with_spectral


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    
    # Test baseline first
    prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("\n=== BASELINE (Original Model) ===")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(baseline_text)
    
    # Now replace with SpectralAttention
    print("\n=== REPLACING ATTENTION ===")
    model = replace_attention_with_spectral(model, SpectralAttention)
    
    print("\n=== SPECTRAL ATTENTION (No Finetuning) ===")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    spectral_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(spectral_text)
    
    # Compare
    print("\n" + "="*60)
    if "palindrome" in spectral_text.lower() and "return" in spectral_text:
        print("✓ SpectralAttention produces coherent code!")
    else:
        print("✗ SpectralAttention still produces garbage")
        print("\nThis is expected - spectral filtering changes the attention pattern.")
        print("We need to finetune the model to adapt to spectral attention.")


if __name__ == "__main__":
    main()
