"""
Comprehensive test of SpectralAttention - multiple prompts + filter verification.
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
    
    # Replace with SpectralAttention
    print("Replacing attention...")
    model = replace_attention_with_spectral(model, SpectralAttention)
    
    test_prompts = [
        "# Implement bubble sort\ndef bubble_sort(arr):",
        "# Calculate fibonacci numbers\ndef fibonacci(n):",
        "# Binary search implementation\ndef binary_search(arr, target):",
        "# Reverse a linked list\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):",
    ]
    
    print("\n" + "="*70)
    print("TESTING SPECTRAL ATTENTION (No Finetuning)")
    print("="*70)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt[:50]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output:\n{text}\n")
    
    # Verify spectral filter is trainable
    print("\n" + "="*70)
    print("VERIFYING SPECTRAL FILTER")
    print("="*70)
    
    # Check layer 0's spectral filter
    layer0_attn = model.model.layers[0].self_attn
    print(f"\nSpectral filter shape: {layer0_attn.spectral_filter.shape}")
    print(f"Spectral filter[:10]: {layer0_attn.spectral_filter[:10].tolist()}")
    print(f"Spectral sharpness: {layer0_attn.spectral_sharpness.item():.4f}")
    print(f"Effective sharpness (after softplus): {torch.nn.functional.softplus(layer0_attn.spectral_sharpness).item():.4f}")
    
    # Count trainable parameters
    spectral_params = sum(
        p.numel() for name, p in model.named_parameters() 
        if 'spectral' in name.lower() and p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nSpectral trainable parameters: {spectral_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Spectral overhead: {100*spectral_params/total_params:.4f}%")


if __name__ == "__main__":
    main()
