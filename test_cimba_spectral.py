"""
Quick test for CIMBA Spectral Attention with moment cache.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

def test_cimba_spectral():
    print("="*60)
    print("CIMBA SPECTRAL ATTENTION TEST")
    print("="*60)
    
    print(f"\nDevice: {DEVICE}, Dtype: {DTYPE}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    # Test generation BEFORE replacement
    print("\n" + "-"*40)
    print("BASELINE (Standard Attention)")
    print("-"*40)
    
    prompt = "def fibonacci(n):\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    print(f"\nBaseline:\n{baseline_text}")
    
    # Replace with CIMBA Spectral Attention
    print("\n" + "-"*40)
    print("REPLACING WITH CIMBA SPECTRAL ATTENTION")
    print("-"*40)
    
    from cimba_spectral_attention import replace_attention_with_spectral, CimbaSpectralAttention
    model = replace_attention_with_spectral(model, bandwidth_target=64)
    
    # Test generation AFTER replacement (no training, just drop-in)
    print("\n" + "-"*40)
    print("CIMBA SPECTRAL (Untrained)")
    print("-"*40)
    
    with torch.no_grad():
        spectral_output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    spectral_text = tokenizer.decode(spectral_output[0], skip_special_tokens=True)
    print(f"\nSpectral:\n{spectral_text}")
    
    # Analysis
    print("\n" + "-"*40)
    print("ANALYSIS")
    print("-"*40)
    
    # Check similarity
    baseline_tokens = baseline_text.split()
    spectral_tokens = spectral_text.split()
    
    common = set(baseline_tokens) & set(spectral_tokens)
    if len(baseline_tokens) > 0:
        similarity = len(common) / len(baseline_tokens)
        print(f"Token overlap: {similarity:.1%}")
    
    # Check for code structure
    code_indicators = ['def ', 'return', 'if ', 'for ', ':', '(', ')']
    code_score = sum(1 for ind in code_indicators if ind in spectral_text)
    print(f"Code indicators: {code_score}/{len(code_indicators)}")
    
    if code_score >= 4:
        print("✓ CIMBA Spectral produces code-like output")
    else:
        print("⚠ May need finetuning")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nCIMBA Spectral preserves Q·K content attention while adding spectral filtering.")
    print("This should work better out-of-the-box than IWM which removes K entirely.")


if __name__ == "__main__":
    test_cimba_spectral()
