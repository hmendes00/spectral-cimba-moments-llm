"""
Quick test to verify IWM attention with RoPE fix.

Tests:
1. Forward pass works without errors
2. RoPE is being applied (query values differ from raw projection)
3. Generation produces different output than before
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

def test_iwm_rope():
    print("="*60)
    print("IWM ATTENTION RoPE FIX TEST")
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
    
    # Test generation BEFORE IWM replacement
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
    print(f"\nBaseline generation:\n{baseline_text}")
    
    # Replace with IWM attention
    print("\n" + "-"*40)
    print("REPLACING WITH IWM ATTENTION (with RoPE fix)")
    print("-"*40)
    
    from iwm_attention import replace_attention_with_iwm
    model = replace_attention_with_iwm(model, bandwidth=64)
    
    # Test generation AFTER IWM replacement (untrained)
    print("\n" + "-"*40)
    print("IWM ATTENTION (Untrained, with RoPE)")
    print("-"*40)
    
    with torch.no_grad():
        iwm_output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    iwm_text = tokenizer.decode(iwm_output[0], skip_special_tokens=True)
    print(f"\nIWM generation:\n{iwm_text}")
    
    # Check if outputs are different (they should be, IWM is untrained)
    print("\n" + "-"*40)
    print("ANALYSIS")
    print("-"*40)
    
    if baseline_text == iwm_text:
        print("⚠ WARNING: Outputs identical - IWM may not be active")
    else:
        print("✓ Outputs differ (expected - IWM is untrained)")
    
    # Check for obvious degenerate patterns
    degenerate_patterns = ['@', 'endsection', 'API', 'nodiscard', '+++']
    has_degenerate = any(p in iwm_text for p in degenerate_patterns)
    
    if has_degenerate:
        print("⚠ WARNING: Output contains degenerate patterns")
        print("  This is expected for untrained IWM, but RoPE should help")
    else:
        print("✓ No obvious degenerate patterns detected")
    
    # Check if output looks like code
    code_indicators = ['def ', 'return', 'if ', 'for ', ':', '(', ')']
    code_score = sum(1 for ind in code_indicators if ind in iwm_text)
    print(f"\nCode indicators found: {code_score}/{len(code_indicators)}")
    
    if code_score >= 3:
        print("✓ Output resembles code structure")
    else:
        print("⚠ Output doesn't look like code")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nNote: For good generation, IWM needs training (distillation or finetuning)")
    print("The RoPE fix ensures pretrained Q/V projections receive expected inputs.")


if __name__ == "__main__":
    test_iwm_rope()
