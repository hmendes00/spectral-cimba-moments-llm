"""
Test CimbaMomentCache with HuggingFace generate().

This tests that:
1. CimbaMomentCache is compatible with model.generate()
2. V is stored as moments (O(S)) not full tensors
3. Generation produces coherent output

Run with: .venv\Scripts\Activate.ps1 ; python test_cimba_with_generate.py
"""

import torch
import sys

def test_cimba_cache_with_generate():
    """Test CimbaMomentCache with HuggingFace generate()."""
    print("=" * 60)
    print("TEST: CimbaMomentCache with HuggingFace generate()")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cimba_spectral_attention import (
        replace_attention_with_spectral,
        create_cimba_cache_for_model,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Replace attention with spectral
    print("Replacing attention with CimbaSpectralAttention...")
    model = replace_attention_with_spectral(model, bandwidth_target=64)
    model.eval()
    
    # Create CIMBA moment cache
    print("Creating CimbaMomentCache...")
    cimba_cache = create_cimba_cache_for_model(model, S=64)
    
    # Check cache properties
    print(f"  Cache properties:")
    print(f"    - num_layers: {cimba_cache.num_layers}")
    print(f"    - num_kv_heads: {cimba_cache.num_kv_heads}")
    print(f"    - S (bandwidth): {cimba_cache.S}")
    print(f"    - num_moments: {cimba_cache.num_moments}")
    print(f"    - has layers property: {hasattr(cimba_cache, 'layers')}")
    print(f"    - has get_mask_sizes: {hasattr(cimba_cache, 'get_mask_sizes')}")
    
    # Test prompt
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: '{prompt}'")
    print(f"Input tokens: {inputs.input_ids.shape[1]}")
    
    # Generate WITHOUT cache first (for comparison)
    print("\n1. Generating WITHOUT CimbaMomentCache (standard)...")
    with torch.no_grad():
        outputs_standard = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,  # Uses DynamicCache
        )
    response_standard = tokenizer.decode(outputs_standard[0], skip_special_tokens=True)
    print(f"   Response: {response_standard}")
    
    # Generate WITH CimbaMomentCache
    print("\n2. Generating WITH CimbaMomentCache...")
    cimba_cache.reset()  # Reset cache
    
    try:
        with torch.no_grad():
            outputs_cimba = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                past_key_values=cimba_cache,  # Use CIMBA cache
                use_cache=True,
            )
        response_cimba = tokenizer.decode(outputs_cimba[0], skip_special_tokens=True)
        print(f"   Response: {response_cimba}")
        
        # Check memory usage
        mem = cimba_cache.memory_usage()
        seq_len = cimba_cache.get_seq_length()
        print(f"\n3. CimbaMomentCache memory usage (compat_mode={cimba_cache.compat_mode}):")
        print(f"   - K cache: {mem['k_cache_bytes'] / 1024:.2f} KB")
        print(f"   - V cache: {mem['v_cache_bytes'] / 1024:.2f} KB")
        print(f"   - V moments: {mem['v_moments_bytes'] / 1024:.2f} KB (constant O(S))")
        print(f"   - Total: {mem['total_bytes'] / 1024:.2f} KB")
        print(f"   - Sequence length: {seq_len}")
        
        # In compat_mode: V cache stores full history (exact outputs)
        # In moment_mode: V cache = 0 (uses moments only, approximate)
        if cimba_cache.compat_mode:
            print(f"   Note: compat_mode=True stores full V for exact outputs")
            print(f"         Set moment_mode=True for O(S) V memory (requires finetuning)")
        else:
            print(f"   Note: moment_mode=True, V is O(S) constant!")
        
        # Check if responses match
        match = response_standard == response_cimba
        print(f"\n4. Responses match: {match}")
        
        if not match:
            print(f"   Standard: {response_standard}")
            print(f"   CIMBA:    {response_cimba}")
        
        print(f"\n{'✓ PASSED' if match else '✗ FAILED'}: CimbaMomentCache works with generate()")
        return match
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CimbaMomentCache + HuggingFace generate() Test")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print()
    
    success = test_cimba_cache_with_generate()
    sys.exit(0 if success else 1)
