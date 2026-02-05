"""
Key comparison: SpectralAttention (Q·K + bandlimit) vs CIMBAAttention (position-only)

SpectralAttention was working well. Let's verify it still does, and understand
why CIMBA (which should reconstruct it exactly) doesn't work.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_attention import SpectralAttention, replace_attention_with_spectral
from cimba_attention import CIMBAAttention, replace_attention_with_cimba

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

PROMPTS = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "def hello_world():",
]

def test_spectral():
    """Test SpectralAttention (Q·K + bandlimit) - this worked before."""
    print("="*60)
    print("SPECTRAL ATTENTION (Q·K + bandlimited filter)")
    print("This uses content-based attention with spectral filtering")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Replace with SpectralAttention (Q·K based)
    model = replace_attention_with_spectral(model, SpectralAttention, bandwidth_target=64)
    model.eval()
    
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"\n{prompt}")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    return model, tokenizer

def test_cimba():
    """Test CIMBAAttention (position-only) - this is struggling."""
    print("\n" + "="*60)
    print("CIMBA ATTENTION (position-only Dirichlet)")
    print("This ignores Q·K, uses only position for attention weights")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Replace with CIMBAAttention (position-only)
    model = replace_attention_with_cimba(model, bandwidth=64)
    model.eval()
    
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"\n{prompt}")
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(result[:300] if len(result) > 300 else result)

def main():
    print("Comparing attention mechanisms...")
    print()
    
    test_spectral()
    test_cimba()
    
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("""
SpectralAttention works because it still uses Q·K content similarity.
The spectral filter just bandlimits the attention pattern.

CIMBAAttention struggles because it removes Q·K entirely.
Attention weights depend ONLY on position: D_S(pos_q - pos_k)

The CIMBA theory is correct for reconstructing Dirichlet attention.
But Dirichlet attention itself may not be expressive enough for
general language modeling without Q·K content awareness.

Options:
1. Hybrid: Use Q·K for weights, CIMBA for V storage (50% memory saving)
2. Train from scratch with position-only attention (research project)
3. Accept position-only works for specific tasks only
""")

if __name__ == "__main__":
    main()
