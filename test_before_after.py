"""
Quick test: Can the trained model reproduce training examples?
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cimba_attention import replace_attention_with_cimba

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

# Training samples
TRAINING_PROMPTS = [
    "def binary_search(arr, target):",
    "def quicksort(arr):",
    "def fibonacci(n):",
    "def is_palindrome(s):",
    "def two_sum(nums, target):",
]

# Novel prompts (not in training)
NOVEL_PROMPTS = [
    "def hello_world():",
    "def add(a, b):",
    "class Calculator:",
]

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test BEFORE replacing attention
    print("\n" + "="*60)
    print("ORIGINAL MODEL (softmax attention)")
    print("="*60)
    
    model.eval()
    for prompt in TRAINING_PROMPTS[:2] + NOVEL_PROMPTS[:1]:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"\n{prompt}")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True)[:200])
    
    # Replace with CIMBAAttention
    print("\n" + "="*60)
    print("CIMBA ATTENTION (position-only, NO training)")
    print("="*60)
    
    model = replace_attention_with_cimba(model, bandwidth=64)
    model.eval()
    
    for prompt in TRAINING_PROMPTS[:2] + NOVEL_PROMPTS[:1]:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"\n{prompt}")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True)[:200])

if __name__ == "__main__":
    main()
