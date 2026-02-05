"""
Phase 5: Train with Dirichlet Attention

This script fine-tunes the model to work with position-based Dirichlet attention,
enabling CIMBA's O(S) memory compression.

Key difference from train_bandlimited.py:
- train_bandlimited.py: Uses Q·K content similarity + spectral filter
- train_dirichlet.py: Uses ONLY position-based Dirichlet kernel (no Q·K)

After training, the model will:
1. Encode content relevance in V projections (not attention weights)
2. Work with position-based attention weights
3. Be compatible with CIMBA O(S) cache for unlimited context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import math

from cimba_attention import CIMBAAttention, replace_attention_with_cimba

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32 if DEVICE == 'cpu' else torch.bfloat16


# Extended coding samples for training
CODING_SAMPLES = [
    # Basic algorithms
    'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
    'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)',
    'def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b',
    'def is_palindrome(s):\n    cleaned = "".join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]',
    'def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []',
    
    # More complex examples
    'def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result',
    'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
    'def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a',
    'def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True',
    'def reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev',
    
    # Data structures
    'class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop() if self.items else None\n    def peek(self):\n        return self.items[-1] if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0',
    'class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n        return self.items.pop(0) if self.items else None\n    def is_empty(self):\n        return len(self.items) == 0',
    'def depth_first_search(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            depth_first_search(graph, neighbor, visited)\n    return visited',
    'def breadth_first_search(graph, start):\n    visited = set([start])\n    queue = [start]\n    while queue:\n        node = queue.pop(0)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return visited',
    
    # String manipulation
    'def longest_common_prefix(strs):\n    if not strs:\n        return ""\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ""\n    return prefix',
    'def anagram_check(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())',
    'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")',
]


class CodingDataset(Dataset):
    """Dataset for training on coding examples.
    
    Key training fix: Each sample ends with explicit EOS token so model
    learns when to stop generating. EOS is NOT masked in labels.
    """
    
    def __init__(self, tokenizer, max_length=256, multiplier=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = CODING_SAMPLES
        self.multiplier = multiplier  # Repeat samples for more training
        self.eos_token = tokenizer.eos_token
        
    def __len__(self):
        return len(self.samples) * self.multiplier
        
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        # Add explicit EOS at end of each sample so model learns to stop
        sample_with_eos = sample + self.eos_token
        
        encoding = self.tokenizer(
            sample_with_eos, 
            truncation=True, 
            max_length=self.max_length, 
            padding='max_length', 
            return_tensors='pt',
            add_special_tokens=False  # We already added EOS explicitly
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels: mask ONLY padding tokens, NOT the EOS token
        # This teaches the model to predict EOS at the right place
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Mask padding only
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': labels
        }


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def test_generation(model, tokenizer, prompt, max_tokens=80):
    """Test generation with a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,  # Explicit EOS for stopping
            pad_token_id=tokenizer.pad_token_id,  # Use actual pad token
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_dirichlet():
    """
    Main training function for Dirichlet attention.
    
    Strategy: Fine-tune ALL model parameters (not just attention) because
    the model needs to learn to encode content relevance in V projections
    rather than relying on Q·K attention patterns.
    """
    print("="*70)
    print("PHASE 5: DIRICHLET ATTENTION TRAINING")
    print("="*70)
    print()
    print("Goal: Train model to work with position-based attention")
    print("This enables CIMBA O(S) cache for unlimited context length")
    print()
    
    # Configuration
    config = {
        'bandwidth': 64,          # S parameter for Dirichlet kernel
        'epochs': 5,              # More epochs needed for this fundamental change
        'batch_size': 2,
        'learning_rate': 5e-5,    # Lower LR for full model fine-tuning
        'warmup_ratio': 0.1,
        'max_length': 256,
        'gradient_accumulation': 4,
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    
    # IMPORTANT: Use dedicated pad token, NOT eos_token
    # This lets the model learn proper stopping behavior
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        # Add a dedicated pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added dedicated pad token: {tokenizer.pad_token}")
    
    # Test before training
    print("\n" + "-"*50)
    print("BEFORE TRAINING (with original attention):")
    print("-"*50)
    test_prompts = ["def fibonacci(n):", "def quicksort(arr):"]
    for prompt in test_prompts:
        result = test_generation(model, tokenizer, prompt, max_tokens=50)
        print(f"\n{prompt}")
        print(result)
    
    # Replace with Dirichlet attention
    print("\n" + "="*70)
    print(f"Replacing attention with CIMBAAttention (S={config['bandwidth']})")
    print("="*70)
    model = replace_attention_with_cimba(model, bandwidth=config['bandwidth'])
    
    # Test immediately after replacement (should be bad)
    print("\n" + "-"*50)
    print("AFTER REPLACEMENT (before training - expect garbage):")
    print("-"*50)
    for prompt in test_prompts[:1]:
        result = test_generation(model, tokenizer, prompt, max_tokens=30)
        print(f"\n{prompt}")
        print(result[:200] + "..." if len(result) > 200 else result)
    
    # Setup training
    print("\n" + "="*70)
    print("SETTING UP TRAINING")
    print("="*70)
    
    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    
    # Dataset and dataloader
    dataset = CodingDataset(tokenizer, max_length=config['max_length'], multiplier=100)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=0.01,
    )
    
    total_steps = len(dataloader) * config['epochs'] // config['gradient_accumulation']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{config['epochs']}")
        print("="*50)
        
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss / config['gradient_accumulation']
            
            # Backward pass
            loss.backward()
            epoch_loss += loss.item() * config['gradient_accumulation']
            
            # Gradient accumulation
            if (step + 1) % config['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * config["gradient_accumulation"]:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        # Test generation after each epoch
        print("\n" + "-"*50)
        print(f"Generation test (Epoch {epoch + 1}):")
        print("-"*50)
        model.eval()
        for prompt in test_prompts:
            result = test_generation(model, tokenizer, prompt, max_tokens=60)
            print(f"\n{prompt}")
            print(result)
        model.train()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"\nNew best loss: {best_loss:.4f}")
            # torch.save(model.state_dict(), "dirichlet_model_best.pt")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    model.eval()
    
    print("\nGeneration tests:")
    print("-"*50)
    final_prompts = [
        "def fibonacci(n):",
        "def quicksort(arr):",
        "def binary_search(arr, target):",
        "class Stack:",
    ]
    
    for prompt in final_prompts:
        result = test_generation(model, tokenizer, prompt, max_tokens=60)
        print(f"\nPrompt: {prompt}")
        print(result)
        print()
    
    # Memory comparison
    print("\n" + "="*70)
    print("MEMORY ANALYSIS")
    print("="*70)
    
    from cimba_cache import CIMBACache
    
    cache = CIMBACache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        S=config['bandwidth'],
        device=DEVICE,
    )
    
    cimba_mem = cache.memory_usage()['total'] / 1e6
    print(f"CIMBA Cache Memory: {cimba_mem:.2f} MB (constant)")
    print()
    print("Comparison at different context lengths:")
    for seq_len in [1024, 10000, 100000, 1000000]:
        comparison = cache.compare_memory_to_standard(seq_len)
        std_mem = comparison['standard_cache_bytes'] / 1e6
        ratio = comparison['compression_ratio']
        print(f"  {seq_len:>7} tokens: Standard={std_mem:>8.1f}MB, CIMBA={cimba_mem:.1f}MB, Compression={ratio:>5.0f}x")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final loss: {best_loss:.4f}")
    print(f"Bandwidth S: {config['bandwidth']}")
    print(f"CIMBA memory: {cimba_mem:.2f} MB (constant for any context length)")
    print()
    print("Next steps:")
    print("1. Evaluate on HumanEval/MBPP benchmarks")
    print("2. Test needle-in-haystack with long context")
    print("3. Scale to larger models")
    
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_dirichlet()
