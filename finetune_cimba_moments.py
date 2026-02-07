"""
CIMBA Moment Mode Finetuning Script

This script finetunes a model to work with moment_mode=True in CimbaMomentCache.
The goal is to train the model so that moment-based V reconstruction produces
acceptable outputs, enabling 50% KV cache memory savings.

Key idea:
- Use CimbaSpectralAttention with moment_mode=True during training
- The model learns to produce attention patterns that reconstruct well from moments
- After training, switch to moment_mode=True for inference with O(S) V memory

Based on finetune_spectral.py
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

from cimba_spectral_attention import (
    CimbaSpectralAttention,
    CimbaMomentCache,
    replace_attention_with_spectral,
    create_cimba_cache_for_model,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for CIMBA moment mode finetuning."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # CIMBA settings
    bandwidth_S: int = 64  # Number of moment pairs (2S+1 total moments)
    moment_mode: bool = True  # Train with moment reconstruction
    
    # Training
    learning_rate: float = 1e-4
    spectral_lr: float = 1e-3  # Higher LR for spectral parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    warmup_steps: int = 100
    max_seq_length: int = 512
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"  # or "float32"
    
    # Logging
    output_dir: str = "outputs/cimba_moment_finetune"
    log_every: int = 10
    save_every: int = 200
    eval_every: int = 50
    
    # Seed
    seed: int = 42


# ============================================================================
# Dataset (same as finetune_spectral.py)
# ============================================================================

CODING_SAMPLES = [
    """# Implement a function to find all prime numbers up to n using the Sieve of Eratosthenes
def sieve_of_eratosthenes(n):
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(n + 1) if is_prime[i]]""",
    
    """# Implement a binary search tree with insert and search operations
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        
        current = self.root
        while True:
            if val < current.val:
                if current.left is None:
                    current.left = TreeNode(val)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = TreeNode(val)
                    return
                current = current.right
    
    def search(self, val):
        current = self.root
        while current:
            if val == current.val:
                return True
            elif val < current.val:
                current = current.left
            else:
                current = current.right
        return False""",
    
    """# Implement merge sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result""",
    
    """# Implement a LRU cache using OrderedDict
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)""",
    
    """# Implement depth-first search for a graph
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    
    return result""",
    
    """// Implement a generic stack in Rust
pub struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }
    
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }
    
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }
    
    pub fn peek(&self) -> Option<&T> {
        self.items.last()
    }
    
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}""",
    
    """// Implement the quicksort algorithm in Rust
fn quicksort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    
    let pivot_idx = partition(arr);
    
    let (left, right) = arr.split_at_mut(pivot_idx);
    quicksort(left);
    quicksort(&mut right[1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let pivot_idx = len / 2;
    
    arr.swap(pivot_idx, len - 1);
    
    let mut store_idx = 0;
    for i in 0..len - 1 {
        if arr[i] < arr[len - 1] {
            arr.swap(i, store_idx);
            store_idx += 1;
        }
    }
    
    arr.swap(store_idx, len - 1);
    store_idx
}""",
    
    """// Implement a debounce function in TypeScript
function debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number
): (...args: Parameters<T>) => void {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    
    return function(this: any, ...args: Parameters<T>) {
        if (timeoutId !== null) {
            clearTimeout(timeoutId);
        }
        
        timeoutId = setTimeout(() => {
            func.apply(this, args);
            timeoutId = null;
        }, wait);
    };
}""",
    
    """# Implement binary search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1""",
    
    """# Implement a simple hash map with chaining
class HashMap:
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
    
    def get(self, key):
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None""",
]


class CodingDataset(Dataset):
    """Simple dataset of coding samples for finetuning."""
    
    def __init__(self, tokenizer, samples: List[str], max_length: int = 512):
        self.tokenizer = tokenizer
        self.samples = samples
        self.max_length = max_length
        
        self.encodings = []
        for sample in samples:
            encoding = tokenizer(
                sample,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })
        
        total_tokens = sum(len(e["input_ids"]) for e in self.encodings)
        print(f"  Dataset: {len(self.encodings)} samples, {total_tokens} total tokens")
        print(f"  Avg tokens per sample: {total_tokens / len(self.encodings):.1f}")
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch):
    """Dynamic padding collator."""
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        
        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }


# ============================================================================
# Training
# ============================================================================

def setup_optimizer(model, config: FinetuneConfig):
    """Create optimizer with different LRs for spectral vs other params."""
    
    spectral_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "spectral" in name or "lowpass" in name or "sharpness" in name:
                spectral_params.append(param)
            else:
                other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": config.learning_rate},
        {"params": spectral_params, "lr": config.spectral_lr},
    ])
    
    print(f"Spectral parameters: {len(spectral_params)}")
    print(f"Other parameters: {len(other_params)}")
    
    return optimizer


def evaluate(model, eval_loader, config: FinetuneConfig):
    """Evaluate model on dataset using moment_mode cache."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def generate_sample(model, tokenizer, prompt: str, config: FinetuneConfig, max_new_tokens: int = 100):
    """Generate a sample completion using moment_mode cache."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    
    # Create moment_mode cache for generation
    cache = create_cimba_cache_for_model(
        model, 
        S=config.bandwidth_S, 
        compat_mode=False,  # Use moment reconstruction
        moment_mode=True,
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            past_key_values=cache,
            use_cache=True,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Report memory usage
    mem = cache.memory_usage()
    print(f"  [Generation] V cache: {mem['v_cache_bytes']} bytes (should be 0)")
    print(f"  [Generation] V moments: {mem['v_moments_bytes'] / 1024:.1f} KB (constant)")
    
    return generated


def train(config: FinetuneConfig):
    """Main training loop with moment_mode."""
    
    torch.manual_seed(config.seed)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # Replace attention with CimbaSpectralAttention
    print(f"\nReplacing attention with CimbaSpectralAttention (S={config.bandwidth_S})...")
    model = replace_attention_with_spectral(
        model,
        attention_class=CimbaSpectralAttention,
        bandwidth_target=config.bandwidth_S,
    )
    
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataset
    print("\nPreparing dataset...")
    dataset = CodingDataset(tokenizer, CODING_SAMPLES, config.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    # Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
    print(f"Mode: moment_mode={config.moment_mode} (training with moment reconstruction)")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    model.train()
    global_step = 0
    total_loss = 0
    loss_history = []
    eval_history = []
    
    start_time = time.time()
    pbar = tqdm(total=config.max_steps, desc="Training")
    
    while global_step < config.max_steps:
        for batch in dataloader:
            if global_step >= config.max_steps:
                break
            
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Create moment_mode cache for this forward pass
            if config.moment_mode:
                cache = create_cimba_cache_for_model(
                    model, 
                    S=config.bandwidth_S,
                    compat_mode=False,
                    moment_mode=True,
                )
            else:
                cache = None
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=cache,
                use_cache=config.moment_mode,
            )
            
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            # Gradient accumulation
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            
            # Logging
            if global_step % config.log_every == 0:
                avg_loss = total_loss / config.log_every
                loss_history.append({"step": global_step, "loss": avg_loss})
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                total_loss = 0
            
            # Evaluation
            if global_step % config.eval_every == 0:
                eval_loss = evaluate(model, dataloader, config)
                eval_history.append({"step": global_step, "eval_loss": eval_loss})
                print(f"\nStep {global_step}: eval_loss = {eval_loss:.4f}")
                
                # Generate a sample with moment_mode
                prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"
                print("Generating sample with moment_mode=True...")
                sample = generate_sample(model, tokenizer, prompt, config, max_new_tokens=80)
                print(f"Sample:\n{sample}\n")
                
                model.train()
            
            # Checkpointing
            if global_step % config.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
    
    pbar.close()
    
    # Final evaluation
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time / 60:.1f} minutes")
    
    final_eval_loss = evaluate(model, dataloader, config)
    print(f"Final eval loss: {final_eval_loss:.4f}")
    
    # Final samples with moment_mode
    print("\n" + "="*60)
    print("FINAL SAMPLE GENERATIONS (moment_mode=True)")
    print("="*60)
    
    test_prompts = [
        "# Implement binary search\ndef binary_search(arr, target):",
        "# Implement factorial\ndef factorial(n):",
        "# Implement a simple stack\nclass Stack:",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        sample = generate_sample(model, tokenizer, prompt, config, max_new_tokens=100)
        print(f"Generated:\n{sample}")
        print("-" * 40)
    
    # Save training history
    history = {
        "config": {
            "model_name": config.model_name,
            "bandwidth_S": config.bandwidth_S,
            "moment_mode": config.moment_mode,
            "learning_rate": config.learning_rate,
            "spectral_lr": config.spectral_lr,
            "max_steps": config.max_steps,
        },
        "loss_history": loss_history,
        "eval_history": eval_history,
        "final_eval_loss": final_eval_loss,
        "training_time_minutes": elapsed_time / 60,
    }
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot loss curve
    if loss_history:
        steps = [h["step"] for h in loss_history]
        losses = [h["loss"] for h in loss_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        
        if eval_history:
            eval_steps = [h["step"] for h in eval_history]
            eval_losses = [h["eval_loss"] for h in eval_history]
            plt.plot(eval_steps, eval_losses, 'r-o', linewidth=2, label='Eval Loss')
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('CIMBA Moment Mode Finetuning Loss Curve', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nLoss curve saved to {output_dir / 'loss_curve.png'}")
    
    # Save final model
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")
    
    return history


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    config = FinetuneConfig()
    
    print("="*60)
    print("CIMBA MOMENT MODE FINETUNING EXPERIMENT")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Bandwidth S: {config.bandwidth_S} ({2*config.bandwidth_S + 1} moments)")
    print(f"moment_mode: {config.moment_mode}")
    print(f"Device: {config.device}")
    print(f"Max steps: {config.max_steps}")
    print("="*60)
    print()
    print("Goal: Train model to work with moment-based V reconstruction")
    print("      This enables 50% KV cache memory savings at inference time")
    print("="*60)
    
    # Run training
    history = train(config)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if history["eval_history"]:
        initial_loss = history["eval_history"][0]["eval_loss"]
        final_loss = history["final_eval_loss"]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        print(f"Improvement: {improvement:.1f}%")
        
        if final_loss < initial_loss:
            print("\n✓ LOSS DECREASED - Moment mode is trainable!")
            print("  The model is learning to work with moment reconstruction.")
            print("  You can now use moment_mode=True for 50% V memory savings.")
        else:
            print("\n✗ LOSS DID NOT DECREASE - May need adjustments")
            print("  Try: longer training, different S, curriculum learning")
    
    print("="*60)
