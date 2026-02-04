"""
Spectral Attention Finetuning Script

This script finetunes Qwen-0.5B with spectral attention replacing softmax.
The goal is to test whether a transformer can learn to code using
natively spectral attention mechanisms.

Key hypothesis: If the loss goes down and the model produces coherent code,
then spectral attention is viable and CIMBA's O(K) memory bound applies.

Uses LoRA for efficient finetuning (only train adapters + spectral params).
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

from spectral_attention import (
    SpectralAttention,
    PureSpectralAttention,
    replace_attention_with_spectral,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for spectral attention finetuning."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    # Spectral attention
    attention_type: str = "spectral"  # "spectral" or "pure_spectral"
    layers_to_replace: Optional[List[int]] = None  # None = all layers
    
    # Training
    learning_rate: float = 1e-4
    spectral_lr: float = 1e-3  # Higher LR for spectral parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 500
    warmup_steps: int = 50
    max_seq_length: int = 512
    
    # LoRA (if enabled)
    use_lora: bool = False  # Start without LoRA to isolate spectral effects
    lora_r: int = 8
    lora_alpha: int = 16
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    output_dir: str = "outputs/spectral_finetune"
    log_every: int = 10
    save_every: int = 100
    eval_every: int = 50
    
    # Seed
    seed: int = 42


# ============================================================================
# Dataset
# ============================================================================

CODING_SAMPLES = [
    # Python samples
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
    
    return result

# Example usage:
# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': []
# }
# print(dfs(graph, 'A'))  # ['A', 'B', 'D', 'E', 'F', 'C']""",
    
    # Rust samples
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
    
    pub fn len(&self) -> usize {
        self.items.len()
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
    
    # JavaScript/TypeScript samples
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
    
    """// Implement a simple Promise.all alternative
function promiseAll<T>(promises: Promise<T>[]): Promise<T[]> {
    return new Promise((resolve, reject) => {
        const results: T[] = [];
        let completed = 0;
        
        if (promises.length === 0) {
            resolve(results);
            return;
        }
        
        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then((value) => {
                    results[index] = value;
                    completed++;
                    
                    if (completed === promises.length) {
                        resolve(results);
                    }
                })
                .catch(reject);
        });
    });
}""",
    
    # SQL samples
    """-- Find the top 3 highest-paid employees in each department
WITH RankedEmployees AS (
    SELECT 
        e.employee_id,
        e.name,
        e.salary,
        d.department_name,
        ROW_NUMBER() OVER (
            PARTITION BY e.department_id 
            ORDER BY e.salary DESC
        ) as salary_rank
    FROM employees e
    JOIN departments d ON e.department_id = d.department_id
)
SELECT 
    employee_id,
    name,
    salary,
    department_name,
    salary_rank
FROM RankedEmployees
WHERE salary_rank <= 3
ORDER BY department_name, salary_rank;""",
]


class CodingDataset(Dataset):
    """Simple dataset of coding samples for finetuning."""
    
    def __init__(
        self,
        tokenizer,
        samples: List[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.samples = samples
        self.max_length = max_length
        
        # Tokenize all samples WITHOUT padding (we'll pad dynamically per batch)
        self.encodings = []
        for sample in samples:
            encoding = tokenizer(
                sample,
                truncation=True,
                max_length=max_length,
                padding=False,  # NO fixed padding - avoids padding dominating loss
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })
        
        # Report actual token counts
        total_tokens = sum(len(e["input_ids"]) for e in self.encodings)
        print(f"  Dataset: {len(self.encodings)} samples, {total_tokens} total tokens")
        print(f"  Avg tokens per sample: {total_tokens / len(self.encodings):.1f}")
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch):
    """Dynamic padding collator - pads to longest in batch, not fixed length."""
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        
        # Pad on the right
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
            if "spectral" in name or "kernel_coeffs" in name:
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


def evaluate(model, eval_loader, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # CRITICAL FIX: Mask padding tokens in labels with -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # -100 is ignored by CrossEntropyLoss
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Generate a sample completion."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def train(config: FinetuneConfig):
    """Main training loop."""
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    # Replace attention with spectral attention
    print(f"\nReplacing attention with {config.attention_type}...")
    attention_class = (
        SpectralAttention if config.attention_type == "spectral"
        else PureSpectralAttention
    )
    model = replace_attention_with_spectral(
        model,
        attention_class=attention_class,
        layers_to_replace=config.layers_to_replace,
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
        collate_fn=collate_fn,  # Use dynamic padding
    )
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    
    num_training_steps = config.max_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
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
            
            # CRITICAL FIX: Mask padding tokens in labels with -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # -100 is ignored by CrossEntropyLoss
            
            # Sanity check: abort if padding dominates (first batch only)
            if global_step == 0:
                total_tokens = input_ids.numel()
                pad_tokens = (attention_mask == 0).sum().item()
                pad_pct = 100 * pad_tokens / total_tokens
                print(f"\n  Padding check: {pad_pct:.1f}% padding tokens")
                if pad_pct > 50:
                    print(f"  WARNING: High padding ratio may affect training")
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
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
                eval_loss = evaluate(model, dataloader, config.device)
                eval_history.append({"step": global_step, "eval_loss": eval_loss})
                print(f"\nStep {global_step}: eval_loss = {eval_loss:.4f}")
                
                # Generate a sample
                prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"
                sample = generate_sample(model, tokenizer, prompt, max_new_tokens=80)
                print(f"Sample generation:\n{sample}\n")
                
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
    
    final_eval_loss = evaluate(model, dataloader, config.device)
    print(f"Final eval loss: {final_eval_loss:.4f}")
    
    # Generate final samples
    print("\n" + "="*60)
    print("FINAL SAMPLE GENERATIONS")
    print("="*60)
    
    test_prompts = [
        "# Implement binary search\ndef binary_search(arr, target):",
        "// Implement factorial in Rust\nfn factorial(n: u64) -> u64 {",
        "# Implement a simple hash map\nclass HashMap:",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        sample = generate_sample(model, tokenizer, prompt, max_new_tokens=150)
        print(f"Generated:\n{sample}")
        print("-" * 40)
    
    # Save training history
    history = {
        "config": {
            "model_name": config.model_name,
            "attention_type": config.attention_type,
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
        plt.title('Spectral Attention Finetuning Loss Curve', fontsize=14)
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
    print("SPECTRAL ATTENTION FINETUNING EXPERIMENT")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Attention type: {config.attention_type}")
    print(f"Device: {config.device}")
    print(f"Max steps: {config.max_steps}")
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
            print("\n✓ LOSS DECREASED - Spectral attention is trainable!")
        else:
            print("\n✗ LOSS DID NOT DECREASE - Spectral attention may need adjustments")
    
    print("="*60)
