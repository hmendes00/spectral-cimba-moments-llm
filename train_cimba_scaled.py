"""
Scaled CIMBA Training with CodeParrot Dataset

This script trains CIMBA attention on real code data from codeparrot-clean,
with proper validation split and HumanEval evaluation checkpoints.

Key improvements over train_dirichlet.py:
- Uses codeparrot/codeparrot-clean instead of 17 hardcoded samples
- Filters to single functions < 512 tokens after tokenization
- Streaming + shuffle buffer (no memory blowup)
- 10% validation split for monitoring
- HumanEval evaluation at checkpoints
- LoRA support for more stable/efficient training

Usage:
    python train_cimba_scaled.py --num-samples 5000
    python train_cimba_scaled.py --num-samples 5000 --eval-humaneval
    python train_cimba_scaled.py --num-samples 25000 --epochs 3 --use-lora
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterator, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from cimba_attention import replace_attention_with_cimba

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32 if DEVICE == 'cpu' else torch.bfloat16


def is_likely_single_function(code: str) -> bool:
    """
    Heuristic to check if code is likely a single function/class.
    
    Filters out:
    - Files with multiple top-level definitions
    - Import-heavy files
    - Very short snippets (< 3 lines)
    """
    lines = code.strip().split('\n')
    
    # Too short
    if len(lines) < 3:
        return False
    
    # Count top-level definitions
    top_level_defs = 0
    for line in lines:
        stripped = line.lstrip()
        # Top-level = no leading whitespace and starts with def/class
        if line == stripped and (stripped.startswith('def ') or stripped.startswith('class ')):
            top_level_defs += 1
    
    # Allow 1-2 top-level definitions (main function + maybe a helper)
    if top_level_defs == 0 or top_level_defs > 2:
        return False
    
    # Too many imports suggests a full file, not a snippet
    import_count = sum(1 for line in lines if line.strip().startswith(('import ', 'from ')))
    if import_count > 3:
        return False
    
    return True


def filter_and_tokenize_samples(
    tokenizer,
    max_length: int = 512,
    num_samples: int = 5000,
    seed: int = 42,
) -> List[Dict]:
    """
    Load codeparrot-clean, filter to single functions, tokenize, and filter by length.
    
    Returns a list of tokenized samples that fit within max_length.
    """
    print(f"Loading codeparrot/codeparrot-clean (streaming)...")
    
    # Load with streaming to avoid memory issues
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )
    
    # Shuffle with a buffer
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    samples = []
    skipped_long = 0
    skipped_multi = 0
    
    print(f"Filtering samples (target: {num_samples})...")
    pbar = tqdm(total=num_samples, desc="Collecting samples")
    
    for example in dataset:
        if len(samples) >= num_samples:
            break
        
        code = example['content']
        
        # Filter: single function heuristic
        if not is_likely_single_function(code):
            skipped_multi += 1
            continue
        
        # Tokenize and check length
        # Add EOS token for proper stopping
        code_with_eos = code + tokenizer.eos_token
        
        # Use truncation to avoid "sequence length longer than max" warning
        # We still filter by actual length, but this prevents HF noise
        encoding = tokenizer(
            code_with_eos,
            truncation=True,
            max_length=max_length + 1,  # +1 to detect too-long samples
            return_tensors='pt',
            add_special_tokens=False,
        )
        
        token_length = encoding['input_ids'].shape[1]
        
        # Filter: must fit in max_length
        if token_length > max_length:
            skipped_long += 1
            continue
        
        # Keep this sample
        samples.append({
            'code': code,
            'token_length': token_length,
        })
        pbar.update(1)
    
    pbar.close()
    
    print(f"\nCollected {len(samples)} samples")
    print(f"Skipped (too long): {skipped_long}")
    print(f"Skipped (multi-def): {skipped_multi}")
    
    # Stats
    lengths = [s['token_length'] for s in samples]
    print(f"Token length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    return samples


class CodeDataset(Dataset):
    """Dataset for code samples with proper EOS handling. Returns unpadded sequences."""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        code = self.samples[idx]['code']
        code_with_eos = code + self.eos_token
        
        # Tokenize WITHOUT padding - collator will handle dynamic padding
        encoding = self.tokenizer(
            code_with_eos,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # No padding here - use DataCollatorWithPadding
            return_tensors='pt',
            add_special_tokens=False,
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels = input_ids (collator will handle padding mask)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),  # Collator will mask PAD positions
        }


def collate_with_padding(batch, tokenizer):
    """
    Custom collator that:
    1. Pads input_ids and attention_mask dynamically to batch max length
    2. Masks PAD positions in labels with -100
    """
    # Find max length in this batch
    max_len = max(len(item['input_ids']) for item in batch)
    # Round up to multiple of 8 for efficiency
    max_len = ((max_len + 7) // 8) * 8
    
    pad_token_id = tokenizer.pad_token_id
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        # Pad on the right
        input_ids = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        labels = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])  # -100 = ignore in loss
        
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_labels.append(labels)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_labels),
    }


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def test_generation(model, tokenizer, prompt, max_tokens=60):
    """Test generation with a prompt (greedy decoding for determinism)."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            # Explicitly clear sampling params to avoid HF warnings
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_humaneval_quick(model, tokenizer, num_problems: int = 20) -> float:
    """
    Quick HumanEval evaluation during training.
    Returns pass@1 on first num_problems.
    """
    from eval_humaneval import load_humaneval, evaluate_humaneval
    
    problems = load_humaneval()[:num_problems]
    results = evaluate_humaneval(model, tokenizer, problems, verbose=False)
    return results['pass@1']


def train_cimba_scaled(args):
    """Main training function with scaled data."""
    
    print("="*70)
    print("SCALED CIMBA TRAINING")
    print("="*70)
    print()
    print(f"Target samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Bandwidth S: {args.bandwidth}")
    print()
    
    # Configuration
    config = {
        'bandwidth': args.bandwidth,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'min_lr': args.min_lr,
        'warmup_ratio': 0.1,
        'max_length': args.max_length,
        'gradient_accumulation': args.grad_accum,
        'num_samples': args.num_samples,
        'val_split': 0.1,
        # LoRA config
        'use_lora': args.use_lora,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
    }
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load tokenizer first (needed for filtering)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    
    # Add dedicated pad token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print(f"Added dedicated pad token: {tokenizer.pad_token}")
    
    # Load and filter samples
    all_samples = filter_and_tokenize_samples(
        tokenizer,
        max_length=config['max_length'],
        num_samples=config['num_samples'],
        seed=args.seed,
    )
    
    # Split into train/val
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * config['val_split'])
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Create datasets
    train_dataset = CodeDataset(train_samples, tokenizer, config['max_length'])
    val_dataset = CodeDataset(val_samples, tokenizer, config['max_length'])
    
    # Custom collator for dynamic padding (more efficient than padding to max_length)
    collate_fn = lambda batch: collate_with_padding(batch, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # Load model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=DTYPE,
    )
    model.to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    
    # Replace with CIMBA attention
    print(f"\nReplacing attention with CIMBAAttention (S={config['bandwidth']})")
    model = replace_attention_with_cimba(model, bandwidth=config['bandwidth'])
    
    # Setup trainable parameters (LoRA or full projection finetuning)
    if config['use_lora']:
        # LoRA: More stable/efficient training with low-rank adapters
        print(f"\nSetting up LoRA (rank={config['lora_rank']}, alpha={config['lora_alpha']})")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("LoRA requires peft library. Install with: pip install peft")
        
        # Target the attention projection modules
        # Note: Module names may vary by model architecture
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in trainable_params)
        total = sum(p.numel() for p in model.parameters())
        print(f"\nLoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # Sanity check: print sample trainable param names to confirm LoRA adapters
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Sample trainable params: {trainable_names[:4]}")
        
    else:
        # Full projection finetuning (original behavior)
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Note: CIMBAAttention currently has NO dedicated learnable params.
        # We're training the attention projections (q/k/v/o_proj) to adapt to the new kernel.
        # This is expensive (~44M params) but necessary since there are no CIMBA-specific params.
        trainable_params = []
        layer_counts = {}  # Track params per layer for verification
        
        for name, param in model.named_parameters():
            # First check for CIMBA-specific params (currently none exist)
            if any(k in name for k in ["lowpass_param", "spectral_sharpness", "spectral_filter"]):
                param.requires_grad = True
                trainable_params.append(param)
                print(f"  Trainable (CIMBA): {name}")
            # Fall back to attention projections if no CIMBA params found
            elif any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']) and 'attn' in name.lower():
                param.requires_grad = True
                trainable_params.append(param)
                # Extract layer number for counting
                import re
                match = re.search(r'layers\.(\d+)', name)
                layer_idx = int(match.group(1)) if match else -1
                layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + param.numel()
        
        # Report what we're training
        cimba_params = [n for n, _ in model.named_parameters() if any(k in n for k in ["lowpass_param", "spectral_sharpness", "spectral_filter"])]
        if cimba_params:
            print(f"  Training {len(cimba_params)} CIMBA-specific parameters")
        else:
            print("  No CIMBA-specific params found - training attention projections instead")
            print(f"  Layers with trainable attn projections: {sorted(layer_counts.keys())}")
            print(f"  Params per layer: {layer_counts.get(0, 0):,} (x{len(layer_counts)} layers)")
            print(f"  (Consider using --use-lora for more stable training)")
        
        trainable = sum(p.numel() for p in trainable_params)
        total = count_parameters(model, trainable_only=False)
        print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Optimizer - LoRA typically uses higher LR
    effective_lr = config['learning_rate'] * 2 if config['use_lora'] else config['learning_rate']
    effective_min_lr = config['min_lr'] * 2 if config['use_lora'] else config['min_lr']
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=effective_lr,
        weight_decay=0.0,  # No weight decay for attention projections
    )
    print(f"  Effective LR: {effective_lr:.2e}" + (" (2x for LoRA)" if config['use_lora'] else ""))
    
    # Calculate optimizer steps correctly:
    # - We step when (batch_idx + 1) % grad_accum == 0
    # - This means floor(batches / grad_accum) steps per epoch PLUS 1 if there's a remainder
    # - We handle the remainder explicitly at epoch end, so use ceil
    batches_per_epoch = len(train_loader)
    grad_accum = config['gradient_accumulation']
    opt_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)  # ceil because we handle remainder
    total_steps = opt_steps_per_epoch * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    # Custom cosine schedule with minimum LR floor
    # This prevents LR from going to ~0 which kills learning in later epochs
    # Use effective_lr (which is 2x for LoRA) for scheduler peak
    max_lr = effective_lr
    min_lr = effective_min_lr
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / warmup_steps
        else:
            # Cosine decay from max_lr to min_lr
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale to [min_lr/max_lr, 1.0] range
            return min_lr / max_lr + (1 - min_lr / max_lr) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nTraining setup:")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Optimizer steps per epoch: {opt_steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps} (for {config['epochs']} epochs)")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  LR schedule: warmup to {max_lr:.2e}, cosine decay to {min_lr:.2e}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'humaneval_pass1': [],
    }
    
    # Initial HumanEval evaluation
    # Note: With pretrained weights + correct CIMBAAttention, initial performance
    # should be non-trivial (compare vs baseline + spectral hard-bandlimit)

    if args.eval_humaneval and not args.skip_initial_eval:
        print("\nInitial HumanEval evaluation...")
        initial_pass1 = evaluate_humaneval_quick(model, tokenizer, num_problems=20)
        print(f"Initial pass@1: {initial_pass1:.1%}")
        history['humaneval_pass1'].append({'step': 0, 'pass1': initial_pass1})
    elif args.skip_initial_eval:
        print("\nSkipping initial HumanEval evaluation...")
        history['humaneval_pass1'].append({'step': 0, 'pass1': None})  # None = skipped, not 0%
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{config['epochs']}")
        print("="*50)
        
        # Training
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / config['gradient_accumulation']
            loss.backward()
            epoch_loss += loss.item() * config['gradient_accumulation']
            
            if (step + 1) % config['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * config["gradient_accumulation"]:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
        
        # Handle remainder mini-batches (don't drop partial accumulation)
        if (step + 1) % config['gradient_accumulation'] != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"\nTrain loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        print(f"Val loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}")
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
        
        # Generation test
        print("\nGeneration test:")
        test_prompts = ["def fibonacci(n):", "def binary_search(arr, target):"]
        for prompt in test_prompts:
            result = test_generation(model, tokenizer, prompt, max_tokens=60)
            print(f"\n{prompt}")
            print(result[:300] + "..." if len(result) > 300 else result)
        
        # HumanEval evaluation
        
        if args.eval_humaneval:
            print("\nHumanEval evaluation (20 problems)...")
            pass1 = evaluate_humaneval_quick(model, tokenizer, num_problems=20)
            print(f"pass@1: {pass1:.1%}")
            history['humaneval_pass1'].append({
                'step': global_step,
                'epoch': epoch + 1,
                'pass1': pass1,
            })
        
        # Save checkpoint
        checkpoint_dir = output_dir / f"checkpoint-epoch{epoch + 1}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    # Final model save
    print("\nSaving final model...")
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
    # LoRA merge option - merge adapters into base model
    if config['use_lora'] and args.merge_lora:
        print("\nMerging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir / "merged_model")
        tokenizer.save_pretrained(output_dir / "merged_model")
        print(f"Merged model saved to {output_dir / 'merged_model'}")
    
    # Final HumanEval (full 164 problems)

    if args.eval_humaneval:
        print("\nFinal HumanEval evaluation (full 164 problems)...")
        from eval_humaneval import load_humaneval, evaluate_humaneval
        problems = load_humaneval()
        results = evaluate_humaneval(model, tokenizer, problems, verbose=True)
    
        history['final_humaneval'] = {
            'pass1': results['pass@1'],
            'passed': results['passed'],
            'total': results['total'],
        }
    
        # Save final results
        with open(output_dir / "humaneval_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Save final history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    if args.eval_humaneval and 'final_humaneval' in history:
        print(f"Final HumanEval pass@1: {history['final_humaneval']['pass1']:.1%}")
    
    return model, tokenizer, history


def main():
    parser = argparse.ArgumentParser(description="Scaled CIMBA training")
    
    # Data
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of training samples to collect")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length in tokens")
    
    # Model
    parser.add_argument("--bandwidth", type=int, default=64,
                        help="CIMBA bandwidth S parameter")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak learning rate (doubled automatically for LoRA)")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate floor (prevents decay to 0)")
    
    # LoRA
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA for more stable/efficient training")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (higher = more capacity, more params)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor, typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--merge-lora", action="store_true",
                        help="Merge LoRA weights into base model after training")
    
    # Evaluation
    parser.add_argument("--eval-humaneval", action="store_true",
                        help="Run HumanEval evaluation at checkpoints")
    parser.add_argument("--skip-initial-eval", action="store_true",
                        help="Skip initial HumanEval evaluation")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/cimba_scaled",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train_cimba_scaled(args)


if __name__ == "__main__":
    main()
