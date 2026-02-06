"""
Scaled IWM Training with CodeParrot Dataset

Trains IWM attention coefficient projection layers on real code data.

Key differences from train_cimba_scaled.py:
- Replaces attention with IWMContentKernelAttention
- Trains only the coefficient projection layers (small param count)
- Adds PSD constraint loss for kernel nonnegativity
- Uses O(S) moment cache for inference

Usage:
    python train_iwm_scaled.py --num-samples 5000
    python train_iwm_scaled.py --num-samples 25000 --epochs 3 --eval-humaneval
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from iwm_attention import replace_attention_with_iwm, get_iwm_psd_loss, get_iwm_diagnostics, get_iwm_diversity_loss

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32 if DEVICE == 'cpu' else torch.bfloat16


def is_likely_single_function(code: str) -> bool:
    """
    Heuristic to check if code is likely a single function/class.
    """
    lines = code.strip().split('\n')
    
    if len(lines) < 3:
        return False
    
    top_level_defs = 0
    for line in lines:
        stripped = line.lstrip()
        if line == stripped and (stripped.startswith('def ') or stripped.startswith('class ')):
            top_level_defs += 1
    
    if top_level_defs == 0 or top_level_defs > 2:
        return False
    
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
    """
    print(f"Loading codeparrot/codeparrot-clean (streaming)...")
    
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )
    
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
        
        if not is_likely_single_function(code):
            skipped_multi += 1
            continue
        
        code_with_eos = code + tokenizer.eos_token
        
        encoding = tokenizer(
            code_with_eos,
            truncation=True,
            max_length=max_length + 1,
            return_tensors='pt',
            add_special_tokens=False,
        )
        
        token_length = encoding['input_ids'].shape[1]
        
        if token_length > max_length:
            skipped_long += 1
            continue
        
        samples.append({
            'code': code,
            'token_length': token_length,
        })
        pbar.update(1)
    
    pbar.close()
    
    print(f"\nCollected {len(samples)} samples")
    print(f"Skipped (too long): {skipped_long}")
    print(f"Skipped (multi-def): {skipped_multi}")
    
    lengths = [s['token_length'] for s in samples]
    print(f"Token length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    return samples


class CodeDataset(Dataset):
    """Dataset for code samples."""
    
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
        
        encoding = self.tokenizer(
            code_with_eos,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt',
            add_special_tokens=False,
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
        }


def collate_with_padding(batch, tokenizer):
    """Custom collator with dynamic padding."""
    max_len = max(len(item['input_ids']) for item in batch)
    max_len = ((max_len + 7) // 8) * 8
    
    pad_token_id = tokenizer.pad_token_id
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        input_ids = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        labels = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        
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
    """Test generation with a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_humaneval_quick(model, tokenizer, num_problems: int = 20) -> float:
    """Quick HumanEval evaluation during training."""
    from eval_humaneval import load_humaneval, evaluate_humaneval
    
    problems = load_humaneval()[:num_problems]
    results = evaluate_humaneval(model, tokenizer, problems, verbose=False)
    return results['pass@1']


def train_iwm_scaled(args):
    """Main training function for IWM attention."""
    
    print("="*70)
    print("IWM CONTENT-KERNEL ATTENTION TRAINING")
    print("="*70)
    print()
    print(f"Target samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Bandwidth S: {args.bandwidth}")
    print(f"PSD loss weight: {args.psd_weight}")
    print(f"Last-token-only training: {args.last_token_only}")
    print(f"Supervise K positions: {args.supervise_k}")
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
        'psd_weight': args.psd_weight,
        'last_token_only': args.last_token_only,
        'supervise_k': args.supervise_k,
    }
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    
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
    
    # Replace with IWM attention
    print(f"\nReplacing attention with IWMContentKernelAttention (S={config['bandwidth']})")
    model = replace_attention_with_iwm(model, bandwidth=config['bandwidth'])
    
    # Enable last-token-only training mode if configured
    if config['last_token_only']:
        print("Enabling last-token-only training mode (memory efficient)")
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'training_last_token_only'):
                layer.self_attn.training_last_token_only = True
    
    # Setup trainable parameters - coeff_proj + Q/V projections
    # Training only coeff_proj isn't enough since IWM attention is fundamentally different
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    coeff_params = []
    qv_params = []
    
    for name, param in model.named_parameters():
        if 'coeff_proj' in name:
            param.requires_grad = True
            coeff_params.append(param)
            print(f"  Trainable (coeff): {name} ({param.numel():,} params)")
        elif 'q_proj' in name or 'v_proj' in name:
            param.requires_grad = True
            qv_params.append(param)
            print(f"  Trainable (QV): {name} ({param.numel():,} params)")
    
    trainable = sum(p.numel() for p in coeff_params) + sum(p.numel() for p in qv_params)
    total = count_parameters(model, trainable_only=False)
    
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    print(f"  Coefficient params: {sum(p.numel() for p in coeff_params):,}")
    print(f"  Q/V projection params: {sum(p.numel() for p in qv_params):,}")
    
    # Optimizer with different LR for coeff vs QV
    # coeff_proj needs higher LR, QV projections need conservative LR
    coeff_lr = config['learning_rate'] * 10  # e.g., 1e-4 * 10 = 1e-3
    qv_lr = config['learning_rate'] * 1  # Keep base LR for pretrained weights
    print(f"\nUsing coefficient LR: {coeff_lr:.1e} (base LR * 10)")
    print(f"Using Q/V LR: {qv_lr:.1e} (base LR)")
    
    optimizer = torch.optim.AdamW([
        {'params': coeff_params, 'lr': coeff_lr},
        {'params': qv_params, 'lr': qv_lr},
    ], weight_decay=0.01)
    
    # LR schedule
    batches_per_epoch = len(train_loader)
    grad_accum = config['gradient_accumulation']
    opt_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    total_steps = opt_steps_per_epoch * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    max_lr = coeff_lr  # Use the boosted LR
    min_lr = config['min_lr'] * 10  # Also boost min_lr proportionally
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / max_lr + (1 - min_lr / max_lr) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nTraining setup:")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Optimizer steps per epoch: {opt_steps_per_epoch}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  LR: {max_lr:.2e} -> {min_lr:.2e}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_psd_loss': [],
        'val_loss': [],
        'humaneval_pass1': [],
    }
    
    # Initial evaluation
    if args.eval_humaneval and not args.skip_initial_eval:
        print("\nInitial HumanEval evaluation...")
        initial_pass1 = evaluate_humaneval_quick(model, tokenizer, num_problems=20)
        print(f"Initial pass@1: {initial_pass1:.1%}")
        history['humaneval_pass1'].append({'step': 0, 'pass1': initial_pass1})
    elif args.skip_initial_eval:
        print("\nSkipping initial HumanEval evaluation...")
        history['humaneval_pass1'].append({'step': 0, 'pass1': None})
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    global_step = 0
    best_val_loss = float('inf')
    diag_log_interval = 50  # Log diagnostics every N steps
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{config['epochs']}")
        print("="*50)
        
        # Training
        model.train()
        epoch_loss = torch.tensor(0.0, device=DEVICE)  # Accumulate on GPU
        epoch_psd_loss = torch.tensor(0.0, device=DEVICE)
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # For last-token-only training, modify labels to supervise K positions
            if config['last_token_only']:
                batch_size, seq_len = attention_mask.shape
                positions_range = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
                masked_positions = positions_range * attention_mask.long() - (1 - attention_mask.long()) * seq_len
                last_indices = masked_positions.argmax(dim=1)  # [batch]
                
                # Create labels with K supervised positions (last K valid tokens)
                K = config['supervise_k']
                labels_sparse = torch.full_like(labels, -100)
                batch_idx = torch.arange(batch_size, device=DEVICE)
                
                for k in range(K):
                    # Position: last_idx - k (but clamped to valid range)
                    pos_k = (last_indices - k).clamp(min=0)
                    # Only supervise if this position is valid (not before start)
                    valid_k = (last_indices >= k)
                    labels_sparse[batch_idx[valid_k], pos_k[valid_k]] = labels[batch_idx[valid_k], pos_k[valid_k]]
                
                labels = labels_sparse
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            lm_loss = outputs.loss
            psd_loss = get_iwm_psd_loss(model)
            diversity_loss = get_iwm_diversity_loss(model)
            
            # Combined loss: LM + PSD constraint
            # Diversity loss disabled - using non-zero initialization instead
            loss = (lm_loss + config['psd_weight'] * psd_loss) / config['gradient_accumulation']
            loss.backward()
            
            # Accumulate without .item() (avoid GPU sync)
            epoch_loss += lm_loss.detach()
            epoch_psd_loss += psd_loss.detach() if isinstance(psd_loss, torch.Tensor) else psd_loss
            
            if (step + 1) % config['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log diagnostics periodically
                if global_step % diag_log_interval == 0:
                    compute_kernel = (global_step % (diag_log_interval * 5) == 0)  # Expensive, less often
                    diag = get_iwm_diagnostics(model, compute_kernel_neg=compute_kernel)
                    tqdm.write(f"  [Step {global_step}] denom: min={diag['denom_min']:.2e}, "
                              f"mean={diag['denom_mean']:.2e}, max={diag['denom_max']:.2e} | "
                              f"c0: [{diag['c0_min']:.3f}, {diag['c0_mean']:.3f}, {diag['c0_max']:.3f}] | "
                              f"ck/c0: {diag['ck_ratio_mean']:.4f} (max={diag['ck_ratio_max']:.4f})"
                              + (f" | kernel_neg={diag['kernel_negative_pct']:.1f}%" if compute_kernel else ""))
            
            pbar.set_postfix({
                'lm': f'{lm_loss.detach().float():.4f}',
                'psd': f'{(psd_loss.detach() if isinstance(psd_loss, torch.Tensor) else psd_loss):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
        
        # Handle remainder
        if (step + 1) % config['gradient_accumulation'] != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Convert accumulated tensors to floats (single GPU sync at end of epoch)
        avg_train_loss = (epoch_loss / len(train_loader)).item() if isinstance(epoch_loss, torch.Tensor) else epoch_loss / len(train_loader)
        avg_psd_loss = (epoch_psd_loss / len(train_loader)).item() if isinstance(epoch_psd_loss, torch.Tensor) else epoch_psd_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_psd_loss'].append(avg_psd_loss)
        print(f"\nTrain LM loss: {avg_train_loss:.4f}")
        print(f"Train PSD loss: {avg_psd_loss:.6f}")
        
        # End-of-epoch diagnostics (with kernel negativity)
        epoch_diag = get_iwm_diagnostics(model, compute_kernel_neg=True)
        print(f"Epoch diagnostics:")
        print(f"  Denominator: min={epoch_diag['denom_min']:.2e}, mean={epoch_diag['denom_mean']:.2e}, max={epoch_diag['denom_max']:.2e}")
        print(f"  Moments: V_norm={epoch_diag['moment_v_norm']:.2e}, Z_norm={epoch_diag['moment_z_norm']:.2e}")
        print(f"  Coefficients: c0=[{epoch_diag['c0_min']:.3f}, {epoch_diag['c0_mean']:.3f}, {epoch_diag['c0_max']:.3f}]")
        print(f"  Higher harmonics: ck/c0 mean={epoch_diag['ck_ratio_mean']:.4f}, max={epoch_diag['ck_ratio_max']:.4f}")
        print(f"  Kernel negative %: {epoch_diag['kernel_negative_pct']:.1f}%")
        
        # Store diagnostics in history
        if 'diagnostics' not in history:
            history['diagnostics'] = []
        history['diagnostics'].append({
            'epoch': epoch + 1,
            'step': global_step,
            **epoch_diag
        })
        
        # Validation with last-token accuracy tracking
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_top5_correct = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                # Find last valid indices
                batch_size, seq_len = attention_mask.shape
                positions_range = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
                masked_positions = positions_range * attention_mask.long() - (1 - attention_mask.long()) * seq_len
                last_indices = masked_positions.argmax(dim=1)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                # Get logits at last position for each sample
                logits = outputs.logits  # [batch, seq, vocab]
                batch_idx = torch.arange(batch_size, device=DEVICE)
                last_logits = logits[batch_idx, last_indices]  # [batch, vocab]
                last_labels = labels[batch_idx, last_indices]  # [batch]
                
                # Compute loss manually for last token
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
                valid_mask = (last_labels != -100)
                if valid_mask.any():
                    val_loss += loss_fct(last_logits[valid_mask], last_labels[valid_mask]).item()
                    
                    # Accuracy: top-1
                    predictions = last_logits.argmax(dim=-1)
                    val_correct += (predictions[valid_mask] == last_labels[valid_mask]).sum().item()
                    
                    # Accuracy: top-5
                    top5_preds = last_logits.topk(5, dim=-1).indices
                    top5_match = (top5_preds[valid_mask] == last_labels[valid_mask].unsqueeze(-1)).any(dim=-1)
                    val_top5_correct += top5_match.sum().item()
                    
                    val_total += valid_mask.sum().item()
        
        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_top5_acc = val_top5_correct / max(val_total, 1)
        
        history['val_loss'].append(avg_val_loss)
        if 'val_acc' not in history:
            history['val_acc'] = []
            history['val_top5_acc'] = []
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5_acc)
        
        print(f"Val loss (last-token): {avg_val_loss:.4f}")
        print(f"Val accuracy: top-1={val_acc:.1%}, top-5={val_top5_acc:.1%}")
        
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
    
    # Final HumanEval
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
    parser = argparse.ArgumentParser(description="IWM Content-Kernel Attention Training")
    
    # Data
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of training samples")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    
    # Model
    parser.add_argument("--bandwidth", type=int, default=64,
                        help="IWM bandwidth S parameter")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--psd-weight", type=float, default=1e-3,
                        help="Weight for PSD constraint loss")
    parser.add_argument("--last-token-only", action="store_true", default=True,
                        help="Use last-token teacher forcing (memory efficient)")
    parser.add_argument("--no-last-token-only", action="store_false", dest="last_token_only",
                        help="Use full-sequence training (more memory)")
    parser.add_argument("--supervise-k", type=int, default=1,
                        help="Number of positions to supervise per sequence (1=last only, 8=denser gradients)")
    
    # Evaluation
    parser.add_argument("--eval-humaneval", action="store_true",
                        help="Run HumanEval evaluation")
    parser.add_argument("--skip-initial-eval", action="store_true",
                        help="Skip initial HumanEval evaluation")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/iwm_scaled",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train_iwm_scaled(args)


if __name__ == "__main__":
    main()
