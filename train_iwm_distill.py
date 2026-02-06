"""
IWM Attention Knowledge Distillation Training

Train IWM attention to match standard attention outputs.
This teaches the spectral coefficients what patterns to produce.

Key insight from CIMBA paper:
- The moment sketch V_k = Σ_i e^{ikθ_i} v_i can exactly compute Att_g(τ)
- But g (the kernel) must be chosen to approximate standard attention
- We learn the kernel coefficients by distillation

Teacher: Standard softmax attention (full Q·K·V)
Student: IWM attention (position-based with learned coefficients)
Loss: MSE(student_output, teacher_output.detach())
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from iwm_attention import (
    IWMContentKernelAttention,
    replace_attention_with_iwm,
    get_iwm_diagnostics,
)


@dataclass
class DistillConfig:
    """Configuration for distillation training."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    bandwidth: int = 64
    
    # Training
    num_samples: int = 5000
    max_length: int = 384
    batch_size: int = 2
    gradient_accumulation: int = 4
    epochs: int = 3
    
    # Learning rates
    learning_rate: float = 1e-4
    coeff_lr_mult: float = 10.0  # Multiplier for coefficient LR
    qv_lr_mult: float = 1.0     # Multiplier for Q/V projection LR
    
    # Loss weights
    distill_weight: float = 1.0   # Weight for distillation loss
    lm_weight: float = 0.1        # Weight for LM loss (optional, helps maintain generation)
    
    # Other
    warmup_ratio: float = 0.1
    min_lr: float = 1e-6
    seed: int = 42
    output_dir: str = "outputs/iwm_distill"


class TeacherStudentModel(nn.Module):
    """
    Wrapper that holds both teacher (standard attention) and student (IWM).
    
    For efficiency, we share everything except attention layers.
    Teacher uses the original Qwen attention, student uses IWM.
    """
    
    def __init__(self, model_name: str, bandwidth: int, device: str = "cuda"):
        super().__init__()
        
        print(f"Loading teacher model (standard attention)...")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        # Freeze teacher completely
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        print(f"Loading student model (IWM attention, S={bandwidth})...")
        self.student = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.student = replace_attention_with_iwm(self.student, bandwidth=bandwidth)
        
        # Freeze most of student, only train attention-related params
        for param in self.student.parameters():
            param.requires_grad = False
        
        # Enable training for IWM-specific params
        self.trainable_params = []
        self.coeff_params = []
        self.qv_params = []
        
        for name, param in self.student.named_parameters():
            if 'coeff_proj' in name:
                param.requires_grad = True
                self.coeff_params.append(param)
            elif 'q_proj' in name or 'v_proj' in name:
                param.requires_grad = True
                self.qv_params.append(param)
        
        self.trainable_params = self.coeff_params + self.qv_params
        
        total_trainable = sum(p.numel() for p in self.trainable_params)
        print(f"Trainable parameters: {total_trainable:,}")
        print(f"  Coefficient params: {sum(p.numel() for p in self.coeff_params):,}")
        print(f"  Q/V params: {sum(p.numel() for p in self.qv_params):,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing both teacher and student outputs.
        
        Returns:
            dict with:
                - teacher_hidden: Teacher's hidden states [batch, seq, hidden]
                - student_hidden: Student's hidden states [batch, seq, hidden]
                - student_logits: Student's logits [batch, seq, vocab]
                - student_loss: Student's LM loss (if labels provided)
        """
        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Use last hidden state before LM head
            teacher_hidden = teacher_out.hidden_states[-1]
        
        # Student forward
        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        student_hidden = student_out.hidden_states[-1]
        
        result = {
            'teacher_hidden': teacher_hidden.detach(),
            'student_hidden': student_hidden,
            'student_logits': student_out.logits,
        }
        
        if labels is not None:
            result['student_loss'] = student_out.loss
        
        return result


def compute_distillation_loss(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute distillation loss between teacher and student hidden states.
    
    Only compute loss on non-padding tokens.
    """
    # Mask: [batch, seq] -> [batch, seq, 1]
    mask = attention_mask.unsqueeze(-1).float()
    
    # MSE loss
    diff = (student_hidden - teacher_hidden) ** 2
    
    # Masked mean
    masked_diff = diff * mask
    loss = masked_diff.sum() / (mask.sum() * teacher_hidden.shape[-1] + 1e-8)
    
    return loss


def is_likely_single_function(code: str) -> bool:
    """Check if code is likely a single function/class definition."""
    lines = code.strip().split('\n')
    if not lines:
        return False
    
    first_line = lines[0].strip()
    if not (first_line.startswith('def ') or 
            first_line.startswith('class ') or
            first_line.startswith('async def ')):
        return False
    
    # Count top-level definitions
    def_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not line[0].isspace():
            if (stripped.startswith('def ') or 
                stripped.startswith('class ') or
                stripped.startswith('async def ')):
                def_count += 1
    
    return def_count == 1


def load_code_dataset(tokenizer, config: DistillConfig):
    """Load and prepare code dataset."""
    print(f"Loading codeparrot/codeparrot-clean (streaming)...")
    
    dataset = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=config.seed, buffer_size=10000)
    
    samples = []
    skipped_long = 0
    skipped_multi = 0
    
    print(f"Filtering samples (target: {config.num_samples})...")
    pbar = tqdm(total=config.num_samples, desc="Collecting samples")
    
    for example in dataset:
        if len(samples) >= config.num_samples:
            break
        
        code = example['content']
        
        if not is_likely_single_function(code):
            skipped_multi += 1
            continue
        
        code_with_eos = code + tokenizer.eos_token
        
        encoding = tokenizer(
            code_with_eos,
            truncation=True,
            max_length=config.max_length + 1,
            return_tensors='pt',
            add_special_tokens=False,
        )
        
        token_length = encoding['input_ids'].shape[1]
        
        if token_length > config.max_length:
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
    
    return samples


class CodeDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        code = self.samples[idx]['code'] + self.tokenizer.eos_token
        encoding = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


def train_distillation(config: DistillConfig):
    """Main distillation training loop."""
    
    print("=" * 70)
    print("IWM ATTENTION KNOWLEDGE DISTILLATION")
    print("=" * 70)
    print(f"\nDistillation weight: {config.distill_weight}")
    print(f"LM weight: {config.lm_weight}")
    print(f"Bandwidth S: {config.bandwidth}")
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    samples = load_code_dataset(tokenizer, config)
    
    # Split train/val
    val_size = min(500, len(samples) // 10)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    train_dataset = CodeDataset(train_samples, tokenizer, config.max_length)
    val_dataset = CodeDataset(val_samples, tokenizer, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create teacher-student model
    print("\n" + "=" * 70)
    model = TeacherStudentModel(config.model_name, config.bandwidth)
    
    # Optimizer with different LRs
    coeff_lr = config.learning_rate * config.coeff_lr_mult
    qv_lr = config.learning_rate * config.qv_lr_mult
    
    print(f"\nLearning rates:")
    print(f"  Coefficient LR: {coeff_lr:.1e}")
    print(f"  Q/V LR: {qv_lr:.1e}")
    
    optimizer = torch.optim.AdamW([
        {'params': model.coeff_params, 'lr': coeff_lr},
        {'params': model.qv_params, 'lr': qv_lr},
    ], weight_decay=0.01)
    
    # LR schedule
    batches_per_epoch = len(train_loader)
    grad_accum = config.gradient_accumulation
    opt_steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    total_steps = opt_steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return config.min_lr / coeff_lr + (1 - config.min_lr / coeff_lr) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nTraining setup:")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training history
    history = {
        'train_distill_loss': [],
        'train_lm_loss': [],
        'val_distill_loss': [],
    }
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    global_step = 0
    best_val_loss = float('inf')
    diag_log_interval = 50
    
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{config.epochs}")
        print(f"{'='*50}")
        
        model.student.train()
        epoch_distill_loss = torch.tensor(0.0, device='cuda')
        epoch_lm_loss = torch.tensor(0.0, device='cuda')
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        for step, batch in pbar:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            
            # Labels for LM loss (shift happens inside model)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels if config.lm_weight > 0 else None)
            
            # Distillation loss
            distill_loss = compute_distillation_loss(
                outputs['teacher_hidden'],
                outputs['student_hidden'],
                attention_mask,
            )
            
            # Combined loss
            loss = config.distill_weight * distill_loss
            
            if config.lm_weight > 0 and 'student_loss' in outputs:
                lm_loss = outputs['student_loss']
                loss = loss + config.lm_weight * lm_loss
                epoch_lm_loss += lm_loss.detach().float()
            
            # Scale for gradient accumulation
            loss = loss / config.gradient_accumulation
            loss.backward()
            
            epoch_distill_loss += distill_loss.detach().float()
            
            # Optimizer step
            if (step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log diagnostics
                if global_step % diag_log_interval == 0:
                    diag = get_iwm_diagnostics(model.student, compute_kernel_neg=False)
                    tqdm.write(f"  [Step {global_step}] "
                              f"distill={distill_loss.item():.4f} | "
                              f"c0: [{diag['c0_min']:.3f}, {diag['c0_mean']:.3f}, {diag['c0_max']:.3f}] | "
                              f"ck/c0: {diag['ck_ratio_mean']:.4f}")
            
            # Use .detach().float() to avoid GPU sync on every iteration
            # The postfix will show last computed value (slightly stale but much faster)
            if (step + 1) % 10 == 0:  # Update postfix less frequently
                pbar.set_postfix({
                    'distill': f'{distill_loss.detach().float():.4f}',
                    'lm': f'{outputs["student_loss"].detach().float():.4f}' if config.lm_weight > 0 and 'student_loss' in outputs else 'N/A',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                })
        
        # Epoch stats (single GPU sync at end of epoch)
        avg_distill = (epoch_distill_loss / len(train_loader)).item()
        avg_lm = (epoch_lm_loss / len(train_loader)).item() if config.lm_weight > 0 else 0
        print(f"\nEpoch {epoch + 1} - Train distill loss: {avg_distill:.4f}, LM loss: {avg_lm:.4f}")
        
        history['train_distill_loss'].append(avg_distill)
        history['train_lm_loss'].append(avg_lm)
        
        # Validation
        print("Validating...")
        model.student.eval()
        val_distill_loss = torch.tensor(0.0, device='cuda')
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                
                outputs = model(input_ids, attention_mask)
                distill_loss = compute_distillation_loss(
                    outputs['teacher_hidden'],
                    outputs['student_hidden'],
                    attention_mask,
                )
                val_distill_loss += distill_loss
        
        avg_val_distill = (val_distill_loss / len(val_loader)).item()
        print(f"Val distill loss: {avg_val_distill:.4f}")
        history['val_distill_loss'].append(avg_val_distill)
        
        # Save best model
        if avg_val_distill < best_val_loss:
            best_val_loss = avg_val_distill
            print(f"New best! Saving to {output_dir / 'best_model'}")
            model.student.save_pretrained(output_dir / 'best_model')
            tokenizer.save_pretrained(output_dir / 'best_model')
        
        # Generation test
        print("\nGeneration test (student):")
        model.student.eval()
        test_prompt = "def fibonacci(n):\n"
        inputs = tokenizer(test_prompt, return_tensors='pt').to('cuda')
        
        with torch.no_grad():
            gen_ids = model.student.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n{generated}\n")
        
        # Save checkpoint
        checkpoint_dir = output_dir / f'checkpoint-epoch{epoch + 1}'
        model.student.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final model
    model.student.save_pretrained(output_dir / 'final_model')
    tokenizer.save_pretrained(output_dir / 'final_model')
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val distill loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="IWM Attention Knowledge Distillation")
    parser.add_argument("--bandwidth", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--lm-weight", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="outputs/iwm_distill")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = DistillConfig(
        bandwidth=args.bandwidth,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.lr,
        distill_weight=args.distill_weight,
        lm_weight=args.lm_weight,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    train_distillation(config)


if __name__ == "__main__":
    main()
