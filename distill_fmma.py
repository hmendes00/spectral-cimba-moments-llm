"""
FMMA Distillation Script

Trains a student model with FMMA attention to match a teacher model with 
standard softmax attention. This is the key to making FMMA work in practice!

The idea:
1. Teacher: Pretrained Qwen with standard softmax attention (frozen)
2. Student: Same Qwen with FMMA attention (trainable FMMA parameters)
3. Loss: KL divergence between student and teacher logits

This teaches the FMMA components (NonnegativeKeyFeatures, QueryMomentNetwork)
to produce attention patterns that approximate softmax attention.

Key insight: The pretrained model's MLP, embeddings, etc. already know "what to do" -
we just need to teach FMMA how to route information like softmax does.

Per FMMA paper §5.4, training also includes:
- PSD Toeplitz penalty: penalize negative eigenvalues to ensure nonnegative densities
- Bandwidth regularization: penalize high-order moments for smoother densities
"""

import math
import os
import sys
import json
import time
import random
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from fmma_attention import (
    FMMAAttention,
    replace_attention_with_fmma,
    create_fmma_cache,
    psd_toeplitz_penalty,
    bandwidth_regularization,
)
from fmma.moment_distill import (
    phases_from_positions,
    compute_teacher_phase_moments,
    compute_moment_loss,
    select_query_indices,
    select_layers_for_moments,
    select_heads,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DistillConfig:
    """Configuration for FMMA distillation training."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    # FMMA parameters
    num_channels: int = 8       # r: number of nonnegative key feature channels
    bandwidth: int = 64         # S: harmonic bandwidth
    layers_to_replace: Optional[List[int]] = None  # None = all layers
    
    # Dataset
    dataset_name: str = "codeparrot/codeparrot-clean-valid"  # HuggingFace dataset
    dataset_split: str = "train"
    num_samples: int = 1000      # Number of samples to use
    
    # Training
    learning_rate: float = 5e-4  # Higher LR for FMMA params
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 2000
    warmup_steps: int = 100
    max_seq_length: int = 512
    
    # Distillation
    temperature: float = 2.0     # Softmax temperature for KL divergence
    alpha_kl: float = 0.9        # Weight for KL loss
    alpha_ce: float = 0.1        # Weight for CE loss (language modeling)
    
    # FMMA regularization (per paper §5.4)
    alpha_psd: float = 0.01      # Weight for PSD Toeplitz penalty
    alpha_bw: float = 0.001      # Weight for bandwidth regularization
    bw_alpha: float = 2.0        # Exponent for bandwidth regularization
    psd_every: int = 4           # Compute PSD penalty every N steps (eigendecomp is expensive)
    psd_max_samples: int = 64    # Max moment vectors to sample for PSD penalty
    
    # Moment distillation (Softmax → FMMA normalization bridge, per KL-projection paper)
    alpha_mom: float = 0.05          # weight for moment loss (start small)
    alpha_mom_warmup: int = 200      # linearly ramp alpha_mom from 0 over this many steps
    mom_every: int = 1               # compute moment loss every N steps
    mom_layers: Optional[List[int]] = None  # None => sample random FMMA layers each step
    mom_max_layers_per_step: int = 2 # max layers to supervise per step
    mom_head_stride: int = 2         # use every N-th head
    mom_query_stride: int = 8        # use every N-th query position
    mom_max_queries: int = 16        # cap #query positions per batch
    mom_detach_teacher_attn: bool = True
    mom_use_fejer_weights: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    output_dir: str = "outputs/fmma_distill"
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 100
    
    # Seed
    seed: int = 42


# ============================================================================
# Dataset - Load from HuggingFace
# ============================================================================

def load_code_dataset(
    dataset_name: str,
    split: str,
    num_samples: int,
    tokenizer,
    max_length: int,
):
    """
    Load code dataset from HuggingFace.
    
    Supported datasets:
    - codeparrot/codeparrot-clean-valid: Clean Python code
    - bigcode/the-stack-smol: Multi-language code samples
    - openai_humaneval: HumanEval benchmark problems
    """
    print(f"  Loading dataset: {dataset_name} ({split})...")
    
    try:
        if "humaneval" in dataset_name.lower():
            # HumanEval format
            ds = load_dataset("openai/openai_humaneval", split="test")
            text_field = "prompt"  # Use the prompt + canonical_solution
            samples = [
                item["prompt"] + item["canonical_solution"] 
                for item in ds
            ][:num_samples]
        elif "codeparrot" in dataset_name.lower():
            # CodeParrot format
            ds = load_dataset(dataset_name, split=split, streaming=True)
            samples = []
            for item in ds:
                if len(samples) >= num_samples:
                    break
                content = item.get("content", item.get("code", ""))
                if len(content) > 50:  # Skip very short samples
                    samples.append(content)
        elif "the-stack" in dataset_name.lower():
            # The Stack format
            ds = load_dataset(dataset_name, split=split, streaming=True)
            samples = []
            for item in ds:
                if len(samples) >= num_samples:
                    break
                content = item.get("content", "")
                if len(content) > 50:
                    samples.append(content)
        else:
            # Generic: try common field names
            ds = load_dataset(dataset_name, split=split, streaming=True)
            samples = []
            for item in ds:
                if len(samples) >= num_samples:
                    break
                # Try common field names
                content = (
                    item.get("content") or 
                    item.get("code") or 
                    item.get("text") or 
                    item.get("prompt", "")
                )
                if len(content) > 50:
                    samples.append(content)
        
        print(f"  Loaded {len(samples)} samples from {dataset_name}")
        
    except Exception as e:
        print(f"  Warning: Failed to load {dataset_name}: {e}")
        print(f"  Falling back to built-in samples...")
        samples = []
    
    return samples


class CodingDataset(Dataset):
    """Dataset of coding samples for distillation."""
    
    def __init__(
        self,
        tokenizer,
        samples: List[str],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.samples = samples
        self.max_length = max_length
        
        # Tokenize all samples WITHOUT padding
        self.encodings = []
        for sample in tqdm(samples, desc="Tokenizing", leave=False):
            encoding = tokenizer(
                sample,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )
            # Only keep samples with enough tokens
            if encoding["input_ids"].shape[1] >= 32:
                self.encodings.append({
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                })
        
        # Report
        total_tokens = sum(len(e["input_ids"]) for e in self.encodings)
        print(f"  Dataset: {len(self.encodings)} samples, {total_tokens:,} total tokens")
        print(f"  Avg tokens per sample: {total_tokens / max(1, len(self.encodings)):.1f}")
    
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
# Distillation Loss
# ============================================================================

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha_kl: float = 0.9,
    alpha_ce: float = 0.1,
) -> dict:
    """
    Compute distillation loss combining KL divergence and cross-entropy.
    
    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len] with -100 for padding
        temperature: softmax temperature for KL
        alpha_kl: weight for KL divergence loss
        alpha_ce: weight for cross-entropy loss
    
    Returns:
        dict with total_loss, kl_loss, ce_loss
    """
    # Mask for valid tokens (not -100)
    valid_mask = labels != -100
    
    # KL divergence loss (soft targets)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    # KL per token
    kl_per_token = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='none'
    ).sum(dim=-1)  # [batch, seq_len]
    
    # Mask and average
    kl_loss = (kl_per_token * valid_mask).sum() / valid_mask.sum()
    kl_loss = kl_loss * (temperature ** 2)  # Scale by T^2 as per Hinton et al.
    
    # Cross-entropy loss (hard targets)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    
    # Combined loss
    total_loss = alpha_kl * kl_loss + alpha_ce * ce_loss
    
    return {
        'total_loss': total_loss,
        'kl_loss': kl_loss,
        'ce_loss': ce_loss,
    }


# ============================================================================
# Training
# ============================================================================

def setup_optimizer(model, config: DistillConfig):
    """Create optimizer for FMMA parameters only."""
    
    fmma_params = []
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then enable FMMA-specific parameters
    for name, module in model.named_modules():
        if isinstance(module, FMMAAttention):
            # Enable gradients for FMMA components
            for param in module.key_features.parameters():
                param.requires_grad = True
                fmma_params.append(param)
            for param in module.query_moments.parameters():
                param.requires_grad = True
                fmma_params.append(param)
            for param in module.o_proj.parameters():
                param.requires_grad = True
                fmma_params.append(param)
    
    trainable_count = sum(p.numel() for p in fmma_params)
    frozen_count = sum(p.numel() for p in model.parameters()) - trainable_count
    
    print(f"FMMA parameters (trainable): {trainable_count:,}")
    print(f"Frozen parameters: {frozen_count:,}")
    
    if not fmma_params:
        raise RuntimeError("No FMMA parameters found! Check model structure.")
    
    optimizer = torch.optim.AdamW(fmma_params, lr=config.learning_rate)
    return optimizer


def evaluate(student, teacher, eval_loader, config: DistillConfig, max_batches: int = 10):
    """Evaluate student model on a subset of data for speed."""
    student.eval()
    teacher.eval()
    
    total_kl = 0.0
    total_ce = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:  # Limit eval batches for speed
                break
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Teacher forward
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Student forward
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # NaN guard: clamp student logits to prevent NaN in loss
            student_logits = student_outputs.logits
            if not torch.isfinite(student_logits).all():
                student_logits = student_logits.nan_to_num(
                    nan=0.0, posinf=1e4, neginf=-1e4
                )
            
            # Compute loss
            losses = distillation_loss(
                student_logits,
                teacher_outputs.logits,
                labels,
                temperature=config.temperature,
                alpha_kl=config.alpha_kl,
                alpha_ce=config.alpha_ce,
            )
            
            kl_val = losses['kl_loss'].item()
            ce_val = losses['ce_loss'].item()
            num_tokens = attention_mask.sum().item()

            # Skip NaN batches in accumulation
            if math.isfinite(kl_val) and math.isfinite(ce_val):
                total_kl += kl_val * num_tokens
                total_ce += ce_val * num_tokens
                total_tokens += num_tokens
    
    if total_tokens == 0:
        return {'kl_loss': float('nan'), 'ce_loss': float('nan')}

    return {
        'kl_loss': total_kl / total_tokens,
        'ce_loss': total_ce / total_tokens,
    }


def _has_fmma_layers(model) -> bool:
    """Check whether any layer in the model is an FMMAAttention module."""
    from fmma.attention import FMMAAttention
    return any(isinstance(m, FMMAAttention) for m in model.modules())


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Generate a sample completion. Uses FMMACache only for FMMA models."""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Only pass an FMMACache when the model actually has FMMA layers.
    # For the teacher (standard attention), let HF create its own cache.
    extra_kwargs = {}
    if _has_fmma_layers(model):
        extra_kwargs["past_key_values"] = create_fmma_cache(model)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            **extra_kwargs,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def train(config: DistillConfig):
    """Main distillation training loop."""
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load TEACHER (frozen softmax attention) in float16 to save memory
    # Must use attn_implementation="eager" so output_attentions=True works
    # (SDPA/FlashAttention2 silently return None for attentions).
    print("\nLoading teacher model (standard attention, float16, eager)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(config.device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Load STUDENT (FMMA attention)
    print("\nLoading student model (FMMA attention)...")
    student = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    
    # Replace attention with FMMA
    print(f"Replacing attention with FMMA (r={config.num_channels}, S={config.bandwidth})...")
    student = replace_attention_with_fmma(
        student,
        num_channels=config.num_channels,
        bandwidth=config.bandwidth,
        layers_to_replace=config.layers_to_replace,
    )
    student = student.to(config.device)

    # Enable gradient checkpointing on FMMA layers (frees FFT + density
    # intermediates between chunks, saving ~80% activation memory).
    for module in student.modules():
        if isinstance(module, FMMAAttention):
            module.gradient_checkpointing = True

    # Enable HF's gradient checkpointing for the transformer blocks.
    # _fmma_training_forward is now fully deterministic (no conditional
    # tensor creation) so nested checkpointing works correctly.
    if hasattr(student, 'gradient_checkpointing_enable'):
        student.gradient_checkpointing_enable()

    # Setup optimizer and scheduler (also freezes non-FMMA params)
    optimizer = setup_optimizer(student, config)

    # Count parameters (after optimizer setup, which freezes non-FMMA params)
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"\nStudent total parameters: {total_params:,}")
    print(f"Student trainable (FMMA only): {trainable_params:,}")
    print(f"Student frozen: {total_params - trainable_params:,}")

    # Create dataset from HuggingFace
    print("\nPreparing dataset...")
    samples = load_code_dataset(
        dataset_name=config.dataset_name,
        split=config.dataset_split,
        num_samples=config.num_samples,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    dataset = CodingDataset(tokenizer, samples, config.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    # Training loop
    print(f"\nStarting FMMA distillation for {config.max_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Temperature: {config.temperature}, α_KL={config.alpha_kl}, α_CE={config.alpha_ce}")
    print(f"Moment distillation: α_mom={config.alpha_mom}, warmup={config.alpha_mom_warmup} steps, every={config.mom_every}")

    # Pre-compute FMMA layer index list for moment distillation subsampling
    fmma_layer_map = {}  # layer_idx -> module
    for module in student.modules():
        if isinstance(module, FMMAAttention):
            fmma_layer_map[module.layer_idx] = module
    fmma_layer_indices = sorted(fmma_layer_map.keys())
    print(f"FMMA layers available for moment supervision: {fmma_layer_indices}")
    
    student.train()
    global_step = 0
    total_loss = torch.tensor(0.0, device=config.device)  # Keep on GPU to avoid sync
    total_kl = torch.tensor(0.0, device=config.device)
    total_ce = torch.tensor(0.0, device=config.device)
    total_mom = torch.tensor(0.0, device=config.device)
    loss_history = []
    eval_history = []
    
    start_time = time.time()
    pbar = tqdm(total=config.max_steps, desc="Distilling")
    
    while global_step < config.max_steps:
        for batch in dataloader:
            if global_step >= config.max_steps:
                break
            
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # ================================================================
            # Moment distillation: decide whether to compute this step
            # ================================================================
            do_moment_distill = (
                config.alpha_mom > 0
                and global_step % config.mom_every == 0
                and len(fmma_layer_indices) > 0
            )

            # ================================================================
            # Teacher forward (no grad) — runs in float16
            # Request attentions only when moment distillation is active.
            # ================================================================
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=do_moment_distill,
                    return_dict=True,
                )
            teacher_logits = teacher_outputs.logits.detach().float()

            # ================================================================
            # Moment distillation: set up position-aware moment storage
            # Before student forward, tell FMMA layers which query positions
            # to store moments for (PLAN §4).
            # ================================================================
            seq_len = input_ids.shape[1]
            t_idx = None
            mom_layers_this_step = []

            if do_moment_distill and teacher_outputs.attentions is not None:
                # Choose query positions and layers to supervise
                t_idx = select_query_indices(
                    seq_len, config.mom_query_stride, config.mom_max_queries
                )
                mom_layers_this_step = select_layers_for_moments(
                    fmma_layer_indices,
                    config.mom_layers,
                    config.mom_max_layers_per_step,
                )

            # Hold ONLY the teacher attention layers we actually need (saves VRAM).
            # Each attention tensor is [B, H, T, T] — keeping all 24 when we only
            # supervise a few is wasteful.  Detach to drop the teacher graph.
            teacher_attentions = None
            if do_moment_distill and teacher_outputs.attentions is not None:
                needed = set(mom_layers_this_step)
                teacher_attentions = {
                    idx: teacher_outputs.attentions[idx].detach()
                    for idx in needed
                    if idx < len(teacher_outputs.attentions)
                }
            del teacher_outputs
                # NOTE: we no longer set _store_query_positions on FMMA layers.
                # Instead, FMMA unconditionally stores _last_all_query_moments
                # and we index into it here.  This avoids conditional tensor
                # creation inside the checkpointed forward.

            # ================================================================
            # Student forward
            # ================================================================
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Compute distillation loss (teacher_logits already detached + float)
            losses = distillation_loss(
                student_outputs.logits,
                teacher_logits,
                labels,
                temperature=config.temperature,
                alpha_kl=config.alpha_kl,
                alpha_ce=config.alpha_ce,
            )
            
            # ================================================================
            # FMMA regularization losses (per paper §5.4)
            # ================================================================
            psd_loss = torch.tensor(0.0, device=config.device)
            bw_loss = torch.tensor(0.0, device=config.device)
            fmma_count = 0
            
            for module in student.modules():
                if isinstance(module, FMMAAttention):
                    if hasattr(module, '_last_query_moments') and module._last_query_moments is not None:
                        moments = module._last_query_moments
                        
                        # PSD Toeplitz penalty: penalize negative eigenvalues
                        if config.alpha_psd > 0 and global_step % config.psd_every == 0:
                            psd_loss = psd_loss + psd_toeplitz_penalty(
                                moments, reduction='mean', max_samples=config.psd_max_samples
                            )
                        
                        # Bandwidth regularization: penalize high-frequency moments
                        if config.alpha_bw > 0:
                            bw_loss = bw_loss + bandwidth_regularization(
                                moments, module.bandwidth, alpha=config.bw_alpha, reduction='mean'
                            )
                        
                        fmma_count += 1
                        # Clear to avoid memory buildup
                        module._last_query_moments = None
            
            if fmma_count > 0:
                psd_loss = psd_loss / fmma_count
                bw_loss = bw_loss / fmma_count
                losses['psd_loss'] = psd_loss
                losses['bw_loss'] = bw_loss
                losses['total_loss'] = losses['total_loss'] + config.alpha_psd * psd_loss + config.alpha_bw * bw_loss

            # ================================================================
            # Moment distillation loss (PLAN §5–6)
            # ================================================================
            moment_loss = torch.tensor(0.0, device=config.device)

            if do_moment_distill and teacher_attentions is not None and t_idx is not None:
                # Determine max_seq_len from any FMMA module
                fmma_max_seq_len = next(iter(fmma_layer_map.values())).max_seq_len
                fmma_bandwidth = next(iter(fmma_layer_map.values())).bandwidth
                fmma_num_heads = next(iter(fmma_layer_map.values())).num_heads

                # Build phases for positions 0..T-1
                position_ids = torch.arange(seq_len, device=config.device)
                phases = phases_from_positions(position_ids, fmma_max_seq_len)  # [T]
                phases = phases.unsqueeze(0).expand(input_ids.shape[0], -1)     # [B, T]

                # Head subsampling
                h_idx = select_heads(fmma_num_heads, config.mom_head_stride)

                # α_mom warmup schedule (PLAN §10)
                if config.alpha_mom_warmup > 0 and global_step < config.alpha_mom_warmup:
                    alpha_mom_eff = config.alpha_mom * (global_step / config.alpha_mom_warmup)
                else:
                    alpha_mom_eff = config.alpha_mom

                mom_count = 0
                for layer_idx in mom_layers_this_step:
                    module = fmma_layer_map[layer_idx]

                    # Get teacher attention for this layer (dict lookup)
                    if layer_idx not in teacher_attentions:
                        continue
                    teacher_attn = teacher_attentions[layer_idx].float()  # [B, H, T, T]

                    # Compute teacher phase moments
                    m_teacher = compute_teacher_phase_moments(
                        attn_weights=teacher_attn,
                        phases=phases,
                        t_idx=t_idx,
                        bandwidth=fmma_bandwidth,
                        h_idx=h_idx,
                        use_fejer_weights=config.mom_use_fejer_weights,
                        detach=config.mom_detach_teacher_attn,
                    )  # [B, Hsel, Q, 2S+1]

                    # Get student moments at the distillation positions.
                    # We call query_moments lazily on just the ~16 positions
                    # needed, keeping memory low.
                    stored_q = getattr(module, '_last_queries', None)
                    if stored_q is None:
                        continue
                    t_idx_dev = t_idx.to(config.device)
                    q_subset = stored_q[:, :, t_idx_dev]  # [B, H, Q, D]
                    mu_student = module.query_moments(q_subset)  # [B, H, Q, r, 2S+1]

                    # Subsample heads from student to match teacher
                    h_idx_dev = h_idx.to(config.device)
                    mu_student = mu_student[:, h_idx_dev]  # [B, Hsel, Q, r, 2S+1]

                    # Ensure shapes match
                    min_q = min(mu_student.shape[2], m_teacher.shape[2])
                    mu_student = mu_student[:, :, :min_q]
                    m_teacher = m_teacher[:, :, :min_q]

                    # Compute moment loss
                    moment_loss = moment_loss + compute_moment_loss(
                        mu_student=mu_student,
                        m_teacher=m_teacher,
                        bandwidth=fmma_bandwidth,
                        use_fejer_weights=config.mom_use_fejer_weights,
                    )
                    mom_count += 1

                    # Clean up stored queries
                    module._last_queries = None

                if mom_count > 0:
                    moment_loss = moment_loss / mom_count
                    losses['moment_loss'] = moment_loss
                    losses['total_loss'] = losses['total_loss'] + alpha_mom_eff * moment_loss

            # Clean up remaining FMMA modules (in case they weren't selected)
            for module in student.modules():
                if isinstance(module, FMMAAttention):
                    module._last_queries = None

            # Free teacher attentions
            del teacher_attentions

            loss = losses['total_loss'] / config.gradient_accumulation_steps

            # NaN guard: if loss is NaN/Inf, skip this batch entirely.
            # Don't let NaN gradients poison the model parameters.
            if not torch.isfinite(loss):
                if global_step % config.log_every == 0:
                    print(f"  [step {global_step}] NaN/Inf loss detected — skipping batch")
                optimizer.zero_grad()
                global_step += 1
                pbar.update(1)
                continue

            loss.backward()
            
            # Accumulate without .item() to avoid GPU sync (detach to prevent graph growth)
            total_loss += losses['total_loss'].detach()
            total_kl += losses['kl_loss'].detach()
            total_ce += losses['ce_loss'].detach()
            if 'moment_loss' in losses:
                total_mom += losses['moment_loss'].detach()
            
            # Gradient accumulation
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            
            # Logging
            if global_step % config.log_every == 0:
                # Only call .item() at logging time (GPU sync is fine here, infrequent)
                avg_loss = (total_loss / config.log_every).item()
                avg_kl = (total_kl / config.log_every).item()
                avg_ce = (total_ce / config.log_every).item()

                # Collect min-density diagnostic (Fix 3.3 monitoring)
                min_density = None
                for module in student.modules():
                    if isinstance(module, FMMAAttention) and module._last_min_density is not None:
                        if min_density is None or module._last_min_density < min_density:
                            min_density = module._last_min_density
                        module._last_min_density = None  # Reset for next window

                log_entry = {
                    "step": global_step,
                    "loss": avg_loss,
                    "kl": avg_kl,
                    "ce": avg_ce,
                }

                # Moment distillation loss
                avg_mom = (total_mom / config.log_every).item()
                if avg_mom > 0:
                    log_entry["mom"] = avg_mom

                if min_density is not None:
                    log_entry["min_density"] = min_density
                loss_history.append(log_entry)

                postfix = {
                    "loss": f"{avg_loss:.4f}",
                    "kl": f"{avg_kl:.4f}",
                    "ce": f"{avg_ce:.4f}",
                }
                if avg_mom > 0:
                    postfix["mom"] = f"{avg_mom:.4f}"
                if min_density is not None:
                    postfix["min_d"] = f"{min_density:.3f}"
                pbar.set_postfix(postfix)

                total_loss = torch.tensor(0.0, device=config.device)
                total_kl = torch.tensor(0.0, device=config.device)
                total_ce = torch.tensor(0.0, device=config.device)
                total_mom = torch.tensor(0.0, device=config.device)
            
            # Evaluation
            if global_step % config.eval_every == 0:
                eval_metrics = evaluate(student, teacher, dataloader, config)
                eval_history.append({
                    "step": global_step,
                    **eval_metrics
                })
                print(f"\nStep {global_step}: eval_kl={eval_metrics['kl_loss']:.4f}, eval_ce={eval_metrics['ce_loss']:.4f}")
                
                # Generate samples from both models
                prompt = "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):"
                
                print(f"\nPrompt: {prompt[:60]}...")
                teacher_sample = generate_sample(teacher, tokenizer, prompt, max_new_tokens=60)
                student_sample = generate_sample(student, tokenizer, prompt, max_new_tokens=60)
                
                print(f"Teacher: {teacher_sample[len(prompt):len(prompt)+100]}...")
                print(f"Student: {student_sample[len(prompt):len(prompt)+100]}...")
                
                student.train()
            
            # Checkpointing
            if global_step % config.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint-{global_step}"
                student.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
    
    pbar.close()
    
    # Final evaluation (use more batches for final eval)
    elapsed_time = time.time() - start_time
    print(f"\nDistillation completed in {elapsed_time / 60:.1f} minutes")
    
    final_metrics = evaluate(student, teacher, dataloader, config, max_batches=50)
    print(f"Final eval KL: {final_metrics['kl_loss']:.4f}")
    print(f"Final eval CE: {final_metrics['ce_loss']:.4f}")
    
    # Generate final samples
    print("\n" + "="*60)
    print("FINAL SAMPLE GENERATIONS")
    print("="*60)
    
    test_prompts = [
        "# Implement binary search\ndef binary_search(arr, target):",
        "# Implement a stack using a list\nclass Stack:",
        "# Find the maximum element in a list\ndef find_max(arr):",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        teacher_sample = generate_sample(teacher, tokenizer, prompt, max_new_tokens=100)
        student_sample = generate_sample(student, tokenizer, prompt, max_new_tokens=100)
        
        print(f"\nTeacher:\n{teacher_sample}")
        print(f"\nStudent (FMMA):\n{student_sample}")
        print("-" * 40)
    
    # Save training history
    history = {
        "config": {
            "model_name": config.model_name,
            "num_channels": config.num_channels,
            "bandwidth": config.bandwidth,
            "learning_rate": config.learning_rate,
            "temperature": config.temperature,
            "alpha_kl": config.alpha_kl,
            "alpha_ce": config.alpha_ce,
            "max_steps": config.max_steps,
        },
        "loss_history": loss_history,
        "eval_history": eval_history,
        "final_metrics": final_metrics,
        "training_time_minutes": elapsed_time / 60,
    }
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Plot loss curves
    if loss_history:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        steps = [h["step"] for h in loss_history]
        
        # Total loss
        axes[0].plot(steps, [h["loss"] for h in loss_history], 'b-', linewidth=2)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Distillation Loss')
        axes[0].grid(True, alpha=0.3)
        
        # KL loss
        axes[1].plot(steps, [h["kl"] for h in loss_history], 'r-', linewidth=2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('KL Loss')
        axes[1].set_title('KL Divergence (Student vs Teacher)')
        axes[1].grid(True, alpha=0.3)
        
        # CE loss
        axes[2].plot(steps, [h["ce"] for h in loss_history], 'g-', linewidth=2)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('CE Loss')
        axes[2].set_title('Cross-Entropy Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nLoss curves saved to {output_dir / 'loss_curves.png'}")
    
    # Save final model
    final_path = output_dir / "final_model"
    student.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")
    
    return history


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FMMA Distillation Training")
    parser.add_argument("--channels", type=int, default=8, help="Number of key feature channels (r)")
    parser.add_argument("--bandwidth", type=int, default=64, help="Harmonic bandwidth (S)")
    parser.add_argument("--steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--temperature", type=float, default=2.0, help="KL temperature")
    parser.add_argument("--output", type=str, default="outputs/fmma_distill", help="Output directory")
    parser.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean-valid", 
                        help="HuggingFace dataset (codeparrot/codeparrot-clean-valid, openai_humaneval, etc)")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to use")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length")
    # FMMA regularization (per paper §5.4)
    parser.add_argument("--alpha-psd", type=float, default=0.01, help="Weight for PSD Toeplitz penalty")
    parser.add_argument("--alpha-bw", type=float, default=0.001, help="Weight for bandwidth regularization")
    parser.add_argument("--bw-alpha", type=float, default=2.0, help="Exponent for bandwidth regularization")
    parser.add_argument("--psd-every", type=int, default=4, help="Compute PSD penalty every N steps (eigendecomp is expensive)")
    parser.add_argument("--psd-max-samples", type=int, default=64, help="Max moment vectors to sample for PSD penalty")
    # Moment distillation (Softmax → FMMA normalization bridge)
    parser.add_argument("--alpha-mom", type=float, default=0.05, help="Weight for moment distillation loss")
    parser.add_argument("--alpha-mom-warmup", type=int, default=200, help="Steps to linearly ramp alpha_mom from 0")
    parser.add_argument("--mom-every", type=int, default=1, help="Compute moment loss every N steps")
    parser.add_argument("--mom-max-layers", type=int, default=2, help="Max FMMA layers to supervise per step")
    parser.add_argument("--mom-head-stride", type=int, default=2, help="Use every N-th head for moment supervision")
    parser.add_argument("--mom-query-stride", type=int, default=8, help="Use every N-th query position")
    parser.add_argument("--mom-max-queries", type=int, default=16, help="Cap number of query positions per batch")
    
    args = parser.parse_args()
    
    config = DistillConfig(
        num_channels=args.channels,
        bandwidth=args.bandwidth,
        max_steps=args.steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        temperature=args.temperature,
        output_dir=args.output,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        eval_every=args.eval_every,
        max_seq_length=args.max_seq_length,
        alpha_psd=args.alpha_psd,
        alpha_bw=args.alpha_bw,
        bw_alpha=args.bw_alpha,
        psd_every=args.psd_every,
        psd_max_samples=args.psd_max_samples,
        alpha_mom=args.alpha_mom,
        alpha_mom_warmup=args.alpha_mom_warmup,
        mom_every=args.mom_every,
        mom_max_layers_per_step=args.mom_max_layers,
        mom_head_stride=args.mom_head_stride,
        mom_query_stride=args.mom_query_stride,
        mom_max_queries=args.mom_max_queries,
    )
    
    print("="*60)
    print("FMMA DISTILLATION TRAINING")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name} ({config.num_samples} samples)")
    print(f"FMMA: r={config.num_channels} channels, S={config.bandwidth} bandwidth")
    print(f"Regularization: α_psd={config.alpha_psd}, α_bw={config.alpha_bw}")
    print(f"Moment distillation: α_mom={config.alpha_mom}, warmup={config.alpha_mom_warmup}")
    print(f"Device: {config.device}")
    print(f"Max steps: {config.max_steps}")
    print(f"Temperature: {config.temperature}")
    print("="*60)
    
    # Run training
    history = train(config)
    
    # Summary
    print("\n" + "="*60)
    print("DISTILLATION SUMMARY")
    print("="*60)
    
    if history["eval_history"]:
        initial_kl = history["eval_history"][0]["kl_loss"]
        final_kl = history["final_metrics"]["kl_loss"]
        improvement = (initial_kl - final_kl) / initial_kl * 100
        
        print(f"Initial KL loss: {initial_kl:.4f}")
        print(f"Final KL loss: {final_kl:.4f}")
        print(f"KL Improvement: {improvement:.1f}%")
        
        if final_kl < initial_kl:
            print("\n✓ KL DECREASED - FMMA is learning to approximate softmax!")
        else:
            print("\n✗ KL DID NOT DECREASE - May need hyperparameter tuning")
    
    print("="*60)
