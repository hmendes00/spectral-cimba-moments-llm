"""
FMMA Inference Evaluation Script

Loads the latest checkpoint from a training run, evaluates perplexity,
KL divergence vs teacher, generation quality, and convergence diagnostics.

Usage:
    python test_fmma_inference.py
    python test_fmma_inference.py --checkpoint-dir outputs/fmma_50k_overnight
    python test_fmma_inference.py --checkpoint-dir outputs/fmma_50k_overnight/checkpoint-20000
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from fmma_attention import (
    FMMAAttention,
    replace_attention_with_fmma,
    create_fmma_cache,
)


# ============================================================================
# Checkpoint Discovery
# ============================================================================

def find_latest_checkpoint(output_dir: str) -> Path:
    """Find the latest checkpoint in a training output directory.
    
    If the path itself is a checkpoint (contains model.safetensors), use it.
    Otherwise, find the checkpoint-N subdirectory with the highest N.
    """
    output_path = Path(output_dir)
    
    # Direct checkpoint path
    if (output_path / "model.safetensors").exists():
        return output_path
    
    # final_model takes priority if it exists
    final_model = output_path / "final_model"
    if (final_model / "model.safetensors").exists():
        print(f"Found final_model at {final_model}")
        return final_model
    
    # Find checkpoint-N directories
    checkpoints = []
    for entry in output_path.iterdir():
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", entry.name)
            if match and (entry / "model.safetensors").exists():
                checkpoints.append((int(match.group(1)), entry))
    
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {output_dir}. "
            f"Expected checkpoint-N/ subdirectories with model.safetensors."
        )
    
    checkpoints.sort(key=lambda x: x[0])
    step, path = checkpoints[-1]
    print(f"Found {len(checkpoints)} checkpoints, using latest: step {step}")
    return path


# ============================================================================
# Model Loading
# ============================================================================

def load_fmma_student(
    checkpoint_path: Path,
    base_model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    num_channels: int = 24,
    bandwidth: int = 64,
    device: str = "cuda",
) -> tuple:
    """Load FMMA student model from checkpoint.
    
    Strategy: load base model → replace attention with FMMA (same r, S) →
    load checkpoint weights on top (overwriting both FMMA and shared params).
    """
    print(f"Loading base model: {base_model}")
    student = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float32, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Replacing attention with FMMA (r={num_channels}, S={bandwidth})...")
    student = replace_attention_with_fmma(
        student, num_channels=num_channels, bandwidth=bandwidth,
    )
    
    # Load trained weights from checkpoint
    print(f"Loading checkpoint weights from {checkpoint_path}")
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path / "model.safetensors")
    
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (may be expected for new FMMA params)")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
        for k in unexpected[:5]:
            print(f"    - {k}")
    
    student = student.to(device).eval()
    
    # Count FMMA layers
    fmma_count = sum(1 for m in student.modules() if isinstance(m, FMMAAttention))
    total_params = sum(p.numel() for p in student.parameters())
    print(f"  FMMA layers: {fmma_count}, Total params: {total_params:,}")
    
    return student, tokenizer


def load_teacher(
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    device: str = "cuda",
) -> tuple:
    """Load the teacher model (standard softmax attention) in float16."""
    print(f"Loading teacher model: {model_name} (float16)")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return teacher, tokenizer


# ============================================================================
# Metrics
# ============================================================================

@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list, max_length: int = 512,
                       device: str = "cuda", use_fmma_cache: bool = False) -> dict:
    """Compute perplexity on a list of text samples."""
    total_nll = 0.0
    total_tokens = 0
    per_sample_ppls = []
    
    for text in texts:
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False,
        )
        input_ids = encodings["input_ids"].to(device)
        
        if input_ids.shape[1] < 2:
            continue
        
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        
        # Per-token cross-entropy
        loss_per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        )
        
        nll = loss_per_token.sum().item()
        n_tokens = shift_labels.numel()
        
        total_nll += nll
        total_tokens += n_tokens
        
        sample_ppl = math.exp(nll / n_tokens) if n_tokens > 0 else float("inf")
        per_sample_ppls.append(sample_ppl)
    
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_nll) if avg_nll < 100 else float("inf")
    
    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "num_samples": len(per_sample_ppls),
        "median_ppl": sorted(per_sample_ppls)[len(per_sample_ppls) // 2] if per_sample_ppls else float("inf"),
    }


@torch.no_grad()
def compute_kl_vs_teacher(
    student, teacher, tokenizer, texts: list,
    max_length: int = 512, device: str = "cuda",
    temperature: float = 2.0,
) -> dict:
    """Compute KL divergence between student and teacher logits."""
    total_kl = 0.0
    total_tokens = 0
    
    for text in texts:
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=False,
        )
        input_ids = encodings["input_ids"].to(device)
        
        if input_ids.shape[1] < 2:
            continue
        
        teacher_logits = teacher(input_ids=input_ids).logits.float()
        student_logits = student(input_ids=input_ids).logits.float()
        
        # KL(teacher || student) with temperature
        t_soft = F.softmax(teacher_logits / temperature, dim=-1)
        s_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
        
        kl = F.kl_div(s_log_soft, t_soft, reduction="none").sum(dim=-1)
        
        total_kl += kl.sum().item()
        total_tokens += kl.numel()
    
    avg_kl = total_kl / total_tokens if total_tokens > 0 else float("inf")
    # Report both raw and T²-scaled (what the training loop reports)
    return {
        "kl_raw": avg_kl,
        "kl_scaled_T2": avg_kl * (temperature ** 2),
        "temperature": temperature,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def compute_density_health(model) -> dict:
    """Check FMMA density nonnegativity and moment statistics across layers."""
    stats = []
    for name, module in model.named_modules():
        if not isinstance(module, FMMAAttention):
            continue
        
        layer_idx = module.layer_idx
        info = {"layer": layer_idx}
        
        # Check query moment network output statistics
        # Create a small random input to probe
        device = next(module.parameters()).device
        dummy_q = torch.randn(1, module.num_heads, 4, module.head_dim, device=device)
        moments = module.query_moments(dummy_q)  # [1, H, 4, r, 2S+1]
        
        S = module.bandwidth
        mu_0 = moments[..., S].real  # Should be ≈ 1.0
        mu_pos = moments[..., S + 1:]  # k=1…S
        
        info["mu_0_mean"] = mu_0.mean().item()
        info["mu_0_std"] = mu_0.std().item()
        info["mu_pos_mag_mean"] = mu_pos.abs().mean().item()
        info["mu_pos_mag_max"] = mu_pos.abs().max().item()
        info["has_nan"] = bool(torch.isnan(moments).any())
        info["has_inf"] = bool(torch.isinf(moments).any())
        
        stats.append(info)
    
    # Aggregate
    all_mu0 = [s["mu_0_mean"] for s in stats]
    all_mag = [s["mu_pos_mag_mean"] for s in stats]
    any_nan = any(s["has_nan"] for s in stats)
    any_inf = any(s["has_inf"] for s in stats)
    
    return {
        "num_layers": len(stats),
        "mu_0_mean_across_layers": sum(all_mu0) / len(all_mu0) if all_mu0 else 0,
        "mu_pos_mag_mean_across_layers": sum(all_mag) / len(all_mag) if all_mag else 0,
        "any_nan": any_nan,
        "any_inf": any_inf,
        "per_layer": stats,
    }


def generate_samples(model, tokenizer, prompts: list, max_new_tokens: int = 100,
                     device: str = "cuda", temperature: float = 0.7) -> list:
    """Generate completions for a list of prompts."""
    results = []
    has_fmma = any(isinstance(m, FMMAAttention) for m in model.modules())
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        extra_kwargs = {}
        if has_fmma:
            extra_kwargs["past_key_values"] = create_fmma_cache(model)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                **extra_kwargs,
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_text[len(prompt):]
        results.append({"prompt": prompt, "completion": completion})
    
    return results


# ============================================================================
# Training Curve Analysis
# ============================================================================

def analyze_training_history(output_dir: Path) -> Optional[dict]:
    """Load and analyze training_history.json if available."""
    history_path = output_dir / "training_history.json"
    if not history_path.exists():
        # Try parent dir (if checkpoint_path is checkpoint-N/)
        history_path = output_dir.parent / "training_history.json"
    if not history_path.exists():
        return None
    
    with open(history_path) as f:
        history = json.load(f)
    
    loss_hist = history.get("loss_history", [])
    eval_hist = history.get("eval_history", [])
    
    if not loss_hist:
        return {"raw": history, "analysis": "No loss history recorded yet."}
    
    # Convergence analysis
    steps = [h["step"] for h in loss_hist]
    losses = [h["loss"] for h in loss_hist]
    kl_losses = [h["kl"] for h in loss_hist]
    ce_losses = [h["ce"] for h in loss_hist]
    mom_losses = [h.get("mom", 0) for h in loss_hist]
    
    n = len(losses)
    first_quarter = losses[:max(n // 4, 1)]
    last_quarter = losses[max(3 * n // 4, n - 1):]
    
    analysis = {
        "total_steps_logged": steps[-1] if steps else 0,
        "loss_first_quarter_avg": sum(first_quarter) / len(first_quarter),
        "loss_last_quarter_avg": sum(last_quarter) / len(last_quarter),
        "loss_reduction_pct": (
            (sum(first_quarter) / len(first_quarter) - sum(last_quarter) / len(last_quarter))
            / (sum(first_quarter) / len(first_quarter)) * 100
        ) if first_quarter else 0,
        "kl_first": kl_losses[0] if kl_losses else None,
        "kl_last": kl_losses[-1] if kl_losses else None,
        "ce_first": ce_losses[0] if ce_losses else None,
        "ce_last": ce_losses[-1] if ce_losses else None,
        "mom_active": any(m > 0 for m in mom_losses),
        "mom_last": mom_losses[-1] if mom_losses else 0,
    }
    
    # Eval history
    if eval_hist:
        valid_evals = [e for e in eval_hist if math.isfinite(e.get("kl_loss", float("nan")))]
        if valid_evals:
            analysis["eval_kl_first"] = valid_evals[0]["kl_loss"]
            analysis["eval_kl_last"] = valid_evals[-1]["kl_loss"]
            analysis["eval_ce_first"] = valid_evals[0]["ce_loss"]
            analysis["eval_ce_last"] = valid_evals[-1]["ce_loss"]
            analysis["eval_kl_reduction_pct"] = (
                (valid_evals[0]["kl_loss"] - valid_evals[-1]["kl_loss"])
                / valid_evals[0]["kl_loss"] * 100
            ) if valid_evals[0]["kl_loss"] > 0 else 0
            analysis["num_nan_evals"] = len(eval_hist) - len(valid_evals)
    
    return {"analysis": analysis, "config": history.get("config", {})}


# ============================================================================
# Evaluation Dataset
# ============================================================================

def load_eval_texts(dataset_name: str = "codeparrot/codeparrot-clean-valid",
                    num_samples: int = 100, max_length: int = 512,
                    tokenizer=None) -> list:
    """Load evaluation text samples from HuggingFace."""
    print(f"Loading eval data: {dataset_name} ({num_samples} samples)...")
    ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    
    texts = []
    for sample in ds:
        content = sample.get("content", sample.get("text", ""))
        if len(content) > 100:  # Skip very short samples
            texts.append(content[:max_length * 4])  # Rough char limit
        if len(texts) >= num_samples:
            break
    
    return texts


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FMMA Inference Evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/fmma_50k_overnight",
                        help="Training output dir or specific checkpoint-N path")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--channels", type=int, default=24, help="FMMA channels (must match training)")
    parser.add_argument("--bandwidth", type=int, default=64, help="FMMA bandwidth (must match training)")
    parser.add_argument("--num-eval-samples", type=int, default=100, help="Samples for perplexity/KL eval")
    parser.add_argument("--temperature", type=float, default=2.0, help="KL temperature (match training)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip teacher comparison (saves VRAM)")
    parser.add_argument("--skip-generation", action="store_true", help="Skip sample generation")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length for eval")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FMMA INFERENCE EVALUATION")
    print("=" * 70)
    
    # ── 1. Find checkpoint ──────────────────────────────────────────────
    checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    print(f"Checkpoint: {checkpoint_path}\n")
    
    # ── 2. Training history analysis ────────────────────────────────────
    print("─" * 70)
    print("TRAINING CURVE ANALYSIS")
    print("─" * 70)
    history_info = analyze_training_history(Path(args.checkpoint_dir))
    if history_info and "analysis" in history_info:
        a = history_info["analysis"]
        if isinstance(a, dict):
            print(f"  Steps logged:           {a.get('total_steps_logged', '?')}")
            print(f"  Loss (first quarter):   {a.get('loss_first_quarter_avg', '?'):.4f}")
            print(f"  Loss (last quarter):    {a.get('loss_last_quarter_avg', '?'):.4f}")
            print(f"  Loss reduction:         {a.get('loss_reduction_pct', 0):.1f}%")
            print(f"  KL: {a.get('kl_first', '?'):.4f} → {a.get('kl_last', '?'):.4f}")
            print(f"  CE: {a.get('ce_first', '?'):.4f} → {a.get('ce_last', '?'):.4f}")
            if a.get("mom_active"):
                print(f"  Moment loss (latest):   {a.get('mom_last', 0):.6f}")
            if "eval_kl_first" in a:
                print(f"  Eval KL: {a['eval_kl_first']:.4f} → {a['eval_kl_last']:.4f} "
                      f"({a.get('eval_kl_reduction_pct', 0):.1f}% reduction)")
                print(f"  Eval CE: {a['eval_ce_first']:.4f} → {a['eval_ce_last']:.4f}")
                if a.get("num_nan_evals", 0) > 0:
                    print(f"  ⚠ NaN eval batches: {a['num_nan_evals']}")
            cfg = history_info.get("config", {})
            if cfg:
                print(f"  Training config: r={cfg.get('num_channels')}, S={cfg.get('bandwidth')}, "
                      f"T={cfg.get('temperature')}, lr={cfg.get('learning_rate')}")
                # Auto-detect channels/bandwidth from training config
                if cfg.get("num_channels"):
                    args.channels = cfg["num_channels"]
                if cfg.get("bandwidth"):
                    args.bandwidth = cfg["bandwidth"]
                if cfg.get("temperature"):
                    args.temperature = cfg["temperature"]
                print(f"  (Auto-detected: channels={args.channels}, bandwidth={args.bandwidth}, T={args.temperature})")
        else:
            print(f"  {a}")
    else:
        print("  No training_history.json found (training may still be running).")
    
    # ── 3. Load models ──────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("LOADING MODELS")
    print("─" * 70)
    student, tokenizer = load_fmma_student(
        checkpoint_path, args.base_model, args.channels, args.bandwidth, args.device,
    )
    
    teacher = None
    if not args.skip_teacher:
        teacher, _ = load_teacher(args.base_model, args.device)
    
    # ── 4. Density health check ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("FMMA DENSITY HEALTH CHECK")
    print("─" * 70)
    health = compute_density_health(student)
    print(f"  Layers checked:         {health['num_layers']}")
    print(f"  μ₀ mean (should be 1):  {health['mu_0_mean_across_layers']:.6f}")
    print(f"  |μ_k| mean (k>0):      {health['mu_pos_mag_mean_across_layers']:.6f}")
    print(f"  Any NaN:                {'⚠ YES' if health['any_nan'] else '✓ No'}")
    print(f"  Any Inf:                {'⚠ YES' if health['any_inf'] else '✓ No'}")
    
    # Per-layer detail (summary)
    problem_layers = [s for s in health["per_layer"] if s["has_nan"] or s["has_inf"]
                      or abs(s["mu_0_mean"] - 1.0) > 0.01]
    if problem_layers:
        print(f"  ⚠ Problem layers: {[s['layer'] for s in problem_layers]}")
        for s in problem_layers:
            print(f"    Layer {s['layer']}: μ₀={s['mu_0_mean']:.4f}, "
                  f"|μ_k|_max={s['mu_pos_mag_max']:.4f}, "
                  f"NaN={s['has_nan']}, Inf={s['has_inf']}")
    else:
        print("  ✓ All layers healthy")
    
    # ── 5. Load eval data ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("COMPUTING METRICS")
    print("─" * 70)
    eval_texts = load_eval_texts(num_samples=args.num_eval_samples, tokenizer=tokenizer)
    
    # ── 6. Perplexity ───────────────────────────────────────────────────
    print("\nStudent perplexity...")
    t0 = time.time()
    student_ppl = compute_perplexity(student, tokenizer, eval_texts,
                                     max_length=args.max_seq_length, device=args.device)
    print(f"  Perplexity:       {student_ppl['perplexity']:.2f}")
    print(f"  Avg NLL:          {student_ppl['avg_nll']:.4f}")
    print(f"  Median PPL:       {student_ppl['median_ppl']:.2f}")
    print(f"  Tokens evaluated: {student_ppl['total_tokens']:,}")
    print(f"  Time: {time.time() - t0:.1f}s")
    
    if teacher is not None:
        print("\nTeacher perplexity...")
        t0 = time.time()
        teacher_ppl = compute_perplexity(teacher, tokenizer, eval_texts,
                                         max_length=args.max_seq_length, device=args.device)
        print(f"  Perplexity:       {teacher_ppl['perplexity']:.2f}")
        print(f"  Avg NLL:          {teacher_ppl['avg_nll']:.4f}")
        print(f"  Median PPL:       {teacher_ppl['median_ppl']:.2f}")
        print(f"  Time: {time.time() - t0:.1f}s")
        
        ppl_ratio = student_ppl['perplexity'] / teacher_ppl['perplexity']
        print(f"\n  PPL ratio (student/teacher): {ppl_ratio:.2f}x")
        print(f"  NLL gap: {student_ppl['avg_nll'] - teacher_ppl['avg_nll']:.4f}")
    
    # ── 7. KL divergence vs teacher ─────────────────────────────────────
    if teacher is not None:
        print("\nKL divergence (student vs teacher)...")
        # Use a smaller subset for KL (it's memory-intensive with 2 models)
        kl_texts = eval_texts[:min(50, len(eval_texts))]
        t0 = time.time()
        kl_metrics = compute_kl_vs_teacher(
            student, teacher, tokenizer, kl_texts,
            max_length=args.max_seq_length, device=args.device,
            temperature=args.temperature,
        )
        print(f"  KL (raw):         {kl_metrics['kl_raw']:.4f}")
        print(f"  KL (×T²={args.temperature}²):    {kl_metrics['kl_scaled_T2']:.4f}  ← comparable to training eval_kl")
        print(f"  Temperature:      {kl_metrics['temperature']}")
        print(f"  Time: {time.time() - t0:.1f}s")
    
    # ── 8. Generation samples ───────────────────────────────────────────
    if not args.skip_generation:
        print("\n" + "─" * 70)
        print("GENERATION SAMPLES")
        print("─" * 70)
        
        test_prompts = [
            "# Implement a function to check if a string is a palindrome\ndef is_palindrome(s):",
            "# Binary search in a sorted array\ndef binary_search(arr, target):",
            "# Compute the factorial of n\ndef factorial(n):",
            "# Reverse a linked list\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):",
        ]
        
        student_samples = generate_samples(student, tokenizer, test_prompts,
                                           max_new_tokens=120, device=args.device)
        
        teacher_samples = None
        if teacher is not None:
            teacher_samples = generate_samples(teacher, tokenizer, test_prompts,
                                               max_new_tokens=120, device=args.device)
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n{'─' * 50}")
            short_prompt = prompt.split('\n')[0][:60]
            print(f"Prompt: {short_prompt}...")
            
            if teacher_samples:
                completion = teacher_samples[i]["completion"][:200]
                print(f"\n  Teacher:\n    {completion.replace(chr(10), chr(10) + '    ')}")
            
            completion = student_samples[i]["completion"][:200]
            print(f"\n  Student:\n    {completion.replace(chr(10), chr(10) + '    ')}")
    
    # ── 9. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"  Checkpoint:       {checkpoint_path.name}")
    print(f"  FMMA config:      r={args.channels}, S={args.bandwidth}")
    print(f"  Student PPL:      {student_ppl['perplexity']:.2f}")
    if teacher is not None:
        print(f"  Teacher PPL:      {teacher_ppl['perplexity']:.2f}")
        print(f"  PPL ratio:        {ppl_ratio:.2f}x")
        print(f"  KL (raw):         {kl_metrics['kl_raw']:.4f}")
        print(f"  KL (×T²):        {kl_metrics['kl_scaled_T2']:.4f}")
    print(f"  Density health:   {'✓ OK' if not health['any_nan'] and not health['any_inf'] else '⚠ ISSUES'}")
    
    # Convergence verdict
    if history_info and isinstance(history_info.get("analysis"), dict):
        a = history_info["analysis"]
        reduction = a.get("loss_reduction_pct", 0)
        if reduction > 20:
            verdict = "✓ CONVERGING WELL"
        elif reduction > 5:
            verdict = "~ SLOW CONVERGENCE (may need more steps or higher LR)"
        elif reduction > 0:
            verdict = "⚠ BARELY CONVERGING (check hyperparameters)"
        else:
            verdict = "✗ NOT CONVERGING (loss increased — check for instability)"
        print(f"  Convergence:      {verdict} ({reduction:.1f}% loss reduction)")
    
    print("=" * 70)
    
    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "student_perplexity": student_ppl,
        "density_health": {k: v for k, v in health.items() if k != "per_layer"},
    }
    if teacher is not None:
        results["teacher_perplexity"] = teacher_ppl
        results["kl_vs_teacher"] = kl_metrics
        results["ppl_ratio"] = ppl_ratio
    if history_info:
        results["training_analysis"] = history_info.get("analysis")
    
    results_path = Path(args.checkpoint_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
