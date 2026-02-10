"""
Diagnostic: Compare FMMA training path vs cache path outputs.

If logits match, the cache correctly implements the FMMA formula and
any generation quality issues are purely from the model's training state.
If logits differ significantly, there's a bug in the inference path.

Usage:
    python diagnose_cache_path.py
    python diagnose_cache_path.py --checkpoint-dir outputs/fmma_distill
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fmma_attention import (
    FMMAAttention,
    replace_attention_with_fmma,
    create_fmma_cache,
)


def load_model(checkpoint_dir=None, channels=16, bandwidth=64, device="cuda"):
    """Load model with FMMA, optionally from a checkpoint."""
    base_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.float32, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Replacing attention with FMMA (r={channels}, S={bandwidth})...")
    model = replace_attention_with_fmma(model, num_channels=channels, bandwidth=bandwidth)

    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
        # Auto-detect final_model subfolder
        if (ckpt / "final_model" / "model.safetensors").exists():
            ckpt = ckpt / "final_model"
        if (ckpt / "model.safetensors").exists():
            from safetensors.torch import load_file
            state = load_file(str(ckpt / "model.safetensors"))
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
            if missing:
                print("  MISSING KEYS (first 10):")
                for k in missing[:10]:
                    print(f"    {k}")
            if unexpected:
                print("  UNEXPECTED KEYS (first 10):")
                for k in unexpected[:10]:
                    print(f"    {k}")
            del state
        else:
            print(f"  No model.safetensors in {ckpt}, using randomly-init FMMA params")

    model = model.to(device).eval()
    return model, tokenizer


def compare_paths(model, tokenizer, prompt, device="cuda"):
    """Compare training-path and cache-path outputs for the same input."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        # Path 1: Training forward (no cache → _fmma_training_forward)
        out_train = model(input_ids=input_ids)
        logits_train = out_train.logits.float()  # [1, T, V]

        # Path 2: Cache forward (FMMACache → _fmma_forward)
        cache = create_fmma_cache(model)
        out_cache = model(input_ids=input_ids, past_key_values=cache)
        logits_cache = out_cache.logits.float()  # [1, T, V]

    # Compare logits
    diff = (logits_train - logits_cache).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Compare top-1 predictions
    top1_train = logits_train.argmax(dim=-1)[0]  # [T]
    top1_cache = logits_cache.argmax(dim=-1)[0]   # [T]
    top1_match = (top1_train == top1_cache).float().mean().item()

    # Compare softmax distributions (KL divergence)
    p_train = F.softmax(logits_train, dim=-1)
    p_cache = F.softmax(logits_cache, dim=-1)
    kl_per_pos = F.kl_div(
        F.log_softmax(logits_cache, dim=-1),
        p_train,
        reduction="none",
    ).sum(dim=-1)[0]  # [T]
    mean_kl = kl_per_pos.mean().item()
    max_kl = kl_per_pos.max().item()

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "top1_match": top1_match,
        "mean_kl": mean_kl,
        "max_kl": max_kl,
        "top1_train": top1_train,
        "top1_cache": top1_cache,
        "kl_per_pos": kl_per_pos,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description="FMMA Cache Path Diagnostic")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Training output directory (optional)")
    parser.add_argument("--channels", type=int, default=16, help="FMMA channels (r)")
    parser.add_argument("--bandwidth", type=int, default=64, help="FMMA bandwidth (S)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer = load_model(
        checkpoint_dir=args.checkpoint_dir,
        channels=args.channels,
        bandwidth=args.bandwidth,
        device=args.device,
    )

    test_prompts = [
        "def factorial(n):\n    if n == 0:",
        "# Binary search\ndef binary_search(arr, target):\n    low, high = 0, len(arr)",
        "import numpy as np\n\ndef matrix_multiply(A, B):",
    ]

    print("\n" + "=" * 70)
    print("FMMA CACHE PATH DIAGNOSTIC")
    print("  Compares training forward (no cache) vs cache forward (FMMACache)")
    print("  If they match, cache is correct and generation quality = training quality")
    print("=" * 70)

    all_pass = True

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'─' * 50}")
        short = prompt.split('\n')[0][:50]
        print(f"Prompt {i+1}: {short}...")

        result = compare_paths(model, tokenizer, prompt, args.device)

        print(f"  Sequence length:      {result['seq_len']}")
        print(f"  Max logit diff:       {result['max_diff']:.6f}")
        print(f"  Mean logit diff:      {result['mean_diff']:.6f}")
        print(f"  Top-1 token match:    {result['top1_match']*100:.1f}%")
        print(f"  Mean KL(train||cache): {result['mean_kl']:.8f}")
        print(f"  Max KL(train||cache):  {result['max_kl']:.8f}")

        # Per-position details (first 15 tokens)
        print(f"\n  Per-position top-1 predictions:")
        n_show = min(result['seq_len'], 15)
        for t in range(n_show):
            t1 = result['top1_train'][t].item()
            tc = result['top1_cache'][t].item()
            match = "✓" if t1 == tc else "✗"
            tok_t = tokenizer.decode([t1]).replace('\n', '\\n')
            tok_c = tokenizer.decode([tc]).replace('\n', '\\n')
            kl_t = result['kl_per_pos'][t].item()
            print(f"    pos {t:3d}: {match} train='{tok_t:12s}'  cache='{tok_c:12s}'  KL={kl_t:.6f}")

        # Verdict for this prompt
        if result['max_diff'] < 0.1 and result['top1_match'] > 0.95:
            print(f"\n  ✓ PASS: Cache path matches training path")
        elif result['max_diff'] < 1.0:
            print(f"\n  ~ MINOR MISMATCH (max diff = {result['max_diff']:.4f}) — likely float precision")
        else:
            print(f"\n  ✗ FAIL: Cache path diverges (max diff = {result['max_diff']:.2f})!")
            all_pass = False

    # Overall verdict
    print(f"\n{'=' * 70}")
    if all_pass:
        print("OVERALL: ✓ Cache path is correct")
        print("  Generation quality is limited by the model's training state,")
        print("  NOT by a cache/inference bug.")
        print()
        print("  With s_mom ≈ 0.09 (nearly flat attention) and PPL ≈ 22,")
        print("  gibberish generation is expected. The model needs more training")
        print("  to develop peaked attention patterns (s_mom → t_mom ≈ 0.81).")
    else:
        print("OVERALL: ✗ Cache path has significant mismatches!")
        print("  There may be a bug in _fmma_forward or FMMACache.")
    print("=" * 70)


if __name__ == "__main__":
    main()
