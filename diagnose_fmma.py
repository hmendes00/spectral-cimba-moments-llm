"""
Quick FMMA diagnostic: checks whether training path == cache path,
whether FMMA weights actually trained, and what the generation looks like
through each path.

Run:  python diagnose_fmma.py --checkpoint-dir outputs/fmma_50k_overnight
"""

import argparse
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from fmma_attention import FMMAAttention, replace_attention_with_fmma, create_fmma_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/fmma_50k_overnight")
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--bandwidth", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find latest checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    if not (ckpt_dir / "model.safetensors").exists():
        # look for highest checkpoint-N
        import re
        candidates = []
        for p in ckpt_dir.iterdir():
            m = re.match(r"checkpoint-(\d+)", p.name)
            if m and (p / "model.safetensors").exists():
                candidates.append((int(m.group(1)), p))
        if not candidates:
            # try final_model
            fm = ckpt_dir / "final_model"
            if (fm / "model.safetensors").exists():
                ckpt_dir = fm
            else:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        else:
            candidates.sort()
            ckpt_dir = candidates[-1][1]
    print(f"Using checkpoint: {ckpt_dir}")

    # ── 1. Load model ────────────────────────────────────────────────────
    print("\n=== Loading model ===")
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get a snapshot of base Q/K/V weights BEFORE loading checkpoint
    base_q0 = base.model.layers[0].self_attn.q_proj.weight.clone()

    base = replace_attention_with_fmma(
        base, num_channels=args.channels, bandwidth=args.bandwidth
    )

    # Snapshot FMMA-specific weights BEFORE loading checkpoint (random init)
    fmma_layer0 = base.model.layers[0].self_attn
    init_key_feat_w = fmma_layer0.key_features.projection.weight.clone()
    init_qm_w = fmma_layer0.query_moments.net[0].weight.clone()

    sd = load_file(ckpt_dir / "model.safetensors")
    missing, unexpected = base.load_state_dict(sd, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    for k in missing:
        print(f"  MISSING: {k}")
    for k in unexpected[:10]:
        print(f"  UNEXPECTED: {k}")

    model = base.to(device).eval()

    # ── 2. Check if FMMA weights actually changed from init ──────────────
    print("\n=== FMMA weight training check ===")
    fmma_layer0 = model.model.layers[0].self_attn
    trained_key_feat_w = fmma_layer0.key_features.projection.weight.cpu()
    trained_qm_w = fmma_layer0.query_moments.net[0].weight.cpu()

    kf_diff = (trained_key_feat_w - init_key_feat_w).abs().mean().item()
    qm_diff = (trained_qm_w - init_qm_w).abs().mean().item()
    print(f"  key_features layer0 weight change from init: {kf_diff:.6f}")
    print(f"  query_moments layer0 weight change from init: {qm_diff:.6f}")
    if kf_diff < 1e-6 and qm_diff < 1e-6:
        print("  ⚠️  FMMA weights did NOT change — checkpoint didn't contain trained FMMA params!")
    else:
        print("  ✓ FMMA weights were trained (changed from init)")

    # Also check if Q projection changed from original base
    trained_q0 = model.model.layers[0].self_attn.q_proj.weight.cpu()
    q_diff = (trained_q0 - base_q0).abs().mean().item()
    print(f"  q_proj layer0 weight change from base: {q_diff:.6f}")

    # ── 3. Compare training path vs cache path ───────────────────────────
    print("\n=== Training path vs Cache path comparison ===")
    prompt = "def fibonacci(n):\n    "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    B, T = input_ids.shape
    print(f"  Input: {repr(prompt)} ({T} tokens)")

    # Training path (no cache → _fmma_training_forward)
    with torch.no_grad():
        out_train = model(input_ids=input_ids)
    logits_train = out_train.logits  # [B, T, V]

    # Cache path (with FMMACache → _fmma_forward)
    cache = create_fmma_cache(model)
    with torch.no_grad():
        out_cache = model(input_ids=input_ids, past_key_values=cache)
    logits_cache = out_cache.logits  # [B, T, V]

    # Compare
    diff = (logits_train.float() - logits_cache.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (logits_train.float().abs() + 1e-8)).mean().item()
    print(f"  Max  |logit_train - logit_cache|: {max_diff:.6f}")
    print(f"  Mean |logit_train - logit_cache|: {mean_diff:.6f}")
    print(f"  Mean relative diff: {rel_diff:.6f}")

    if max_diff > 0.1:
        print("  ⚠️  SIGNIFICANT MISMATCH between training and cache paths!")
        # Show per-position comparison
        for t in range(min(T, 8)):
            lt = logits_train[0, t]
            lc = logits_cache[0, t]
            pos_diff = (lt - lc).abs().max().item()
            top_train = lt.topk(3).indices.tolist()
            top_cache = lc.topk(3).indices.tolist()
            print(f"    pos {t}: max_diff={pos_diff:.4f}, "
                  f"top3_train={top_train}, top3_cache={top_cache}")
    else:
        print("  ✓ Training and cache paths agree")

    # ── 4. Check actual prediction quality (training path) ───────────────
    print("\n=== Prediction quality (training path) ===")
    probs_train = F.softmax(logits_train[0, -1].float(), dim=-1)
    top_vals, top_ids = probs_train.topk(10)
    print(f"  Top-10 next-token predictions after '{prompt.strip()}':")
    for i in range(10):
        tok = tokenizer.decode([top_ids[i].item()])
        print(f"    {top_vals[i].item():.4f}  {repr(tok)}")

    # Check entropy of last position
    entropy = -(probs_train * (probs_train + 1e-10).log()).sum().item()
    max_entropy = math.log(probs_train.shape[0])
    print(f"  Entropy: {entropy:.2f} (max: {max_entropy:.2f}, ratio: {entropy/max_entropy:.4f})")

    # ── 5. Quick perplexity on a known string ────────────────────────────
    print("\n=== Quick perplexity check ===")
    test_text = "def hello():\n    print('Hello, world!')\n\nhello()"
    enc = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"])
    shift_logits = out.logits[:, :-1, :].float()
    shift_labels = enc["input_ids"][:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    ).item()
    ppl = math.exp(loss)
    print(f"  PPL on hello-world snippet: {ppl:.2f} (loss: {loss:.4f})")

    # ── 6. Generate with both paths ──────────────────────────────────────
    print("\n=== Generation comparison ===")
    gen_prompt = "# Python function to sort a list\ndef sort_list(lst):\n"
    gen_inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)

    # Generate via training path (no cache — will be slow but uses training forward)
    print("  [Training path generation]")
    with torch.no_grad():
        out_t = model.generate(
            **gen_inputs,
            max_new_tokens=50,
            do_sample=False,  # greedy for determinism
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_text_train = tokenizer.decode(out_t[0], skip_special_tokens=True)
    print(f"  {repr(gen_text_train[len(gen_prompt):])}")

    # Generate via cache path
    print("  [Cache path generation]")
    cache2 = create_fmma_cache(model)
    with torch.no_grad():
        out_c = model.generate(
            **gen_inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            past_key_values=cache2,
        )
    gen_text_cache = tokenizer.decode(out_c[0], skip_special_tokens=True)
    print(f"  {repr(gen_text_cache[len(gen_prompt):])}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if kf_diff < 1e-6 and qm_diff < 1e-6:
        print("🔴 FMMA weights NOT trained — checkpoint doesn't contain FMMA params")
    elif ppl > 50:
        print("🟡 FMMA weights trained but PPL very high — needs more training")
    elif ppl > 15:
        print("🟡 FMMA weights trained, PPL mediocre — partially converged")
    else:
        print("🟢 FMMA weights trained and PPL reasonable")

    if max_diff > 1.0:
        print("🔴 LARGE mismatch between training and cache paths — likely cache bug")
    elif max_diff > 0.1:
        print("🟡 Moderate mismatch between paths — investigate numerical issues")
    else:
        print("🟢 Training and cache paths agree")


if __name__ == "__main__":
    main()
