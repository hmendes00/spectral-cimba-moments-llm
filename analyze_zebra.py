"""
ZEBRA Compressibility Analysis — Measure per-layer compressibility of model weights.

Produces the empirical data the ZEBRA paper needs:
  1. Per-layer transform-domain coefficient concentration
  2. Certificate curves (tail energy vs K, residual SVD tail vs r)
  3. ZEBRA vs pure SVD at matched parameter budgets
  4. Compression/error Pareto front
  5. Works for both square and rectangular matrices

Usage:
    python analyze_zebra.py
    python analyze_zebra.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct
    python analyze_zebra.py --save-plots
"""

import argparse
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from zebra.core import (
    build_fast_orthogonal,
    build_orthogonal_matrix,
    top_k_sparsify,
    top_k_tail_energy,
    zebra_decompose,
    certificate_curve_sparse,
    certificate_curve_residual,
    select_transforms,
    analyze_compressibility,
    compressibility_sweep,
)


def get_weight_matrices(model) -> list:
    """Extract all weight matrices from the model with metadata."""
    weights = []
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        if param.dim() != 2:
            continue
        # Skip embeddings (very large, not attention/MLP)
        if "embed" in name or "lm_head" in name:
            continue

        m, n = param.shape
        category = "unknown"
        if "q_proj" in name:
            category = "Q projection"
        elif "k_proj" in name:
            category = "K projection"
        elif "v_proj" in name:
            category = "V projection"
        elif "o_proj" in name:
            category = "O projection"
        elif "gate_proj" in name:
            category = "MLP gate"
        elif "up_proj" in name:
            category = "MLP up"
        elif "down_proj" in name:
            category = "MLP down"

        layer_idx = None
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break

        weights.append({
            "name": name,
            "shape": (m, n),
            "is_square": m == n,
            "category": category,
            "layer_idx": layer_idx,
            "param": param.data,
        })

    return weights


def analyze_weight(W: torch.Tensor, name: str, verbose: bool = True,
                   n_candidates: int = 4) -> dict:
    """Full ZEBRA analysis of a weight matrix (square or rectangular)."""
    m, n = W.shape
    mn = m * n

    # Structural analysis (no (K,r) commitment)
    comp = analyze_compressibility(W.float(), name=name, n_candidates=n_candidates)

    if verbose:
        print(f"  ||W||_F = {comp['W_norm']:.4f}")
        print(f"  Transform coefficients: 50% energy in {comp['coeff_k50']} "
              f"({comp['coeff_k50_frac']*100:.1f}%), "
              f"90% in {comp['coeff_k90']} "
              f"({comp['coeff_k90_frac']*100:.1f}%), "
              f"95% in {comp['coeff_k95']} "
              f"({comp['coeff_k95_frac']*100:.1f}%) "
              f"(out of {mn})")

    # Sweep (K, r) for compression/error tradeoff
    # 3.2: Log-spaced K for better coverage at small K (where ZEBRA should shine)
    K_values = sorted(set(
        [max(1, int(mn * 2**(-j))) for j in range(1, 11)]  # log-spaced: mn/2..mn/1024
        + [mn // 6, mn // 3, mn // 2, 2*mn // 3]           # a few linear midpoints
    ))
    K_values = [k for k in K_values if 1 <= k <= mn]
    min_dim = min(m, n)
    # 2.1: Dense r-grid so optimizer doesn't jump straight to max r
    r_values = sorted(set(
        list(range(0, min(65, min_dim + 1), 8))         # 0,8,16,...,64
        + list(range(64, min(257, min_dim + 1), 32))    # 64,96,128,...,256
        + list(range(256, min(513, min_dim + 1), 64))   # 256,320,...,512
        + list(range(512, min_dim + 1, 128))             # 512,640,...
    ))
    r_values = [r for r in r_values if 0 <= r <= min_dim]

    sweep = compressibility_sweep(
        W.float(),
        K_values=K_values,
        r_values=r_values,
        seed_A=comp["best_seed_A"],
        seed_B=comp["best_seed_B"],
    )

    # Find best (K, r) for various error budgets
    # Include r=0 (pure transform sparsity) — often the only compressing regime
    zebra_only = [r for r in sweep if r["method"] == "ZEBRA"]
    svd_only = [r for r in sweep if r["method"] == "SVD-only"]

    targets = {"10%": 0.10, "5%": 0.05, "2%": 0.02, "1%": 0.01}
    best_at = {}
    for label, target in targets.items():
        # 2.2: Explicitly minimize params (then error) among feasible configs
        candidates = [r for r in zebra_only if r["relative_error"] <= target]
        if candidates:
            best = min(candidates, key=lambda r: (r["zebra_params"], r["relative_error"]))
            best_at[label] = best

            svd_candidates = [r for r in svd_only if r["relative_error"] <= target]
            svd_best = min(svd_candidates, key=lambda s: (s["zebra_params"], s["relative_error"])) if svd_candidates else None

            if verbose:
                params_ratio = best['zebra_params'] / max(best['dense_params'], 1)
                ratio_str = (f"{1/params_ratio:.1f}× smaller" if params_ratio < 1.0
                             else f"{params_ratio:.2f}× of dense (expansion)")
                msg = (f"  Best ZEBRA at ≤{label} error: K={best['K']}, r={best['r']} "
                       f"→ {best['zebra_params']:,} params ({ratio_str}, "
                       f"err: {best['relative_error']*100:.1f}%)")
                if svd_best:
                    svd_pr = svd_best['zebra_params'] / max(svd_best['dense_params'], 1)
                    svd_str = (f"{1/svd_pr:.1f}×" if svd_pr < 1.0
                               else f"{svd_pr:.2f}× of dense")
                    msg += f" | SVD: {svd_str}"
                print(msg)

    # Always report best achievable numbers
    if verbose:
        if zebra_only:
            best_err = min(zebra_only, key=lambda r: r["relative_error"])
            best_comp = min(zebra_only, key=lambda r: r["zebra_params"])
            pr_err = best_err['zebra_params'] / max(best_err['dense_params'], 1)
            pr_comp = best_comp['zebra_params'] / max(best_comp['dense_params'], 1)
            print(f"  Min achievable error: {best_err['relative_error']*100:.1f}% "
                  f"(K={best_err['K']}, r={best_err['r']}, "
                  f"{pr_err:.2f}× of dense)")
            print(f"  Min params: {best_comp['zebra_params']:,} "
                  f"({pr_comp:.2f}× of dense, "
                  f"err: {best_comp['relative_error']*100:.1f}%)")

        # Item 4: min error subject to params ≤ dense (compression view)
        if zebra_only:
            compressing = [r for r in zebra_only if r["zebra_params"] <= r["dense_params"]]
            if compressing:
                best_under_budget = min(compressing, key=lambda r: r["relative_error"])
                bub_pr = best_under_budget['zebra_params'] / max(best_under_budget['dense_params'], 1)
                print(f"  Best error with actual compression: "
                      f"{best_under_budget['relative_error']*100:.1f}% "
                      f"(K={best_under_budget['K']}, r={best_under_budget['r']}, "
                      f"{bub_pr:.2f}× of dense, "
                      f"values_only={best_under_budget.get('zebra_params_values_only', '?'):,})")
            else:
                print(f"  ⚠ No ZEBRA config achieves actual compression (all expand)")

        if svd_only:
            best_svd = min(svd_only, key=lambda r: r["relative_error"])
            svd_pr = best_svd['zebra_params'] / max(best_svd['dense_params'], 1)
            print(f"  SVD-only min error: {best_svd['relative_error']*100:.1f}% "
                  f"(r={best_svd['r']}, {svd_pr:.2f}× of dense)")

    comp["sweep"] = sweep
    comp["best_at"] = best_at
    return comp


def main():
    parser = argparse.ArgumentParser(description="ZEBRA Compressibility Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save coefficient distribution and compression plots")
    parser.add_argument("--output-dir", type=str, default="outputs/zebra_analysis",
                        help="Directory for results and plots")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Analyze only first N layers (for speed)")
    parser.add_argument("--n-candidates", type=int, default=4,
                        help="Number of transform candidates to try per weight")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ZEBRA COMPRESSIBILITY ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Transform candidates: {args.n_candidates}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=True,
    ).to(args.device)
    config = model.config
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  hidden_size={config.hidden_size}, "
          f"num_heads={config.num_attention_heads}, "
          f"num_kv_heads={getattr(config, 'num_key_value_heads', config.num_attention_heads)}, "
          f"intermediate_size={config.intermediate_size}")
    print()

    # Extract weights
    weights = get_weight_matrices(model)
    if args.max_layers is not None:
        weights = [w for w in weights
                   if w["layer_idx"] is None or w["layer_idx"] < args.max_layers]

    print(f"Analyzing {len(weights)} weight matrices...")
    print()

    all_results = []
    total_dense_params = 0
    total_zebra_params_10pct = 0
    total_zebra_params_5pct = 0
    total_zebra_params_1pct = 0

    for i, winfo in enumerate(weights):
        name = winfo["name"]
        W = winfo["param"]
        m, n = winfo["shape"]
        total_dense_params += m * n

        print(f"{'─' * 60}")
        print(f"[{i + 1}/{len(weights)}] {name} ({m}×{n}) — {winfo['category']}")

        result = analyze_weight(W, name, n_candidates=args.n_candidates)
        result["shape"] = (m, n)
        result["category"] = winfo["category"]
        result["layer_idx"] = winfo["layer_idx"]
        all_results.append(result)

        # Accumulate compression stats
        if "5%" in result.get("best_at", {}):
            total_zebra_params_5pct += result["best_at"]["5%"]["zebra_params"]
        else:
            total_zebra_params_5pct += m * n  # no compression possible

        if "1%" in result.get("best_at", {}):
            total_zebra_params_1pct += result["best_at"]["1%"]["zebra_params"]
        else:
            total_zebra_params_1pct += m * n

        if "10%" in result.get("best_at", {}):
            total_zebra_params_10pct += result["best_at"]["10%"]["zebra_params"]
        else:
            total_zebra_params_10pct += m * n

        print()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal weight matrices analyzed: {len(weights)}")
    print(f"Total dense parameters: {total_dense_params:,}")

    # Aggregate by category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r["category"]].append(r)

    print(f"\n{'─' * 50}")
    print("PER-CATEGORY ANALYSIS")
    print(f"{'─' * 50}")

    for cat, results in sorted(by_category.items()):
        mean_k90_frac = sum(r["coeff_k90_frac"] for r in results) / len(results)
        shapes = set(r["shape"] for r in results)
        shape_str = ", ".join(f"{s[0]}×{s[1]}" for s in shapes)
        print(f"\n  {cat} ({len(results)} layers, {shape_str}):")
        print(f"    Mean coeffs for 90% energy: {mean_k90_frac * 100:.1f}% of total")

        for err_label in ["10%", "5%", "2%", "1%"]:
            ratios = []
            for r in results:
                if err_label in r.get("best_at", {}):
                    b = r["best_at"][err_label]
                    ratios.append(b["zebra_params"] / max(b["dense_params"], 1))
            if ratios:
                mean_pr = sum(ratios) / len(ratios)
                label = (f"{1/mean_pr:.1f}× smaller" if mean_pr < 1.0
                         else f"{mean_pr:.2f}× of dense")
                print(f"    ZEBRA at ≤{err_label} error: {label} "
                      f"({len(ratios)}/{len(results)} layers)")

    # Overall compression potential
    print(f"\n{'─' * 50}")
    print("OVERALL COMPRESSION POTENTIAL")
    print(f"{'─' * 50}")
    print(f"  Dense (baseline):       {total_dense_params:>12,} params")
    for label, total_z in [("10%", total_zebra_params_10pct),
                            ("5%", total_zebra_params_5pct),
                            ("1%", total_zebra_params_1pct)]:
        pr = total_z / max(total_dense_params, 1)
        if pr <= 1.0:
            ratio_str = f"{1/pr:.2f}× smaller"
        else:
            ratio_str = f"{pr:.2f}× of dense (EXPANSION)"
        print(f"  ZEBRA (≤{label} error):  {total_z:>12,} params ({ratio_str})")

    # SVD-only comparison
    total_svd = {"10%": 0, "5%": 0, "1%": 0}
    for r in all_results:
        m, n = r["shape"]
        mn = m * n
        svd_items = [s for s in r.get("sweep", []) if s["method"] == "SVD-only"]
        for label in total_svd:
            target = float(label.rstrip("%")) / 100
            candidates = [s for s in svd_items if s["relative_error"] <= target]
            if candidates:
                best = min(candidates, key=lambda s: (s["zebra_params"], s["relative_error"]))
                total_svd[label] += best["zebra_params"]
            else:
                total_svd[label] += mn

    for label in ["10%", "5%", "1%"]:
        svd_total = total_svd[label]
        svd_pr = svd_total / max(total_dense_params, 1)
        if svd_pr <= 1.0:
            svd_str = f"{1/svd_pr:.2f}× smaller"
        else:
            svd_str = f"{svd_pr:.2f}× of dense (EXPANSION)"
        print(f"  SVD-only (≤{label} error):  {svd_total:>12,} params ({svd_str})")
    print("=" * 70)

    # ================================================================
    # Save detailed results
    # ================================================================
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, torch.Tensor):
                jr[k] = v.tolist()
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                jr[k] = [{kk: (vv.tolist() if isinstance(vv, torch.Tensor) else vv)
                           for kk, vv in item.items()} for item in v]
            else:
                jr[k] = v
        json_results.append(jr)

    results_path = output_dir / "compressibility_results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {results_path}")

    # ================================================================
    # Plots (optional)
    # ================================================================
    if args.save_plots and all_results:
        try:
            import matplotlib.pyplot as plt

            # Plot 1: Coefficient energy concentration across layers
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            categories = sorted(set(r["category"] for r in all_results))
            for cat in categories:
                cat_results = [r for r in all_results if r["category"] == cat]
                layers = [r["layer_idx"] for r in cat_results if r["layer_idx"] is not None]
                k90_fracs = [r["coeff_k90_frac"] * 100 for r in cat_results
                             if r["layer_idx"] is not None]
                if layers:
                    axes[0].plot(layers, k90_fracs, "o-", label=cat, markersize=3)
            axes[0].set_xlabel("Layer index")
            axes[0].set_ylabel("% coefficients for 90% energy")
            axes[0].set_title("Transform-Domain Energy Concentration")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # Right: SVD singular value decay (first weight of each category)
            for cat in categories:
                cat_results = [r for r in all_results if r["category"] == cat]
                if cat_results and "singular_values_W" in cat_results[0]:
                    sv = cat_results[0]["singular_values_W"]
                    if isinstance(sv, torch.Tensor):
                        sv = sv.numpy()
                    axes[1].semilogy(sv[:min(64, len(sv))], label=cat)
            axes[1].set_xlabel("Singular value index")
            axes[1].set_ylabel("σ_j")
            axes[1].set_title("Singular Value Decay (first layer)")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / "zebra_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Analysis plot saved to {plot_path}")

        except ImportError:
            print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
