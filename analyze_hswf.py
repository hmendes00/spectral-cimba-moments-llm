"""
HSWF Compressibility Analysis — Measure per-layer compressibility of Qwen weights.

Produces the empirical data the HSWF paper needs:
  1. Per-layer non-circulant error (how close each weight is to circulant)
  2. Fourier energy concentration (determines optimal S)
  3. HSWF vs pure SVD at matched parameter budgets
  4. Compression/error Pareto front
  5. Blockwise analysis for rectangular matrices

Usage:
    python analyze_hswf.py
    python analyze_hswf.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct
    python analyze_hswf.py --save-plots
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from hswf.core import (
    wrap_diagonal_average,
    circulant_spectrum,
    circulant_from_column,
    bandlimit_spectrum,
    hswf_decompose,
    analyze_compressibility,
    analyze_rectangular_svd,
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

        # Classify the weight
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

        # Extract layer index
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


def analyze_square_weight(W: torch.Tensor, name: str, verbose: bool = True) -> dict:
    """Full HSWF analysis of a square weight matrix."""
    d = W.shape[0]

    # Structural analysis (no (S,r) commitment)
    comp = analyze_compressibility(W.float(), name=name)

    if verbose:
        print(f"  ||W||_F = {comp['W_norm']:.4f}")
        print(f"  Non-circulant error: {comp['non_circ_error']:.4f} "
              f"({comp['non_circ_relative'] * 100:.1f}% of ||W||)")
        print(f"  Fourier energy: 50% in {comp['fourier_s50']} modes, "
              f"90% in {comp['fourier_s90']}, "
              f"95% in {comp['fourier_s95']}, "
              f"99% in {comp['fourier_s99']} (out of {d})")

    # Sweep (S, r) for compression/error tradeoff
    S_values = [d // 16, d // 8, d // 4, d // 2]
    S_values = [max(1, s) for s in S_values]
    r_values = [0, 2, 4, 8, 16, 32]

    sweep = compressibility_sweep(W.float(), S_values=S_values, r_values=r_values)

    # Find best (S, r) for various error budgets
    hswf_only = [r for r in sweep if r["method"] == "HSWF" and r["r"] > 0]
    svd_only = [r for r in sweep if r["method"] == "SVD-only"]

    targets = {"5%": 0.05, "2%": 0.02, "1%": 0.01, "0.5%": 0.005}
    best_at = {}
    for label, target in targets.items():
        candidates = [r for r in hswf_only if r["relative_error"] <= target]
        if candidates:
            best = max(candidates, key=lambda r: r["compression_ratio"])
            best_at[label] = best

            # Find best SVD for same error budget
            svd_candidates = [r for r in svd_only if r["relative_error"] <= target]
            svd_best = max(svd_candidates, key=lambda r: r["compression_ratio"]) if svd_candidates else None

            if verbose:
                msg = (f"  Best HSWF at {label} error: S={best['S']}, r={best['r']} "
                       f"→ {best['compression_ratio']:.1f}× compression")
                if svd_best:
                    msg += f"  (SVD: r={svd_best['r']} → {svd_best['compression_ratio']:.1f}×)"
                print(msg)

    comp["sweep"] = sweep
    comp["best_at"] = best_at
    return comp


def analyze_rectangular_weight(W: torch.Tensor, name: str,
                                verbose: bool = True) -> dict:
    """Analyze a rectangular weight via SVD + optional blockwise HSWF."""
    m, n = W.shape
    result = analyze_rectangular_svd(W.float(), name=name)

    if verbose:
        print(f"  ||W||_F = {result['W_norm']:.4f}")
        print(f"  SVD rank for 50%={result['rank_50']}, 90%={result['rank_90']}, "
              f"95%={result['rank_95']}, 99%={result['rank_99']} "
              f"(out of {min(m, n)})")

    # Blockwise HSWF analysis: split into square sub-blocks
    g = math.gcd(m, n)
    if g >= 16:  # Only analyze if blocks are large enough
        block_size = g
        n_row_blocks = m // block_size
        n_col_blocks = n // block_size

        block_results = []
        total_circ_energy = 0.0
        total_energy = 0.0

        for bi in range(n_row_blocks):
            for bj in range(n_col_blocks):
                block = W[bi * block_size:(bi + 1) * block_size,
                          bj * block_size:(bj + 1) * block_size].float()
                c_star = wrap_diagonal_average(block)
                C_star = circulant_from_column(c_star)

                block_norm_sq = block.square().sum().item()
                circ_norm_sq = C_star.square().sum().item()
                non_circ_sq = (block - C_star).square().sum().item()

                total_energy += block_norm_sq
                total_circ_energy += circ_norm_sq

                block_results.append({
                    "bi": bi, "bj": bj,
                    "block_norm": math.sqrt(block_norm_sq),
                    "circ_fraction": circ_norm_sq / max(block_norm_sq, 1e-10),
                    "non_circ_relative": math.sqrt(non_circ_sq) / max(math.sqrt(block_norm_sq), 1e-10),
                })

        mean_circ_frac = total_circ_energy / max(total_energy, 1e-10)
        mean_non_circ = 1.0 - mean_circ_frac

        result["block_size"] = block_size
        result["n_blocks"] = n_row_blocks * n_col_blocks
        result["mean_circulant_fraction"] = mean_circ_frac
        result["block_results"] = block_results

        if verbose:
            print(f"  Blockwise HSWF ({block_size}×{block_size}, "
                  f"{n_row_blocks}×{n_col_blocks} blocks):")
            print(f"    Mean circulant fraction: {mean_circ_frac * 100:.1f}% "
                  f"(non-circulant: {mean_non_circ * 100:.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="HSWF Compressibility Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save Fourier spectrum and compression plots")
    parser.add_argument("--output-dir", type=str, default="outputs/hswf_analysis",
                        help="Directory for results and plots")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Analyze only first N layers (for speed)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HSWF COMPRESSIBILITY ANALYSIS")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True,
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

    # Analyze each weight
    square_results = []
    rect_results = []
    all_results = []

    total_dense_params = 0
    total_hswf_params_5pct = 0
    total_hswf_params_1pct = 0

    for i, winfo in enumerate(weights):
        name = winfo["name"]
        W = winfo["param"]
        m, n = winfo["shape"]
        total_dense_params += m * n

        print(f"{'─' * 60}")
        print(f"[{i + 1}/{len(weights)}] {name} ({m}×{n}) — {winfo['category']}")

        if winfo["is_square"]:
            result = analyze_square_weight(W, name)
            result["shape"] = (m, n)
            result["category"] = winfo["category"]
            result["layer_idx"] = winfo["layer_idx"]
            square_results.append(result)

            # Accumulate compression stats
            if "5%" in result.get("best_at", {}):
                total_hswf_params_5pct += result["best_at"]["5%"]["hswf_params"]
            else:
                total_hswf_params_5pct += m * n  # Can't compress at 5%

            if "1%" in result.get("best_at", {}):
                total_hswf_params_1pct += result["best_at"]["1%"]["hswf_params"]
            else:
                total_hswf_params_1pct += m * n
        else:
            result = analyze_rectangular_weight(W, name)
            result["category"] = winfo["category"]
            result["layer_idx"] = winfo["layer_idx"]
            rect_results.append(result)

            # Rectangular: count full params (not HSWF-compressed)
            total_hswf_params_5pct += m * n
            total_hswf_params_1pct += m * n

        all_results.append(result)
        print()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal weight matrices analyzed: {len(weights)}")
    print(f"  Square (HSWF applicable): {len(square_results)}")
    print(f"  Rectangular (SVD only):   {len(rect_results)}")

    print(f"\nTotal dense parameters: {total_dense_params:,}")

    if square_results:
        print(f"\n{'─' * 50}")
        print("SQUARE MATRICES (full HSWF)")
        print(f"{'─' * 50}")

        # Aggregate by category
        by_category = defaultdict(list)
        for r in square_results:
            by_category[r["category"]].append(r)

        for cat, results in sorted(by_category.items()):
            mean_non_circ = sum(r["non_circ_relative"] for r in results) / len(results)
            mean_s90 = sum(r["fourier_s90"] for r in results) / len(results)
            d = results[0]["d"]
            print(f"\n  {cat} ({len(results)} layers, {d}×{d}):")
            print(f"    Mean non-circulant: {mean_non_circ * 100:.1f}%")
            print(f"    Mean S for 90% Fourier energy: {mean_s90:.0f}/{d}")

            # Best compression stats
            for err_label in ["5%", "2%", "1%"]:
                ratios = []
                for r in results:
                    if err_label in r.get("best_at", {}):
                        ratios.append(r["best_at"][err_label]["compression_ratio"])
                if ratios:
                    mean_ratio = sum(ratios) / len(ratios)
                    print(f"    Mean compression at ≤{err_label} error: "
                          f"{mean_ratio:.1f}× ({len(ratios)}/{len(results)} layers)")

    if rect_results:
        print(f"\n{'─' * 50}")
        print("RECTANGULAR MATRICES (SVD / blockwise)")
        print(f"{'─' * 50}")

        by_category = defaultdict(list)
        for r in rect_results:
            by_category[r["category"]].append(r)

        for cat, results in sorted(by_category.items()):
            mean_r90 = sum(r["rank_90"] for r in results) / len(results)
            shape = results[0]["shape"]
            print(f"\n  {cat} ({len(results)} layers, {shape[0]}×{shape[1]}):")
            print(f"    Mean SVD rank for 90% energy: {mean_r90:.0f}/{min(shape)}")

            if "block_size" in results[0]:
                mean_circ = sum(r.get("mean_circulant_fraction", 0) for r in results) / len(results)
                block_size = results[0]["block_size"]
                print(f"    Blockwise ({block_size}×{block_size}): "
                      f"mean circulant fraction = {mean_circ * 100:.1f}%")

    # Overall compression potential
    print(f"\n{'─' * 50}")
    print("OVERALL COMPRESSION POTENTIAL")
    print(f"{'─' * 50}")
    print(f"  Dense (baseline):       {total_dense_params:>12,} params")
    print(f"  HSWF (≤5% error):       {total_hswf_params_5pct:>12,} params "
          f"({total_dense_params / max(total_hswf_params_5pct, 1):.2f}×)")
    print(f"  HSWF (≤1% error):       {total_hswf_params_1pct:>12,} params "
          f"({total_dense_params / max(total_hswf_params_1pct, 1):.2f}×)")
    print(f"\n  Note: Rectangular matrices counted at full size (future: BCCB extension)")
    print("=" * 70)

    # ================================================================
    # Save detailed results
    # ================================================================
    # Convert to JSON-serializable format
    json_results = []
    for r in all_results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, torch.Tensor):
                continue  # Skip tensors
            if isinstance(v, list) and v and isinstance(v[0], dict):
                # Nested dicts (sweep, block_results) — keep a summary
                if k == "sweep":
                    jr[k] = v  # Already JSON-safe
                elif k == "block_results":
                    jr[k + "_count"] = len(v)
                continue
            jr[k] = v
        json_results.append(jr)

    results_path = output_dir / "compressibility_results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {results_path}")

    # ================================================================
    # Plots (optional)
    # ================================================================
    if args.save_plots and square_results:
        try:
            import matplotlib.pyplot as plt

            # Plot 1: Fourier energy concentration across layers
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: Fourier energy concentration (S needed for X% energy)
            categories = sorted(set(r["category"] for r in square_results))
            for cat in categories:
                cat_results = [r for r in square_results if r["category"] == cat]
                layers = [r["layer_idx"] for r in cat_results]
                s90 = [r["fourier_s90"] for r in cat_results]
                d = cat_results[0]["d"]
                axes[0].plot(layers, [s / d * 100 for s in s90],
                             'o-', label=f"{cat}", markersize=4)
            axes[0].set_xlabel("Layer index")
            axes[0].set_ylabel("% of modes for 90% Fourier energy")
            axes[0].set_title("Fourier Energy Concentration")
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # Right: Non-circulant error across layers
            for cat in categories:
                cat_results = [r for r in square_results if r["category"] == cat]
                layers = [r["layer_idx"] for r in cat_results]
                non_circ = [r["non_circ_relative"] * 100 for r in cat_results]
                axes[1].plot(layers, non_circ, 'o-', label=f"{cat}", markersize=4)
            axes[1].set_xlabel("Layer index")
            axes[1].set_ylabel("Non-circulant error (% of ||W||)")
            axes[1].set_title("Distance from Circulant Subspace")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / "fourier_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Fourier analysis plot saved to {plot_path}")

            # Plot 2: Compression vs error Pareto (first square layer)
            if square_results and "sweep" in square_results[0]:
                fig, ax = plt.subplots(figsize=(8, 5))
                sweep = square_results[0]["sweep"]
                layer_name = square_results[0]["name"]

                hswf_pts = [r for r in sweep if r["method"] == "HSWF" and r["r"] > 0]
                svd_pts = [r for r in sweep if r["method"] == "SVD-only"]

                if hswf_pts:
                    ax.scatter(
                        [r["compression_ratio"] for r in hswf_pts],
                        [r["relative_error"] * 100 for r in hswf_pts],
                        c="blue", alpha=0.6, label="HSWF (C_S + UV^T)", s=30,
                    )
                if svd_pts:
                    ax.scatter(
                        [r["compression_ratio"] for r in svd_pts],
                        [r["relative_error"] * 100 for r in svd_pts],
                        c="red", alpha=0.6, label="SVD-only (UV^T)", s=30, marker="x",
                    )

                ax.set_xlabel("Compression ratio (dense / compressed)")
                ax.set_ylabel("Relative error (%)")
                ax.set_title(f"HSWF vs SVD Pareto — {layer_name}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")

                plot_path = output_dir / "pareto_front.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Pareto front plot saved to {plot_path}")

        except ImportError:
            print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
