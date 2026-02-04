"""
Golden Layer CIMBA Validation Test

This script validates whether CIMBA moment sketching can faithfully reconstruct
Qwen attention across diverse coding prompts. It tests the hypothesis that
learned Fourier coefficients can capture the "Motif Algebra" of each attention head.

Metrics:
- Relative MSE: Overall fidelity
- Cosine Similarity: Direction preservation (which tokens are attended to)
- Max Absolute Error: Worst-case token loss detection
- Peak Preservation: Rank correlation of top-k attention weights

Author: Hugo Leonardo Simas Mendes
Date: February 2026
"""

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from prompts import PROMPTS, PROMPT_NAMES
from cimba import CIMBASketch, CIMBARecovery


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the Golden Layer test."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    layer_idx: int = 15  # Target layer for extraction
    S_values: List[int] = None  # Moment counts to test
    learning_rate: float = 1e-3
    num_steps: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"
    seed: int = 42
    
    def __post_init__(self):
        if self.S_values is None:
            self.S_values = [8, 16, 32, 64, 128, 256]


# ============================================================================
# Model Loading and Attention Extraction
# ============================================================================

def load_model_and_tokenizer(config: ExperimentConfig):
    """Load the Qwen model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # Use float32 for precision in comparisons
        device_map=config.device,
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer


def get_attention_layer(model, layer_idx: int):
    """Extract the attention module from a specific layer."""
    # Qwen architecture: model.model.layers[i].self_attn
    try:
        return model.model.layers[layer_idx].self_attn
    except (AttributeError, IndexError) as e:
        # Fallback: try to find layers automatically
        if hasattr(model, 'transformer'):
            return model.transformer.h[layer_idx].attn
        raise ValueError(f"Could not find attention layer {layer_idx}: {e}")


def compute_ground_truth_attention(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ground-truth Q, K, V, attention matrix, and output for a prompt.
    
    Returns:
        Q: [batch, heads, seq_len, head_dim]
        K: [batch, heads, seq_len, head_dim]
        V: [batch, heads, seq_len, head_dim]
        A_true: [batch, heads, seq_len, seq_len] attention matrix
        Y_true: [batch, heads, seq_len, head_dim] attention output
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    # We need to capture intermediate activations
    # Use hooks to grab Q, K, V from the target layer
    captured = {}
    
    def capture_qkv_hook(module, args, kwargs, output):
        """Hook to capture the hidden states before attention."""
        # The input to self_attn is the hidden_states
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        captured["hidden_states"] = hidden_states.detach()
        return output
    
    attn_layer = get_attention_layer(model, layer_idx)
    
    # Register hook
    handle = attn_layer.register_forward_hook(capture_qkv_hook, with_kwargs=True)
    
    try:
        with torch.no_grad():
            # Forward pass to capture hidden states
            _ = model(input_ids, output_attentions=True)
    finally:
        handle.remove()
    
    hidden_states = captured["hidden_states"]
    
    # Now manually compute Q, K, V using the layer's projections
    # Qwen uses separate q_proj, k_proj, v_proj
    with torch.no_grad():
        Q = attn_layer.q_proj(hidden_states)
        K = attn_layer.k_proj(hidden_states)
        V = attn_layer.v_proj(hidden_states)
    
    # Reshape to [batch, heads, seq_len, head_dim]
    batch = Q.shape[0]
    
    # Get head configuration from model config
    num_heads = attn_layer.config.num_attention_heads
    head_dim = attn_layer.config.hidden_size // num_heads
    
    # Handle GQA (Grouped Query Attention) if present
    num_kv_heads = getattr(attn_layer.config, 'num_key_value_heads', num_heads)
    
    Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # If GQA, expand K and V to match Q's head count
    if num_kv_heads != num_heads:
        n_rep = num_heads // num_kv_heads
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)
    
    # Compute attention: A = softmax(QK^T / sqrt(d))
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply causal mask (lower triangular)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1
    )
    attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    
    A_true = F.softmax(attn_weights, dim=-1)
    
    # Compute output: Y = A @ V
    Y_true = torch.matmul(A_true, V)
    
    return Q, K, V, A_true, Y_true


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class MetricsResult:
    """Container for all computed metrics."""
    relative_mse: float
    cosine_similarity: float
    max_absolute_error_output: float
    max_absolute_error_attention: float
    peak_preservation_score: float
    per_head_mse: List[float]
    per_prompt_mae: Dict[str, float]


def compute_metrics(
    Y_true: torch.Tensor,
    Y_cimba: torch.Tensor,
    A_true: torch.Tensor,
    A_cimba: Optional[torch.Tensor],
    prompt_name: str,
) -> dict:
    """Compute all metrics for a single prompt."""
    
    # Relative MSE
    mse = ((Y_true - Y_cimba) ** 2).mean()
    rel_mse = mse / (Y_true ** 2).mean()
    
    # Cosine similarity (per-token, then averaged)
    # Flatten heads and head_dim for cosine computation
    Y_true_flat = Y_true.reshape(-1, Y_true.shape[-1])  # [batch*heads*seq, head_dim]
    Y_cimba_flat = Y_cimba.reshape(-1, Y_cimba.shape[-1])
    
    cos_sim = F.cosine_similarity(Y_true_flat, Y_cimba_flat, dim=-1).mean()
    
    # Max Absolute Error (output)
    mae_output = (Y_true - Y_cimba).abs().max()
    
    # Max Absolute Error (attention) and Peak Preservation
    mae_attention = torch.tensor(0.0)
    peak_score = torch.tensor(0.0)
    
    if A_cimba is not None:
        mae_attention = (A_true - A_cimba).abs().max()
        
        # Peak preservation: for each query, check if top-k tokens are preserved
        # Use rank correlation (Spearman) for top-10 attention weights
        peak_score = compute_peak_preservation(A_true, A_cimba, top_k=10)
    
    # Per-head MSE
    per_head_mse = ((Y_true - Y_cimba) ** 2).mean(dim=(0, 2, 3))  # [heads]
    
    return {
        "relative_mse": rel_mse.item(),
        "cosine_similarity": cos_sim.item(),
        "max_absolute_error_output": mae_output.item(),
        "max_absolute_error_attention": mae_attention.item(),
        "peak_preservation_score": peak_score.item() if torch.is_tensor(peak_score) else peak_score,
        "per_head_mse": per_head_mse.tolist(),
        "prompt_name": prompt_name,
    }


def compute_peak_preservation(
    A_true: torch.Tensor,
    A_cimba: torch.Tensor,
    top_k: int = 10,
) -> float:
    """
    Compute peak preservation score via rank correlation.
    
    For each query position, compare the ranking of top-k attended tokens
    in ground truth vs CIMBA reconstruction.
    """
    batch, heads, seq_len, _ = A_true.shape
    
    correlations = []
    
    for b in range(batch):
        for h in range(heads):
            for t in range(seq_len):
                # Get attention row for this query
                a_true_row = A_true[b, h, t, :t+1]  # Only attend to past (causal)
                a_cimba_row = A_cimba[b, h, t, :t+1]
                
                if len(a_true_row) < 2:
                    continue
                
                # Get top-k indices from ground truth
                k = min(top_k, len(a_true_row))
                top_k_indices = a_true_row.topk(k).indices
                
                # Get ranks in both
                true_ranks = a_true_row[top_k_indices].argsort(descending=True).float()
                cimba_values = a_cimba_row[top_k_indices]
                cimba_ranks = cimba_values.argsort(descending=True).float()
                
                # Spearman correlation (on ranks)
                if k > 1:
                    true_ranks = true_ranks - true_ranks.mean()
                    cimba_ranks = cimba_ranks - cimba_ranks.mean()
                    
                    numerator = (true_ranks * cimba_ranks).sum()
                    denominator = (true_ranks.norm() * cimba_ranks.norm()) + 1e-8
                    corr = numerator / denominator
                    correlations.append(corr.item())
    
    return np.mean(correlations) if correlations else 0.0


# ============================================================================
# Training Loop
# ============================================================================

def train_cimba_coefficients(
    sketch: CIMBASketch,
    recovery: CIMBARecovery,
    prompt_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    config: ExperimentConfig,
) -> List[float]:
    """
    Train the CIMBA recovery coefficients to minimize reconstruction error.
    
    Args:
        sketch: CIMBASketch module
        recovery: CIMBARecovery module with learnable coefficients
        prompt_data: List of (V, Y_true, A_true) tuples for each prompt
        config: Experiment configuration
        
    Returns:
        loss_history: Training loss at each step
    """
    optimizer = AdamW(recovery.parameters(), lr=config.learning_rate)
    loss_history = []
    
    pbar = tqdm(range(config.num_steps), desc="Training coefficients")
    
    for step in pbar:
        optimizer.zero_grad()
        total_loss = 0.0
        
        for V, Y_true, A_true in prompt_data:
            # Sketch the values
            moments, phases = sketch(V)
            
            # Recover with learned coefficients
            Y_cimba, _ = recovery(moments, phases, return_attention_matrix=False)
            
            # MSE loss
            loss = F.mse_loss(Y_cimba, Y_true)
            total_loss += loss
        
        # Average across prompts
        total_loss = total_loss / len(prompt_data)
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{total_loss.item():.6f}"})
    
    return loss_history


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(
    results: Dict[int, Dict],
    worst_prompt_data: Tuple[str, torch.Tensor, torch.Tensor],
    config: ExperimentConfig,
):
    """Generate all output plots."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. MSE vs S plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    S_vals = sorted(results.keys())
    
    # Aggregate metrics across prompts for each S
    rel_mse = [np.mean([r["relative_mse"] for r in results[s]["per_prompt"]]) for s in S_vals]
    cos_sim = [np.mean([r["cosine_similarity"] for r in results[s]["per_prompt"]]) for s in S_vals]
    mae_output = [np.max([r["max_absolute_error_output"] for r in results[s]["per_prompt"]]) for s in S_vals]
    
    # Plot 1: Relative MSE
    axes[0].semilogy(S_vals, rel_mse, 'b-o', linewidth=2, markersize=8)
    axes[0].axhline(y=0.01, color='g', linestyle='--', label='1% threshold')
    axes[0].set_xlabel('S (number of moments)', fontsize=12)
    axes[0].set_ylabel('Relative MSE', fontsize=12)
    axes[0].set_title('Reconstruction Error vs Moment Count', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cosine Similarity
    axes[1].plot(S_vals, cos_sim, 'g-o', linewidth=2, markersize=8)
    axes[1].axhline(y=0.98, color='r', linestyle='--', label='0.98 threshold')
    axes[1].set_xlabel('S (number of moments)', fontsize=12)
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].set_title('Direction Preservation vs Moment Count', fontsize=14)
    axes[1].set_ylim([0.9, 1.01])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Max Absolute Error
    axes[2].semilogy(S_vals, mae_output, 'r-o', linewidth=2, markersize=8)
    axes[2].set_xlabel('S (number of moments)', fontsize=12)
    axes[2].set_ylabel('Max Absolute Error', fontsize=12)
    axes[2].set_title('Worst-Case Token Error vs Moment Count', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "mse_vs_S.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Attention heatmap comparison (if we have attention data)
    prompt_name, A_true, A_cimba = worst_prompt_data
    
    if A_true is not None and A_cimba is not None:
        # Take first batch, first head, limited sequence for visibility
        max_tokens = min(50, A_true.shape[2])
        A_true_plot = A_true[0, 0, :max_tokens, :max_tokens].cpu().numpy()
        A_cimba_plot = A_cimba[0, 0, :max_tokens, :max_tokens].cpu().numpy()
        A_diff = np.abs(A_true_plot - A_cimba_plot)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im0 = axes[0].imshow(A_true_plot, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Ground Truth Attention\n({prompt_name})', fontsize=12)
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(A_cimba_plot, cmap='viridis', aspect='auto')
        axes[1].set_title('CIMBA Reconstructed Attention', fontsize=12)
        axes[1].set_xlabel('Key Position')
        axes[1].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(A_diff, cmap='hot', aspect='auto')
        axes[2].set_title('Absolute Difference |A_true - A_cimba|', fontsize=12)
        axes[2].set_xlabel('Key Position')
        axes[2].set_ylabel('Query Position')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_dir / "attention_heatmap_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Error heatmap only
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(A_diff, cmap='hot', aspect='auto')
        ax.set_title(f'Attention Reconstruction Error\n({prompt_name})', fontsize=14)
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        plt.colorbar(im, ax=ax, label='|A_true - A_cimba|')
        plt.savefig(output_dir / "attention_error_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def create_peak_scatter(
    A_true: torch.Tensor,
    A_cimba: torch.Tensor,
    output_dir: Path,
    top_k: int = 50,
):
    """Create scatter plot of top attention weights (true vs reconstructed)."""
    # Flatten and get top-k from ground truth
    A_true_flat = A_true.flatten()
    A_cimba_flat = A_cimba.flatten()
    
    # Get indices of top-k attention weights
    top_k_indices = A_true_flat.topk(top_k).indices
    
    true_vals = A_true_flat[top_k_indices].cpu().numpy()
    cimba_vals = A_cimba_flat[top_k_indices].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(true_vals, cimba_vals, alpha=0.6, s=50)
    
    # Perfect reconstruction line
    max_val = max(true_vals.max(), cimba_vals.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect reconstruction')
    
    ax.set_xlabel('Ground Truth Attention Weight', fontsize=12)
    ax.set_ylabel('CIMBA Reconstructed Weight', fontsize=12)
    ax.set_title(f'Peak Preservation: Top-{top_k} Attention Weights', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "peak_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(config: ExperimentConfig):
    """Run the full Golden Layer validation experiment."""
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Adjust layer index if model is smaller
    num_layers = len(model.model.layers)
    if config.layer_idx >= num_layers:
        config.layer_idx = num_layers // 2
        print(f"Adjusted layer_idx to {config.layer_idx} (model has {num_layers} layers)")
    
    # Get attention layer info
    attn_layer = get_attention_layer(model, config.layer_idx)
    # Qwen2 uses config attributes, not layer attributes
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    print(f"\nLayer {config.layer_idx} attention config:")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    
    # Compute ground truth for all prompts
    print("\nComputing ground truth attention for all prompts...")
    prompt_data = []  # List of (V, Y_true, A_true, prompt_name)
    
    for prompt, name in zip(PROMPTS, PROMPT_NAMES):
        print(f"  Processing: {name}")
        Q, K, V, A_true, Y_true = compute_ground_truth_attention(
            model, tokenizer, prompt, config.layer_idx, config.device
        )
        prompt_data.append({
            "name": name,
            "V": V,
            "Y_true": Y_true,
            "A_true": A_true,
            "Q": Q,
            "K": K,
        })
        print(f"    Sequence length: {V.shape[2]}")
    
    # Run experiment for each S value
    all_results = {}
    best_S = None
    best_cos_sim = -1
    worst_prompt_for_best_S = None
    
    for S in config.S_values:
        print(f"\n{'='*60}")
        print(f"Testing S = {S} ({2*S+1} moments)")
        print(f"{'='*60}")
        
        # Initialize sketch and recovery modules
        sketch = CIMBASketch(S=S, device=config.device)
        recovery = CIMBARecovery(
            S=S,
            num_heads=num_heads,
            head_dim=head_dim,
            device=config.device,
        )
        
        # Prepare training data
        train_data = [(d["V"], d["Y_true"], d["A_true"]) for d in prompt_data]
        
        # Train coefficients
        loss_history = train_cimba_coefficients(sketch, recovery, train_data, config)
        
        # Evaluate on all prompts
        prompt_metrics = []
        worst_mae = -1
        worst_prompt_name = None
        worst_A_true = None
        worst_A_cimba = None
        
        with torch.no_grad():
            for data in prompt_data:
                V = data["V"]
                Y_true = data["Y_true"]
                A_true = data["A_true"]
                name = data["name"]
                
                # Compute CIMBA reconstruction
                moments, phases = sketch(V)
                Y_cimba, A_cimba = recovery(moments, phases, return_attention_matrix=True)
                
                # Compute metrics
                metrics = compute_metrics(Y_true, Y_cimba, A_true, A_cimba, name)
                prompt_metrics.append(metrics)
                
                # Track worst prompt
                if metrics["max_absolute_error_attention"] > worst_mae:
                    worst_mae = metrics["max_absolute_error_attention"]
                    worst_prompt_name = name
                    worst_A_true = A_true.clone()
                    worst_A_cimba = A_cimba.clone()
                
                print(f"\n  {name}:")
                print(f"    Relative MSE: {metrics['relative_mse']:.6f}")
                print(f"    Cosine Sim:   {metrics['cosine_similarity']:.4f}")
                print(f"    MAE (output): {metrics['max_absolute_error_output']:.6f}")
                print(f"    MAE (attn):   {metrics['max_absolute_error_attention']:.6f}")
                print(f"    Peak Score:   {metrics['peak_preservation_score']:.4f}")
        
        # Aggregate metrics
        avg_cos_sim = np.mean([m["cosine_similarity"] for m in prompt_metrics])
        avg_rel_mse = np.mean([m["relative_mse"] for m in prompt_metrics])
        
        all_results[S] = {
            "per_prompt": prompt_metrics,
            "avg_cosine_similarity": avg_cos_sim,
            "avg_relative_mse": avg_rel_mse,
            "loss_history": loss_history,
            "coefficient_stats": recovery.get_coefficient_stats(),
        }
        
        # Memory comparison
        sample_seq_len = prompt_data[0]["V"].shape[2]
        mem_comparison = CIMBASketch.compare_to_full_kv(
            sample_seq_len, 1, num_heads, head_dim, S
        )
        print(f"\n  Memory comparison (seq_len={sample_seq_len}):")
        print(f"    Full KV: {mem_comparison['full_kv_bytes'] / 1024:.1f} KB")
        print(f"    CIMBA:   {mem_comparison['cimba_bytes'] / 1024:.1f} KB")
        print(f"    Compression: {mem_comparison['compression_ratio']:.1f}x")
        
        # Track best S
        if avg_cos_sim > best_cos_sim:
            best_cos_sim = avg_cos_sim
            best_S = S
            worst_prompt_for_best_S = (worst_prompt_name, worst_A_true, worst_A_cimba)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(all_results, worst_prompt_for_best_S, config)
    
    if worst_prompt_for_best_S[1] is not None:
        create_peak_scatter(
            worst_prompt_for_best_S[1],
            worst_prompt_for_best_S[2],
            output_dir,
        )
    
    # Save metrics report
    report = {
        "config": {
            "model_name": config.model_name,
            "layer_idx": config.layer_idx,
            "S_values": config.S_values,
            "num_steps": config.num_steps,
            "learning_rate": config.learning_rate,
        },
        "results": {
            str(s): {
                "avg_relative_mse": all_results[s]["avg_relative_mse"],
                "avg_cosine_similarity": all_results[s]["avg_cosine_similarity"],
                "per_prompt": all_results[s]["per_prompt"],
                "coefficient_stats": all_results[s]["coefficient_stats"],
            }
            for s in config.S_values
        },
        "summary": {
            "best_S": best_S,
            "best_cosine_similarity": best_cos_sim,
        },
    }
    
    with open(output_dir / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/metrics_report.json")
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print("\nResults by S:")
    print(f"{'S':>6} | {'Rel MSE':>12} | {'Cos Sim':>10} | {'Status':>10}")
    print("-" * 50)
    
    for S in config.S_values:
        rel_mse = all_results[S]["avg_relative_mse"]
        cos_sim = all_results[S]["avg_cosine_similarity"]
        
        # Check success criteria
        mse_ok = rel_mse < 0.01
        cos_ok = cos_sim > 0.98
        status = "✓ PASS" if (mse_ok and cos_ok) else "✗ FAIL"
        
        print(f"{S:>6} | {rel_mse:>12.6f} | {cos_sim:>10.4f} | {status:>10}")
    
    print("\n" + "="*60)
    
    return all_results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    config = ExperimentConfig()
    
    # Allow command-line override of model
    if len(sys.argv) > 1:
        config.model_name = sys.argv[1]
    
    print("Golden Layer CIMBA Validation Test")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Layer: {config.layer_idx}")
    print(f"S values: {config.S_values}")
    print(f"Device: {config.device}")
    print("="*60)
    
    results = run_experiment(config)
