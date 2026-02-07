"""
Needle-in-Haystack Benchmark for CIMBA Spectral Attention

Tests the model's ability to retrieve a specific "needle" (fact) hidden at
various positions within a long "haystack" (filler text).

Hypothesis: CIMBA Spectral Attention should perform AT LEAST as well as standard
attention because:
1. It preserves Q·K content-based similarity (what to attend to)
2. Spectral filtering only smooths the attention pattern
3. The needle should still have highest attention weight in its region

Usage:
    python benchmark_needle_haystack.py --max-context 8192 --depths 5
"""

import argparse
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cimba_spectral_attention import replace_attention_with_spectral
from fmma_attention import replace_attention_with_fmma, FMMACache, create_fmma_cache
from gc import collect


# Filler text for the haystack (technical documentation style)
HAYSTACK_FILLER = """
The system architecture employs a distributed computing framework with redundant
failover mechanisms. Load balancing is achieved through consistent hashing algorithms
that minimize cache invalidation during node failures. The message queue implements
at-least-once delivery semantics with idempotent consumers handling duplicate messages.
Database sharding follows a geographic distribution pattern with cross-region replication
for disaster recovery scenarios. The authentication layer uses JWT tokens with RSA-256
signatures and implements token refresh mechanisms with sliding expiration windows.
API rate limiting enforces per-user quotas tracked via Redis with sliding window counters.
The caching strategy combines CDN edge caching for static assets with application-level
caching using a write-through pattern for frequently accessed data. Monitoring infrastructure
collects metrics via Prometheus exporters aggregated through Grafana dashboards with
alerting rules configured for SLA thresholds. Deployment pipelines implement blue-green
strategies with automated canary analysis for detecting regression before full rollout.
Configuration management uses immutable infrastructure principles with all changes
tracked through version-controlled infrastructure-as-code repositories.
"""

# Different needles (facts to hide)
NEEDLES = [
    {
        "needle": "The secret password for the vault is 'BlueHorse42'.",
        "question": "What is the secret password for the vault?",
        "answer": "BlueHorse42"
    },
    {
        "needle": "The meeting is scheduled for March 15th at 3:00 PM in Room 407.",
        "question": "When is the meeting scheduled and in which room?",
        "answer": "March 15th at 3:00 PM in Room 407"
    },
    {
        "needle": "The API key for the production server is 'sk-prod-7x9mK2nP4qR8'.",
        "question": "What is the API key for the production server?",
        "answer": "sk-prod-7x9mK2nP4qR8"
    },
]


def generate_haystack(target_tokens: int, tokenizer) -> str:
    """Generate filler text of approximately target_tokens length."""
    filler_tokens = len(tokenizer.encode(HAYSTACK_FILLER, add_special_tokens=False))
    repeats = (target_tokens // filler_tokens) + 1
    haystack = (HAYSTACK_FILLER.strip() + "\n\n") * repeats
    
    # Trim to approximately target length
    tokens = tokenizer.encode(haystack, add_special_tokens=False)
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def insert_needle(haystack: str, needle: str, depth_percent: float, tokenizer) -> str:
    """
    Insert needle at depth_percent position in the haystack.
    
    Args:
        haystack: The filler text
        needle: The fact to hide
        depth_percent: 0.0 = beginning, 0.5 = middle, 1.0 = end
    """
    tokens = tokenizer.encode(haystack, add_special_tokens=False)
    needle_tokens = tokenizer.encode(f"\n\n{needle}\n\n", add_special_tokens=False)
    
    insert_pos = int(len(tokens) * depth_percent)
    
    new_tokens = tokens[:insert_pos] + needle_tokens + tokens[insert_pos:]
    return tokenizer.decode(new_tokens)


def create_prompt(context: str, question: str) -> str:
    """Create the full prompt with context and question."""
    return f"""<|im_start|>system
You are a helpful assistant. Answer questions based on the provided context.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""


def check_answer(generated: str, expected: str) -> Tuple[bool, float]:
    """
    Check if the generated answer contains the expected answer.
    
    Returns:
        (exact_match, partial_score)
    """
    generated_lower = generated.lower().strip()
    expected_lower = expected.lower().strip()
    
    # Exact match
    if expected_lower in generated_lower:
        return True, 1.0
    
    # Partial match (check key parts)
    parts = expected_lower.split()
    matches = sum(1 for part in parts if part in generated_lower)
    partial_score = matches / len(parts) if parts else 0.0
    
    return False, partial_score


def run_needle_test(
    model,
    tokenizer,
    context_length: int,
    depth_percent: float,
    needle_info: dict,
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> Dict:
    """
    Run a single needle-in-haystack test.
    
    Returns:
        Dict with results including success, generated text, timing
    """
    # Generate haystack and insert needle
    haystack = generate_haystack(context_length, tokenizer)
    context_with_needle = insert_needle(haystack, needle_info["needle"], depth_percent, tokenizer)
    
    # Create prompt
    prompt = create_prompt(context_with_needle, needle_info["question"])
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    actual_context_len = inputs.input_ids.shape[1]
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )
    gen_time = time.time() - start_time
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Check answer
    exact_match, partial_score = check_answer(response, needle_info["answer"])
    
    # Clean up to prevent memory accumulation
    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "context_length": actual_context_len,
        "depth_percent": depth_percent,
        "needle": needle_info["needle"][:50] + "...",
        "expected": needle_info["answer"],
        "generated": response.strip()[:100],
        "exact_match": exact_match,
        "partial_score": partial_score,
        "gen_time": gen_time,
    }


def run_benchmark(
    model,
    tokenizer,
    context_lengths: List[int],
    depth_percents: List[float],
    model_name: str,
    device: str = "cuda"
) -> List[Dict]:
    """Run full needle-in-haystack benchmark."""
    results = []
    
    for ctx_len in context_lengths:
        for depth in depth_percents:
            for needle_info in NEEDLES:
                print(f"  Testing: ctx={ctx_len}, depth={depth:.0%}, needle='{needle_info['answer'][:20]}...'")
                
                try:
                    result = run_needle_test(
                        model, tokenizer, ctx_len, depth, needle_info, device=device
                    )
                    result["model"] = model_name
                    results.append(result)
                    
                    status = "PASS" if result["exact_match"] else f"FAIL ({result['partial_score']:.0%})"
                    print(f"    -> {status}: '{result['generated'][:50]}...'")
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"    -> OOM at context length {ctx_len}")
                    results.append({
                        "model": model_name,
                        "context_length": ctx_len,
                        "depth_percent": depth,
                        "error": "OOM"
                    })
                    torch.cuda.empty_cache()
                    
    return results


def print_summary(results: List[Dict], model_name: str):
    """Print summary statistics."""
    valid_results = [r for r in results if "exact_match" in r]
    
    if not valid_results:
        print(f"\n{model_name}: No valid results")
        return
    
    total = len(valid_results)
    exact_matches = sum(1 for r in valid_results if r["exact_match"])
    avg_partial = sum(r["partial_score"] for r in valid_results) / total
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Total tests: {total}")
    print(f"Exact matches: {exact_matches}/{total} ({100*exact_matches/total:.1f}%)")
    print(f"Average partial score: {avg_partial:.1%}")
    
    # By depth
    print(f"\nBy Depth:")
    depths = sorted(set(r["depth_percent"] for r in valid_results))
    for depth in depths:
        depth_results = [r for r in valid_results if r["depth_percent"] == depth]
        depth_matches = sum(1 for r in depth_results if r["exact_match"])
        print(f"  {depth:.0%}: {depth_matches}/{len(depth_results)} ({100*depth_matches/len(depth_results):.1f}%)")
    
    # By context length
    print(f"\nBy Context Length:")
    ctx_lens = sorted(set(r["context_length"] for r in valid_results))
    for ctx_len in ctx_lens:
        ctx_results = [r for r in valid_results if r["context_length"] == ctx_len]
        ctx_matches = sum(1 for r in ctx_results if r["exact_match"])
        print(f"  {ctx_len:,}: {ctx_matches}/{len(ctx_results)} ({100*ctx_matches/len(ctx_results):.1f}%)")

def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    collect()

def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--bandwidth", type=int, default=128, help="CIMBA bandwidth S")
    parser.add_argument("--max-context", type=int, default=4096, help="Maximum context length")
    parser.add_argument("--depths", type=int, default=5, help="Number of depth positions to test")
    parser.add_argument("--output", default="outputs/needle_haystack_results.json", help="Output file")
    parser.add_argument("--skip-standard", action="store_true", help="Skip standard attention test")
    parser.add_argument("--skip-cimba", action="store_true", help="Skip CIMBA test")
    parser.add_argument("--skip-fmma", action="store_true", help="Skip FMMA test")
    parser.add_argument("--fmma-channels", type=int, default=4, help="FMMA num_channels (r)")
    parser.add_argument("--fmma-bandwidth", type=int, default=32, help="FMMA bandwidth (S)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Context lengths to test (exponential scale)
    context_lengths = []
    ctx = 4096
    while ctx <= args.max_context:
        context_lengths.append(ctx)
        ctx *= 2
    
    # Depth positions (0% = start, 50% = middle, 100% = end)
    depth_percents = [i / (args.depths - 1) for i in range(args.depths)]
    
    print(f"\nBenchmark Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Depth positions: {[f'{d:.0%}' for d in depth_percents]}")
    print(f"  Needles: {len(NEEDLES)}")
    print(f"  Total tests per model: {len(context_lengths) * len(depth_percents) * len(NEEDLES)}")
    
    all_results = {
        "config": {
            "model": args.model,
            "bandwidth": args.bandwidth,
            "fmma_channels": args.fmma_channels,
            "fmma_bandwidth": args.fmma_bandwidth,
            "context_lengths": context_lengths,
            "depth_percents": depth_percents,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {}
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Test 1: Standard Attention
    if not args.skip_standard:
        print(f"\n{'='*60}")
        print("Testing: Standard Attention")
        print(f"{'='*60}")
        
        model_std = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16
        ).to(device)
        
        model_std.eval()  # Ensure eval mode for fair comparison

        reset_memory_stats()
        csa_base_memory = get_gpu_memory_mb()

        results_std = run_benchmark(
            model_std, tokenizer, context_lengths, depth_percents,
            "Standard Attention", device=device
        )
        all_results["results"]["standard"] = results_std
        print_summary(results_std, "Standard Attention")
        
        del model_std
        torch.cuda.empty_cache()
    
    # Test 2: CIMBA Spectral Attention
    if not args.skip_cimba:
        print(f"\n{'='*60}")
        print(f"Testing: CIMBA Spectral Attention (S={args.bandwidth})")
        print(f"{'='*60}")
        
        model_cimba = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16
        ).to(device)
        model_cimba = replace_attention_with_spectral(
            model_cimba, bandwidth_target=args.bandwidth
        )
        # Enable filter caching optimization
        model_cimba.eval()
        
    
        reset_memory_stats()
        csa_base_memory = get_gpu_memory_mb()
        results_cimba = run_benchmark(
            model_cimba, tokenizer, context_lengths, depth_percents,
            f"CIMBA Spectral (S={args.bandwidth})", device=device
        )
        all_results["results"]["cimba_spectral"] = results_cimba
        print_summary(results_cimba, f"CIMBA Spectral (S={args.bandwidth})")
        
        del model_cimba
        torch.cuda.empty_cache()
    
    # Test 3: FMMA (Fejér Mixture Moment Attention)
    if not args.skip_fmma:
        print(f"\n{'='*60}")
        print(f"Testing: FMMA (r={args.fmma_channels}, S={args.fmma_bandwidth})")
        print(f"{'='*60}")
        print(f"  Note: FMMA has CONSTANT memory O(rS) regardless of context length!")
        print(f"  Theoretical memory: {args.fmma_channels * args.fmma_bandwidth * 64 * 8 / 1024:.1f} KB per head")
        
        model_fmma = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16
        ).to(device)
        model_fmma = replace_attention_with_fmma(
            model_fmma,
            num_channels=args.fmma_channels,
            bandwidth=args.fmma_bandwidth,
            max_seq_len=args.max_context * 2,
        )
        model_fmma.eval()
        
        reset_memory_stats()
        results_fmma = run_benchmark(
            model_fmma, tokenizer, context_lengths, depth_percents,
            f"FMMA (r={args.fmma_channels}, S={args.fmma_bandwidth})", device=device
        )
        all_results["results"]["fmma"] = results_fmma
        print_summary(results_fmma, f"FMMA (r={args.fmma_channels}, S={args.fmma_bandwidth})")
        
        del model_fmma
        torch.cuda.empty_cache()
    
    # Comparison
    if "standard" in all_results["results"]:
        print(f"\n{'='*60}")
        print("COMPARISON: All Methods")
        print(f"{'='*60}")
        
        std_results = [r for r in all_results["results"]["standard"] if "exact_match" in r]
        
        if std_results:
            std_exact = sum(1 for r in std_results if r["exact_match"])
            std_partial = sum(r["partial_score"] for r in std_results) / len(std_results)
            
            print(f"\n{'Method':<30} {'Exact Match':<15} {'Partial Score':<15} {'Memory':<15}")
            print("-" * 75)
            print(f"{'Standard':<30} {100*std_exact/len(std_results):<15.1f}% {100*std_partial:<15.1f}% {'O(N)':<15}")
            
            if "cimba_spectral" in all_results["results"]:
                cimba_results = [r for r in all_results["results"]["cimba_spectral"] if "exact_match" in r]
                if cimba_results:
                    cimba_exact = sum(1 for r in cimba_results if r["exact_match"])
                    cimba_partial = sum(r["partial_score"] for r in cimba_results) / len(cimba_results)
                    print(f"{'CIMBA Spectral':<30} {100*cimba_exact/len(cimba_results):<15.1f}% {100*cimba_partial:<15.1f}% {'O(N) K, O(S) V':<15}")
            
            if "fmma" in all_results["results"]:
                fmma_results = [r for r in all_results["results"]["fmma"] if "exact_match" in r]
                if fmma_results:
                    fmma_exact = sum(1 for r in fmma_results if r["exact_match"])
                    fmma_partial = sum(r["partial_score"] for r in fmma_results) / len(fmma_results)
                    print(f"{'FMMA':<30} {100*fmma_exact/len(fmma_results):<15.1f}% {100*fmma_partial:<15.1f}% {'O(rS) CONSTANT':<15}")
            
            print("\nMemory Legend:")
            print("  O(N) = grows linearly with sequence length")
            print("  O(S) = grows with bandwidth S only")
            print("  O(rS) = grows with channels r × bandwidth S (CONSTANT in sequence length)")
    
    # Save results
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
