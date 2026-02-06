"""
HumanEval Evaluation Harness for CIMBA Models

Evaluates code generation models on the OpenAI HumanEval benchmark.
Computes pass@1 with deterministic decoding.

Usage:
    python eval_humaneval.py --model qwen  # Baseline Qwen
    python eval_humaneval.py --model cimba --checkpoint outputs/spectral_finetune/final_model
"""

import argparse
import json
import subprocess
import tempfile
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32 if DEVICE == 'cpu' else torch.bfloat16

# Evaluation settings
MAX_NEW_TOKENS = 512
TIMEOUT_SECONDS = 10  # Per-problem execution timeout


def load_humaneval() -> List[Dict]:
    """Load HumanEval dataset from HuggingFace."""
    dataset = load_dataset("openai_humaneval", split="test")
    return list(dataset)


def extract_completion(completion: str) -> str:
    """
    Clean up the completion by stopping at structural markers.
    
    The completion is already just the generated tokens (prompt not included).
    We stop at markers that indicate end of the target function.
    """
    # Stop at structural end-of-function markers only
    # Don't stop at "\n#" - valid code often has comments
    stop_sequences = [
        "\nclass ",
        "\ndef ",
        "\nif __name__",
        "\n\n\n\n",  # 4+ blank lines strongly indicates end
    ]
    
    for stop in stop_sequences:
        if stop in completion:
            completion = completion[:completion.index(stop)]
    
    return completion


def execute_code_safely(
    code: str,
    test_code: str,
    timeout: int = TIMEOUT_SECONDS,
) -> Tuple[bool, str]:
    """
    Execute code + tests in a sandboxed subprocess with timeout.
    
    Returns:
        (passed: bool, error_message: str)
    """
    # Combine the generated code with the test
    full_code = code + "\n" + test_code
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(full_code)
        temp_path = f.name
    
    try:
        # Run in subprocess with timeout
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            error = result.stderr[:500] if result.stderr else "Unknown error"
            return False, error
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)[:500]
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Generate completion with deterministic decoding.
    
    Settings for pass@1:
    - do_sample=False (greedy)
    - Explicitly clear sampling params (temperature/top_p/top_k/repetition_penalty)
    - Stop at EOS
    - Decode only NEW tokens (not including prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for pass@1
            # Explicitly clear sampling params to avoid HF warnings
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the NEW tokens (completion), not the prompt
    gen_ids = outputs[0][input_len:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return completion


def evaluate_humaneval(
    model,
    tokenizer,
    problems: List[Dict],
    num_problems: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model on HumanEval problems.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        problems: List of HumanEval problems
        num_problems: Limit number of problems (None = all 164)
        verbose: Print progress and results
        
    Returns:
        Dictionary with pass@1 and per-problem results
    """
    if num_problems:
        problems = problems[:num_problems]
    
    results = []
    passed = 0
    total = len(problems)
    
    model.eval()
    
    iterator = tqdm(problems, desc="Evaluating") if verbose else problems
    
    for problem in iterator:
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        test = problem["test"]
        entry_point = problem["entry_point"]
        
        # Generate completion (returns only new tokens, not prompt)
        completion = generate_completion(model, tokenizer, prompt)
        completion = extract_completion(completion)
        
        # Combine prompt + completion for full function
        full_code = prompt + completion
        
        # Build test code
        test_code = test + f"\ncheck({entry_point})"
        
        # Execute
        success, error = execute_code_safely(full_code, test_code)
        
        results.append({
            "task_id": task_id,
            "prompt": prompt,
            "completion": completion,
            "passed": success,
            "error": error if not success else None,
        })
        
        if success:
            passed += 1
        
        if verbose:
            iterator.set_postfix({"pass@1": f"{passed}/{len(results)}"})
    
    pass_at_1 = passed / total if total > 0 else 0.0
    
    summary = {
        "pass@1": pass_at_1,
        "passed": passed,
        "total": total,
        "results": results,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"HumanEval Results: {passed}/{total} = {pass_at_1:.1%} pass@1")
        print(f"{'='*50}")
    
    return summary


def load_qwen_baseline():
    """Load baseline Qwen2.5-Coder-0.5B-Instruct model."""
    print("Loading Qwen2.5-Coder-0.5B-Instruct (baseline)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_cimba_model(checkpoint_path: Optional[str] = None, bandwidth: int = 64):
    """
    Load model with CIMBA attention.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint (None = untrained CIMBA)
        bandwidth: CIMBA S parameter
    """
    from cimba_attention import replace_attention_with_cimba
    
    if checkpoint_path:
        print(f"Loading CIMBA model from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        print("Loading Qwen base + replacing with CIMBA attention (untrained)...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Replace attention with CIMBA (whether checkpoint or fresh)
    # For checkpoint, this re-applies CIMBA structure with trained weights
    model = replace_attention_with_cimba(model, bandwidth=bandwidth)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="HumanEval evaluation for CIMBA")
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen", "cimba"],
        default="qwen",
        help="Model type: 'qwen' (baseline) or 'cimba'"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to CIMBA checkpoint (for --model cimba)"
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=64,
        help="CIMBA bandwidth S parameter"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Limit number of problems (default: all 164)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with 10 problems"
    )
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.num_problems = 10
    
    # Load model
    if args.model == "qwen":
        model, tokenizer = load_qwen_baseline()
        model_name = "qwen-baseline"
    else:
        model, tokenizer = load_cimba_model(args.checkpoint, args.bandwidth)
        model_name = f"cimba-S{args.bandwidth}"
        if args.checkpoint:
            model_name += f"-{Path(args.checkpoint).name}"
        else:
            model_name += "-untrained"
    
    # Load HumanEval
    print("Loading HumanEval dataset...")
    problems = load_humaneval()
    print(f"Loaded {len(problems)} problems")
    
    # Evaluate
    print(f"\nEvaluating {model_name}...")
    results = evaluate_humaneval(
        model,
        tokenizer,
        problems,
        num_problems=args.num_problems,
        verbose=True,
    )
    
    # Add metadata
    results["model"] = model_name
    results["settings"] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "decoding": "greedy (do_sample=False)",
    }
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"outputs/humaneval_{model_name}.json"
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
