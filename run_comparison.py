"""Quick two-model ZEBRA comparison: reference model + Qwen Coder next size."""
import subprocess, sys

models = [
    ("microsoft/phi-2", "outputs/zebra_phi2", "2"),         # ~2.7B reference
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "outputs/zebra_qwen1.5b", "2"),  # Qwen next
]

for model, outdir, layers in models:
    print(f"\n{'='*70}")
    print(f"RUNNING: {model}  (max-layers {layers})")
    print(f"{'='*70}\n")
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable, "analyze_zebra.py",
         "--model", model,
         "--max-layers", layers,
         "--output-dir", outdir],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"FAILED with code {result.returncode}")
