"""Check what attention mask Qwen actually passes to attention layers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
inputs = tokenizer("def hello():", return_tensors="pt")

# Hook to capture all args
def hook(module, args, kwargs):
    print(f"\n=== Attention Forward Called ===")
    print(f"Number of positional args: {len(args)}")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            print(f"  arg[{i}] shape: {arg.shape}")
        else:
            print(f"  arg[{i}] type: {type(arg)}")
    print(f"Keyword args: {list(kwargs.keys())}")
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k} shape: {v.shape}")
        elif isinstance(v, tuple) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            print(f"  {k} is tuple of tensors, first shape: {v[0].shape}")
        else:
            print(f"  {k}: {type(v)}")
    return None

# Register hook on first layer's attention
layer0 = model.model.layers[0].self_attn
handle = layer0.register_forward_pre_hook(hook, with_kwargs=True)

# Run forward
with torch.no_grad():
    outputs = model(**inputs)

handle.remove()
