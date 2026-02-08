"""
Integration helpers: model surgery (replace_attention_with_fmma) and cache creation.
"""

from typing import Optional, List

import torch

from .cache import FMMACache
from .attention import FMMAAttention


def create_fmma_cache(
    model,
    num_channels: int = None,
    bandwidth: int = None,
    max_seq_len: int = None,
) -> FMMACache:
    """
    Create an FMMA cache for a model.

    Auto-detects num_channels, bandwidth, and max_seq_len from the first
    FMMAAttention layer found in the model.  Explicit overrides take priority.
    """
    config = model.config

    # Auto-detect FMMA hyperparameters from the model's own layers
    detected_channels, detected_bw, detected_msl = None, None, None
    for module in model.modules():
        if isinstance(module, FMMAAttention):
            detected_channels = module.num_channels
            detected_bw = module.bandwidth
            detected_msl = module.max_seq_len
            break

    num_channels = num_channels or detected_channels or 4
    bandwidth = bandwidth or detected_bw or 64
    max_seq_len = max_seq_len or detected_msl or 8192

    return FMMACache(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        num_channels=num_channels,
        bandwidth=bandwidth,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )


# Fix 4.1 (PLAN.md): Known attention class names for targeted replacement.
_KNOWN_ATTENTION_CLASSES = {
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen2FlashAttention2",
    "LlamaAttention",
    "LlamaSdpaAttention",
    "LlamaFlashAttention2",
    "MistralAttention",
    "MistralSdpaAttention",
    "MistralFlashAttention2",
    "GemmaAttention",
    "GemmaSdpaAttention",
    "Phi3Attention",
    "Phi3SdpaAttention",
    "Starcoder2Attention",
}


def replace_attention_with_fmma(
    model,
    num_channels: int = 4,
    bandwidth: int = 64,
    max_seq_len: int = 8192,
    layers_to_replace: Optional[List[int]] = None,
):
    """
    Replace model's attention layers with FMMA attention.

    Fix 4.1: Targets known transformer attention classes instead of fragile
             string matching with 'attn'.
    Fix 4.2: Uses repeat_interleave for GQA head mapping.
    """
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    if layers_to_replace is None:
        layers_to_replace = list(range(num_layers))

    replaced = 0

    for name, module in list(model.named_modules()):
        class_name = type(module).__name__

        # Fix 4.1: Match by known class OR by structural signature
        is_known_class = class_name in _KNOWN_ATTENTION_CLASSES
        is_structural_match = (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
        )

        if not (is_known_class or is_structural_match):
            continue

        # Extract layer index
        layer_idx = None
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                break

        if layer_idx is None or layer_idx not in layers_to_replace:
            continue

        device = module.q_proj.weight.device
        dtype = module.q_proj.weight.dtype

        fmma = FMMAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_channels=num_channels,
            bandwidth=bandwidth,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            num_kv_heads=num_kv_heads,
        )
        fmma = fmma.to(device=device, dtype=dtype)

        # Copy Q projection (always full num_heads)
        fmma.q_proj.weight.data.copy_(module.q_proj.weight.data)

        # Copy K, V projections — FMMA stores them at num_kv_heads size
        # (expansion happens in forward via _expand_kv_heads)
        k_weight = module.k_proj.weight.data
        v_weight = module.v_proj.weight.data

        if k_weight.shape[0] == fmma.k_proj.weight.shape[0]:
            fmma.k_proj.weight.data.copy_(k_weight)
            fmma.v_proj.weight.data.copy_(v_weight)
        elif k_weight.shape[0] < fmma.k_proj.weight.shape[0]:
            # Original is GQA (fewer KV heads) but FMMA was created with
            # num_kv_heads — this shouldn't happen with new constructor, but
            # handle gracefully.  Fix 4.2: repeat_interleave
            repeat_factor = fmma.k_proj.weight.shape[0] // k_weight.shape[0]
            fmma.k_proj.weight.data.copy_(
                k_weight.repeat_interleave(repeat_factor, dim=0)
            )
            fmma.v_proj.weight.data.copy_(
                v_weight.repeat_interleave(repeat_factor, dim=0)
            )
        else:
            fmma.k_proj.weight.data.copy_(k_weight)
            fmma.v_proj.weight.data.copy_(v_weight)

        fmma.o_proj.weight.data.copy_(module.o_proj.weight.data)

        # Replace module in parent
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, fmma)

        replaced += 1
        print(f"  Replaced layer {layer_idx} ({class_name}) with FMMA")

    print(f"Total layers replaced: {replaced}")
    return model
