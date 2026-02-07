"""
Fejér Mixture Moment Attention (FMMA)

A constant-memory, content-aware attention operator on the circle.

Key properties:
- Content awareness via r nonnegative key feature channels
- Position awareness via bandlimited Fejér densities
- EXACT evaluation from O(rS) cached moments (independent of sequence length)
- Nonnegative weights guaranteed by PSD Toeplitz + Fejér reconstruction
- Both K and V become O(rS) — 100% KV cache savings!

Package layout:
    fmma.regularization   — PSD Toeplitz penalty, bandwidth regularization
    fmma.cache            — FMMACache (O(rS) moment cache)
    fmma.features         — NonnegativeKeyFeatures, QueryMomentNetwork
    fmma.attention        — FMMAAttention (core module)
    fmma.integration      — replace_attention_with_fmma, create_fmma_cache
    fmma.tests            — 11 acceptance tests
"""

# Regularization losses
from .regularization import (
    build_toeplitz_matrix,
    psd_toeplitz_penalty,
    bandwidth_regularization,
)

# Cache
from .cache import FMMACache

# Feature networks
from .features import NonnegativeKeyFeatures, QueryMomentNetwork

# Core attention
from .attention import FMMAAttention

# Integration helpers
from .integration import (
    create_fmma_cache,
    replace_attention_with_fmma,
    _KNOWN_ATTENTION_CLASSES,
)

# Tests
from .tests import run_all_tests

__all__ = [
    # Regularization
    "build_toeplitz_matrix",
    "psd_toeplitz_penalty",
    "bandwidth_regularization",
    # Cache
    "FMMACache",
    # Features
    "NonnegativeKeyFeatures",
    "QueryMomentNetwork",
    # Attention
    "FMMAAttention",
    # Integration
    "create_fmma_cache",
    "replace_attention_with_fmma",
    "_KNOWN_ATTENTION_CLASSES",
    # Tests
    "run_all_tests",
]
