"""
Backwards-compatibility shim — re-exports everything from the fmma/ package.

All implementation has moved to:
    fmma/regularization.py  — PSD Toeplitz penalty, bandwidth regularization
    fmma/cache.py           — FMMACache
    fmma/features.py        — NonnegativeKeyFeatures, QueryMomentNetwork
    fmma/attention.py       — FMMAAttention
    fmma/integration.py     — replace_attention_with_fmma, create_fmma_cache
    fmma/tests.py           — All acceptance tests

Existing code that does `from fmma_attention import ...` will continue to work.
"""

from fmma import (
    # Regularization
    build_toeplitz_matrix,
    psd_toeplitz_penalty,
    bandwidth_regularization,
    # Cache
    FMMACache,
    # Features
    NonnegativeKeyFeatures,
    QueryMomentNetwork,
    # Attention
    FMMAAttention,
    # Integration
    create_fmma_cache,
    replace_attention_with_fmma,
    _KNOWN_ATTENTION_CLASSES,
    # Tests
    run_all_tests,
)


if __name__ == "__main__":
    run_all_tests()
