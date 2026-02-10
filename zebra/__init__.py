"""
ZEBRA — Zero-Training Eigen-Bandlimited Residual Approximation

Post-training weight compression with certified error bounds:
    W ≈ A^T · P_K(AWB^T) · B  +  UV^T

where A,B are from a fast orthogonal family (scrambled Hadamard),
P_K keeps the K largest-magnitude transform-domain coefficients,
and UV^T is a rank-r low-rank correction of the residual.

Storage: mn → O(K + r(m+n))
Cost:    O(mn) → O(m log m + K + n log n + r(m+n))
"""

from .core import (
    hadamard_matrix,
    fast_hadamard_transform,
    random_sign_diagonal,
    random_permutation,
    build_fast_orthogonal,
    apply_fast_orthogonal,
    build_orthogonal_matrix,
    compute_W_tilde,
    reconstruct_from_sparse,
    reconstruct_from_indices,
    top_k_sparsify,
    top_k_indices_only,
    top_k_tail_energy,
    zebra_decompose,
    certificate_curve_sparse,
    certificate_curve_residual,
    select_transforms,
    analyze_compressibility,
    compressibility_sweep,
)
from .layer import ZEBRALinear

__all__ = [
    "hadamard_matrix",
    "fast_hadamard_transform",
    "random_sign_diagonal",
    "random_permutation",
    "build_fast_orthogonal",
    "apply_fast_orthogonal",
    "build_orthogonal_matrix",
    "compute_W_tilde",
    "reconstruct_from_sparse",
    "reconstruct_from_indices",
    "top_k_sparsify",
    "top_k_indices_only",
    "top_k_tail_energy",
    "zebra_decompose",
    "certificate_curve_sparse",
    "certificate_curve_residual",
    "select_transforms",
    "analyze_compressibility",
    "compressibility_sweep",
    "ZEBRALinear",
]
