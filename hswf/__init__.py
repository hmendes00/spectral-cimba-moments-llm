"""
Harmonic Structured Weight Factorization (HSWF)

Deterministic, post-training weight compression:
    W ≈ Ŵ = C_S + UV^T
where C_S is a bandlimited circulant (S Fourier coefficients)
and UV^T is a rank-r low-rank correction.

Storage: d² → O(S + rd)
Cost:    O(d²) → O(d log d + rd)
"""

from .core import (
    wrap_diagonal_average,
    circulant_spectrum,
    circulant_from_column,
    bandlimit_spectrum,
    circulant_matvec,
    hswf_decompose,
    analyze_compressibility,
    compressibility_sweep,
)
from .layer import HSWFLinear

__all__ = [
    "wrap_diagonal_average",
    "circulant_spectrum",
    "circulant_from_column",
    "bandlimit_spectrum",
    "circulant_matvec",
    "hswf_decompose",
    "analyze_compressibility",
    "compressibility_sweep",
    "HSWFLinear",
]
