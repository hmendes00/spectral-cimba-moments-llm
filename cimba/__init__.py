"""
CIMBA: Constant-Size Invertible Moment Sketch for Bandlimited Attention

This module implements the moment sketching and recovery mechanisms
described in the CIMBA paper for approximating attention via learned
Fourier coefficients.
"""

from .sketch import CIMBASketch
from .recovery import CIMBARecovery

__all__ = ["CIMBASketch", "CIMBARecovery"]
