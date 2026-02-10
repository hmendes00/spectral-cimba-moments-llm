"""
HSWFLinear — Drop-in replacement for nn.Linear using HSWF compression.

Forward cost:  O(d log d + rd)  instead of  O(d²)
Storage:       O(S + rd)         instead of  O(d²)

Usage:
    # Compress an existing layer
    compressed = HSWFLinear.from_linear(model.layer.q_proj, S=64, r=16)

    # Replace in model
    model.layer.q_proj = compressed

    # Same forward API
    output = compressed(input)  # [..., d] -> [..., d]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    wrap_diagonal_average,
    circulant_spectrum,
    circulant_from_column,
    bandlimit_spectrum,
)


class HSWFLinear(nn.Module):
    """
    Square linear layer compressed via HSWF: y = x @ (C_S + UV^T)^T + bias.

    The circulant component C_S is applied via FFT in O(d log d).
    The low-rank correction UV^T is applied as two matmuls in O(rd).

    Attributes:
        c_S:         [d]    first column of bandlimited circulant (buffer)
        _c_spectrum: [d]    precomputed FFT(c_S) for fast forward (buffer)
        U:           [d, r] left factor of low-rank correction (buffer)
        V:           [d, r] right factor of low-rank correction (buffer)
        bias:        [d]    optional bias (buffer)
    """

    def __init__(
        self,
        d: int,
        S: int,
        r: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d = d
        self.S = S
        self.r = r

        factory = {"device": device, "dtype": dtype or torch.float32}

        self.register_buffer("c_S", torch.zeros(d, **factory))
        self.register_buffer(
            "_c_spectrum", torch.zeros(d, dtype=torch.complex64, device=device)
        )
        self.register_buffer("U", torch.zeros(d, r, **factory))
        self.register_buffer("V", torch.zeros(d, r, **factory))

        if bias:
            self.register_buffer("bias", torch.zeros(d, **factory))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        S: int,
        r: int,
        bandlimit_strategy: str = "energy",
    ) -> "HSWFLinear":
        """
        Compress an existing nn.Linear using HSWF (deterministic, no training).

        Args:
            linear: the nn.Linear to compress (must be square: in == out)
            S: number of Fourier modes to retain
            r: rank of low-rank correction
            bandlimit_strategy: "energy" (optimal) or "lowpass"

        Returns:
            HSWFLinear with pre-computed components
        """
        W = linear.weight.data.float()   # [d, d]
        m, n = W.shape
        assert m == n, (
            f"HSWFLinear requires square weight (got {W.shape}). "
            f"For rectangular, use block-wise HSWF or pure low-rank."
        )
        d = m
        device = W.device

        # Step 1: Best circulant via wrap-diagonal averaging
        c_star = wrap_diagonal_average(W)
        spectrum_full = circulant_spectrum(c_star)

        # Step 2: Bandlimit
        S = min(S, d)
        spectrum_S, kept_idx = bandlimit_spectrum(
            spectrum_full, S, strategy=bandlimit_strategy
        )
        c_S = torch.fft.ifft(spectrum_S).real

        # Step 3-4: Residual + rank-r SVD
        C_S_mat = circulant_from_column(c_S)
        R = W - C_S_mat

        r = min(r, d)
        U_full, sigma, Vt_full = torch.linalg.svd(R, full_matrices=False)
        sigma_r = sigma[:r]
        sqrt_sigma = sigma_r.sqrt()

        # Absorb σ^{1/2} into U and V: UV^T = (U√σ)(V√σ)^T = UσV^T = R_r
        U_r = U_full[:, :r] * sqrt_sigma.unsqueeze(0)     # [d, r]
        V_r = Vt_full[:r, :].T * sqrt_sigma.unsqueeze(0)  # [d, r]

        # Compute reconstruction error
        R_r = U_r @ V_r.T
        total_err = (W - C_S_mat - R_r).norm().item()
        rel_err = total_err / max(W.norm().item(), 1e-10)

        # Build module
        layer = cls(
            d=d, S=S, r=r,
            bias=linear.bias is not None,
            device=device, dtype=torch.float32,
        )

        layer.c_S.copy_(c_S)
        layer._c_spectrum.copy_(torch.fft.fft(c_S.float()))
        layer.U.copy_(U_r)
        layer.V.copy_(V_r)

        if linear.bias is not None:
            layer.bias.copy_(linear.bias.data.float())

        # Report
        dense = d * d
        compressed = 2 * S + 2 * r * d
        print(f"  HSWF compressed: {dense:,} → {compressed:,} params "
              f"({dense / compressed:.1f}× compression, "
              f"{rel_err * 100:.2f}% error)")

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = x @ (C_S + UV^T)^T + bias
          = x @ C_S^T + x @ V @ U^T + bias

        Circulant part via FFT: x @ C_S^T = IFFT(FFT(c_S) · FFT(x))
        Low-rank part via matmul: x @ V @ U^T

        Cost: O(d log d + rd)
        """
        dtype_in = x.dtype
        x_f = x.float()

        # Circulant: O(d log d)
        x_fft = torch.fft.fft(x_f, dim=-1)
        circ_out = torch.fft.ifft(self._c_spectrum * x_fft, dim=-1).real

        # Low-rank: O(rd)
        lr_out = (x_f @ self.V) @ self.U.T

        out = circ_out + lr_out

        if self.bias is not None:
            out = out + self.bias

        return out.to(dtype_in)

    def reconstruction_error(self, linear: nn.Linear) -> float:
        """Compute ||W - Ŵ||_F / ||W||_F against the original dense layer."""
        W = linear.weight.data.float()
        W_hat = circulant_from_column(self.c_S) + self.U @ self.V.T
        return (W - W_hat).norm().item() / max(W.norm().item(), 1e-10)

    def compression_info(self) -> dict:
        """Report compression statistics."""
        dense = self.d * self.d
        compressed = 2 * self.S + 2 * self.r * self.d
        if self.bias is not None:
            dense += self.d
            compressed += self.d

        return {
            "d": self.d,
            "S": self.S,
            "r": self.r,
            "dense_params": dense,
            "hswf_params": compressed,
            "compression_ratio": dense / max(compressed, 1),
        }

    def extra_repr(self) -> str:
        return (
            f"d={self.d}, S={self.S}, r={self.r}, "
            f"bias={self.bias is not None}"
        )
