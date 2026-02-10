"""
ZEBRALinear — Drop-in replacement for nn.Linear using ZEBRA compression.

Forward cost:  O(m log m + K + n log n + r(m+n))  instead of  O(mn)
Storage:       O(K + r(m+n))                       instead of  O(mn)

No dense d×d matrices are ever materialized — all transforms use
apply_fast_orthogonal (O(d log d) per vector), and the transform-domain
coefficients are stored as a true sparse COO tensor.

Usage:
    # Compress an existing layer
    compressed = ZEBRALinear.from_linear(model.layer.q_proj, K=256, r=16)

    # Replace in model
    model.layer.q_proj = compressed

    # Same forward API
    output = compressed(input)  # [..., in] -> [..., out]
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .core import (
    build_fast_orthogonal,
    apply_fast_orthogonal,
    compute_W_tilde,
    reconstruct_from_sparse,
    top_k_sparsify,
    select_transforms,
)


class ZEBRALinear(nn.Module):
    """
    Linear layer compressed via ZEBRA:
        y = x @ (A^T W̃_K B + UV^T)^T + bias

    The transform-sparse component is applied via fast transforms
    and cached sparse COO tensor (no dense matrix materialization):
        1. x' = x @ B^T        (fast orthogonal, O(n log n))
        2. z  = x' @ W̃_K^T     (sparse matmul via COO, O(K))
        3. y₀ = z @ A           (fast orthogonal, O(m log m))

    The low-rank correction is applied as:
        y₁ = (x @ V) @ U^T     (O(r(m+n)))

    Works for rectangular weight matrices (m ≠ n).

    Attributes:
        transform_A: dict     left transform descriptor (stored implicitly)
        transform_B: dict     right transform descriptor (stored implicitly)
        sparse_indices: [K,2] indices of kept transform coefficients
        sparse_values:  [K]   values of kept transform coefficients
        U:              [m,r] left factor of low-rank correction
        V:              [n,r] right factor of low-rank correction
        bias:           [m]   optional bias
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        K: int,
        r: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.out_features = out_features  # m
        self.in_features = in_features    # n
        self.K = K
        self.r = r

        factory = {"device": device, "dtype": dtype or torch.float32}

        # Sparse component: stored as index-value pairs
        self.register_buffer(
            "sparse_indices",
            torch.zeros(K, 2, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "sparse_values",
            torch.zeros(K, **factory),
        )

        # Low-rank correction: initialized as empty [m,0] / [n,0]
        # Actual r-rank tensors assigned in from_linear via _buffers[]
        self.register_buffer("U", torch.empty(out_features, 0, **factory))
        self.register_buffer("V", torch.empty(in_features, 0, **factory))

        # Bias
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, **factory))
        else:
            self.bias = None

        # Transform descriptors (will be set by from_linear)
        self.transform_A = None
        self.transform_B = None

        # Cached sparse COO tensor for efficient forward pass
        self._sparse_coo_cache = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        K: int,
        r: int,
        n_candidates: int = 4,
        n_factors: int = 3,
        seed_A: Optional[int] = None,
        seed_B: Optional[int] = None,
    ) -> "ZEBRALinear":
        """
        Compress an existing nn.Linear using ZEBRA (deterministic, no training).

        Uses fast transforms throughout — never materializes dense d×d matrices.

        Args:
            linear: the nn.Linear to compress
            K: number of transform-domain coefficients to keep
            r: rank of low-rank residual correction
            n_candidates: number of transform candidates to try
            n_factors: scrambling depth per transform
            seed_A: fixed seed for left transform (overrides search)
            seed_B: fixed seed for right transform (overrides search)

        Returns:
            ZEBRALinear with pre-computed components
        """
        W = linear.weight.data.float()  # [out, in] = [m, n]
        m, n = W.shape
        mn = m * n
        K = min(K, mn)
        r = min(r, min(m, n))
        device = W.device

        # Step 0: Select best transforms (or use fixed seeds)
        if seed_A is not None and seed_B is not None:
            best_sa, best_sb = seed_A, seed_B
        else:
            best_sa, best_sb, _ = select_transforms(
                W, K=K, n_candidates=n_candidates, n_factors=n_factors,
            )

        # Step 1: Build transforms
        transform_A = build_fast_orthogonal(m, seed=best_sa, device=device, n_factors=n_factors)
        transform_B = build_fast_orthogonal(n, seed=best_sb, device=device, n_factors=n_factors)

        # Step 2: W̃ = A W B^T via fast transforms (no dense matrix)
        W_tilde = compute_W_tilde(W, transform_A, transform_B)

        # Step 3: Top-K sparsification
        W_tilde_K, indices, values = top_k_sparsify(W_tilde, K)

        # Step 4: Reconstruct and get residual via fast transforms
        W_ABK = reconstruct_from_sparse(W_tilde_K, transform_A, transform_B)
        R = W - W_ABK

        # Step 5: Rank-r SVD of residual
        U_full, sigma, Vt_full = torch.linalg.svd(R, full_matrices=False)
        r = min(r, len(sigma))
        sigma_r = sigma[:r]

        if r > 0:
            sqrt_sigma = sigma_r.sqrt()
            U_r = U_full[:, :r] * sqrt_sigma.unsqueeze(0)     # [m, r]
            V_r = Vt_full[:r, :].T * sqrt_sigma.unsqueeze(0)  # [n, r]
            R_r = U_r @ V_r.T
        else:
            U_r = torch.empty(m, 0, device=device)
            V_r = torch.empty(n, 0, device=device)
            R_r = torch.zeros_like(W)

        # Compute reconstruction error
        total_err_sq = (W - W_ABK - R_r).square().sum().item()
        total_err = math.sqrt(max(total_err_sq, 0))
        rel_err = total_err / max(W.norm().item(), 1e-10)

        # Build module
        layer = cls(
            out_features=m,
            in_features=n,
            K=K,
            r=r,
            bias=linear.bias is not None,
            device=device,
            dtype=torch.float32,
        )

        layer.sparse_indices.copy_(indices[:K])
        layer.sparse_values.copy_(values[:K])

        # B1 fix: assign U/V directly into the buffer dict, no re-registration
        layer._buffers["U"] = U_r.contiguous()
        layer._buffers["V"] = V_r.contiguous()

        if linear.bias is not None:
            layer.bias.copy_(linear.bias.data.float())

        layer.transform_A = transform_A
        layer.transform_B = transform_B
        layer._dense_W_tilde_K = None  # lazily built in forward if needed
        layer._build_sparse_coo_cache()

        # Report (realistic: count index storage — 3K float-equiv per sparse entry)
        dense = mn
        compressed = K * 3 + r * (m + n)  # 3K (value + 2 idx) + low-rank
        params_ratio = compressed / max(dense, 1)
        if params_ratio <= 1.0:
            print(f"  ZEBRA compressed: {dense:,} → {compressed:,} params "
                  f"({1/params_ratio:.1f}× smaller, {rel_err * 100:.2f}% error)")
        else:
            print(f"  ZEBRA compressed: {dense:,} → {compressed:,} params "
                  f"({params_ratio:.1f}× EXPANSION, {rel_err * 100:.2f}% error)")

        return layer

    def _build_sparse_coo_cache(self):
        """Build a cached sparse COO tensor for efficient forward pass.

        E1 fix: stores W̃_K as true sparse COO instead of dense m×n matrix.
        Memory: O(K) instead of O(mn).
        """
        m, n = self.out_features, self.in_features
        dev = self.sparse_values.device
        if self.K > 0:
            idx = self.sparse_indices[:self.K].T.long()  # [2, K]
            vals = self.sparse_values[:self.K].float()
            self._sparse_coo_cache = torch.sparse_coo_tensor(
                idx, vals, size=(m, n), device=dev,
            ).coalesce()
        else:
            self._sparse_coo_cache = torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long, device=dev),
                torch.zeros(0, device=dev, dtype=torch.float32),
                size=(m, n),
            ).coalesce()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ (A^T W̃_K B + UV^T)^T + bias

        D2 fix: uses apply_fast_orthogonal — no dense d×d matrix
        materialization.  Sparse matmul via cached COO tensor (E1).

        Derivation:
            W ≈ A^T W̃_K B + UV^T
            y = x @ W^T
              = x @ B^T @ W̃_K^T @ A + x @ V @ U^T

        Steps:
            1. x' = apply_fast_orthogonal(x, B, transpose=True)     [O(n log n)]
            2. z  = sparse.mm(W̃_K, x'^T)^T                         [O(K)]
            3. y₀ = apply_fast_orthogonal(z, A, transpose=False)     [O(m log m)]
            4. y  = y₀ + (x @ V) @ U^T                              [O(r(m+n))]
        """
        dtype_in = x.dtype
        x_f = x.float()

        # Rebuild sparse COO cache if needed (e.g. after device move)
        if self._sparse_coo_cache is None or self._sparse_coo_cache.device != x.device:
            self._build_sparse_coo_cache()

        # Step 1: x' = x @ B^T via fast transform
        # v @ B^T = apply(v, B, transpose=True)
        x_Bt = apply_fast_orthogonal(x_f, self.transform_B, transpose=True, dim=-1)

        # Step 2: z = x' @ W̃_K^T
        batch_shape = x_Bt.shape[:-1]
        x_flat = x_Bt.reshape(-1, self.in_features)            # [B, n]

        # 2.3: Dense fallback when sparsity ratio > 0.2 (GEMM is faster)
        m, n = self.out_features, self.in_features
        if self.K > 0 and self.K / (m * n) > 0.2:
            # Dense path: materialize W̃_K and use GEMM
            if not hasattr(self, '_dense_W_tilde_K') or self._dense_W_tilde_K is None:
                self._dense_W_tilde_K = self._sparse_coo_cache.to_dense()
            z = x_flat @ self._dense_W_tilde_K.float().T          # [B, m]
        else:
            # Sparse path: COO matmul
            S = self._sparse_coo_cache.float()                    # [m, n] sparse
            z = torch.sparse.mm(S, x_flat.T).T                   # [B, m]

        z = z.reshape(*batch_shape, self.out_features)

        # Step 3: y₀ = z @ A via fast transform
        # v @ A = apply(v, A, transpose=False)
        y_sparse = apply_fast_orthogonal(z, self.transform_A, transpose=False, dim=-1)

        # Low-rank correction: x @ V @ U^T
        if self.r > 0 and self.U.shape[1] > 0:
            y_lr = (x_f @ self.V.float()) @ self.U.float().T
        else:
            y_lr = 0.0

        out = y_sparse + y_lr

        if self.bias is not None:
            out = out + self.bias

        return out.to(dtype_in)

    def reconstruction_error(self, linear: nn.Linear) -> float:
        """Compute ||W - Ŵ||_F / ||W||_F against the original dense layer."""
        W = linear.weight.data.float()
        W_hat = self._reconstruct_weight()
        return (W - W_hat).norm().item() / max(W.norm().item(), 1e-10)

    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the full approximate weight matrix Ŵ = A^T W̃_K B + UV^T.

        Uses fast transforms — no dense d×d matrix materialization.
        """
        m, n = self.out_features, self.in_features

        # Build dense W̃_K from sparse representation (small: only m×n for the weight)
        W_tilde_K = torch.zeros(m, n, device=self.sparse_values.device,
                                dtype=torch.float32)
        if self.K > 0:
            rows = self.sparse_indices[:self.K, 0]
            cols = self.sparse_indices[:self.K, 1]
            W_tilde_K[rows, cols] = self.sparse_values[:self.K].float()

        # Reconstruct via fast transforms: A^T @ W̃_K @ B
        W_sparse = reconstruct_from_sparse(W_tilde_K, self.transform_A, self.transform_B)

        if self.r > 0 and self.U.shape[1] > 0:
            W_lr = self.U.float() @ self.V.float().T
        else:
            W_lr = 0.0

        return W_sparse + W_lr

    def compression_info(self) -> dict:
        """Report compression statistics (realistic: includes index storage + descriptor overhead)."""
        m, n = self.out_features, self.in_features
        dense = m * n
        # Descriptor overhead: perms + signs for both transforms
        descriptor_overhead = 2 * (m + n)  # pre/post perm+signs for A and B
        # 3K: 1 value + 2 int32 indices (= 2 float32 equiv) per sparse entry
        compressed = self.K * 3 + self.r * (m + n) + descriptor_overhead
        if self.bias is not None:
            dense += m
            compressed += m

        return {
            "out_features": m,
            "in_features": n,
            "K": self.K,
            "r": self.r,
            "dense_params": dense,
            "zebra_params": compressed,
            "descriptor_overhead": descriptor_overhead,
            "compression_ratio": dense / max(compressed, 1),
            "params_ratio": compressed / max(dense, 1),  # >1 = expansion
        }

    def extra_repr(self) -> str:
        return (
            f"out={self.out_features}, in={self.in_features}, "
            f"K={self.K}, r={self.r}, bias={self.bias is not None}"
        )
