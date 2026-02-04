"""
CIMBA Recovery Module with Learnable Coefficients

Implements the recovery operation using learned Fourier coefficients ĝ_k:
    Y_cimba[t] = Re(Σ_k ĝ_k · exp(-i·k·θ_t) · M_k)

The coefficients ĝ_k are parameterized as real [num_heads, 2S+1, 2] tensors
and converted to complex for computation (numerical stability).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class CIMBARecovery(nn.Module):
    """
    Learnable recovery module for reconstructing attention output from CIMBA moments.
    
    Each attention head has its own set of 2S+1 complex coefficients, respecting
    the MAST principle that each head may have a unique "Motif Algebra."
    """
    
    def __init__(
        self,
        S: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = None,
        init_scale: float = 0.01,
    ):
        """
        Args:
            S: Maximum harmonic degree (moments for k ∈ [-S, S]).
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            device: Target device.
            init_scale: Scale for random initialization of coefficients.
        """
        super().__init__()
        self.S = S
        self.num_moments = 2 * S + 1
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learnable coefficients: parameterized as real [num_heads, 2S+1, 2]
        # The last dimension is [real_part, imag_part]
        # This avoids PyTorch complex autograd instabilities
        self.g_coeffs_real = nn.Parameter(
            torch.randn(num_heads, self.num_moments, 2, device=self.device) * init_scale
        )
        
        # Precompute harmonic indices
        self.register_buffer(
            "k_values",
            torch.arange(-S, S + 1, dtype=torch.float32, device=self.device)
        )
    
    @property
    def g_coeffs(self) -> torch.Tensor:
        """Convert real parameterization to complex coefficients."""
        return torch.view_as_complex(self.g_coeffs_real.contiguous())
    
    def forward(
        self,
        moments: torch.Tensor,
        phases: torch.Tensor,
        return_attention_matrix: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Recover attention output (and optionally attention matrix) from moments.
        
        Args:
            moments: Complex tensor [batch, heads, 2S+1, head_dim] from CIMBASketch
            phases: Real tensor [seq_len] of position phases
            return_attention_matrix: If True, also reconstruct A_cimba
            
        Returns:
            Y_cimba: Real tensor [batch, heads, seq_len, head_dim]
            A_cimba: (optional) Real tensor [batch, heads, seq_len, seq_len]
        """
        batch, heads, num_k, head_dim = moments.shape
        seq_len = phases.shape[0]
        
        assert num_k == self.num_moments, f"Moment count mismatch: {num_k} vs {self.num_moments}"
        assert heads == self.num_heads, f"Head count mismatch: {heads} vs {self.num_heads}"
        
        # Get complex coefficients: [heads, 2S+1]
        g = self.g_coeffs  # [heads, 2S+1]
        
        # Compute -k * θ_t for recovery
        # k_values: [2S+1], phases: [seq_len]
        # neg_k_theta: [2S+1, seq_len]
        neg_k_theta = -self.k_values.unsqueeze(1) * phases.unsqueeze(0)
        
        # exp(-i·k·θ_t): [2S+1, seq_len]
        exp_neg_ik_theta = torch.complex(
            torch.cos(neg_k_theta),
            torch.sin(neg_k_theta)
        )
        
        # Weighted by coefficients: ĝ_k · exp(-i·k·θ_t)
        # g: [heads, 2S+1] -> [heads, 2S+1, 1]
        # exp_neg_ik_theta: [2S+1, seq_len] -> [1, 2S+1, seq_len]
        # Result: [heads, 2S+1, seq_len]
        weighted_exp = g.unsqueeze(-1) * exp_neg_ik_theta.unsqueeze(0)
        
        # Sum over k, then multiply by moments
        # weighted_exp: [heads, 2S+1, seq_len]
        # moments: [batch, heads, 2S+1, head_dim]
        # 
        # For each query position t:
        #   Y[t] = Σ_k ĝ_k · exp(-ikθ_t) · M_k
        #        = Σ_k weighted_exp[k,t] · M_k
        #
        # Result: [batch, heads, seq_len, head_dim]
        
        # Reshape for einsum:
        # weighted_exp: [heads, 2S+1, seq_len] -> [1, heads, 2S+1, seq_len]
        weighted_exp = weighted_exp.unsqueeze(0)
        
        # Y_cimba = Σ_k weighted_exp[k,t] · M_k
        # [batch, heads, seq_len, head_dim]
        Y_cimba_complex = torch.einsum(
            "bhkt,bhkd->bhtd",
            weighted_exp.expand(batch, -1, -1, -1),
            moments
        )
        
        # Take real part (imaginary should be ~0 for valid reconstruction)
        Y_cimba = Y_cimba_complex.real
        
        # Optionally reconstruct the attention matrix
        A_cimba = None
        if return_attention_matrix:
            A_cimba = self._reconstruct_attention_matrix(phases, batch)
        
        return Y_cimba, A_cimba
    
    def _reconstruct_attention_matrix(
        self,
        phases: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """
        Reconstruct the attention matrix from learned coefficients.
        
        The attention kernel is approximated as:
            A[t, i] ≈ Re(Σ_k ĝ_k · exp(i·k·(θ_i - θ_t)))
        
        This follows from the CIMBA kernel interpretation where the
        attention weights form a bandlimited trigonometric polynomial.
        
        Args:
            phases: [seq_len] position phases
            batch: batch size for output shape
            
        Returns:
            A_cimba: [batch, heads, seq_len, seq_len]
        """
        seq_len = phases.shape[0]
        
        # Compute phase differences: θ_i - θ_t
        # phases: [seq_len]
        # phase_diff[t, i] = θ_i - θ_t
        phase_diff = phases.unsqueeze(0) - phases.unsqueeze(1)  # [seq_len, seq_len]
        
        # k * (θ_i - θ_t) for all k
        # k_values: [2S+1]
        # k_phase_diff: [2S+1, seq_len, seq_len]
        k_phase_diff = self.k_values.view(-1, 1, 1) * phase_diff.unsqueeze(0)
        
        # exp(i·k·(θ_i - θ_t)): [2S+1, seq_len, seq_len]
        exp_ik_diff = torch.complex(
            torch.cos(k_phase_diff),
            torch.sin(k_phase_diff)
        )
        
        # Weight by coefficients: ĝ_k · exp(i·k·Δθ)
        # g: [heads, 2S+1] -> [heads, 2S+1, 1, 1]
        g = self.g_coeffs.view(self.num_heads, self.num_moments, 1, 1)
        
        # exp_ik_diff: [2S+1, seq_len, seq_len] -> [1, 2S+1, seq_len, seq_len]
        exp_ik_diff = exp_ik_diff.unsqueeze(0)
        
        # Weighted sum over k: [heads, seq_len, seq_len]
        A_complex = (g * exp_ik_diff).sum(dim=1)
        
        # Take real part and expand batch dimension
        A_cimba = A_complex.real.unsqueeze(0).expand(batch, -1, -1, -1)
        
        return A_cimba
    
    def get_coefficient_stats(self) -> dict:
        """Return statistics about learned coefficients for diagnostics."""
        g = self.g_coeffs  # [heads, 2S+1]
        magnitudes = g.abs()
        
        return {
            "mean_magnitude": magnitudes.mean().item(),
            "max_magnitude": magnitudes.max().item(),
            "min_magnitude": magnitudes.min().item(),
            "magnitude_std": magnitudes.std().item(),
            "dc_component_mean": magnitudes[:, self.S].mean().item(),  # k=0 component
        }
