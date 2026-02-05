"""
CIMBA Moment Sketching Module

Implements the core sketching operation:
    M_k = Σ_i exp(i·k·θ_i) · V_i

where θ_i = 2π·i/L maps token positions to phases on the unit circle.
"""

import torch
import torch.nn as nn
from typing import Tuple
import math


class CIMBASketch(nn.Module):
    """
    Computes complex Fourier moments of the Value vectors.
    
    Given V of shape [batch, heads, seq_len, head_dim], produces moments
    M of shape [batch, heads, 2S+1, head_dim] in complex64.
    
    The moment index k ranges from -S to +S (inclusive), giving 2S+1 moments.
    """
    
    def __init__(self, S: int, device: torch.device = None):
        """
        Args:
            S: Maximum harmonic degree. Will compute moments for k ∈ [-S, S].
            device: Target device for computations.
        """
        super().__init__()
        self.S = S
        self.num_moments = 2 * S + 1
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute harmonic indices: [-S, -S+1, ..., -1, 0, 1, ..., S-1, S]
        self.register_buffer(
            "k_values",
            torch.arange(-S, S + 1, dtype=torch.float32, device=self.device)
        )
    
    def compute_phases(self, seq_len: int) -> torch.Tensor:
        """
        Map token positions to phases on [0, 2π).
        
        Args:
            seq_len: Number of tokens in sequence.
            
        Returns:
            phases: Tensor of shape [seq_len] with θ_i = 2π·i/L
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        phases = 2 * math.pi * positions / seq_len
        return phases
    
    def forward(self, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CIMBA moments from Value vectors.
        
        Args:
            V: Value tensor of shape [batch, heads, seq_len, head_dim]
            
        Returns:
            moments: Complex tensor of shape [batch, heads, 2S+1, head_dim]
            phases: Real tensor of shape [seq_len] (for use in recovery)
        """
        batch, heads, seq_len, head_dim = V.shape
        
        # Compute phases for each position
        phases = self.compute_phases(seq_len)  # [seq_len]
        
        # Compute k * θ for all combinations
        # k_values: [2S+1], phases: [seq_len]
        # k_theta: [2S+1, seq_len]
        k_theta = self.k_values.unsqueeze(1) * phases.unsqueeze(0)
        
        # Complex exponentials: exp(i·k·θ)
        # Shape: [2S+1, seq_len]
        exp_ik_theta = torch.complex(
            torch.cos(k_theta),
            torch.sin(k_theta)
        )
        
        # Convert V to complex for multiplication
        # V: [batch, heads, seq_len, head_dim] -> complex
        V_complex = V.to(torch.complex64)
        
        # Compute moments: M_k = Σ_i exp(i·k·θ_i) · V_i
        # exp_ik_theta: [2S+1, seq_len]
        # V_complex: [batch, heads, seq_len, head_dim]
        # Result: [batch, heads, 2S+1, head_dim]
        
        # Reshape for batch matrix multiply:
        # exp_ik_theta: [1, 1, 2S+1, seq_len]
        # V_complex: [batch, heads, seq_len, head_dim]
        exp_ik_theta = exp_ik_theta.unsqueeze(0).unsqueeze(0)
        
        # M_k = sum over seq_len: exp(ik·θ) * V
        # [batch, heads, 2S+1, seq_len] @ [batch, heads, seq_len, head_dim]
        # -> [batch, heads, 2S+1, head_dim]
        moments = torch.einsum("bhks,bhsd->bhkd", exp_ik_theta.expand(batch, heads, -1, -1), V_complex)
        
        return moments, phases
    
    def get_memory_size(self, batch: int, heads: int, head_dim: int) -> int:
        """
        Calculate memory footprint of the sketch in bytes.
        
        Complex64 = 8 bytes per element.
        """
        num_elements = batch * heads * self.num_moments * head_dim
        return num_elements * 8  # complex64 = 8 bytes
    
    @staticmethod
    def compare_to_full_kv(seq_len: int, batch: int, heads: int, head_dim: int, S: int) -> dict:
        """
        Compare CIMBA sketch memory to full KV cache.
        
        Returns:
            dict with 'full_kv_bytes', 'cimba_bytes', 'compression_ratio'
        """
        # Full KV cache: K and V, both [batch, heads, seq_len, head_dim] in float16
        full_kv_bytes = 2 * batch * heads * seq_len * head_dim * 2  # 2 for K+V, 2 for float16
        
        # CIMBA: moments only, [batch, heads, 2S+1, head_dim] in complex64
        cimba_bytes = batch * heads * (2 * S + 1) * head_dim * 8
        
        return {
            "full_kv_bytes": full_kv_bytes,
            "cimba_bytes": cimba_bytes,
            "compression_ratio": full_kv_bytes / cimba_bytes if cimba_bytes > 0 else float("inf"),
        }
