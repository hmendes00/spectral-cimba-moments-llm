"""
IWM Content-Kernel Attention

Implements attention with content-aware but CIMBA-sketchable weights.

Key properties:
1. Attention weights W_t(j) are nonnegative and bandlimited to |k|≤S
2. Computed from O(S) cached V-moments (infinite context, constant memory)
3. Content-dependent via query-predicted Fourier coefficients

Kernel form (Fejér-smoothed):
    p_t(Δθ) = Σ_{k=-S..S} w_k * g_k(t) * e^{ikΔθ}
where:
    - g_k(t) are content-predicted coefficients from Q_t
    - w_k = 1 - |k|/(S+1) are Fejér weights (ensure nonnegativity)
    - Δθ = θ_j - θ_t is relative position angle

Output via moments:
    y_t = (Σ_k g_k(t) e^{-ikθ_t} M_k) / (Σ_k g_k(t) e^{-ikθ_t} Z_k)
where:
    M_k = Σ_{j≤t} e^{ikθ_j} V_j  (V moments)
    Z_k = Σ_{j≤t} e^{ikθ_j}      (normalization moments)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class IWMCache:
    """O(S) cache for IWM attention - stores moments, not full KV."""
    v_moments: torch.Tensor      # [num_layers, num_kv_heads, 2S+1, head_dim] complex
    norm_moments: torch.Tensor   # [num_layers, num_kv_heads, 2S+1] complex
    seen_tokens: int
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen_tokens


class IWMContentKernelAttention(nn.Module):
    """
    IWM-style attention with content-aware bandlimited kernel.
    
    Replaces standard attention with O(S) memory via moment caching.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        layer_idx: int = 0,
        bandwidth: int = 64,  # S: maximum harmonic
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.attention_dropout = attention_dropout
        self.layer_idx = layer_idx
        self.is_causal = True
        self.max_seq_len = max_position_embeddings
        
        # Bandwidth
        self.S = bandwidth
        self.num_moments = 2 * bandwidth + 1  # k ∈ [-S, S]
        
        # GQA
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Standard projections (copied from base model)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE (copied from base)
        self.rotary_emb = None
        
        # Content -> coefficient projection (the NEW trainable part)
        # Maps Q_t (per head) to S+1 real coefficients c_0..c_S
        # Shared across KV heads within a group
        self.coeff_proj = nn.Linear(self.head_dim, bandwidth + 1, bias=True)
        
        # Initialize: c0 ≈ 1 (dominant), others small but non-zero
        # This avoids the degenerate uniform kernel where gradients vanish
        with torch.no_grad():
            # Small random weights so different queries get different kernels
            self.coeff_proj.weight.normal_(0, 0.01)
            # Bias: c0=1, others=0.05 (small but non-zero)
            self.coeff_proj.bias.zero_()
            self.coeff_proj.bias[0] = 1.0
            self.coeff_proj.bias[1:] = 0.05  # Small non-zero initialization
        
        # Precompute Fejér weights: w_k = 1 - |k|/(S+1)
        k_vals = torch.arange(-bandwidth, bandwidth + 1, dtype=torch.float32)
        fejer_weights = 1.0 - torch.abs(k_vals) / (bandwidth + 1)
        self.register_buffer('fejer_weights', fejer_weights)
        self.register_buffer('k_vals', k_vals)
        
        # Position angle scale
        self.theta_scale = 2 * math.pi / max_position_embeddings
        
        # Training mode flag: if True, only compute output for last valid token
        # This dramatically reduces memory from O(seq × 2S+1 × head_dim) to O(2S+1 × head_dim)
        self.training_last_token_only = False
        
        # Diagnostics (updated each forward, for monitoring)
        self._diagnostics = {
            'denom_min': 0.0,
            'denom_mean': 0.0,
            'denom_max': 0.0,
            'kernel_negative_pct': 0.0,
            'moment_v_norm': 0.0,
            'moment_z_norm': 0.0,
            # Coefficient stats (more informative than PSD loss)
            'c0_min': 0.0,
            'c0_mean': 0.0,
            'c0_max': 0.0,
            'ck_ratio_mean': 0.0,  # mean of |c_k|/|c0| for k>0
            'ck_ratio_max': 0.0,   # max of |c_k|/|c0| for k>0
        }
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Apply rotary position embeddings."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_pos_emb_q_only(self, q, cos, sin):
        """
        Apply rotary position embeddings to query only.
        
        This is critical for IWM: the pretrained Q projection expects RoPE-encoded
        positions. The coefficient predictor then learns from position-aware queries.
        
        Args:
            q: [batch, heads, seq_len, head_dim]
            cos: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
            sin: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        """
        # Unsqueeze for head dimension: [batch, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        return q_embed
    
    def _predict_coefficients(self, query: torch.Tensor) -> torch.Tensor:
        """
        Predict Fejér kernel coefficients from query.
        
        Args:
            query: [batch, num_heads, seq_len, head_dim]
            
        Returns:
            coeffs: [batch, num_heads, seq_len, S+1] real coefficients c_0..c_S
        """
        # query: [batch, heads, seq, head_dim]
        coeffs = self.coeff_proj(query)  # [batch, heads, seq, S+1]
        return coeffs
    
    def _build_kernel_coeffs(self, c_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Build two-sided Fourier coefficients g_k for k=-S..S from c_0..c_S.
        
        Uses even (cosine) kernel: g_k = g_{-k}
        Applies Fejér smoothing for nonnegativity.
        
        Args:
            c_coeffs: [batch, heads, seq, S+1] real coefficients
            
        Returns:
            g_coeffs: [batch, heads, seq, 2S+1] complex coefficients for k=-S..S
        """
        batch, heads, seq, S_plus_1 = c_coeffs.shape
        S = S_plus_1 - 1
        device = c_coeffs.device
        
        # Build two-sided: g_0 = c_0, g_k = g_{-k} = 0.5*c_k for k>0
        # Index mapping: k=-S..S -> idx 0..2S
        # k=0 -> idx=S, k=+1 -> idx=S+1, k=-1 -> idx=S-1, etc.
        
        g_coeffs = torch.zeros(batch, heads, seq, 2*S + 1, device=device, dtype=c_coeffs.dtype)
        
        # c_0 -> g_0 (center)
        g_coeffs[..., S] = c_coeffs[..., 0]
        
        # c_k -> g_k and g_{-k} for k=1..S (loop is fine, runs on GPU slices)
        for k in range(1, S + 1):
            half_ck = 0.5 * c_coeffs[..., k]
            g_coeffs[..., S + k] = half_ck  # g_k
            g_coeffs[..., S - k] = half_ck  # g_{-k}
        
        # Apply Fejér weights for nonnegativity
        g_coeffs = g_coeffs * self.fejer_weights.view(1, 1, 1, -1)
        
        # Convert to complex (real coefficients, zero imaginary)
        # Note: torch.complex doesn't support bfloat16, so convert to float32
        g_coeffs_f32 = g_coeffs.float()
        g_coeffs = torch.complex(g_coeffs_f32, torch.zeros_like(g_coeffs_f32))
        
        return g_coeffs
    
    def _compute_kernel_negativity(self, c_coeffs: torch.Tensor, num_grid: int = 128) -> float:
        """
        Compute percentage of grid points where kernel is negative.
        
        This is a diagnostic - expensive, so call sparingly.
        """
        with torch.no_grad():
            g_coeffs = self._build_kernel_coeffs(c_coeffs)
            
            # Evaluate on grid
            theta_grid = torch.linspace(0, 2*math.pi, num_grid, device=g_coeffs.device)
            phase = torch.exp(1j * self.k_vals.unsqueeze(0) * theta_grid.unsqueeze(1))  # [grid, 2S+1]
            
            # g_coeffs: [batch, heads, seq, 2S+1]
            # For efficiency, sample a few positions
            sample_g = g_coeffs[:, :, :min(8, g_coeffs.shape[2]), :]  # [batch, heads, ≤8, 2S+1]
            
            # kernel[b,h,t,θ] = Σ_k g_k * e^{ikθ}
            kernel = torch.einsum('bhsk,gk->bhsg', sample_g, phase)  # [batch, heads, seq, grid]
            kernel_real = kernel.real
            
            negative_pct = (kernel_real < -1e-6).float().mean().item() * 100
            return negative_pct
    
    def _compute_psd_loss(self, c_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft PSD constraint loss for Toeplitz realizability.
        
        For a trigonometric polynomial to be nonnegative, its Toeplitz matrix
        of coefficients must be positive semidefinite.
        
        Note: This is a heuristic regularizer, not exact PSD enforcement.
        
        Args:
            c_coeffs: [batch, heads, seq, S+1]
            
        Returns:
            loss: scalar penalty for negative eigenvalues
        """
        batch, heads, seq, S_plus_1 = c_coeffs.shape
        
        # c_0 should be positive and dominant
        c0_penalty = F.relu(-c_coeffs[..., 0]).mean()
        
        # Higher harmonics should be bounded relative to c_0
        c0_safe = c_coeffs[..., 0:1].abs() + 1e-6
        ratio = c_coeffs[..., 1:].abs() / c0_safe
        ratio_penalty = F.relu(ratio - 0.5).mean()  # Penalize if |c_k| > 0.5*|c_0|
        
        return c0_penalty + 0.1 * ratio_penalty
    
    def _compute_diversity_loss(self, c_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that ENCOURAGES using higher harmonics while keeping
        coefficients bounded.
        
        Two components:
        1. Diversity: push c_rest away from 0 (target ~0.05)
        2. Stability: penalize c0 deviating too far from 1
        
        Args:
            c_coeffs: [..., S+1] real coefficients
            
        Returns:
            loss: scalar
        """
        c0 = c_coeffs[..., 0]
        c_rest = c_coeffs[..., 1:]
        
        # Diversity: encourage c_rest to be small but non-zero
        target = 0.05 * torch.ones_like(c_rest)  # Small target
        diversity_loss = ((c_rest - target) ** 2).mean()
        
        # Stability: keep c0 close to 1 (positive, dominant)
        c0_target = 1.0
        stability_loss = ((c0 - c0_target) ** 2).mean()
        
        # Combined: diversity + strong stability
        return diversity_loss + 0.5 * stability_loss
    
    def _track_coefficient_stats(self, c_coeffs: torch.Tensor) -> None:
        """
        Track coefficient statistics for monitoring learning progress.
        
        More informative than PSD loss - shows if coefficients are actually moving.
        
        Args:
            c_coeffs: [..., S+1] real coefficients (any leading dims)
        """
        with torch.no_grad():
            c0 = c_coeffs[..., 0]
            c_rest = c_coeffs[..., 1:]
            
            self._diagnostics['c0_min'] = c0.min().detach()
            self._diagnostics['c0_mean'] = c0.mean().detach()
            self._diagnostics['c0_max'] = c0.max().detach()
            
            # Ratio of higher harmonics to c0
            c0_safe = c0.abs().unsqueeze(-1) + 1e-6
            ratio = c_rest.abs() / c0_safe
            
            self._diagnostics['ck_ratio_mean'] = ratio.mean().detach()
            self._diagnostics['ck_ratio_max'] = ratio.max().detach()
    
    def _compute_prefix_moments(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal prefix moments for all positions (for training).
        
        M_k(t) = Σ_{j<=t} e^{ikθ_j} V_j  (masked by attention_mask)
        Z_k(t) = Σ_{j<=t} e^{ikθ_j}      (masked by attention_mask)
        
        Uses cumsum to maintain causality: position t only sees j <= t.
        
        Args:
            value: [batch, kv_heads, seq_len, head_dim]
            positions: [seq_len]
            attention_mask: [batch, seq_len], 1=valid, 0=pad
            
        Returns:
            v_moments: [batch, kv_heads, seq_len, 2S+1, head_dim] complex
            norm_moments: [batch, kv_heads, seq_len, 2S+1] complex
        """
        batch, kv_heads, seq_len, head_dim = value.shape
        device = value.device
        
        # Position angles
        theta = positions.float() * self.theta_scale  # [seq_len]
        
        # Phase matrix: e^{ikθ_j} for k=-S..S
        # Result: [seq_len, 2S+1]
        phase = torch.exp(1j * self.k_vals.unsqueeze(0) * theta.unsqueeze(1))
        
        # Convert V to complex
        v_complex = value.to(torch.complex64)  # [batch, kv_heads, seq_len, head_dim]
        
        # Apply attention mask to V if provided (zero out padding)
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, seq_len, 1]
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(v_complex.dtype)
            v_complex = v_complex * mask
        
        # Compute per-position contribution: v_contrib[b,h,j,k,d] = phase[j,k] * V[b,h,j,d]
        # v_complex: [batch, kv_heads, seq_len, head_dim]
        # phase: [seq_len, 2S+1]
        # Result: [batch, kv_heads, seq_len, 2S+1, head_dim]
        v_contrib = v_complex.unsqueeze(3) * phase.view(1, 1, seq_len, -1, 1)
        
        # Cumsum for causal prefix moments: M_k(t) = Σ_{j<=t} v_contrib[j,k]
        v_moments = v_contrib.cumsum(dim=2)
        
        # Z moments: phase contribution with mask
        # Must have shape [batch, kv_heads, seq_len, 2S+1] for proper broadcasting
        if attention_mask is not None:
            # phase_mask: [batch, 1, seq_len, 1]
            phase_mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(phase.dtype)
            z_contrib = phase.view(1, 1, seq_len, -1) * phase_mask
            # Expand to full batch and kv_heads dimensions
            z_contrib = z_contrib.expand(batch, kv_heads, -1, -1).contiguous()
        else:
            # Expand phase to [batch, kv_heads, seq_len, 2S+1]
            z_contrib = phase.view(1, 1, seq_len, -1).expand(batch, kv_heads, -1, -1).contiguous()
        
        # Cumsum for causal: Z_k(t) = Σ_{j<=t} e^{ikθ_j}
        norm_moments = z_contrib.cumsum(dim=2)  # [batch, kv_heads, seq_len, 2S+1]
        
        return v_moments, norm_moments
    
    def _reconstruct_from_prefix_moments(
        self,
        v_moments: torch.Tensor,
        norm_moments: torch.Tensor,
        g_coeffs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct attention output from prefix moments (for training).
        
        y_t = (Σ_k g_k(t) e^{-ikθ_t} M_k(t)) / (Σ_k g_k(t) e^{-ikθ_t} Z_k(t))
        
        Args:
            v_moments: [batch, kv_heads, seq_len, 2S+1, head_dim] complex
            norm_moments: [batch, kv_heads, seq_len, 2S+1] complex
            g_coeffs: [batch, heads, seq_len, 2S+1] complex
            positions: [seq_len]
            
        Returns:
            output: [batch, heads, seq_len, head_dim]
        """
        batch, heads, seq_len, num_k = g_coeffs.shape
        device = g_coeffs.device
        
        # Query phases: e^{-ikθ_t} for each position
        theta_q = positions.float() * self.theta_scale
        query_phase = torch.exp(-1j * self.k_vals.unsqueeze(0) * theta_q.unsqueeze(1))  # [seq_len, 2S+1]
        
        # Expand moments for GQA
        if self.num_kv_heads != self.num_heads:
            v_moments = v_moments.repeat_interleave(self.num_key_value_groups, dim=1)
            norm_moments = norm_moments.repeat_interleave(self.num_key_value_groups, dim=1)
        # Now: v_moments [batch, heads, seq_len, 2S+1, head_dim]
        #      norm_moments [batch, heads, seq_len, 2S+1]
        
        # Combined weight: g_k(t) * e^{-ikθ_t}
        # g_coeffs: [batch, heads, seq_len, 2S+1]
        # query_phase: [seq_len, 2S+1]
        combined = g_coeffs * query_phase.unsqueeze(0).unsqueeze(0)  # [batch, heads, seq_len, 2S+1]
        
        # Numerator: Σ_k combined[b,h,t,k] * M_k(t)[b,h,t,k,d]
        numerator = (combined.unsqueeze(-1) * v_moments).sum(dim=3)  # [batch, heads, seq_len, head_dim]
        
        # Denominator: Σ_k combined[b,h,t,k] * Z_k(t)[b,h,t,k]
        denominator = (combined * norm_moments).sum(dim=3)  # [batch, heads, seq_len]
        
        # Track diagnostics (store tensors detached, convert to scalars only when needed)
        # Detach to avoid keeping computation graph; avoids GPU-CPU sync on every forward
        denom_abs = denominator.abs().detach()
        self._diagnostics['denom_min'] = denom_abs.min()
        self._diagnostics['denom_mean'] = denom_abs.mean()
        self._diagnostics['denom_max'] = denom_abs.max()
        self._diagnostics['moment_v_norm'] = v_moments.abs().mean().detach()
        self._diagnostics['moment_z_norm'] = norm_moments.abs().mean().detach()
        
        # Output (keep as float32 from complex, convert back in forward_with_cache)
        output = numerator / (denominator.unsqueeze(-1) + 1e-10)
        
        return output.real
    
    def _compute_last_prefix_moments(
        self,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prefix moments ONLY at the last valid position (O(1) in seq memory).
        
        Instead of building [batch, heads, seq, 2S+1, head_dim] and doing cumsum,
        we directly compute the sum over all valid positions:
            V_last = Σ_{t: mask[t]=1} e^{ikθ_t} * v_t
            Z_last = Σ_{t: mask[t]=1} e^{ikθ_t}
        
        This is O(batch × heads × (2S+1) × head_dim) memory, not O(seq × ...).
        
        Args:
            value: [batch, kv_heads, seq_len, head_dim]
            attention_mask: [batch, seq_len], 1=valid, 0=pad
            
        Returns:
            V_last: [batch, kv_heads, 2S+1, head_dim] complex
            Z_last: [batch, kv_heads, 2S+1] complex
        """
        batch, kv_heads, seq_len, head_dim = value.shape
        device = value.device
        
        # Position angles for all positions
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta = positions * self.theta_scale  # [seq_len]
        
        # Phase matrix: e^{ikθ_t} for k=-S..S and t=0..seq-1
        # k_vals: [2S+1], theta: [seq_len]
        # Result: [seq_len, 2S+1] complex
        phase = torch.exp(1j * theta.unsqueeze(1) * self.k_vals.unsqueeze(0))
        
        # Convert value to complex
        v_complex = value.to(torch.complex64)  # [batch, kv_heads, seq_len, head_dim]
        
        # Apply attention mask (zero out padding)
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, seq_len, 1]
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(v_complex.dtype)
            v_masked = v_complex * mask
            # For Z moments, we need mask as [batch, seq_len]
            mask_for_z = attention_mask.to(phase.dtype)  # [batch, seq_len]
        else:
            v_masked = v_complex
            mask_for_z = torch.ones(batch, seq_len, device=device, dtype=phase.dtype)
        
        # V_last = Σ_t phase[t,k] * v[t]
        # einsum: 'tk,bhte->bhke' where t=seq, k=2S+1, b=batch, h=heads, e=head_dim
        V_last = torch.einsum('tk,bhte->bhke', phase, v_masked)
        # V_last: [batch, kv_heads, 2S+1, head_dim] complex
        
        # Z_last = Σ_t phase[t,k] * mask[t]
        # einsum: 'tk,bt->bk' then expand to [batch, kv_heads, 2S+1]
        Z_last = torch.einsum('tk,bt->bk', phase, mask_for_z)
        # Expand to [batch, kv_heads, 2S+1]
        Z_last = Z_last.unsqueeze(1).expand(-1, kv_heads, -1).contiguous()
        
        return V_last, Z_last
    
    def forward_train_last_token(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Last-token teacher forcing: compute attention output ONLY for last valid token.
        
        Uses O(1)-in-seq memory by computing moments directly without cumsum.
        Memory: O(batch × heads × (2S+1) × head_dim) - no seq dimension!
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple (unused, kept for API compatibility)
            attention_mask: [batch, seq_len], 1=valid, 0=pad
            
        Returns:
            output: [batch, seq_len, hidden_size] - zeros everywhere except last_idx
            last_indices: [batch] - indices of last valid tokens
            psd_loss: scalar
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        original_dtype = hidden_states.dtype
        
        # Project Q, V
        query = self.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        value = self.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
        
        # Reshape
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # query: [batch, num_heads, seq_len, head_dim]
        # value: [batch, num_kv_heads, seq_len, head_dim]
        
        # Apply RoPE to query (critical for pretrained weights!)
        cos, sin = position_embeddings
        query = self._apply_rotary_pos_emb_q_only(query, cos, sin)
        
        # Find last valid index per sample
        if attention_mask is not None:
            # attention_mask: [batch, seq_len], 1=valid, 0=pad
            positions_range = torch.arange(seq_len, device=device).unsqueeze(0)
            masked_positions = positions_range * attention_mask.long() - (1 - attention_mask.long()) * seq_len
            last_indices = masked_positions.argmax(dim=1)  # [batch]
        else:
            last_indices = torch.full((batch,), seq_len - 1, device=device, dtype=torch.long)
        
        # Compute moments at last position directly (O(1) in seq memory!)
        # This avoids building the [batch, heads, seq, 2S+1, head_dim] tensor
        M_k_last, Z_k_last = self._compute_last_prefix_moments(value, attention_mask)
        # M_k_last: [batch, kv_heads, 2S+1, head_dim]
        # Z_k_last: [batch, kv_heads, 2S+1]
        
        # Get query at last position
        batch_idx = torch.arange(batch, device=device)
        q_last = query[batch_idx, :, last_indices, :]  # [batch, num_heads, head_dim]
        
        # Predict kernel coefficients
        c_coeffs_last = self.coeff_proj(q_last)  # [batch, num_heads, S+1]
        g_coeffs_last = self._build_kernel_coeffs_unbatched(c_coeffs_last)
        
        # Track coefficient stats (more useful than PSD loss)
        self._track_coefficient_stats(c_coeffs_last)
        
        # PSD loss and diversity loss
        c_coeffs_for_loss = c_coeffs_last.unsqueeze(2)
        psd_loss = self._compute_psd_loss(c_coeffs_for_loss)
        diversity_loss = self._compute_diversity_loss(c_coeffs_for_loss)
        self._last_diversity_loss = diversity_loss  # Store for retrieval
        
        # Query phase at last position
        theta_last = last_indices.float() * self.theta_scale
        query_phase_last = torch.exp(-1j * self.k_vals.unsqueeze(0) * theta_last.unsqueeze(1))
        
        # Expand for GQA
        if self.num_kv_heads != self.num_heads:
            M_k_expanded = M_k_last.repeat_interleave(self.num_key_value_groups, dim=1)
            Z_k_expanded = Z_k_last.repeat_interleave(self.num_key_value_groups, dim=1)
        else:
            M_k_expanded = M_k_last
            Z_k_expanded = Z_k_last
        
        # Combined weight: g_k(t) * e^{-ikθ_t}
        combined = g_coeffs_last * query_phase_last.unsqueeze(1)
        
        # Numerator and denominator
        numerator = (combined.unsqueeze(-1) * M_k_expanded).sum(dim=2)
        denominator = (combined * Z_k_expanded).sum(dim=2)
        
        # Track diagnostics
        denom_abs = denominator.abs().detach()
        self._diagnostics['denom_min'] = denom_abs.min()
        self._diagnostics['denom_mean'] = denom_abs.mean()
        self._diagnostics['denom_max'] = denom_abs.max()
        self._diagnostics['moment_v_norm'] = M_k_last.abs().mean().detach()
        self._diagnostics['moment_z_norm'] = Z_k_last.abs().mean().detach()
        
        # Output
        attn_output_last = numerator / (denominator.unsqueeze(-1) + 1e-10)
        attn_output_last = attn_output_last.real.to(original_dtype)
        
        # Project and reshape
        attn_output_last = attn_output_last.view(batch, self.num_heads * self.head_dim)
        attn_output_last = self.o_proj(attn_output_last)
        attn_output_last = attn_output_last.unsqueeze(1)
        
        # Scatter into full output tensor
        output = torch.zeros(batch, seq_len, attn_output_last.shape[-1],
                            device=device, dtype=original_dtype)
        output.scatter_(1, last_indices.view(batch, 1, 1).expand(-1, -1, output.shape[-1]),
                       attn_output_last)
        
        return output, last_indices, psd_loss
    
    def _build_kernel_coeffs_unbatched(self, c_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Build two-sided Fourier coefficients for last-token path.
        
        Args:
            c_coeffs: [batch, heads, S+1] real coefficients (no seq dim)
            
        Returns:
            g_coeffs: [batch, heads, 2S+1] complex coefficients
        """
        batch, heads, S_plus_1 = c_coeffs.shape
        S = S_plus_1 - 1
        device = c_coeffs.device
        
        g_coeffs = torch.zeros(batch, heads, 2*S + 1, device=device, dtype=c_coeffs.dtype)
        
        # c_0 -> g_0 (center)
        g_coeffs[..., S] = c_coeffs[..., 0]
        
        # c_k -> g_k and g_{-k}
        for k in range(1, S + 1):
            half_ck = 0.5 * c_coeffs[..., k]
            g_coeffs[..., S + k] = half_ck
            g_coeffs[..., S - k] = half_ck
        
        # Apply Fejér weights
        g_coeffs = g_coeffs * self.fejer_weights.view(1, 1, -1)
        
        # Convert to complex
        g_coeffs_f32 = g_coeffs.float()
        g_coeffs = torch.complex(g_coeffs_f32, torch.zeros_like(g_coeffs_f32))
        
        return g_coeffs
    
    def _update_moments_incremental(
        self,
        cache: IWMCache,
        value: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update cache moments incrementally (for generation).
        
        Designed for autoregressive generation where we add one token at a time.
        This is naturally causal since cache only contains past tokens.
        
        Args:
            cache: IWMCache to update
            value: [batch, kv_heads, seq_len, head_dim] (typically seq_len=1)
            positions: [seq_len]
            attention_mask: [batch, seq_len], 1=valid, 0=pad
        """
        batch, kv_heads, seq_len, head_dim = value.shape
        device = value.device
        
        # Generation should be batch=1 (for now)
        assert batch == 1, f"Generation cache expects batch=1, got {batch}"
        
        # Position angles
        theta = positions.float() * self.theta_scale
        
        # Phase: e^{ikθ_j}
        phase = torch.exp(1j * self.k_vals.unsqueeze(1) * theta.unsqueeze(0))  # [2S+1, seq_len]
        
        # Convert V to complex
        v_complex = value.to(torch.complex64)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(v_complex.dtype)  # [batch, 1, seq_len, 1]
            v_complex = v_complex * mask
            phase_mask = attention_mask.to(phase.dtype)  # [batch, seq_len]
        
        # Update V moments: M_k += Σ_j e^{ikθ_j} V_j
        delta_v = torch.einsum('ks,bhsd->bhkd', phase, v_complex)  # [batch, kv_heads, 2S+1, head_dim]
        
        # Update Z moments
        if attention_mask is not None:
            delta_z = torch.einsum('ks,bs->bk', phase, phase_mask)  # [batch, 2S+1]
            delta_z = delta_z.unsqueeze(1).expand(-1, kv_heads, -1)  # [batch, kv_heads, 2S+1]
        else:
            delta_z = phase.sum(dim=1)  # [2S+1]
            delta_z = delta_z.unsqueeze(0).unsqueeze(0).expand(batch, kv_heads, -1)
        
        # Add to cache (batch=0 since batch=1 for generation)
        cache.v_moments[self.layer_idx] = cache.v_moments[self.layer_idx] + delta_v[0]
        cache.norm_moments[self.layer_idx] = cache.norm_moments[self.layer_idx] + delta_z[0]
        cache.seen_tokens += seq_len
    
    def _reconstruct_from_cache(
        self,
        cache: IWMCache,
        g_coeffs: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct attention output from cached moments (for generation).
        
        Args:
            cache: IWMCache with accumulated moments
            g_coeffs: [batch, heads, q_len, 2S+1] complex
            positions: [q_len]
            
        Returns:
            output: [batch, heads, q_len, head_dim]
        """
        batch, heads, q_len, num_k = g_coeffs.shape
        device = g_coeffs.device
        
        # Query angles
        theta_q = positions.float() * self.theta_scale
        
        # Query phases: e^{-ikθ_t}
        query_phase = torch.exp(-1j * self.k_vals.unsqueeze(1) * theta_q.unsqueeze(0))  # [2S+1, q_len]
        
        # Get cached moments for this layer
        v_moments = cache.v_moments[self.layer_idx]  # [kv_heads, 2S+1, head_dim]
        norm_moments = cache.norm_moments[self.layer_idx]  # [kv_heads, 2S+1]
        
        # Expand for GQA
        if self.num_kv_heads != self.num_heads:
            v_moments = v_moments.repeat_interleave(self.num_key_value_groups, dim=0)
            norm_moments = norm_moments.repeat_interleave(self.num_key_value_groups, dim=0)
        
        # Combined weight: g_k(t) * e^{-ikθ_t}
        combined = g_coeffs * query_phase.T.unsqueeze(0).unsqueeze(0)  # [batch, heads, q_len, 2S+1]
        
        # Numerator: Σ_k combined_k * M_k
        numerator = torch.einsum('bhqk,hkd->bhqd', combined, v_moments)
        
        # Denominator: Σ_k combined_k * Z_k
        denominator = torch.einsum('bhqk,hk->bhq', combined, norm_moments)
        
        # Output (keep as float32 from complex, convert back in forward_with_cache)
        output = numerator / (denominator.unsqueeze(-1) + 1e-10)
        
        return output.real
    
    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache: Optional[IWMCache] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using IWM moments.
        
        For training (seq_len > 1): Uses prefix moments with cumsum for causality.
        For generation (seq_len = 1): Updates cache incrementally (naturally causal).
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple
            cache: IWMCache for generation (optional for training)
            cache_position: position indices
            attention_mask: [batch, seq_len], 1=valid, 0=pad
            
        Returns:
            output: [batch, seq_len, hidden_size]
            psd_loss: scalar loss for PSD constraint
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project Q, V (K not needed for IWM - attention weights come from kernel)
        query = self.q_proj(hidden_states)
        # NOTE: k_proj exists for weight compatibility but is not used
        value = self.v_proj(hidden_states)
        
        # Reshape
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to query (critical for pretrained weights!)
        # position_embeddings = (cos, sin) from rotary_emb
        cos, sin = position_embeddings
        query = self._apply_rotary_pos_emb_q_only(query, cos, sin)
        
        # Get positions for IWM moment computation
        if cache_position is not None:
            positions = cache_position
        elif cache is not None:
            positions = torch.arange(seq_len, device=device) + cache.seen_tokens
        else:
            positions = torch.arange(seq_len, device=device)
        
        # Predict kernel coefficients from RoPE-encoded query
        c_coeffs = self._predict_coefficients(query)  # [batch, heads, seq, S+1]
        self._track_coefficient_stats(c_coeffs)  # Track for monitoring
        psd_loss = self._compute_psd_loss(c_coeffs)
        diversity_loss = self._compute_diversity_loss(c_coeffs)
        self._last_diversity_loss = diversity_loss  # Store for retrieval
        g_coeffs = self._build_kernel_coeffs(c_coeffs)  # [batch, heads, seq, 2S+1]
        
        # Choose computation path based on mode
        if seq_len == 1 and cache is not None:
            # GENERATION MODE: Update cache incrementally, reconstruct from cache
            # This is naturally causal since cache only contains past tokens
            self._update_moments_incremental(cache, value, positions, attention_mask)
            attn_output = self._reconstruct_from_cache(cache, g_coeffs, positions)
        else:
            # TRAINING MODE: Compute causal prefix moments with cumsum
            # This ensures position t only sees tokens j <= t
            v_moments, norm_moments = self._compute_prefix_moments(
                value, positions, attention_mask
            )
            attn_output = self._reconstruct_from_prefix_moments(
                v_moments, norm_moments, g_coeffs, positions
            )
            # Don't update cache in training mode (fresh computation each forward)
        
        # Convert back to original dtype (from float32 complex operations)
        original_dtype = hidden_states.dtype
        attn_output = attn_output.to(original_dtype)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, psd_loss
    
    def forward_explicit(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit weight computation (for debugging/testing).
        
        Returns:
            output: [batch, seq_len, hidden_size]
            attn_weights: [batch, heads, seq_len, seq_len]
            psd_loss: scalar
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Project
        query = self.q_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape
        query = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to query (critical for pretrained weights!)
        cos, sin = position_embeddings
        query = self._apply_rotary_pos_emb_q_only(query, cos, sin)
        
        # Positions for IWM moments
        if cache_position is not None:
            positions = cache_position
        else:
            positions = torch.arange(seq_len, device=device)
        
        theta = positions.float() * self.theta_scale  # [seq_len]
        
        # Predict coefficients from RoPE-encoded query
        c_coeffs = self._predict_coefficients(query)  # [batch, heads, seq, S+1]
        psd_loss = self._compute_psd_loss(c_coeffs)
        g_coeffs = self._build_kernel_coeffs(c_coeffs)  # [batch, heads, seq, 2S+1]
        
        # Explicit kernel computation
        # p_t(θ_j - θ_t) = Σ_k g_k(t) e^{ik(θ_j - θ_t)}
        # For all pairs (t, j): [seq, seq]
        
        delta_theta = theta.unsqueeze(0) - theta.unsqueeze(1)  # [seq, seq] (j, t)
        delta_theta = delta_theta.T  # [seq, seq] now (t, j)
        
        # Phase matrix: e^{ikΔθ} for each (t, j) and k
        # [2S+1, seq, seq]
        phase_matrix = torch.exp(1j * self.k_vals.view(-1, 1, 1) * delta_theta.unsqueeze(0))
        
        # Kernel values: Σ_k g_k(t) * phase[k, t, j]
        # g_coeffs: [batch, heads, seq, 2S+1]
        # phase_matrix: [2S+1, seq, seq]
        # Result: [batch, heads, seq, seq]
        kernel = torch.einsum('bhtk,ktj->bhtj', g_coeffs, phase_matrix)
        kernel = kernel.real  # Should be real for even kernel
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        kernel = kernel.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        
        # Normalize
        kernel_sum = kernel.sum(dim=-1, keepdim=True) + 1e-10
        attn_weights = kernel / kernel_sum
        
        # Expand V for GQA
        if self.num_kv_heads != self.num_heads:
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Attend
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, psd_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[IWMCache] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Main forward matching Qwen interface.
        
        ALWAYS uses moment-based computation to ensure training and inference
        use the same operator.
        
        Training mode with last-token-only (self.training_last_token_only=True):
          - Uses streaming scan to compute output only at last valid token
          - Memory: O(2S+1 × head_dim) instead of O(seq × 2S+1 × head_dim)
          - Returns zeros except at last_idx positions
          
        Training mode full (self.training_last_token_only=False):
          - Uses prefix moments with cumsum for causality
          - Respects attention_mask to exclude padding
          
        Generation mode (past_key_values=IWMCache, seq_len=1):
          - Updates cache incrementally
          - Naturally causal since cache only has past tokens
        """
        batch = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        
        # Normalize attention_mask to [batch, seq_len] with 1=valid, 0=pad
        # NOTE: For IWM, we only need a simple padding mask, not a full causal mask.
        # The causality is handled by the cumsum-based prefix moments.
        if attention_mask is not None:
            # HF attention_mask can come in different formats:
            # 1. [batch, seq_len] with 1=valid, 0=pad (input to model)
            # 2. [batch, 1, tgt_len, src_len] with 0=valid, -inf=masked (causal mask)
            if attention_mask.dim() == 4:
                # This is a causal mask. For padding info, check the diagonal or last row.
                # The last row shows which source positions are valid for any target.
                # Actually, for a simple padding mask, we can just look at the first row:
                # if position j is padding, it's masked for all queries.
                # attention_mask[b, 0, 0, j] == 0 means position j is valid
                # attention_mask[b, 0, 0, j] == -inf means position j is masked
                padding_mask = (attention_mask[:, 0, 0, :] == 0).float()  # [batch, src_len]
                attention_mask = padding_mask
            elif attention_mask.dim() == 2:
                # Already [batch, seq_len] with 1=valid, 0=pad
                attention_mask = attention_mask.float()
        
        # Check if we should use last-token-only path
        # Use it for BOTH training AND eval when training_last_token_only=True
        # (except for generation which needs the cache path)
        is_generation = isinstance(past_key_values, IWMCache) and seq_len == 1
        use_last_token_path = (self.training_last_token_only and 
                               not is_generation and seq_len > 1)
        
        if use_last_token_path:
            # LAST-TOKEN PATH: Only compute output at last valid position
            # Memory-efficient: O(batch × heads × (2S+1) × head_dim) - no seq dimension!
            output, last_indices, psd_loss = self.forward_train_last_token(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
            )
            # Store last_indices for training script to use
            self._last_indices = last_indices
        else:
            # Standard path: full sequence or generation
            output, psd_loss = self.forward_with_cache(
                hidden_states,
                position_embeddings,
                cache=past_key_values if isinstance(past_key_values, IWMCache) else None,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        
        # Store PSD loss in module for training retrieval
        self._last_psd_loss = psd_loss
        
        if output_attentions:
            # For debugging: compute explicit weights if requested
            _, attn_weights, _ = self.forward_explicit(
                hidden_states, position_embeddings, attention_mask, cache_position
            )
            return output, attn_weights
        return output, None


def create_iwm_cache(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bandwidth: int,
    device: torch.device,
) -> IWMCache:
    """Create an empty IWM cache."""
    num_moments = 2 * bandwidth + 1
    
    v_moments = torch.zeros(
        num_layers, num_kv_heads, num_moments, head_dim,
        device=device, dtype=torch.complex64
    )
    norm_moments = torch.zeros(
        num_layers, num_kv_heads, num_moments,
        device=device, dtype=torch.complex64
    )
    
    return IWMCache(
        v_moments=v_moments,
        norm_moments=norm_moments,
        seen_tokens=0,
    )


def replace_attention_with_iwm(model, bandwidth: int = 64):
    """Replace all attention layers with IWMContentKernelAttention."""
    config = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn
        
        new_attn = IWMContentKernelAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            bandwidth=bandwidth,
        )
        
        # Copy projection weights
        new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
        new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
        new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
        new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
        
        # Copy rotary embedding
        if hasattr(old_attn, 'rotary_emb'):
            new_attn.rotary_emb = old_attn.rotary_emb
        
        new_attn = new_attn.to(device=device, dtype=dtype)
        layer.self_attn = new_attn
        
        print(f"Replaced layer {layer_idx} with IWMContentKernelAttention (S={bandwidth})")
    
    return model


def get_iwm_psd_loss(model) -> torch.Tensor:
    """Collect PSD loss from all IWM attention layers."""
    total_loss = 0.0
    count = 0
    
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, '_last_psd_loss'):
            total_loss = total_loss + attn._last_psd_loss
            count += 1
    
    if count > 0:
        return total_loss / count
    return torch.tensor(0.0)


def get_iwm_diversity_loss(model) -> torch.Tensor:
    """
    Collect diversity loss from all IWM attention layers.
    
    This loss encourages the model to use higher harmonics instead of
    staying stuck at uniform attention (c0=1, rest=0).
    """
    total_loss = 0.0
    count = 0
    
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, '_last_diversity_loss'):
            total_loss = total_loss + attn._last_diversity_loss
            count += 1
    
    if count > 0:
        return total_loss / count
    return torch.tensor(0.0)


def get_iwm_diagnostics(model, compute_kernel_neg: bool = False) -> Dict[str, float]:
    """
    Collect diagnostics from all IWM attention layers.
    
    Returns aggregated stats across all layers:
    - denom_min/mean/max: Denominator values (should not be near 0 or exploding)
    - moment_v_norm/moment_z_norm: Moment magnitudes
    - kernel_negative_pct: % of kernel grid points that are negative (expensive)
    - c0_min/mean/max: DC component stats (should stay ~1 if not learning much)
    - ck_ratio_mean/max: Higher harmonics relative to c0 (movement here = learning)
    
    Note: This function calls .item() to convert tensors to floats, causing GPU sync.
    Only call this periodically (e.g., every N steps), not every forward pass.
    """
    stats = {
        'denom_min': float('inf'),
        'denom_mean': 0.0,
        'denom_max': 0.0,
        'moment_v_norm': 0.0,
        'moment_z_norm': 0.0,
        'kernel_negative_pct': 0.0,
        # Coefficient stats - more informative than PSD loss
        'c0_min': float('inf'),
        'c0_mean': 0.0,
        'c0_max': float('-inf'),
        'ck_ratio_mean': 0.0,
        'ck_ratio_max': float('-inf'),
    }
    count = 0
    
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, '_diagnostics'):
            d = attn._diagnostics
            # Convert tensors to floats (this causes GPU sync, but only when diagnostics are read)
            def to_float(v):
                if isinstance(v, torch.Tensor):
                    return v.item()
                return v if v is not None else 0.0
            
            dmin = to_float(d.get('denom_min', float('inf')))
            dmean = to_float(d.get('denom_mean', 0.0))
            dmax = to_float(d.get('denom_max', 0.0))
            
            stats['denom_min'] = min(stats['denom_min'], dmin)
            stats['denom_mean'] += dmean
            stats['denom_max'] = max(stats['denom_max'], dmax)
            stats['moment_v_norm'] += to_float(d.get('moment_v_norm', 0.0))
            stats['moment_z_norm'] += to_float(d.get('moment_z_norm', 0.0))
            
            # Coefficient stats
            c0_min = to_float(d.get('c0_min', float('inf')))
            c0_mean = to_float(d.get('c0_mean', 0.0))
            c0_max = to_float(d.get('c0_max', float('-inf')))
            ck_ratio_mean = to_float(d.get('ck_ratio_mean', 0.0))
            ck_ratio_max = to_float(d.get('ck_ratio_max', float('-inf')))
            
            stats['c0_min'] = min(stats['c0_min'], c0_min)
            stats['c0_mean'] += c0_mean
            stats['c0_max'] = max(stats['c0_max'], c0_max)
            stats['ck_ratio_mean'] += ck_ratio_mean
            stats['ck_ratio_max'] = max(stats['ck_ratio_max'], ck_ratio_max)
            
            count += 1
    
    if count > 0:
        stats['denom_mean'] /= count
        stats['moment_v_norm'] /= count
        stats['moment_z_norm'] /= count
        stats['c0_mean'] /= count
        stats['ck_ratio_mean'] /= count
    
    if stats['denom_min'] == float('inf'):
        stats['denom_min'] = 0.0
    if stats['c0_min'] == float('inf'):
        stats['c0_min'] = 0.0
    if stats['c0_max'] == float('-inf'):
        stats['c0_max'] = 0.0
    if stats['ck_ratio_max'] == float('-inf'):
        stats['ck_ratio_max'] = 0.0
    
    # Kernel negativity is expensive, only compute if requested
    if compute_kernel_neg:
        neg_pcts = []
        for layer in model.model.layers:
            attn = layer.self_attn
            if isinstance(attn, IWMContentKernelAttention):
                # Use random coefficients to test learned projection
                with torch.no_grad():
                    # Match dtype of model parameters
                    param = next(attn.parameters())
                    test_q = torch.randn(1, attn.num_heads, 4, attn.head_dim, 
                                        device=param.device, dtype=param.dtype)
                    c_coeffs = attn._predict_coefficients(test_q)
                    neg_pct = attn._compute_kernel_negativity(c_coeffs)
                    neg_pcts.append(neg_pct)
        if neg_pcts:
            stats['kernel_negative_pct'] = sum(neg_pcts) / len(neg_pcts)
    
    return stats


def get_iwm_layer_coeff_stats(model, layer_indices: list = None) -> Dict[int, Dict[str, float]]:
    """
    Get coefficient stats for specific layers (useful for monitoring learning).
    
    Args:
        model: The model with IWM attention layers
        layer_indices: List of layer indices to report (default: [0, num_layers//2, num_layers-1])
        
    Returns:
        Dict mapping layer_idx -> {c0_mean, c0_std, ck_ratio_mean, ck_ratio_max}
    """
    num_layers = len(model.model.layers)
    
    if layer_indices is None:
        # Default: first, middle, last
        layer_indices = [0, num_layers // 2, num_layers - 1]
    
    results = {}
    
    for layer_idx in layer_indices:
        if layer_idx >= num_layers:
            continue
            
        attn = model.model.layers[layer_idx].self_attn
        if hasattr(attn, '_diagnostics'):
            d = attn._diagnostics
            
            def to_float(v):
                if isinstance(v, torch.Tensor):
                    return v.item()
                return v if v is not None else 0.0
            
            results[layer_idx] = {
                'c0_min': to_float(d.get('c0_min', 0.0)),
                'c0_mean': to_float(d.get('c0_mean', 0.0)),
                'c0_max': to_float(d.get('c0_max', 0.0)),
                'ck_ratio_mean': to_float(d.get('ck_ratio_mean', 0.0)),
                'ck_ratio_max': to_float(d.get('ck_ratio_max', 0.0)),
            }
    
    return results
