"""
FMMA Feature Networks: Nonnegative Key Features and Query Moment Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonnegativeKeyFeatures(nn.Module):
    """
    Produces r nonnegative features from key vectors.

    φ_m(k) = softplus(u_m^T k + b_m) for m = 1, ..., r

    Ensures φ_m(k) ≥ 0 which is required for FMMA nonnegativity.
    """

    def __init__(self, key_dim: int, num_channels: int, beta: float = 1.0):
        super().__init__()
        self.key_dim = key_dim
        self.num_channels = num_channels

        self.projection = nn.Linear(key_dim, num_channels, bias=True)
        self.beta = beta

        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keys: [batch, num_heads, seq_len, head_dim]
        Returns:
            features: [batch, num_heads, seq_len, r] nonnegative
        """
        projected = self.projection(keys)
        return F.softplus(projected, beta=self.beta)


class QueryMomentNetwork(nn.Module):
    """
    Produces query-conditioned harmonic moments μ^{(m)}_k(q) with PSD Toeplitz structure.

    For each channel m, outputs complex moments satisfying:
    - μ_0 = 1 (normalized)
    - μ_{-k} = conj(μ_k) (Hermitian symmetry)
    - T_S(μ) ≽ 0 (positive semidefinite Toeplitz matrix)

    Fix 1.2 (PLAN.md): Uses autocorrelation parameterization to *hard-enforce*
    PSD Toeplitz.  We parameterize a vector a ∈ C^{S+1} and compute moments as
    the autocorrelation μ_k = (a ⋆ a*)_k / ||a||², which is guaranteed PSD
    Toeplitz with μ_0 = 1.  This replaces the old |μ_k| < 1 heuristic.

    When enforce_psd=False (legacy), falls back to soft |μ_k| < 1.
    """

    def __init__(
        self,
        query_dim: int,
        num_channels: int,
        bandwidth: int,
        hidden_dim: int = None,
        enforce_psd: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.enforce_psd = enforce_psd

        if hidden_dim is None:
            hidden_dim = query_dim

        if enforce_psd:
            # Autocorrelation parameterization: output (S+1) complex coefficients
            # per channel.  The autocorrelation of a is guaranteed PSD Toeplitz.
            output_dim = num_channels * (bandwidth + 1) * 2  # real + imag
        else:
            # Legacy: S complex numbers per channel (μ_1 … μ_S)
            output_dim = num_channels * bandwidth * 2

        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _autocorrelation_psd(raw_flat: torch.Tensor, S: int) -> torch.Tensor:
        """
        Autocorrelation PSD parameterization (Fix 1.2).

        Isolated as a static method so it can be wrapped with
        torch.utils.checkpoint to trade compute for memory.

        Args:
            raw_flat: [N, r, S+1, 2]  (flattened over B*H*T)
            S: bandwidth
        Returns:
            moments: [N, r, 2S+1] complex64
        """
        N, r = raw_flat.shape[:2]
        raw_f = raw_flat.float()
        a = torch.complex(raw_f[..., 0], raw_f[..., 1])  # [N, r, S+1]

        # Autocorrelation via FFT: μ = IFFT(|FFT(a)|²), normalised so μ_0=1
        n_fft = 2 * S + 1
        A = torch.fft.fft(a, n=n_fft, dim=-1)  # [N, r, 2S+1]
        power = A.abs().square()                 # |A|²  (fused, avoids separate .real/.imag)
        del A                                    # free immediately
        autocorr = torch.fft.ifft(power, dim=-1) # [N, r, 2S+1]
        del power

        # Normalise so μ_0 = 1
        mu_0 = autocorr[..., 0:1].real.clamp(min=1e-8)
        autocorr = autocorr / mu_0

        # Build full Hermitian moment vector [μ_{-S}…μ_0…μ_S]
        mu_positive = autocorr[..., 1 : S + 1]   # [N, r, S]
        mu_negative = torch.conj(torch.flip(mu_positive, dims=[-1]))
        mu_zero = torch.ones(N, r, 1, dtype=torch.complex64, device=raw_flat.device)
        return torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)  # [N, r, 2S+1]

    def forward(self, queries: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        """
        Args:
            queries: [B, H, T, D] or [B, H, D]
            use_checkpoint: if True, checkpoint the FFT autocorrelation to save memory
        Returns:
            moments: [B, H, T, r, 2S+1] or [B, H, r, 2S+1] complex
        """
        is_single_query = queries.dim() == 3
        if is_single_query:
            queries = queries.unsqueeze(2)

        B, H, T, D = queries.shape
        S = self.bandwidth
        r = self.num_channels

        raw = self.net(queries)  # [B, H, T, output_dim]

        if self.enforce_psd:
            # Flatten B*H*T so the FFT operates on a 3-D tensor (much less memory)
            raw_flat = raw.view(B * H * T, r, S + 1, 2)

            if use_checkpoint and raw_flat.requires_grad:
                moments_flat = torch.utils.checkpoint.checkpoint(
                    QueryMomentNetwork._autocorrelation_psd,
                    raw_flat, S,
                    use_reentrant=False,
                )
            else:
                moments_flat = QueryMomentNetwork._autocorrelation_psd(raw_flat, S)

            moments = moments_flat.view(B, H, T, r, 2 * S + 1)
        else:
            # --- Legacy soft constraint ---
            raw = raw.view(B, H, T, r, S, 2)
            raw_f = raw.float()
            mu_positive = torch.complex(raw_f[..., 0], raw_f[..., 1])

            magnitude = torch.abs(mu_positive)
            phase = torch.angle(mu_positive)
            scaled_mag = 0.95 * torch.sigmoid(magnitude)
            mu_positive = scaled_mag * torch.exp(1j * phase)

            mu_negative = torch.conj(torch.flip(mu_positive, dims=[-1]))
            mu_zero = torch.ones(B, H, T, r, 1, dtype=torch.complex64, device=queries.device)
            moments = torch.cat([mu_negative, mu_zero, mu_positive], dim=-1)

        if is_single_query:
            moments = moments.squeeze(2)

        return moments
