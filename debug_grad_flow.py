"""Debug: trace gradient flow through IWM attention"""

import torch
import torch.nn as nn

# Simple test of the gradient path
print("Testing gradient flow through complex operations...")

# Simulate the coefficient path
S = 8
batch, heads = 2, 4
seq = 16
head_dim = 32

# Input with gradient
c_coeffs = torch.randn(batch, heads, S+1, requires_grad=True)

# Build two-sided coefficients (simplified)
g_coeffs = torch.zeros(batch, heads, 2*S + 1)
g_coeffs[..., S] = c_coeffs[..., 0]
for k in range(1, S + 1):
    half_ck = 0.5 * c_coeffs[..., k]
    g_coeffs[..., S + k] = half_ck
    g_coeffs[..., S - k] = half_ck

# Apply Fejér weights
k_vals = torch.arange(-S, S + 1, dtype=torch.float32)
fejer = 1.0 - torch.abs(k_vals) / (S + 1)
g_coeffs = g_coeffs * fejer

print(f"After Fejér: g_coeffs.requires_grad = {g_coeffs.requires_grad}")

# Convert to complex
g_complex = torch.complex(g_coeffs, torch.zeros_like(g_coeffs))
print(f"After complex: g_complex.requires_grad = {g_complex.requires_grad}")

# Simulate query phase
theta = torch.linspace(0, 1, seq)
query_phase = torch.exp(-1j * k_vals.unsqueeze(0) * theta.unsqueeze(1))

# Expand g_complex for seq dimension
g_expanded = g_complex.unsqueeze(2).expand(-1, -1, seq, -1)

# Combine
combined = g_expanded * query_phase.unsqueeze(0).unsqueeze(0)
print(f"After combine: combined.requires_grad = {combined.requires_grad}")

# Simulate moments (these don't need gradients)
v_moments = torch.randn(batch, heads, seq, 2*S+1, head_dim, dtype=torch.complex64)
z_moments = torch.randn(batch, heads, seq, 2*S+1, dtype=torch.complex64)

# Compute output
numerator = (combined.unsqueeze(-1) * v_moments).sum(dim=3)
denominator = (combined * z_moments).sum(dim=3)

output = numerator / (denominator.unsqueeze(-1) + 1e-10)
output_real = output.real

print(f"output_real.requires_grad = {output_real.requires_grad}")

# Compute loss and backward
loss = output_real.sum()
loss.backward()

print(f"\nc_coeffs.grad norm: {c_coeffs.grad.norm().item():.6f}")
print(f"c_coeffs.grad max: {c_coeffs.grad.abs().max().item():.6f}")

if c_coeffs.grad.norm().item() > 0:
    print("\n✓ Gradients ARE flowing through complex operations")
else:
    print("\n✗ Gradients are NOT flowing")
