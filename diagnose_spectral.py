"""
Diagnose why spectral attention produces garbage.
Compare attention patterns between standard and spectral attention.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")

# Short test sequence
test_text = "def hello():\n    print('Hello')\n"
inputs = tokenizer(test_text, return_tensors="pt")
print(f"Input: {repr(test_text)}")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
print(f"Tokens: {[tokenizer.decode([t]) for t in inputs['input_ids'][0]]}")

seq_len = inputs['input_ids'].shape[1]
print(f"\nSequence length: {seq_len}")

# Simulate what standard attention does
print("\n" + "="*60)
print("ANALYZING ATTENTION MECHANISMS")
print("="*60)

# Load model to get Q, K, V projections
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    torch_dtype=torch.float32,  # Use float32 for analysis
).to("cpu")

# Get hidden states from embedding
with torch.no_grad():
    embed_output = model.model.embed_tokens(inputs['input_ids'])
    
# Get Q, K, V from first layer
layer0 = model.model.layers[0]
attn = layer0.self_attn

with torch.no_grad():
    hidden = embed_output
    
    # Get Q, K
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    
    # Reshape to [batch, heads, seq, head_dim]
    num_heads = attn.config.num_attention_heads
    num_kv_heads = attn.config.num_key_value_heads
    head_dim = q.shape[-1] // num_heads
    
    q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Expand K for GQA
    num_groups = num_heads // num_kv_heads
    k = k.repeat_interleave(num_groups, dim=1)
    
    # Apply RoPE
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(k, position_ids)
    
    # Standard attention scores
    scores_standard = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    
    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores_standard = scores_standard.masked_fill(causal_mask, float('-inf'))
    
    attn_standard = F.softmax(scores_standard, dim=-1)
    
    print("\n1. STANDARD ATTENTION (head 0):")
    print(f"   Score range: [{scores_standard[0,0].min():.2f}, {scores_standard[0,0].max():.2f}]")
    print(f"   Attention sum per row: {attn_standard[0,0].sum(dim=-1)}")  # Should be ~1
    
    # Now simulate spectral attention
    # Normalize Q and K
    q_norm = F.normalize(q, dim=-1)
    k_norm = F.normalize(k, dim=-1)
    
    # Raw dot-product scores (like spectral attention does)
    raw_scores = torch.matmul(q_norm, k_norm.transpose(-2, -1))
    
    print("\n2. NORMALIZED DOT-PRODUCT SCORES (before FFT):")
    print(f"   Score range: [{raw_scores[0,0].min():.2f}, {raw_scores[0,0].max():.2f}]")
    
    # FFT along key dimension
    scores_fft = torch.fft.rfft(raw_scores, dim=-1)
    
    print(f"\n3. FFT of scores:")
    print(f"   FFT shape: {scores_fft.shape}")  # [batch, heads, seq_len, seq_len//2+1]
    print(f"   FFT magnitude (DC component): {scores_fft[0,0,:,0].abs().mean():.4f}")
    print(f"   FFT magnitude (high freq): {scores_fft[0,0,:,-1].abs().mean():.4f}")
    
    # Spectral filter (all ones initially = no filtering)
    spectral_filter = torch.ones(scores_fft.shape[-1])
    scores_filtered = scores_fft * spectral_filter
    
    # IFFT back
    attn_spectral = torch.fft.irfft(scores_filtered, n=seq_len, dim=-1)
    
    print(f"\n4. AFTER IFFT (with no filtering):")
    print(f"   Range: [{attn_spectral[0,0].min():.4f}, {attn_spectral[0,0].max():.4f}]")
    
    # Apply causal mask
    attn_spectral = attn_spectral.masked_fill(causal_mask, float('-inf'))
    
    # Softmax
    attn_spectral_final = F.softmax(attn_spectral, dim=-1)
    
    print(f"\n5. AFTER CAUSAL MASK + SOFTMAX:")
    print(f"   Attention sum per row: {attn_spectral_final[0,0].sum(dim=-1)}")
    
    # Compare attention distributions
    print("\n" + "="*60)
    print("ATTENTION COMPARISON (first row, head 0)")
    print("="*60)
    
    print(f"\nStandard attention (where does token 0 attend?):")
    print(f"   {attn_standard[0,0,0,:].numpy()}")
    
    print(f"\nSpectral attention (where does token 0 attend?):")
    print(f"   {attn_spectral_final[0,0,0,:].numpy()}")
    
    # Check for row 5 (a token in the middle)
    mid = min(5, seq_len-1)
    print(f"\nRow {mid} - Standard: {attn_standard[0,0,mid,:].numpy()}")
    print(f"Row {mid} - Spectral: {attn_spectral_final[0,0,mid,:].numpy()}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Standard attention heatmap
    ax = axes[0, 0]
    im = ax.imshow(attn_standard[0, 0].numpy(), cmap='viridis', aspect='auto')
    ax.set_title("Standard Attention (head 0)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax)
    
    # Spectral attention heatmap  
    ax = axes[0, 1]
    im = ax.imshow(attn_spectral_final[0, 0].numpy(), cmap='viridis', aspect='auto')
    ax.set_title("Spectral Attention (head 0)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax)
    
    # Difference
    ax = axes[1, 0]
    diff = (attn_spectral_final[0, 0] - attn_standard[0, 0]).abs().numpy()
    im = ax.imshow(diff, cmap='Reds', aspect='auto')
    ax.set_title(f"Absolute Difference (max={diff.max():.3f})")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax)
    
    # FFT magnitudes
    ax = axes[1, 1]
    fft_mags = scores_fft[0, 0].abs().mean(dim=0).numpy()
    ax.plot(fft_mags)
    ax.set_title("FFT Magnitude (averaged over queries)")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Magnitude")
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("outputs/attention_comparison.png", dpi=150)
    print(f"\nSaved attention comparison to outputs/attention_comparison.png")
    
    # Key diagnostic: entropy
    def entropy(p):
        p = p.clamp(min=1e-10)
        return -(p * p.log()).sum(dim=-1)
    
    standard_entropy = entropy(attn_standard[0, 0]).mean()
    spectral_entropy = entropy(attn_spectral_final[0, 0]).mean()
    
    print(f"\n" + "="*60)
    print("ENTROPY (higher = more uniform attention)")
    print("="*60)
    print(f"Standard attention entropy: {standard_entropy:.4f}")
    print(f"Spectral attention entropy: {spectral_entropy:.4f}")
    
    if spectral_entropy < standard_entropy * 0.5:
        print("\n⚠️  Spectral attention is MUCH more concentrated!")
        print("   This could cause the model to focus on wrong tokens.")
    elif spectral_entropy > standard_entropy * 1.5:
        print("\n⚠️  Spectral attention is MUCH more uniform!")
        print("   This could cause the model to lose focus entirely.")
    else:
        print("\n✓ Entropy is similar - attention distribution preserved")
