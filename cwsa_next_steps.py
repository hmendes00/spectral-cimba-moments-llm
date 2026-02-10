"""
CWSA Next Steps: Turning Hypotheses Into Measurements
======================================================

The cwsa_sha256_analysis.py script identified three hypotheses that were
NOT measured. This script measures them.

HYPOTHESIS A: Cocycle-guided backtracking → replaced by the more useful
              FILTRATION COMPLEXITY PROFILE (rotation collisions at each
              Witt level, diffusion deficit measurement).

HYPOTHESIS B: Spectral carry tracking → extended into a REDUCED-ROUND
              CWSA DISTINGUISHER (at how few rounds does carry structure
              become detectable?).

HYPOTHESIS C: Cohomological constraint counting → implemented as an
              explicit CONSTRAINT RANK measurement using the carry
              cocycle identity across rounds.

Author: Hugo L.S. Mendes  (framework by Copilot)
"""

import numpy as np
from math import gcd
from itertools import combinations
import struct

BITS = 32
MOD = 2**BITS
MASK = MOD - 1


# ============================================================================
# UTILITIES (shared with cwsa_sha256_analysis.py)
# ============================================================================

def rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (BITS - n))) & MASK

def carry_cocycle(a: int, b: int) -> int:
    """Lifted carry: (a XOR b) XOR ((a+b) mod 2^32)."""
    return (a ^ b) ^ ((a + b) & MASK)

def word_to_bits(w: int) -> np.ndarray:
    return np.array([(w >> j) & 1 for j in range(BITS)], dtype=np.float64)

def rand32(rng) -> int:
    lo = int(rng.randint(0, 0x10000))
    hi = int(rng.randint(0, 0x10000))
    return (hi << 16) | lo

def Sigma0(a): return rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
def Sigma1(e): return rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
def Ch(e, f, g):
    e32 = e & MASK
    return (e32 & f) ^ ((~e32 & MASK) & g)
def Maj(a, b, c): return (a & b) ^ (a & c) ^ (b & c)
def sigma0(x): return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
def sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)

K_SHA = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]
H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


# ============================================================================
# EXPERIMENT 1: FILTRATION COMPLEXITY PROFILE
# ============================================================================
#
# At each Witt filtration level j (working mod 2^j), SHA-256's rotation
# constants may COLLIDE. When two rotations are identical mod j, their
# XOR cancels → the Σ function loses a diffusion path at that level.
#
# This is a MEASURABLE, ALGEBRAIC quantity — not a hypothesis.

def filtration_complexity_profile():
    """
    Compute the filtration complexity profile of SHA-256.
    
    For each filtration level j = 1..32:
      - How many distinct rotation values does each Σ function have mod j?
      - What is the EFFECTIVE number of diffusion paths?
      - What is the diffusion deficit Δ_j?
    
    Then MEASURE the actual diffusion: how well does truncated-Σ spread
    information across the j-bit word?
    """
    print("=" * 72)
    print("EXPERIMENT 1: FILTRATION COMPLEXITY PROFILE")
    print("=" * 72)
    
    sigma0_rots = [2, 13, 22]   # Σ₀ rotation amounts
    sigma1_rots = [6, 11, 25]   # Σ₁ rotation amounts
    
    # Also analyze message schedule σ functions
    msched_sigma0_rots = [7, 18]   # σ₀ (plus >>3 shift, not rotation)
    msched_sigma1_rots = [17, 19]  # σ₁ (plus >>10 shift, not rotation)
    
    print("\n--- Rotation collision analysis at each filtration level ---")
    print(f"  {'Level j':>8s}  {'Σ₀ distinct':>11s}  {'Σ₀ Δ_j':>8s}  "
          f"{'Σ₁ distinct':>11s}  {'Σ₁ Δ_j':>8s}  {'Weakness?':>10s}")
    
    sigma0_deficits = []
    sigma1_deficits = []
    
    for j in range(1, 33):
        # Σ₀: rotations mod j
        s0_mod = [r % j for r in sigma0_rots]
        s0_distinct = len(set(s0_mod))
        s0_deficit = 1.0 - s0_distinct / 3.0
        sigma0_deficits.append(s0_deficit)
        
        # Σ₁: rotations mod j  
        s1_mod = [r % j for r in sigma1_rots]
        s1_distinct = len(set(s1_mod))
        s1_deficit = 1.0 - s1_distinct / 3.0
        sigma1_deficits.append(s1_deficit)
        
        weak = ""
        if s0_deficit > 0 or s1_deficit > 0:
            parts = []
            if s0_deficit > 0:
                parts.append(f"Σ₀:{s0_mod}")
            if s1_deficit > 0:
                parts.append(f"Σ₁:{s1_mod}")
            weak = " ".join(parts)
        
        print(f"  {j:>8d}  {s0_distinct:>8d}/3   {s0_deficit:>8.3f}  "
              f"{s1_distinct:>8d}/3   {s1_deficit:>8.3f}  {weak}")
    
    # Compute the SECURITY PRODUCT across all levels
    print("\n--- Security product formula ---")
    print("  Security(r) ≥ ∏ⱼ (1 - Δ_j)^r")
    print("  where Δ_j = max(Δ_j(Σ₀), Δ_j(Σ₁))")
    
    combined_deficits = [max(s0, s1) for s0, s1 in zip(sigma0_deficits, sigma1_deficits)]
    non_zero_levels = [(j+1, d) for j, d in enumerate(combined_deficits) if d > 0]
    
    print(f"\n  Levels with non-zero deficit:")
    for j, d in non_zero_levels:
        print(f"    j={j}: Δ={d:.3f}  (1-Δ)={1-d:.3f}")
    
    for num_rounds in [4, 16, 32, 64]:
        product = 1.0
        for _, d in non_zero_levels:
            product *= (1 - d) ** num_rounds
        bits_lost = -np.log2(product) if product > 0 else float('inf')
        print(f"\n  SHA-256 at {num_rounds:2d} rounds: security product = {product:.6f}"
              f"  ({bits_lost:.1f} spectral bits lost)")
    
    # ─── Now MEASURE actual diffusion at each level ───
    print("\n--- Measured diffusion at weak filtration levels ---")
    print("  (How well does truncated-Σ₀ spread a 1-bit change at level j?)")
    
    rng = np.random.RandomState(42)
    
    for j in [2, 3, 4, 6, 8, 11, 13, 16, 22, 32]:
        if j > 32:
            continue
        j_mask = (1 << j) - 1
        n_trials = 50000
        
        # Measure: for random x, flip bit i, check how many output bits change
        total_flips = 0
        total_bit_changes = 0
        
        for _ in range(n_trials):
            x = rand32(rng) & j_mask
            
            # Apply Σ₀ truncated to j bits
            # For j < 32, rotation mod j
            if j >= 32:
                sx = Sigma0(x) & j_mask
            else:
                # Compute ROTR amounts mod j
                r1, r2, r3 = [r % j for r in sigma0_rots]
                # Rotate within j-bit word
                def rotr_j(val, rot, width):
                    return ((val >> rot) | (val << (width - rot))) & ((1 << width) - 1)
                sx = rotr_j(x, r1, j) ^ rotr_j(x, r2, j) ^ rotr_j(x, r3, j)
            
            # Flip each input bit, measure output change
            for bit in range(j):
                x_flipped = x ^ (1 << bit)
                if j >= 32:
                    sx_flipped = Sigma0(x_flipped) & j_mask
                else:
                    sx_flipped = rotr_j(x_flipped, r1, j) ^ rotr_j(x_flipped, r2, j) ^ rotr_j(x_flipped, r3, j)
                
                diff = sx ^ sx_flipped
                bits_changed = bin(diff).count('1')
                total_flips += 1
                total_bit_changes += bits_changed
        
        avg_avalanche = total_bit_changes / total_flips
        ideal_avalanche = j / 2.0  # ideal: each input bit flips half of output bits
        avalanche_ratio = avg_avalanche / ideal_avalanche if ideal_avalanche > 0 else 0
        
        # Check if this level has a rotation collision
        s0_mod = set([r % j for r in sigma0_rots])
        has_collision = len(s0_mod) < 3
        
        print(f"  Level j={j:2d}: avg flipped bits = {avg_avalanche:.2f}/{j}  "
              f"avalanche ratio = {avalanche_ratio:.3f}  "
              f"{'⚠ COLLISION' if has_collision else '✓ ok'}")
    
    # ─── Compare Σ₀ vs Σ₁ at all collision levels ───
    print("\n--- Σ₀ vs Σ₁ avalanche comparison at collision levels ---")
    for j in range(2, 33):
        s0_mod = set([r % j for r in sigma0_rots])
        s1_mod = set([r % j for r in sigma1_rots])
        
        if len(s0_mod) < 3 or len(s1_mod) < 3:
            j_mask = (1 << j) - 1
            n_trials = 20000
            
            def measure_avalanche(rot_amounts, width, n):
                total_changes = 0
                total_flips = 0
                r_mods = [r % width for r in rot_amounts]
                def rotr_w(val, rot, w):
                    return ((val >> rot) | (val << (w - rot))) & ((1 << w) - 1)
                for _ in range(n):
                    x = rand32(rng) & ((1 << width) - 1)
                    sx = rotr_w(x, r_mods[0], width) ^ rotr_w(x, r_mods[1], width) ^ rotr_w(x, r_mods[2], width)
                    for bit in range(width):
                        xf = x ^ (1 << bit)
                        sxf = rotr_w(xf, r_mods[0], width) ^ rotr_w(xf, r_mods[1], width) ^ rotr_w(xf, r_mods[2], width)
                        total_changes += bin(sx ^ sxf).count('1')
                        total_flips += 1
                return total_changes / total_flips if total_flips > 0 else 0
            
            s0_aval = measure_avalanche(sigma0_rots, j, n_trials)
            s1_aval = measure_avalanche(sigma1_rots, j, n_trials)
            ideal = j / 2.0
            
            s0_flag = "⚠ COLLISION" if len(s0_mod) < 3 else ""
            s1_flag = "⚠ COLLISION" if len(s1_mod) < 3 else ""
            
            print(f"  j={j:2d}: Σ₀ avalanche={s0_aval:.2f}/{ideal:.1f} {s0_flag:>14s}"
                  f"  |  Σ₁ avalanche={s1_aval:.2f}/{ideal:.1f} {s1_flag}")

    return combined_deficits


# ============================================================================
# EXPERIMENT 2: REDUCED-ROUND CWSA DISTINGUISHER
# ============================================================================
#
# If CWSA structure is real but Σ functions destroy it, then at FEWER
# rounds the structure should be detectable. At how many rounds does
# carry structure become indistinguishable from random?
#
# This is the key experiment: find the "crossover round count."

def reduced_round_distinguisher():
    """
    Test whether CWSA-based analysis can distinguish r-round SHA-256
    from a random function, for r = 1, 2, 4, 8, 16, 32, 64.
    
    METHOD:
    For each round count r, run SHA-256 for only r rounds on many
    random inputs. Then measure:
    
    1. CARRY SPECTRAL BIAS: Is the carry spectrum across rounds
       more structured than expected from random additions?
       
    2. WITT-FOURIER COHERENCE: Do consecutive rounds' Witt-Fourier
       moments have higher correlation than random?
       
    3. LOW-BIT PREDICTABILITY: Can we predict the low bits of the
       output from the carry structure better than chance?
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: REDUCED-ROUND CWSA DISTINGUISHER")
    print("=" * 72)
    
    rng = np.random.RandomState(42)
    n_trials = 2000
    
    round_counts = [1, 2, 4, 8, 16, 32, 64]
    
    print("\n--- Test 1: Carry spectral bias per round count ---")
    print("  (Do r-round carry spectra have more structure than random?)")
    print(f"  {'Rounds':>8s}  {'Carry spec variance':>20s}  {'Random baseline':>16s}  {'Ratio':>8s}  {'Verdict':>12s}")
    
    for n_rounds in round_counts:
        # Run n_rounds of SHA-256 on random messages, collect carry spectra
        carry_spec_variances = []
        random_spec_variances = []
        
        for trial in range(n_trials):
            # Random initial state
            state = [rand32(rng) for _ in range(8)]
            W = [rand32(rng) for _ in range(64)]
            
            a, b, c, d, e, f, g, h = state
            round_carry_specs = []
            
            for t in range(n_rounds):
                round_carry_word = 0
                
                s1 = Sigma1(e)
                ch = Ch(e, f, g)
                
                acc = h
                for operand in [s1, ch, K_SHA[t], W[t]]:
                    c_word = carry_cocycle(acc, operand)
                    round_carry_word ^= c_word
                    acc = (acc + operand) & MASK
                T1 = acc
                
                s0 = Sigma0(a)
                maj = Maj(a, b, c)
                c_word = carry_cocycle(s0, maj)
                round_carry_word ^= c_word
                T2 = (s0 + maj) & MASK
                
                c_word = carry_cocycle(d, T1)
                round_carry_word ^= c_word
                c_word = carry_cocycle(T1, T2)
                round_carry_word ^= c_word
                
                carry_spec = np.abs(np.fft.fft(word_to_bits(round_carry_word)))**2
                round_carry_specs.append(carry_spec)
                
                h = g; g = f; f = e
                e = (d + T1) & MASK
                d = c; c = b; b = a
                a = (T1 + T2) & MASK
            
            if len(round_carry_specs) > 1:
                spec_matrix = np.array(round_carry_specs)
                # Variance of spectrum across rounds (high = structured)
                carry_spec_variances.append(np.mean(np.var(spec_matrix, axis=0)))
            
            # Baseline: random words' carry spectra
            random_specs = []
            for t in range(n_rounds):
                rx = rand32(rng)
                random_specs.append(np.abs(np.fft.fft(word_to_bits(rx)))**2)
            if len(random_specs) > 1:
                rspec_matrix = np.array(random_specs)
                random_spec_variances.append(np.mean(np.var(rspec_matrix, axis=0)))
        
        if carry_spec_variances and random_spec_variances:
            sha_var = np.mean(carry_spec_variances)
            rand_var = np.mean(random_spec_variances)
            ratio = sha_var / rand_var if rand_var > 0 else float('inf')
            verdict = "STRUCTURED" if ratio > 1.5 else "MARGINAL" if ratio > 1.1 else "≈ RANDOM"
            print(f"  {n_rounds:>8d}  {sha_var:>20.2f}  {rand_var:>16.2f}  {ratio:>8.3f}  {verdict:>12s}")
        else:
            print(f"  {n_rounds:>8d}  (need >1 round for variance)")
    
    # ─── Test 2: Cross-round Witt moment correlation ───
    print("\n--- Test 2: Cross-round carry correlation by round count ---")
    print("  (At how many rounds does carry correlation drop to noise?)")
    print(f"  {'Rounds':>8s}  {'Mean |corr|':>12s}  {'Max |corr|':>12s}  {'Verdict':>12s}")
    
    for n_rounds in round_counts:
        if n_rounds < 2:
            print(f"  {n_rounds:>8d}  (need ≥2 rounds)")
            continue
            
        all_corrs = []
        
        for trial in range(n_trials):
            state = [rand32(rng) for _ in range(8)]
            W = [rand32(rng) for _ in range(64)]
            
            a, b, c, d, e, f, g, h = state
            carry_words = []
            
            for t in range(n_rounds):
                round_carry_word = 0
                
                s1 = Sigma1(e)
                ch = Ch(e, f, g)
                acc = h
                for operand in [s1, ch, K_SHA[t], W[t]]:
                    c_word = carry_cocycle(acc, operand)
                    round_carry_word ^= c_word
                    acc = (acc + operand) & MASK
                T1 = acc
                
                s0 = Sigma0(a)
                maj = Maj(a, b, c)
                c_word = carry_cocycle(s0, maj)
                round_carry_word ^= c_word
                T2 = (s0 + maj) & MASK
                
                c_word = carry_cocycle(d, T1)
                round_carry_word ^= c_word
                c_word = carry_cocycle(T1, T2)
                round_carry_word ^= c_word
                
                carry_words.append(round_carry_word)
                
                h = g; g = f; f = e
                e = (d + T1) & MASK
                d = c; c = b; b = a
                a = (T1 + T2) & MASK
            
            # Correlation between consecutive rounds' carry bit patterns
            for t in range(n_rounds - 1):
                bits_t = word_to_bits(carry_words[t])
                bits_t1 = word_to_bits(carry_words[t + 1])
                cc = np.corrcoef(bits_t, bits_t1)[0, 1]
                if not np.isnan(cc):
                    all_corrs.append(abs(cc))
        
        if all_corrs:
            mean_corr = np.mean(all_corrs)
            max_corr = np.max(all_corrs)
            verdict = "DETECTABLE" if mean_corr > 0.1 else "MARGINAL" if mean_corr > 0.05 else "≈ NOISE"
            print(f"  {n_rounds:>8d}  {mean_corr:>12.4f}  {max_corr:>12.4f}  {verdict:>12s}")
    
    # ─── Test 3: Low-bit carry prediction ───
    print("\n--- Test 3: Low-bit output prediction from carry structure ---")
    print("  (Can knowing carry bits help predict output bits?)")
    print(f"  {'Rounds':>8s}  {'Bit 0 pred':>11s}  {'Bit 1 pred':>11s}  "
          f"{'Bit 7 pred':>11s}  {'Bit 15 pred':>11s}  {'Verdict':>10s}")
    
    for n_rounds in round_counts:
        # For each trial: run n_rounds, record (carry_features, output_bits)
        bit_correct = {0: 0, 1: 0, 7: 0, 15: 0}
        total = 0
        
        for trial in range(n_trials):
            state = [rand32(rng) for _ in range(8)]
            W = [rand32(rng) for _ in range(64)]
            
            a, b, c, d, e, f, g, h = state
            total_carry_hw = 0  # total Hamming weight of all carries
            
            for t in range(n_rounds):
                s1 = Sigma1(e)
                ch = Ch(e, f, g)
                acc = h
                for operand in [s1, ch, K_SHA[t], W[t]]:
                    cw = carry_cocycle(acc, operand)
                    total_carry_hw += bin(cw).count('1')
                    acc = (acc + operand) & MASK
                T1 = acc
                
                s0 = Sigma0(a)
                maj = Maj(a, b, c)
                cw = carry_cocycle(s0, maj)
                total_carry_hw += bin(cw).count('1')
                T2 = (s0 + maj) & MASK
                
                cw = carry_cocycle(d, T1)
                total_carry_hw += bin(cw).count('1')
                cw = carry_cocycle(T1, T2)
                total_carry_hw += bin(cw).count('1')
                
                h = g; g = f; f = e
                e = (d + T1) & MASK
                d = c; c = b; b = a
                a = (T1 + T2) & MASK
            
            # Predict: if total carry HW is even, guess output bit = 0
            # (This is a simple parity-based distinguisher)
            carry_parity = total_carry_hw % 2
            
            for target_bit in [0, 1, 7, 15]:
                output_bit = (a >> target_bit) & 1
                if output_bit == carry_parity:
                    bit_correct[target_bit] += 1
            total += 1
        
        accs = {b: bit_correct[b] / total for b in bit_correct}
        # Bias = |accuracy - 0.5| (0 = random, 0.5 = perfect predictor)
        max_bias = max(abs(a - 0.5) for a in accs.values())
        verdict = "BIASED" if max_bias > 0.02 else "MARGINAL" if max_bias > 0.01 else "≈ RANDOM"
        
        print(f"  {n_rounds:>8d}  {accs[0]:>10.4f}  {accs[1]:>10.4f}  "
              f"{accs[7]:>10.4f}  {accs[15]:>10.4f}  {verdict:>10s}")


# ============================================================================
# EXPERIMENT 3: CARRY CONSTRAINT RANK (H¹ PROXY)
# ============================================================================
#
# The cocycle identity says:
#   L(c(a,b)) ⊕ L(c(a⊞b, z)) = L(c(b,z)) ⊕ L(c(a, b⊞z))
#
# Across r rounds of SHA-256, each round introduces 7 additions.
# The cocycle identity constrains how these carries relate.
#
# Naively: 7 additions × 32 bits × r rounds = 224r carry variables.
# But the cocycle identity creates linear (over GF(2)) constraints
# that reduce the degrees of freedom.
#
# H¹ ≈ (carry variables) - rank(constraint matrix)
#
# We measure the rank empirically.

def constraint_rank_measurement():
    """
    Build the carry constraint matrix for r rounds of SHA-256
    and measure its rank over GF(2).
    
    The constraint matrix has:
      - Rows: one per cocycle identity application
      - Columns: one per carry bit (32 × 7 × r)
      - Entry: 1 if that carry bit appears in that identity
    
    We approximate this by sampling: for many random inputs,
    record which carry bits are correlated (via the identity),
    and measure the effective rank of the correlation matrix.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: CARRY CONSTRAINT RANK (H¹ PROXY)")
    print("=" * 72)
    
    rng = np.random.RandomState(42)
    
    print("\n--- Measuring carry degrees of freedom per round count ---")
    print("  Naive DOF = 7 additions × 32 bits × r rounds = 224r")
    print("  Actual DOF = rank of carry-bit correlation matrix")
    print(f"\n  {'Rounds':>8s}  {'Naive DOF':>10s}  {'Effective rank':>15s}  "
          f"{'Reduction':>10s}  {'DOF ratio':>10s}")
    
    n_samples = 3000
    
    for n_rounds in [1, 2, 4, 8, 16]:
        naive_dof = 7 * 32 * n_rounds
        
        # Collect carry bit vectors for many random inputs
        # Each trial produces a binary vector of length 7*32*n_rounds
        carry_vectors = []
        
        for trial in range(n_samples):
            state = [rand32(rng) for _ in range(8)]
            W = [rand32(rng) for _ in range(64)]
            
            a, b, c, d, e, f, g, h = state
            carry_bits = []
            
            for t in range(n_rounds):
                s1 = Sigma1(e)
                ch = Ch(e, f, g)
                
                # 4 additions for T1
                acc = h
                for operand in [s1, ch, K_SHA[t], W[t]]:
                    cw = carry_cocycle(acc, operand)
                    for bit in range(32):
                        carry_bits.append((cw >> bit) & 1)
                    acc = (acc + operand) & MASK
                T1 = acc
                
                # 1 addition for T2
                s0 = Sigma0(a)
                maj = Maj(a, b, c)
                cw = carry_cocycle(s0, maj)
                for bit in range(32):
                    carry_bits.append((cw >> bit) & 1)
                T2 = (s0 + maj) & MASK
                
                # Addition 6: d + T1
                cw = carry_cocycle(d, T1)
                for bit in range(32):
                    carry_bits.append((cw >> bit) & 1)
                
                # Addition 7: T1 + T2
                cw = carry_cocycle(T1, T2)
                for bit in range(32):
                    carry_bits.append((cw >> bit) & 1)
                
                h = g; g = f; f = e
                e = (d + T1) & MASK
                d = c; c = b; b = a
                a = (T1 + T2) & MASK
            
            carry_vectors.append(carry_bits)
        
        # Build correlation matrix and measure effective rank
        carry_matrix = np.array(carry_vectors, dtype=np.float64)  # (n_samples, 224*n_rounds)
        
        # Center the matrix
        carry_matrix -= carry_matrix.mean(axis=0, keepdims=True)
        
        # SVD to find effective rank
        # (number of singular values above threshold)
        if carry_matrix.shape[1] <= carry_matrix.shape[0]:
            try:
                U, S, Vt = np.linalg.svd(carry_matrix, full_matrices=False)
                # Effective rank: count singular values > 1% of max
                threshold = 0.01 * S[0] if S[0] > 0 else 0
                eff_rank = int(np.sum(S > threshold))
            except np.linalg.LinAlgError:
                eff_rank = carry_matrix.shape[1]
        else:
            # More columns than samples — use covariance matrix
            cov = carry_matrix.T @ carry_matrix / n_samples
            eigvals = np.linalg.eigvalsh(cov)
            threshold = 0.01 * max(eigvals) if max(eigvals) > 0 else 0
            eff_rank = int(np.sum(eigvals > threshold))
        
        reduction = naive_dof - eff_rank
        dof_ratio = eff_rank / naive_dof if naive_dof > 0 else 0
        
        print(f"  {n_rounds:>8d}  {naive_dof:>10d}  {eff_rank:>15d}  "
              f"{reduction:>10d}  {dof_ratio:>10.3f}")
    
    # ─── Deeper analysis: WHERE are the redundant dimensions? ───
    print("\n--- Per-addition carry rank analysis (1 round) ---")
    print("  Testing which of the 7 additions contribute independent carries")
    
    addition_labels = [
        "h + Σ₁(e)",
        "acc + Ch(e,f,g)", 
        "acc + K_t",
        "acc + W_t",
        "Σ₀(a) + Maj(a,b,c)",
        "d + T1",
        "T1 + T2"
    ]
    
    # Collect per-addition carry vectors
    per_addition_carries = {i: [] for i in range(7)}
    
    for trial in range(n_samples):
        state = [rand32(rng) for _ in range(8)]
        W_vals = [rand32(rng) for _ in range(64)]
        
        a, b, c, d, e, f, g, h = state
        
        s1 = Sigma1(e)
        ch = Ch(e, f, g)
        
        add_idx = 0
        acc = h
        for operand in [s1, ch, K_SHA[0], W_vals[0]]:
            cw = carry_cocycle(acc, operand)
            bits = [(cw >> bit) & 1 for bit in range(32)]
            per_addition_carries[add_idx].append(bits)
            acc = (acc + operand) & MASK
            add_idx += 1
        T1 = acc
        
        s0 = Sigma0(a)
        maj = Maj(a, b, c)
        cw = carry_cocycle(s0, maj)
        per_addition_carries[4].append([(cw >> bit) & 1 for bit in range(32)])
        T2 = (s0 + maj) & MASK
        
        cw = carry_cocycle(d, T1)
        per_addition_carries[5].append([(cw >> bit) & 1 for bit in range(32)])
        
        cw = carry_cocycle(T1, T2)
        per_addition_carries[6].append([(cw >> bit) & 1 for bit in range(32)])
    
    print(f"  {'Addition':>25s}  {'Rank/32':>8s}  {'Info bits':>10s}")
    for i in range(7):
        mat = np.array(per_addition_carries[i], dtype=np.float64)
        mat -= mat.mean(axis=0, keepdims=True)
        try:
            S = np.linalg.svd(mat, compute_uv=False)
            thresh = 0.01 * S[0] if len(S) > 0 and S[0] > 0 else 0
            rank = int(np.sum(S > thresh))
        except:
            rank = 32
        print(f"  {addition_labels[i]:>25s}  {rank:>5d}/32  {rank:>10d}")
    
    # Cross-addition mutual information (correlation between addition carries)
    print("\n--- Cross-addition carry correlation (1 round) ---")
    print("  (Are carries from different additions within the same round correlated?)")
    
    all_carries = np.hstack([np.array(per_addition_carries[i], dtype=np.float64) 
                             for i in range(7)])
    all_carries -= all_carries.mean(axis=0, keepdims=True)
    
    # Correlation between addition blocks
    for i, j_idx in [(0, 1), (0, 4), (0, 5), (3, 5), (4, 6), (5, 6)]:
        block_i = np.array(per_addition_carries[i], dtype=np.float64)
        block_j = np.array(per_addition_carries[j_idx], dtype=np.float64)
        
        # Average correlation between all bit pairs across the two blocks
        corrs = []
        for bi in range(32):
            for bj in range(32):
                cc = np.corrcoef(block_i[:, bi], block_j[:, bj])[0, 1]
                if not np.isnan(cc):
                    corrs.append(abs(cc))
        
        mean_corr = np.mean(corrs) if corrs else 0
        max_corr = np.max(corrs) if corrs else 0
        
        print(f"  {addition_labels[i]:>25s}  ↔  {addition_labels[j_idx]:<25s}"
              f"  mean|r|={mean_corr:.4f}  max|r|={max_corr:.4f}")


# ============================================================================
# EXPERIMENT 4: OPTIMAL ROTATION CONSTANT SEARCH
# ============================================================================
#
# From the filtration analysis: SHA-256's Σ₀ has a collision at level 4
# ({2,13,22} mod 4 = {2,1,2}). Can we find rotation triples that have
# NO collisions at ANY level?

def optimal_rotation_search():
    """
    Search for rotation triples (r1, r2, r3) that avoid collisions
    at ALL filtration levels j = 2..32.
    
    Constraint: {r1 mod j, r2 mod j, r3 mod j} must have 3 distinct
    values for all j ∈ {2, 3, ..., 32}.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: OPTIMAL ROTATION CONSTANT SEARCH")
    print("=" * 72)
    
    print("\n--- Searching for collision-free rotation triples in [0..31] ---")
    
    best_triples = []
    
    for r1 in range(32):
        for r2 in range(r1 + 1, 32):
            for r3 in range(r2 + 1, 32):
                collision_free = True
                for j in range(2, 33):
                    mods = {r1 % j, r2 % j, r3 % j}
                    if len(mods) < 3:
                        collision_free = False
                        break
                
                if collision_free:
                    # Also check: do they generate the full rotation group?
                    # gcd of all rotation amounts should be 1 with 32
                    g = gcd(gcd(r1, r2), gcd(r3, 32))
                    full_group = (g == 1)
                    
                    best_triples.append((r1, r2, r3, full_group))
    
    print(f"  Total collision-free triples: {len(best_triples)}")
    
    if best_triples:
        # Sort by whether they generate full group, then by spread
        best_triples.sort(key=lambda t: (-int(t[3]), -(t[2] - t[0])))
        
        print(f"\n  Top 20 collision-free triples (sorted by spread):")
        print(f"  {'(r1, r2, r3)':>16s}  {'Full group?':>12s}  {'Spread':>8s}  {'Sum':>6s}")
        for r1, r2, r3, fg in best_triples[:20]:
            spread = r3 - r1
            print(f"  ({r1:2d}, {r2:2d}, {r3:2d})    {'YES' if fg else 'no':>12s}  {spread:>8d}  {r1+r2+r3:>6d}")
        
        # Compare SHA-256's actual choices
        print(f"\n  SHA-256's actual choices:")
        for name, rots in [("Σ₀", [2, 13, 22]), ("Σ₁", [6, 11, 25])]:
            is_cf = (rots[0], rots[1], rots[2]) in [(t[0], t[1], t[2]) for t in best_triples]
            # Check unsorted too
            sorted_rots = tuple(sorted(rots))
            is_cf = any(tuple(sorted([t[0], t[1], t[2]])) == sorted_rots for t in best_triples)
            
            first_collision = None
            for j in range(2, 33):
                mods = {r % j for r in rots}
                if len(mods) < 3:
                    first_collision = j
                    break
            
            status = "COLLISION-FREE ✓" if is_cf else f"COLLISION at j={first_collision}"
            print(f"    {name}: {rots}  →  {status}")
    else:
        print("  No collision-free triples found!")
        print("  This would mean collisions are unavoidable — interesting in itself.")
        
        # Find the best (fewest collisions) triples
        print("\n  Searching for triples with fewest collisions...")
        min_collisions = 32
        best_few = []
        
        for r1 in range(32):
            for r2 in range(r1 + 1, 32):
                for r3 in range(r2 + 1, 32):
                    collisions = 0
                    for j in range(2, 33):
                        mods = {r1 % j, r2 % j, r3 % j}
                        if len(mods) < 3:
                            collisions += 1
                    
                    if collisions < min_collisions:
                        min_collisions = collisions
                        best_few = [(r1, r2, r3, collisions)]
                    elif collisions == min_collisions:
                        best_few.append((r1, r2, r3, collisions))
        
        print(f"  Minimum achievable collisions: {min_collisions}")
        print(f"  Triples achieving minimum ({len(best_few)} found):")
        for r1, r2, r3, nc in best_few[:10]:
            collision_levels = []
            for j in range(2, 33):
                mods = {r1 % j, r2 % j, r3 % j}
                if len(mods) < 3:
                    collision_levels.append(j)
            print(f"    ({r1:2d}, {r2:2d}, {r3:2d})  collisions at j={collision_levels}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  CWSA Next Steps: Turning Hypotheses Into Measurements          ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    deficits = filtration_complexity_profile()
    reduced_round_distinguisher()
    constraint_rank_measurement()
    optimal_rotation_search()
    
    print("\n" + "=" * 72)
    print("SUMMARY OF NEW FINDINGS")
    print("=" * 72)
    print("""
  These experiments convert three hypotheses from cwsa_sha256_analysis.py
  into measured results:
  
  1. FILTRATION COMPLEXITY PROFILE: Actual rotation collision counts and
     measured avalanche deficits at each Witt level. The security product
     formula gives concrete bit-loss numbers.
  
  2. REDUCED-ROUND DISTINGUISHER: Empirical measurement of at how many
     rounds the CWSA carry structure becomes indistinguishable from random.
     This identifies the "crossover round count" — the minimum rounds
     needed for spectral security.
  
  3. CONSTRAINT RANK: The effective degrees of freedom in carry bits
     across r rounds, measured via SVD rank. The ratio (effective / naive)
     quantifies how much the cocycle identity constrains the carry space.
  
  4. OPTIMAL ROTATION SEARCH: Exhaustive enumeration of collision-free
     rotation triples, showing whether SHA-256's choices are optimal
     in the filtration sense.
    """)


if __name__ == "__main__":
    main()
