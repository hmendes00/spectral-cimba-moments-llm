"""
CWSA Algebraic Analysis - Polynomial GCDs over F2
Implements Appendix H of CWSA paper

Key theorem: collision at level j iff gcd(P(z), z^j - 1) != 1 in F2[z]
(Over F2, z^j - 1 = z^j + 1; we use z^j - 1 throughout for standard notation.)

Two types of weakness:
  Type A (diffusion reduction): 2 of 3 rotations coincide mod j,
         reducing Sigma from 3 XOR paths to 1. Still bijective (no kernel).
  Type B (kernel annihilation): gcd(P(z), z^j - 1) != 1 -> nontrivial kernel,
         some cyclic input patterns are completely destroyed. NOT injective.
"""

import numpy as np
import random as _random
from itertools import combinations
from collections import Counter

# SHA-256 rotation constants
SIGMA0_ROTS = (2, 13, 22)
SIGMA1_ROTS = (6, 11, 25)
WORD_BITS = 32

# ======================================================================
# F2[z] polynomial arithmetic  (bit i = coefficient of z^i)
# ======================================================================

def f2_deg(p):
    if p == 0:
        return -1
    return p.bit_length() - 1

def f2_mul(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        b >>= 1
    return result

def f2_divmod(a, b):
    if b == 0:
        raise ZeroDivisionError
    da, db = f2_deg(a), f2_deg(b)
    q = 0
    while da >= db:
        shift = da - db
        q ^= (1 << shift)
        a ^= (b << shift)
        da = f2_deg(a)
    return q, a

def f2_mod(a, b):
    return f2_divmod(a, b)[1]

def f2_gcd(a, b):
    while b:
        a, b = b, f2_mod(a, b)
    return a

def f2_poly_str(p):
    if p == 0:
        return "0"
    d = f2_deg(p)
    terms = []
    for i in range(d, -1, -1):
        if p & (1 << i):
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append("z")
            else:
                terms.append(f"z^{i}")
    return " + ".join(terms)

def rots_to_poly(rots):
    p = 0
    for r in rots:
        p ^= (1 << r)
    return p

P0 = rots_to_poly(SIGMA0_ROTS)
P1 = rots_to_poly(SIGMA1_ROTS)

# ======================================================================
# F2 matrix operations (Gaussian elimination mod 2)
# ======================================================================

def f2_matrix_rank(M):
    """Rank of binary matrix over F2 via Gaussian elimination."""
    A = np.array(M, dtype=np.int64) % 2
    rows, cols = A.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if A[row, col] % 2 == 1:
                pivot = row
                break
        if pivot is None:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(rows):
            if row != rank and A[row, col] % 2 == 1:
                A[row] = (A[row] + A[rank]) % 2
        rank += 1
    return rank

def f2_kernel_vectors(M, max_vecs=5):
    """Find kernel vectors of M over F2 by brute force (small matrices only)."""
    mat = np.array(M, dtype=np.int64) % 2
    r, c = mat.shape
    rank = f2_matrix_rank(mat)
    ker = []
    if c > 20:
        return ker
    for bits in range(1, 1 << c):
        v = np.array([(bits >> i) & 1 for i in range(c)], dtype=np.int64)
        if np.all(mat @ v % 2 == 0):
            ker.append(v)
            if len(ker) >= max_vecs:
                break
    return ker

def build_sigma_matrix_f2(rots, j):
    """Build j x j circulant matrix for Sigma function mod j over F2."""
    M = np.zeros((j, j), dtype=np.int64)
    for r in rots:
        for row in range(j):
            col = (row + r) % j
            M[row, col] ^= 1
    return M

# ======================================================================
# F2[z] factoring (trial division with fast irreducibility test)
# ======================================================================

def _f2_is_irreducible(p):
    """Test if p is irreducible over F2 using z^(2^k) criterion."""
    d = f2_deg(p)
    if d <= 0:
        return False
    if d == 1:
        return True
    z = 0b10
    for k in range(1, d // 2 + 1):
        z2k = z
        for _ in range(k):
            z2k = f2_mod(f2_mul(z2k, z2k), p)
        g = f2_gcd(z2k ^ z, p)
        if g != 1:  # includes g == p: means p | z^(2^k)-z, so p is reducible
            return False
    z2d = z
    for _ in range(d):
        z2d = f2_mod(f2_mul(z2d, z2d), p)
    return z2d == z

def f2_factor_trial(p):
    """Factor p over F2[z] by trial division. Only searches up to deg/2.
    NOTE: Demo-scale factorer — correct but O(2^(deg/2)). Fine for
    the small SHA-256 polynomials here (deg ≤ 25). Not suitable for
    polynomials of degree > ~30."""
    if p <= 1:
        return {}
    factors = {}
    remaining = p
    max_trial_deg = f2_deg(remaining) // 2 + 1

    for d in range(1, max_trial_deg + 1):
        for bits in range(1 << d, 1 << (d + 1)):
            candidate = bits
            if not _f2_is_irreducible(candidate):
                continue
            while f2_deg(remaining) >= d:
                q, r = f2_divmod(remaining, candidate)
                if r == 0:
                    factors[candidate] = factors.get(candidate, 0) + 1
                    remaining = q
                else:
                    break
            if f2_deg(remaining) <= 0:
                break
        if f2_deg(remaining) <= 0:
            break

    if remaining > 1:
        factors[remaining] = factors.get(remaining, 0) + 1
    return factors

def f2_multiplicative_order(p):
    """Smallest n such that p divides z^n - 1.
    Returns None if p is divisible by z (no finite order)."""
    d = f2_deg(p)
    if d <= 0:
        return 0
    # z-divisible polynomials never divide z^n - 1 (since z^n-1 has nonzero constant term)
    if p & 1 == 0:
        return None
    max_order = (1 << d) - 1
    # Compute z^n mod p incrementally: start with z mod p
    z_pow = f2_mod(0b10, p)  # z mod p (always reduce, regardless of degree)
    for n in range(1, max_order + 1):
        if z_pow == 1:
            return n
        z_pow = f2_mod(f2_mul(z_pow, 0b10), p)  # z^(n+1) = z * z^n
    return max_order


# ======================================================================
# EXPERIMENT 1: Two Collision Types Compared
# ======================================================================

def experiment_1():
    print("=" * 72)
    print("EXPERIMENT 1: Two Types of Sigma-Weakness")
    print("=" * 72)
    print()
    print("Type A (diffusion reduction): 2 of 3 rotations coincide mod j")
    print("  -> Sigma degenerates from 3 XOR paths to 1. Still bijective.")
    print()
    print("Type B (kernel annihilation): gcd(P(z), z^j-1) != 1 in F2[z]")
    print("  (Over F2: z^j-1 = z^j+1, so both notations are equivalent.)")
    print("  -> Nontrivial kernel: some inputs mapped to zero. NOT injective.")
    print()
    print(f"P0(z) = {f2_poly_str(P0)}   (Sigma0 rots: {SIGMA0_ROTS})")
    print(f"P1(z) = {f2_poly_str(P1)}   (Sigma1 rots: {SIGMA1_ROTS})")
    print()

    typeA_0, typeA_1 = [], []
    typeB_0, typeB_1 = [], []

    for j in range(1, WORD_BITS + 1):
        rots0_mod = set(r % j for r in SIGMA0_ROTS)
        rots1_mod = set(r % j for r in SIGMA1_ROTS)
        if len(rots0_mod) < 3:
            typeA_0.append(j)
        if len(rots1_mod) < 3:
            typeA_1.append(j)

        zj = (1 << j) ^ 1
        g0 = f2_gcd(P0, zj)
        g1 = f2_gcd(P1, zj)
        if g0 > 1:
            typeB_0.append(j)
        if g1 > 1:
            typeB_1.append(j)

    print("TYPE A (diffusion reduction):")
    print(f"  Sigma0 weak at: {typeA_0}")
    print(f"  Sigma1 weak at: {typeA_1}")
    print()
    print("TYPE B (kernel annihilation):")
    print(f"  Sigma0 weak at: {typeB_0}")
    print(f"  Sigma1 weak at: {typeB_1}")
    print()

    typeA_both = set(typeA_0) & set(typeA_1)
    typeB_both = set(typeB_0) & set(typeB_1)

    print("STAGGERING CHECK:")
    nontrivial_A = {j for j in typeA_both if j >= 3}
    if typeA_both:
        if nontrivial_A:
            print(f"  Type A simultaneous at j >= 3: {sorted(nontrivial_A)}")
        else:
            print(f"  Type A simultaneous only at trivial j in {sorted(typeA_both)}")
            print(f"  -> Type A staggering holds for j >= 3!")
    else:
        print(f"  Type A: perfectly staggered at all levels!")

    print(f"  Type B simultaneous: {sorted(typeB_both) if typeB_both else 'NONE'}")
    if typeB_both:
        print(f"    KERNEL-STAGGERING FAILS at j = {sorted(typeB_both)}")
    print()

    # Cross-check Type B with F2 matrix rank
    print("Cross-check: Type B via F2 Gaussian elimination")
    bf_0, bf_1 = [], []
    for j in range(1, WORD_BITS + 1):
        M0 = build_sigma_matrix_f2(SIGMA0_ROTS, j)
        M1 = build_sigma_matrix_f2(SIGMA1_ROTS, j)
        r0 = f2_matrix_rank(M0)
        r1 = f2_matrix_rank(M1)
        if r0 < j:
            bf_0.append(j)
        if r1 < j:
            bf_1.append(j)

    match0 = "EXACT" if bf_0 == typeB_0 else f"MISMATCH bf={bf_0}"
    match1 = "EXACT" if bf_1 == typeB_1 else f"MISMATCH bf={bf_1}"
    print(f"  Sigma0: GCD={typeB_0}  brute={bf_0}  {match0}")
    print(f"  Sigma1: GCD={typeB_1}  brute={bf_1}  {match1}")
    print()

    # Summary table
    print(f" {'j':>4} | {'--- Type A ---':^14} | {'--- Type B (kernel) ---':^22}")
    print(f" {'':>4} | {'S0':>4} {'S1':>4} {'Both':>4} | {'S0':>5} {'S1':>5} {'Both':>5}")
    print("-" * 55)
    for j in range(1, WORD_BITS + 1):
        a0 = "X" if j in typeA_0 else "."
        a1 = "X" if j in typeA_1 else "."
        aB = "!!" if j in typeA_both else ""
        b0 = "X" if j in typeB_0 else "."
        b1 = "X" if j in typeB_1 else "."
        bB = "!!!" if j in typeB_both else ""
        if j in typeA_0 or j in typeA_1 or j in typeB_0 or j in typeB_1:
            print(f" {j:>4} | {a0:>4} {a1:>4} {aB:>4} | {b0:>5} {b1:>5} {bB:>5}")
    print()


# ======================================================================
# EXPERIMENT 2: Kernel Structure
# ======================================================================

def experiment_2():
    print("=" * 72)
    print("EXPERIMENT 2: Kernel Dimension and Structure")
    print("=" * 72)
    print()
    print("dim ker(Sigma_j) = deg gcd(P(z), z^j-1)   [z^j-1 = z^j+1 over F2]")
    print()

    print(f" {'j':>4} | {'ker0':>5} | {'ker1':>5} | {'deficit':>8} | Kernel info")
    print("-" * 70)

    total_ker0 = 0
    total_ker1 = 0

    for j in range(1, WORD_BITS + 1):
        zj = (1 << j) ^ 1
        g0 = f2_gcd(P0, zj)
        g1 = f2_gcd(P1, zj)
        d0 = f2_deg(g0)
        d1 = f2_deg(g1)

        if d0 == 0 and d1 == 0:
            continue

        total_ker0 += d0
        total_ker1 += d1
        deficit = max(d0, d1) / j

        ker_str = ""
        if d0 > 0 and j <= 14:
            M0 = build_sigma_matrix_f2(SIGMA0_ROTS, j)
            ker = f2_kernel_vectors(M0, max_vecs=3)
            if ker:
                vecs = ["".join(str(int(x)) for x in v) for v in ker[:2]]
                ker_str = "ker0: " + ", ".join(vecs)
        elif d1 > 0 and j <= 14:
            M1 = build_sigma_matrix_f2(SIGMA1_ROTS, j)
            ker = f2_kernel_vectors(M1, max_vecs=3)
            if ker:
                vecs = ["".join(str(int(x)) for x in v) for v in ker[:2]]
                ker_str = "ker1: " + ", ".join(vecs)

        print(f" {j:>4} | {d0:>5} | {d1:>5} | {deficit:>8.4f} | {ker_str}")

    print()
    print(f"Total kernel dims: Sigma0 = {total_ker0}, Sigma1 = {total_ker1}")
    print()


# ======================================================================
# EXPERIMENT 3: Coprimality (Staggering Criterion)
# ======================================================================

def experiment_3():
    print("=" * 72)
    print("EXPERIMENT 3: Coprimality of P0 and P1 (Staggering Criterion)")
    print("=" * 72)
    print()
    print("Lemma (Appendix H): kernel-staggering holds iff P0, P1 share")
    print("no irreducible factor that divides z^j-1 for any j >= 3")
    print()

    g = f2_gcd(P0, P1)
    print(f"P0(z) = {f2_poly_str(P0)}")
    print(f"P1(z) = {f2_poly_str(P1)}")
    print(f"gcd(P0, P1) = {f2_poly_str(g)}")
    print()

    if g == 1:
        print("P0 and P1 are COPRIME -> kernel-staggering holds at all levels!")
    else:
        print(f"P0 and P1 share factor {f2_poly_str(g)} of degree {f2_deg(g)}")
        ord_g = f2_multiplicative_order(g)
        if ord_g is None:
            # Shared factor is z^k — has no multiplicative order,
            # never divides z^j-1, so irrelevant for kernel collisions
            print(f"  This is a power of z (no cyclotomic order).")
            print(f"  z^k never divides z^j-1, so this factor does NOT create")
            print(f"  simultaneous kernel collisions.")
            print()
            # Check reduced coprimality (remove z factors)
            P0_red = P0
            while P0_red & 1 == 0 and P0_red > 0:
                P0_red >>= 1
            P1_red = P1
            while P1_red & 1 == 0 and P1_red > 0:
                P1_red >>= 1
            g_red = f2_gcd(P0_red, P1_red)
            print(f"  Reduced polynomials (z-free):")
            print(f"    P0/z^{f2_deg(P0)-f2_deg(P0_red)} = {f2_poly_str(P0_red)}")
            print(f"    P1/z^{f2_deg(P1)-f2_deg(P1_red)} = {f2_poly_str(P1_red)}")
            print(f"    gcd = {f2_poly_str(g_red)}")
            if g_red == 1:
                print("  -> Reduced polys COPRIME: no shared cyclotomic factor!")
                print("  -> Simultaneous kernel at j=21 arises from DIFFERENT")
                print("     irreducible factors with orders 7 and 3, lcm=21.")
            else:
                print(f"  -> Reduced polys share {f2_poly_str(g_red)}")
        else:
            print(f"Order of shared factor: {ord_g}")
            print(f"-> Simultaneous kernel at ALL multiples of {ord_g}")
            print()
            for j in range(3, WORD_BITS + 1):
                zj = (1 << j) ^ 1
                if f2_mod(zj, g) == 0:
                    g0 = f2_gcd(P0, zj)
                    g1 = f2_gcd(P1, zj)
                    print(f"  j={j}: ker0={f2_deg(g0)}, ker1={f2_deg(g1)}")
    print()


# ======================================================================
# EXPERIMENT 4: Cyclotomic Factor Analysis
# ======================================================================

def experiment_4():
    print("=" * 72)
    print("EXPERIMENT 4: Cyclotomic Factor Analysis")
    print("=" * 72)
    print()
    print("Factoring P0(z) and P1(z) over F2[z]...")
    print("(Trial division with fast irreducibility test)")
    print()

    factors_P0 = f2_factor_trial(P0)
    factors_P1 = f2_factor_trial(P1)

    print(f"P0(z) = {f2_poly_str(P0)}")
    print("  Irreducible factors:")
    for f, mult in sorted(factors_P0.items(), key=lambda x: f2_deg(x[0])):
        order = f2_multiplicative_order(f)
        m = f"^{mult}" if mult > 1 else ""
        ord_str = str(order) if order is not None else 'N/A (z-factor)'
        print(f"    ({f2_poly_str(f)}){m}  deg={f2_deg(f)}  ord={ord_str}")
    print()

    print(f"P1(z) = {f2_poly_str(P1)}")
    print("  Irreducible factors:")
    for f, mult in sorted(factors_P1.items(), key=lambda x: f2_deg(x[0])):
        order = f2_multiplicative_order(f)
        m = f"^{mult}" if mult > 1 else ""
        ord_str = str(order) if order is not None else 'N/A (z-factor)'
        print(f"    ({f2_poly_str(f)}){m}  deg={f2_deg(f)}  ord={ord_str}")
    print()

    # Verify
    prod0 = 1
    for f, m in factors_P0.items():
        for _ in range(m):
            prod0 = f2_mul(prod0, f)
    prod1 = 1
    for f, m in factors_P1.items():
        for _ in range(m):
            prod1 = f2_mul(prod1, f)
    print(f"Verification: P0 = product? {prod0 == P0}")
    print(f"Verification: P1 = product? {prod1 == P1}")
    print()

    shared = set(factors_P0.keys()) & set(factors_P1.keys())
    if shared:
        print("SHARED factors:")
        for f in sorted(shared, key=f2_deg):
            order = f2_multiplicative_order(f)
            if order is None:
                print(f"  {f2_poly_str(f)}  (deg {f2_deg(f)}, z-factor, no cyclotomic order)")
                print(f"  -> Harmless: z^k never divides z^j-1")
            else:
                print(f"  {f2_poly_str(f)}  (deg {f2_deg(f)}, ord {order})")
                print(f"  -> Simultaneous kernel at multiples of {order}")
    else:
        print("No shared factors -> staggering provably holds!")
    print()

    # Factor table (exclude z-factors for sorting)
    all_factors = set(list(factors_P0.keys()) + list(factors_P1.keys()))
    def sort_key(f):
        o = f2_multiplicative_order(f)
        return (0, 0) if o is None else (1, o)
    print(f"  {'Factor':>25} | {'deg':>3} | {'ord':>6} | {'P0':>3} | {'P1':>3} | z^j-1 at j=")
    print("  " + "-" * 75)
    for f in sorted(all_factors, key=sort_key):
        order = f2_multiplicative_order(f)
        in0 = "Y" if f in factors_P0 else " "
        in1 = "Y" if f in factors_P1 else " "
        if order is None:
            print(f"  {f2_poly_str(f):>25} | {f2_deg(f):>3} | {'N/A':>6} | {in0:>3} | {in1:>3} | (z-factor)")
        else:
            multiples = [str(m) for m in range(order, WORD_BITS + 1, order)]
            print(f"  {f2_poly_str(f):>25} | {f2_deg(f):>3} | {order:>6} | {in0:>3} | {in1:>3} | {','.join(multiples)}")
    print()


# ======================================================================
# EXPERIMENT 5: Refined Security Product
# ======================================================================

def experiment_5():
    print("=" * 72)
    print("EXPERIMENT 5: Security Product (Type A vs Type B)")
    print("=" * 72)
    print()
    print("Type A deficit: delta_j = 1/3 if rotations degenerate mod j")
    print("Type B deficit: delta_j = dim_ker(Sigma_j) / j")
    print()

    bits_A = 0.0
    bits_B = 0.0

    print(f" {'j':>4} | {'dA0':>7} {'dA1':>7} {'maxA':>7} | {'dB0':>7} {'dB1':>7} {'maxB':>7}")
    print("-" * 65)

    for j in range(1, WORD_BITS + 1):
        rots0 = set(r % j for r in SIGMA0_ROTS)
        rots1 = set(r % j for r in SIGMA1_ROTS)
        dA0 = 1 - len(rots0) / 3
        dA1 = 1 - len(rots1) / 3
        max_A = max(dA0, dA1)

        zj = (1 << j) ^ 1
        g0 = f2_gcd(P0, zj)
        g1 = f2_gcd(P1, zj)
        dB0 = f2_deg(g0) / j if f2_deg(g0) > 0 else 0
        dB1 = f2_deg(g1) / j if f2_deg(g1) > 0 else 0
        max_B = max(dB0, dB1)

        if max_A > 0 or max_B > 0:
            bits_A += -np.log2(1 - max_A) if max_A < 1 else 32
            bits_B += -np.log2(1 - max_B) if max_B < 1 else 32
            print(f" {j:>4} | {dA0:>7.4f} {dA1:>7.4f} {max_A:>7.4f} | {dB0:>7.4f} {dB1:>7.4f} {max_B:>7.4f}")

    print()
    print("Security bits lost per round:")
    print(f"  Type A (diffusion): {bits_A:.1f} bits/round")
    print(f"  Type B (kernel):    {bits_B:.1f} bits/round")
    print()
    for r in [4, 16, 32, 64]:
        print(f"  {r:>3} rounds: Type A = {r*bits_A:.1f} bits, Type B = {r*bits_B:.1f} bits")
    print()
    if bits_A > 0:
        print(f"Type B / Type A ratio: {bits_B/bits_A:.3f}")
        print("(Type B is the algebraically precise measure; Type A overcounts)")
    print()


# ======================================================================
# EXPERIMENT 6: Optimal Sigma-Pair Search  
# ======================================================================

def experiment_6():
    print("=" * 72)
    print("EXPERIMENT 6: Optimal Sigma-Pair Search")
    print("=" * 72)
    print()
    print("Search for rotation triples minimizing total kernel deficit.")
    print()

    # Find single triples with fewest Type B collisions
    print("--- Single triple: minimize kernel collision count (j >= 3) ---")
    best_count = WORD_BITS + 1
    best_singles = []

    for r1, r2, r3 in combinations(range(WORD_BITS), 3):
        p = (1 << r1) ^ (1 << r2) ^ (1 << r3)
        count = 0
        for j in range(3, WORD_BITS + 1):
            zj = (1 << j) ^ 1
            if f2_gcd(p, zj) > 1:
                count += 1
        if count < best_count:
            best_count = count
            best_singles = [(r1, r2, r3)]
        elif count == best_count:
            best_singles.append((r1, r2, r3))

    print(f"  Minimum kernel collisions (j>=3): {best_count}")
    print(f"  Triples achieving minimum: {len(best_singles)}")
    for t in best_singles[:10]:
        p = (1 << t[0]) ^ (1 << t[1]) ^ (1 << t[2])
        levels = [j for j in range(3, WORD_BITS+1) if f2_gcd(p, (1<<j)^1)>1]
        print(f"    {t} at j={levels}")
    print()

    # Compare SHA-256
    sha0_count = sum(1 for j in range(3, WORD_BITS+1) if f2_gcd(P0, (1<<j)^1)>1)
    sha1_count = sum(1 for j in range(3, WORD_BITS+1) if f2_gcd(P1, (1<<j)^1)>1)
    print(f"  SHA-256 Sigma0 {SIGMA0_ROTS}: {sha0_count} kernel collisions")
    print(f"  SHA-256 Sigma1 {SIGMA1_ROTS}: {sha1_count} kernel collisions")
    print()

    # Pair search
    print("--- Pair search: minimize combined kernel deficit ---")
    candidates = best_singles[:30]
    candidates.append(SIGMA0_ROTS)
    candidates.append(SIGMA1_ROTS)
    seen = set()
    unique = []
    for t in candidates:
        key = tuple(sorted(t))
        if key not in seen:
            seen.add(key)
            unique.append(t)
    candidates = unique

    best_deficit = float('inf')
    best_pairs = []

    for i, t1 in enumerate(candidates):
        p1 = (1 << t1[0]) ^ (1 << t1[1]) ^ (1 << t1[2])
        for t2 in candidates[i+1:]:
            p2 = (1 << t2[0]) ^ (1 << t2[1]) ^ (1 << t2[2])
            g = f2_gcd(p1, p2)
            total = 0.0
            for j in range(3, WORD_BITS + 1):
                zj = (1 << j) ^ 1
                d1 = f2_deg(f2_gcd(p1, zj))
                d2 = f2_deg(f2_gcd(p2, zj))
                total += max(d1, d2) / j
            if total < best_deficit:
                best_deficit = total
                best_pairs = [(t1, t2, g)]
            elif abs(total - best_deficit) < 1e-10:
                best_pairs.append((t1, t2, g))

    print(f"  Best pair deficit: {best_deficit:.4f}")
    for t1, t2, g in best_pairs[:5]:
        coprime = "coprime" if g == 1 else f"gcd={f2_poly_str(g)}"
        print(f"    Sa={t1} Sb={t2}  {coprime}")
    print()

    sha_deficit = sum(
        max(f2_deg(f2_gcd(P0, (1<<j)^1)), f2_deg(f2_gcd(P1, (1<<j)^1))) / j
        for j in range(3, WORD_BITS + 1)
    )
    print(f"  SHA-256 pair deficit: {sha_deficit:.4f}")
    print(f"  Optimal pair deficit: {best_deficit:.4f}")
    if best_deficit > 0:
        print(f"  SHA-256 is {sha_deficit/best_deficit:.2f}x worse than optimal")
    print()


# ======================================================================
# EXPERIMENT 7: Kernel-Carry Correlation
# ======================================================================

def experiment_7():
    print("=" * 72)
    print("EXPERIMENT 7: Do Kernel Collisions Amplify Output Collisions?")
    print("=" * 72)
    print()

    np.random.seed(42)
    _random.seed(42)
    N = 10000

    print(f" {'j':>4} | {'Expected':>12} | {'Observed':>12} | {'Ratio':>7} | Type")
    print("-" * 65)

    for j in [3, 5, 7, 10, 14, 21, 32]:
        mask = (1 << j) - 1
        outputs = Counter()
        for _ in range(N):
            x = _random.randrange(1 << j)
            r1, r2, r3 = SIGMA0_ROTS
            y = (((x >> (r1%j)) | (x << (j - r1%j))) & mask) ^ \
                (((x >> (r2%j)) | (x << (j - r2%j))) & mask) ^ \
                (((x >> (r3%j)) | (x << (j - r3%j))) & mask)
            outputs[y] += 1

        obs = sum(c*(c-1)//2 for c in outputs.values())
        exp = N*(N-1) / (2 * (1 << j)) if j < 20 else 0

        zj = (1 << j) ^ 1
        g0 = f2_gcd(P0, zj)
        rots_mod = set(r % j for r in SIGMA0_ROTS)
        if f2_deg(g0) > 0:
            ctype = f"B (ker={f2_deg(g0)})"
        elif len(rots_mod) < 3:
            ctype = "A (diffusion)"
        else:
            ctype = "none"

        ratio = obs / exp if exp > 0 else float('inf')
        print(f" {j:>4} | {exp:>12.1f} | {obs:>12} | {ratio:>7.2f} | {ctype}")

    print()
    print("Ratio > 1 -> MORE collisions than random -> weaker")
    print()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print("CWSA Algebraic Analysis - Polynomial GCDs over F2")
    print("Implements Appendix H of CWSA paper")
    print()

    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()
    experiment_7()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("""
KEY FINDINGS:

1. TWO DISTINCT WEAKNESS TYPES:
   - Type A (diffusion): rotation degeneracy, still bijective, no kernel
   - Type B (kernel): gcd(P,z^j-1) != 1, information LOST, not injective
   [Over F2, z^j-1 = z^j+1; we standardize on z^j-1 notation.]
   Type B is algebraically precise; Type A overcounts.

2. SHA-256 KERNEL STRUCTURE:
   - Sigma0: kernel at multiples of 7 (factor z^3+z^2+1, order 7)
   - Sigma1: kernel at multiples of 3 (factor z^2+z+1, order 3)
   - j=21 = lcm(3,7): BOTH have kernel -> staggering fails!

3. COPRIMALITY:
   - If gcd(P0,P1) = 1: kernel-staggering guaranteed
   - SHA-256: check experimentally via Experiments 3 and 4

4. SECURITY PRODUCT:
   - Type A overcounts weakness (~8 bits/round)
   - Type B is precise (~5 bits/round)
   - At 64 rounds: Type B ~ 323 bits (vs ~ 513 Type A)""")

    # Derive optimal-design claim from actual experiment data instead of hardcoding
    best_p = rots_to_poly((0, 1, 2))  # test a consecutive triple
    best_kc = sum(1 for j in range(3, WORD_BITS + 1) if f2_gcd(best_p, (1 << j) ^ 1) > 1)
    # Scan ALL triples to verify
    min_kc = WORD_BITS + 1
    min_count = 0
    are_consecutive = False
    for r1, r2, r3 in combinations(range(WORD_BITS), 3):
        p = (1 << r1) ^ (1 << r2) ^ (1 << r3)
        kc = sum(1 for j in range(3, WORD_BITS + 1) if f2_gcd(p, (1 << j) ^ 1) > 1)
        if kc < min_kc:
            min_kc = kc
            min_count = 1
        elif kc == min_kc:
            min_count += 1
    # Check if consecutive triples are among the optimal
    consec_optimal = all(
        sum(1 for j in range(3, WORD_BITS + 1)
            if f2_gcd(rots_to_poly((r, r+1, r+2)), (1 << j) ^ 1) > 1) == min_kc
        for r in range(WORD_BITS - 2)
    )
    print(f"""
5. OPTIMAL DESIGN:
   - Best triples minimize kernel deficit across all levels
   - Minimum kernel collisions (j>=3): {min_kc}
   - {min_count} triples achieve this minimum
   - Consecutive triples (r,r+1,r+2) all optimal: {consec_optimal}""")
