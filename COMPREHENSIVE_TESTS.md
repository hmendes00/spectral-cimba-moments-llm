Below is a **detailed unit test plan** for a single file: `test_fmma_comprehensive.py`.

---

# test_fmma_comprehensive.py - Comprehensive Unit Test Plan (FMMA)

## Goals

These tests must falsify/validate:

1. **Causality correctness** (no future leakage within a chunk and across generation steps).
2. **Training vs inference consistency** (same phase mapping and same (\Delta) convention).
3. **Moment theorem correctness** (cached moment formula matches explicit definition).
4. **Nonnegativity + normalization stability** (scores nonnegative, denominator positive, weights sum to 1).
5. **Masking correctness** (padding mask and causal mask behavior).
6. **HF cache integration sanity** (no KV growth, deterministic step-by-step generation).
7. **Numerical stability** (long streams, no NaNs/Inf, bounded denominator).
8. **Gradient flow sanity** (QueryMomentNetwork and feature maps get gradients).

---

## Test file structure (recommended)

```python
# test_fmma_comprehensive.py

import math
import pytest
import torch

# import your FMMA classes/functions:
# from fmma_attention import FMMAAttention, FMMACache, QueryMomentNetwork, replace_attention_with_fmma

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 1234
```

---

## Common helpers to implement (in test file)

### Helper 1: deterministic seed

```python
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Helper 2: build small FMMA module

Create a minimal FMMA head config for tests (or what you use as default):

* `hidden_size=32`
* `num_heads=4`
* `head_dim=8`
* `S=4` or `S=8`
* `r=4` or `r=8`
* `max_seq_len=64`

### Helper 3: explicit FMMA weights from definition (slow path)

Given:

* queries `q_t`
* keys `k_i`
* values `v_i`
* phases `theta_i`, query phase `tau`
* `phi_m(k)` and `p_m(delta|q_t)` from your module

Compute:

* `s_i = sum_m phi_m(k_i) * p_m(theta_i - tau | q_t)`
* apply causal mask `i<=t` if necessary
* `w_i = s_i / sum_j s_j`
* `y_t = sum_i w_i * v_i`

This should match cached theorem output within tolerance.

### Helper 4: cached theorem output (fast path)

Given cache moments:

* `M_V[m,k]`, `M_1[m,k]`
  Compute:
* numerator and denominator via theorem formula
  Return `y_t`

---

## Test Group A: Theorem Exactness (must pass before anything else)

### A1) Moment theorem equality vs explicit definition (single step)

**Purpose:** Validate core math: cached formula equals explicit sum.

**Setup:**

* `B=1`, `L=10`, `S=4`, `r=4`, `d_v=8`
* Random keys/values.
* Random phases `theta_i` in `[0, 2π)`.
* Pick a single query time `t=7`.

**Procedure:**

1. Compute explicit output `y_explicit` using definition.
2. Build cache moments from tokens `i<=t`.
3. Compute `y_cached` using theorem ratio.
4. Assert `max_abs(y_explicit - y_cached) < 1e-5` (fp32), tighter in fp64.

**Failure signals:**

* Δ sign mismatch
* indexing mismatch for k bins
* Hermitian mismatch / complex errors
* future leakage

---

### A2) Moment theorem equality for all t in a sequence (streamed)

**Purpose:** Catch causality bug within chunk.

**Procedure:**
For each `t in [0..L-1]`:

* cache built only from `i<=t`
* compare `y_cached(t)` to explicit `y_explicit(t)`

**Assert:**

* all t match within tolerance

This test will fail immediately if implementation pre-updates cache with the full chunk.

---

### A3) Multi-batch theorem equality

Same as A2 but with `B=2` and random lengths. Confirms batch indexing.

---

## Test Group B: Causality / No Future Leakage

### B1) Chunk leakage detector (critical)

**Purpose:** Detect your current bug: cache updated with all tokens before per-t outputs.

**Setup:**

* Create `x` for `L=16`
* Create a modified copy `x2` where only future tokens `i>t0` are changed (randomized).
* Choose `t0=6`.

**Procedure:**

1. Run FMMA forward on `x` and record `y(t0)`.
2. Run FMMA forward on `x2` and record `y2(t0)`.

**Expected:**

* `y(t0)` MUST equal `y2(t0)` within tolerance.

If `y(t0)` changes, you are leaking future tokens.

---

### B2) Generation-step causality (incremental updates)

**Purpose:** Ensure step-by-step inference equals full forward with causal definition.

**Procedure:**

1. Stream tokens one by one:

   * update cache with token i
   * compute output at step i
2. Separately, compute outputs with explicit definition for each i.
3. Compare all steps.

---

## Test Group C: Phase mapping consistency

### C1) Training vs inference mapping equivalence (must enforce same mapping)

**Purpose:** catch denominator mismatch (`seq_len` vs `max_seq_len`).

**Setup:**

* Use a fixed `position_ids=[0..L-1]`, with `L < max_seq_len`
* Compute phases using training path and inference path.

**Assert:**

* `max_abs(theta_train - theta_infer) == 0` (or < 1e-8)

If not equal, hard fail with message:
“Phase mapping differs between train and infer. Fix to a single canonical mapping.”

---

### C2) Δ sign consistency test

**Purpose:** ensure training delta and inference delta match.

**Procedure:**

* Use a known single frequency `k=1`
* Construct a scenario where sin term matters:

  * Choose `mu_1` purely imaginary or choose phases such that sin contribution nonzero.
* Compute density in training code path and compare to the density implied by cached formula.

**Assert:**

* densities match within tolerance.
  If signs flipped, you’ll see systematic mismatch.

---

## Test Group D: Nonnegativity, normalization, PSD constraints

### D1) Nonnegativity of densities (when PSD enforced)

**Purpose:** If PSD Toeplitz is enforced (or penalty is strong), Fejér density should be ≥ 0.

**Procedure:**

* Generate μ using your enforced parameterization.
* Evaluate `p_m(Δ)` on a grid of `Δ` values (e.g., 256 points).
* Assert `min(p_m) >= -1e-6` (allow tiny numerical noise).

If this fails, PSD isn’t enforced or reconstruction differs.

---

### D2) Nonnegativity of attention scores and weight normalization

**Procedure:**

* For random inputs, compute `attn_scores` and `attn_weights`.
* Assert:

  * `attn_scores.min() >= -1e-6` (or exactly >=0 if strict)
  * `attn_weights.min() >= -1e-6`
  * `abs(attn_weights.sum(dim=-1) - 1).max() < 1e-5`
* Assert denominator `Den` is > `1e-8`.

---

### D3) Hermitian symmetry / real output check

**Procedure:**

* After computing complex numerator:

  * assert `imag.abs().max() < 1e-5`
* If you intentionally discard imaginary part, require it to be tiny.

This catches silent complex drift.

---

## Test Group E: Masking (padding + causal)

### E1) Padding mask must exclude padded tokens

**Setup:**

* Batch with padding: example lengths `[8, 5]` padded to `L=8`.
* Build `attention_mask` where padded positions are 0.

**Procedure:**

* Compare outputs from:

  1. full padded batch
  2. same batch but explicitly remove padded tokens (shorter L) and run again
* Assert outputs on real positions match.

This test forces you to implement mask handling in FMMA update/scoring.

---

### E2) Causal mask correctness on training path

**Procedure:**

* For training forward, verify that token t does not attend to i>t.
* Compare `attn_weights[..., i>t]` are zero.

---

## Test Group F: HF cache integration sanity (minimal)

### F1) Cache length tracking

**Purpose:** Ensure `past_key_value.get_seq_length()` equals number of streamed tokens and does not depend on allocated KV.

**Procedure:**

* Create a cache, step through updates for N tokens.
* Assert seq length increments by 1 each step.

---

### F2) No KV growth (if intended)

If you intend no KV caching:

* After many steps, assert underlying stored K/V tensors are empty or constant size.
* If HF requires dummy tensors, ensure size does not grow with N.

---

## Test Group G: Numerical stability

### G1) Long stream stability (e.g., 50k tokens in test, not 1M)

**Purpose:** Catch NaNs, denom collapse, drift.

**Procedure:**

* Stream `N=50000` tokens with random keys/values.
* Track:

  * `Den` min/mean/max
  * `output` finite checks
* Assert:

  * no NaNs/Inf
  * `Den.min() > 1e-8`
  * `output.abs().max()` is finite and not exploding (set a reasonable bound like `< 1e3`)

Mark as `@pytest.mark.slow`.

---

### G2) Extreme input stability

* Very large key magnitudes
* Very large query magnitudes
* Check that feature maps don’t overflow (if using exp, must stabilize with max subtraction / clamp).

Assert no NaNs/Inf.

---

## Test Group H: Gradient flow

### H1) Gradients exist for moment network parameters

**Procedure:**

* Forward with random inputs
* Loss = `output.sum()`
* Backward
* Assert:

  * some parameters in QueryMomentNetwork have `grad` not None
  * grad norms are nonzero

---

### H2) Gradients for feature projection params

Same as H1 but for parameters controlling `phi_m(k)`.

---

## Mandatory fixtures / parametrization

### Parameter sweep

Use `pytest.mark.parametrize` over:

* `S in [2, 4, 8]`
* `r in [2, 4, 8]`
* `L in [8, 16, 32]`

But keep slow tests separated.

---

## Required tolerances (guideline)

* fp32: `atol=1e-5`, `rtol=1e-4`
* fp64 (optional): `atol=1e-10`, `rtol=1e-8`

---

## Expected “initial failures” (good sign)

Given your current code issues, these will likely fail until fixed:

* **B1 (chunk leakage)** should fail immediately
* **C1 (phase mapping)** should fail if you keep train denom = seq_len
* **C2 / A1 (Δ sign)** may fail depending on sin handling
* **E1 (padding mask)** should fail because it’s not implemented

---

## Deliverable checklist for dev AI

The dev AI must implement:

* `explicit_fmma_output()` helper
* `cached_fmma_output()` helper
* `streaming_cache_runner()` (update then query)
* all tests A–H with clear error messages
