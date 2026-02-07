Below is a **copy-paste friendly .md** of the **inconsistencies / correctness hazards** I can flag in your current `fmma_attention.py`. 

---

# FMMA Attention - Inconsistencies / Bugs / Spec Mismatches

Source: `fmma_attention.py` 

## 0) Biggest blockers (must fix before training is meaningful)

### 0.1 Causality is violated in cached inference path (chunk update then per-t read)

In `_fmma_forward()`, you do:

1. `cache.update_moments(... seq_len tokens ...)` (adds **all tokens in the chunk**)
2. then loop `for t in range(seq_len): compute_attention(...)`

That means when computing output for early `t`, the cache already contains contributions from future tokens `t+1..end` in the same forward call. This breaks causal attention and invalidates any “causal masking” assumptions.

**Fix:** stream token-by-token:

* update cache with token `t`
* compute output for token `t`
  (or maintain separate “prefix cache” states within the chunk).

---

### 0.2 Training phase definition differs from inference phase definition (distribution shift)

`_get_phases()` uses:

* **training**: `2π * positions / seq_len` (when `seq_len < max_seq_len`)
* **inference/cache**: `2π * positions / max_seq_len`

So the operator you train is not the operator you deploy. The learned harmonic structure is tied to a different geometry.

**Fix options:**

* Always use `max_seq_len` mapping in both training and inference (recommended).
* Or train with the same mapping schedule you’ll use at inference (progressive max_seq_len curriculum), but **never change the mapping silently**.

---

### 0.3 Training Δ sign likely inconsistent with cache Δ sign

Theory/cache path implements terms consistent with:

* cache uses (e^{ik\theta_i}) in moments and multiplies by (e^{-ik\tau}) at query time → depends on ((\theta_i - \tau))

In training path (`_fmma_training_forward`):

```python
delta = phases_q.unsqueeze(2) - phases.unsqueeze(1)  # τ - θ_i
k_delta = delta * k
cos_basis = cos(k_delta)
sin_basis = sin(k_delta)
densities = mu_0 + 2*(cos_contrib - sin_contrib)
```

This corresponds to ((\tau - \theta_i)) (opposite sign). Cos is even but sin flips sign, so your imaginary part contribution changes sign unless you compensate.

**Fix:** make the training Δ match inference:

* use `delta = phases.unsqueeze(1) - phases_q.unsqueeze(2)` (θ_i − τ), **or**
* change the density formula accordingly (flip the sign of the sin term).

---

## 1) FMMA math vs implementation inconsistencies

### 1.1 You pre-weight moments with Fejér weights, but training density formula assumes exact harmonic reconstruction

You do:

```python
weighted = q_moments_complex * c_k
mu_0_real = weighted[..., S].real
mu_pos_real/imag = weighted[..., S+1:].real/imag
densities = mu_0 + 2*(cos - sin)
```

This is okay **only if** you intend the kernel to be exactly the Fejér-smoothed reconstruction. But then:

* you must ensure the **same Fejér reconstruction** is used in cached inference (it is, via `combined = c_k * μ_k * e^{-ikτ}`).

✅ Good intention
⚠️ But this becomes inconsistent if you ever train with one mapping (scaled by seq_len) and infer with another (max_seq_len). See 0.2.

---

### 1.2 “Nonnegative weights” is not guaranteed by current `QueryMomentNetwork`

You claim PSD Toeplitz + Fejér gives nonnegativity, but the network does not actually enforce PSD Toeplitz.

Current enforcement:

```python
scaled_mag = 0.95 * sigmoid(|mu|)
mu_positive = scaled_mag * exp(i phase)
```

This only enforces (|\mu_k|<1) (necessary, not sufficient). Toeplitz PSD can still fail, and then Fejér reconstruction can be negative.

You have a `psd_toeplitz_penalty()` function, but it’s not integrated into the module forward/loss by default.

**Fix:**

* either hard-parameterize PSD Toeplitz moments (preferred; see below), or
* always add PSD penalty to training loss and monitor min density / min score.

**Hard parameterization idea (more “correct”):**
Parameterize a PSD Toeplitz matrix via Cholesky factor (T = LL^*), then read moments from its first column (normalized). This enforces PSD exactly.

---

### 1.3 Cache attention returns `output.real` (silently discards imaginary leakage)

`compute_attention()` returns:

```python
return output.real.to(self.dtype)
```

If Hermitian symmetry is slightly violated (which can happen), imaginary parts can appear and you’re discarding them silently. That can hide major issues in training/inference consistency.

**Fix:**

* assert imaginary magnitude is small during debug (`imag.abs().max()`), or
* explicitly symmetrize moments so the result is provably real.

---

## 2) Cache mechanics & HF integration issues

### 2.1 `FMMACache.update()` doesn’t actually satisfy HF cache expectations

You implement:

```python
def update(...):
    return key_states, value_states
```

But comments say “Return empty tensors - actual attention uses moments”. Yet you return the input states unchanged. Depending on HF internals, this may:

* still allocate standard KV cache upstream,
* break cache accounting,
* or cause incorrect generation behavior.

**Fix:** implement the proper HF `DynamicCache` contract for “no KV storage”.

* If HF expects `update()` to return stored KV, you likely need to return dummy tensors of shape (0) or intercept caching differently.
* Also ensure `past_key_value.get_seq_length()` used in forward returns a correct length across generation steps.

---

### 2.2 In `_fmma_forward`, you convert phases back to positions using a float transform

```python
positions = phases * self.max_seq_len / (2 * math.pi)
```

But `phases` may have been computed using different denominator (`seq_len` in training vs `max_seq_len` in inference). Also this produces floats, not integer positions, which then re-enters:

```python
phases = 2π * positions / max_seq_len
```

This can drift numerically and breaks the guarantee that the same (\theta) is used for moment update and query.

**Fix:**

* pass integer `position_ids` to cache update, not phases.
* keep one canonical mapping `theta = f(position_id)` everywhere.

---

### 2.3 Padding / attention_mask is ignored in FMMA paths

In FMMA forward paths you never apply `attention_mask` (padding, packed sequences, etc.). Standard attention uses it.

This can cause:

* training instability on padded batches,
* incorrect outputs vs teacher during distillation.

**Fix:** incorporate masks by zeroing contributions from padded tokens:

* in training: mask out keys/values in `key_feats` / `values` before accumulation
* in cache: do not update moments for padded positions

---

## 3) Training path issues (accuracy, stability, complexity)

### 3.1 Training forward does O(L²) work (still too heavy for long sequences)

`_fmma_training_forward` computes per chunk:

* delta: `[B, chunk, L]`
* cos/sin basis: `[B, chunk, L, S]`

This is still quadratic in L. It’s okay for short distillation, but it’s not aligned with the “constant-memory” story for training long contexts.

**Fix:** for long-context training, you should train using the same streaming cache mechanism (but with correct causality), possibly with truncated BPTT or stop-grad through caches.

---

### 3.2 Causal masking uses hard zeros, not `-inf`, which changes gradients

You do:

```python
attn_scores = attn_scores.masked_fill(causal_mask, 0.0)
attn_weights = attn_scores / (sum + 1e-8)
```

This is not equivalent to masking logits in softmax. Here, masked positions contribute exactly 0, but the normalization is a linear normalization, not exponential. That might be intended (FMMA uses nonnegative scores), but then you must ensure:

* all unmasked scores are nonnegative, otherwise weights can become negative.

**Fix:** if you keep linear normalization, ensure `attn_scores >= 0` always (requires PSD enforcement + nonnegative features + density positivity).

---

### 3.3 Density can go negative in training (because PSD not guaranteed)

Even if `key_feats` is nonnegative, if `densities` goes negative you will get negative `attn_scores`, and then linear normalization yields negative “weights” (invalid probability).

**Fix:** either enforce PSD moments hard, or clamp density at 0 (but clamping breaks exactness and gradients), or add a positivity barrier loss + monitoring.

---

## 4) Replacement logic & architecture mismatches

### 4.1 `replace_attention_with_fmma()` module matching is fragile

You replace any module with `'attn' in name.lower()` and `hasattr(module, 'q_proj')`. This can catch unintended modules or miss model-specific patterns.

**Fix:** target known module classes for the specific architecture (Qwen/Llama/Mistral attention class), or match by exact attribute set and expected shapes.

---

### 4.2 GQA handling may be wrong if `num_heads // num_kv_heads` is not integer

You do:

```python
repeat_factor = num_heads // num_kv_heads
k_weight = k_weight.repeat(repeat_factor, 1)
```

This assumes divisibility and does not handle interleaving patterns used by some architectures.

**Fix:** implement the exact head mapping used by the model (often “repeat_interleave” on head dimension with known grouping).

---

## 5) Spec mismatches vs “FMMA theorem” requirements

### 5.1 The theorem expects per-token update then query evaluation (causal)

Your cached path currently violates the theorem’s causal streaming assumption (0.1). The theorem’s whole point is that you can compute:
[
y_t = \frac{\sum_{m,k} c_k \mu_{m,k}(q_t) e^{-ik\tau} M^{(m,V)}*k(t)}
{\sum*{m,k} c_k \mu_{m,k}(q_t) e^{-ik\tau} M^{(m,1)}_k(t)}
]
where (M(t)) is built from tokens (\le t). Your implementation currently computes (M) from the full chunk before each t.

---

### 5.2 “RoPE / circular phases” claim is not implemented

You don’t use RoPE or actual multi-dim torus phases. You use:

* a single scalar phase mapped from token index around a circle.

If your plan is “compatibility with RoPE”, you’ll eventually need:

* per head and per dimension phase vectors (torus),
* multi-index moments (or separable product structure).

---

## 6) Concrete checklist for the dev AI (what to change first)

### Priority A (correctness)

* [ ] Fix causality in cached inference: update cache incrementally per token.
* [ ] Make phase mapping consistent between training and inference (choose one).
* [ ] Make Δ sign consistent between training and inference.
* [ ] Integrate padding masks into FMMA update and scoring.
* [ ] Stop silently discarding imaginary parts; add assertions/diagnostics.

### Priority B (math guarantees)

* [ ] Replace “|μ_k|<1” heuristic with a real PSD Toeplitz parameterization (Cholesky).
* [ ] Or: enforce PSD with a strong penalty + monitor min density and min score.

### Priority C (engineering)

* [ ] Fix `DynamicCache.update()` behavior so HF doesn’t allocate KV and generation works.
* [ ] Improve module replacement targeting for the specific transformer architecture.
* [ ] Validate GQA head mapping rigorously.
