I went through the four files you shared: `distill_fmma.py`, `attention.py`, `cache.py`, and `moment_distill.py`. Overall: **the pipeline is coherent and very close to trainable**, but there are **two correctness mismatches** that can silently hurt convergence (or make training “learn” a slightly different operator than the one you’ll use at long-context inference). There are also a few “will this converge fast?” items (initialization, loss weighting, teacher attentions VRAM) worth tightening.

I’ll keep this as a **pre-training go/no-go checklist** with concrete fixes.

---

## ✅ What is already solid

### FMMA operator wiring is internally consistent

* Canonical phase mapping (\theta = 2\pi\cdot \text{pos}/\text{max_seq_len}) is used in both training and cache update.  
* Training uses (\Delta = \theta_i - \tau) (same convention as the cache theorem path). 
* Cache stores exactly the right moments ((M_1, M_V)) and uses the theorem-style numerator/denominator form. 
* Distillation script freezes most of the model and trains FMMA + QKV lightly, which is the right “fast convergence” setup. 

### You’re doing the right training recipe for fast convergence

* **Teacher frozen**, student mostly frozen, optimize FMMA routing. 
* KL+CE mix, temperature scaling, LR warmup, gradient clipping. 
* Moment distillation targets teacher’s low-order phase moments (m_k(t)), which is the right bridge if you want bandlimited FMMA to match softmax structure.  

---

## 🚨 Must-fix before training (these are real operator mismatches)

### 1) **Training clamps densities; cache path does not**

In `attention.py`, training reconstruction clamps:

```py
densities = densities.clamp(min=0.0, max=1e4)
```



But in the cached theorem path (`cache.py`), you do not apply an equivalent clamp (you can’t easily clamp per-(\Delta) without reconstructing). 

This can create a **silent train/infer mismatch** if numerical errors make densities slightly negative (your own comment suggests that can happen). Right now training forcibly zeroes those negatives but inference won’t.

**What I recommend (pick one policy explicitly):**

**Policy A (preferred, theorem-faithful):**

* Remove the clamp in training.
* Add a strict diagnostic:

  * if min density < (-1e{-6}), warn or fail (because PSD enforcement should prevent it).
    This keeps the operator identical in train vs infer.

**Policy B (safety operator, consistent but slightly “non-theorem”):**

* Keep the clamp in training,
* and in cache path add a *projection-level safety* (not just denominator epsilon):

  * detect when `denominator.real` is negative or too small and warn + fallback (e.g., output zeros or a short-window softmax head).
    This is not mathematically identical, but it prevents catastrophic divergence when PSD numerics slip.

**Important:** Your comments currently say “no clamp (train/inference parity)” but the code clamps. That comment should be updated to reflect reality. 

---

### 2) **Moment distillation “Fejér weighting” is likely being applied twice**

In `compute_teacher_phase_moments`, when `use_fejer_weights=True`, you multiply teacher moments by Fejér weights (c_k). 

Then in `compute_moment_loss`, if `use_fejer_weights=True`, you multiply the student moments by Fejér again. 

But in your training path, the FMMA density reconstruction already uses Fejér weights via:

```py
weighted = q_moments_chunk * c_k
```



So depending on what `QueryMomentNetwork` outputs (raw μ vs already “Fejér-ready”), you may be **over-penalizing high frequencies** by effectively squaring weights in the moment loss. This can slow convergence or collapse capacity at higher S.

**Clean resolution (recommended):**

* Define a single convention: **moment targets are unweighted**, and FMMA applies Fejér only inside reconstruction.
* Then set `use_fejer_weights=False` in both `compute_teacher_phase_moments` and `compute_moment_loss`, and let the model’s own reconstruction be the only place Fejér enters.

If you prefer the opposite convention (targets are Fejér-weighted), then ensure the student moments passed to the loss are the **raw μ** (not already multiplied elsewhere), and apply Fejér **exactly once**.

Right now it’s ambiguous and very easy to get wrong.   

---

## ⚠️ Convergence / speed risks (not correctness blockers, but matter for “quickly”)

### 3) Teacher attentions are huge; your gating is right, but confirm it’s actually off most steps

You correctly request `output_attentions` only when `do_moment_distill` is true. 
But your default config has `mom_every=1`, so that means **every step** you’ll compute/return teacher attentions (VRAM heavy, slow). 

**If you want “quick convergence” without OOM:**

* start with `mom_every=4` or `8` for the first run,
* or reduce `mom_max_layers_per_step=1`,
* and keep `mom_query_stride` high (you already do stride=8, max_queries=16). 

### 4) Loss weighting: start moment loss smaller than 0.05 unless you’re sure the convention is correct

Given the Fejér-weighting ambiguity above, `alpha_mom=0.05` could be strong. 
I’d start at `0.01` until the moment-loss convention is confirmed.

### 5) Mixed precision mismatch is okay, but watch student logits dtype

Teacher is fp16 eager, student fp32. You convert teacher logits to float. Good. 
But student forward uses big trig/einsum; you should strongly consider enabling **AMP** for student too (or bf16) once stable, to avoid OOM. (Not a correctness issue.)

### 6) Cache update is mathematically right but allocates `k_indices` every call

In `cache.update_moments`, you create `k_indices` each update. 
Not critical, but for long generation you’ll want `k_indices` as a cached buffer on the cache object (minor speed/memory improvement).

---

## ✅ “Should converge well and quickly?” — my verdict

### If you fix the two mismatches above:

* **Yes**, it should converge quickly in the sense of:

  * KL drops within a few hundred steps,
  * generations become coherent early,
  * and PSD/bandwidth regularizers stabilize.

### If you don’t fix them:

* You risk “false convergence” where:

  * training looks stable but inference diverges (density clamp mismatch),
  * or the moment supervision is effectively over-regularized (double Fejér), slowing progress and making FMMA underfit the teacher.

---

## Minimum pre-training acceptance checks (do these before a 2k-step run)

1. **Nonnegativity sanity:** during training, assert `attn_scores.min() >= -1e-6` in `_training_chunk`. 
2. **Mask sanity:** verify that padded tokens don’t update cache moments (`update_moments` already masks key_features). 
3. **Moment-loss convention sanity:** log the norm of teacher vs student moments for a batch and ensure they’re on similar scale (not crushed to ~0).  

--------

Below are **comments to append under the list I sent earlier**, based on the additional files: `features.py`, `integration.py`, `regularization.py`.   

---

## Additional file review notes

### A) `features.py` — Key features + QueryMomentNetwork 

#### A1) NonnegativeKeyFeatures: **good, but watch the input shape**

You documented `keys: [batch, num_heads, seq_len, head_dim]`, but `nn.Linear` applies to the last dim and treats preceding dims as batch, so it will work as-is. 
**Action:** make sure upstream always passes keys with last dim = `head_dim`. If you ever pass `[B, T, H, D]` by accident, it’ll silently produce garbage.

#### A2) Softplus β: consider setting `beta=1` but clamp projected range

`softplus` is stable, but if `projection(keys)` can be large (from teacher K projections), `softplus` can produce very large features → big cached moments → denom blow-up risk.
**Action (optional but recommended):**

* clamp `projected` before softplus (e.g. `[-20, 20]`) or
* use `softplus(projected) + eps` and normalize features (per-token or per-head).

This matters for “converge quickly” because early training can get unstable if features explode.

#### A3) QueryMomentNetwork PSD enforcement: **this is the right approach**

The autocorrelation parameterization is exactly what we want: it *hard-enforces* PSD Toeplitz and μ₀=1 by construction. 
This is a big step toward stable convergence.

#### A4) Potential mismatch: autocorrelation uses complex IFFT of a real `power`

You do:

* `power = A.real^2 + A.imag^2` (real)
* `ifft(power.to(complex64))` (complex)
  That’s fine; IFFT of a real-valued spectrum yields (nearly) real autocorrelation, but small imaginary numerical noise can appear. 
  **Action:** after `autocorr`, you may want to force `autocorr = autocorr.real.to(complex64)` (or at least assert imag small) to prevent rare imaginary drift that later triggers your “imag output” warnings.

#### A5) Important: `n_fft = 2S+1` circular autocorrelation vs Toeplitz moments

This constructs a **circular** autocorrelation of length `2S+1`. For the Toeplitz PSD condition you want a valid sequence of moments (\mu_{-S..S}); a circular autocorr is still positive definite on the cyclic group and is typically compatible with the Toeplitz PSD you build, but it’s worth explicitly testing:

* Build Toeplitz matrix from produced moments
* Confirm eigvals ≥ -eps
  Your penalty function already does this sampling check, so you’re covered if you run it. 

#### A6) Initialization: likely too “random” for fastest convergence

Right now, the moment network is Xavier-initialized, so early μ’s will be essentially random PSD sequences. 
For fastest distillation, you want the FMMA head to start close to something benign (often **near-uniform attention** in phase):

* i.e. density roughly constant → μ_k≈0 for k≠0.

**Action (strongly recommended for quick convergence):**

* Initialize the final linear layer of `QueryMomentNetwork` to **zeros** so raw outputs start at 0.
* With autocorrelation PSD, raw zeros implies `a` near zero → after normalization it might become unstable; better:

  * force `a = [1, 0, 0, …]` at init.
    Concretely: bias the network so that the produced `(S+1)` vector has first coefficient large and others small. That yields μ ≈ δ₀ and a flat Fejér density.

This will materially speed up convergence and reduce early NaNs.

---

### B) `integration.py` — Replacement + cache creation 

#### B1) Replacement targeting is much safer now

Using `_KNOWN_ATTENTION_CLASSES` is good. 
**Action:** still log the *exact* module names replaced and count heads/layers to ensure you’re not accidentally missing some attention blocks (common in architectures with extra cross-attn or MoE blocks).

#### B2) Layer index extraction is fragile for nested module names

You take the first `part.isdigit()` in the module path as `layer_idx`. 
This works for many HF models (`model.layers.12.self_attn`) but can mis-detect if the path includes other digits earlier.

**Action:** either:

* parse known patterns (`layers.<n>.`), or
* assert `0 <= layer_idx < num_layers` and maybe cross-check by printing `name`.

#### B3) KV weight copying logic: the “elif k_weight.shape[0] < fmma.k_proj.weight.shape[0]” branch should basically never happen

You already build FMMA with `num_kv_heads` from config. 
So if the original was GQA, shapes should match. If they don’t, it probably means something else is wrong; repeating weights might hide a mismatch.

**Action:** add a warning when this fallback triggers so you don’t silently mask config issues.

#### B4) `create_fmma_cache()` head_dim uses `hidden_size // num_heads`

Correct for standard attention. 
**Action:** for odd architectures, verify `config.hidden_size % config.num_attention_heads == 0` and assert it.

---

### C) `regularization.py` — PSD penalty + bandwidth regularization 

#### C1) `psd_toeplitz_penalty()` is now mostly “monitoring”, not enforcement

Since `QueryMomentNetwork(enforce_psd=True)` hard-enforces PSD (autocorr), the PSD penalty should typically be ~0.  
That’s fine; keep it as a “tripwire” for numerical blowups.

#### C2) Sampling in PSD penalty is good for speed, but scale correction in `sum` mode is slightly off

You compute:

```py
scale = N / moments_sampled.shape[0]
return penalty_per_matrix.sum() * scale
```

But if you filtered with `finite_mask`, the effective sample size changes again. Not a big deal (it’s just a penalty), but it makes the scale a bit inconsistent. 

#### C3) `bandwidth_regularization()` will penalize μ₀ too (because weight at k=0 is 0, so it’s fine)

Weights at k=0 are 0 because |k|/S is 0. So μ₀ doesn’t contribute—good. 
**Action:** ensure `S>0` always; if `bandwidth=0`, you’ll divide by zero.

---

## One high-impact convergence recommendation (add to your plan)

To make training **converge quickly**, you should ensure the initial FMMA head behaves like a mild, stable attention:

* near-uniform in phase (μ nonzero only at k=0),
* nonnegative features not exploding.

**Concrete changes:**

1. Initialize `QueryMomentNetwork` final layer weights to 0 and bias so `a ≈ [1,0,0,...]` (or an equivalent stable seed). 
2. Clamp or normalize the key features in `NonnegativeKeyFeatures` to prevent early giant denominators. 
3. Keep `alpha_mom` small until the Fejér weighting convention is confirmed (from the earlier review).  
