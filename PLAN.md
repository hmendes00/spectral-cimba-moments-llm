I reviewed the updated `fmma_attention.py`. Overall it’s **much closer to “correct FMMA”** (causality fixed, canonical phase mapping, PSD autocorr option, mask propagation, and some memory-minded chunking/checkpointing). 

That said, there are **a few remaining correctness bugs + a couple of performance/oom footguns** I’d fix before we invest in a big test suite.

---

## ✅ What looks good now (big improvements)

* **Causality in cached path**: `_fmma_forward()` now does “update token t → query token t” instead of pre-updating the full chunk. 
* **Phase mapping**: `_get_phases()` always uses `max_seq_len`. 
* **Δ sign**: training uses `delta = θ_i − τ` (matches cache theorem). 
* **Masking**: padding mask is actually applied (both in cache update and training scores). 
* **PSD enforcement path exists**: autocorrelation PSD parameterization via FFT. 
* **GQA expansion**: uses `repeat_interleave` which is the right style (assuming divisibility). 
* **Training OOM mitigations**: chunked query processing + optional checkpointing. 

---

## 🚨 Must-fix correctness issues (before asking for tests)

### 1) **Bug: `if imag_mag > 1e-4:` will crash**

In `compute_attention()`:

```py
imag_mag = output.imag.abs().max()
if imag_mag > 1e-4:
    warnings.warn(...)
```

`imag_mag > 1e-4` returns a **0-d tensor**, and Python `if` can’t evaluate that as a bool. This will error during inference the first time it runs.

**Fix:** either:

* `if float(imag_mag) > 1e-4:` (sync), or better,
* gate behind a debug flag and only check every N steps, and do `imag_mag.detach().cpu().item()` rarely.



---

### 2) **Training “density clamp” changes the operator**

In `_training_chunk()` you do:

```py
densities = densities.clamp(min=0.0)
```

But in the **cached theorem path**, you do **no clamp** — you assume PSD ⇒ nonnegative. So training is learning with a different operator than inference if PSD ever fails (or numerically goes slightly negative).

**Fix options:**

* If PSD is truly hard-enforced, remove the clamp and instead assert / log violations.
* If you keep clamp as safety, you should also mirror it in inference (but that breaks “exactness” and can complicate stability).

Right now, clamp is a silent distribution shift risk. 

---

### 3) **4D `attention_mask` conversion is shaky**

You convert a 4D causal mask to a “simple mask” by:

```py
simple_mask = (attention_mask[:, 0, -1, :] > -1e4).long()
```

This is not equivalent to padding validity in general, and will be wrong for some HF mask patterns (and may mark tokens valid/invalid incorrectly).

**Fix:** prefer:

* if you want padding validity: derive from `position_ids` + known lengths, or keep the original 2D attention mask from HF (most models provide it).
* treat 4D masks as “attention constraints”, not token validity.



---

### 4) `_fmma_forward()` per-token Python loop will be slow on GPU

You fixed causality, but now inference does:

* loop over `t`
* update moments and query per token

That’s correct but can become a throughput killer.

**Better causal GPU approach (still exact):**

* process in **micro-chunks** and compute prefix-sums *within the chunk* (scan):

  * compute all `delta_M` for the chunk
  * compute cumulative moments `M_prefix[t]`
  * query each `t` from `M_prefix[t]`
    This keeps causality while removing Python-level per-token overhead.

Not required for correctness, but if you already care about GPU sync + OOM, you’ll care about throughput next. 

---

## ⚠️ Things that are “probably fine” but should be explicitly tested

### 5) PSD autocorrelation parameterization indexing

Your autocorr builds:

* `mu_positive = autocorr[..., 1:S+1]`
* `mu_negative = conj(flip(mu_positive))`
  This assumes the circular autocorr aligns so that negative lags correspond exactly to the flipped positive lags. It’s likely okay, but the “PSD autocorrelation test” must verify:
* Hermitian symmetry holds tightly
* Toeplitz PSD penalty is near zero
* density is nonnegative on a grid



### 6) KV head expansion assumes divisibility

`repeat_factor = num_heads // num_kv_heads` assumes exact divisibility. Should assert it. 

### 7) Output tuple shape for HF compatibility

Your `forward()` returns `(output, None)` even though many HF attention modules return 3 outputs (hidden, attn_weights, past_kv). If you’re using a custom integration path, fine; otherwise might break. 
