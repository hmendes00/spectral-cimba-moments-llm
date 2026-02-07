## Task: Fix CIMBA V-cache so it’s actually used, stays **O(S)**, and **does not break pretrained outputs**

### Context (do not debate)

* You previously added/rewired **CimbaMomentCache** and started using **Fourier reconstruction** for `attn @ V`, which causes **garbage output** on pretrained models.
* Before this change, **V bandlimiting / moment sketching “worked” in practice** (quality + benchmarks), even if cache plumbing wasn’t HF-native.
* **Goal now**: restore the previous working behavior **while making the cache real and HF-compatible**, and make **V memory constant O(S)** during generation.

### Non-negotiable goals

1. **Pretrained correctness gate**: With the default settings, generation must match the pretrained baseline behavior (no gibberish).
2. **CIMBA cache is actually used**: during `model.generate()`, each layer should use the CIMBA cache path without requiring hacks.
3. **V memory is O(S)**: **do not store full `V`** in the cache when `use_cimba_moments=True` (except in explicit debug mode).
4. **Keep K cache standard (O(N))**: we are only fixing V now.

---

## What to implement

### A) Split the cache into two modes (explicit flags)

Implement these flags on `CimbaMomentCache`:

* `compat_mode=True` (DEFAULT):

  * **Outputs MUST be identical to standard attention**
  * Cache still **updates and stores V moments incrementally**
  * But **attn_output uses standard V path computed from current `value_states` ONLY**, not from stored full V.
  * **Do NOT store full V history** (`v_cache_bytes` must remain 0).

* `moment_mode=True` (EXPERIMENTAL, off by default):

  * Uses the moment reconstruction path.
  * Allowed to be approximate / require finetuning.
  * Only enabled explicitly by the user.

In other words:

* **compat_mode=True** = “shadow moments” (track moments for constant-memory, but don’t use them to compute outputs yet).
* **moment_mode=True** = “use moments for output”.

### B) Make V moments truly O(S) without recomputation

Delete/avoid the current approach that:

* stores `self.v_cache[layer]` and
* recomputes moments from scratch using kv_len-based matrices.

Instead implement **incremental moment updates**:

For each new token value vector `v_t` at absolute position `pos_t`:

* Use a **fixed circle basis** with period `max_seq_len` (or `max_position_embeddings`), not kv_len.
* Maintain:

  * `V_moments[layer]` shape: `[batch, kv_heads, 2S+1, head_dim]` complex
* Update:

  * `V_k += exp(+i*k*theta(pos_t)) * v_t`
  * where `theta(pos) = 2π * pos / max_seq_len`

This avoids storing full V and avoids O(kv_len) recomputation.

### C) In compat_mode, compute output without storing full V

In `CimbaSpectralAttention.forward()`:

* If using the CIMBA cache and `compat_mode=True`:

  * do **not** call any `attend_with_moments()`
  * do **not** fetch `v_cache`
  * just do standard attention using the **current full `value` returned by cache.update()**, where `cache.update()` returns the concatenated **value_states for this step only if needed**.
  * But since we refuse storing full history, you must compute standard attention output using:

    * either the model’s normal DynamicCache for V (ONLY in compat_mode baseline), **or**
    * the simplest safe route: in compat_mode, **don’t replace V caching at all** (use HF DynamicCache for V), while still building the moment cache side-channel.

**Pick one and implement cleanly:**

* **Recommended**: keep HF’s normal V caching for compat_mode (so exactness holds), while moments are updated in parallel and can later replace V. This keeps output correct and makes your moment cache “real” without quality regression.

### D) Stop mixing bases

Right now the file mixes:

* moments computed with one basis (sometimes `max_seq_len`)
* coefficients computed with another (sometimes `kv_len` FFT / phase matrices)

For this task:

* **compat_mode path must not depend on coefficient/moment basis at all** (so it stays correct).
* In moment_mode path (later), use the **same fixed-circle basis** for:

  * moment accumulation
  * coefficient extraction (if/when you do it)

### E) Add 3 acceptance tests (must pass)

1. **HF generate sanity**:

   * Replace attention with CSA + CimbaMomentCache (compat_mode default).
   * Generate 256 tokens on a known prompt.
   * Output must be non-gibberish and similar to baseline (at least not repeating `def ... def ...` with `nodiscard` spam).
2. **Memory check**:

   * After generating N tokens:

     * `v_cache_bytes == 0` (or constant small debug-only)
     * `v_moments_bytes` stays constant w.r.t. N
     * `k_cache_bytes` grows with N (expected)
3. **Cache usage check**:

   * Ensure the model actually routes into CIMBA cache update for every layer (one log line per layer once per run is enough).

---

## “Don’t do these” (important)

* Do NOT try to “make Fourier reconstruction exact” right now.
* Do NOT recompute moments from scratch from stored full V.
* Do NOT change SpectralAttention math again.
* Do NOT add new operators (Dirichlet-only, IWM-style, etc.) unless asked.

---

## Deliverables

* Update `cimba_spectral_attention.py` with:

  1. A **CimbaMomentCache** that supports “shadow moments” + constant memory for V.
  2. `compat_mode=True` default path that preserves pretrained outputs.
  3. The three acceptance tests above.
