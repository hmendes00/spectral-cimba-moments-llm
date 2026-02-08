Below is a **copy/paste plan** for your dev AI to integrate **Softmax → FMMA moment distillation** (the KL-projection idea, implemented as **moment targets**) into your existing `distill_fmma.py`, without rewriting the whole training loop. This plan is designed to fit what you already have: **teacher/student logits KL + CE**, plus FMMA PSD/BW regularizers.

---

# PLAN: Integrate Softmax→FMMA Moment Distillation into `distill_fmma.py`

## Goal

Add an **auxiliary loss** that teaches FMMA’s `QueryMomentNetwork` to produce **phase moments** matching the **teacher softmax attention’s phase moments**:
[
m_k(t)=\sum_{i\le t} w^{\text{soft}}_{t,i},e^{ik(\theta_i-\theta_t)},\quad |k|\le S
]
This addresses the “softmax normalization barrier” by training the student to output the **bandlimited sufficient statistics** (moments) that FMMA can represent.

You will keep existing:

* logits KL distillation + CE (already implemented)
* PSD Toeplitz penalty + bandwidth regularization (already implemented)

We will add:

* `moment_loss` (new) computed from teacher attentions and student query moments.

---

## 0) Minimal design choices (keep it simple)

We are **NOT** solving the inner KL optimization in training. We implement the practical equivalent: **moment matching**. The paper covers why that is the correct target.

We’ll compute teacher moments on a **small set** of layers/heads and a **few query positions** per batch (for speed).

We’ll match these teacher moments to the student’s `QueryMomentNetwork` outputs (already stored as `_last_query_moments` in FMMAAttention during training). 

---

## 1) Add new config knobs in `DistillConfig` (distill_fmma.py)

Add these fields:

```py
# Moment distillation
alpha_mom: float = 0.05          # weight for moment loss (start small)
mom_every: int = 1               # compute moment loss every N steps
mom_layers: Optional[List[int]] = None  # None => sample a few random FMMA layers each step
mom_max_layers_per_step: int = 2
mom_head_stride: int = 2         # use every N heads (or sample heads)
mom_query_stride: int = 8        # use every N query positions
mom_max_queries: int = 16        # cap #query positions per batch
mom_detach_teacher_attn: bool = True
mom_use_fejer_weights: bool = True
```

Reason: you don’t want full attention tensors everywhere.

---

## 2) Make teacher return attentions (only when needed)

In the training loop, teacher forward is currently:

```py
teacher_outputs = teacher(input_ids=..., attention_mask=...)
```

Modify to optionally request attentions:

```py
teacher_outputs = teacher(
  input_ids=input_ids,
  attention_mask=attention_mask,
  output_attentions=(global_step % config.mom_every == 0),
  return_dict=True,
)
```

Important:

* This may increase memory; we’ll **subsample** layers/heads/queries after.

---

## 3) Build a helper to compute teacher phase moments efficiently

Add a helper function in `distill_fmma.py`:

### 3.1 Canonical phase mapping (must match FMMAAttention)

FMMAAttention uses:
[
\theta = 2\pi \cdot \text{pos} / \text{max_seq_len}
]
So in distill, define:

```py
def phases_from_positions(position_ids, max_seq_len):
    return 2 * math.pi * position_ids.float() / max_seq_len
```

Use integer positions `0..T-1` (or model position_ids if provided).

### 3.2 Teacher moments from attention probs

Teacher attention tensor per layer is usually:

* `[B, H, T, T]` (query, key)
  Call it `A = attn_probs`.

For chosen query indices `t_idx` (shape `[Q]`):

* Extract `A_q = A[:, :, t_idx, :]` → `[B, H, Q, T]`
* Build phase difference basis:

  * `theta = phases` → `[B, T]` (same across heads)
  * `theta_q = theta[:, t_idx]` → `[B, Q]`
  * `delta = theta[:, None, :] - theta_q[:, :, None]` → `[B, Q, T]` equals `(θ_i - θ_t)`
* For k = 1..S:

  * `E_k = exp(1j * k * delta)` → `[B, Q, T]`
  * `m_k = sum_i A_q * E_k` (sum over T) → `[B, H, Q]`
* Pack to `[B, H, Q, 2S+1]` with ordering `[-S..S]`:

  * `m_0 = 1` (or compute as sum_i A_q = 1)
  * `m_-k = conj(m_k)`

Optional: if `mom_use_fejer_weights`, multiply target moments by Fejér weights `c_k` (same weights FMMA uses in attention.py) to match the exact density reconstruction basis. 

---

## 4) Get student moments to supervise (from FMMAAttention)

Your FMMAAttention stores query moments during training:

* `module._last_query_moments` is set for the first chunk. 

### 4.1 Make it store the query moments for the same positions as teacher

Small change in FMMAAttention training path:

* add an optional argument `store_query_positions` (list of positions indices)
* when computing q_moments_chunk, extract only those positions and save them into `_last_query_moments` (instead of only first chunk), or save a dict keyed by absolute positions.

Minimal patch approach (fastest):

* In FMMAAttention._training_chunk, instead of only storing when `q_start == 0`, store if the chunk overlaps a set of requested positions.
* Save a tensor shaped `[B, H, Q, r, 2S+1]` for those positions.

This avoids mismatch where teacher moments are for positions across the sequence but student stored only first chunk.

---

## 5) Define the moment loss (practical + stable)

Teacher target moments: `m_teacher` → `[B, H, Q, 2S+1]` complex
Student predicted moments: `mu_student` → `[B, H, Q, r, 2S+1]` complex

We need to compare apples to apples.

### 5.1 Reduce channels for supervision (keep simple)

We do **distribution-shape** supervision, not channel decomposition.
Define effective student moment by averaging channels:

```py
mu_eff = mu_student.mean(dim=3)  # average over r → [B, H, Q, 2S+1]
```

(Alternative: learn a small channel-weight gate later; start with mean.)

### 5.2 Compare only positive k (or full vector)

Compute L2 over k=1..S (skip k=0 since it’s fixed):

```py
diff = mu_eff[..., S+1:] - m_teacher[..., S+1:]
moment_loss = (diff.real**2 + diff.imag**2).mean()
```

If using Fejér weighting, apply to both (or only to student moments consistently).

---

## 6) Integrate into the existing training loop

Right now you compute:

* teacher logits (no grad)
* student forward
* distillation_loss (KL + CE)
* psd_loss, bw_loss from `_last_query_moments`

Add after student forward:

1. If `global_step % config.mom_every == 0`:

   * choose query indices `t_idx` (example: last half tokens, stride, cap to mom_max_queries)
   * compute `m_teacher` from teacher attentions for chosen layers/heads and t_idx
   * gather `mu_student` for same layers/heads and t_idx from FMMA modules
   * compute `moment_loss`
2. Add to total loss:

```py
losses['moment_loss'] = moment_loss
losses['total_loss'] = losses['total_loss'] + config.alpha_mom * moment_loss
```

3. Log it (like psd/bw).

---

## 7) Layer/head subsampling strategy (so it’s fast)

Do NOT compute moments for every layer/head.

Recommended:

* Pick `mom_max_layers_per_step = 2`
* For those layers:

  * take heads with stride `mom_head_stride` (or random 4 heads)
* For queries:

  * take `t_idx = torch.arange(0, T, mom_query_stride)` then subsample to `mom_max_queries`

This keeps overhead manageable.

---

## 8) Validation checks (quick sanity)

Add these debug checks early:

* Ensure teacher attention rows sum to ~1 for selected t (after mask).
* Ensure `m_teacher[..., 0] ≈ 1` if you compute it.
* Ensure shapes align exactly across teacher/student for selected layers/heads/queries.
* Track `|m_teacher|` magnitudes: should be ≤ 1 generally.

---

## 9) Why this fits your current distill script

* You already do KL logits + CE (main behavior matching). 
* You already store `_last_query_moments` and apply PSD/BW penalties.
* This adds the missing “softmax normalization bridge” as an auxiliary, representation-aligned loss (moments) rather than token-wise weight matching.

---

## 10) Suggested schedule (so it doesn’t destabilize)

* Start with `alpha_mom = 0.01` for first 200–500 steps, then ramp to 0.05.
* Keep PSD penalty on (alpha_psd) and bandwidth reg on (alpha_bw).
* Keep teacher window length short (your max_seq_length is already 256–512). 

---

If you want, I can also give you a **minimal pseudocode patch** (exact insertion points in your current loop) using your variable names, but this plan should already be “dev AI ready” given it can read the paper + your files.
