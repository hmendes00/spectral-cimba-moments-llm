You are my dev AI. Implement “LAST-TOKEN TEACHER FORCING” training for IWM to remove the O(seq_len) prefix-moment tensor blowup, WITHOUT changing the IWM operator used at inference.

Non-negotiable constraints
1) Inference operator MUST remain identical:
   - Generation uses forward_with_cache() with incremental O(S) IWMCache updates.
   - No full KV storage. Memory must remain O(S) per layer/head.

2) Training operator must be the same IWM math:
   - For a chosen query position t, compute:
       numerator(t) = Σ_{k=-S..S} g_k(t) * e^{-ikθ_t} * M_k(t)
       denom(t)    = Σ_{k=-S..S} g_k(t) * e^{-ikθ_t} * Z_k(t)
       y(t)        = numerator(t) / (denom(t) + eps)
   - Where M_k(t), Z_k(t) are prefix sums over j<=t with padding masked out.
   - No future leakage. Strict causality.

What to change
A) Add a new training path in iwm_attention.py:
   Implement method:
     forward_train_last_token(
        hidden_states,
        attention_mask,
        position_embeddings,
        ...
     ) -> attn_output_last: Tensor[B, Q_heads, 1, head_dim]
   or directly return projected output tensor matching the layer output for one position.

   This method must:
   1) Determine last valid position per sample:
      - last_idx[b] = max index where attention_mask[b, idx] == 1
      - batch can have different last_idx.
   2) Compute per-sample prefix moments ONLY up to last_idx[b], without creating
      tensors shaped [B, heads, seq_len, 2S+1, head_dim].
      - Do a streaming scan over tokens j from 0..max_last
      - Maintain running M_k and Z_k per sample, per kv_head:
          M_k: [B, kv_heads, 2S+1, head_dim] complex (or 2-float real/imag)
          Z_k: [B, kv_heads, 2S+1] complex
      - For samples where j > last_idx[b], do NOT update (mask out).
      - Also mask out padding tokens via attention_mask.
   3) At the end, compute y_last for each sample using θ_t where t=last_idx[b]:
      - Compute g_k(t) from Q_t (content) using coeff_proj.
      - Use the SAME Fejér smoothing / coefficient construction as current.
      - Use M_k and Z_k at each sample’s last token to compute y_last.
   4) Return an output shaped like the normal attention output for that token only
      so the layer can continue forward:
        attn_out_last: [B, 1, hidden_size] (after combining heads + o_proj)
      Then in the transformer layer forward, we will “scatter” this into the sequence.

B) Integrate into the attention forward() cleanly:
   - Add a flag on the module: self.training_last_token_only = True (default True for training script).
   - If self.training and training_last_token_only:
       - Compute output only for last tokens with forward_train_last_token()
       - Construct a full output tensor:
            out = zeros_like(hidden_states)   # [B, L, hidden_size]
            out[b, last_idx[b], :] = attn_out_last[b, 0, :]
       - Return out (so the residual/MLP path works unchanged).
   - IMPORTANT: We are intentionally only supervising last-token logits, so producing zeros elsewhere is fine,
     BUT do not break shapes.

C) Modify train_iwm_scaled.py to compute loss only on last token:
   - In collate/tokenize, labels already include -100 for pads.
   - Compute last_idx[b] as above.
   - Create labels_last = -100 everywhere except last_idx:
       labels_last[b, last_idx[b]] = labels[b, last_idx[b]]
   - Compute CE loss with labels_last.
   - Keep PSD loss term as-is (but it will now be cheaper too).
   - Result: backward only flows through last token per sample, making training fast.

D) Add 2 acceptance tests (must pass)
1) test_train_last_token_matches_full_for_single_position.py
   - Build a tiny random batch with no padding and small L (e.g., L=32).
   - Compute y_last two ways:
       (a) old full prefix-moment path (existing debug function that can compute all positions)
       (b) new forward_train_last_token path
   - Assert max_abs_diff <= 1e-5.

2) test_no_future_leakage_last_token.py
   - Create two sequences identical up to last_idx, but differ after last_idx (add extra tokens/padding).
   - Ensure output at last_idx is identical within tolerance.
   - This proves no future leakage.

E) Logging (so we can see speed win)
   - Print step time every 50 optimizer steps.
   - Confirm training iter time drops by at least ~10x vs previous full-seq prefix tensor.

Do NOT do these mistakes
- Do NOT compute full logits and then just pick last token (that doesn’t fix runtime).
- Do NOT use past_key_values cache for training full sequences (that breaks batching and causality).
- Do NOT accidentally include future tokens in M_k/Z_k when scanning.
- Do NOT forget attention_mask when updating moments.

Deliverables
- Updated iwm_attention.py with last-token-only training path.
- Updated train_iwm_scaled.py to use labels_last loss.
- Two new test files passing.
- A short run log showing iteration time improvement.
