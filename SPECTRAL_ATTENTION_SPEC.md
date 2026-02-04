# SPECIFICATION: Spectral Attention Mechanism (SAM)
## Implementation Guide for CIMBA-Native KV Cache & Mastery-Inference

### 1. Executive Summary
The goal is to replace the standard $O(N)$ KV Cache in a Transformer model (target: the new Qwen 80B - https://huggingface.co/Qwen/Qwen3-Coder-Next) with a constant-memory $O(K)$ **Spectral Cache**. This implementation is based on the **CIMBA** (Moment Sketching), **MAST** (Minimal Sufficiency), and **SBGP** (Context Gluing) research papers. 

By representing the context as a set of circular moments rather than raw token vectors, we enable "infinite" context within a fixed VRAM budget, allowing 80B+ models to run comfortably on 16GB consumer GPUs (like the RTX 4070 Ti Super).

---

### 2. Theoretical Foundations
The Dev AI must strictly adhere to these principles from the provided papers:
*   **MAST (Motif Algebra Sufficiency Theorem):** The KV Cache for a block should be stored as the projection of the orbit-averaged outer product $M_G(x)$ onto the Motif Algebra $\mathcal{A}$. This is the *minimal* sufficient statistic for rotation-invariant queries (Self-Attention).
*   **SBGP (Sheaf-of-Bars Gluing Principle):** When context length exceeds the "bar" (local window), use the Čech coboundary $\delta^0$ consistency check to "glue" new token moments into the global spectral state without increasing the state dimension.
*   **CIMBA (Moment Sketching):** Implementation of the complex moments $M_k = \sum e^{ik\theta} v$.

---

### 3. Technical Requirements

#### 3.1 Spectral KV Layer (The Core Task)
Replace the standard `forward` pass logic in `Qwen2Attention` with the following SAM logic:

1.  **Sketching (Input):**
    *   Map token positions to phase angles $\theta \in [0, 2\pi)$.
    *   For each new token $i$, compute $K$ complex moments: $M_k^{(i)} = V_i \cdot e^{jk\theta_i}$.
    *   Update the Global Motif Statistic: $\Phi_{\mathcal{A}} \leftarrow \Phi_{\mathcal{A}} + M_k^{(i)}$.

2.  **Recovery (Query):**
    *   Given a Query vector $Q$ at position $t$, reconstruct the attention result $Y_t$ directly from $\Phi_{\mathcal{A}}$ using **Prony's Method** (for sparse important tokens) or a direct **Inverse Fourier Transform** approximation.
    *   $\text{Attention}(Q, \Phi_{\mathcal{A}}) \approx \sum_{k} \text{Project}(Q, k) \cdot \Phi_{\mathcal{A}}[k]$.

3.  **Gluing (SBGP):**
    *   Maintain a "Boundary Stalk" (overlap) between consecutive context windows.
    *   If $\delta^0 (\Phi_{current} - \Phi_{previous}) \neq 0$, apply the SBGP one-pass gluing operator $\Pi$ to correct the global section.

#### 3.2 Performance Targets
*   **Complexity:** Memory must remain $O(K)$ where $K$ is the number of moments (harmonic degree), regardless of context length $L$.
*   **Precision:** Use `torch.complex64` for moment calculations to preserve phase information.
*   **Hardware:** Optimized for 16GB VRAM; the cache for a 128k context should take < 500MB.

---

### 4. Implementation Steps for the Dev AI

1.  **Phase I: Spectral Basis Mapping**
    *   Implement a module that transforms standard Keys/Values into the Motif Algebra $\mathcal{A}$.
    *   Validate that $\tr(L \, \Phi_{\mathcal{A}}(x))$ recovers the intended query response.

2.  **Phase II: The SBGP Gluer**
    *   Build the linear data sheaf for context chunks.
    *   Precompute the right-inverse $R$ for the coboundary operator to allow one-pass correction.

3.  **Phase III: Integration with Qwen3-Coder**
    *   Monkey-patch the `modeling_qwen.py` Attention class.
    *   Implement "Spectral Distillation" finetuning: Since the base model weights expect standard attention, a brief LoRA finetune (on ~1000 coding samples) is required to teach the model to "speak" in moments.

---

### 5. Critical Constraints
*   **Truth over Agreement:** If the Dev AI finds that $H^1 \neq 0$ for a specific configuration, it must stop and reconfigure the bar/overlap summaries. It should NOT attempt to bypass the cohomology requirement with a heuristic.
*   **Minimalism:** coordinate vectors in the Hilbert-Schmidt basis of $\mathcal{A}$ are the only things that should persist in VRAM.

---
**Reference Papers (Context):**
1. `CIMBA`: Invertible Moment Sketches.
2. `CircleStat`: Circular Transformations.
3. `MAST`: Minimal Invariant Statistics.
4. `SBGP`: Cohomological Gluing.
