## LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention* 

The **core attention computation** in large language models (LLMs):

$$
S_h = \frac{q_h K_{\hat{h}}^T}{\sqrt{D}}, \quad 
o_h = \text{softmax}(S_h)V_{\hat{h}}, \quad 
\hat{h} = \left\lfloor \frac{h}{n} \right\rfloor
$$


1. **Context**

   * Each transformer layer has **multiple attention heads**, allowing the model to attend to different parts of the sequence in parallel.
   * The model computes attention over **queries (Q)**, **keys (K)**, and **values (V)**.

---

2. **Query-Key Similarity**

   $$
   S_h = \frac{q_h K_{\hat{h}}^T}{\sqrt{D}}
   $$

   * $q_h$ is the **query vector** for head $h$.

   * $K_{\hat{h}}$ is the **matrix of key vectors** associated with head $\hat{h}$.

   * $D$ is the **head dimension**.

   * The dot product $q_h K_{\hat{h}}^T$ measures how similar each query is to each key.

   * Dividing by $\sqrt{D}$ normalizes the values to stabilize training (avoids large variance in dot products).

   > This step computes **attention scores** between the query token and all previous tokens in the sequence.

---

3. **Attention Weights and Output**

   $$
   o_h = \text{softmax}(S_h) V_{\hat{h}}
   $$

   * A **softmax** is applied to $S_h$ to convert raw similarity scores into a probability distribution (attention weights).
   * These weights are used to compute a **weighted sum** over the value vectors $V_{\hat{h}}$.
   * The result $o_h$ represents the **aggregated contextual information** for head $h$.

---

4. **Grouped-Query Attention (GQA)**

   $$
   \hat{h} = \left\lfloor \frac{h}{n} \right\rfloor
   $$

   * In modern models (like LLaMA 3), **Grouped-Query Attention** is used.
   * This means **multiple query heads share the same key/value heads** to save memory.
   * The index $\hat{h}$ maps query head $h$ to its corresponding key/value group (if $n>1$).

---

5. **Computational Complexity**

   $$
   O(N(S + N)HD)
   $$

   where:

   * $N$: number of query tokens (prefill batch)

   * $S$: number of stored key/value tokens (history)

   * $H$: number of heads

   * $D$: head dimension

   * Complexity grows **quadratically** with input length during **prefill** and **linearly** during **decoding**.

---

### \ud83d\udd39 Intuition

Equation (1) describes the **core computation of attention**, where:

* Each query token decides which past tokens to focus on (via softmax over $qK^T$).
* The model then synthesizes this information using $V$.
* LServe builds upon this standard operation by **skipping unnecessary KV blocks** (via block sparsity) to accelerate these computations.

---

**LServe** modifies and extends the standard attention formula through its **unified block-sparse attention** framework.

---

## \ud83e\udde0 Starting Point — Dense Attention (Eq. 1 Recap)

$$
S_h = \frac{q_h K_{\hat{h}}^T}{\sqrt{D}}, \quad
o_h = \text{softmax}(S_h)V_{\hat{h}}
$$

Every query token attends to **all** previous key–value pairs.
This is **dense** attention — powerful but **O(N^2)** in prefill and **O(N)** in decoding.

---

## \u2699\ufe0f LServe’s Core Idea — Block-Sparse Modification

LServe observes that **not all tokens are equally important**.
So instead of computing $q_h K^T$ for every token pair, it divides the KV history into **blocks (pages)** and **skips entire blocks** that contribute little to the output.

Formally, for each query block $Q_b$:

$$
S_h^{(b)} =
\begin{cases}
\frac{Q_b K_{\hat{b}}^T}{\sqrt{D}}, & \text{if block } b \text{ is active} \
0, & \text{if block } b \text{ is skipped}
\end{cases}
$$

Then attention proceeds as:

$$
o_h = \text{softmax}!\left(\sum_{b \in \mathcal{B}*{\text{active}}} S_h^{(b)}\right)
\sum*{b \in \mathcal{B}*{\text{active}}} V*{\hat{b}}
$$

where $\mathcal{B}_{\text{active}}$ is the set of **non-skipped KV blocks**.

---

## \ud83e\udde9 Two Complementary Sparsity Types

LServe unifies **static** and **dynamic** sparsity within this block structure:

1. **Static Sparsity — Streaming Heads**

   * Certain attention heads use a *fixed Λ-shaped mask*.
   * Each token only attends to **local** and **sink** (initial) blocks.
   * These are pre-determined (“offline”), yielding *constant-cost* heads.

   \u2192 Equation (1) becomes a *subset sum* over fixed neighboring blocks.

2. **Dynamic Sparsity — Page-Pruned Heads**

   * Other heads adaptively choose relevant blocks per query during decoding.
   * A **hierarchical page selector** computes importance scores:

     $$
     S_j = \sum_i \max(q[i] , k_{j,\max}[i], , q[i] , k_{j,\min}[i])
     $$

     and retains only top-K blocks.

   \u2192 Equation (1) is computed **only over these selected pages**, bounding decoding cost to a constant.

---

## \ud83d\ude80 Efficiency Consequence

If the sparsity ratio is $r$ (fraction of skipped blocks),
the theoretical speedup is approximately:

$$
\text{Speedup} \approx \frac{1}{1 - r}
$$

Example from the paper:
If 10 of 21 blocks are kept \u2192 $r = 11/21 \approx 0.52$ \u2192 ~2.1\u00d7 faster.

---

## \ud83d\udca1 Unified Kernel Implementation

LServe fuses both static and dynamic sparse patterns into a **single GPU kernel**:

* **Prefill:** executes static (streaming + dense) heads together.
* **Decode:** reuses the same kernel but feeds shortened page tables from the selector.

Thus, the **modified Eq. (1)** is evaluated only for chosen blocks / heads, achieving the same attention semantics as dense models but at a fraction of the computation.

---

### \u2705 Summary

| Aspect               | Dense Eq. (1)            | LServe Modification         |
| -------------------- | ------------------------ | --------------------------- |
| Scope of computation | All KV pairs             | Only selected KV blocks     |
| Sparsity type        | None                     | Static + Dynamic            |
| Complexity           | $O(N^2)$/ $O(N)$         | $O((1-r)N^2)$/ constant     |
| Implementation       | Per-token attention loop | Block-sparse unified kernel |

