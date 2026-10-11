# Phase 4: Autoregressive Inference & KV Cache

## Objective
Implement an optimized inference engine with Key-Value (KV) caching and modern decoding sampling algorithms.

---

## ⚡ The KV Cache Mechanics

During autoregressive generation, generating token $t+1$ only requires the query vector $Q$ of token $t$. Computing keys and values for all past tokens $1 \dots t-1$ repeatedly costs $O(T^2)$ compute.

### Solution:
Maintain a cache tensor for Keys and Values in every attention layer:
- `k_cache`: `(B, H_kv, max_seq_len, D_head)`
- `v_cache`: `(B, H_kv, max_seq_len, D_head)`
- At step $t$, only pass input token $x_t \in (B, 1)$ through the network.
- Compute $Q_t \in (B, H, 1, D_h)$, $K_t \in (B, H_{kv}, 1, D_h)$, $V_t \in (B, H_{kv}, 1, D_h)$.
- Insert $K_t, V_t$ into cache at index $t$.
- Attend $Q_t$ over all cached keys $K_{1..t}$ and values $V_{1..t}$.
- Speedup: Transforms per-step complexity from $O(t)$ to $O(1)$ matrix multiplies!

---

## 🎲 Decoding Strategies

Given next-token logits $z \in \mathbb{R}^V$:

1. **Greedy Decoding**:
   $$\text{next\_token} = \arg\max_i(z_i)$$
2. **Temperature Scaling**:
   $$p_i = \text{softmax}\left(\frac{z_i}{T}\right)$$
   - Lower temperature ($T \to 0$): more deterministic, greedy.
   - Higher temperature ($T > 1$): more diverse, creative.
3. **Top-$k$ Sampling**:
   - Filter logits so only the $k$ highest scoring tokens remain; set others to $-\infty$.
4. **Top-$p$ (Nucleus) Sampling**:
   - Sort probabilities in descending order.
   - Keep the smallest set of tokens whose cumulative probability $\ge p$ (e.g. $p = 0.9$).
5. **Repetition Penalty**:
   - Penalize logits of tokens that have already appeared in the context window.

---

## 🧪 Verification Checkpoints
- [ ] Speed comparison: Measure token generation latency with and without KV cache as sequence length increases from 50 to 500.
- [ ] Consistency check: Verify that greedy decoding with KV-cache produces identical token outputs as full-context re-computation.
