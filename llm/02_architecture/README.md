# Phase 2: Transformer Decoder Architecture From Scratch

## Objective
Implement every foundational building block of a modern decoder-only Transformer (Llama 3 style) using PyTorch `nn.Module` and tensor operations.

---

## 🧱 Component Breakdown & Tensor Specifications

### 1. `RMSNorm`
- **Formula**:
  $$y = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$
- **Parameters**: `weight = nn.Parameter(torch.ones(dim))`
- **Input**: `(B, T, D)`
- **Output**: `(B, T, D)`

---

### 2. `RotaryEmbedding` (RoPE)
- **Precompute**: Frequencies $\theta_k = \text{base}^{-2k/d_{head}}$ for $k \in [0, d_{head}/2 - 1]$.
- **Complex rotation** or interleaved 2D rotation:
  $$\begin{pmatrix} q_1' \\ q_2' \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$$
- **Input**: Query $Q \in (B, H, T, D_h)$, Key $K \in (B, H_{kv}, T, D_h)$
- **Output**: Rotated $Q, K$ with same shapes.

---

### 3. `GroupedQueryAttention` (GQA)
- Supports $H_Q$ query heads and $H_{KV}$ key-value heads.
- Repeat / broadcast KV heads by a factor of $H_Q / H_{KV}$.
- Apply causal lower-triangular mask (prevent attending to future tokens).
- Compute scaled dot-product attention: $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + \text{mask}\right) V$.
- Project output with linear layer $W_O \in \mathbb{R}^{D \times D}$.
- **Input**: `x: (B, T, D)`
- **Output**: `(B, T, D)`

---

### 4. `SwiGLU` (Gated Feed-Forward Network)
- Linear layers: `w_gate` $(D \to D_{hidden})$, `w_up` $(D \to D_{hidden})$, `w_down` $(D_{hidden} \to D)$.
- Forward pass:
  $$\text{output} = (\text{SiLU}(x W_{gate}) \odot (x W_{up})) W_{down}$$
- **Input**: `(B, T, D)`
- **Output**: `(B, T, D)`

---

### 5. `TransformerBlock`
- Pre-norm structure:
  $$x = x + \text{GQA}(\text{RMSNorm}(x))$$
  $$x = x + \text{SwiGLU}(\text{RMSNorm}(x))$$

---

### 6. `TransformerLM` (Complete Model)
- `tok_embeddings = nn.Embedding(vocab_size, dim)`
- `layers = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])`
- `norm = RMSNorm(dim)`
- `lm_head = nn.Linear(dim, vocab_size, bias=False)` (optionally weight-tied to `tok_embeddings.weight`)

---

## 🧪 Verification Checkpoints
- [ ] Dimension checking: Run random tensor `torch.randint(0, vocab_size, (2, 32))` through model and verify output is `(2, 32, vocab_size)`.
- [ ] Causal property verification: Ensure perturbation of token at position $t > 10$ does not change output logits at position $t \le 10$.
- [ ] Parameter count sanity check: Verify total parameters against theoretical formula.
