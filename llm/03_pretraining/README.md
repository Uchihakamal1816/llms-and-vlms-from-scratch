# Phase 3: Pretraining Pipeline From Scratch

## Objective
Implement an end-to-end causal language modeling training loop from raw data ingestion, token batching, learning rate schedules, and loss calculation to checkpointing in Google Colab.

---

## ⚙️ Key Concepts & Components

### 1. Data Pipeline & Sequence Packing
- **Dataset**: Start with **TinyStories** or **TinyShakespeare**.
- **Chunking**: Break token stream into non-overlapping or sliding chunks of length $T+1$:
  - `inputs = tokens[:-1]` (length $T$)
  - `targets = tokens[1:]` (length $T$)
- **Colab DataLoader**: Efficient PyTorch `Dataset` & `DataLoader` with `pin_memory=True`.

### 2. Loss Computation
- Cross-Entropy Loss over the flattened sequence:
  - Flatten logits from `(B, T, V)` to `(B*T, V)`.
  - Flatten targets from `(B, T)` to `(B*T)`.
  - `loss = F.cross_entropy(logits_flat, targets_flat)`.
- **Initial Loss Sanity Check**: For a vocabulary of size $V$, the initial random loss should be approximately:
  $$\mathcal{L}_{initial} \approx -\ln\left(\frac{1}{V}\right) = \ln(V)$$
  *(e.g., for $V=4096$, $\ln(4096) \approx 8.31$)*.

### 3. Optimization Strategy
- **Optimizer**: AdamW with $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$.
- **Learning Rate Schedule**:
  - Linear Warmup for first $N_{warmup}$ steps.
  - Cosine Decay down to $0.1 \times \text{lr}_{max}$.
- **Gradient Clipping**: Clip global gradient norm to $1.0$ (`torch.nn.utils.clip_grad_norm_`).
- **Gradient Accumulation**: Simulate larger effective batch sizes on Colab T4.
- **Mixed Precision**: Use `torch.cuda.amp.autocast(dtype=torch.bfloat16)` or `torch.float16` with `GradScaler`.

---

## 🧪 Verification Checkpoints
- [ ] Overfitting test on a single batch: A 5M–15M parameter model should overfit a batch of 10 samples to loss $< 0.1$ within 100 steps.
- [ ] Convergence: Loss steadily declines on TinyStories from $\sim 8.0$ down to $< 1.8$.
- [ ] Drive persistence: Checkpoint files (`model_step_*.pt`) correctly saved to Google Drive.
