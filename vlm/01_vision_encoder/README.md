# Phase 1: Vision Encoder (Vision Transformer ViT from Scratch)

## Objective
Implement a Vision Transformer (ViT) that converts a 2D image into a sequence of continuous visual embedding vectors.

---

## 🧱 Architectural Components & Math

### 1. Patch Projection
An image $x \in \mathbb{R}^{B \times C \times H \times W}$ is divided into non-overlapping patches of size $P \times P$:
- Method A (Convolution): A 2D convolution `nn.Conv2d(in_channels=3, out_channels=D, kernel_size=P, stride=P)`.
- Method B (Unfold / Reshape): Reshape `(B, C, H/P, P, W/P, P)` $\to$ `(B, N, C*P*P)` followed by linear layer `nn.Linear(C*P*P, D)`.
- Both methods yield patch tokens of shape: `(B, N, D)`.

### 2. Positional Embeddings
Since images have spatial 2D structure, tokens need spatial information:
- Learnable parameter `pos_embed = nn.Parameter(torch.zeros(1, N, D))` or `1 + N` if using a `[CLS]` token.
- Additive positional encoding:
  $$z_0 = [x_{class} \, ; \, x_{patch}^1 \, ; \, \dots \, ; \, x_{patch}^N] + E_{pos}$$

### 3. Transformer Encoder Block
Unlike decoder models, standard vision encoders use **bidirectional (non-causal) attention**:
- No triangular causal mask! Every visual patch attends to every other visual patch.
- LayerNorm / RMSNorm + Multi-Head Self-Attention + MLP (GELU).

---

## 🧪 Verification Checkpoints
- [ ] Tensor shape verification: Pass `torch.randn(2, 3, 224, 224)` through patch embedding with $P=14$. Output must be `(2, 256, D)`.
- [ ] Invariance & Permutation check: Verify bidirectional attention mechanics.
- [ ] Gradient flow: Verify backpropagation passes clean gradients to all patch weights.
