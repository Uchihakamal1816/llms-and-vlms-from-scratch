# Large Language Models (LLMs) From Scratch

This directory houses the complete roadmap, architectural specifications, and implementation notebooks for building a modern decoder-only Large Language Model (GPT / Llama style) using pure PyTorch.

---

## 🏛️ Architectural Blueprint

The target model adopts the modern open-source LLM standard (similar to LLaMA 3 / Mistral):

```text
[Input Token IDs] -> (B, T)
       ↓
[Token Embedding Layer] -> (B, T, D)
       ↓
+==============================================+
| Transformer Decoder Block (Repeated N times) |
|                                              |
|   1. RMSNorm (Pre-normalization)             |
|   2. Rotary Positional Embeddings (RoPE)     |
|   3. Grouped Query Attention (GQA)           |
|      + Causal Attention Mask                 |
|   4. Residual Add: x = x + Attention(RMSNorm)|
|   5. RMSNorm (Pre-normalization)             |
|   6. SwiGLU Feed-Forward Network             |
|   7. Residual Add: x = x + FFN(RMSNorm)      |
+==============================================+
       ↓
[Final RMSNorm] -> (B, T, D)
       ↓
[LM Head Linear Layer (Unembedded)] -> (B, T, Vocab_Size)
       ↓
[CrossEntropyLoss (Pretraining) or Softmax / Sampling (Inference)]
```

---

## 📐 Mathematical Specifications & Tensor Shapes

### 1. RMSNorm (Root Mean Square Layer Normalization)
Instead of standard LayerNorm which computes both mean and variance, RMSNorm simplifies by normalizing only by the root mean square:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

- **Input Shape**: `(B, T, D)`
- **Output Shape**: `(B, T, D)`
- **Learnable Parameters**: Gain vector $\gamma \in \mathbb{R}^D$

---

### 2. Rotary Position Embedding (RoPE)
Instead of additive learned positional embeddings, RoPE rotates pairs of coordinates in the 2D plane:

$$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$

where $\theta_i = 10000^{-2(i-1)/d}$. RoPE naturally incorporates relative position decay into query-key dot products:

$$\langle R_m q, R_n k \rangle = g(q, k, m - n)$$

- **Input Q, K Shape**: `(B, H, T, D_head)`
- **Output Q, K Shape**: `(B, H, T, D_head)`

---

### 3. Grouped Query Attention (GQA)
Standard Multi-Head Attention (MHA) has $H_Q = H_{KV}$ heads. Multi-Query Attention (MQA) shares 1 KV head across all query heads. GQA generalises this by having $H_Q$ query heads grouped into $H_{KV}$ key-value heads (e.g., 8 query heads per 2 KV heads):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

where $M$ is the causal triangular mask where upper triangular elements are $-\infty$.

- **Q Shape**: `(B, H_Q, T, D_head)`
- **K, V Shape**: `(B, H_KV, T, D_head)` -> broadcast / repeat to `(B, H_Q, T, D_head)`
- **Attention Weights**: `(B, H_Q, T, T)`
- **Output**: `(B, T, D)` after linear projection $W_O$

---

### 4. SwiGLU Feed-Forward Network
Replaces standard ReLU/GELU MLP with a Gated Linear Unit with Swish/SiLU activation:

$$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{gate}) \odot x W_{up}\right) W_{down}$$

- **Input Shape**: `(B, T, D)`
- **Intermediate Shape**: `(B, T, D_hidden)` where typically $D_{hidden} \approx \frac{8}{3} D$
- **Output Shape**: `(B, T, D)`

---

## 🗂️ Phased Implementation Guide

| Directory | What to Build | Checkpoints to Reach |
| :--- | :--- | :--- |
| [`01_tokenization/`](01_tokenization/) | Byte-Pair Encoding (BPE) from byte sequences | Train vocabulary on raw text, handle unknown tokens, encode & decode round-trip tests |
| [`02_architecture/`](02_architecture/) | Modular PyTorch Transformer blocks | Forward pass matches expected tensor shapes, causal masking verified, no nan/inf |
| [`03_pretraining/`](03_pretraining/) | Dataset loader, AdamW, loss & training loop | Train on TinyStories or Shakespeare in Colab; loss decreases from ~10.0 to < 1.5 |
| [`04_inference/`](04_inference/) | KV Cache & Generation loop | Implement greedy, temperature, top-p, top-k; benchmark KV-cache vs non-KV cache speed |
| [`05_finetuning/`](05_finetuning/) | SFT & LoRA | Format instruction-response pairs; mask prompt loss; test parameter-efficient adapters |
