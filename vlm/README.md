# Vision-Language Models (VLMs) From Scratch

This directory houses the comprehensive roadmap, architectural specifications, and implementation notebooks for building a multimodal Vision-Language Model (LLaVA-style) from scratch using PyTorch.

---

## 🏛️ Multimodal Architecture Blueprint

The target VLM connects a Vision Transformer (ViT) to our Causal LLM Decoder using a Multimodal Projection Layer:

```text
[Input Image (B, 3, H, W)]                   [Input Text Prompt]
          ↓                                           ↓
[Patch Embedding & ViT Encoder]              [Token Embedding Layer]
          ↓                                           ↓
[Vision Tokens (B, N_patches, D_vision)]     [Text Tokens (B, N_text, D_llm)]
          ↓                                           │
[Multimodal Projector (Linear / MLP)]                 │
          ↓                                           │
[Projected Visual Tokens (B, N_patches, D_llm)]       │
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ↓
        [Unified Multimodal Sequence: [Vision ; Text]]
                     (B, N_patches + N_text, D_llm)
                                ↓
      +==============================================+
      | Autoregressive LLM Decoder (from llm/ series)|
      |                                              |
      |   Self-Attention across visual & text tokens |
      |   Causal Masking on text generation          |
      +==============================================+
                                ↓
        [Output Text Logits / Predicted Visual Answers]
```

---

## 📐 Mathematical Specifications & Dimension Mapping

### 1. Vision Patch Embedding
- Given image $I \in \mathbb{R}^{B \times C \times H \times W}$ (e.g. $C=3, H=224, W=224$) and patch size $P=14$ (or $16$):
  - Number of patches $N = \frac{H}{P} \times \frac{W}{P} = \left(\frac{224}{14}\right)^2 = 16 \times 16 = 256$.
  - Patch projection maps each $(C \times P \times P)$ flattened patch to $D_{vision}$.
  - Add 2D learnable or sinusoidal positional embeddings $E_{pos} \in \mathbb{R}^{1 \times N \times D_{vision}}$.
- **Output Shape**: `(B, N_patches, D_vision)`

---

### 2. Multimodal Projector
The vision encoder and language model often operate in different vector spaces and dimensions ($D_{vision} \ne D_{llm}$, e.g. $D_{vision} = 768$ or $1024$, while $D_{llm} = 512$ or $2048$):

#### A. Linear Projector (LLaVA-1.0):
$$H_v = X_v W_p + b$$
where $W_p \in \mathbb{R}^{D_{vision} \times D_{llm}}$.

#### B. 2-Layer MLP Projector (LLaVA-1.5):
$$H_v = \text{GELU}(X_v W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{D_{vision} \times D_{llm}}$ and $W_2 \in \mathbb{R}^{D_{llm} \times D_{llm}}$.

- **Input Shape**: `(B, N_patches, D_vision)`
- **Output Shape**: `(B, N_patches, D_llm)` (visual tokens now match text embedding dimension!)

---

### 3. Unified Sequence Concatenation
Let visual token embeddings be $H_v \in \mathbb{R}^{B \times N_v \times D_{llm}}$ and text token embeddings be $H_t \in \mathbb{R}^{B \times N_t \times D_{llm}}$:

$$H_{multimodal} = [H_v \, ; \, H_t] \in \mathbb{R}^{B \times (N_v + N_t) \times D_{llm}}$$

This sequence is passed directly through the transformer decoder blocks implemented in `llm/`.

---

## 🏋️ 2-Stage Training Recipe

### Stage 1: Feature Alignment Pretraining
- **Objective**: Align vision features with the language embedding space.
- **Dataset**: Image-caption pairs (e.g. Flickr8k / Flickr30k / CC3M subset).
- **Weights**: Freeze ViT encoder and freeze LLM decoder. **Only train the Multimodal Projector weights**.
- **Loss**: Next-token prediction loss on caption tokens.

### Stage 2: Visual Instruction Fine-Tuning
- **Objective**: Teach the model to converse about images (VQA, detailed descriptions, reasoning).
- **Dataset**: Multimodal instruction-response pairs (LLaVA-Instruct).
- **Weights**: Keep ViT encoder frozen (or fine-tune with low lr); train Projector and LLM decoder (or LoRA on LLM).

---

## 🗂️ Phased Implementation Guide

| Phase | Milestone | Focus Areas | Guide / Spec | Notebook |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Vision Encoder** | ViT Patch Embedding, 2D Positional Embeddings, Transformer Encoder Blocks | [`01_vision_encoder/README.md`](01_vision_encoder/README.md) | [`01_vision_transformer.ipynb`](vlm/notebooks/01_vision_transformer.ipynb) |
| **02** | **Projector** | Linear & 2-Layer MLP Projectors, Dimension Alignment, Spatial Pooling | [`02_projector/README.md`](02_projector/README.md) | [`02_multimodal_projector.ipynb`](vlm/notebooks/02_multimodal_projector.ipynb) |
| **03** | **Multimodal Architecture** | Interleaving tokens, special image tokens `<image>`, attention masking | [`03_multimodal_architecture/README.md`](03_multimodal_architecture/README.md) | [`03_vlm_architecture.ipynb`](vlm/notebooks/03_vlm_architecture.ipynb) |
| **04** | **2-Stage Training** | Stage 1 Projector Alignment & Stage 2 Instruction Tuning in Colab | [`04_training/README.md`](04_training/README.md) | [`04_vlm_pretraining.ipynb`](vlm/notebooks/04_vlm_pretraining.ipynb) |
| **05** | **Multimodal Inference** | Image loading, preprocessing, prompt formatting, autoregressive visual generation | [`05_inference/README.md`](05_inference/README.md) | [`05_vlm_inference.ipynb`](vlm/notebooks/05_vlm_inference.ipynb) |
