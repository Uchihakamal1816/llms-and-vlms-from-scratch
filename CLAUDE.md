# CLAUDE.md — LLMs & VLMs From Scratch

This document defines the architecture, project conventions, current implementation status, and operating standards for **Claude Code** when working in this repository.

---

## 🎯 1. Project Mission & Core Ethos

This repository is an end-to-end implementation journey building modern **Large Language Models (LLMs)** and **Vision-Language Models (VLMs)** completely from scratch using foundational mathematical principles and pure PyTorch.

### Non-Negotiable Core Rule
> **Zero High-Level Wrapper Libraries**:
> Never use high-level abstractions like Hugging Face `transformers.AutoModel`, `transformers.Trainer`, LangChain, or Keras.
> All tokenizers, attention layers, normalization, projection layers, positional embeddings, training loops, KV caches, and inference engines must be written by hand using pure PyTorch (`torch`, `torch.nn`, `torch.nn.functional`), `einops`, and raw Python.

---

## 🏛️ 2. Repository Architecture & Layout

```text
.
├── CLAUDE.md                       # Claude Code project guide and persistent context
├── README.md                       # Project master index and curriculum roadmap
├── requirements.txt                # Core dependencies (PyTorch, einops, datasets, etc.)
├── COLAB_WORKFLOW.md               # Guide for executing & persisting on Google Colab
├── assets/                         # Architecture diagrams and visual workflows
├── llm/                            # Part 1: Decoder-Only Large Language Models
│   ├── README.md                   # LLM Master Architecture Guide & Math Spec
│   ├── 01_tokenization/            # Phase 1: BPE tokenizer specs & verification criteria
│   ├── 02_architecture/            # Phase 2: RoPE, RMSNorm, GQA, SwiGLU specs
│   ├── 03_pretraining/             # Phase 3: Causal LM loss, data packing, training loop
│   ├── 04_inference/               # Phase 4: KV-cache generation & sampling strategies
│   ├── 05_finetuning/              # Phase 5: SFT, prompt masking, LoRA from scratch
│   └── notebooks/                  # Colab-ready implementation notebooks (01 to 05)
└── vlm/                            # Part 2: Multimodal Vision-Language Models (LLaVA-style)
    ├── README.md                   # VLM Master Architecture Guide & Math Spec
    ├── 01_vision_encoder/          # Phase 1: ViT patch embeddings & Transformer encoder
    ├── 02_projector/               # Phase 2: Linear & 2-Layer MLP projector (GELU)
    ├── 03_multimodal_architecture/ # Phase 3: Cross-modal sequence interleaving & masks
    ├── 04_training/                # Phase 4: 2-stage training (Alignment -> SFT)
    ├── 05_inference/               # Phase 5: Multimodal image-text generation engine
    └── notebooks/                  # Colab-ready implementation notebooks (01 to 05)
```

---

## 📊 3. Implementation Status & Next Milestones

### Part 1: Large Language Models (`llm/`)
- [x] **Phase 01: Tokenization (`01_bpe_tokenizer.ipynb`)**
  - Implemented `get_stats(ids)`, `merge(ids, pair, idx)`, and `BytePairTokenizer` class (Tasks 1 to 3).
  - *Note*: Advanced SuperBPE tokenizer and end-to-end unified architecture live on the dedicated `unified-llm` branch (`unified_llm.ipynb`).
- [ ] **Phase 02: Architecture (`02_transformer_decoder.ipynb`) — 📍 CURRENT ACTIVE MILESTONE**
  - **Task 1: RMSNorm** — $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$
  - **Task 2: Rotary Positional Embeddings (RoPE)** — $\theta_i = 10000^{-2(i-1)/d}$, complex rotation on query/key coordinate pairs.
  - **Task 3: Grouped Query Attention (GQA)** — $H_Q$ query heads, $H_{KV}$ key-value heads, causal mask, scaled dot-product.
  - **Task 4: SwiGLU FFN** — $\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \cdot (x W_{\text{up}})) W_{\text{down}}$.
  - **Task 5: TransformerBlock & Full Causal LLM Stack** — Pre-norm residual blocks, unembedding head, tied weights option.
- [ ] **Phase 03: Pretraining (`03_pretraining.ipynb`)**
  - Causal next-token prediction loss (`CrossEntropyLoss`), AdamW optimizer with cosine warmup/decay, gradient accumulation, mixed precision (`torch.cuda.amp` / `autocast`).
- [ ] **Phase 04: Inference & KV Cache (`04_inference_kv_cache.ipynb`)**
  - Autoregressive loop with cached K and V tensors, temperature scaling, top-$k$, top-$p$ (nucleus) sampling, repetition penalties.
- [ ] **Phase 05: Instruction Fine-Tuning & LoRA (`05_sft_finetuning.ipynb`)**
  - SFT formatting, prompt loss masking (`label = -100` on prompt tokens), custom Low-Rank Adaptation (`LoRALinear` with $W + \frac{\alpha}{r} B A$).

### Part 2: Vision-Language Models (`vlm/`)
- [ ] **Phase 01: Vision Encoder (`01_vision_transformer.ipynb`)** — Patch projection $(P \times P \times C \to D_{vis})$, 2D positional embeddings, ViT encoder.
- [ ] **Phase 02: Multimodal Projector (`02_multimodal_projector.ipynb`)** — Projection ($D_{vis} \to D_{llm}$), Linear & 2-Layer MLP with GELU.
- [ ] **Phase 03: Multimodal Architecture (`03_vlm_architecture.ipynb`)** — Sequence fusion: `[<image_tokens>, <text_tokens>]`, causal masking.
- [ ] **Phase 04: 2-Stage Training (`04_vlm_pretraining.ipynb`)** — Stage 1: Freeze ViT + LLM, train projector. Stage 2: Visual instruction tuning.
- [ ] **Phase 05: Multimodal Inference (`05_vlm_inference.ipynb`)** — Image preprocessing, prompt formatting, multimodal autoregressive sampling.

---

## 🛠️ 4. Coding Standards & Conventions

1. **Explicit Tensor Shapes**:
   - Always annotate tensor shapes at input, key intermediate transformations, and output.
   - Example: `# x: (B, T, D) -> Q: (B, H_q, T, D_head), K: (B, H_kv, T, D_head)`
2. **Vectorization**:
   - Zero Python loops over batch size $B$, sequence length $T$, or head counts $H$. Use PyTorch broadcasting, matrix multiplications, and `einops.rearrange` or `torch.reshape` / `torch.transpose`.
3. **Notebook Structure**:
   - Each notebook is structured with paired cells:
     1. `# Step: Task N: <Name>` — Implementation cell.
     2. `# Test for Task N: <Name>` — Verification cell with rigorous `assert` checks for outputs, gradients, and shape invariants.
   - Do NOT delete the Google Colab setup cell, Google Drive mounting cell, or Git Commit & Push cell at the end of each notebook.
4. **Numerical Stability**:
   - Always use an $\epsilon$ (e.g. `1e-5` or `1e-6`) inside square roots and denominators.
   - For attention logits, scale by $\frac{1}{\sqrt{d_k}}$ before adding causal masks ($-\infty$ or `-1e9`).
5. **Colab & Device Agility**:
   - Code must run seamlessly across `cpu`, `cuda` (NVIDIA T4 / A100), and `mps` (Apple Silicon).
   - Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## ⚡ 5. Verification Commands

To verify a notebook or script locally without opening a browser:
```bash
# Run syntax and basic test check
python3 -c "import torch; print('PyTorch Version:', torch.__version__, 'CUDA Available:', torch.cuda.is_available())"

# Execute a notebook headless via jupyter nbconvert (or test exported scripts)
jupyter nbconvert --to notebook --execute llm/notebooks/01_bpe_tokenizer.ipynb --output executed_01.ipynb
```
