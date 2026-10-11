# Phase 2: Multimodal Projector From Scratch

## Objective
Implement projection architectures that map visual token representations into the language model's hidden embedding space.

---

## 🔌 Projector Architectures

### 1. Linear Projector (LLaVA-1.0 style)
The simplest projection: a single affine transformation:
```text
(B, N, D_vision) ──[ Linear(D_vision, D_llm) ]──> (B, N, D_llm)
```
- Computationally lightweight.
- Works surprisingly well when the vision encoder is already well-aligned (e.g. CLIP).

---

### 2. Multi-Layer Perceptron (MLP) Projector (LLaVA-1.5 style)
A 2-layer feedforward network with non-linear activation:
```text
(B, N, D_vision) ──[ Linear(D_vision, D_llm) ]──> GELU ──[ Linear(D_llm, D_llm) ]──> (B, N, D_llm)
```
- Provides non-linear mapping capacity to adapt visual features to language syntax and semantics.

---

### 3. Spatial Resampler / Pooling (Optional Advanced)
If $N_{patches}$ is large (e.g. 576 or 1024 patches), projecting all patches consumes excessive LLM context:
- Pixel shuffle / 2x2 spatial pooling downsamples patches by $4\times$ (e.g. 256 patches $\to$ 64 tokens).
- Concatenate 4 adjacent patch vectors and project: `nn.Linear(4 * D_vision, D_llm)`.

---

## 🧪 Verification Checkpoints
- [ ] Shape alignment: Ensure projected tokens match `(B, N, D_llm)` exactly.
- [ ] Trainable parameter count: Compute parameters of Linear vs 2-Layer MLP projector.
