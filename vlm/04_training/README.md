# Phase 4: 2-Stage Multimodal Training Recipe

## Objective
Implement the two-stage training pipeline in Google Colab to train a functional Vision-Language Model from scratch.

---

## 🏋️ The 2-Stage Paradigm

```text
+---------------------------------------------------------------------------------+
| STAGE 1: Feature Alignment Pretraining                                          |
|                                                                                 |
|  [Image] ──> [ViT Encoder (FROZEN)] ──> [Projector (TRAINABLE)]                 |
|                                                     │                           |
|  [Caption Tokens] ────────────────────────────> [LLM Decoder (FROZEN)]          |
|                                                     │                           |
|                                              Cross-Entropy Loss on Captions     |
+---------------------------------------------------------------------------------+
                                       ↓
+---------------------------------------------------------------------------------+
| STAGE 2: Visual Instruction Fine-Tuning                                         |
|                                                                                 |
|  [Image] ──> [ViT Encoder (FROZEN)] ──> [Projector (TRAINABLE)]                 |
|                                                     │                           |
|  [QA Prompt + Answer] ────────────────────────> [LLM Decoder (TRAINABLE / LoRA)]|
|                                                     │                           |
|                                              Cross-Entropy Loss on Answer Only  |
+---------------------------------------------------------------------------------+
```

---

## 📊 Dataset Recommendations for Free Colab T4

1. **Stage 1 (Feature Alignment)**:
   - **Flickr8k / Flickr30k**: ~8,000 to 30,000 images with 5 captions each.
   - Very lightweight, downloads quickly in Colab, easily trains in 1–2 hours.
2. **Stage 2 (Instruction Tuning)**:
   - **LLaVA-Instruct-150k (subsample 5k–10k samples)**:
   - High-quality question-answering pairs about image contents.

---

## ⚡ Colab Training Tips
- Use `requires_grad = False` on frozen module parameters to save memory and skip backward gradient calculations.
- Cache pre-extracted ViT embeddings if GPU memory is tight!
  - If the ViT is completely frozen during Stage 1, you can pre-extract visual features for your dataset and store them on Google Drive. This reduces Stage 1 training to essentially training a small MLP on token vectors, which runs in minutes on a T4!

---

## 🧪 Verification Checkpoints
- [ ] Freeze verification: Confirm only projector parameters have `grad != None` in Stage 1.
- [ ] Loss tracking: Loss on image caption tokens drops steadily.
