# Phase 3: Multimodal Architecture & Token Interleaving

## Objective
Unify the visual representations and text token sequences into a single coherent multimodal sequence that the Transformer decoder processes autoregressively.

---

## 🧩 Sequence Fusion Strategies

### 1. Prefix Image Tokens
The simplest multimodal layout places image tokens at the start of the sequence:
```text
[ Visual Tokens ] + [ Text Prompt Tokens ] + [ Response Tokens ]
|<--- N_v ---->|   |<------- N_p ------->|   |<----- N_r ----->|
```
- Total sequence length: $T = N_v + N_p + N_r$.
- Embedding vectors:
  - Positions $0 \dots N_v-1$: from Projector output.
  - Positions $N_v \dots T-1$: from Token Embedding table.

---

### 2. The `<image>` Placeholder Token
In modern conversational setups (LLaVA), the prompt text includes a special placeholder token:
```text
"User: <image>\nDescribe what you see in this photo.\nAssistant: A golden retriever playing in a park."
```
1. Tokenize prompt text into token IDs.
2. Locate the index of `<image>` token ID in the sequence.
3. Slice and substitute the single `<image>` vector with the $N_v$ projected visual tokens!

---

### 3. Attention Mask & Loss Masking
- Causal Mask: Tokens at position $j > i$ cannot be attended to.
- Targets Tensor:
  - Set target label to `-100` for all visual tokens and all user prompt tokens.
  - Only compute cross-entropy loss on the assistant's response tokens!

---

## 🧪 Verification Checkpoints
- [ ] Placeholder replacement: Correctly replace `<image>` token index with the full block of $N_v$ visual embeddings.
- [ ] End-to-end forward pass: Pass an image tensor `(1, 3, 224, 224)` and text token sequence `(1, 40)` through the unified VLM and verify logits shape `(1, N_v + 40, vocab_size)`.
