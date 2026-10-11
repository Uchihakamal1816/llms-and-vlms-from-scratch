# Phase 5: Multimodal Inference Engine

## Objective
Build an interactive multimodal generation loop that takes an image and a natural language question and generates visual descriptions and answers autoregressively.

---

## 🖼️ Inference Pipeline

1. **Image Preprocessing**:
   - Load image using `PIL.Image`.
   - Resize to model input resolution (e.g. $224 \times 224$ or $336 \times 336$).
   - Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
   - Convert to tensor `(1, 3, H, W)`.

2. **Prompt Encoding**:
   - Format prompt string: `"User: <image>\nWhat color is the car?\nAssistant:"`.
   - Tokenize text.

3. **Multimodal Prefix Assembly**:
   - Pass image tensor through Vision Encoder $\to$ Projector to produce visual tokens $H_v \in (1, N_v, D_{llm})$.
   - Embed text tokens through token embedding table.
   - Assemble continuous prompt vectors: `[H_prompt_before, H_v, H_prompt_after]`.

4. **Autoregressive Generation with KV Cache**:
   - Prefill phase: Process the entire multimodal prefix once, populating the KV cache with keys and values for all visual and prompt tokens.
   - Generation phase: Feed only the single most recently generated token at each step, sample using Temperature/Top-$p$, and decode token IDs to text until `<eos>` token is generated.

---

## 🧪 Verification Checkpoints
- [ ] End-to-end question answering demo in Colab with user-uploaded images.
- [ ] Visual grounding check: Verify model response changes appropriately when given different input images for the same question.
