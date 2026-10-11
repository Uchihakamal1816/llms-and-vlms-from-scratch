# Phase 5: Instruction Fine-Tuning (SFT) & LoRA

## Objective
Transform the raw pretrained base model into an instruction-following conversational assistant using Supervised Fine-Tuning (SFT) and Low-Rank Adaptation (LoRA).

---

## 💬 SFT Data Formatting & Masking

A conversational instruction dataset consists of pairs:
```text
System: You are a helpful AI assistant.
User: Explain gravity in one sentence.
Assistant: Gravity is the invisible force that attracts objects toward one another.
```

### The Prompt Loss Masking Principle
During training, we must **only compute loss on the Assistant's response tokens**, NOT on the User's prompt tokens:
1. Tokenize the full conversation string: `[tokens_prompt + tokens_response]`.
2. Construct target tensor:
   $$\text{targets}_i = \begin{cases} -100 & \text{if } i \in \text{prompt} \\ \text{tokens}_{i+1} & \text{if } i \in \text{response} \end{cases}$$
3. PyTorch's `F.cross_entropy` ignores labels equal to `-100` (`ignore_index=-100`).

---

## 🧩 LoRA (Low-Rank Adaptation) from Scratch

Instead of updating the full weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$, freeze $W$ and introduce two low-rank trainable matrices $A$ and $B$:

$$W' = W + \Delta W = W + \frac{\alpha}{r} (B A)$$

- $A \in \mathbb{R}^{r \times d_{in}}$ initialized from Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d_{out} \times r}$ initialized to zeros.
- $r \ll \min(d_{in}, d_{out})$ (typically $r = 4, 8, 16$).
- $\alpha$ is a scaling hyperparameter.
- At initialization, $\Delta W = B A = 0 \cdot A = 0$, so the model's behavior is unchanged before training begins.

---

## 🧪 Verification Checkpoints
- [ ] Verify trainable parameter reduction: LoRA should reduce trainable parameters to $< 1\%$ of the base model.
- [ ] Verify loss masking: Confirm gradient with respect to prompt tokens is zero.
- [ ] Instruction following test: Compare assistant output before and after SFT on test prompts.
