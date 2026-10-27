"""
Unified LLM: Modern Decoder-Only Large Language Model from Scratch
===================================================================
An end-to-end, zero-high-level-wrapper Large Language Model stack combining:
  1. SuperBPE Tokenizer (Two-Stage Pre-tokenization Curriculum)
  2. RMSNorm (Root Mean Square Layer Normalization)
  3. Rotary Positional Embeddings (RoPE)
  4. Grouped Query Attention (GQA) with Causal Masking
  5. SwiGLU Gated Feed-Forward Network
  6. Pre-norm Residual Transformer Blocks
  7. Full Autoregressive Causal Language Model with KV-Cache Generation

Written purely in PyTorch (`torch`, `torch.nn`, `torch.nn.functional`) and Python.
"""

from __future__ import annotations

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.superbpe import SuperBPETokenizer, BytePairTokenizer, GPT4_PAT


# =============================================================================
# 1. RMSNorm (Root Mean Square Layer Normalization)
# =============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) = (x / RMS(x)) * gamma
    where RMS(x) = sqrt( mean(x^2) + eps )
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        # Calculate mean square across hidden dimension D
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


# =============================================================================
# 2. Rotary Position Embeddings (RoPE)
# =============================================================================
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Applies 2D rotation to coordinate pairs of query and key projections.
    theta_i = base^(-2(i-1)/dim_head)
    """

    def __init__(self, dim_head: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        assert dim_head % 2 == 0, "dim_head must be even for 2D pair rotation"
        self.dim_head = dim_head

        # Precompute frequency inverse powers
        freqs = 1.0 / (theta ** (torch.arange(0, dim_head, 2).float() / dim_head))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs_matrix = torch.outer(t, freqs)  # (max_seq_len, dim_head // 2)

        # Register cos and sin buffers
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # Rotates half the hidden dimensions: [-x2, x1]
        x1 = x[..., : self.dim_head // 2]
        x2 = x[..., self.dim_head // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: (B, H_q, T, D_head)
        # k: (B, H_kv, T, D_head)
        T = q.shape[2]
        cos = self.cos_cached[offset : offset + T, :]  # (T, D_head // 2)
        sin = self.sin_cached[offset : offset + T, :]  # (T, D_head // 2)

        # Broadcast across both halves of head dimension
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(1)  # (1, 1, T, D_head)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(1)  # (1, 1, T, D_head)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


# =============================================================================
# 3. Grouped Query Attention (GQA) with Causal Masking
# =============================================================================
class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    H_q query heads grouped with H_kv key-value heads (H_q % H_kv == 0).
    Includes causal masking, scaled dot-product, and optional KV caching.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = dim // n_heads
        self.num_rep = n_heads // n_kv_heads

        self.w_q = nn.Linear(dim, n_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(dim, n_kv_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.d_head, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_head, dim, bias=False)

        self.rope = RotaryEmbedding(dim_head=self.d_head, max_seq_len=max_seq_len)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: (B, H_kv, T, D_head) -> (B, H_q, T, D_head)
        if n_rep == 1:
            return x
        B, H_kv, T, D = x.shape
        return (
            x[:, :, None, :, :]
            .expand(B, H_kv, n_rep, T, D)
            .reshape(B, H_kv * n_rep, T, D)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, D)
        B, T, _ = x.shape

        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        offset = 0 if kv_cache is None else kv_cache[0].shape[2]
        q, k = self.rope(q, k, offset=offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Broadcast KV heads to match Q heads
        k_rep = self._repeat_kv(k, self.num_rep)
        v_rep = self._repeat_kv(v, self.num_rep)

        # Scaled dot-product attention
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.d_head)

        T_k = k_rep.shape[2]
        if mask is not None:
            scores = scores + mask
        elif T > 1:
            # Standard causal mask for training
            causal_mask = torch.full(
                (T, T_k), float("-inf"), device=x.device, dtype=scores.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=offset + 1)
            scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_rep)  # (B, H_q, T, D_head)

        out = out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.w_o(out), new_kv_cache


# =============================================================================
# 4. SwiGLU Feed-Forward Network
# =============================================================================
class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, T, D)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =============================================================================
# 5. Transformer Decoder Block
# =============================================================================
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Decoder Block:
      x = x + Attention(RMSNorm(x))
      x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
        )
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm Attention with residual
        normed_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(normed_x, mask=mask, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-norm SwiGLU with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv_cache


# =============================================================================
# 6. Unified Transformer Causal Language Model
# =============================================================================
class UnifiedTransformerLM(nn.Module):
    """
    Unified Causal Transformer Language Model Stack.

    Architecture:
      Token Embeddings -> N x TransformerBlock -> Final RMSNorm -> LM Head Linear
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        hidden_dim: int | None = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        hidden_dim = hidden_dim or int(2 * (4 * dim) / 3)

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    hidden_dim=hidden_dim,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie embedding and unembedding weights for parameter efficiency
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        # idx: (B, T)
        B, T = idx.shape
        x = self.tok_embeddings(idx)  # (B, T, D)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        x = self.norm(x)  # (B, T, D)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten across batch and sequence length for CrossEntropyLoss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-caching, temperature, top-k, and nucleus (top-p) sampling.
        """
        self.eval()
        B, T = prompt_ids.shape
        curr_ids = prompt_ids
        kv_caches = None

        # Warm-up KV cache on initial prompt
        logits, _, kv_caches = self(curr_ids, kv_caches=None)
        next_token_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            if temperature > 0:
                scaled_logits = next_token_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    val, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < val[:, [-1]]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Step single next token forward with cached past keys and values
            logits, _, kv_caches = self(next_token, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]

        return curr_ids


# =============================================================================
# 7. Unified LLM Pipeline (Tokenizer + Model Engine)
# =============================================================================
class UnifiedLLM:
    """
    End-to-end Unified LLM Pipeline:
      - Uses SuperBPETokenizer for tokenization and superwords.
      - Uses UnifiedTransformerLM for neural causal language modeling.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.tokenizer = SuperBPETokenizer()
        self.tokenizer.register_special_tokens(
            {"<|endoftext|>": vocab_size - 1, "<|pad|>": vocab_size - 2}
        )
        self.model = UnifiedTransformerLM(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        ).to(self.device)

    def train_tokenizer(self, corpus: str, vocab_size: int = 512, subword_ratio: float = 0.7):
        """Trains SuperBPE tokenizer on input text."""
        self.tokenizer.train(corpus, vocab_size=vocab_size, subword_ratio=subword_ratio)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
    ) -> str:
        """Encodes prompt, runs autoregressive generation, and decodes back to string."""
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [0]
        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        out_ids = self.model.generate(
            idx, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return self.tokenizer.decode(out_ids[0].tolist())


if __name__ == "__main__":
    print("Verifying Unified LLM Stack from scratch...")

    # 1. Test SuperBPE
    corpus = (
        "In the beginning, language models processed words. " * 15
        + "With SuperBPE, tokens traverse whitespace and form superwords. " * 10
    )
    llm = UnifiedLLM(vocab_size=350, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, device="cpu")
    llm.train_tokenizer(corpus, vocab_size=350, subword_ratio=0.6)

    # 2. Test forward pass and loss
    dummy_input = torch.randint(0, 350, (2, 16))
    dummy_targets = torch.randint(0, 350, (2, 16))
    logits, loss, _ = llm.model(dummy_input, targets=dummy_targets)

    assert logits.shape == (2, 16, 350), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Loss computed incorrectly"
    print(f"Forward pass successful! Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    # 3. Test generation with KV cache
    generated = llm.generate_text("In the beginning", max_new_tokens=10)
    print("Sample generation output:", repr(generated))
    print("Unified LLM validation completed successfully!")
