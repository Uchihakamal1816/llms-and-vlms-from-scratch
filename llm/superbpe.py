"""
SuperBPE: Space Travel for Language Models (Tokenization from Scratch)
======================================================================
This module implements Byte-Pair Encoding (BPE) and SuperBPE from first principles
without high-level wrapper libraries.

SuperBPE Innovation:
Traditional BPE tokenizers (GPT-2, LLaMA) restrict token merges within word/regex
boundaries, forcing common multi-word expressions (e.g., 'in the', 'by the way')
to consume multiple tokens. SuperBPE uses a two-stage pre-tokenization curriculum:
  - Stage 1: Subword Learning — BPE merges strictly within whitespace/regex chunks.
  - Stage 2: Superword Learning — Boundary-relaxed BPE merges across whitespace,
    learning multi-word 'superwords' for higher compression and faster inference.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Sequence
import regex as re

# GPT-4 pre-tokenization regex pattern
GPT4_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids: list[int], counts: Counter | None = None) -> Counter:
    """
    Counts frequency of consecutive integer pairs.
    Optionally accumulates counts into an existing Counter.

    Args:
        ids: List of integer token IDs.
        counts: Optional existing Counter instance to update in-place.

    Returns:
        Counter with pair tuples (id1, id2) mapped to frequency counts.
    """
    counts = Counter() if counts is None else counts
    counts.update(zip(ids, ids[1:]))
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    Replaces all non-overlapping occurrences of `pair` with new token ID `idx`.

    Args:
        ids: Input list of token IDs.
        pair: Tuple of two consecutive IDs (id1, id2) to merge.
        idx: New replacement token ID.

    Returns:
        New list of token IDs with `pair` replaced by `idx`.
    """
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BytePairTokenizer:
    """
    Standard Byte-Pair Encoding Tokenizer with regex chunk pre-tokenization.
    Merges are strictly isolated within chunks.
    """

    def __init__(self, pattern: str = GPT4_PAT):
        self.pattern = re.compile(pattern)
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int):
        """Trains BPE merges up to vocab_size starting from 256 byte tokens."""
        assert vocab_size >= 256, "vocab_size must be >= 256"
        chunks = [list(c.encode("utf-8")) for c in self.pattern.findall(text)]
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = Counter()
            for c in chunks:
                get_stats(c, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(c, pair, idx) for c in chunks]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        """Iteratively applies the lowest-rank merge present in chunk."""
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encodes string to token IDs."""
        return [
            t
            for c in self.pattern.findall(text)
            for t in self._encode_chunk(list(c.encode("utf-8")))
        ]

    def decode(self, ids: Sequence[int]) -> str:
        """Decodes token IDs back to string."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


class SuperBPETokenizer:
    """
    SuperBPE: Space Travel for Language Models.

    Implements a two-stage pre-tokenization curriculum:
      Stage 1 (Subwords): BPE merges within regex chunks up to subword_vocab_size.
      Stage 2 (Superwords): BPE merges across chunk boundaries, capturing multi-word phrases.
    """

    def __init__(
        self,
        pattern: str = GPT4_PAT,
        special_tokens: list[str] | dict[str, int] | None = None,
    ):
        self.pattern = re.compile(pattern)
        self.subword_merges: dict[tuple[int, int], int] = {}
        self.superword_merges: dict[tuple[int, int], int] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        if isinstance(special_tokens, dict):
            self.special_tokens = dict(special_tokens)
            self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        elif isinstance(special_tokens, (list, tuple)):
            for i, tok in enumerate(special_tokens):
                self.special_tokens[tok] = 100000 + i
                self.inverse_special_tokens[100000 + i] = tok
        self._special_pattern: re.Pattern | None = None
        self._update_special_pattern()

    def _update_special_pattern(self):
        if self.special_tokens:
            escaped = [
                re.escape(k)
                for k in sorted(self.special_tokens.keys(), key=len, reverse=True)
            ]
            self._special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pattern = None

    def register_special_tokens(self, special_tokens: list[str] | dict[str, int]):
        """Registers special tokens (e.g. {'<|endoftext|>': 10000} or ['<|endoftext|>'])."""
        if isinstance(special_tokens, dict):
            self.special_tokens.update(special_tokens)
            self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        elif isinstance(special_tokens, (list, tuple)):
            start_id = max(self.inverse_special_tokens.keys(), default=99999) + 1
            for i, tok in enumerate(special_tokens):
                self.special_tokens[tok] = start_id + i
                self.inverse_special_tokens[start_id + i] = tok
        self._update_special_pattern()

    def train(self, text: str, vocab_size: int, subword_ratio: float = 0.7):
        """
        Trains SuperBPE tokenizer using a two-stage curriculum.

        Args:
            text: Training corpus.
            vocab_size: Target vocabulary size (including 256 byte tokens and special tokens).
            subword_ratio: Ratio of total learned merges dedicated to Stage 1 (subwords).
        """
        assert vocab_size >= 256, "vocab_size must be >= 256"
        num_special = len(self.special_tokens)
        bpe_vocab_target = vocab_size - num_special
        assert bpe_vocab_target >= 256, "vocab_size too small for byte alphabet and special tokens"

        total_merges = bpe_vocab_target - 256
        n_subword = int(total_merges * subword_ratio)
        n_superword = total_merges - n_subword

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.subword_merges = {}
        self.superword_merges = {}

        # Place special tokens at top of vocab [bpe_vocab_target .. vocab_size - 1]
        for i, st in enumerate(sorted(self.special_tokens.keys())):
            st_id = bpe_vocab_target + i
            self.special_tokens[st] = st_id
            self.inverse_special_tokens[st_id] = st
        self._update_special_pattern()

        # -----------------------------------------------------------------
        # Stage 1: Subword Learning (chunk-constrained merges)
        # -----------------------------------------------------------------
        chunks = [list(c.encode("utf-8")) for c in self.pattern.findall(text)]
        for i in range(n_subword):
            stats = Counter()
            for c in chunks:
                get_stats(c, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = [merge(c, pair, idx) for c in chunks]
            self.subword_merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        # -----------------------------------------------------------------
        # Stage 2: Superword Learning (boundary-relaxed cross-word merges)
        # -----------------------------------------------------------------
        corpus_ids = [tok for c in chunks for tok in c]
        for i in range(n_superword):
            stats = get_stats(corpus_ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + len(self.subword_merges) + i
            corpus_ids = merge(corpus_ids, pair, idx)
            self.superword_merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        self.merges = {**self.subword_merges, **self.superword_merges}

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.subword_merges.get(p, float("inf")))
            if pair not in self.subword_merges:
                break
            ids = merge(ids, pair, self.subword_merges[pair])
        return ids

    def _encode_text_segment(self, text: str) -> list[int]:
        # Stage 1: Encode within regex chunks
        chunks = [list(c.encode("utf-8")) for c in self.pattern.findall(text)]
        ids = [t for c in chunks for t in self._encode_chunk(c)]

        # Stage 2: Apply superword merges across flattened token sequence
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.superword_merges.get(p, float("inf")))
            if pair not in self.superword_merges:
                break
            ids = merge(ids, pair, self.superword_merges[pair])
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encodes text into token IDs, handling special tokens and superwords."""
        if not text:
            return []
        if not allowed_special or not self._special_pattern:
            return self._encode_text_segment(text)

        parts = self._special_pattern.split(text)
        tokens: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                tokens.extend(self._encode_text_segment(part))
        return tokens

    def decode(self, ids: Sequence[int]) -> str:
        """Decodes token IDs into string, lossless UTF-8 roundtrip."""
        parts: list[str] = []
        byte_chunks: list[bytes] = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                if byte_chunks:
                    parts.append(b"".join(byte_chunks).decode("utf-8", errors="replace"))
                    byte_chunks = []
                parts.append(self.inverse_special_tokens[idx])
            elif idx in self.vocab:
                byte_chunks.append(self.vocab[idx])
            else:
                byte_chunks.append("\ufffd".encode("utf-8"))
        if byte_chunks:
            parts.append(b"".join(byte_chunks).decode("utf-8", errors="replace"))
        return "".join(parts)

    def save(self, file_prefix: str):
        """Saves tokenizer state to JSON file."""
        model_data = {
            "subword_merges": [list(k) + [v] for k, v in self.subword_merges.items()],
            "superword_merges": [list(k) + [v] for k, v in self.superword_merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(file_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

    def load(self, file_prefix: str):
        """Loads tokenizer state from JSON file."""
        with open(file_prefix + ".json", "r", encoding="utf-8") as f:
            model_data = json.load(f)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.subword_merges = {}
        for p0, p1, idx in model_data["subword_merges"]:
            self.subword_merges[(p0, p1)] = idx
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.superword_merges = {}
        for p0, p1, idx in model_data["superword_merges"]:
            self.superword_merges[(p0, p1)] = idx
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.merges = {**self.subword_merges, **self.superword_merges}
        self.special_tokens = model_data.get("special_tokens", {})
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self._update_special_pattern()


if __name__ == "__main__":
    sample = (
        "in the long run, by the way, in the long run! " * 20
        + "Building LLMs and VLMs from scratch with pure PyTorch. " * 5
        + "Hello world! こんにちは 🌍 12345!"
    )
    tokenizer = SuperBPETokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 10000})
    tokenizer.train(sample, vocab_size=350, subword_ratio=0.6)

    print(f"Subword merges: {len(tokenizer.subword_merges)}")
    print(f"Superword merges: {len(tokenizer.superword_merges)}")

    cross_spaces = [
        tokenizer.vocab[idx]
        for idx in tokenizer.superword_merges.values()
        if b" " in tokenizer.vocab[idx]
    ]
    print(f"Learned {len(cross_spaces)} multi-word superwords.")
    assert len(cross_spaces) > 0

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample
    print("SuperBPE tests passed successfully!")
