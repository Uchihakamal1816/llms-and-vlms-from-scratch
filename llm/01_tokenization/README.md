# Phase 1: Tokenization (Byte-Pair Encoding from Scratch)

## Objective
Implement a Byte-Pair Encoding (BPE) tokenizer from first principles without using third-party tokenizer packages during execution.

---

## 📚 Concepts to Study & Implement

1. **UTF-8 Byte Representation**:
   - Every string is an array of bytes `[0, 255]`.
   - Starting with a base vocabulary of 256 byte tokens ensures the tokenizer can encode **any arbitrary unicode text** with zero out-of-vocabulary (OOV) tokens.

2. **BPE Training Algorithm**:
   - Given a corpus of raw text converted to integers `[0..255]`:
   - Compute frequency of all adjacent pairs `(id1, id2)`.
   - Find the most frequent pair `(id_a, id_b)`.
   - Create a new token ID `vocab_size + i` representing that pair.
   - Replace all occurrences of `(id_a, id_b)` in the corpus with the new token ID.
   - Repeat until desired `vocab_size` (e.g. 512, 1024, or 4096) is reached.

3. **Inference (Encode & Decode)**:
   - **Encode**: Given a new input string, convert to initial byte IDs, and iteratively apply learned merges in chronological order of rank until no more merges apply.
   - **Decode**: Given a list of token IDs, lookup the underlying byte sequences from the vocabulary table and decode as UTF-8.

4. **Special Tokens**:
   - `<|endoftext|>` / `<|pad|>` / `<|bos|>` / `<|eos|>`.
   - Handling regex splits (e.g. GPT-4 pattern) so punctuation and words are tokenized predictably.

## 🚀 Advanced Tokenization: SuperBPE (Space Travel for Language Models)

Traditional BPE enforces strict whitespace boundaries, breaking text down exclusively into subwords. However, common multi-word expressions (e.g., *"by the way"*, *"in the long run"*) act as single semantic concepts. SuperBPE relaxes this constraint via a **two-stage pre-tokenization curriculum**:

1. **Stage 1 (Subword Learning)**: Enforces regex/whitespace pre-tokenization (`GPT4_PAT`) to learn foundational subwords within word boundaries up to `subword_vocab_size`.
2. **Stage 2 (Superword Learning)**: Relaxes whitespace boundaries. Merges continue across word boundaries on the concatenated token stream to discover multi-word **"superwords"**.
3. **Inference & Compression**: Encodes subwords first within chunks, then merges superwords across chunks. Achieves up to 30-40% fewer tokens per sequence with zero changes to downstream transformer architectures.

---

## 🧪 Verification Checkpoints
- [x] Round-trip invariance: `decode(encode(text)) == text` for all ASCII, Unicode, emoji, and multilingual test strings.
- [x] Compression ratio: Verified SuperBPE token compression advantage over standard BPE.
- [x] Multi-word superwords: Discovered cross-whitespace tokens (e.g. `b" in the long run"`).
- [x] Special tokens handling: Seamless preservation of `<|endoftext|>`, `<|pad|>`.
- [x] Unknown character robustness: Zero crashes on unseen Unicode characters via UTF-8 byte base alphabet.

