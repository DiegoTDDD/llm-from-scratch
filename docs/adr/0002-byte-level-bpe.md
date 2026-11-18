# ADR 0002: A byte-level BPE tokenizer trained on the corpus, with incremental pair counts

**Status:** accepted

## Context
A character-level model wastes context on spelling, and borrowing the GPT-2 tokenizer would
skip a part of the stack worth building. The tokenizer has to encode any string, including
symbols and non-English text found in filings.

## Decision
- Byte-level BPE as in GPT-2: the 256 bytes are the base vocabulary, so every string encodes
  and decoding is lossless.
- A GPT-2-style regex splits text first, so merges never cross word boundaries.
- 4,096 tokens, trained only on the training 90% of the corpus, so validation text is unseen.
- Training counts pairs over distinct chunks weighted by frequency, and after each merge updates
  counts only for the words containing the merged pair, using a max-heap with lazy deletion.

## Consequences
- The naive algorithm recounts every pair per merge: O(merges × corpus). The incremental one
  trains 4,096 tokens on 11M characters in about 17 s in pure Python.
- A test checks the incremental trainer learns exactly the same merges as the naive reference,
  so the optimisation cannot silently change the tokenizer.
- Ties are broken by the smallest pair, so training is deterministic.
