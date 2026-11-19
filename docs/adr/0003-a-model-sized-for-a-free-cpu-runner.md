# ADR 0003: A ~4.3M-parameter model trained on a free GitHub Actions CPU runner

**Status:** accepted

## Context
The project must cost nothing: no paid GPU. GitHub's free runners give 4 CPU cores and allow
jobs of up to 6 hours, and the run has to be reproducible by anyone who forks the repo.

## Decision
4 layers, 4 heads, width 256, context 256, vocabulary 4,096, tied input/output embeddings
(about 4.3M parameters). AdamW with warmup and cosine decay, gradient clipping, dropout 0.1,
best-validation checkpointing. The `train` workflow builds the corpus, trains with a time
budget, benchmarks, commits `results/` and attaches the checkpoint to a release.

## Consequences
- A full run takes a few hours on the runner and nothing else; anyone can re-run it.
- The model is far too small to be useful as an assistant. The claim is narrower and
  measurable: it learns the corpus (validation loss well below a unigram baseline) and its
  samples read like 10-K prose.
- Weight tying cuts about a third of the parameters at this vocabulary size.
