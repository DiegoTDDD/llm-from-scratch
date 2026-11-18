# ADR 0004: KV cache for generation, verified against full recomputation

**Status:** accepted

## Context
Without a cache, generating each token re-runs attention over the whole prefix, so generating
n tokens costs O(n²) forward work. Caches are also where subtle bugs live: an off-by-one in
the position embedding or the mask slice gives plausible but wrong text.

## Decision
Each attention layer returns its keys and values; `GPT.forward` accepts the cache and only
processes new tokens, offsetting position embeddings and slicing the causal mask to the new
rows. `generate` uses the cache by default and can turn it off.

## Consequences
- Tests require cached logits to equal a full forward pass, token by token and after a
  multi-token prompt prefill, and greedy outputs to be identical with and without the cache.
- `scripts/benchmark.py` measures the speedup on the trained model and fails if outputs ever
  differ; the speedup grows with sequence length, as expected.
- Generation is capped at the context length (no sliding window), which keeps the cache simple.
