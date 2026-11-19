# llm-from-scratch

[![tests](https://github.com/DiegoTDDD/llm-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/DiegoTDDD/llm-from-scratch/actions/workflows/tests.yml)
[![train](https://github.com/DiegoTDDD/llm-from-scratch/actions/workflows/train.yml/badge.svg)](https://github.com/DiegoTDDD/llm-from-scratch/actions/workflows/train.yml)

A GPT-style language model written from scratch in PyTorch: byte-level BPE tokenizer, causal
self-attention, training loop, sampling (greedy, temperature, top-k, top-p) and KV-cache
decoding. It is trained on Business, Risk Factors and MD&A sections of SEC 10-K filings, on the
free CPU runner of GitHub Actions, so the whole run costs nothing and anyone can reproduce it.

## Problem

Most LLM projects call an API or fine-tune a model someone else built, which shows how to use
a model but not how one works. When something breaks in production, such as a tokenizer
mismatch, a cache bug or a sampling setting, you need to understand the parts.

## Solution

Every part of a decoder-only transformer, built and tested on its own:

| Part | File | What the tests prove |
| --- | --- | --- |
| Byte-level BPE tokenizer | `llm_from_scratch/tokenizer.py` | Lossless on any string (emoji, CJK, control characters); deterministic; incremental trainer learns exactly the merges of the naive algorithm |
| Causal self-attention, pre-norm blocks, tied embeddings | `llm_from_scratch/model.py` | Changing future tokens never changes earlier logits; attention above the diagonal is exactly 0; parameter count matches the formula |
| KV cache | `llm_from_scratch/model.py` | Cached logits equal a full forward pass, token by token and after a prompt prefill; greedy output identical with and without cache |
| Sampling | `llm_from_scratch/sampling.py` | top-k and top-p keep exactly the right tokens, sorted or not; seeded sampling is reproducible |
| Training | `llm_from_scratch/training.py` | Warmup + cosine schedule; a tiny model learns real text; time budget is respected |

```
SEC EDGAR 10-K sections ──► corpus.txt ──► BPE (4,096 tokens, trained on the train split only)
                                              │
                                              ▼
              token stream ──► 90% train / 10% validation (held-out tail)
                                              │
                                              ▼
   GPT: 4 layers × 4 heads, width 256, context 256, tied embeddings (~4.3M parameters)
   AdamW, warmup + cosine decay, grad clipping, best-validation checkpoint
                                              │
                                              ▼
   results/: metrics.json, loss_curve.png, samples.md, benchmark.md   +   checkpoint in a release
```

## Results

<!-- RESULTS:START -->
From training run [44395273109](https://github.com/DiegoTDDD/llm-from-scratch/actions/runs/44395273109)
on a free GitHub runner, 2 PyTorch threads (1 h 58 min, 3,000 steps, 24.6M tokens seen). Full numbers in
[`results/metrics.json`](results/metrics.json).

| | |
| --- | --- |
| Corpus | 13.3M characters: Business, Risk Factors and MD&A of the two latest 10-Ks of 30 large US companies ([manifest](results/corpus_manifest.json)) |
| Tokenizer | 4,096-token byte-level BPE, trained in 2.6 s; 4.29 characters per token; 3.09M tokens |
| Model | 4.27M parameters (4 layers × 4 heads, width 256, context 256, tied embeddings) |
| **Validation perplexity** | **32.0** (loss 3.47, best at step 2,700) vs **837** for a unigram baseline on the same tokens |
| Bits per character | 1.17 on held-out text |
| KV cache | 2.5× faster at 32 new tokens, **5.8× at 224**, identical tokens in every case |

![Training and validation loss](results/loss_curve.png)

Validation loss flattens after about step 2,200 while training loss keeps falling: at 4.3M
parameters and 3M tokens the model is starting to memorise, and the best checkpoint is kept
by validation loss, not taken from the last step.

KV-cache benchmark ([`results/benchmark.md`](results/benchmark.md)), greedy decoding on 2 CPU threads:

| New tokens | With cache (tok/s) | Without (tok/s) | Speed-up |
| --- | --- | --- | --- |
| 32 | 542 | 218 | 2.49× |
| 64 | 553 | 180 | 3.08× |
| 128 | 545 | 132 | 4.12× |
| 224 | 538 | 93 | 5.78× |

With the cache the speed stays flat as the text grows; without it every new token recomputes
the whole prefix.

Samples ([`results/samples.md`](results/samples.md), 60 tokens, fixed seed). The model has
learned the register of a 10-K; it has not learned facts, and greedy decoding shows the classic
repetition loop that sampling avoids:

> **top-p 0.9:** Our business is subject to risks that are substantial and risks related to
> intellectual property, including claims that are outside of the United States. These claims
> have been developed and may lead to significant technical compliance costs…

> **greedy:** Interest rates, interest rates, interest rates, interest rates, …

The best checkpoint and the tokenizer are attached to the
[release](https://github.com/DiegoTDDD/llm-from-scratch/releases/tag/model-44395273109).
<!-- RESULTS:END -->

## Design decisions

- [ADR 0001](docs/adr/0001-attention-written-by-hand.md): attention written by hand instead of the fused PyTorch kernel, so the mask can be tested.
- [ADR 0002](docs/adr/0002-byte-level-bpe.md): byte-level BPE with incremental pair counts (11M characters in ~17 s, pure Python).
- [ADR 0003](docs/adr/0003-a-model-sized-for-a-free-cpu-runner.md): a ~4.3M-parameter model sized for a free 4-core runner.
- [ADR 0004](docs/adr/0004-kv-cache.md): KV cache, verified against full recomputation and benchmarked.

## Run it

```bash
pip install -r requirements.txt -r requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
pytest -q                                   # 38 tests, a few seconds on a CPU

export SEC_IDENTITY="Your Name your@email.com"   # SEC asks automated clients to identify themselves
python scripts/prepare_corpus.py            # data/corpus.txt + data/corpus_manifest.json
python scripts/train.py --max-minutes 60    # results/ and checkpoints/gpt_sec.pt
python scripts/benchmark.py                 # results/benchmark.md
```

Or run the `train` workflow from the Actions tab (it needs a `SEC_IDENTITY` repository secret
with your name and email, which SEC requires of automated clients): it builds the corpus, trains, benchmarks,
commits `results/` and publishes the checkpoint and tokenizer as a release.

## Limits

- At ~4.3M parameters and a few hours of CPU, the model writes fluent-looking 10-K prose, not
  facts. It is a study of the mechanism, not an assistant.
- No sliding window: generation stops at the 256-token context.
- Attention is O(T²) and unfused; fine at this size, the first thing to replace at scale.

## Stack

Python 3.11, PyTorch (CPU), edgartools for SEC EDGAR, matplotlib, pytest, ruff, GitHub Actions.

## License

MIT
