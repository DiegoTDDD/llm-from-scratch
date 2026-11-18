"""Train the tokenizer and the model on the SEC corpus, and write every result to results/.

Steps:
  1. Train a 4,096-token byte-level BPE on the training part of the corpus.
  2. Encode the corpus; hold out the last 10% of tokens for validation.
  3. Train the GPT (CPU-friendly size) and keep the checkpoint with the best validation loss.
  4. Write results/metrics.json, results/loss_curve.png, results/samples.md and
     results/tokenizer.json. The checkpoint goes to checkpoints/ (not versioned;
     the training workflow attaches it to a GitHub release).

Usage:
    python scripts/train.py [--steps 3000] [--max-minutes 40]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_from_scratch.model import GPT, GPTConfig  # noqa: E402
from llm_from_scratch.sampling import generate  # noqa: E402
from llm_from_scratch.tokenizer import BPETokenizer  # noqa: E402
from llm_from_scratch.training import TrainConfig, split_tokens, train  # noqa: E402

log = logging.getLogger(__name__)

CORPUS = ROOT / "data" / "corpus.txt"
RESULTS = ROOT / "results"
CHECKPOINT = ROOT / "checkpoints" / "gpt_sec.pt"
VOCAB_SIZE = 4096
PROMPTS = [
    "Our business is subject to risks",
    "Net revenue increased",
    "We compete with",
    "Interest rates",
]
STRATEGIES = {
    "greedy": {"greedy": True},
    "temperature 0.8": {"temperature": 0.8},
    "top-k 40": {"temperature": 1.0, "top_k": 40},
    "top-p 0.9": {"temperature": 1.0, "top_p": 0.9},
}


def unigram_cross_entropy(train_ids: torch.Tensor, val_ids: torch.Tensor, vocab: int) -> float:
    """Baseline: cross-entropy of a model that only knows how often each token appears."""
    counts = Counter(train_ids.tolist())
    total = len(train_ids) + vocab  # add-one smoothing
    return -sum(math.log((counts.get(t, 0) + 1) / total) for t in val_ids.tolist()) / len(val_ids)


def save_loss_curve(history: list[dict], baseline: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    plt.figure(figsize=(7, 4.2))
    plt.plot(steps, [h["train_loss"] for h in history], label="train")
    plt.plot(steps, [h["val_loss"] for h in history], label="validation")
    plt.axhline(baseline, color="gray", linestyle="--", label="unigram baseline")
    plt.xlabel("step")
    plt.ylabel("cross-entropy (nats per token)")
    plt.title("GPT trained from scratch on SEC 10-K text")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_samples(model: GPT, tok: BPETokenizer, path: Path, new_tokens: int = 60) -> None:
    lines = ["# Samples", "",
             f"{new_tokens} new tokens per prompt from the best checkpoint. Sampled strategies use "
             "a fixed seed, so the file is reproducible.", ""]
    for prompt in PROMPTS:
        lines += [f"## \"{prompt}\"", ""]
        ids = torch.tensor([tok.encode(prompt)])
        for name, kwargs in STRATEGIES.items():
            out = generate(model, ids, new_tokens, generator=torch.Generator().manual_seed(0),
                           **kwargs)
            text = tok.decode(out[0].tolist()).replace("\n", " ")
            lines.append(f"- **{name}:** {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    RESULTS.mkdir(exist_ok=True)
    CHECKPOINT.parent.mkdir(exist_ok=True)

    text = CORPUS.read_text(encoding="utf-8")
    cut = int(len(text) * 0.9)
    t0 = time.perf_counter()
    tok = BPETokenizer.train(text[:cut], VOCAB_SIZE)  # never sees the validation text
    tok_seconds = time.perf_counter() - t0
    tok.save(RESULTS / "tokenizer.json")
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = split_tokens(ids, 0.1)
    log.info("corpus %s chars -> %s tokens (%.2f chars/token); tokenizer trained in %.0fs",
             f"{len(text):,}", f"{len(ids):,}", len(text) / len(ids), tok_seconds)

    model_cfg = GPTConfig(vocab_size=tok.vocab_size, context_length=256, n_layer=4, n_head=4,
                          d_model=256, dropout=0.1)
    train_cfg = TrainConfig(steps=args.steps, max_minutes=args.max_minutes)
    model = GPT(model_cfg)
    log.info("model: %s parameters", f"{model.num_parameters():,}")

    def save_best(step: int, val_loss: float) -> None:
        torch.save({"model": model.state_dict(), "config": model_cfg.to_dict(),
                    "step": step, "val_loss": val_loss}, CHECKPOINT)

    result = train(model, train_ids, val_ids, train_cfg, on_best=save_best)
    model.load_state_dict(torch.load(CHECKPOINT)["model"])
    model.eval()

    baseline = unigram_cross_entropy(train_ids, val_ids, tok.vocab_size)
    chars_per_token = len(text) / len(ids)
    metrics = {
        "corpus_chars": len(text),
        "tokens": len(ids),
        "chars_per_token": round(chars_per_token, 3),
        "vocab_size": tok.vocab_size,
        "tokenizer_train_seconds": round(tok_seconds, 1),
        "parameters": model.num_parameters(),
        "model": model_cfg.to_dict(),
        "training": train_cfg.to_dict(),
        "steps_run": result.steps_run,
        "tokens_seen": result.tokens_seen,
        "train_seconds": round(result.seconds, 1),
        "best_step": result.best_step,
        "best_val_loss": round(result.best_val_loss, 4),
        "best_val_perplexity": round(math.exp(result.best_val_loss), 2),
        "val_bits_per_char": round(result.best_val_loss / math.log(2) / chars_per_token, 4),
        "unigram_val_loss": round(baseline, 4),
        "unigram_val_perplexity": round(math.exp(baseline), 2),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "history": result.history,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_loss_curve(result.history, baseline, RESULTS / "loss_curve.png")
    write_samples(model, tok, RESULTS / "samples.md")
    log.info("best val loss %.3f (perplexity %.1f) at step %d; unigram baseline %.3f",
             result.best_val_loss, math.exp(result.best_val_loss), result.best_step, baseline)


if __name__ == "__main__":
    main()
