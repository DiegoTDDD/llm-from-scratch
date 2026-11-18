"""Measure generation speed with and without the KV cache, and check they agree.

For several output lengths, greedy-decode the same prompt both ways, check the
tokens are identical, and record tokens per second (best of 3 runs). Uses the
trained checkpoint if present, otherwise a randomly initialised model of the
same size (speed does not depend on the weights).

Writes results/benchmark.json and results/benchmark.md.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_from_scratch.model import GPT, GPTConfig  # noqa: E402
from llm_from_scratch.sampling import generate  # noqa: E402

log = logging.getLogger(__name__)

CHECKPOINT = ROOT / "checkpoints" / "gpt_sec.pt"
RESULTS = ROOT / "results"
PROMPT_TOKENS = 16
LENGTHS = [32, 64, 128, 224]
REPEATS = 3


def load_model() -> tuple[GPT, str]:
    if CHECKPOINT.exists():
        ckpt = torch.load(CHECKPOINT)
        model = GPT(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        return model, "trained checkpoint"
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=4096, context_length=256, n_layer=4, n_head=4,
                         d_model=256)), "random weights"


def best_time(fn: callable) -> float:
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model, source = load_model()
    model.eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, PROMPT_TOKENS),
                           generator=torch.Generator().manual_seed(0))
    rows = []
    for n in LENGTHS:
        with_cache = generate(model, prompt, n, greedy=True, use_cache=True)
        without = generate(model, prompt, n, greedy=True, use_cache=False)
        identical = torch.equal(with_cache, without)
        t_cache = best_time(lambda n=n: generate(model, prompt, n, greedy=True, use_cache=True))
        t_full = best_time(lambda n=n: generate(model, prompt, n, greedy=True, use_cache=False))
        rows.append({"new_tokens": n, "with_cache_tok_s": round(n / t_cache, 1),
                     "without_cache_tok_s": round(n / t_full, 1),
                     "speedup": round(t_full / t_cache, 2), "identical_output": identical})
        log.info("%4d tokens: cache %.0f tok/s, no cache %.0f tok/s, %.2fx, identical=%s",
                 n, n / t_cache, n / t_full, t_full / t_cache, identical)
    report = {"weights": source, "device": "cpu", "threads": torch.get_num_threads(),
              "torch": torch.__version__, "prompt_tokens": PROMPT_TOKENS,
              "parameters": model.num_parameters(), "rows": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"Greedy decoding on CPU ({torch.get_num_threads()} threads), {source}, "
             f"{PROMPT_TOKENS}-token prompt, best of {REPEATS} runs.", "",
             "| New tokens | With KV cache (tok/s) | Without (tok/s) | Speed-up | Same tokens |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['new_tokens']} | {r['with_cache_tok_s']} | {r['without_cache_tok_s']} | "
              f"{r['speedup']}x | {'yes' if r['identical_output'] else 'NO'} |" for r in rows]
    (RESULTS / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(r["identical_output"] for r in rows):
        sys.exit("cached and uncached decoding disagree")


if __name__ == "__main__":
    main()
