"""Training loop: batches from a token stream, AdamW, warmup + cosine schedule, validation.

Kept free of file paths and plotting so tests can train a tiny model in seconds
(tests/test_training.py); scripts/train.py wires it to the real corpus.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from llm_from_scratch.model import GPT

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 20
    max_minutes: float | None = None  # stop early when the time budget runs out
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)  # one row per evaluation
    best_val_loss: float = math.inf
    best_step: int = 0
    steps_run: int = 0
    seconds: float = 0.0
    tokens_seen: int = 0


def split_tokens(tokens: Tensor, val_fraction: float = 0.1) -> tuple[Tensor, Tensor]:
    """Hold out the last `val_fraction` of the stream, so validation text is never trained on."""
    cut = int(len(tokens) * (1 - val_fraction))
    return tokens[:cut], tokens[cut:]


def get_batch(tokens: Tensor, batch_size: int, context: int,
              generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Random windows of `context` tokens; targets are the same windows shifted by one."""
    if len(tokens) <= context:
        raise ValueError("token stream is shorter than the context length")
    starts = torch.randint(len(tokens) - context, (batch_size,), generator=generator)
    x = torch.stack([tokens[s:s + context] for s in starts.tolist()])
    y = torch.stack([tokens[s + 1:s + 1 + context] for s in starts.tolist()])
    return x, y


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from learning_rate to min_learning_rate."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (not on biases, norms or embeddings' 1-D params)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(0.9, 0.95),
    )


@torch.no_grad()
def estimate_loss(model: GPT, tokens: Tensor, cfg: TrainConfig,
                  generator: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(tokens, cfg.batch_size, model.cfg.context_length, generator)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def train(model: GPT, train_tokens: Tensor, val_tokens: Tensor, cfg: TrainConfig,
          on_best: callable | None = None) -> TrainResult:
    """Train in place. `on_best(step, val_loss)` is called whenever validation improves."""
    torch.manual_seed(cfg.seed)
    batches = torch.Generator().manual_seed(cfg.seed)
    evals = torch.Generator().manual_seed(cfg.seed + 1)
    optimizer = make_optimizer(model, cfg)
    result = TrainResult()
    start = time.perf_counter()
    model.train()
    window_loss, window_steps = 0.0, 0
    for step in range(cfg.steps):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_tokens, cfg.batch_size, model.cfg.context_length, batches)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        window_loss += loss.item()
        window_steps += 1
        result.steps_run = step + 1
        result.tokens_seen += x.numel()

        last = step == cfg.steps - 1
        out_of_time = (cfg.max_minutes is not None
                       and time.perf_counter() - start > cfg.max_minutes * 60)
        if (step + 1) % cfg.eval_every == 0 or last or out_of_time:
            val_loss = estimate_loss(model, val_tokens, cfg, evals)
            row = {"step": step + 1, "train_loss": window_loss / window_steps,
                   "val_loss": val_loss, "lr": lr,
                   "elapsed_s": round(time.perf_counter() - start, 1)}
            result.history.append(row)
            log.info("step %5d  train %.3f  val %.3f  lr %.2e  %ss", row["step"],
                     row["train_loss"], val_loss, lr, row["elapsed_s"])
            window_loss, window_steps = 0.0, 0
            if val_loss < result.best_val_loss:
                result.best_val_loss, result.best_step = val_loss, step + 1
                if on_best is not None:
                    on_best(step + 1, val_loss)
        if out_of_time:
            log.info("time budget of %s minutes reached at step %d", cfg.max_minutes, step + 1)
            break
    result.seconds = time.perf_counter() - start
    return result
