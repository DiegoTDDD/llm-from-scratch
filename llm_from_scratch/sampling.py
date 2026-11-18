"""Decoding: greedy, temperature, top-k and top-p (nucleus), with or without the KV cache."""

from __future__ import annotations

import torch
from torch import Tensor

from llm_from_scratch.model import GPT


def filter_logits(logits: Tensor, temperature: float = 1.0, top_k: int | None = None,
                  top_p: float | None = None) -> Tensor:
    """Divide by the temperature, then keep the top-k tokens and/or the smallest set of tokens
    whose probability reaches top_p. Removed tokens get -inf. Works on (..., vocab) tensors."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy=True for argmax decoding)")
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Drop a token when the tokens ranked above it already reach top_p.
        drop_sorted = (probs.cumsum(dim=-1) - probs) >= top_p
        drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
        logits = logits.masked_fill(drop, float("-inf"))
    return logits


@torch.no_grad()
def generate(model: GPT, idx: Tensor, max_new_tokens: int, greedy: bool = False,
             temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
             use_cache: bool = True, generator: torch.Generator | None = None) -> Tensor:
    """Append `max_new_tokens` tokens to `idx`, shaped (batch, time)."""
    model.eval()
    limit = model.cfg.context_length
    if idx.shape[1] + max_new_tokens > limit:
        raise ValueError(f"prompt + new tokens must fit in context_length ({limit})")
    cache = None
    step_input = idx
    for _ in range(max_new_tokens):
        if use_cache:
            logits, _, cache = model(step_input, cache=cache)  # only the new token(s)
        else:
            logits, _, _ = model(idx)  # the whole sequence, every step
        last = logits[:, -1, :]
        if greedy:
            next_token = last.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(filter_logits(last, temperature, top_k, top_p), dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, next_token], dim=1)
        step_input = next_token
    return idx
