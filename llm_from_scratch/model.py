"""A decoder-only transformer (GPT style) with an explicit KV cache.

Attention is written out by hand (no nn.MultiheadAttention, no
scaled_dot_product_attention) so every step is visible and testable:

    scores = Q K^T / sqrt(d_head)  ->  hide the future  ->  softmax  ->  @ V

During generation the keys and values of past positions never change, so the
cache keeps them and each step only computes Q, K and V for the new token.
tests/test_model.py proves both properties: the mask hides the future, and
cached decoding gives the same logits as recomputing everything.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# One (keys, values) pair per layer, each shaped (batch, heads, positions, head_dim).
LayerCache = tuple[Tensor, Tensor]
KVCache = list[LayerCache]


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    context_length: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        # future[i, j] is True where position i must NOT see position j (j > i).
        n = cfg.context_length
        self.register_buffer("future", torch.triu(torch.ones(n, n, dtype=torch.bool), 1),
                             persistent=False)
        self.last_attention: Tensor | None = None  # kept for inspection and tests

    def forward(self, x: Tensor, past: LayerCache | None = None) -> tuple[Tensor, LayerCache]:
        batch, new, _ = x.shape
        q, k, v = self.qkv(x).split(self.cfg.d_model, dim=2)
        # (batch, new, d_model) -> (batch, heads, new, head_dim)
        q, k, v = (t.view(batch, new, self.cfg.n_head, self.cfg.head_dim).transpose(1, 2)
                   for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        total = k.shape[2]
        start = total - new  # absolute position of the first new token

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)
        scores = scores.masked_fill(self.future[start:total, :total], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach()
        out = self.dropout(weights) @ v
        out = out.transpose(1, 2).contiguous().view(batch, new, self.cfg.d_model)
        return self.proj(out), (k, v)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor, past: LayerCache | None = None) -> tuple[Tensor, LayerCache]:
        attended, present = self.attn(self.ln1(x), past)
        x = x + attended
        return x + self.mlp(self.ln2(x)), present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: one matrix for input and output
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx: Tensor, targets: Tensor | None = None,
                cache: KVCache | None = None) -> tuple[Tensor, Tensor | None, KVCache]:
        """Logits for every new position, the loss if targets are given, and the updated cache."""
        start = cache[0][0].shape[2] if cache else 0
        end = start + idx.shape[1]
        if end > self.cfg.context_length:
            raise ValueError(f"{end} tokens exceed context_length {self.cfg.context_length}")
        positions = torch.arange(start, end, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        new_cache: KVCache = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, cache[i] if cache else None)
            new_cache.append(present)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_cache
