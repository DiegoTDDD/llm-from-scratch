"""The training loop learns: on real text the loss falls well below the untrained level."""

import math

import pytest
import torch
from conftest import SAMPLE_TEXT

from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.tokenizer import BPETokenizer
from llm_from_scratch.training import TrainConfig, get_batch, learning_rate, split_tokens, train


def test_batches_are_windows_shifted_by_one():
    stream = torch.arange(100)
    x, y = get_batch(stream, 4, 8, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_validation_is_the_tail_of_the_stream():
    train_part, val_part = split_tokens(torch.arange(100), 0.1)
    assert train_part[-1] == 89 and val_part[0] == 90 and len(val_part) == 10


def test_learning_rate_warms_up_then_decays():
    cfg = TrainConfig(steps=100, learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10)
    assert learning_rate(0, cfg) == pytest.approx(1e-4)
    assert learning_rate(9, cfg) == pytest.approx(1e-3)
    assert learning_rate(99, cfg) == pytest.approx(1e-4, rel=0.01)
    assert learning_rate(50, cfg) < learning_rate(20, cfg)


def test_a_tiny_model_learns_real_text():
    tok = BPETokenizer.train(SAMPLE_TEXT, 320)
    ids = torch.tensor(tok.encode(SAMPLE_TEXT * 4))
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, context_length=32, n_layer=2, n_head=2,
                          d_model=48))
    cfg = TrainConfig(steps=150, batch_size=16, learning_rate=3e-3, warmup_steps=10,
                      eval_every=50, eval_batches=4)
    best_steps = []
    result = train(model, ids, ids, cfg, on_best=lambda s, v: best_steps.append(s))
    untrained = math.log(tok.vocab_size)
    assert result.history[0]["train_loss"] > result.history[-1]["train_loss"]
    assert result.best_val_loss < 0.6 * untrained  # memorising a short text is easy
    assert best_steps and best_steps[-1] == result.best_step
    assert result.tokens_seen == 150 * 16 * 32


def test_time_budget_stops_training_early():
    ids = torch.randint(0, 50, (500,))
    model = GPT(GPTConfig(vocab_size=50, context_length=16, n_layer=1, n_head=1, d_model=16))
    result = train(model, ids, ids, TrainConfig(steps=10_000, batch_size=2, eval_batches=1,
                                                max_minutes=0.0))
    assert result.steps_run == 1
    assert len(result.history) == 1
