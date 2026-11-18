"""The two properties a decoder must have: no peeking at the future, and a correct KV cache."""

import pytest
import torch

from llm_from_scratch.model import GPT, GPTConfig
from llm_from_scratch.sampling import generate


def tokens(n, vocab=97, seed=0):
    return torch.randint(0, vocab, (2, n), generator=torch.Generator().manual_seed(seed))


def test_output_shapes_and_loss(tiny_model, tiny_cfg):
    x = tokens(10)
    logits, loss, cache = tiny_model(x, targets=x)
    assert logits.shape == (2, 10, tiny_cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert len(cache) == tiny_cfg.n_layer
    assert cache[0][0].shape == (2, tiny_cfg.n_head, 10, tiny_cfg.head_dim)


def test_untrained_loss_is_close_to_uniform(tiny_model, tiny_cfg):
    # Independent targets: with tied weights, targets=x would favour copying the input token.
    _, loss, _ = tiny_model(tokens(32, seed=0), targets=tokens(32, seed=1))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(97.0)).item(), abs=0.3)


def test_changing_future_tokens_never_changes_past_logits(tiny_model):
    """Causal mask: the output at position t depends only on tokens 0..t."""
    x = tokens(20)
    for t in [0, 5, 13, 18]:
        altered = x.clone()
        altered[:, t + 1:] = (altered[:, t + 1:] + 7) % 97  # rewrite everything after t
        before, _, _ = tiny_model(x)
        after, _, _ = tiny_model(altered)
        assert torch.equal(before[:, : t + 1], after[:, : t + 1])
        assert not torch.allclose(before[:, t + 1:], after[:, t + 1:])


def test_attention_weights_are_zero_above_the_diagonal(tiny_model):
    tiny_model(tokens(12))
    weights = tiny_model.blocks[0].attn.last_attention  # (batch, heads, 12, 12)
    upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), 1)
    assert torch.all(weights[..., upper] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))


def test_cached_decoding_matches_full_recompute(tiny_model):
    """Feeding tokens one at a time through the cache gives the same logits as one full pass."""
    x = tokens(24)
    full, _, _ = tiny_model(x)
    cache, steps = None, []
    for t in range(x.shape[1]):
        logits, _, cache = tiny_model(x[:, t:t + 1], cache=cache)
        steps.append(logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)


def test_prompt_prefill_then_cache_matches_full_recompute(tiny_model):
    x = tokens(20)
    full, _, _ = tiny_model(x)
    first, _, cache = tiny_model(x[:, :12])
    rest, _, _ = tiny_model(x[:, 12:], cache=cache)
    torch.testing.assert_close(torch.cat([first, rest], dim=1), full, atol=1e-5, rtol=1e-4)


def test_greedy_generation_is_identical_with_and_without_cache(tiny_model):
    prompt = tokens(5)
    with_cache = generate(tiny_model, prompt, 20, greedy=True, use_cache=True)
    without = generate(tiny_model, prompt, 20, greedy=True, use_cache=False)
    assert torch.equal(with_cache, without)
    assert torch.equal(with_cache[:, :5], prompt)


def test_context_length_is_enforced(tiny_model, tiny_cfg):
    with pytest.raises(ValueError):
        tiny_model(tokens(tiny_cfg.context_length + 1))
    with pytest.raises(ValueError):
        generate(tiny_model, tokens(30), 10)


def test_weights_are_tied_between_input_embedding_and_output(tiny_model):
    assert tiny_model.head.weight is tiny_model.tok_emb.weight


def test_parameter_count_matches_the_formula():
    cfg = GPTConfig(vocab_size=100, context_length=16, n_layer=2, n_head=2, d_model=8)
    d, v, n = cfg.d_model, cfg.vocab_size, cfg.context_length
    norms, qkv, proj = 2 * 2 * d, 3 * d * d + 3 * d, d * d + d
    mlp = (4 * d * d + 4 * d) + (4 * d * d + d)
    per_block = norms + qkv + proj + mlp
    expected = v * d + n * d + cfg.n_layer * per_block + 2 * d
    assert GPT(cfg).num_parameters() == expected


def test_heads_must_divide_the_model_width():
    with pytest.raises(ValueError):
        GPT(GPTConfig(vocab_size=10, d_model=30, n_head=4))
