"""Temperature, top-k and top-p filtering, and seeded sampling."""

import math

import pytest
import torch

from llm_from_scratch.sampling import filter_logits, generate

LOGITS = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.07, 0.03]]))


def kept(logits):
    return torch.isfinite(logits[0]).nonzero().flatten().tolist()


def test_top_k_keeps_exactly_the_k_most_likely():
    assert kept(filter_logits(LOGITS, top_k=2)) == [0, 1]
    assert kept(filter_logits(LOGITS, top_k=99)) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("top_p", "expected"), [
    (0.4, [0]),            # the first token alone already reaches 0.4
    (0.5, [0]),
    (0.6, [0, 1]),         # 0.5 < 0.6, so the second is needed
    (0.89, [0, 1, 2]),     # 0.75 < 0.89 <= 0.90
    (1.0, [0, 1, 2, 3, 4]),
])
def test_top_p_keeps_the_smallest_set_reaching_p(top_p, expected):
    assert kept(filter_logits(LOGITS, top_p=top_p)) == expected


def test_top_p_works_when_logits_are_not_sorted():
    shuffled = LOGITS[:, [3, 0, 4, 2, 1]]
    assert kept(filter_logits(shuffled, top_p=0.6)) == [1, 4]  # the 0.5 and 0.25 tokens


def test_temperature_sharpens_or_flattens():
    probs = lambda t: torch.softmax(filter_logits(LOGITS, temperature=t), -1)[0]  # noqa: E731
    assert probs(0.5)[0] > 0.5 > probs(2.0)[0]
    torch.testing.assert_close(probs(1.0), torch.tensor([0.5, 0.25, 0.15, 0.07, 0.03]))


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        filter_logits(LOGITS, temperature=0)
    with pytest.raises(ValueError):
        filter_logits(LOGITS, top_p=1.5)


def test_sampling_is_reproducible_with_a_seed(tiny_model):
    prompt = torch.zeros((1, 3), dtype=torch.long)
    run = lambda: generate(tiny_model, prompt, 15, temperature=1.0, top_k=10,  # noqa: E731
                           generator=torch.Generator().manual_seed(7))
    assert torch.equal(run(), run())


def test_sampled_tokens_stay_inside_the_top_k(tiny_model):
    prompt = torch.zeros((1, 1), dtype=torch.long)
    logits, _, _ = tiny_model(prompt)
    allowed = set(torch.topk(logits[0, -1], 3).indices.tolist())
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        out = generate(tiny_model, prompt, 1, top_k=3, generator=gen)
        assert out[0, -1].item() in allowed
    assert not math.isnan(logits.sum().item())
