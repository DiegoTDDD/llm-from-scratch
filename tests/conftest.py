import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_from_scratch.model import GPT, GPTConfig

SAMPLE_TEXT = (Path(__file__).parent / "fixtures" / "sample_10k.txt").read_text(encoding="utf-8")


@pytest.fixture
def tiny_cfg() -> GPTConfig:
    return GPTConfig(vocab_size=97, context_length=32, n_layer=2, n_head=4, d_model=32)


@pytest.fixture
def tiny_model(tiny_cfg) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_cfg).eval()
