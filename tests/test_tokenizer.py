"""Byte-level BPE: lossless, deterministic, compressing, and identical after save/load."""

import itertools

import pytest
from conftest import SAMPLE_TEXT

from llm_from_scratch.tokenizer import SPLIT_PATTERN, BPETokenizer, merge_pair


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(SAMPLE_TEXT, vocab_size=400)


def test_merge_replaces_non_overlapping_pairs():
    assert merge_pair([1, 2, 1, 2, 2], (1, 2), 9) == [9, 9, 2]
    assert merge_pair([1, 1, 1], (1, 1), 7) == [7, 1]  # left to right, no overlap


def test_round_trip_is_lossless_on_any_text(tok):
    for text in [SAMPLE_TEXT, "", " ", "unseen word zyxq 123456", "emoji 🚀 and 中文",
                 "tabs\tand\nnewlines\n\n", "\x00 control   nbsp"]:
        assert tok.decode(tok.encode(text)) == text


def test_bytes_alone_still_encode_everything():
    empty = BPETokenizer()
    assert empty.vocab_size == 256
    assert empty.encode("é") == list("é".encode())


def test_training_learns_the_requested_merges_and_compresses(tok):
    assert tok.vocab_size == 400
    assert len(tok.encode(SAMPLE_TEXT)) < 0.5 * len(SAMPLE_TEXT.encode("utf-8"))
    # A frequent word in the sample becomes a single token.
    assert len(tok.encode(" Company")) == 1


def test_training_is_deterministic():
    a = BPETokenizer.train(SAMPLE_TEXT, 300)
    b = BPETokenizer.train(SAMPLE_TEXT, 300)
    assert a.merges == b.merges


def test_merges_never_cross_word_boundaries(tok):
    for token_id in range(256, tok.vocab_size):
        piece = tok.vocab[token_id].decode("utf-8", errors="ignore")
        # Every learned token is a piece of one pre-tokenizer chunk.
        assert len(SPLIT_PATTERN.findall(piece)) <= 1 or not piece.strip()


def test_save_and_load_give_the_same_encoding(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    again = BPETokenizer.load(path)
    assert again.merges == tok.merges
    assert again.encode(SAMPLE_TEXT) == tok.encode(SAMPLE_TEXT)


def test_vocab_size_below_the_byte_alphabet_is_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", 100)


def naive_train(text: str, vocab_size: int) -> dict:
    """Reference implementation: recount every pair after every merge."""
    from collections import Counter

    chunk_counts = Counter(SPLIT_PATTERN.findall(text))
    words = [list(c.encode("utf-8")) for c in chunk_counts]
    freqs = list(chunk_counts.values())
    merges = {}
    for new_id in range(256, vocab_size):
        pairs = Counter()
        for ids, f in zip(words, freqs, strict=True):
            for pair in itertools.pairwise(ids):
                pairs[pair] += f
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges[best] = new_id
        words = [merge_pair(ids, best, new_id) for ids in words]
    return merges


@pytest.mark.parametrize("vocab_size", [300, 600, 2000])
def test_incremental_training_matches_the_naive_algorithm(vocab_size):
    text = SAMPLE_TEXT + " aaaa bbbb abab ababab 1234 1234 12345 ...!!! " * 3
    assert BPETokenizer.train(text, vocab_size).merges == naive_train(text, vocab_size)
