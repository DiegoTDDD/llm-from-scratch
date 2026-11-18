"""Byte-level Byte Pair Encoding (BPE), as in GPT-2, written from scratch.

Text is first split into chunks by a regex (words with their leading space,
numbers, punctuation runs, whitespace), so merges never cross word boundaries.
Each chunk becomes its UTF-8 bytes, which makes every possible string
encodable: the 256 single bytes are always in the vocabulary. Training then
repeatedly merges the most frequent adjacent pair into a new token.

Training counts pairs over the *distinct* chunks weighted by their frequency and
updates those counts incrementally after each merge (only words containing the
merged pair change), so it stays fast on tens of millions of characters in pure
Python. tests/test_tokenizer.py checks it learns exactly the merges of the naive
recount-everything algorithm.
"""

from __future__ import annotations

import heapq
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Simplified GPT-2 pre-tokenizer: contractions, letters, up to 3 digits, symbols, whitespace.
SPLIT_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+", re.UNICODE
)
Pair = tuple[int, int]


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE. Token ids 0-255 are raw bytes; 256+ are learned merges, in order."""

    def __init__(self, merges: dict[Pair, int] | None = None) -> None:
        self.merges: dict[Pair, int] = dict(merges or {})
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        """Learn `vocab_size - 256` merges from `text`."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")
        chunk_counts = Counter(SPLIT_PATTERN.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        # Pair counts and an index of which words contain each pair. After a merge only the
        # words containing the merged pair are touched, instead of recounting the corpus.
        counts: Counter[Pair] = Counter()
        where: dict[Pair, set[int]] = defaultdict(set)
        for wi, (ids, freq) in enumerate(zip(words, freqs, strict=True)):
            for pair in itertools.pairwise(ids):
                counts[pair] += freq
                where[pair].add(wi)
        # Max-heap on (count, then smallest pair) so training is deterministic; stale entries
        # are skipped lazily when popped.
        heap = [(-c, p) for p, c in counts.items()]
        heapq.heapify(heap)

        merges: dict[Pair, int] = {}
        for new_id in range(256, vocab_size):
            best = None
            while heap:
                neg, pair = heapq.heappop(heap)
                if counts.get(pair, 0) == -neg and neg < 0:
                    best = pair
                    break
            if best is None:
                break  # every chunk is already a single token
            merges[best] = new_id
            touched: set[Pair] = set()
            for wi in where.pop(best):
                ids, freq = words[wi], freqs[wi]
                for pair in itertools.pairwise(ids):
                    counts[pair] -= freq
                    touched.add(pair)
                ids = merge_pair(ids, best, new_id)
                words[wi] = ids
                for pair in itertools.pairwise(ids):
                    counts[pair] += freq
                    where[pair].add(wi)
                    touched.add(pair)
            for pair in touched:
                if counts[pair] > 0:
                    heapq.heappush(heap, (-counts[pair], pair))
                else:
                    del counts[pair]
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge present, exactly as training did.
            pair = min(itertools.pairwise(ids),
                       key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_pair(ids, pair, self.merges[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for chunk in SPLIT_PATTERN.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {"type": "byte-bpe", "merges": [[a, b, i] for (a, b), i in self.merges.items()]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({(a, b): i for a, b, i in data["merges"]})
