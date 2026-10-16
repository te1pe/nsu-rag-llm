"""BM25 over the chunk file, written out rather than imported.

    score(q, d) = sum over terms t of q:
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

k1 = 1.2 and b = 0.75 are Lucene's defaults, and the idf is Lucene's too: it
never goes negative, so a term present in most chunks adds a little rather
than subtracting. The index is an inverted list per term held in numpy
arrays; a query touches only the postings of its own terms (ADR-022).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from nsu_rag_llm.retrieval.base import Hit
from nsu_rag_llm.retrieval.text import tokenize

K1 = 1.2
B = 0.75


class BM25Index:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        # The context line ("course › document › section") is indexed with the
        # text, as it is embedded with it: a section title is often the best
        # match a question has.
        documents = [tokenize(f"{c['context']}\n{c['text']}") for c in chunks]
        self.lengths = np.array([len(d) for d in documents], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if len(documents) else 0.0
        self.courses = np.array([c["course"] for c in chunks])

        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for index, tokens in enumerate(documents):
            for term, frequency in Counter(tokens).items():
                ids, frequencies = postings[term]
                ids.append(index)
                frequencies.append(frequency)
        total = len(documents)
        self.postings = {
            term: (np.array(ids, dtype=np.int32), np.array(freqs, dtype=np.float32))
            for term, (ids, freqs) in postings.items()
        }
        self.idf = {
            term: math.log(1 + (total - len(ids) + 0.5) / (len(ids) + 0.5))
            for term, (ids, _) in self.postings.items()
        }

    @classmethod
    def from_file(cls, path: Path) -> BM25Index:
        with path.open(encoding="utf-8") as handle:
            return cls([json.loads(line) for line in handle if line.strip()])

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every chunk; terms repeated in the query count once."""
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, frequencies = self.postings[term]
            norm = K1 * (1 - B + B * self.lengths[ids] / self.average_length)
            scores[ids] += self.idf[term] * frequencies * (K1 + 1) / (frequencies + norm)
        return scores

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        scores = self.scores(query)
        if course is not None:
            scores[self.courses != course] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if not len(candidates):
            return []
        top = candidates[np.argsort(-scores[candidates], kind="stable")[:k]]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]
