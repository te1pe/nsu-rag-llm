"""Reciprocal Rank Fusion: combine rankings by position, not by score.

    fused(d) = sum over rankings r containing d of 1 / (k + rank_r(d))

BM25 scores and cosine similarities live on unrelated scales, so adding or
averaging them would need calibration the two retrievers do not offer. RRF
looks only at ranks. k = 60 is the constant from Cormack, Clarke and Büttcher
(2009); it flattens the weight of the very top ranks, so one retriever's
favourite cannot outvote agreement between both.
"""

from __future__ import annotations

from collections.abc import Sequence

from nsu_rag_llm.retrieval.base import Hit, Retriever

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = RRF_K) -> list[Hit]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(hit.id, hit.chunk)
    # Ties are broken by id so that the same inputs always give the same order.
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(chunks[chunk_id], score) for chunk_id, score in ordered]


class HybridRetriever:
    """BM25 and dense search, each asked for `candidates`, fused by RRF."""

    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        candidates: int = 50,
        k: int = RRF_K,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.candidates = candidates
        self.k = k

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        rankings = [
            self.lexical.search(query, self.candidates, course),
            self.dense.search(query, self.candidates, course),
        ]
        return reciprocal_rank_fusion(rankings, self.k)[:k]
