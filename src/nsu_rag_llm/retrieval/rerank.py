from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nsu_rag_llm.retrieval.base import Hit, Retriever


class PairScorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    def __init__(self, model_name: str, device: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        return self.model.predict(pairs, batch_size=16, show_progress_bar=False).tolist()


def rerank(scorer: PairScorer, query: str, shortlist: Sequence[Hit]) -> list[Hit]:
    if not shortlist:
        return []
    scores = scorer.score(query, [hit.passage for hit in shortlist])
    ranked = sorted(zip(shortlist, scores, strict=True), key=lambda pair: -pair[1])
    return [Hit(hit.chunk, float(score)) for hit, score in ranked]


class RerankRetriever:
    def __init__(self, base: Retriever, scorer: PairScorer, candidates: int = 30) -> None:
        self.base = base
        self.scorer = scorer
        self.candidates = candidates

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        shortlist = self.base.search(query, self.candidates, course)
        return rerank(self.scorer, query, shortlist)[:k]
