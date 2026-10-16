"""Re-order a retriever's candidates with a cross-encoder.

A bi-encoder embeds question and passage separately and compares vectors; a
cross-encoder reads them together, so it can weigh one against the other word
by word. It is far too slow to run over the whole corpus, and far better at
ordering a shortlist - hence: hybrid search proposes, the cross-encoder
decides.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nsu_rag_llm.retrieval.base import Hit, Retriever


class PairScorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    """BAAI/bge-reranker-v2-m3 through sentence-transformers."""

    def __init__(self, model_name: str, device: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        # A chunk and its context line stay under ~490 tokens, so 512 fits a
        # question plus the passage without truncating the passage.
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        return self.model.predict(pairs, batch_size=16, show_progress_bar=False).tolist()


def rerank(scorer: PairScorer, query: str, shortlist: Sequence[Hit]) -> list[Hit]:
    """The shortlist in the cross-encoder's order, with its scores."""
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
