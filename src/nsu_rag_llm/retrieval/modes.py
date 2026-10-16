"""The four retrieval modes, built from settings, each model loaded once."""

from __future__ import annotations

from functools import cached_property
from typing import get_args

from nsu_rag_llm.config import RetrievalMode, Settings, resolve_device
from nsu_rag_llm.retrieval.base import Retriever

# Declared in config, where RETRIEVAL_MODE is validated against it.
Mode = RetrievalMode
MODES: tuple[str, ...] = get_args(Mode)


class Retrievers:
    """Builds what a run asks for and nothing more.

    The embedding model, the reranker and the Qdrant client are loaded on
    first use: a BM25-only search must not wait seconds for a model it never
    touches, and an evaluation over all four modes loads each model once.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)

    @cached_property
    def bm25(self) -> Retriever:
        from nsu_rag_llm.retrieval.bm25 import BM25Index

        return BM25Index.from_file(self.settings.chunks_file)

    @cached_property
    def dense(self) -> Retriever:
        from qdrant_client import QdrantClient

        from nsu_rag_llm.retrieval.dense import DenseRetriever, E5QueryEmbedder

        return DenseRetriever(
            QdrantClient(url=self.settings.qdrant_url, timeout=30),
            self.settings.qdrant_collection,
            E5QueryEmbedder(self.settings.embedding_model, self.device),
        )

    @cached_property
    def hybrid(self) -> Retriever:
        from nsu_rag_llm.retrieval.fusion import HybridRetriever

        return HybridRetriever(
            self.bm25,
            self.dense,
            candidates=self.settings.retrieval_candidates,
            k=self.settings.rrf_k,
        )

    @cached_property
    def scorer(self):
        from nsu_rag_llm.retrieval.rerank import CrossEncoderScorer

        return CrossEncoderScorer(self.settings.reranker_model, self.device)

    @cached_property
    def reranked(self) -> Retriever:
        from nsu_rag_llm.retrieval.rerank import RerankRetriever

        return RerankRetriever(self.hybrid, self.scorer, candidates=self.settings.rerank_candidates)

    def uses_reranker(self, mode: str) -> bool:
        return mode == "hybrid+rerank" and self.settings.reranker_enabled

    def get(self, mode: str) -> Retriever:
        if mode == "hybrid+rerank" and not self.settings.reranker_enabled:
            # The reranker is the heaviest model here; RERANKER_ENABLED=false
            # turns it off everywhere at once, e.g. on a CPU-only container.
            return self.hybrid
        return {
            "bm25": lambda: self.bm25,
            "dense": lambda: self.dense,
            "hybrid": lambda: self.hybrid,
            "hybrid+rerank": lambda: self.reranked,
        }[mode]()
