"""Vector search: the question embedded as a query, nearest chunks from Qdrant."""

from __future__ import annotations

import time
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from nsu_rag_llm.retrieval.base import Hit
from nsu_rag_llm.retrieval.filters import course_filter


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class E5QueryEmbedder:
    """multilingual-e5 on the query side.

    e5 was trained asymmetrically: passages carry "passage: " (ingest adds it)
    and questions carry "query: ". Swapping or dropping the prefixes quietly
    degrades retrieval.
    """

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()


#: Transport failures - a timeout, a dropped connection - are retried this many
#: times. The first full evaluation run died two hours in on one timed-out
#: request: the laptop had slept with the request in flight.
RETRIES = 2


class DenseRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: QueryEmbedder,
        backoff: float = 1.0,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.backoff = backoff

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        for attempt in range(RETRIES + 1):
            try:
                points = self.client.query_points(
                    self.collection,
                    query=vector,
                    limit=k,
                    query_filter=course_filter(course),
                    with_payload=True,
                ).points
                break
            except ResponseHandlingException:
                # Only transport errors; a malformed request would fail the
                # same way again and is left to surface at once.
                if attempt == RETRIES:
                    raise
                time.sleep(self.backoff * (attempt + 1))
        return [Hit(point.payload, point.score) for point in points]
