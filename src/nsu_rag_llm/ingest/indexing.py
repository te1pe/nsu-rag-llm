"""Embed chunks and keep the Qdrant collection in step with them.

Indexing is incremental by construction: a chunk's id is derived from the
embedding model, its document's bytes, its first page and the exact text that
gets embedded (see `chunking.chunk_id`). Syncing is then set arithmetic:

    to embed   = ids the chunks have and the collection does not
    to delete  = ids the collection has and the chunks do not
    unchanged  = everything else - not re-embedded, not even read

Editing one page re-embeds the few chunks that page produced; changing the
embedding model changes every id and so rebuilds the collection; excluding a
document in courses.yaml deletes its points on the next run (ADR-021).

The embedder sits behind a two-method protocol, so tests run against a fake
one and Qdrant's in-memory mode: no model, no GPU, no server.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from nsu_rag_llm import jsonl
from nsu_rag_llm.ingest.chunking import Chunk

#: Fields a search can filter on (phase 2: by course; eval: by source type).
KEYWORD_FIELDS = ("course", "origin", "file")


class Embedder(Protocol):
    dimension: int
    model_name: str

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """multilingual-e5 through sentence-transformers, on MPS/CUDA/CPU."""

    def __init__(self, model_name: str, device: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        # e5 was trained with these prefixes; leaving "passage: " off would
        # embed a different kind of text from the one it learned on.
        vectors = self.model.encode(
            [f"passage: {text}" for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


#: Writes that fail in transport - a timeout, a dropped connection - are
#: retried this many times. Safe, because writes are idempotent: a point's id
#: comes from its content (ADR-021), so a batch written twice is written once.
#: Indexing the full corpus died at 71% on one timed-out write without this.
RETRIES = 3


class QdrantIndex:
    """The one collection this project writes to."""

    def __init__(self, client: QdrantClient, collection: str, backoff: float = 2.0) -> None:
        self.client = client
        self.collection = collection
        self.backoff = backoff

    def _write(self, operation: Callable[[], object]) -> None:
        for attempt in range(RETRIES + 1):
            try:
                operation()
                return
            except ResponseHandlingException:
                if attempt == RETRIES:
                    raise
                time.sleep(self.backoff * (attempt + 1))

    def ensure(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        info = self.client.get_collection(self.collection)
        existing = info.config.params.vectors.size
        if existing != dimension:
            raise RuntimeError(
                f"collection '{self.collection}' holds {existing}-dimensional vectors, "
                f"the embedding model produces {dimension}. It was built with another "
                "model; delete the collection and ingest again."
            )
        indexed = set((info.payload_schema or {}).keys())
        for name in KEYWORD_FIELDS:
            if name not in indexed:
                self.client.create_payload_index(
                    self.collection,
                    field_name=name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    def ids(self, course: str | None = None) -> set[str]:
        """Point ids in the collection - of one course, if given.

        Only ids: no payloads, no vectors. An absent collection has none.
        """
        if not self.client.collection_exists(self.collection):
            return set()
        scope = (
            models.Filter(
                must=[models.FieldCondition(key="course", match=models.MatchValue(value=course))]
            )
            if course
            else None
        )
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=scope,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            found.update(str(point.id) for point in points)
            if offset is None:
                return found

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        points = [
            models.PointStruct(id=chunk.id, vector=list(vector), payload=chunk.to_json())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._write(lambda: self.client.upsert(self.collection, points=points, wait=True))

    def delete(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        for start in range(0, len(ids), 1000):
            selector = models.PointIdsList(points=ids[start : start + 1000])
            self._write(
                lambda selector=selector: self.client.delete(
                    self.collection, points_selector=selector, wait=True
                )
            )


@dataclass
class SyncResult:
    added: int = 0
    deleted: int = 0
    unchanged: int = 0
    embed_seconds: float = 0.0
    batches: list[tuple[int, float]] = field(default_factory=list)
    """(chunks, seconds) per batch, for the speed log."""

    @property
    def rate(self) -> float:
        return self.added / self.embed_seconds if self.embed_seconds else 0.0


@dataclass
class SyncPlan:
    fresh: list[Chunk]
    """Chunks the collection does not have: to embed."""
    stale: set[str]
    """Ids the collection has and the chunks do not: to delete."""
    unchanged: int


def plan(index: QdrantIndex, chunks: Sequence[Chunk], course: str | None = None) -> SyncPlan:
    """What it takes to make the collection hold exactly `chunks`.

    With `course`, only that course's points are compared: a run over one
    course must not read every other course's points as stale.
    """
    wanted = {chunk.id: chunk for chunk in chunks}
    present = index.ids(course)
    fresh = [chunk for chunk_id, chunk in wanted.items() if chunk_id not in present]
    return SyncPlan(
        fresh=fresh,
        stale=present - wanted.keys(),
        unchanged=len(wanted) - len(fresh),
    )


def apply(
    index: QdrantIndex,
    embedder: Callable[[], Embedder],
    work: SyncPlan,
    batch_size: int = 256,
    on_batch: Callable[[int, float], None] | None = None,
) -> SyncResult:
    """Embed and upsert what is new, delete what is gone.

    `embedder` is a factory: loading the model costs seconds and memory, and a
    run that finds nothing new should pay neither.
    """
    result = SyncResult(unchanged=work.unchanged)
    if work.fresh:
        model = embedder()
        index.ensure(model.dimension)
        for start in range(0, len(work.fresh), batch_size):
            batch = work.fresh[start : start + batch_size]
            began = time.perf_counter()
            vectors = model.embed_passages([chunk.embedding_text for chunk in batch])
            index.upsert(batch, vectors)
            seconds = time.perf_counter() - began
            result.added += len(batch)
            result.embed_seconds += seconds
            result.batches.append((len(batch), seconds))
            if on_batch:
                on_batch(len(batch), seconds)
    if work.stale:
        index.delete(work.stale)
        result.deleted = len(work.stale)
    return result


def sync(
    index: QdrantIndex,
    embedder: Callable[[], Embedder],
    chunks: Sequence[Chunk],
    batch_size: int = 256,
    course: str | None = None,
) -> SyncResult:
    """Make the collection hold exactly `chunks`, embedding only what is new."""
    return apply(index, embedder, plan(index, chunks, course), batch_size)


def write_chunks(chunks: Sequence[Chunk], path: Path, course: str | None = None) -> None:
    """All chunks, one JSON object per line: BM25 (phase 2) reads this file.

    With `course`, only that course's lines are replaced and every other
    course's lines are kept - otherwise a run over one course would leave the
    file holding nothing else.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    if course and path.exists():
        kept = [
            line
            for line in jsonl.lines(path.read_text(encoding="utf-8"))
            if json.loads(line)["course"] != course
        ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line + "\n")
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)
