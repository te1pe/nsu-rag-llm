"""Incremental indexing, against Qdrant's in-memory mode and a fake embedder.

No model, no GPU, no server: `QdrantClient(":memory:")` is the real client
running locally, and the fake embedder derives a vector from each text's hash.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from nsu_rag_llm.ingest.chunking import Chunk, DocInfo, chunk_document
from nsu_rag_llm.ingest.indexing import QdrantIndex, apply, plan, sync, write_chunks

# Payload indexes are a server feature; the in-memory client warns and ignores them.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


class FakeEmbedder:
    model_name = "fake"

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls = 0
        self.embedded = 0

    def embed_passages(self, texts):
        self.calls += 1
        self.embedded += len(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([b / 255 + 0.01 for b in digest[: self.dimension]])
        return vectors


def doc(course: str, name: str) -> DocInfo:
    return DocInfo(
        course=course,
        course_title=course.upper(),
        doc_title=name,
        file=f"{course}/sem1/{name}.pdf",
        sha256=hashlib.sha256(name.encode()).hexdigest(),
        semester=1,
        origin="digital",
    )


def chunks_of(info: DocInfo, pages: list[str]) -> list[Chunk]:
    return chunk_document(
        info,
        list(enumerate(pages, start=1)),
        lambda text: len(text.split()),
        "fake",
        budget=40,
        overlap=8,
        min_before_break=12,
    )


def pages(topic: str, n: int = 4) -> list[str]:
    return [
        f"# {topic} {i}\n\n" + " ".join(f"Утверждение {j} о теме {topic} {i}." for j in range(5))
        for i in range(n)
    ]


@pytest.fixture
def index() -> QdrantIndex:
    return QdrantIndex(QdrantClient(location=":memory:"), "test")


def test_the_first_run_embeds_everything(index: QdrantIndex) -> None:
    chunks = chunks_of(doc("os", "a"), pages("процесс"))
    embedder = FakeEmbedder()

    result = sync(index, lambda: embedder, chunks)

    assert result.added == len(chunks) > 0
    assert index.ids() == {c.id for c in chunks}
    assert embedder.embedded == len(chunks)


def test_a_second_run_embeds_nothing_and_does_not_load_the_model(index: QdrantIndex) -> None:
    chunks = chunks_of(doc("os", "a"), pages("процесс"))
    sync(index, FakeEmbedder, chunks)

    def must_not_load():
        raise AssertionError("the model was loaded although nothing was new")

    result = sync(index, must_not_load, chunks)

    assert (result.added, result.deleted, result.unchanged) == (0, 0, len(chunks))


def test_an_edit_re_embeds_only_what_it_touched(index: QdrantIndex) -> None:
    before = pages("процесс", n=6)
    sync(index, FakeEmbedder, chunks_of(doc("os", "a"), before))
    after = list(before)
    after[4] = after[4].replace("Утверждение 2", "Исправленное утверждение 2")
    embedder = FakeEmbedder()

    result = sync(index, lambda: embedder, chunks_of(doc("os", "a"), after))

    assert 0 < result.added <= 2
    assert result.deleted == result.added
    assert result.unchanged > result.added


def test_a_document_taken_out_of_the_corpus_is_deleted(index: QdrantIndex) -> None:
    kept = chunks_of(doc("os", "a"), pages("процесс"))
    dropped = chunks_of(doc("os", "b"), pages("поток"))
    sync(index, FakeEmbedder, kept + dropped)

    result = sync(index, FakeEmbedder, kept)

    assert result.deleted == len(dropped)
    assert index.ids() == {c.id for c in kept}


def test_a_run_over_one_course_leaves_the_others_alone(index: QdrantIndex) -> None:
    os_chunks = chunks_of(doc("os", "a"), pages("процесс"))
    matan_chunks = chunks_of(doc("matan", "b"), pages("предел"))
    sync(index, FakeEmbedder, os_chunks + matan_chunks)

    result = sync(index, FakeEmbedder, os_chunks, course="os")

    assert result.deleted == 0
    assert index.ids() == {c.id for c in os_chunks + matan_chunks}


def test_the_plan_is_what_apply_carries_out(index: QdrantIndex) -> None:
    chunks = chunks_of(doc("os", "a"), pages("процесс"))

    work = plan(index, chunks)
    result = apply(index, FakeEmbedder, work, batch_size=3)

    assert len(work.fresh) == result.added == len(chunks)
    assert len(result.batches) == -(-len(chunks) // 3)  # ceil: batches of three


def test_a_collection_built_by_another_model_is_refused(index: QdrantIndex) -> None:
    chunks = chunks_of(doc("os", "a"), pages("процесс"))
    sync(index, lambda: FakeEmbedder(dimension=8), chunks)

    with pytest.raises(RuntimeError, match="another"):
        index.ensure(dimension=16)


def test_payload_carries_what_citations_and_filters_need(index: QdrantIndex) -> None:
    chunks = chunks_of(doc("os", "a"), pages("процесс"))
    sync(index, FakeEmbedder, chunks)

    points, _ = index.client.scroll("test", limit=1, with_payload=True)
    payload = points[0].payload

    for key in ("course", "doc_title", "file", "page_start", "page_end", "section_path", "text"):
        assert key in payload
    assert payload["course"] == "os"


def test_writing_one_course_keeps_the_other_courses_lines(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    os_chunks = chunks_of(doc("os", "a"), pages("процесс"))
    matan_chunks = chunks_of(doc("matan", "b"), pages("предел"))
    write_chunks(os_chunks + matan_chunks, path)

    replacement = chunks_of(doc("os", "c"), pages("сигнал", n=2))
    write_chunks(replacement, path, course="os")

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {line["id"] for line in lines if line["course"] == "matan"} == {
        c.id for c in matan_chunks
    }
    assert {line["id"] for line in lines if line["course"] == "os"} == {c.id for c in replacement}


def test_ids_survive_the_round_trip_through_qdrant(index: QdrantIndex) -> None:
    """Qdrant normalises UUIDs; a mismatch in spelling would re-embed everything."""
    chunks = chunks_of(doc("os", "a"), pages("процесс"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sync(index, FakeEmbedder, chunks)

    assert plan(index, chunks).fresh == []


class FlakyWrites:
    """A Qdrant client whose first `failures` upserts time out in transport."""

    def __init__(self, inner: QdrantClient, failures: int) -> None:
        self.inner = inner
        self.failures = failures
        self.upserts = 0

    def upsert(self, *args, **kwargs):
        from qdrant_client.http.exceptions import ResponseHandlingException

        self.upserts += 1
        if self.upserts <= self.failures:
            raise ResponseHandlingException(TimeoutError("timed out"))
        return self.inner.upsert(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_write_that_times_out_is_retried() -> None:
    """Regression: indexing the full corpus died at 71% on one timed-out write.
    Writes are idempotent - ids come from content - so retrying is safe."""
    flaky = FlakyWrites(QdrantClient(location=":memory:"), failures=2)
    index = QdrantIndex(flaky, "test", backoff=0)
    chunks = chunks_of(doc("os", "a"), pages("процесс"))

    result = sync(index, FakeEmbedder, chunks)

    assert result.added == len(chunks)
    assert index.ids() == {c.id for c in chunks}
    assert flaky.upserts == 3  # two failures, then the batch went in


def test_a_dead_server_still_fails_the_write_after_the_retries() -> None:
    from qdrant_client.http.exceptions import ResponseHandlingException

    flaky = FlakyWrites(QdrantClient(location=":memory:"), failures=100)
    index = QdrantIndex(flaky, "test", backoff=0)

    with pytest.raises(ResponseHandlingException):
        sync(index, FakeEmbedder, chunks_of(doc("os", "a"), pages("процесс")))
    assert flaky.upserts == 4  # one try and three retries
