"""Retrieval components, on synthetic chunks, with fake models.

BM25 and the fusion are pure computation; dense search runs against Qdrant's
in-memory mode with a fake query embedder; the reranker gets a fake scorer.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import ClassVar

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from nsu_rag_llm.config import Settings
from nsu_rag_llm.retrieval.base import Hit
from nsu_rag_llm.retrieval.bm25 import BM25Index
from nsu_rag_llm.retrieval.dense import DenseRetriever
from nsu_rag_llm.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion
from nsu_rag_llm.retrieval.modes import Retrievers
from nsu_rag_llm.retrieval.rerank import RerankRetriever
from nsu_rag_llm.retrieval.text import stem, tokenize

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def chunk(cid: str, text: str, course: str = "os", context: str = "Курс › Книга") -> dict:
    return {
        "id": cid,
        "course": course,
        "context": context,
        "text": text,
        "file": f"{course}/sem1/book.pdf",
        "doc_sha256": "a" * 64,
        "page_start": 1,
        "page_end": 1,
    }


CHUNKS = [
    chunk("p1", "Процесс — это программа в стадии выполнения со своим адресным пространством."),
    chunk("p2", "Потоки одного процесса разделяют память и дескрипторы файлов."),
    chunk("p3", "Системный вызов fork создаёт копию процесса."),
    chunk("m1", "Предел последовательности определяется через эпсилон.", course="matan"),
    chunk("m2", "Интегрирование по частям сводит интеграл к более простому.", course="matan"),
]


def test_words_are_stemmed_in_their_own_language() -> None:
    assert stem("процессы") == stem("процесса") == "процесс"
    assert stem("threads") == "thread"
    # An identifier with digits or mixed scripts is kept as it is.
    assert stem("x86") == "x86"


def test_tokens_fold_yo_and_drop_single_letters() -> None:
    assert tokenize("Ёмкость и x") == ["емкост"]


def test_bm25_ranks_the_chunk_sharing_the_rare_word_first() -> None:
    index = BM25Index(CHUNKS)

    hits = index.search("что делает fork", 3)

    assert hits[0].id == "p3"


def test_bm25_matches_inflected_forms() -> None:
    index = BM25Index(CHUNKS)

    assert index.search("потока", 1)[0].id == "p2"


def test_bm25_course_filter_keeps_only_that_course() -> None:
    index = BM25Index(CHUNKS)

    hits = index.search("процесс предел интеграл", 10, course="matan")

    assert {h.chunk["course"] for h in hits} == {"matan"}


def test_bm25_indexes_the_context_line_too() -> None:
    """A section title is often the best match a question has."""
    chunks = [
        chunk("a", "Общий текст без ключевого слова.", context="ОС › Книга › Семафоры"),
        chunk("b", "Другой общий текст.", context="ОС › Книга › Прерывания"),
    ]

    assert BM25Index(chunks).search("семафор", 1)[0].id == "a"


def test_bm25_returns_nothing_rather_than_noise() -> None:
    index = BM25Index(CHUNKS)

    assert index.search("квантовая хромодинамика", 5) == []


def test_bm25_idf_never_goes_negative() -> None:
    """Lucene's idf: a term in every chunk adds a little, never subtracts."""
    chunks = [chunk(str(i), "общий термин и что-то ещё") for i in range(10)]

    assert all(value > 0 for value in BM25Index(chunks).idf.values())


def _hits(*ids: str) -> list[Hit]:
    return [Hit(chunk(i, i), 1.0) for i in ids]


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = reciprocal_rank_fusion([_hits("a", "b", "c"), _hits("b", "c", "a")], k=60)

    assert [h.id for h in fused][:1] == ["b"]  # 1/62 + 1/61 beats 1/61 + 1/63
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_keeps_what_only_one_ranking_found() -> None:
    fused = reciprocal_rank_fusion([_hits("a"), _hits("z")], k=60)

    assert {h.id for h in fused} == {"a", "z"}


def test_rrf_breaks_ties_the_same_way_every_time() -> None:
    first = reciprocal_rank_fusion([_hits("b"), _hits("a")])
    second = reciprocal_rank_fusion([_hits("b"), _hits("a")])

    assert [h.id for h in first] == [h.id for h in second] == ["a", "b"]


class Fixed:
    """A retriever that returns a fixed ranking and records what it was asked."""

    def __init__(self, *ids: str) -> None:
        self.ids = ids
        self.calls: list[tuple] = []

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        self.calls.append((query, k, course))
        return _hits(*self.ids)[:k]


def test_hybrid_asks_both_retrievers_for_candidates_and_fuses() -> None:
    lexical, dense = Fixed("a", "b"), Fixed("b", "c")
    hybrid = HybridRetriever(lexical, dense, candidates=7)

    hits = hybrid.search("вопрос", 2, course="os")

    assert [h.id for h in hits] == ["b", "a"]
    assert lexical.calls == [("вопрос", 7, "os")]
    assert dense.calls == [("вопрос", 7, "os")]


class Keyword:
    """A fake cross-encoder: scores a passage by whether it holds a word."""

    def __init__(self, word: str) -> None:
        self.word = word

    def score(self, query, passages):
        return [1.0 if self.word in passage else 0.0 for passage in passages]


class Texts:
    """A retriever returning chunks with given texts, in the given order."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        return [Hit(chunk(cid, text), 1.0) for cid, text in self.texts.items()][:k]


def test_rerank_reorders_the_shortlist() -> None:
    base = Texts({"x": "без него", "y": "без него", "target": "в этом тексте есть ключ"})

    hits = RerankRetriever(base, Keyword("ключ"), candidates=3).search("вопрос", 2)

    assert hits[0].id == "target"
    assert len(hits) == 2


def test_rerank_reads_the_context_line_with_the_text() -> None:
    seen: list[str] = []

    class Recorder:
        def score(self, query, passages):
            seen.extend(passages)
            return [0.0] * len(passages)

    RerankRetriever(Fixed("a"), Recorder(), candidates=1).search("вопрос", 1)

    assert seen == ["Курс › Книга\n\na"]


class Vectors:
    """Fake query embedder: a query "about" a topic points at that topic."""

    topics: ClassVar[dict[str, list[float]]] = {
        "процесс": [1.0, 0.0, 0.0],
        "предел": [0.0, 1.0, 0.0],
        "fork": [0.0, 0.0, 1.0],
    }

    def embed_query(self, text: str) -> list[float]:
        return next((v for word, v in self.topics.items() if word in text), [0.3, 0.3, 0.3])


@pytest.fixture
def dense() -> DenseRetriever:
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "c", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    points = [
        ("p1", [1.0, 0.1, 0.0], "os"),
        ("p3", [0.0, 0.1, 1.0], "os"),
        ("m1", [0.1, 1.0, 0.0], "matan"),
    ]
    client.upsert(
        "c",
        points=[
            models.PointStruct(
                id=str(uuid.UUID(bytes=hashlib.sha256(cid.encode()).digest()[:16])),
                vector=vector,
                payload=chunk(cid, cid, course),
            )
            for cid, vector, course in points
        ],
        wait=True,
    )
    return DenseRetriever(client, "c", Vectors())


def test_dense_returns_the_nearest_chunk_with_its_payload(dense: DenseRetriever) -> None:
    hits = dense.search("что такое процесс", 1)

    assert hits[0].id == "p1"
    assert hits[0].chunk["course"] == "os"


def test_dense_course_filter_is_applied_in_qdrant(dense: DenseRetriever) -> None:
    hits = dense.search("что такое процесс", 5, course="matan")

    assert [h.id for h in hits] == ["m1"]


def test_disabling_the_reranker_turns_hybrid_rerank_into_hybrid() -> None:
    retrievers = Retrievers(Settings(_env_file=None, reranker_enabled=False))
    stand_in = Fixed("a")
    retrievers.__dict__["hybrid"] = stand_in  # what cached_property would hold

    assert retrievers.get("hybrid+rerank") is stand_in


class Flaky:
    """A Qdrant client whose first `failures` calls time out."""

    def __init__(self, inner: QdrantClient, failures: int) -> None:
        self.inner = inner
        self.failures = failures
        self.calls = 0

    def query_points(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise ResponseHandlingException(TimeoutError("timed out"))
        return self.inner.query_points(*args, **kwargs)


def test_a_transient_timeout_is_retried(dense: DenseRetriever) -> None:
    """Regression: one timed-out request - the laptop slept mid-run - killed a
    two-hour evaluation."""
    flaky = Flaky(dense.client, failures=2)
    retriever = DenseRetriever(flaky, "c", Vectors(), backoff=0)

    hits = retriever.search("что такое процесс", 1)

    assert hits[0].id == "p1"
    assert flaky.calls == 3


def test_a_dead_server_still_fails_after_the_retries(dense: DenseRetriever) -> None:
    flaky = Flaky(dense.client, failures=10)
    retriever = DenseRetriever(flaky, "c", Vectors(), backoff=0)

    with pytest.raises(ResponseHandlingException):
        retriever.search("что такое процесс", 1)
    assert flaky.calls == 3
