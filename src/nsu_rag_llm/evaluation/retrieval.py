"""Retrieval metrics, per mode: recall@5, recall@10 and MRR@10.

The hit rule is CLAUDE.md's: a retrieved chunk answers a gold entry when it
comes from the same file and its page range overlaps the gold pages. "Same
file" is decided by content hash, not by path, so that a gold entry naming a
duplicate copy is matched by chunks of the copy that was indexed.

* recall@k of a question is the share of its gold entries that at least one
  of the top k chunks answers - 0 or 1 for a single source, partial for a
  multi-hop question with several; the mode's recall@k is the mean.
* MRR@10 is the mean of 1 / (rank of the first chunk answering any gold
  entry), 0 when none of the top 10 does.

Unanswerable questions have no gold and are not scored here; phase 4 measures
whether the model declines them.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from nsu_rag_llm.evaluation.questions import Gold, Question
from nsu_rag_llm.retrieval.base import Hit, Retriever

CUTOFFS = (5, 10)
DEPTH = max(CUTOFFS)


def answers(chunk: dict, gold: Gold, gold_sha: dict[str, str]) -> bool:
    """Does this chunk come from the gold document and span a gold page?"""
    if chunk.get("doc_sha256") != gold_sha.get(gold.file):
        return False
    return any(chunk["page_start"] <= page <= chunk["page_end"] for page in gold.pages)


@dataclass
class QuestionResult:
    question: Question
    origin: str
    first_rank: int | None
    """Rank of the first chunk answering any gold entry, within the top DEPTH."""
    coverage: dict[int, float]
    """k -> share of gold entries answered by the top k."""
    seconds: float


@dataclass
class ModeResult:
    mode: str
    filtered: bool
    """Searched with the question's course as a filter."""
    results: list[QuestionResult] = field(default_factory=list)

    def recall(self, k: int, keep: Callable[[QuestionResult], bool] | None = None) -> float:
        values = [r.coverage[k] for r in self._subset(keep)]
        return statistics.fmean(values) if values else float("nan")

    def mrr(self, keep: Callable[[QuestionResult], bool] | None = None) -> float:
        values = [1 / r.first_rank if r.first_rank else 0.0 for r in self._subset(keep)]
        return statistics.fmean(values) if values else float("nan")

    def interval(self, k: int) -> float:
        """Half-width of a 95% normal-approximation interval for recall@k."""
        values = [r.coverage[k] for r in self.results]
        if len(values) < 2:
            return float("nan")
        return 1.96 * statistics.stdev(values) / math.sqrt(len(values))

    def median_ms(self) -> float:
        return statistics.median(r.seconds for r in self.results) * 1000 if self.results else 0.0

    def count(self, keep: Callable[[QuestionResult], bool] | None = None) -> int:
        return len(self._subset(keep))

    def _subset(self, keep):
        return [r for r in self.results if keep is None or keep(r)]


def question_origin(question: Question, gold_sha: dict[str, str], origins: dict[str, str]) -> str:
    """Where the answer's text came from: born-digital, someone else's OCR
    layer, or this project's own OCR (phase 4.5)."""
    kinds = {origins.get(gold_sha.get(g.file, ""), "unknown") for g in question.gold}
    if "ocr" in kinds:
        return "ocr"
    if "ocr-layer" in kinds:
        return "ocr-layer"
    if kinds == {"digital"}:
        return "digital"
    return "unknown"


def score_question(
    question: Question,
    hits: list[Hit],
    gold_sha: dict[str, str],
    origin: str,
    seconds: float,
) -> QuestionResult:
    first_rank = next(
        (
            rank
            for rank, hit in enumerate(hits[:DEPTH], start=1)
            if any(answers(hit.chunk, gold, gold_sha) for gold in question.gold)
        ),
        None,
    )
    coverage = {
        k: sum(any(answers(h.chunk, gold, gold_sha) for h in hits[:k]) for gold in question.gold)
        / len(question.gold)
        for k in CUTOFFS
    }
    return QuestionResult(question, origin, first_rank, coverage, seconds)


def evaluate(
    mode: str,
    retriever: Retriever,
    questions: Iterable[Question],
    gold_sha: dict[str, str],
    origins: dict[str, str],
    filtered: bool = False,
) -> ModeResult:
    result = ModeResult(mode, filtered)
    for question in questions:
        if not question.answerable:
            continue
        started = time.perf_counter()
        hits = retriever.search(question.question, DEPTH, question.course if filtered else None)
        seconds = time.perf_counter() - started
        origin = question_origin(question, gold_sha, origins)
        result.results.append(score_question(question, hits, gold_sha, origin, seconds))
    return result
