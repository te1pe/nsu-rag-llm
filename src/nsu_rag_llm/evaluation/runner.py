"""Run the retrieval evaluation end to end: validate, search, score, report."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nsu_rag_llm.config import PROJECT_ROOT, Settings
from nsu_rag_llm.evaluation.questions import Validation, validate
from nsu_rag_llm.evaluation.report import RunInfo, render
from nsu_rag_llm.evaluation.retrieval import ModeResult, evaluate
from nsu_rag_llm.retrieval.base import Retriever


@dataclass
class RetrievalRun:
    validation: Validation
    unfiltered: list[ModeResult] = field(default_factory=list)
    filtered: list[ModeResult] = field(default_factory=list)
    report: Path | None = None


def index_facts(chunks_file: Path) -> tuple[set[str], dict[str, str], int]:
    """sha256 of every indexed document, its text origin, and the chunk count."""
    indexed: set[str] = set()
    origins: dict[str, str] = {}
    count = 0
    with chunks_file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            indexed.add(chunk["doc_sha256"])
            origins[chunk["doc_sha256"]] = chunk["origin"]
            count += 1
    return indexed, origins, count


def report_path(settings: Settings, day: str, kind: str) -> Path:
    """`<date>-<kind>.md`; the full corpus's reports get `-full`, so that
    the same questions on both indexes (phase 4.5) leave two reports."""
    suffix = "-full" if settings.corpus_scope == "full" else ""
    return settings.eval_reports_dir / f"{day}-{kind}{suffix}.md"


def commit() -> str:
    """The code a report was produced by: short hash, "+dirty" if edited."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return f"{head}+dirty" if dirty else head


def run_retrieval(
    settings: Settings,
    modes: Iterable[str],
    retriever_for: Callable[[str], Retriever],
    questions_file: Path | None = None,
    with_filter: bool = True,
    on_mode: Callable[[str, bool], None] | None = None,
    write_report: bool = True,
) -> RetrievalRun:
    """Validate the questions, then score every mode on them.

    `retriever_for` builds a mode's retriever; each one is warmed up with a
    throwaway query first, so the latency in the report is search time, not
    model loading.
    """
    indexed, origins, chunk_count = index_facts(settings.chunks_file)
    validation = validate(questions_file or settings.questions_file, settings.raw_dir, indexed)
    run = RetrievalRun(validation)
    if not validation.ok:
        return run

    for mode in modes:
        retriever = retriever_for(mode)
        retriever.search("прогрев", 1)
        for filtered in (False, True) if with_filter else (False,):
            if on_mode:
                on_mode(mode, filtered)
            result = evaluate(
                mode, retriever, validation.questions, validation.gold_sha, origins, filtered
            )
            (run.filtered if filtered else run.unfiltered).append(result)

    if write_report:
        info = RunInfo(
            date=date.today().isoformat(),
            commit=commit(),
            chunks=chunk_count,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
            candidates=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
            rerank_candidates=settings.rerank_candidates,
            reranker_enabled=settings.reranker_enabled,
        )
        settings.eval_reports_dir.mkdir(parents=True, exist_ok=True)
        run.report = report_path(settings, info.date, "retrieval")
        run.report.write_text(
            render(run.unfiltered, run.filtered, validation, info), encoding="utf-8"
        )
    return run
