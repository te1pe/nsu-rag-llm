"""Ingest report: how the corpus was cut, and what indexing did.

Numbers and file names only, like the other stage reports; it lives in
`data/`, outside git.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from nsu_rag_llm.ingest.build import DocumentChunks
from nsu_rag_llm.ingest.chunking import MIN_BEFORE_BREAK, Chunk
from nsu_rag_llm.ingest.indexing import SyncResult
from nsu_rag_llm.ingest.inventory_report import thousands

#: Chunks this small carry too little text to retrieve well; they are counted.
SMALL_CHUNK = 50


@dataclass
class IngestRun:
    documents: list[DocumentChunks]
    unparsed: list
    model: str
    device: str
    budget: int
    overlap: int
    chunk_seconds: float
    sync: SyncResult | None
    """None for a dry run, which chunks but does not touch the index."""
    embedding_tokens: list[int]
    """Tokens of what is actually embedded - prefix, context line and text."""
    model_limit: int


def quantile(values: list[int], share: float) -> int:
    ordered = sorted(values)
    return ordered[int(share * (len(ordered) - 1))] if ordered else 0


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def length_table(chunks_by_course: dict[str, list[Chunk]], pages: dict[str, int]) -> list[str]:
    out = [
        "| курс | документов | страниц | чанков | токенов: min | p10 | медиана | p90 | max |"
        f" < {SMALL_CHUNK} | с кодом | с таблицей | разрезано |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for course, chunks in sorted(chunks_by_course.items()):
        tokens = [c.tokens for c in chunks]
        out.append(
            _row(
                [
                    f"`{course}`",
                    str(len({c.file for c in chunks})),
                    thousands(pages[course]),
                    thousands(len(chunks)),
                    str(min(tokens)),
                    str(quantile(tokens, 0.1)),
                    f"{statistics.median(tokens):.0f}",
                    str(quantile(tokens, 0.9)),
                    str(max(tokens)),
                    str(sum(t < SMALL_CHUNK for t in tokens)),
                    str(sum(c.has_code for c in chunks)),
                    str(sum(c.has_table for c in chunks)),
                    str(sum(c.split_structure for c in chunks)),
                ]
            )
        )
    return out


def render(run: IngestRun) -> str:
    chunks = [c for d in run.documents for c in d.chunks]
    by_course: dict[str, list[Chunk]] = defaultdict(list)
    pages: dict[str, int] = defaultdict(int)
    for d in run.documents:
        by_course[d.document.course].extend(d.chunks)
        pages[d.document.course] += d.pages
    over = sum(t > run.model_limit for t in run.embedding_tokens)

    out = [
        f"# Чанкинг и индексация — {date.today().isoformat()}",
        "",
        "Отчёт не попадает в git: он перечисляет имена файлов из `data/`. "
        "Содержимого методичек здесь нет — только измерения.",
        "",
        "## Итог",
        "",
        f"- документов: **{len(run.documents)}**, страниц: "
        f"**{thousands(sum(d.pages for d in run.documents))}**, "
        f"чанков: **{thousands(len(chunks))}**",
        f"- бюджет чанка {run.budget} токенов, перекрытие ~{run.overlap}; заголовок начинает "
        f"новый чанк, если в текущем уже не меньше {MIN_BEFORE_BREAK} токенов",
        f"- эмбеддинги: `{run.model}` на `{run.device}`; чанкинг занял {run.chunk_seconds:.0f} с",
        f"- то, что уходит в модель (префикс, строка контекста, текст): "
        f"максимум {max(run.embedding_tokens, default=0)} токенов, p99 "
        f"{quantile(run.embedding_tokens, 0.99)}; **длиннее {run.model_limit}: {over}**",
    ]
    if run.sync is None:
        out += ["- **сухой прогон**: индекс не трогали"]
    else:
        s = run.sync
        out += [
            f"- индекс: добавлено **{thousands(s.added)}**, удалено {thousands(s.deleted)}, "
            f"без изменений {thousands(s.unchanged)}",
        ]
        if s.added:
            rates = [n / sec for n, sec in s.batches if sec]
            out += [
                f"- скорость эмбеддингов: **{s.rate:.1f} чанков/с** в среднем "
                f"(по батчам: min {min(rates):.1f}, медиана {statistics.median(rates):.1f}, "
                f"max {max(rates):.1f}); всего {s.embed_seconds / 60:.1f} мин",
            ]
    out.append("")

    if run.unparsed:
        out += ["## Не разобраны", "", "Запустите `nsu-rag parse`:", ""]
        out += [f"- `{path}`" for path in run.unparsed]
        out += [""]

    out += ["## Распределение длин по курсам", "", "Длины — в токенах токенизатора модели.", ""]
    out += length_table(by_course, pages)
    out += [
        "",
        "«Разрезано» — чанки с куском листинга или таблицы, которые целиком в бюджет не "
        "влезли и разрезаны по структуре: листинг — по пустым строкам, иначе между строками; "
        "таблица — между строками, с повтором шапки (ADR-020).",
        "",
        "## По документам",
        "",
        "| документ | страниц | чанков | токенов, медиана | разрезано |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in sorted(run.documents, key=lambda x: str(x.document.source)):
        tokens = [c.tokens for c in d.chunks] or [0]
        out.append(
            _row(
                [
                    f"`{d.document.source}`",
                    str(d.pages),
                    str(len(d.chunks)),
                    f"{statistics.median(tokens):.0f}",
                    str(sum(c.split_structure for c in d.chunks)),
                ]
            )
        )
    out.append("")
    return "\n".join(out)
