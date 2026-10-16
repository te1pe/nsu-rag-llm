"""The retrieval report: metrics and question ids, never question or corpus text.

CLAUDE.md allows only ids and metrics in evaluation reports, so this file can
be committed: nothing in it quotes a handbook, or even the questions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from nsu_rag_llm.evaluation.questions import QUESTION_TYPES, Validation
from nsu_rag_llm.evaluation.retrieval import DEPTH, ModeResult, QuestionResult


@dataclass
class RunInfo:
    date: str
    commit: str
    chunks: int
    embedding_model: str
    reranker_model: str
    candidates: int
    rrf_k: int
    rerank_candidates: int
    reranker_enabled: bool


def _pct(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.0%}"


def _num(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.3f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _breakdown(
    title: str,
    groups: list[tuple[str, Callable[[QuestionResult], bool]]],
    results: list[ModeResult],
) -> list[str]:
    header = ["группа", "n"] + [f"{r.mode}: r@10 / MRR" for r in results]
    out = [f"### {title}", "", _row(header), _row(["---"] + ["---:"] * (len(header) - 1))]
    for label, keep in groups:
        n = results[0].count(keep) if results else 0
        if not n:
            continue
        cells = [f"{_pct(r.recall(10, keep))} / {_num(r.mrr(keep))}" for r in results]
        out.append(_row([label, str(n), *cells]))
    return [*out, ""]


def render(
    unfiltered: list[ModeResult],
    filtered: list[ModeResult],
    validation: Validation,
    info: RunInfo,
) -> str:
    questions = validation.questions
    answerable = [q for q in questions if q.answerable]
    out = [
        f"# Оценка поиска — {info.date}",
        "",
        f"Коммит `{info.commit}`. Чанков в индексе: {info.chunks}. Эмбеддинги "
        f"`{info.embedding_model}`, реранкер `{info.reranker_model}`"
        + ("" if info.reranker_enabled else " (выключен)")
        + f". Кандидатов на ретривер: {info.candidates}, RRF k = {info.rrf_k}, "
        f"реранкер читает {info.rerank_candidates}.",
        "",
        f"Вопросов: **{len(questions)}**, из них с ответом **{len(answerable)}**, "
        f"без ответа {len(questions) - len(answerable)} (здесь не оцениваются — это фаза 4).",
        "",
        "Попадание: чанк из того же документа (по хешу содержимого), диапазон страниц "
        "которого пересекает страницы gold. recall@k — доля gold-источников вопроса, "
        "найденных в топ-k, в среднем по вопросам; MRR@10 — средняя обратная позиция "
        "первого попадания в топ-10.",
        "",
        "## Итог",
        "",
        "| режим | recall@5 | recall@10 | ±95% | MRR@10 | мс/запрос |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in unfiltered:
        out.append(
            _row(
                [
                    f"`{r.mode}`",
                    _pct(r.recall(5)),
                    _pct(r.recall(10)),
                    _pct(r.interval(10)),
                    _num(r.mrr()),
                    f"{r.median_ms():.0f}",
                ]
            )
        )
    out += [
        "",
        f"±95% — полуширина нормального доверительного интервала recall@10 при "
        f"n = {len(answerable)}: разница между режимами меньше неё — не результат, а шум.",
        "",
    ]

    if filtered:
        out += [
            "## С фильтром по курсу вопроса",
            "",
            "Сколько дало бы знание курса заранее — то, что в фазе 5 будет угадывать "
            "маршрутизатор.",
            "",
            "| режим | recall@5 | recall@10 | MRR@10 |",
            "|---|---:|---:|---:|",
        ]
        out += [
            _row([f"`{r.mode}`", _pct(r.recall(5)), _pct(r.recall(10)), _num(r.mrr())])
            for r in filtered
        ]
        out += [""]

    courses = sorted({q.course for q in answerable})
    out += ["## Разбивка", ""]
    out += _breakdown(
        "По курсам",
        [(f"`{c}`", lambda r, c=c: r.question.course == c) for c in courses],
        unfiltered,
    )
    out += _breakdown(
        "По типу исходника",
        [
            ("текст (digital)", lambda r: r.origin == "digital"),
            ("скан с OCR-слоем", lambda r: r.origin == "ocr-layer"),
            ("скан, наш OCR", lambda r: r.origin == "ocr"),
            ("неизвестно", lambda r: r.origin == "unknown"),
        ],
        unfiltered,
    )
    out += _breakdown(
        "По типу вопроса",
        [(t, lambda r, t=t: r.question.type == t) for t in QUESTION_TYPES],
        unfiltered,
    )

    out += [
        "## По вопросам",
        "",
        f"Позиция первого попадания в топ-{DEPTH}; «—» — не найдено.",
        "",
        _row(["id", "курс", "тип", "исходник", *[f"`{r.mode}`" for r in unfiltered]]),
        _row(["---"] * 4 + ["---:"] * len(unfiltered)),
    ]
    by_mode = [{res.question.id: res for res in r.results} for r in unfiltered]
    for question in answerable:
        first = by_mode[0][question.id] if by_mode else None
        ranks = [
            str(m[question.id].first_rank) if m[question.id].first_rank else "—" for m in by_mode
        ]
        origin = first.origin if first else "—"
        out.append(_row([question.id, question.course, question.type, origin, *ranks]))
    out += [""]

    if validation.warnings:
        out += ["## Предупреждения валидатора", ""]
        out += [f"- {w}" for w in validation.warnings]
        out += [""]
    return "\n".join(out)
