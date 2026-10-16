"""The generation report: metrics per model and question ids, never text.

Like the retrieval report it can be committed: no answer, no question and no
fragment appears in it, only ids, verdicts and numbers. The answers
themselves stay in `data/eval-runs/`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from nsu_rag_llm.evaluation.generation import Answered, VariantResult
from nsu_rag_llm.evaluation.questions import QUESTION_TYPES, Validation


@dataclass
class GenerationInfo:
    date: str
    commit: str
    mode: str
    chunks: int
    answer_fragments: int
    judge: str | None


def _pct(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.0%}"


def _seconds(ms: float) -> str:
    return "—" if math.isnan(ms) else f"{ms / 1000:.1f} с"


def _speed(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.1f}"


def _gb(size: int | None) -> str:
    return "—" if size is None else f"{size / 1e9:.1f} ГБ"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _mark(value: bool | None) -> str:
    return {True: "✓", False: "✗", None: "·"}[value]


def _summary(results: list[VariantResult]) -> list[str]:
    header = [
        "модель",
        "верность",
        "полнота",
        "ссылки валидны",
        "ссылки по делу",
        "отказ на unanswerable",
        "ложные отказы",
        "TTFT (медиана)",
        "ток/с",
        "пик памяти",
    ]
    out = [_row(header), _row(["---"] + ["---:"] * (len(header) - 1))]
    for r in results:
        out.append(
            _row(
                [
                    f"`{r.variant.label}`",
                    _pct(r.faithfulness()),
                    _pct(r.completeness()),
                    _pct(r.citations_valid()),
                    _pct(r.citations_supported()),
                    _pct(r.refusal_accuracy()),
                    _pct(r.false_refusals()),
                    _seconds(r.ttft_ms()),
                    _speed(r.tokens_per_second()),
                    _gb(r.peak_memory_bytes),
                ]
            )
        )
    return out


def _breakdown(
    title: str, groups: list[tuple[str, Callable[[Answered], bool]]], results: list[VariantResult]
) -> list[str]:
    header = ["группа", "n"] + [f"`{r.variant.label}`: верность / полнота" for r in results]
    out = [f"### {title}", "", _row(header), _row(["---"] + ["---:"] * (len(header) - 1))]
    for label, keep in groups:
        n = results[0].count(keep) if results else 0
        if n:
            cells = [
                f"{_pct(r.faithfulness(keep))} / {_pct(r.completeness(keep))}" for r in results
            ]
            out.append(_row([label, str(n), *cells]))
    return [*out, ""]


def _health(results: list[VariantResult]) -> list[str]:
    header = [
        "модель",
        "прогон",
        "обрезано по max_tokens",
        "промпт вне бюджета",
        "оценка − факт, ток.",
        "сбоев судьи",
        "противоречивых вердиктов",
    ]
    out = [_row(header), _row(["---", "---"] + ["---:"] * (len(header) - 2))]
    for r in results:
        error = r.estimate_error()
        out.append(
            _row(
                [
                    f"`{r.variant.label}`",
                    f"`{r.run_id}`",
                    str(r.truncated()),
                    str(r.over_budget()),
                    "—" if error is None else f"{error[0]}…{error[1]}",
                    str(r.judge_failures()),
                    str(r.inconsistent()),
                ]
            )
        )
    return out


def _per_question(results: list[VariantResult]) -> list[str]:
    header = ["id", "тип"] + [f"`{r.variant.label}`" for r in results]
    out = [_row(header), _row(["---", "---"] + [":---:"] * len(results))]
    for index, first in enumerate(results[0].answered if results else []):
        cells = []
        for r in results:
            answer = r.answered[index]
            judged = r.judged.get(answer.id)
            if answer.refused:
                cells.append("отказ")
                continue
            marks = [
                _mark(judged.faithful if judged else None),
                _mark(not answer.citation_errors),
                _mark(judged.citations_correct if judged else None),
                _mark(judged.complete if judged else None) if answer.answerable else "·",
            ]
            cells.append(" ".join(marks))
        out.append(_row([first.id, first.type, *cells]))
    return out


def render(results: list[VariantResult], validation: Validation, info: GenerationInfo) -> str:
    answerable = sum(q.answerable for q in validation.questions)
    lines = [
        f"# Оценка ответов — {info.date}",
        "",
        f"Код `{info.commit}` · поиск `{info.mode}`, фрагментов в промпте до "
        f"{info.answer_fragments} · в индексе {info.chunks} чанков · судья "
        + (f"`{info.judge}`" if info.judge else "не вызывался"),
        "",
        f"Вопросов {len(validation.questions)}: с ответом {answerable}, "
        f"без ответа (unanswerable) {len(validation.questions) - answerable}. "
        "Все модели отвечали по одним и тем же фрагментам.",
        "",
        "## Сводка",
        "",
        *_summary(results),
        "",
        "Верность и «ссылки по делу» — по ответам без отказа; полнота — по вопросам "
        "с ответом (отказ = неполный); «ссылки валидны» проверяет код, остальное — "
        "судья. TTFT — от запроса к модели до первого токена.",
        "",
        "## Разбивка",
        "",
    ]
    courses = sorted({a.course for r in results for a in r.answered})
    lines += _breakdown("По курсам", [(c, lambda a, c=c: a.course == c) for c in courses], results)
    lines += _breakdown(
        "По типам вопросов",
        [(t, lambda a, t=t: a.type == t) for t in QUESTION_TYPES if t != "unanswerable"],
        results,
    )
    agents = [r for r in results if r.routed()]
    if agents:
        lines += [
            "## Агент",
            "",
            "| система | маршрутизатор угадал курс | поисков на вопрос |",
            "| --- | ---: | ---: |",
        ]
        lines += [
            f"| `{r.variant.label}` | {_pct(r.router_accuracy())} | {r.mean_searches():.1f} |"
            for r in agents
        ]
        lines += [
            "",
            "Маршрутизатор — по вопросам с ответом: выбранный курс совпал с курсом вопроса.",
            "",
        ]
    lines += ["## Надёжность прогона", "", *_health(results), ""]
    lines += [
        "## По вопросам",
        "",
        "Верность · ссылки валидны · ссылки по делу · полнота; ✓ да, ✗ нет, · не оценивалось.",
        "",
        *_per_question(results),
        "",
    ]
    if validation.warnings:
        lines += ["## Предупреждения валидатора", ""]
        lines += [f"- {w}" for w in validation.warnings]
        lines.append("")
    return "\n".join(lines)
