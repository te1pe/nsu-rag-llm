"""Run the generation evaluation: retrieve once, answer with every model, judge.

1. The questions are validated, exactly as for retrieval.
2. Every question is searched **once**, and the fragments are kept: every
   model then answers from the same ones, so only the model differs. The
   search models are released before any answering model is loaded - on a
   16 GB laptop they would otherwise page each other out (ADR-026).
3. Each model answers every question with the others unloaded; its peak
   memory is sampled from Ollama after every answer.
4. Only then is anything judged, so that the judge's model never shares
   memory with a model whose speed is being measured.

Answers and verdicts go to `data/eval-runs/<date>-<mode>-<label>/`: they quote
the handbooks and stay out of git. The report holds ids and numbers only.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nsu_rag_llm.config import Settings
from nsu_rag_llm.evaluation.generation import (
    Answered,
    Judged,
    Variant,
    VariantResult,
    by_id,
    write_jsonl,
)
from nsu_rag_llm.evaluation.judge import Judge, JudgeError
from nsu_rag_llm.evaluation.ollama import Memory
from nsu_rag_llm.evaluation.questions import Question, Validation, validate
from nsu_rag_llm.evaluation.runner import commit, index_facts, report_path
from nsu_rag_llm.generation.answer import Answerer, Result
from nsu_rag_llm.generation.citations import Fragment
from nsu_rag_llm.generation.prompt import fragment_block
from nsu_rag_llm.retrieval.base import Hit

#: Builds the pipeline a variant answers with: its model, tokenizer, settings.
AnswererFor = Callable[[Variant], Answerer]
#: Answers one question, however it likes: from fixed fragments, or end to end.
Answer = Callable[[Question], Awaitable[Result]]
Progress = Callable[[str, int, int], None]


@dataclass
class GenerationRun:
    validation: Validation
    mode: str
    judge: str | None
    results: list[VariantResult] = field(default_factory=list)
    report: Path | None = None


def run_id(day: str, mode: str, variant: Variant) -> str:
    return f"{day}-{mode}-{variant.label}"


def record(question: Question, result: Result) -> Answered:
    return Answered(
        id=question.id,
        type=question.type,
        course=question.course,
        refused=result.refused,
        raw_answer=result.raw_answer,
        answer=result.answer,
        cited=result.cited,
        citation_errors=result.citation_errors,
        truncated=result.truncated,
        dropped=result.dropped,
        prompt_tokens=result.prompt_tokens,
        prompt_tokens_estimated=result.prompt_tokens_estimated,
        completion_tokens=result.completion_tokens,
        llm_first_output_ms=result.timings.llm_first_output_ms,
        llm_first_token_ms=result.timings.llm_first_token_ms,
        total_ms=result.timings.total_ms,
        sources=[
            {"number": s.number, "file": s.file, "pages": [s.page_start, s.page_end]}
            for s in result.sources
        ],
        routed_course=result.routed_course,
        steps=result.steps,
    )


async def answer_all(
    answer: Answer,
    questions: Sequence[Question],
    memory: Memory | None,
    model: str,
    on_progress: Progress | None = None,
    label: str = "",
) -> tuple[list[Answered], list[list[Fragment]], int | None]:
    """Every question answered; what each answer was built from; peak memory."""
    answered, shown, peak = [], [], None
    for done, question in enumerate(questions, start=1):
        result = await answer(question)
        answered.append(record(question, result))
        shown.append([Fragment(s.number, _chunk(s)) for s in result.sources])
        if memory is not None:
            size = memory.size_of(model)
            if size is not None:
                peak = max(peak or 0, size)
        if on_progress:
            on_progress(label, done, len(questions))
    return answered, shown, peak


def from_fragments(answerer: Answerer, fragments: dict[str, list[Hit]]) -> Answer:
    """Answer from the fragments retrieved once for every model."""

    async def answer(question: Question) -> Result:
        async for event in answerer.events_from(
            question.question, question.course, fragments[question.id]
        ):
            if event.kind == "done":
                return event.data
        raise RuntimeError("the answer stream ended without a result")

    return answer


def _chunk(source) -> dict:
    """The fields a fragment block is built from, back from a Source."""
    return {
        "context": source.context,
        "text": source.text,
        "page_start": source.page_start,
        "page_end": source.page_end,
        "course_title": source.course_title,
        "doc_title": source.doc_title,
    }


async def judge_all(
    judge: Judge,
    questions: dict[str, Question],
    answered: Sequence[Answered],
    shown: Sequence[list[Fragment]],
    on_progress: Progress | None = None,
    label: str = "",
) -> dict[str, Judged]:
    verdicts: dict[str, Judged] = {}
    for done, (answer, fragments) in enumerate(zip(answered, shown, strict=True), start=1):
        question = questions[answer.id]
        judged = Judged(answer.id)
        if not answer.refused:
            await _judge_one(judged, judge, question, answer, fragments)
        verdicts[answer.id] = judged
        if on_progress:
            on_progress(label, done, len(answered))
    return verdicts


async def _judge_one(
    judged: Judged, judge: Judge, question: Question, answer: Answered, fragments
) -> None:
    try:
        faith = await judge.faithfulness(question.question, fragments, answer.raw_answer)
        judged.faithful, judged.citations_correct = faith.faithful, faith.citations_correct
        judged.unsupported, judged.miscited, judged.uncited = (
            faith.unsupported,
            faith.miscited,
            faith.uncited,
        )
        judged.inconsistent |= not faith.consistent
    except JudgeError as exc:
        judged.errors.append(f"faithfulness: {exc}")
    if not question.answerable:
        return
    try:
        whole = await judge.completeness(
            question.question, question.reference_answer, answer.raw_answer
        )
        judged.complete, judged.missing = whole.complete, whole.missing
        judged.inconsistent |= not whole.consistent
    except JudgeError as exc:
        judged.errors.append(f"completeness: {exc}")


def write_review(
    path: Path,
    run: str,
    questions: dict[str, Question],
    answered: Sequence[Answered],
    shown: Sequence[list[Fragment]],
) -> None:
    """Everything a person needs to label a run's answers, in one readable file:
    question, reference, the answer, and the fragments the model read. It
    quotes the handbooks, so it lives in data/eval-runs/, never in git."""
    lines = [
        f"# {run} — ответы для ручной разметки",
        "",
        "Для каждого ответа решите: **faithful** — каждое утверждение есть во фрагментах "
        "ниже; **complete** — ответ покрывает эталон. Запишите в `eval/manual_labels.jsonl`:",
        "",
        f'    {{"id": "…", "answer_run": "{run}", "faithful": true, "complete": false}}',
        "",
    ]
    for answer, fragments in zip(answered, shown, strict=True):
        question = questions[answer.id]
        lines += [
            f"## {answer.id} · {question.type} · {question.course}",
            "",
            f"**Вопрос.** {question.question}",
            "",
            f"**Эталон.** {question.reference_answer or '—'}",
            "",
            "**Ответ.** " + ("*(отказ)* " if answer.refused else "") + answer.answer,
            "",
        ]
        if answer.citation_errors:
            lines += [f"*Ошибки ссылок: {'; '.join(answer.citation_errors)}*", ""]
        lines += ["**Фрагменты, которые читала модель.**", ""]
        lines += [f"> {fragment_block(f)}".replace("\n", "\n> ") + "\n" for f in fragments]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _store(settings: Settings, result: VariantResult, info: dict) -> None:
    folder = settings.eval_runs_dir / result.run_id
    write_jsonl(folder / "answers.jsonl", result.answered)
    (folder / "run.json").write_text(
        json.dumps(
            info | {"variant": vars(result.variant), "peak_memory_bytes": result.peak_memory_bytes},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


async def _judge(
    settings: Settings,
    run: GenerationRun,
    judge: Judge,
    questions: Sequence[Question],
    shown_by_label: dict[str, list[list[Fragment]]],
    on_progress: Progress | None,
) -> None:
    for result in run.results:
        result.judged = await judge_all(
            judge,
            by_id(list(questions)),
            result.answered,
            shown_by_label[result.variant.label],
            on_progress,
            f"судья: {result.variant.label}",
        )
        folder = settings.eval_runs_dir / result.run_id
        write_jsonl(folder / "judgements.jsonl", list(result.judged.values()))
        (folder / "judge.json").write_text(
            json.dumps(
                {"judge": judge.name, "reasoning_effort": settings.judge_reasoning_effort},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def _report(settings, run: GenerationRun, day: str, chunks: int, kind: str) -> None:
    from nsu_rag_llm.evaluation.generation_report import GenerationInfo, render

    info = GenerationInfo(
        date=day,
        commit=commit(),
        mode=run.mode,
        chunks=chunks,
        answer_fragments=settings.answer_fragments,
        judge=run.judge,
    )
    settings.eval_reports_dir.mkdir(parents=True, exist_ok=True)
    run.report = report_path(settings, day, kind)
    run.report.write_text(render(run.results, run.validation, info), encoding="utf-8")


async def run_generation(
    settings: Settings,
    variants: Sequence[Variant],
    mode: str,
    retrieve: Callable[[Question], list[Hit]],
    answerer_for: AnswererFor,
    judge: Judge | None,
    memory: Memory | None,
    questions_file: Path | None = None,
    release_retrieval: Callable[[], None] | None = None,
    on_progress: Progress | None = None,
    write_report: bool = True,
) -> GenerationRun:
    """Phase 4: the models compared on the same fragments."""
    indexed, _, chunk_count = index_facts(settings.chunks_file)
    validation = validate(questions_file or settings.questions_file, settings.raw_dir, indexed)
    run = GenerationRun(validation, mode, judge.name if judge else None)
    if not validation.ok:
        return run

    questions = validation.questions
    fragments = {}
    for done, question in enumerate(questions, start=1):
        fragments[question.id] = retrieve(question)
        if on_progress:
            on_progress("поиск", done, len(questions))
    if release_retrieval:
        release_retrieval()

    day = date.today().isoformat()
    shown_by_label = {}
    for variant in variants:
        if memory is not None:
            memory.unload_all()
        answer = from_fragments(answerer_for(variant), fragments)
        answered, shown, peak = await answer_all(
            answer, questions, memory, variant.model, on_progress, variant.label
        )
        result = VariantResult(
            variant,
            run_id(day, mode, variant),
            answered=answered,
            peak_memory_bytes=peak,
            room=settings.llm_context_tokens - variant.max_output_tokens,
        )
        run.results.append(result)
        shown_by_label[variant.label] = shown
        _store(
            settings,
            result,
            {
                "date": day,
                "commit": commit(),
                "mode": mode,
                "questions": len(questions),
                "chunks": chunk_count,
            },
        )
        write_review(
            settings.eval_runs_dir / result.run_id / "review.md",
            result.run_id,
            by_id(questions),
            answered,
            shown,
        )

    if judge is not None:
        if memory is not None:
            memory.unload_all()
        await _judge(settings, run, judge, questions, shown_by_label, on_progress)
    if write_report:
        _report(settings, run, day, chunk_count, "generation")
    return run


async def run_systems(
    settings: Settings,
    systems: dict[str, Answer],
    judge: Judge | None,
    questions_file: Path | None = None,
    on_progress: Progress | None = None,
    write_report: bool = True,
) -> GenerationRun:
    """Phase 5: whole systems compared end to end - the phase 3 pipeline and
    the agent search for themselves, as a student would ask, without a course."""
    indexed, _, chunk_count = index_facts(settings.chunks_file)
    validation = validate(questions_file or settings.questions_file, settings.raw_dir, indexed)
    run = GenerationRun(validation, settings.retrieval_mode, judge.name if judge else None)
    if not validation.ok:
        return run

    questions = validation.questions
    day = date.today().isoformat()
    shown_by_label = {}
    for label, answer in systems.items():
        variant = Variant(label, settings.llm_model, settings.llm_tokenizer)
        answered, shown, _ = await answer_all(
            answer, questions, None, settings.llm_model, on_progress, label
        )
        result = VariantResult(
            variant,
            run_id(day, settings.retrieval_mode, variant),
            answered=answered,
            room=settings.llm_context_tokens - settings.llm_max_output_tokens,
        )
        run.results.append(result)
        shown_by_label[label] = shown
        _store(
            settings,
            result,
            {
                "date": day,
                "commit": commit(),
                "mode": settings.retrieval_mode,
                "questions": len(questions),
                "chunks": chunk_count,
            },
        )
        write_review(
            settings.eval_runs_dir / result.run_id / "review.md",
            result.run_id,
            by_id(questions),
            answered,
            shown,
        )
    if judge is not None:
        await _judge(settings, run, judge, questions, shown_by_label, on_progress)
    if write_report:
        _report(settings, run, day, chunk_count, "agent")
    return run
