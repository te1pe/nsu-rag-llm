"""Phase 4: the judge, its calibration, the metrics and the model comparison.

All synthetic: a scripted fake LLM answers, a scripted fake judge rules, and
the questions and labels are written for these tests - the author's own
eval files are never read or written here.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from nsu_rag_llm.config import PROJECT_ROOT, Settings
from nsu_rag_llm.evaluation.agreement import agreement
from nsu_rag_llm.evaluation.calibration import calibrate
from nsu_rag_llm.evaluation.calibration import render as render_calibration
from nsu_rag_llm.evaluation.generation import (
    Answered,
    Judged,
    Variant,
    VariantResult,
    load_variants,
    write_jsonl,
)
from nsu_rag_llm.evaluation.generation_runner import run_generation, run_systems
from nsu_rag_llm.evaluation.judge import (
    CompletenessVerdict,
    FaithfulnessVerdict,
    Judge,
    JudgeError,
    parse,
)
from nsu_rag_llm.evaluation.labels import read_labels
from nsu_rag_llm.evaluation.questions import file_sha256
from nsu_rag_llm.generation.citations import Fragment
from nsu_rag_llm.generation.prompt import REFUSAL
from nsu_rag_llm.retrieval.base import Hit
from tests.test_evaluation import line, raw, write  # noqa: F401 - raw is a fixture
from tests.test_generation import ScriptedChat, chunk, make

# --- agreement ---------------------------------------------------------------


def test_kappa_matches_a_worked_example() -> None:
    """20 both yes, 5 only a, 10 only b, 15 both no: p_o 0.70, p_e 0.50, kappa 0.40."""
    a = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15

    result = agreement(a, b)

    assert result.observed == pytest.approx(0.70)
    assert result.kappa == pytest.approx(0.40)
    assert (result.both_yes, result.only_a, result.only_b, result.both_no) == (20, 5, 10, 15)


def test_a_judge_that_always_says_yes_scores_zero_however_often_it_agrees() -> None:
    result = agreement([True] * 9 + [False], [True] * 10)

    assert result.observed == pytest.approx(0.9)
    assert result.kappa == pytest.approx(0.0)


def test_perfect_agreement_is_one_and_one_label_everywhere_is_undefined() -> None:
    assert agreement([True, False, True], [True, False, True]).kappa == pytest.approx(1.0)
    assert math.isnan(agreement([True, True], [True, True]).kappa)


def test_labels_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError):
        agreement([True], [True, False])


# --- the author's labels -------------------------------------------------------


def label(**overrides) -> str:
    record = {"id": "q1", "answer_run": "run-a", "faithful": True, "complete": False}
    return json.dumps(record | overrides, ensure_ascii=False)


def test_the_example_labels_file_is_valid() -> None:
    labels = read_labels(PROJECT_ROOT / "eval" / "manual_labels.example.jsonl")

    assert labels.ok, labels.errors
    assert len(labels.labels) == 2


@pytest.mark.parametrize(
    ("record", "fragment"),
    [
        ({"faithful": "да"}, "faithful должен быть true или false"),
        ({"answer_run": ""}, "answer_run должен быть строкой"),
        ({"note": 5}, "note должен быть строкой"),
    ],
)
def test_each_label_error_is_reported(tmp_path: Path, record: dict, fragment: str) -> None:
    labels = read_labels(write(tmp_path / "l.jsonl", label(**record)))

    assert any(fragment in e for e in labels.errors), labels.errors


def test_a_missing_field_a_repeat_and_an_unknown_run_are_errors(tmp_path: Path) -> None:
    (tmp_path / "runs" / "run-a").mkdir(parents=True)
    path = write(
        tmp_path / "l.jsonl",
        json.dumps({"id": "q1", "answer_run": "run-a"}),
        label(),
        label(),
        label(answer_run="run-b"),
    )

    errors = read_labels(path, tmp_path / "runs").errors

    assert any("нет полей" in e for e in errors)
    assert any("уже размечен" in e for e in errors)
    assert any("нет прогона run-b" in e for e in errors)


# --- the judge -----------------------------------------------------------------


class ScriptedJson:
    """A fake judge model: replies from a script, records what it was asked."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[dict], str, dict]] = []

    async def complete(self, messages, name, schema) -> str:
        self.calls.append((messages, name, schema))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


FAITHFUL = json.dumps(
    {
        "unsupported": [],
        "miscited": [],
        "uncited": [],
        "faithful": True,
        "citations_correct": True,
    }
)
COMPLETE = json.dumps({"missing": [], "complete": True})


def test_a_verdict_is_read_through_a_fence_or_prose() -> None:
    assert parse(f"```json\n{COMPLETE}\n```", CompletenessVerdict).complete
    assert parse(f"Вот вердикт: {COMPLETE} Всё.", CompletenessVerdict).complete


@pytest.mark.parametrize("reply", ["не знаю", '{"complete": "наверное"}', '{"missing": []}'])
def test_a_reply_that_is_not_a_verdict_is_a_judge_error(reply: str) -> None:
    with pytest.raises(JudgeError):
        parse(reply, CompletenessVerdict)


def test_yes_or_no_contradicting_the_list_is_noticed() -> None:
    verdict = FaithfulnessVerdict(
        unsupported=["формула не из фрагментов"],
        miscited=[],
        uncited=[],
        faithful=True,
        citations_correct=True,
    )

    assert not verdict.consistent


def test_an_unsupported_claim_with_a_label_may_count_against_the_citations() -> None:
    """Seen live: the judge lists such a claim as unsupported only, and rightly
    says the citations are wrong - that is not a contradiction."""
    verdict = FaithfulnessVerdict(
        unsupported=["объём кэша L1 всегда 64 КБ"],
        miscited=[],
        uncited=[],
        faithful=False,
        citations_correct=False,
    )

    assert verdict.consistent


async def test_the_judge_sees_the_labelled_fragments_and_the_raw_answer() -> None:
    model = ScriptedJson(FAITHFUL)
    fragments = [Fragment(1, chunk("a", "Синтетический текст о процессах."))]

    judge = Judge(model, "fake")

    verdict = await judge.faithfulness("Что такое процесс?", fragments, "Ответ [Ф1].")

    messages, name, schema = model.calls[0]
    assert verdict.faithful
    assert name == "faithfulness"
    assert re.search(r"\[Ф\d+\]", messages[0]["content"])  # the rules show the label format
    user = messages[1]["content"]
    assert "[Ф1] Операционные системы" in user and "Синтетический текст" in user
    assert user.endswith("Ответ ассистента:\nОтвет [Ф1].")
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "unsupported",
        "miscited",
        "uncited",
        "faithful",
        "citations_correct",
    }


async def test_completeness_compares_with_the_reference_answer() -> None:
    model = ScriptedJson(json.dumps({"missing": ["второе отличие"], "complete": False}))

    verdict = await Judge(model, "fake").completeness("Вопрос?", "Эталон.", "Ответ.")

    assert verdict.missing == ["второе отличие"]
    assert "Эталонный ответ:\nЭталон." in model.calls[0][0][1]["content"]


# --- the models compared ---------------------------------------------------------


def test_the_repository_variants_load_and_each_model_has_a_modelfile() -> None:
    """Every compared model must be buildable, with the 8k window, from ollama/."""
    variants = load_variants(PROJECT_ROOT / "eval" / "models.yaml")
    built = {}
    for path in (PROJECT_ROOT / "ollama").glob("Modelfile*"):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"ollama create (\S+) -f", text)
        if match:
            built[match.group(1)] = text

    assert len({v.label for v in variants}) == len(variants)
    for variant in variants:
        assert variant.model in built, f"no Modelfile builds {variant.model}"
        assert "PARAMETER num_ctx 8192" in built[variant.model]


def test_repeated_labels_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "- {label: a, model: m, tokenizer: t}\n- {label: a, model: n, tokenizer: t}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="повторяются"):
        load_variants(path)


# --- metrics ---------------------------------------------------------------------


def answered(qid: str, qtype: str = "factual", refused: bool = False, **fields) -> Answered:
    base = {
        "id": qid,
        "type": qtype,
        "course": "os",
        "refused": refused,
        "raw_answer": "",
        "answer": "",
        "cited": [1],
        "citation_errors": [],
        "truncated": False,
        "dropped": 0,
        "prompt_tokens": 1000,
        "prompt_tokens_estimated": 1010,
        "completion_tokens": 100,
        "llm_first_output_ms": 2000.0,
        "llm_first_token_ms": 2000.0,
        "total_ms": 12000.0,
        "sources": [{"number": 1}],
    }
    return Answered(**(base | fields))


def test_metrics_count_refusals_where_they_belong() -> None:
    result = VariantResult(
        Variant("v", "m", "t"),
        "run",
        answered=[
            answered("a"),
            answered("b", citation_errors=["ссылка [Ф9] на несуществующий фрагмент"]),
            answered("c", refused=True),
            answered("u1", "unanswerable", refused=True),
            answered("u2", "unanswerable"),
        ],
        judged={
            "a": Judged("a", faithful=True, citations_correct=True, complete=True),
            "b": Judged("b", faithful=False, citations_correct=False, complete=False),
            "u2": Judged("u2", faithful=True, citations_correct=True),
        },
        room=7168,
    )

    assert result.faithfulness() == pytest.approx(2 / 3)  # a, b, u2: refusals aside
    assert result.citations_valid() == pytest.approx(2 / 3)  # b cites a missing fragment
    assert result.completeness() == pytest.approx(1 / 3)  # a of a, b, c(refused)
    assert result.refusal_accuracy() == pytest.approx(1 / 2)
    assert result.false_refusals() == pytest.approx(1 / 3)
    assert result.tokens_per_second() == pytest.approx(10.0)  # 100 tokens in 10 s
    assert result.ttft_ms() == pytest.approx(2000.0)
    assert result.estimate_error() == (10, 10)


def test_a_metric_over_nothing_is_undefined_not_zero() -> None:
    result = VariantResult(Variant("v", "m", "t"), "run", answered=[answered("a")])

    assert math.isnan(result.refusal_accuracy())  # no unanswerable questions
    assert math.isnan(result.faithfulness())  # nothing judged


def test_thinking_tokens_are_timed_from_the_first_output_not_the_answer() -> None:
    """Seen live: 394 completion tokens, of which the answer came 17 s in."""
    thinking = answered("a", llm_first_output_ms=1000.0, llm_first_token_ms=6000.0)

    assert thinking.tokens_per_second == pytest.approx(100 / 11)  # 100 tokens, 1 s .. 12 s


def test_a_prompt_the_server_counted_over_the_room_is_flagged() -> None:
    result = VariantResult(
        Variant("v", "m", "t"), "run", answered=[answered("a", prompt_tokens=7200)], room=7168
    )

    assert result.over_budget() == 1


# --- the whole comparison ----------------------------------------------------------


class Memory:
    def __init__(self, sizes: list[int], log: list[str]) -> None:
        self.sizes = sizes
        self.log = log

    def size_of(self, model: str) -> int | None:
        return self.sizes.pop(0) if self.sizes else None

    def unload_all(self) -> None:
        self.log.append("unload")


SECRET_QUESTION = "Синтетический вопрос о семафорах?"
SECRET_FRAGMENT = "Секретный синтетический текст фрагмента."


@pytest.fixture
def comparison(tmp_path: Path, raw: Path):  # noqa: F811 - raw is the fixture
    sha = file_sha256(raw / "os" / "sem4" / "book.pdf")
    fragment = chunk("c1", SECRET_FRAGMENT, start=3) | {"doc_sha256": sha, "origin": "digital"}
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps(fragment, ensure_ascii=False) + "\n", encoding="utf-8")
    questions = write(
        tmp_path / "q.jsonl",
        line(id="q1"),
        line(id="q2", type="unanswerable", gold=[], reference_answer=""),
    )
    settings = Settings(
        _env_file=None,
        raw_dir=raw,
        chunks_file=chunks,
        questions_file=questions,
        eval_reports_dir=tmp_path / "reports",
        eval_runs_dir=tmp_path / "runs",
    )
    return settings, fragment


async def test_every_model_answers_the_same_fragments_and_is_judged(comparison) -> None:
    settings, fragment = comparison
    log: list[str] = []
    variants = [Variant("good", "m-good", "t"), Variant("shy", "m-shy", "t")]
    scripts = {"good": ("Семафор — счётчик [Ф1].",), "shy": (REFUSAL,)}

    def retrieve(question):
        log.append(f"retrieve {question.id}")
        return [Hit(fragment, 1.0)]

    def answerer_for(variant):
        log.append(f"answer with {variant.label}")
        return make([], ScriptedChat(*scripts[variant.label]))

    judge_model = ScriptedJson(FAITHFUL, COMPLETE, FAITHFUL)
    run = await run_generation(
        settings,
        variants,
        "hybrid+rerank",
        retrieve,
        answerer_for,
        Judge(judge_model, "fake-judge"),
        Memory([5_000_000_000, 6_000_000_000, 4_000_000_000, 4_000_000_000], log),
        release_retrieval=lambda: log.append("release"),
    )

    assert run.validation.ok
    # Searched once per question, released before any model answered; every
    # model starts alone, and so does the judge.
    assert log == [
        "retrieve q1",
        "retrieve q2",
        "release",
        "unload",
        "answer with good",
        "unload",
        "answer with shy",
        "unload",
    ]
    good, shy = run.results
    assert good.peak_memory_bytes == 6_000_000_000
    assert good.faithfulness() == 1.0 and good.completeness() == 1.0
    assert good.refusal_accuracy() == 0.0  # it answered the unanswerable one
    assert shy.refusal_accuracy() == 1.0 and shy.false_refusals() == 1.0
    # Refusals are never sent to the judge; completeness only for answerable ones.
    assert [call[1] for call in judge_model.calls] == [
        "faithfulness",
        "completeness",
        "faithfulness",
    ]
    folder = settings.eval_runs_dir / good.run_id
    assert (folder / "answers.jsonl").exists() and (folder / "judgements.jsonl").exists()
    # For labelling by hand: question, answer and the fragments, in one file.
    review = (folder / "review.md").read_text(encoding="utf-8")
    assert "## q1 · definition · os" in review and SECRET_FRAGMENT in review
    assert json.loads((folder / "judge.json").read_text(encoding="utf-8"))["judge"] == "fake-judge"


async def test_the_report_holds_ids_and_numbers_but_no_text(comparison) -> None:
    settings, fragment = comparison

    run = await run_generation(
        settings,
        [Variant("good", "m", "t")],
        "hybrid",
        lambda q: [Hit(fragment, 1.0)],
        lambda v: make([], ScriptedChat("Секретный ответ модели [Ф1].")),
        Judge(ScriptedJson(FAITHFUL, COMPLETE, FAITHFUL), "fake-judge"),
        None,
    )

    text = run.report.read_text(encoding="utf-8")
    assert "q1" in text and "q2" in text
    for secret in (SECRET_QUESTION, SECRET_FRAGMENT, "Секретный ответ", "Синтетический ответ"):
        assert secret not in text
    # The answers themselves are kept, outside the repository's reports.
    stored = (settings.eval_runs_dir / run.results[0].run_id / "answers.jsonl").read_text(
        encoding="utf-8"
    )
    assert "Секретный ответ модели" in stored


async def test_an_invalid_questions_file_stops_before_any_search(comparison, tmp_path) -> None:
    settings, _ = comparison

    def never(*_):
        raise AssertionError("nothing may run on an invalid file")

    run = await run_generation(
        settings,
        [Variant("v", "m", "t")],
        "hybrid",
        never,
        never,
        None,
        None,
        questions_file=write(tmp_path / "bad.jsonl", line(type="opinion")),
    )

    assert not run.validation.ok
    assert run.report is None


async def test_a_judge_that_talks_nonsense_is_counted_not_trusted(comparison) -> None:
    settings, fragment = comparison

    run = await run_generation(
        settings,
        [Variant("v", "m", "t")],
        "hybrid",
        lambda q: [Hit(fragment, 1.0)],
        lambda v: make([], ScriptedChat("Ответ [Ф1].")),
        Judge(ScriptedJson("не знаю"), "fake-judge"),
        None,
    )

    result = run.results[0]
    assert result.judge_failures() == 2
    assert math.isnan(result.faithfulness())


# --- calibration -----------------------------------------------------------------


def test_calibration_pairs_the_authors_labels_with_the_judges_verdicts(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_jsonl(
        runs / "run-a" / "judgements.jsonl",
        [
            Judged("q1", faithful=True, complete=True),
            Judged("q2", faithful=False, complete=True),
            Judged("q3", errors=["faithfulness: нет JSON"]),
        ],
    )
    (runs / "run-a" / "judge.json").write_text('{"judge": "fake-judge"}', encoding="utf-8")
    labels = write(
        tmp_path / "labels.jsonl",
        label(id="q1", faithful=True, complete=True, note="Личная заметка автора"),
        label(id="q2", faithful=True, complete=False),
        label(id="q3", faithful=True, complete=True),
    )

    calibration = calibrate(labels, runs)

    assert calibration.labels.ok
    assert calibration.faithful.n == 2 and calibration.faithful.observed == 0.5
    assert calibration.complete.n == 2 and calibration.complete.observed == 0.5
    assert calibration.unmatched == ["run-a/q3"]
    assert calibration.judges == ["fake-judge"]
    report = render_calibration(calibration)
    assert "fake-judge" in report
    assert "Личная заметка автора" not in report


# --- whole systems (phase 5) ---------------------------------------------------------


async def test_systems_are_compared_end_to_end_with_the_routers_accuracy(comparison) -> None:
    settings, fragment = comparison
    pipeline = make([Hit(fragment, 1.0)], ScriptedChat("Семафор — счётчик [Ф1]."))

    async def plain(question):
        return await pipeline.ask(question.question)

    async def routed(question):
        result = await pipeline.ask(question.question)
        # q1 is an os question: routed right; q2 is unanswerable and not scored.
        result.routed_course = "os" if question.id == "q1" else "none"
        result.steps = ["маршрут: os", "поиск (os, попытка 1): лучший скор 0.9"]
        return result

    run = await run_systems(
        settings,
        {"pipeline": plain, "agent": routed},
        Judge(ScriptedJson(FAITHFUL, COMPLETE, FAITHFUL), "fake-judge"),
    )

    pipeline_result, agent_result = run.results
    assert not pipeline_result.routed() and agent_result.routed()
    assert agent_result.router_accuracy() == 1.0
    assert agent_result.mean_searches() == 1.0
    report = run.report.read_text(encoding="utf-8")
    assert run.report.name.endswith("-agent.md")
    assert "## Агент" in report and "`agent`" in report
    assert "маршрут: os" not in report  # steps stay in data/eval-runs
