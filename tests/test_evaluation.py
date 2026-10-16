"""The retrieval evaluation: the validator, the hit rule, the metrics, the report.

Every question here is synthetic and written for the test: the real
eval/questions.jsonl is the author's, and no test reads or writes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from nsu_rag_llm.config import Settings
from nsu_rag_llm.evaluation.questions import Gold, Question, Validation, file_sha256, validate
from nsu_rag_llm.evaluation.report import RunInfo, render
from nsu_rag_llm.evaluation.retrieval import answers, evaluate, score_question
from nsu_rag_llm.evaluation.runner import run_retrieval
from nsu_rag_llm.retrieval.base import Hit


def line(**overrides) -> str:
    record = {
        "id": "q1",
        "course": "os",
        "question": "Синтетический вопрос о семафорах?",
        "type": "definition",
        "gold": [{"file": "os/sem4/book.pdf", "pages": [3]}],
        "reference_answer": "Синтетический ответ.",
    } | overrides
    return json.dumps(record, ensure_ascii=False)


def write(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """A corpus of one 5-page synthetic PDF, plus a byte-identical copy."""
    root = tmp_path / "raw"
    (root / "os" / "sem4").mkdir(parents=True)
    (root / "os" / "sem5").mkdir(parents=True)
    doc = pymupdf.open()
    for n in range(5):
        doc.new_page().insert_text((72, 72), f"Page {n + 1}")
    doc.save(root / "os" / "sem4" / "book.pdf")
    doc.close()
    (root / "os" / "sem5" / "copy.pdf").write_bytes(
        (root / "os" / "sem4" / "book.pdf").read_bytes()
    )
    return root


def test_a_well_formed_file_passes(tmp_path: Path, raw: Path) -> None:
    path = write(tmp_path / "q.jsonl", line(), line(id="q2", type="unanswerable", gold=[]))

    result = validate(path, raw)

    assert result.ok, result.errors
    assert [q.id for q in result.questions] == ["q1", "q2"]


@pytest.mark.parametrize(
    ("record", "fragment"),
    [
        ({"type": "opinion"}, "type"),
        ({"gold": []}, "хотя бы один gold"),
        ({"type": "unanswerable"}, "должен быть пустым"),
        ({"gold": [{"file": "os/sem4/book.pdf", "pages": [0]}]}, "страниц ≥ 1"),
        ({"gold": [{"file": "os/sem4/book.pdf"}]}, "gold[0]"),
        ({"reference_answer": "   "}, "reference_answer пустой"),
        ({"question": ""}, "question пустой"),
    ],
)
def test_each_schema_error_is_reported(tmp_path: Path, record: dict, fragment: str) -> None:
    result = validate(write(tmp_path / "q.jsonl", line(**record)))

    assert not result.ok
    assert any(fragment in error for error in result.errors), result.errors


def test_broken_json_and_duplicate_ids_are_errors(tmp_path: Path) -> None:
    result = validate(write(tmp_path / "q.jsonl", line(), "{not json", line()))

    assert any("не JSON" in e for e in result.errors)
    assert any("уже был" in e for e in result.errors)


def test_gold_must_exist_and_its_pages_must_be_inside(tmp_path: Path, raw: Path) -> None:
    path = write(
        tmp_path / "q.jsonl",
        line(gold=[{"file": "os/sem4/missing.pdf", "pages": [1]}]),
        line(id="q2", gold=[{"file": "os/sem4/book.pdf", "pages": [9]}]),
    )

    result = validate(path, raw)

    assert any("нет файла" in e for e in result.errors)
    assert any("5 стр., а указана 9" in e for e in result.errors)


def test_an_unindexed_gold_document_is_a_warning(tmp_path: Path, raw: Path) -> None:
    result = validate(write(tmp_path / "q.jsonl", line()), raw, indexed=set())

    assert result.ok
    assert any("нет в индексе" in w for w in result.warnings)


def test_a_single_source_multi_hop_is_a_warning(tmp_path: Path) -> None:
    result = validate(write(tmp_path / "q.jsonl", line(type="multi_hop")))

    assert any("multi_hop с одним источником" in w for w in result.warnings)


def chunk(sha: str, start: int, end: int, cid: str = "c") -> dict:
    return {"id": cid, "doc_sha256": sha, "page_start": start, "page_end": end}


def test_a_chunk_answers_when_its_pages_overlap_the_gold_pages() -> None:
    gold = Gold("os/sem4/book.pdf", (34, 35))
    sha = {"os/sem4/book.pdf": "s"}

    assert answers(chunk("s", 33, 34), gold, sha)
    assert not answers(chunk("s", 36, 37), gold, sha)
    assert not answers(chunk("other", 34, 34), gold, sha)


def test_a_duplicate_copy_named_as_gold_still_matches(raw: Path) -> None:
    """Gold may name any copy; chunks carry the indexed copy's hash."""
    sha_of_copy = file_sha256(raw / "os" / "sem5" / "copy.pdf")
    gold = Gold("os/sem5/copy.pdf", (2,))

    assert answers(chunk(sha_of_copy, 2, 2), gold, {"os/sem5/copy.pdf": sha_of_copy})


def question(*gold: Gold, qtype: str = "factual") -> Question:
    return Question("q", "os", "?", qtype, tuple(gold), "…")


def test_recall_and_rank_for_a_single_source() -> None:
    q = question(Gold("f", (5,)))
    hits = [Hit(chunk("s", p, p, str(p)), 1.0) for p in (1, 2, 3, 4, 9, 5, 7)]

    result = score_question(q, hits, {"f": "s"}, "digital", 0.0)

    assert result.first_rank == 6
    assert result.coverage == {5: 0.0, 10: 1.0}


def test_multi_hop_recall_is_partial_when_one_source_is_found() -> None:
    q = question(Gold("f", (5,)), Gold("g", (9,)), qtype="multi_hop")
    hits = [Hit(chunk("sf", 5, 5), 1.0)]

    result = score_question(q, hits, {"f": "sf", "g": "sg"}, "digital", 0.0)

    assert result.coverage[10] == 0.5
    assert result.first_rank == 1


class Fixed:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.courses: list[str | None] = []

    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]:
        self.courses.append(course)
        return self.hits[:k]


def test_unanswerable_questions_are_not_scored_here() -> None:
    questions = [question(Gold("f", (1,))), Question("u", "os", "?", "unanswerable", (), "")]

    result = evaluate("bm25", Fixed([Hit(chunk("s", 1, 1), 1.0)]), questions, {"f": "s"}, {})

    assert [r.question.id for r in result.results] == ["q"]
    assert result.recall(10) == 1.0
    assert result.mrr() == 1.0


def test_the_course_filter_passes_the_questions_course() -> None:
    retriever = Fixed([])

    evaluate("bm25", retriever, [question(Gold("f", (1,)))], {"f": "s"}, {}, filtered=True)

    assert retriever.courses == ["os"]


def test_the_report_holds_ids_and_numbers_but_no_question_text(tmp_path: Path) -> None:
    secret = "Секретная формулировка вопроса"
    q = Question("q7", "os", secret, "factual", (Gold("f", (1,)),), "Секретный ответ")
    result = evaluate(
        "bm25", Fixed([Hit(chunk("s", 1, 1), 1.0)]), [q], {"f": "s"}, {"s": "digital"}
    )
    info = RunInfo("2026-01-01", "abc", 1, "e5", "bge", 50, 60, 30, True)

    text = render([result], [], Validation(questions=[q]), info)

    assert "q7" in text
    assert secret not in text
    assert "Секретный ответ" not in text


def test_the_runner_validates_scores_and_writes_a_report(tmp_path: Path, raw: Path) -> None:
    sha = file_sha256(raw / "os" / "sem4" / "book.pdf")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(chunk(sha, 3, 3, "c1") | {"origin": "digital"}) + "\n", encoding="utf-8"
    )
    questions = write(tmp_path / "q.jsonl", line())
    settings = Settings(
        _env_file=None,
        raw_dir=raw,
        chunks_file=chunks,
        questions_file=questions,
        eval_reports_dir=tmp_path / "reports",
    )
    hits = [Hit(chunk(sha, 3, 3, "c1"), 1.0)]

    run = run_retrieval(settings, ["bm25", "dense"], lambda mode: Fixed(hits))

    assert run.validation.ok
    assert [r.mode for r in run.unfiltered] == ["bm25", "dense"]
    assert all(r.recall(5) == 1.0 for r in run.unfiltered)
    assert run.report is not None and run.report.exists()


def test_the_runner_stops_on_an_invalid_file(tmp_path: Path, raw: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        raw_dir=raw,
        chunks_file=chunks,
        questions_file=write(tmp_path / "q.jsonl", line(type="opinion")),
        eval_reports_dir=tmp_path / "reports",
    )

    def never(mode):
        raise AssertionError("no retriever should be built for an invalid file")

    run = run_retrieval(settings, ["bm25"], never)

    assert not run.validation.ok
    assert run.report is None


def test_the_questions_skeleton_holds_the_planned_slots() -> None:
    """The skeleton the author fills in: structure only, never a question."""
    from collections import Counter

    from nsu_rag_llm.config import PROJECT_ROOT

    path = PROJECT_ROOT / "eval" / "questions.skeleton.jsonl"
    slots = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]

    assert len(slots) == 60
    assert all(slot["question"] == "" for slot in slots)
    assert len({slot["id"] for slot in slots}) == 60
    mvp = Counter(slot["course"] for slot in slots[:50])
    assert mvp == dict.fromkeys(("lop", "matan", "os", "tvims", "vychmat"), 10)
    for slot in slots:
        if slot["type"] == "unanswerable":
            assert slot["gold"] == [] and slot["reference_answer"] == "В методичках этого нет."
        if slot["type"] == "multi_hop":
            assert len(slot["gold"]) == 2
