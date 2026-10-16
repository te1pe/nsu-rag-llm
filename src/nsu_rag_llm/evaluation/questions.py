"""The evaluation questions: their schema, and a validator for them.

`eval/questions.jsonl` is written by the author and by nobody else - a set of
questions produced by the system under test would measure nothing. This module
only reads it, and says precisely what is wrong with it: errors stop an
evaluation, warnings do not.

    {"id": "q001", "course": "os", "question": "...", "type": "comparison",
     "gold": [{"file": "os/sem4/<file>.pdf", "pages": [34, 35]}],
     "reference_answer": "..."}

`file` is relative to `data/raw/`; `pages` are 1-based positions in the PDF
file - what a viewer's page field shows, not the number printed on the page
(ADR-011). A gold page is a hit when a retrieved chunk of the same document
spans it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from nsu_rag_llm import jsonl

QUESTION_TYPES = ("factual", "definition", "comparison", "multi_hop", "unanswerable")
REQUIRED = ("id", "course", "question", "type", "gold", "reference_answer")


@dataclass(frozen=True)
class Gold:
    file: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class Question:
    id: str
    course: str
    question: str
    type: str
    gold: tuple[Gold, ...]
    reference_answer: str

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


@dataclass
class Validation:
    questions: list[Question] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gold_sha: dict[str, str] = field(default_factory=dict)
    """Gold file path -> sha256 of its bytes, for matching chunks by content."""

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse(line_no: int, raw: dict, result: Validation) -> Question | None:
    where = f"строка {line_no}"
    missing = [key for key in REQUIRED if key not in raw]
    if missing:
        result.errors.append(f"{where}: нет полей {', '.join(missing)}")
        return None
    extra = sorted(set(raw) - set(REQUIRED))
    if extra:
        result.warnings.append(f"{where} ({raw['id']}): лишние поля {', '.join(extra)}")

    problems = []
    for key in ("id", "course", "question", "reference_answer"):
        if not isinstance(raw[key], str):
            problems.append(f"{key} должен быть строкой")
    for key in ("id", "course", "question"):
        if isinstance(raw[key], str) and not raw[key].strip():
            problems.append(f"{key} пустой")
    if raw["type"] not in QUESTION_TYPES:
        problems.append(f"type {raw['type']!r} не из {', '.join(QUESTION_TYPES)}")
    if not isinstance(raw["gold"], list):
        problems.append("gold должен быть списком")

    gold: list[Gold] = []
    for index, item in enumerate(raw["gold"] if isinstance(raw["gold"], list) else []):
        label = f"gold[{index}]"
        if not isinstance(item, dict) or set(item) != {"file", "pages"}:
            problems.append(f"{label} должен быть {{'file': ..., 'pages': [...]}}")
            continue
        pages = item["pages"]
        if not isinstance(item["file"], str) or not item["file"].strip():
            problems.append(f"{label}.file должен быть непустой строкой")
            continue
        if (
            not isinstance(pages, list)
            or not pages
            or not all(isinstance(p, int) and not isinstance(p, bool) and p >= 1 for p in pages)
        ):
            problems.append(f"{label}.pages должен быть непустым списком номеров страниц ≥ 1")
            continue
        gold.append(Gold(item["file"], tuple(pages)))

    if raw["type"] == "unanswerable" and raw.get("gold"):
        problems.append("у unanswerable gold должен быть пустым")
    if raw["type"] in QUESTION_TYPES and raw["type"] != "unanswerable" and not raw.get("gold"):
        problems.append("у вопроса с ответом нужен хотя бы один gold")
    reference = raw["reference_answer"]
    if raw["type"] != "unanswerable" and isinstance(reference, str) and not reference.strip():
        problems.append("reference_answer пустой")

    if problems:
        result.errors += [f"{where} ({raw['id']}): {p}" for p in problems]
        return None
    if raw["type"] == "multi_hop" and len(gold) < 2:
        result.warnings.append(f"{where} ({raw['id']}): multi_hop с одним источником")
    return Question(
        id=raw["id"],
        course=raw["course"],
        question=raw["question"],
        type=raw["type"],
        gold=tuple(gold),
        reference_answer=raw["reference_answer"],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def page_count(path: Path) -> int | None:
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001 - the validator reports, it does not crash
        return None


def validate(
    path: Path,
    raw_dir: Path | None = None,
    indexed: set[str] | None = None,
) -> Validation:
    """Parse and check the questions file.

    With `raw_dir`, every gold file must exist and every page must lie inside
    it. With `indexed` - the sha256 of every document in the index - a gold
    document that retrieval can never find earns a warning.
    """
    result = Validation()
    if not path.exists():
        result.errors.append(f"нет файла {path}")
        return result

    seen: set[str] = set()
    for line_no, line in jsonl.numbered(path.read_text(encoding="utf-8")):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            result.errors.append(f"строка {line_no}: не JSON ({exc.msg})")
            continue
        if not isinstance(raw, dict):
            result.errors.append(f"строка {line_no}: ожидался объект JSON")
            continue
        question = _parse(line_no, raw, result)
        if question is None:
            continue
        if question.id in seen:
            result.errors.append(f"строка {line_no}: id {question.id!r} уже был")
            continue
        seen.add(question.id)
        result.questions.append(question)

    if raw_dir is not None:
        _check_corpus(result, raw_dir, indexed)
    return result


def _check_corpus(result: Validation, raw_dir: Path, indexed: set[str] | None) -> None:
    hashes = result.gold_sha
    counts: dict[str, int | None] = {}
    for question in result.questions:
        for gold in question.gold:
            path = raw_dir / gold.file
            if not path.is_file():
                result.errors.append(f"{question.id}: нет файла data/raw/{gold.file}")
                continue
            if gold.file not in counts:
                counts[gold.file] = page_count(path)
                hashes[gold.file] = file_sha256(path)
            pages = counts[gold.file]
            if pages is not None and max(gold.pages) > pages:
                result.errors.append(
                    f"{question.id}: в {gold.file} {pages} стр., а указана {max(gold.pages)}"
                )
            if gold.file.split("/")[0] != question.course:
                result.warnings.append(
                    f"{question.id}: курс вопроса {question.course}, а источник из "
                    f"{gold.file.split('/')[0]}"
                )
            if indexed is not None and hashes[gold.file] not in indexed:
                result.warnings.append(
                    f"{question.id}: {gold.file} нет в индексе (не в MVP или скан) — "
                    "поиск его не найдёт, вопрос будет промахом"
                )
