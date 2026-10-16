"""The author's own verdicts on answers: `eval/manual_labels.jsonl`.

    {"id": "q001", "answer_run": "2026-10-05-hybrid+rerank-qwen3.5-9b-8k",
     "faithful": true, "complete": false, "note": "..."}

Written by the author and by nobody else: these labels are what the judge is
calibrated against, so a judge - or this code - writing them would be marking
its own homework. `answer_run` names a directory under `data/eval-runs/`,
where the answers that were read are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nsu_rag_llm import jsonl

REQUIRED = ("id", "answer_run", "faithful", "complete")
OPTIONAL = ("note",)


@dataclass(frozen=True)
class Label:
    id: str
    answer_run: str
    faithful: bool
    complete: bool
    note: str = ""


@dataclass
class LabelFile:
    labels: list[Label] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_labels(path: Path, runs_dir: Path | None = None) -> LabelFile:
    """Parse and check the labels; with `runs_dir`, each run must exist."""
    result = LabelFile()
    if not path.exists():
        result.errors.append(f"нет файла {path}")
        return result
    seen: set[tuple[str, str]] = set()
    for line_no, line in jsonl.numbered(path.read_text(encoding="utf-8")):
        where = f"строка {line_no}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            result.errors.append(f"{where}: не JSON ({exc.msg})")
            continue
        if not isinstance(raw, dict):
            result.errors.append(f"{where}: ожидался объект JSON")
            continue
        missing = [key for key in REQUIRED if key not in raw]
        if missing:
            result.errors.append(f"{where}: нет полей {', '.join(missing)}")
            continue
        extra = sorted(set(raw) - set(REQUIRED) - set(OPTIONAL))
        if extra:
            result.warnings.append(f"{where}: лишние поля {', '.join(extra)}")
        problems = [
            f"{key} должен быть строкой"
            for key in ("id", "answer_run")
            if not isinstance(raw[key], str) or not raw[key].strip()
        ]
        problems += [
            f"{key} должен быть true или false"
            for key in ("faithful", "complete")
            if not isinstance(raw[key], bool)
        ]
        if not isinstance(raw.get("note", ""), str):
            problems.append("note должен быть строкой")
        if problems:
            result.errors += [f"{where}: {p}" for p in problems]
            continue
        key = (raw["answer_run"], raw["id"])
        if key in seen:
            result.errors.append(f"{where}: {raw['id']} в {raw['answer_run']} уже размечен")
            continue
        seen.add(key)
        if runs_dir is not None and not (runs_dir / raw["answer_run"]).is_dir():
            result.errors.append(f"{where}: нет прогона {raw['answer_run']} в {runs_dir}")
            continue
        note = raw.get("note", "")
        result.labels.append(
            Label(raw["id"], raw["answer_run"], raw["faithful"], raw["complete"], note)
        )
    return result
