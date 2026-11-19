from __future__ import annotations

import json
from pathlib import Path


def lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def numbered(text: str) -> list[tuple[int, str]]:
    return [(n, line) for n, line in enumerate(text.split("\n"), start=1) if line.strip()]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in lines(path.read_text(encoding="utf-8"))]
