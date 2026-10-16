"""Reading JSON Lines: one record per "\\n", and nothing else.

`str.splitlines()` also breaks lines at U+2028, U+2029, U+0085 and a few
other separators - characters that `json.dumps(..., ensure_ascii=False)`
leaves unescaped inside strings. A page holding one - U+0085, in the garbage
a font with no Unicode mapping left in two documents of the full corpus - was
cut in two in the middle of a string and failed to parse. The JSON Lines
format separates records by "\\n" alone, and so does this.
"""

from __future__ import annotations

import json
from pathlib import Path


def lines(text: str) -> list[str]:
    """The records' lines, blank ones dropped."""
    return [line for line in text.split("\n") if line.strip()]


def numbered(text: str) -> list[tuple[int, str]]:
    """(1-based line number, line) for every non-blank line - for messages."""
    return [(n, line) for n, line in enumerate(text.split("\n"), start=1) if line.strip()]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in lines(path.read_text(encoding="utf-8"))]
