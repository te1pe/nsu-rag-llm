"""Source files must not carry characters a reader cannot see.

This project handles soft hyphens, zero-width spaces and no-break spaces on
purpose, so the code mentions them constantly - always as `\\u` escapes. A
literal one in the source is invisible in every editor: a test that looks like
it compares "процесс" with "процесс" would really be comparing two different
strings. Editors and tooling can turn an escape into the literal character
without anyone noticing, hence a test rather than a convention. The documents
are written with the same tools: report.md once picked up a literal U+2028.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("eval/*.md"),
    ]
)


def hidden_characters(text: str) -> list[str]:
    found = []
    for char in text:
        category = unicodedata.category(char)
        invisible = (
            category in ("Cf", "Zl", "Zp")
            or (category == "Zs" and char != " ")
            # Control characters: U+0085 and its kind read as line breaks too.
            or (category == "Cc" and char not in "\n\t\r")
        )
        if invisible or 0xFB00 <= ord(char) <= 0xFB06:  # ligatures read as plain letters
            found.append(f"U+{ord(char):04X}")
    return sorted(set(found))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_invisible_characters_in_source(path: Path) -> None:
    assert hidden_characters(path.read_text(encoding="utf-8")) == []
