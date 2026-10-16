"""Clean parser output without damaging what it cannot understand.

Four jobs, in this order:

1. Unicode: NFC, and the invisible characters that split one word into two
   tokens for BM25 (soft hyphens, zero-width spaces), plus typographic
   ligatures. Deliberately *not* NFKC: that would turn `x²` into `x2` and
   quietly rewrite formulas.
2. Hyphenation: `пере-\\nнос` becomes `перенос`, but only when the next line
   starts in lower case, so that a compound broken exactly at its own hyphen
   (`северо-\\nЗападный`) is left alone.
3. Whitespace: trailing spaces, runs of spaces, runs of blank lines.
4. Running headers and footers: lines that recur at the top or bottom edge of
   many pages. The layout model already drops the ones it recognises; this is
   the safety net for the ones it does not.

Code listings are fenced in Markdown, and nothing inside a fence is touched
except the invisible characters: indentation and line breaks are the
structure of code, and CLAUDE.md forbids "fixing" listings and formulas.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

#: Recorded next to every cleaned document, so a report can say which rules
#: produced it. Bump when a rule changes.
CLEANING_REVISION = 1

FORMULA = "[формула]"
PICTURE = "[рисунок]"
PLACEHOLDERS = frozenset({FORMULA, PICTURE})

INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)
#: Spaces that are typographic, not structural: no-break, thin, narrow.
ODD_SPACES = re.compile(r"[\u00a0\u2007\u2009\u202f]")

HYPHENATED = re.compile(r"([^\W\d_])-[ \t]*\n[ \t]*([a-zа-яё])")
SPACE_RUN = re.compile(r"(?<=\S)[ \t]{2,}")
BLANK_RUN = re.compile(r"\n{3,}")
FENCE = re.compile(r"^\s*```")

#: How many non-empty lines at each end of a page count as its edge. One: a
#: running header is by definition the outermost line, and on a short page a
#: wider edge would cover the whole text.
EDGE_LINES = 1
#: A line recurring at an edge on at least this share of pages is a running
#: header or footer - "most pages", as CLAUDE.md puts it. Deliberately strict:
#: digits are normalised away so that page numbers match, which also makes
#: "Задача 12." and "Задача 13." the same line, and a problem book can start
#: many pages with a problem. Missing a header costs a little noise; removing
#: a line of content loses it, and the layout model catches most headers first.
EDGE_SHARE = 0.5
#: Short documents cannot tell a running header from a coincidence.
EDGE_MIN_PAGES = 5


@dataclass
class PageStats:
    hyphens_joined: int = 0
    edge_lines_removed: int = 0


def normalise_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(INVISIBLE)
    return text.translate(LIGATURES)


def split_fenced(markdown: str) -> list[tuple[bool, str]]:
    """Split into (is_code, segment) pieces along ``` fences.

    An unclosed fence runs to the end of the page: better to leave a stretch of
    prose uncleaned than to reflow a listing.
    """
    pieces: list[tuple[bool, str]] = []
    current: list[str] = []
    in_code = False
    for line in markdown.splitlines(keepends=True):
        if FENCE.match(line):
            if in_code:
                current.append(line)
                pieces.append((True, "".join(current)))
                current = []
            else:
                if current:
                    pieces.append((False, "".join(current)))
                current = [line]
            in_code = not in_code
            continue
        current.append(line)
    if current:
        pieces.append((in_code, "".join(current)))
    return pieces


def clean_prose(text: str) -> tuple[str, int]:
    """Hyphenation and whitespace, for text known to be outside code."""
    text, joined = HYPHENATED.subn(r"\1\2", text)
    text = ODD_SPACES.sub(" ", text)
    text = SPACE_RUN.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text, joined


def clean_page(markdown: str) -> tuple[str, PageStats]:
    stats = PageStats()
    out: list[str] = []
    for is_code, segment in split_fenced(normalise_unicode(markdown)):
        if is_code:
            out.append(segment)
            continue
        cleaned, joined = clean_prose(segment)
        stats.hyphens_joined += joined
        out.append(cleaned)
    text = BLANK_RUN.sub("\n\n", "".join(out)).strip()
    return text, stats


def edge_key(line: str) -> str:
    """Normalise a line so that page 12's header matches page 13's."""
    key = line.strip().lower()
    key = re.sub(r"[#*_`>|]", "", key)
    key = re.sub(r"\d+", "#", key)
    return re.sub(r"\s+", " ", key).strip()


def _prose_lines(page: str) -> list[tuple[int, str]]:
    """(line index, line) for non-empty lines outside code fences."""
    lines: list[tuple[int, str]] = []
    in_code = False
    for index, line in enumerate(page.split("\n")):
        if FENCE.match(line):
            in_code = not in_code
            continue
        if not in_code and line.strip():
            lines.append((index, line))
    return lines


def _edge_candidates(page: str) -> list[tuple[int, str]]:
    prose = _prose_lines(page)
    edges = prose[:EDGE_LINES] + prose[-EDGE_LINES:]
    seen: set[int] = set()
    out = []
    for index, line in edges:
        if index in seen or line.strip() in PLACEHOLDERS or line.lstrip().startswith("|"):
            continue
        seen.add(index)
        out.append((index, line))
    return out


def running_lines(pages: list[str]) -> set[str]:
    """Edge keys that recur on enough pages to be headers or footers."""
    if len(pages) < EDGE_MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({edge_key(line) for _, line in _edge_candidates(page)})
    threshold = max(EDGE_MIN_PAGES, EDGE_SHARE * len(pages))
    return {key for key, count in counts.items() if key and count >= threshold}


def drop_running_lines(page: str, running: set[str]) -> tuple[str, int]:
    """Remove running-header lines, but only where they sit at the edge.

    The same words in the middle of a page are content - a chapter title in a
    running header also appears once as the chapter's own heading.
    """
    if not running:
        return page, 0
    doomed = {i for i, line in _edge_candidates(page) if edge_key(line) in running}
    if not doomed:
        return page, 0
    kept = [line for i, line in enumerate(page.split("\n")) if i not in doomed]
    return "\n".join(kept), len(doomed)


def clean_document(pages: list[str]) -> tuple[list[str], list[PageStats]]:
    """Clean every page, then strip lines that run along the edges."""
    cleaned, stats = [], []
    for page in pages:
        text, page_stats = clean_page(page)
        cleaned.append(text)
        stats.append(page_stats)

    running = running_lines(cleaned)
    for i, page in enumerate(cleaned):
        text, removed = drop_running_lines(page, running)
        cleaned[i] = BLANK_RUN.sub("\n\n", text).strip()
        stats[i].edge_lines_removed = removed
    return cleaned, stats
