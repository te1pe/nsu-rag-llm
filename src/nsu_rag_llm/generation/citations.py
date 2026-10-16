"""Citations: the model writes [Ф2], the reader sees [course · «document» · p. N].

Asking a 9B model to copy course titles, document titles and page ranges into
every citation invites typos and invented pages. It is asked only for the
label of the fragment it used - `[Ф2]`, `[Ф1, Ф3]` - which this module checks
against the fragments the model was actually given, and expands into the full
form. A label with no fragment behind it is an error, reported as such:
CLAUDE.md counts a citation of a non-existent fragment as a failure, and
phase 4 scores citation correctness from these errors.

The letter is there because a bare number is ambiguous: `[0, 1]` is a segment
in an answer about calculus or probability, three of the five MVP courses
(ADR-025).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The label the prompt gives a fragment, and the one the model copies.
LABEL = "Ф"

#: [Ф1], [Ф1, Ф3], [Ф1; Ф3], [Ф1, 3], [ф2], [Фрагмент 2].
CITATION = re.compile(
    r"\[\s*Ф(?:рагмент)?\s*\d+(?:\s*[,;]\s*(?:Ф(?:рагмент)?\s*)?\d+)*\s*\]",
    re.IGNORECASE,
)
#: An open bracket that may still become a citation while tokens arrive. A
#: bracket open for longer than this is prose, and is released as it is.
PENDING = re.compile(r"\[[^\[\]\n]{0,48}$")


@dataclass(frozen=True)
class Fragment:
    """A chunk as the model sees it: labelled, with where it comes from."""

    number: int
    chunk: dict

    @property
    def label(self) -> str:
        return f"[{LABEL}{self.number}]"

    @property
    def pages(self) -> str:
        start, end = self.chunk["page_start"], self.chunk["page_end"]
        return f"{start}" if start == end else f"{start}–{end}"

    @property
    def citation(self) -> str:
        return f"[{self.chunk['course_title']} · «{self.chunk['doc_title']}» · с. {self.pages}]"


@dataclass
class Rendered:
    text: str
    cited: list[int] = field(default_factory=list)
    """Fragment numbers the answer cites, in order of first appearance."""
    errors: list[str] = field(default_factory=list)


def render(text: str, fragments: dict[int, Fragment]) -> Rendered:
    """Replace every [Фn] with the full citation; flag labels with no fragment."""
    result = Rendered(text="")

    def expand(match: re.Match) -> str:
        parts: list[str] = []
        for number in (int(n) for n in re.findall(r"\d+", match.group(0))):
            fragment = fragments.get(number)
            if fragment is None:
                result.errors.append(f"ссылка [{LABEL}{number}] на несуществующий фрагмент")
                part = f"[нет фрагмента {LABEL}{number}]"
            else:
                if number not in result.cited:
                    result.cited.append(number)
                part = fragment.citation
            # Two chunks of one page cite identically; say it once.
            if part not in parts:
                parts.append(part)
        return " ".join(parts)

    result.text = CITATION.sub(expand, text)
    return result


class StreamingRenderer:
    """Expand citations in a stream of tokens as it arrives.

    Text is released as soon as it cannot be part of a citation; from an open
    bracket onwards it is held back until the bracket either closes - and is
    expanded if it is a citation - or has run on too long to be one. The
    held-back tail is a few tokens, so the reader sees the answer with barely
    any delay and with full citations.
    """

    def __init__(self, fragments: dict[int, Fragment]) -> None:
        self.fragments = fragments
        self.buffer = ""
        self.cited: list[int] = []
        self.errors: list[str] = []

    def feed(self, delta: str) -> str:
        self.buffer += delta
        pending = PENDING.search(self.buffer)
        cut = pending.start() if pending else len(self.buffer)
        ready, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return self._render(ready)

    def flush(self) -> str:
        ready, self.buffer = self.buffer, ""
        return self._render(ready)

    def _render(self, text: str) -> str:
        if not text:
            return ""
        rendered = render(text, self.fragments)
        for number in rendered.cited:
            if number not in self.cited:
                self.cited.append(number)
        self.errors += rendered.errors
        return rendered.text
