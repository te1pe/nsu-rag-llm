"""Generate and refresh `data/courses.yaml`.

The file is the author's: it carries the course titles they wrote and the
`include` flags that decide what the MVP corpus is. This module regenerates
the measured parts around those decisions - page counts, text-layer verdicts,
which copies are duplicates - without touching either.

Regeneration is therefore idempotent: run it after every inventory and the
author's edits survive.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from nsu_rag_llm.ingest.inventory import FileReport, Inventory
from nsu_rag_llm.ingest.inventory_report import (
    GIANT_PAGES,
    is_usable,
    propose_mvp,
    summarise_courses,
)

HEADER = """\
# Course catalogue and MVP selection.
#
# `title` is yours: it is what a citation shows, [<title> · «document» · p. 34].
# `include` is yours too - it decides what phases 1b-4 actually index.
# Everything else is measured by `nsu-rag inventory` and refreshed on every
# run; your titles and your include flags are preserved across regeneration.
#
# A document is indexed when its course is `include: true` AND the document
# itself is `include: true`. A file is proposed for inclusion when it reads
# without OCR, is not a second copy of something already listed, is under
# {giant} pages and is in Russian; the comment after an excluded file says
# which of those it failed.
#
# Copies of a duplicate are always `include: false`: the decision belongs to
# the content, not to a path, and is carried by the primary copy. If the
# primary ever changes, your decision moves with the content.
#
# `origin: ocr-layer` marks a scan whose text layer somebody else produced by
# OCR: dense text, but with that OCR's mistakes (ADR-017).
"""


@dataclass
class Decisions:
    """What the author has already decided, read back off disk."""

    title_lines: dict[str, str]
    """Slug -> the verbatim `title:` line, so inline comments survive."""
    course_include: dict[str, bool]
    file_include: dict[tuple[str, str], bool]


def read_decisions(path: Path) -> Decisions:
    if not path.exists():
        return Decisions({}, {}, {})

    text = path.read_text(encoding="utf-8")

    # Titles are lifted as raw text: the author writes "# ПРОВЕРЬ ..." notes
    # after them, and a YAML round-trip would silently drop every comment.
    title_lines: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        course = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if course:
            current = course.group(1)
            continue
        if current and (stripped := line.strip()).startswith("title:"):
            title_lines[current] = stripped
            current = None

    course_include: dict[str, bool] = {}
    file_include: dict[tuple[str, str], bool] = {}
    loaded = yaml.safe_load(text) or {}
    for slug, body in (loaded.get("courses") or {}).items():
        if not isinstance(body, dict):
            continue
        if isinstance(body.get("include"), bool):
            course_include[slug] = body["include"]
        for entry in body.get("documents") or []:
            if isinstance(entry, dict) and isinstance(entry.get("include"), bool):
                file_include[(slug, str(entry.get("file")))] = entry["include"]
    return Decisions(title_lines, course_include, file_include)


def exclusion_reason(report: FileReport) -> str | None:
    """Why this file is not proposed for the MVP, in the author's language."""
    if report.is_duplicate:
        return f"дубликат {report.duplicate_of}"
    if report.text_layer == "none":
        return "скан, нужен OCR"
    if report.text_layer == "weak":
        return "текстовый слой рваный, надёжнее OCR"
    if report.text_layer == "unknown":
        return report.error or "не читается"
    if report.text_layer == "n/a":
        return "не PDF, обрабатывается отдельно"
    if (report.pages or 0) >= GIANT_PAGES:
        return f"{report.pages} страниц — забьёт выдачу"
    if report.language == "en":
        return "по-английски, вопросы для eval будут русские"
    return None


def _relative(report: FileReport) -> str:
    """Path inside the course directory, as written in `courses.yaml`."""
    return str(report.path.relative_to(report.course))


def _scalar(value: str) -> str:
    """Quote a value safely. JSON strings are valid YAML double-quoted ones."""
    return json.dumps(value, ensure_ascii=False)


def render(inventory: Inventory, decisions: Decisions) -> str:
    summaries = {s.course: s for s in summarise_courses(inventory)}
    proposed = {s.course for s in propose_mvp(list(summaries.values()))}

    by_course: dict[str, list[FileReport]] = {}
    by_content: dict[str, list[FileReport]] = {}
    for report in inventory.files:
        by_course.setdefault(report.course, []).append(report)
        by_content.setdefault(report.sha256, []).append(report)

    def decided(report: FileReport) -> bool | None:
        """The author's decision about this file's *content*, if any.

        Read across every copy, because the copy that was primary when the
        author decided may no longer be primary now. Copies are written as
        `false`, so a `true` anywhere can only be the author's.
        """
        votes = [
            decisions.file_include.get((copy.course, _relative(copy)))
            for copy in by_content[report.sha256]
        ]
        cast = [vote for vote in votes if vote is not None]
        return any(cast) if cast else None

    lines = [HEADER.format(giant=GIANT_PAGES), "courses:"]
    for slug in sorted(by_course):
        summary = summaries[slug]
        reports = sorted(by_course[slug], key=lambda r: str(r.path))
        semesters = sorted({r.semester for r in reports if r.semester is not None})

        title_line = decisions.title_lines.get(slug, 'title: ""   # TODO')
        include = decisions.course_include.get(slug, slug in proposed)

        lines += [
            f"  {slug}:",
            f"    {title_line}",
            f"    semesters: [{', '.join(str(s) for s in semesters)}]",
            f"    include: {str(include).lower()}",
            f"    # измерено: {summary.usable_files} пригодных файлов, "
            f"{summary.usable_pages} страниц из {summary.pages}",
            "    documents:",
        ]
        for report in reports:
            reason = exclusion_reason(report)
            if report.is_duplicate:
                file_include = False
            else:
                previous = decided(report)
                file_include = previous if previous is not None else include and is_usable(report)
            suffix = f"   # {reason}" if reason and not file_include else ""
            lines += [
                f"      - file: {_scalar(_relative(report))}",
                f"        include: {str(file_include).lower()}{suffix}",
                f"        pages: {report.pages if report.pages is not None else 'null'}",
                f"        layer: {report.text_layer}",
                f"        origin: {report.text_origin}",
                f"        lang: {report.language}",
            ]
        lines.append("")
    return "\n".join(lines)


def write(inventory: Inventory, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(inventory, read_decisions(path)), encoding="utf-8")
    return path
