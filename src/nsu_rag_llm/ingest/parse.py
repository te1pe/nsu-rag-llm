"""Parse the selected corpus into cleaned Markdown pages.

Per unique document (identified by the sha256 of its bytes) the stage writes a
folder `data/parsed/<sha256>/`:

    raw.jsonl     what the parser produced, page by page - the expensive part
    meta.json     source, parser key, timing; written last, so its presence
                  means the parse completed
    pages.jsonl   the cleaned pages that phase 1c chunks
    document.md   the same, as one file a person can read

Parsing is cached on (content hash, parser key): a second run parses nothing,
and an interrupted run resumes where it stopped. Cleaning is cheap and is
re-done from `raw.jsonl` on every run, so it always reflects the current rules
without re-parsing (ADR-013).
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from nsu_rag_llm import jsonl
from nsu_rag_llm.ingest.cleaning import (
    CLEANING_REVISION,
    PLACEHOLDERS,
    clean_document,
    split_fenced,
)
from nsu_rag_llm.ingest.courses import read_decisions
from nsu_rag_llm.ingest.inventory import FileReport, Inventory, is_junk, text_stats
from nsu_rag_llm.ingest.ocr import ocr_key
from nsu_rag_llm.ingest.parsers.docx import docling_available, parse_docx
from nsu_rag_llm.ingest.parsers.pdf import parse_pdf, parser_key

ROUTE_PDF = "pdf"
ROUTE_DOCX = "docx"
#: A scan, recognised by `nsu-rag ocr` (ingest/ocr.py) - never by this stage,
#: which only cleans what OCR has finished.
ROUTE_OCR = "ocr"


@dataclass
class Document:
    """One unique document chosen for this stage."""

    source: Path
    """The canonical copy, relative to the corpus root."""
    sha256: str
    course: str
    semester: int | None
    pages: int | None
    route: str | None = None
    skip_reason: str | None = None
    text_origin: str = "digital"
    """`digital`; `ocr-layer` for a scan somebody else OCR'd (see inventory);
    `ocr` for a scan this project OCR'd (phase 4.5)."""


@dataclass
class DocumentResult:
    document: Document
    pages: list[dict]
    parse_seconds: float
    from_cache: bool


@dataclass
class ParseRun:
    selected: list[Document] = field(default_factory=list)
    results: list[DocumentResult] = field(default_factory=list)
    skipped: list[Document] = field(default_factory=list)
    failed: dict[Path, str] = field(default_factory=dict)
    undecided: list[Path] = field(default_factory=list)
    """Files in an included course that courses.yaml does not mention yet."""
    seconds: float = 0.0


def route_for(report: FileReport, scope: str = "mvp") -> tuple[str | None, str | None]:
    """Which parser a file goes to, or why it is skipped."""
    if report.fmt == "pdf":
        if report.text_layer == "good":
            return ROUTE_PDF, None
        if report.needs_ocr:
            if scope == "full":
                return ROUTE_OCR, None
            return None, "нужен OCR — только в полном корпусе (CORPUS_SCOPE=full)"
        return None, report.error or "PDF не читается"
    if report.fmt == "docx":
        if docling_available():
            return ROUTE_DOCX, None
        return None, "нужен docling: uv sync --extra ingest"
    return None, f"формат .{report.fmt} не поддерживается"


def select_documents(
    inventory: Inventory,
    courses_file: Path,
    only_course: str | None = None,
    scope: str = "mvp",
) -> tuple[list[Document], list[Path]]:
    """The documents of a corpus scope, one per unique content.

    In the MVP a document qualifies when its course is included and the
    author's decision about its *content* - read across every copy, as in
    courses.yaml - is to include it. In the full corpus every unique document
    does: the include flags describe the MVP, and phase 4.5 is precisely
    everything else. The canonical copy is always the primary one.
    """
    decisions = read_decisions(courses_file)
    copies: dict[str, list[FileReport]] = {}
    for report in inventory.files:
        copies.setdefault(report.sha256, []).append(report)

    selected: list[Document] = []
    undecided: list[Path] = []
    for report in inventory.primary_files:
        if only_course and report.course != only_course:
            continue
        if scope == "mvp":
            if not decisions.course_include.get(report.course, False):
                continue
            votes = [
                decisions.file_include.get((c.course, str(c.path.relative_to(c.course))))
                for c in copies[report.sha256]
            ]
            cast = [vote for vote in votes if vote is not None]
            if not cast:
                undecided.append(report.path)
                continue
            if not any(cast):
                continue
        route, reason = route_for(report, scope)
        selected.append(
            Document(
                source=report.path,
                sha256=report.sha256,
                course=report.course,
                semester=report.semester,
                pages=report.pages,
                route=route,
                skip_reason=reason,
                text_origin="ocr" if route == ROUTE_OCR else report.text_origin,
            )
        )
    return selected, undecided


def expected_key(route: str, ocr_engine: str = "ocrmac") -> str:
    if route == ROUTE_PDF:
        return parser_key()
    if route == ROUTE_OCR:
        return ocr_key(ocr_engine)
    import docling

    return f"docling/{getattr(docling, '__version__', '?')}"


def cache_is_valid(folder: Path, key: str) -> bool:
    meta = folder / "meta.json"
    if not meta.exists() or not (folder / "raw.jsonl").exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8"))["parser_key"] == key
    except json.JSONDecodeError, KeyError:
        return False


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file, so an interrupted run leaves no half-file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def parse_into_cache(source: str, sha256: str, route: str, folder: str, key: str) -> dict:
    """Parse one file and write its raw pages. Runs in a worker process.

    Takes and returns plain values only: it crosses a process boundary.
    """
    started = time.perf_counter()
    path = Path(source)
    pages = parse_pdf(path) if route == ROUTE_PDF else parse_docx(path)
    seconds = time.perf_counter() - started

    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        target / "raw.jsonl",
        "".join(json.dumps(p.to_json(), ensure_ascii=False) + "\n" for p in pages),
    )
    meta = {
        "sha256": sha256,
        "source": source,
        "route": route,
        "parser_key": key,
        "pages": len(pages),
        "parse_seconds": round(seconds, 2),
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }
    # Last: a folder with meta.json is a folder whose parse finished.
    _write_atomic(target / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def _without_markers(markdown: str) -> str:
    for placeholder in PLACEHOLDERS:
        markdown = markdown.replace(placeholder, "")
    return markdown


def prose_characters(markdown: str) -> int:
    """Characters of actual text on a page, not counting our markers."""
    return len(_without_markers(markdown).strip())


HEADING = re.compile(r"^#{1,6} ", re.MULTILINE)


def count_headings(markdown: str) -> int:
    """Markdown headings, outside code.

    A C listing is full of lines that start with `#` - `#include`, `#define` -
    and a shell listing has `# comments`. Neither is a heading, and phase 1c,
    which splits text at headings, must not think so either.
    """
    return sum(
        len(HEADING.findall(segment)) for is_code, segment in split_fenced(markdown) if not is_code
    )


def text_characters(markdown: str) -> int:
    """Non-space characters, not counting our markers.

    Counted the same way as `source_chars`, the size of the page's own text
    layer, so that the two can be compared to spot text the parser lost.
    """
    return len("".join(_without_markers(markdown).split()))


def clean_into_cache(folder: Path, document: Document) -> list[dict]:
    """Clean the raw pages of one document and write what phase 1c reads."""
    raw = jsonl.read(folder / "raw.jsonl")
    cleaned, stats = clean_document([page["markdown"] for page in raw])

    records = []
    for page, text, page_stats in zip(raw, cleaned, stats, strict=True):
        junk_ratio, language = text_stats(text)
        records.append(
            {
                "number": page["number"],
                "label": page["label"],
                "markdown": text,
                "chars": prose_characters(text),
                "text_chars": text_characters(text),
                "source_chars": page.get("source_chars", 0),
                "junk_chars": sum(1 for char in text if is_junk(char)),
                "junk_ratio": round(junk_ratio, 4),
                "lang": language,
                # Phase 2 breaks retrieval metrics down by where text came from.
                "origin": document.text_origin,
                "headings": count_headings(text),
                "formulas": page["formulas"],
                "formula_prose": page.get("formula_prose", 0),
                "pictures": page["pictures"],
                "code_blocks": page["code_blocks"],
                "running_boxes": page["running_boxes"],
                "hyphens_joined": page_stats.hyphens_joined,
                "edge_lines_removed": page_stats.edge_lines_removed,
            }
            # Only on pages the OCR engine failed on (ingest/ocr.py).
            | ({"ocr_error": page["ocr_error"]} if page.get("ocr_error") else {})
        )

    _write_atomic(
        folder / "pages.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
    )
    header = [
        f"<!-- source: {document.source} | sha256: {document.sha256[:12]} "
        f"| cleaning r{CLEANING_REVISION} -->",
        "",
    ]
    body = [f"<!-- page {r['number']} -->\n\n{r['markdown']}\n" for r in records]
    _write_atomic(folder / "document.md", "\n".join(header + body))
    return records


def run(
    inventory: Inventory,
    root: Path,
    courses_file: Path,
    parsed_dir: Path,
    workers: int = 4,
    force: bool = False,
    only_course: str | None = None,
    on_parsed: Callable[[Document], None] | None = None,
    scope: str = "mvp",
    ocr_engine: str = "ocrmac",
) -> ParseRun:
    """Select, parse what is not cached, clean everything selected."""
    started = time.perf_counter()
    result = ParseRun()
    documents, result.undecided = select_documents(inventory, courses_file, only_course, scope)
    result.selected = documents
    for d in documents:
        # Scans are cleaned here once `nsu-rag ocr` has recognised them.
        if d.route == ROUTE_OCR and not cache_is_valid(
            parsed_dir / d.sha256, expected_key(ROUTE_OCR, ocr_engine)
        ):
            d.route, d.skip_reason = None, "OCR ещё не выполнен — nsu-rag ocr"
    result.skipped = [d for d in documents if d.route is None]
    runnable = [d for d in documents if d.route is not None]

    keys = {d.sha256: expected_key(d.route, ocr_engine) for d in runnable}
    todo = [
        d
        for d in runnable
        if d.route != ROUTE_OCR
        and (force or not cache_is_valid(parsed_dir / d.sha256, keys[d.sha256]))
    ]
    fresh = {d.sha256 for d in todo}
    # Longest first: the biggest book should not be the one left running alone
    # at the end while every other worker sits idle.
    todo.sort(key=lambda d: -(d.pages or 0))

    def task(d: Document) -> tuple:
        return (str(root / d.source), d.sha256, d.route, str(parsed_dir / d.sha256), keys[d.sha256])

    if workers <= 1:
        for d in todo:
            try:
                parse_into_cache(*task(d))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                result.failed[d.source] = f"{type(exc).__name__}: {exc}"
            if on_parsed:
                on_parsed(d)
    elif todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(parse_into_cache, *task(d)): d for d in todo}
            for future in as_completed(futures):
                d = futures[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    result.failed[d.source] = f"{type(exc).__name__}: {exc}"
                if on_parsed:
                    on_parsed(d)

    for d in runnable:
        if d.source in result.failed:
            continue
        folder = parsed_dir / d.sha256
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        result.results.append(
            DocumentResult(
                document=d,
                pages=clean_into_cache(folder, d),
                parse_seconds=meta["parse_seconds"],
                from_cache=d.sha256 not in fresh,
            )
        )
    result.seconds = time.perf_counter() - started
    return result


def median_chars(pages: list[dict]) -> float:
    values = [p["chars"] for p in pages]
    return statistics.median(values) if values else 0.0
