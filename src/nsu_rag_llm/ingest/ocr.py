"""Scans -> Markdown through docling's OCR, one page at a time, resumable.

A scan has no text layer worth parsing: every page is a picture of text. OCR
is the most expensive step in the project - hours for the full corpus
(ADR-009, ADR-030) - so it is a command of its own, meant to run in the
background, and it survives being stopped: every recognised page is appended
to `raw.partial.jsonl` the moment it is done, and a restart skips the pages
already there. When a document's last page is in, its pages are written as
`raw.jsonl` with a `meta.json`, exactly like a parsed PDF's, and
`nsu-rag parse` cleans and reports them with the rest.

The whole page is recognised (`OcrMode.FULL_PAGE`): these documents either
have no text layer or one somebody else produced badly enough that the
inventory rejected it, and mixing the two would be worse than either.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from nsu_rag_llm import jsonl
from nsu_rag_llm.ingest.cleaning import FORMULA, PICTURE
from nsu_rag_llm.ingest.parsers.pdf import ParsedPage

Engine = Literal["ocrmac", "rapidocr"]

#: Bump when anything that decides the OCR output changes.
OCR_REVISION = 1

#: Languages per engine. Apple's Vision reads Russian and English in one pass;
#: RapidOCR takes exactly one language per pass (ADR-009), so with it an
#: English scan comes out as Cyrillic noise.
LANGUAGES: dict[str, list[str]] = {"ocrmac": ["ru-RU", "en-US"], "rapidocr": ["iso:ru"]}

PARTIAL = "raw.partial.jsonl"

OnPage = Callable[[int, float], None]
"""Called after each page with its number and the seconds it took."""


def ocr_key(engine: str) -> str:
    """Identifies everything that decides the OCR output, like parser_key().

    Without docling installed nothing can have been recognised by it, and no
    cached page may be trusted as its work: the key says so, and never matches.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        docling = version("docling")
    except PackageNotFoundError:
        docling = "not-installed"
    return f"docling-ocr/{docling}/{engine}/r{OCR_REVISION}"


def make_converter(engine: Engine):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        OcrMacOptions,
        OcrMode,
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True
    # Table structure is a model of its own and a bill of its own; tables in
    # these scans are rare, and their text is recognised anyway.
    options.do_table_structure = False
    ocr = OcrMacOptions if engine == "ocrmac" else RapidOcrOptions
    options.ocr_options = ocr(lang=LANGUAGES[engine], mode=OcrMode.FULL_PAGE)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


#: docling's markers for what it saw but could not read, and ours for the same
#: (the PDF parser's, ADR-012): the prompt and the parse report know only ours.
DOCLING_FORMULA = "<!-- formula-not-decoded -->"
DOCLING_PICTURE = "<!-- image -->"


def normalise(markdown: str) -> tuple[str, int, int]:
    """docling's markers replaced with ours; how many formulas and pictures."""
    formulas, pictures = markdown.count(DOCLING_FORMULA), markdown.count(DOCLING_PICTURE)
    markdown = markdown.replace(DOCLING_FORMULA, FORMULA).replace(DOCLING_PICTURE, PICTURE)
    return markdown, formulas, pictures


def recognise_page(converter, path: Path, number: int) -> ParsedPage:
    """One page, 1-based, as Markdown."""
    result = converter.convert(path, page_range=(number, number))
    markdown, formulas, pictures = normalise(result.document.export_to_markdown())
    return ParsedPage(
        number=number, label=None, markdown=markdown, formulas=formulas, pictures=pictures
    )


def done_pages(folder: Path) -> dict[int, dict]:
    """Pages already recognised in an unfinished run, by number."""
    partial = folder / PARTIAL
    if not partial.exists():
        return {}
    pages = {}
    for line in jsonl.lines(partial.read_text(encoding="utf-8")):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short when the run was killed mid-write
        pages[record["number"]] = record
    return pages


def ocr_document(
    path: Path,
    folder: Path,
    pages: int,
    key: str,
    converter,
    sha256: str,
    source: str,
    on_page: OnPage | None = None,
    recognise: Callable[..., ParsedPage] = recognise_page,
) -> dict:
    """Recognise every page not yet done; finish the document when all are."""
    folder.mkdir(parents=True, exist_ok=True)
    done = done_pages(folder)
    started = time.perf_counter()
    with (folder / PARTIAL).open("a", encoding="utf-8") as out:
        for number in range(1, pages + 1):
            if number in done:
                continue
            page_started = time.perf_counter()
            try:
                page = recognise(converter, path, number)
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop hours of work
                page = ParsedPage(number=number, label=None, markdown="")
                page_error = f"{type(exc).__name__}: {exc}"
            else:
                page_error = None
            record = page.to_json() | ({"ocr_error": page_error} if page_error else {})
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            done[number] = record
            if on_page:
                on_page(number, time.perf_counter() - page_started)

    records = [done[n] for n in sorted(done)]
    _write_atomic(
        folder / "raw.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    meta = {
        "sha256": sha256,
        "source": source,
        "route": "ocr",
        "parser_key": key,
        "pages": len(records),
        "failed_pages": sum("ocr_error" in r for r in records),
        "parse_seconds": round(time.perf_counter() - started, 2),
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }
    # As for any parse: meta.json last, so its presence means the work is done.
    _write_atomic(folder / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    (folder / PARTIAL).unlink(missing_ok=True)
    return meta


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
