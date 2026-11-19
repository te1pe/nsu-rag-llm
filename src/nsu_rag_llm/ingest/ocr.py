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

# поднять при любом изменении OCR
OCR_REVISION = 1

LANGUAGES: dict[str, list[str]] = {"ocrmac": ["ru-RU", "en-US"], "rapidocr": ["iso:ru"]}

PARTIAL = "raw.partial.jsonl"

OnPage = Callable[[int, float], None]
"""Called after each page with its number and the seconds it took."""


def ocr_key(engine: str) -> str:
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
    options.do_table_structure = False
    ocr = OcrMacOptions if engine == "ocrmac" else RapidOcrOptions
    options.ocr_options = ocr(lang=LANGUAGES[engine], mode=OcrMode.FULL_PAGE)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


DOCLING_FORMULA = "<!-- formula-not-decoded -->"
DOCLING_PICTURE = "<!-- image -->"


def normalise(markdown: str) -> tuple[str, int, int]:
    formulas, pictures = markdown.count(DOCLING_FORMULA), markdown.count(DOCLING_PICTURE)
    markdown = markdown.replace(DOCLING_FORMULA, FORMULA).replace(DOCLING_PICTURE, PICTURE)
    return markdown, formulas, pictures


def recognise_page(converter, path: Path, number: int) -> ParsedPage:
    result = converter.convert(path, page_range=(number, number))
    markdown, formulas, pictures = normalise(result.document.export_to_markdown())
    return ParsedPage(
        number=number, label=None, markdown=markdown, formulas=formulas, pictures=pictures
    )


def done_pages(folder: Path) -> dict[int, dict]:
    partial = folder / PARTIAL
    if not partial.exists():
        return {}
    pages = {}
    for line in jsonl.lines(partial.read_text(encoding="utf-8")):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
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
    _write_atomic(folder / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    (folder / PARTIAL).unlink(missing_ok=True)
    return meta


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
