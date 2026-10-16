"""DOCX -> Markdown through docling.

A DOCX has no pages: its layout is decided by whichever program opens it. The
whole document therefore becomes a single record numbered 1, and the metadata
says so, rather than inventing page numbers a citation could not be checked
against.

docling lives in the optional `ingest` extra; it is imported here, not at the
top of the package, so that the rest of the pipeline works without it.
"""

from __future__ import annotations

from pathlib import Path

from nsu_rag_llm.ingest.parsers.pdf import ParsedPage


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def parse_docx(path: Path) -> list[ParsedPage]:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(path)
    return [ParsedPage(number=1, label=None, markdown=result.document.export_to_markdown())]
