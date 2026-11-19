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
