from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from nsu_rag_llm import jsonl
from nsu_rag_llm.ingest.chunking import Chunk, DocInfo, TokenCounter, chunk_document
from nsu_rag_llm.ingest.parse import Document


def document_title(path: Path) -> str:
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip()


def course_titles(courses_file: Path) -> dict[str, str]:
    if not courses_file.exists():
        return {}
    loaded = yaml.safe_load(courses_file.read_text(encoding="utf-8")) or {}
    return {
        slug: (body or {}).get("title") or slug
        for slug, body in (loaded.get("courses") or {}).items()
    }


@dataclass
class DocumentChunks:
    document: Document
    pages: int
    chunks: list[Chunk]


def build_chunks(
    documents: list[Document],
    parsed_dir: Path,
    titles: dict[str, str],
    count: TokenCounter,
    model: str,
    budget: int,
    overlap: int,
    on_document: Callable[[Document], None] | None = None,
) -> tuple[list[DocumentChunks], list[Path]]:
    built: list[DocumentChunks] = []
    unparsed: list[Path] = []
    for document in documents:
        if document.route is None:
            continue
        pages_file = parsed_dir / document.sha256 / "pages.jsonl"
        if not pages_file.exists():
            unparsed.append(document.source)
            continue
        pages = jsonl.read(pages_file)
        info = DocInfo(
            course=document.course,
            course_title=titles.get(document.course) or document.course,
            doc_title=document_title(document.source),
            file=str(document.source),
            sha256=document.sha256,
            semester=document.semester,
            origin=document.text_origin,
        )
        chunks = chunk_document(
            info,
            [(page["number"], page["markdown"]) for page in pages],
            count,
            model,
            budget=budget,
            overlap=overlap,
        )
        built.append(DocumentChunks(document, len(pages), chunks))
        if on_document:
            on_document(document)
    return built, unparsed
