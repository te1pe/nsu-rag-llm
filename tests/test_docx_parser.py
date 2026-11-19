from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("docling")
docx = pytest.importorskip("docx")

from nsu_rag_llm.ingest.parsers.docx import parse_docx  # noqa: E402


def test_a_docx_becomes_one_unpaged_record(tmp_path: Path) -> None:
    document = docx.Document()
    document.add_heading("Lab 3. Signals", level=1)
    document.add_paragraph("Install a handler with sigaction and block the signal meanwhile.")
    path = tmp_path / "lab.docx"
    document.save(path)

    pages = parse_docx(path)

    assert len(pages) == 1
    assert pages[0].number == 1
    assert "# Lab 3. Signals" in pages[0].markdown
    assert "sigaction" in pages[0].markdown
