"""OCR of scans: resumable page by page, and never stopped by one bad page.

The recogniser is a fake - docling and its models are not needed here.
"""

from __future__ import annotations

import json
from pathlib import Path

from nsu_rag_llm.ingest.ocr import PARTIAL, done_pages, normalise, ocr_document
from nsu_rag_llm.ingest.parsers.pdf import ParsedPage


class Recogniser:
    """Recognises page n as "Страница n", and records which pages it saw."""

    def __init__(self, failing: set[int] = frozenset()) -> None:
        self.seen: list[int] = []
        self.failing = failing

    def __call__(self, converter, path: Path, number: int) -> ParsedPage:
        self.seen.append(number)
        if number in self.failing:
            raise RuntimeError("the engine choked on this page")
        return ParsedPage(number=number, label=None, markdown=f"Страница {number}")


def ocr(folder: Path, recogniser: Recogniser, pages: int = 3, on_page=None) -> dict:
    return ocr_document(
        Path("scan.pdf"),
        folder,
        pages,
        key="docling-ocr/test",
        converter=None,
        sha256="abc",
        source="os/sem4/scan.pdf",
        on_page=on_page,
        recognise=recogniser,
    )


def raw_pages(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]


def test_every_page_is_recognised_and_the_document_finished(tmp_path: Path) -> None:
    progress = []

    meta = ocr(tmp_path, Recogniser(), on_page=lambda n, s: progress.append(n))

    assert [p["markdown"] for p in raw_pages(tmp_path)] == [
        "Страница 1",
        "Страница 2",
        "Страница 3",
    ]
    assert meta["route"] == "ocr" and meta["parser_key"] == "docling-ocr/test"
    assert (tmp_path / "meta.json").exists()
    assert not (tmp_path / PARTIAL).exists()
    assert progress == [1, 2, 3]


def test_a_restart_skips_the_pages_already_recognised(tmp_path: Path) -> None:
    """Killed during page 3: page 3's line was cut short, pages 1-2 are safe."""
    first = ParsedPage(number=1, label=None, markdown="Страница 1").to_json()
    second = ParsedPage(number=2, label=None, markdown="Страница 2").to_json()
    (tmp_path / PARTIAL).write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n" + '{"number": 3, "mark',
        encoding="utf-8",
    )
    recogniser = Recogniser()

    ocr(tmp_path, recogniser, pages=4)

    assert recogniser.seen == [3, 4]
    assert [p["number"] for p in raw_pages(tmp_path)] == [1, 2, 3, 4]


def test_a_page_the_engine_fails_on_is_recorded_and_the_rest_go_on(tmp_path: Path) -> None:
    meta = ocr(tmp_path, Recogniser(failing={2}))

    pages = raw_pages(tmp_path)
    assert meta["failed_pages"] == 1
    assert pages[1]["markdown"] == "" and "choked" in pages[1]["ocr_error"]
    assert pages[2]["markdown"] == "Страница 3"


def test_progress_survives_between_runs_until_the_document_is_done(tmp_path: Path) -> None:
    (tmp_path / PARTIAL).write_text(
        json.dumps(ParsedPage(number=1, label=None, markdown="x").to_json()) + "\n",
        encoding="utf-8",
    )

    assert set(done_pages(tmp_path)) == {1}
    assert done_pages(tmp_path / "nothing-yet") == {}


def test_doclings_markers_become_the_parsers_own() -> None:
    """The prompt tells the model what [формула] means; docling's comment it
    would not recognise, and the parse report would not count."""
    text = "Метрика задана формулой\n\n<!-- formula-not-decoded -->\n\n<!-- image -->"

    markdown, formulas, pictures = normalise(text)

    assert markdown == "Метрика задана формулой\n\n[формула]\n\n[рисунок]"
    assert (formulas, pictures) == (1, 1)
