"""Inventory heuristics, exercised on synthetic documents.

Every fixture here is generated in the test. The real corpus is third-party
material and never appears in the test suite.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from nsu_rag_llm.ingest.inventory import (
    FileReport,
    Inventory,
    ManifestRow,
    course_and_semester,
    inspect_file,
    mark_duplicates,
    reconcile,
    sample_indices,
    text_stats,
)

RUSSIAN = (
    "Процесс — это программа в стадии выполнения, со своим адресным "
    "пространством и таблицей открытых файлов. Поток — единица планирования "
    "внутри процесса; потоки одного процесса делят память. "
)
ENGLISH = (
    "A process is a program in execution, with its own address space and file "
    "descriptor table. A thread is the unit of scheduling inside a process; "
    "threads of one process share memory. "
)


def make_pdf(path: Path, pages: int, line: str = "", repeats: int = 12) -> Path:
    """Write a PDF whose every page carries `line` repeated, or nothing.

    The text is Latin on purpose: PDF base-14 fonts cannot encode Cyrillic,
    and embedding a font that can would mean shipping a font file with the
    tests. Language detection is covered directly against `text_stats`, and
    end to end through the DOCX fixture, which needs no font at all.
    """
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if line:
            page.insert_textbox(
                pymupdf.Rect(36, 36, 559, 806),
                (line * repeats),
                fontsize=9,
                fontname="helv",
            )
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    (root / "os" / "sem4").mkdir(parents=True)
    (root / "matan" / "sem1").mkdir(parents=True)
    return root


def test_language_is_detected_from_the_dominant_script() -> None:
    assert text_stats(RUSSIAN)[1] == "ru"
    assert text_stats(ENGLISH)[1] == "en"
    assert text_stats(RUSSIAN + ENGLISH)[1] == "mixed"


def test_too_little_text_is_not_guessed_at() -> None:
    assert text_stats("абв")[1] == "unknown"
    assert text_stats("")[1] == "unknown"


def test_junk_ratio_counts_broken_glyphs_not_punctuation() -> None:
    clean, _ = text_stats(RUSSIAN + "«ёлки-палки» — 42% (см. §3).")
    assert clean == 0.0

    # A damaged ToUnicode map extracts as private-use code points, and a
    # mis-declared encoding produces U+FFFD.
    broken, _ = text_stats("�\x00" + RUSSIAN)
    assert broken > 0.0


def test_page_sampling_never_starves_a_short_document() -> None:
    assert sample_indices(0) == []
    assert sample_indices(3) == [0, 1, 2]
    assert len(sample_indices(5000)) == 12
    # Front matter is skipped only when there is a body left to sample.
    assert sample_indices(100)[0] == 2


def test_a_text_pdf_is_recognised_as_such(corpus: Path) -> None:
    path = make_pdf(corpus / "os" / "sem4" / "lectures.pdf", pages=8, line=ENGLISH)

    report = inspect_file(path, corpus)

    assert report.pages == 8
    assert report.text_layer == "good"
    assert report.needs_ocr is False
    assert report.course == "os"
    assert report.semester == 4


def make_docx(path: Path, text: str) -> Path:
    """Write the one archive member the inventory reads out of a DOCX."""
    import zipfile

    body = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in [text] * 6)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>',
        )
    return path


def test_a_russian_docx_is_read_and_its_language_recognised(corpus: Path) -> None:
    path = make_docx(corpus / "os" / "sem4" / "notes.docx", RUSSIAN)

    report = inspect_file(path, corpus)

    assert report.error is None
    assert report.language == "ru"
    assert report.fmt == "docx"
    # Page count is meaningless for a flow-layout format.
    assert report.pages is None
    assert report.text_layer == "n/a"


def test_a_docx_that_is_not_a_zip_is_reported_not_raised(corpus: Path) -> None:
    path = corpus / "os" / "sem4" / "broken.docx"
    path.write_bytes(b"this is not a zip archive")

    report = inspect_file(path, corpus)

    assert report.error is not None


def test_a_pdf_without_a_text_layer_is_routed_to_ocr(corpus: Path) -> None:
    path = make_pdf(corpus / "os" / "sem4" / "scan.pdf", pages=5)

    report = inspect_file(path, corpus)

    assert report.text_layer == "none"
    assert report.needs_ocr is True


def test_a_thin_text_layer_is_called_weak_not_good(corpus: Path) -> None:
    # One short line per page: text is present, but nowhere near a full page.
    path = make_pdf(corpus / "os" / "sem4" / "thin.pdf", pages=6, line="Chapter 1. ", repeats=10)

    report = inspect_file(path, corpus)

    assert report.text_layer == "weak"
    assert report.needs_ocr is True


def test_an_unreadable_file_is_reported_not_raised(corpus: Path) -> None:
    path = corpus / "os" / "sem4" / "truncated.pdf"
    path.write_bytes(b"%PDF-1.7\nnot actually a pdf")

    report = inspect_file(path, corpus)

    assert report.error is not None
    assert report.text_layer == "unknown"


def test_identical_files_collapse_to_one_primary(corpus: Path) -> None:
    first = make_pdf(corpus / "os" / "sem4" / "book.pdf", pages=4, line=ENGLISH)
    second = corpus / "matan" / "sem1" / "book.pdf"
    second.write_bytes(first.read_bytes())

    reports = [inspect_file(first, corpus), inspect_file(second, corpus)]
    mark_duplicates(reports)

    primaries = [r for r in reports if not r.is_duplicate]
    assert len(primaries) == 1
    assert all(r.duplicate_of == primaries[0].path for r in reports if r.is_duplicate)


def test_different_files_are_not_called_duplicates(corpus: Path) -> None:
    a = make_pdf(corpus / "os" / "sem4" / "a.pdf", pages=4, line=ENGLISH)
    b = make_pdf(corpus / "os" / "sem4" / "b.pdf", pages=9, line=ENGLISH)

    reports = [inspect_file(a, corpus), inspect_file(b, corpus)]
    mark_duplicates(reports)

    assert [r.is_duplicate for r in reports] == [False, False]


def test_course_and_semester_come_from_the_path() -> None:
    assert course_and_semester(Path("os/sem4/x.pdf")) == ("os", 4)
    assert course_and_semester(Path("os/x.pdf")) == ("os", None)


def _report(path: str) -> FileReport:
    course, semester = course_and_semester(Path(path))
    return FileReport(
        path=Path(path),
        course=course,
        semester=semester,
        size_bytes=1,
        sha256="deadbeef",
        fmt="pdf",
        pages=10,
    )


def test_a_book_listed_in_two_semesters_claims_both_copies() -> None:
    """Regression: both rows used to be handed the same copy.

    The other copy then looked like a book that was never in the manifest,
    which is the signal used to tell nsumedia files from loose ones.
    """
    inventory = Inventory(
        files=[_report("os/sem4/tanenbaum.pdf"), _report("os/sem5/tanenbaum.pdf")]
    )
    rows = [
        ManifestRow("2", "4", "ОСи", "tanenbaum.pdf", "уже есть"),
        ManifestRow("3", "5", "ОСи", "tanenbaum.pdf", "уже есть"),
    ]

    reconcile(inventory, rows)

    assert inventory.unlisted == []
    assert inventory.manifest_missing == []
    assert [f.manifest_semester for f in inventory.files] == [4, 5]


def test_a_row_with_no_file_is_reported_missing() -> None:
    inventory = Inventory(files=[_report("os/sem4/present.pdf")])
    rows = [
        ManifestRow("2", "4", "ОСи", "present.pdf", "уже есть"),
        ManifestRow("2", "4", "ОСи", "gone.pdf", "ОШИБКА: HTTP Error 404: Not Found"),
    ]

    reconcile(inventory, rows)

    assert [r.filename for r in inventory.manifest_missing] == ["gone.pdf"]
    assert inventory.unlisted == []


def test_a_file_no_row_mentions_is_reported_unlisted() -> None:
    inventory = Inventory(files=[_report("matan/sem1/loose-book.pdf")])

    reconcile(inventory, [])

    assert [f.path.name for f in inventory.unlisted] == ["loose-book.pdf"]


def test_measured_costs_survive_a_run_without_the_benchmark(tmp_path: Path) -> None:
    """A plain inventory run must not throw away a measurement that cost minutes."""
    from nsu_rag_llm.ingest.inventory_report import CostEstimate

    sidecar = tmp_path / "costs.json"
    CostEstimate(
        ocr_seconds_per_page=2.0,
        ocr_pages=20_040,
        embed_chunks_per_second=22.0,
        embed_chunks_total=125_399,
        notes=["замерено на 4 страницах"],
    ).save(sidecar)

    loaded = CostEstimate.load(sidecar)

    assert loaded is not None
    assert loaded.ocr_seconds_per_page == 2.0
    assert round(loaded.ocr_hours) == 11
    assert loaded.notes == ["замерено на 4 страницах"]


def test_a_missing_or_corrupt_cost_file_is_not_fatal(tmp_path: Path) -> None:
    from nsu_rag_llm.ingest.inventory_report import CostEstimate

    assert CostEstimate.load(tmp_path / "nothing.json") is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert CostEstimate.load(broken) is None


def test_the_manifest_copy_of_a_duplicate_is_the_primary() -> None:
    """CLAUDE.md: the nsumedia copy wins over a book that lay loose in raw/.

    The copies are byte-identical, so only provenance can decide - and a loose
    book sorts first by path, which is exactly the wrong answer.
    """
    loose = _report("matan/sem1/demidovich.pdf")
    listed = _report("matan/sem2/demidovich.pdf")
    listed.manifest_subject = "Матанализ"

    mark_duplicates([loose, listed])

    assert listed.is_duplicate is False
    assert loose.duplicate_of == listed.path


def test_without_a_manifest_copy_the_earliest_semester_wins() -> None:
    later = _report("os/sem5/tanenbaum.pdf")
    earlier = _report("os/sem4/tanenbaum.pdf")

    mark_duplicates([later, earlier])

    assert earlier.is_duplicate is False
    assert later.duplicate_of == earlier.path


def make_scan_with_text(path: Path, pages: int, cover: float) -> Path:
    """Pages carrying a picture over `cover` of their area, with text on top -
    what a scanned book looks like after someone else has run OCR on it."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        width, height = page.rect.width * cover**0.5, page.rect.height * cover**0.5
        picture = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 56), 0)
        picture.clear_with(200)
        page.insert_image(pymupdf.Rect(0, 0, width, height), pixmap=picture)
        page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), ENGLISH * 12, fontsize=9)
    doc.save(path)
    doc.close()
    return path


def test_a_scan_under_someone_elses_ocr_layer_is_recognised(corpus: Path) -> None:
    path = make_scan_with_text(corpus / "os" / "sem4" / "scanned.pdf", pages=6, cover=1.0)

    report = inspect_file(path, corpus)

    # Dense text, so the layer is "good" - which is exactly why it needs a second label.
    assert report.text_layer == "good"
    assert report.text_origin == "ocr-layer"


def test_a_figure_does_not_make_a_page_a_scan(corpus: Path) -> None:
    path = make_scan_with_text(corpus / "os" / "sem4" / "figure.pdf", pages=6, cover=0.3)

    report = inspect_file(path, corpus)

    assert report.text_origin == "digital"


def test_born_digital_text_is_digital(corpus: Path) -> None:
    path = make_pdf(corpus / "os" / "sem4" / "digital.pdf", pages=6, line=ENGLISH)

    assert inspect_file(path, corpus).text_origin == "digital"
