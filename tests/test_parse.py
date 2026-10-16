"""Selection, caching and failure handling of the parse stage.

The parser itself is replaced by a fake here - its own behaviour is covered in
test_pdf_parser - so these tests are about the orchestration: what gets
parsed, when the cache is trusted, and what happens when a file breaks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from nsu_rag_llm.ingest import parse as stage
from nsu_rag_llm.ingest.inventory import take_inventory
from nsu_rag_llm.ingest.parsers.pdf import ParsedPage

TEXT = "A process is a program in execution; the kernel schedules its threads. " * 12


def make_pdf(path: Path, pages: int = 3, text: str = TEXT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), f"{n} {text}", fontsize=9)
    doc.save(path)
    doc.close()


COURSES = """\
courses:
  os:
    title: "Operating systems"
    include: true
    documents:
      - file: "sem4/kernel.pdf"
        include: true
      - file: "sem4/threads.pdf"
        include: true
      - file: "sem4/excluded.pdf"
        include: false
      - file: "sem4/scan.pdf"
        include: true
  matan:
    title: "Calculus"
    include: false
    documents:
      - file: "sem1/limits.pdf"
        include: true
"""


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    root = tmp_path / "raw"
    make_pdf(root / "os/sem4/kernel.pdf")
    make_pdf(root / "os/sem4/threads.pdf", text=TEXT.replace("process", "thread"))
    make_pdf(root / "os/sem4/excluded.pdf", text=TEXT.replace("kernel", "shell"))
    make_pdf(root / "os/sem4/scan.pdf", text="")  # no text layer at all
    make_pdf(root / "matan/sem1/limits.pdf", text=TEXT.replace("threads", "limits"))
    courses = tmp_path / "courses.yaml"
    courses.write_text(COURSES, encoding="utf-8")
    return {"root": root, "courses": courses, "parsed": tmp_path / "parsed"}


@pytest.fixture
def fake_parser(monkeypatch):
    """Replace the PDF parser with one that records its calls."""
    calls: list[str] = []

    def parse_pdf(path: Path) -> list[ParsedPage]:
        calls.append(path.name)
        return [
            ParsedPage(
                number=1, label=None, markdown=f"# {path.stem}\n\nFirst page of {path.stem}."
            ),
            ParsedPage(number=2, label="ii", markdown="Second page.", formulas=1),
        ]

    monkeypatch.setattr(stage, "parse_pdf", parse_pdf)
    return calls


def run(corpus: dict, **kwargs) -> stage.ParseRun:
    inventory = take_inventory(corpus["root"])
    return stage.run(
        inventory,
        root=corpus["root"],
        courses_file=corpus["courses"],
        parsed_dir=corpus["parsed"],
        workers=1,
        **kwargs,
    )


def test_only_included_documents_of_included_courses_are_parsed(corpus, fake_parser) -> None:
    result = run(corpus)

    assert sorted(fake_parser) == ["kernel.pdf", "threads.pdf"]
    assert {r.document.source.name for r in result.results} == {"kernel.pdf", "threads.pdf"}


def test_a_scan_is_skipped_with_a_reason_not_parsed(corpus, fake_parser) -> None:
    result = run(corpus)

    assert [d.source.name for d in result.skipped] == ["scan.pdf"]
    assert "OCR" in result.skipped[0].skip_reason
    assert "scan.pdf" not in fake_parser


def test_a_file_courses_yaml_does_not_mention_is_reported(corpus, fake_parser) -> None:
    make_pdf(corpus["root"] / "os/sem4/new-arrival.pdf", text=TEXT.replace("the", "a"))

    result = run(corpus)

    assert [p.name for p in result.undecided] == ["new-arrival.pdf"]
    assert "new-arrival.pdf" not in fake_parser


def test_outputs_are_written_per_content_hash(corpus, fake_parser) -> None:
    result = run(corpus)
    kernel = next(r for r in result.results if r.document.source.name == "kernel.pdf")
    folder = corpus["parsed"] / kernel.document.sha256

    assert {p.name for p in folder.iterdir()} == {
        "raw.jsonl",
        "meta.json",
        "pages.jsonl",
        "document.md",
    }
    pages = [json.loads(line) for line in (folder / "pages.jsonl").read_text().splitlines()]
    assert [p["number"] for p in pages] == [1, 2]
    assert pages[1]["label"] == "ii"
    assert pages[1]["formulas"] == 1
    assert "<!-- page 2 -->" in (folder / "document.md").read_text(encoding="utf-8")


def test_a_second_run_parses_nothing(corpus, fake_parser) -> None:
    run(corpus)
    fake_parser.clear()

    result = run(corpus)

    assert fake_parser == []
    assert all(r.from_cache for r in result.results)


def test_force_parses_again(corpus, fake_parser) -> None:
    run(corpus)
    fake_parser.clear()

    run(corpus, force=True)

    assert sorted(fake_parser) == ["kernel.pdf", "threads.pdf"]


def test_a_new_parser_version_invalidates_the_cache(corpus, fake_parser, monkeypatch) -> None:
    run(corpus)
    fake_parser.clear()
    monkeypatch.setattr(stage, "parser_key", lambda: "pymupdf4llm-layout/9.9.9/r99")

    run(corpus)

    assert sorted(fake_parser) == ["kernel.pdf", "threads.pdf"]


def test_an_interrupted_parse_is_not_mistaken_for_a_finished_one(corpus, fake_parser) -> None:
    """meta.json is written last; a folder without it is parsed again."""
    first = run(corpus)
    folder = corpus["parsed"] / first.results[0].document.sha256
    (folder / "meta.json").unlink()
    fake_parser.clear()

    run(corpus)

    assert fake_parser == [first.results[0].document.source.name]


def test_one_broken_file_does_not_stop_the_others(corpus, monkeypatch) -> None:
    def parse_pdf(path: Path) -> list[ParsedPage]:
        if path.name == "kernel.pdf":
            raise RuntimeError("damaged xref table")
        return [ParsedPage(number=1, label=None, markdown="Fine.")]

    monkeypatch.setattr(stage, "parse_pdf", parse_pdf)

    result = run(corpus)

    assert [p.name for p in result.failed] == ["kernel.pdf"]
    assert "damaged xref table" in next(iter(result.failed.values()))
    assert [r.document.source.name for r in result.results] == ["threads.pdf"]


def test_cleaning_is_redone_from_the_cache_without_parsing(corpus, fake_parser) -> None:
    """Cleaning rules change more often than the parser; they must not cost a re-parse."""
    first = run(corpus)
    folder = corpus["parsed"] / first.results[0].document.sha256
    (folder / "pages.jsonl").unlink()
    fake_parser.clear()

    run(corpus)

    assert fake_parser == []
    assert (folder / "pages.jsonl").exists()


def test_only_course_limits_the_run(corpus, fake_parser) -> None:
    corpus["courses"].write_text(
        COURSES.replace("include: false\n    documents", "include: true\n    documents")
    )

    run(corpus, only_course="matan")

    assert fake_parser == ["limits.pdf"]


def test_preprocessor_lines_in_a_listing_are_not_headings() -> None:
    """`#include` starts with a hash; a chunker splitting at headings would cut
    the listing apart if it believed that made it a heading."""
    page = (
        "# Signals\n\nText.\n\n```\n#include <signal.h>\n# not a heading either\n```\n\n"
        "## Handlers\n#hashtag is not a heading"
    )

    assert stage.count_headings(page) == 2


# --- the full corpus (phase 4.5) ------------------------------------------------


def test_the_full_corpus_takes_every_unique_document_whatever_the_flags(
    corpus, fake_parser
) -> None:
    """The include flags describe the MVP; the full corpus is everything else too."""
    result = run(corpus, scope="full")

    parsed = {r.document.source.name for r in result.results}
    assert parsed == {"kernel.pdf", "threads.pdf", "excluded.pdf", "limits.pdf"}
    assert result.undecided == []


def test_a_scan_waits_for_ocr_in_the_full_corpus(corpus, fake_parser) -> None:
    result = run(corpus, scope="full")

    (scan,) = [d for d in result.skipped if d.source.name == "scan.pdf"]
    assert "nsu-rag ocr" in scan.skip_reason
    assert "scan.pdf" not in fake_parser


def test_a_recognised_scan_is_cleaned_and_marked_as_our_ocr(corpus, fake_parser) -> None:
    documents, _ = stage.select_documents(
        take_inventory(corpus["root"]), corpus["courses"], scope="full"
    )
    (scan,) = [d for d in documents if d.source.name == "scan.pdf"]
    folder = corpus["parsed"] / scan.sha256
    folder.mkdir(parents=True)
    page = ParsedPage(number=1, label=None, markdown="# Распознанный заголовок\n\nТекст.")
    (folder / "raw.jsonl").write_text(json.dumps(page.to_json()) + "\n", encoding="utf-8")
    meta = {"parser_key": stage.expected_key(stage.ROUTE_OCR, "ocrmac"), "parse_seconds": 1.0}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    result = run(corpus, scope="full", ocr_engine="ocrmac")

    (cleaned,) = [r for r in result.results if r.document.source.name == "scan.pdf"]
    assert cleaned.document.text_origin == "ocr"
    assert cleaned.pages[0]["origin"] == "ocr"
    assert "scan.pdf" not in fake_parser  # parse never runs OCR itself


def test_a_scan_recognised_by_another_engine_is_not_trusted(corpus, fake_parser) -> None:
    documents, _ = stage.select_documents(
        take_inventory(corpus["root"]), corpus["courses"], scope="full"
    )
    (scan,) = [d for d in documents if d.source.name == "scan.pdf"]
    folder = corpus["parsed"] / scan.sha256
    folder.mkdir(parents=True)
    (folder / "raw.jsonl").write_text("", encoding="utf-8")
    meta = {"parser_key": stage.expected_key(stage.ROUTE_OCR, "rapidocr"), "parse_seconds": 1.0}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    result = run(corpus, scope="full", ocr_engine="ocrmac")

    assert "scan.pdf" in {d.source.name for d in result.skipped}


def test_parsing_needs_no_docling_unless_there_is_a_scan(corpus, fake_parser, monkeypatch) -> None:
    """Regression: the OCR cache key asked docling for its version for every
    document, and without docling - the environment CI runs in - nothing at
    all was parsed."""
    import importlib.metadata as metadata

    real = metadata.version

    def version(name: str) -> str:
        if name == "docling":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", version)

    mvp = run(corpus)
    full = run(corpus, scope="full")

    assert {r.document.source.name for r in mvp.results} == {"kernel.pdf", "threads.pdf"}
    assert "scan.pdf" in {d.source.name for d in full.skipped}


def test_a_page_holding_a_unicode_line_separator_is_cleaned(corpus, monkeypatch) -> None:
    """Regression: U+0085 in a page's text broke reading the cache back."""

    def parse_pdf(path: Path) -> list[ParsedPage]:
        return [ParsedPage(number=1, label=None, markdown="Текст\x85с\u2028разделителями.")]

    monkeypatch.setattr(stage, "parse_pdf", parse_pdf)

    result = run(corpus)

    assert all(len(r.pages) == 1 for r in result.results)
