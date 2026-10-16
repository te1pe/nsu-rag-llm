"""`courses.yaml` is the author's file; regenerating it must not edit their work."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nsu_rag_llm.ingest.courses import exclusion_reason, read_decisions, render, write
from nsu_rag_llm.ingest.inventory import FileReport, Inventory


def report(
    path: str,
    pages: int = 200,
    chars_per_page: float = 2000.0,
    language: str = "ru",
    duplicate_of: str | None = None,
    content: str | None = None,
) -> FileReport:
    """A synthetic inventory entry. `content` names the bytes: two reports with
    the same `content` are copies of one document."""
    parts = Path(path).parts
    return FileReport(
        path=Path(path),
        course=parts[0],
        semester=int(parts[1].removeprefix("sem")),
        size_bytes=1024,
        sha256=hashlib.sha256((content or path).encode()).hexdigest(),
        fmt="pdf",
        pages=pages,
        chars_per_page=chars_per_page,
        language=language,
        duplicate_of=Path(duplicate_of) if duplicate_of else None,
    )


@pytest.fixture
def inventory() -> Inventory:
    return Inventory(
        files=[
            report("matan/sem1/lectures.pdf", pages=300),
            report("matan/sem1/problems.pdf", pages=250),
            report("matan/sem2/scanned.pdf", pages=400, chars_per_page=5.0),
            report("evm/sem3/intel-manual.pdf", pages=4800, language="en"),
            report("evm/sem3/notes.pdf", pages=120),
        ]
    )


def test_a_fresh_file_proposes_the_measured_selection(inventory: Inventory, tmp_path: Path) -> None:
    path = write(inventory, tmp_path / "courses.yaml")
    text = path.read_text(encoding="utf-8")

    assert 'title: ""' in text  # left for the author to fill
    assert "  matan:" in text and "  evm:" in text
    # Facts the author should not have to look up.
    assert "pages: 4800" in text
    assert "layer: none" in text


def test_excluded_files_say_why(inventory: Inventory, tmp_path: Path) -> None:
    text = write(inventory, tmp_path / "courses.yaml").read_text(encoding="utf-8")

    assert "скан, нужен OCR" in text
    assert "забьёт выдачу" in text


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"chars_per_page": 5.0}, "скан, нужен OCR"),
        ({"chars_per_page": 120.0}, "текстовый слой рваный, надёжнее OCR"),
        ({"pages": 4800}, "забьёт выдачу"),
        ({"language": "en"}, "по-английски"),
        ({"duplicate_of": "matan/sem1/other.pdf"}, "дубликат"),
    ],
)
def test_every_exclusion_has_a_readable_reason(kwargs: dict, expected: str) -> None:
    reason = exclusion_reason(report("matan/sem1/x.pdf", **kwargs))

    assert reason is not None
    assert expected in reason


def test_a_usable_file_has_no_exclusion_reason() -> None:
    assert exclusion_reason(report("matan/sem1/good.pdf")) is None


def test_author_titles_survive_regeneration(inventory: Inventory, tmp_path: Path) -> None:
    """Titles are lifted as raw text, so the notes written after them survive."""
    path = write(inventory, tmp_path / "courses.yaml")
    edited = path.read_text(encoding="utf-8").replace(
        '    title: ""   # TODO',
        '    title: "Математический анализ"   # ПРОВЕРЬ точное название',
        1,
    )
    path.write_text(edited, encoding="utf-8")

    write(inventory, path)
    text = path.read_text(encoding="utf-8")

    assert '"Математический анализ"' in text
    assert "# ПРОВЕРЬ точное название" in text


def test_author_include_flags_survive_regeneration(inventory: Inventory, tmp_path: Path) -> None:
    path = write(inventory, tmp_path / "courses.yaml")
    before = read_decisions(path)
    flipped = {slug: not value for slug, value in before.course_include.items()}

    text = path.read_text(encoding="utf-8").splitlines()
    current = None
    for i, line in enumerate(text):
        if line.startswith("  ") and line.endswith(":") and not line.startswith("    "):
            current = line.strip().rstrip(":")
        elif current and line.startswith("    include:"):
            text[i] = f"    include: {str(flipped[current]).lower()}"
            current = None
    path.write_text("\n".join(text) + "\n", encoding="utf-8")

    write(inventory, path)

    assert read_decisions(path).course_include == flipped


def test_regeneration_is_idempotent(inventory: Inventory, tmp_path: Path) -> None:
    path = write(inventory, tmp_path / "courses.yaml")
    first = path.read_bytes()

    write(inventory, path)

    assert path.read_bytes() == first


def test_filenames_with_awkward_characters_stay_valid_yaml(tmp_path: Path) -> None:
    import yaml

    inventory = Inventory(files=[report('matan/sem1/Zorich: "Analysis", vol. 2 (2012).pdf')])

    path = write(inventory, tmp_path / "courses.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    entry = loaded["courses"]["matan"]["documents"][0]
    assert entry["file"] == 'sem1/Zorich: "Analysis", vol. 2 (2012).pdf'


def test_render_does_not_touch_disk(inventory: Inventory, tmp_path: Path) -> None:
    text = render(inventory, read_decisions(tmp_path / "missing.yaml"))

    assert text.startswith("# Course catalogue")
    assert not (tmp_path / "missing.yaml").exists()


def test_a_duplicate_copy_is_never_included(tmp_path: Path) -> None:
    inventory = Inventory(
        files=[
            report("matan/sem2/demidovich.pdf", content="demidovich"),
            report(
                "matan/sem1/demidovich (1).pdf",
                content="demidovich",
                duplicate_of="matan/sem2/demidovich.pdf",
            ),
            report("matan/sem1/lectures.pdf"),
        ]
    )

    text = write(inventory, tmp_path / "courses.yaml").read_text(encoding="utf-8")

    assert '"sem1/demidovich (1).pdf"\n        include: false   # дубликат' in text


def test_the_decision_follows_the_content_when_the_primary_changes(tmp_path: Path) -> None:
    """The author included a document while copy A was primary. A change to the
    duplicate rule then makes copy B primary; the document must stay included,
    and must not be included twice."""
    path = tmp_path / "courses.yaml"
    before = Inventory(
        files=[
            report("matan/sem1/book.pdf", content="book"),
            report("matan/sem2/book.pdf", content="book", duplicate_of="matan/sem1/book.pdf"),
            report("matan/sem1/other.pdf"),
        ]
    )
    write(before, path)
    # The author excludes the other document and keeps the book.
    edited = path.read_text(encoding="utf-8").replace(
        '"sem1/other.pdf"\n        include: true',
        '"sem1/other.pdf"\n        include: false',
    )
    path.write_text(edited, encoding="utf-8")

    after = Inventory(
        files=[
            report("matan/sem1/book.pdf", content="book", duplicate_of="matan/sem2/book.pdf"),
            report("matan/sem2/book.pdf", content="book"),
            report("matan/sem1/other.pdf"),
        ]
    )
    write(after, path)

    decisions = read_decisions(path).file_include
    assert decisions[("matan", "sem2/book.pdf")] is True
    assert decisions[("matan", "sem1/book.pdf")] is False
    assert decisions[("matan", "sem1/other.pdf")] is False
