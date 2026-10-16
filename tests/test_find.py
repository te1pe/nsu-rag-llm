"""`nsu-rag find`: where a phrase is, for writing gold pages by. Synthetic pages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nsu_rag_llm.evaluation.find import find


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    raw, parsed = tmp_path / "raw", tmp_path / "parsed"
    documents = {
        "aaa": (
            "os/sem4/Синтетическая книга.pdf",
            "digital",
            ["Вступление.", "Семафор —\nсчётчик ресурсов."],
        ),
        "bbb": ("matan/sem1/Скан.pdf", "ocr", ["Предел последовательности; ёлка."]),
    }
    for sha, (file, origin, pages) in documents.items():
        folder = parsed / sha
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(json.dumps({"source": str(raw / file)}), encoding="utf-8")
        lines = [
            json.dumps({"number": n, "markdown": text, "origin": origin}, ensure_ascii=False)
            for n, text in enumerate(pages, start=1)
        ]
        (folder / "pages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw, parsed


def test_a_phrase_is_found_with_its_file_and_page(corpus) -> None:
    raw, parsed = corpus

    (match,) = find("семафор — счетчик", parsed, raw, {"mvp": {"aaa"}})

    assert match.file == "os/sem4/Синтетическая книга.pdf"  # as a gold entry wants it
    assert match.page == 2
    assert match.indexes == ("mvp",)


def test_case_yo_and_line_breaks_do_not_matter(corpus) -> None:
    raw, parsed = corpus

    assert find("ЕЛКА", parsed, raw, {})[0].origin == "ocr"  # scans are searchable too
    assert find("семафор —   счётчик", parsed, raw, {})[0].page == 2


def test_the_course_filter_and_the_limit(corpus) -> None:
    raw, parsed = corpus

    assert find("предел", parsed, raw, {}, course="os") == []
    assert len(find("с", parsed, raw, {}, limit=1)) == 1


def test_nothing_is_ranked_or_stemmed(corpus) -> None:
    """Gold is the author's judgement: an inflected form is not "found"."""
    raw, parsed = corpus

    assert find("семафоры", parsed, raw, {}) == []
    assert find("", parsed, raw, {}) == []
