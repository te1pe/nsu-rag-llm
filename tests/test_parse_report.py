"""Which pages the report sends a person to look at."""

from __future__ import annotations

import pytest

from nsu_rag_llm.ingest.parse_report import _ranges, classify


def page(
    number: int = 50,
    chars: int = 1800,
    source: int = 1600,
    junk: float = 0.0,
    junk_chars: int = 0,
    pictures: int = 0,
    formulas: int = 0,
) -> dict:
    return {
        "number": number,
        "chars": chars,
        "text_chars": int(chars * 0.85),  # roughly: prose minus spaces
        "source_chars": source,
        "junk_ratio": junk,
        "junk_chars": junk_chars,
        "pictures": pictures,
        "formulas": formulas,
    }


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        # The parser kept the page: nothing to say.
        (page(), None),
        # Blank in the PDF itself - a verso before a new chapter. Not a loss.
        (page(chars=0, source=3), ("expected", "пустая и в самом PDF")),
        # The PDF has a full page of text and almost none came out: open it.
        (page(chars=200, source=2000), ("review", "разбор потерял текст")),
        # Same shortfall, but formulas were replaced by markers: explained.
        (page(chars=200, source=2000, formulas=6), None),
        (page(chars=60, source=900, pictures=1), ("expected", "рисунок")),
        (page(chars=60, source=400, formulas=3), ("expected", "формулы")),
        (page(number=1, chars=60, source=400), ("expected", "обложка или концевая")),
        (page(chars=60, source=400), ("review", "почти пусто")),
    ],
)
def test_pages_are_judged_against_their_own_text_layer(p: dict, expected) -> None:
    assert classify(p, page_count=100) == expected


def test_a_few_broken_glyphs_on_a_short_page_are_not_worth_a_look() -> None:
    """Four U+FFFD on a 180-character page is 2% - and nothing anyone can fix."""
    assert classify(page(chars=180, source=170, junk=0.022, junk_chars=4), page_count=100) is None


def test_many_broken_glyphs_are_flagged() -> None:
    kind, reason = classify(page(junk=0.05, junk_chars=90), page_count=100)

    assert kind == "review"
    assert "мусор" in reason


def test_a_cover_set_in_an_unmapped_font_is_expected() -> None:
    verdict = classify(page(number=1, chars=62, junk=0.77, junk_chars=48), page_count=100)

    assert verdict == ("expected", "обложка: шрифт без Unicode")


def test_page_numbers_are_listed_as_ranges() -> None:
    assert _ranges([9, 3, 4, 5, 12, 13]) == "3–5, 9, 12–13"
    assert _ranges([]) == ""


def test_a_scan_page_ocr_found_nothing_on_is_worth_a_look() -> None:
    """A scan has no text layer: empty after OCR is not "empty in the PDF"."""
    scanned = page(chars=0, source=0) | {"origin": "ocr"}

    assert classify(scanned, 300) == ("review", "OCR не нашёл текста")
    assert classify(scanned | {"number": 1}, 300) == ("expected", "обложка или концевая")


def test_a_page_the_ocr_engine_failed_on_is_flagged() -> None:
    failed = page(chars=0, source=0) | {"origin": "ocr", "ocr_error": "RuntimeError: boom"}

    assert classify(failed, 300) == ("review", "OCR упал на странице")
