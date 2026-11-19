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
        "text_chars": int(chars * 0.85),
        "source_chars": source,
        "junk_ratio": junk,
        "junk_chars": junk_chars,
        "pictures": pictures,
        "formulas": formulas,
    }


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (page(), None),
        (page(chars=0, source=3), ("expected", "пустая и в самом PDF")),
        (page(chars=200, source=2000), ("review", "разбор потерял текст")),
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
    scanned = page(chars=0, source=0) | {"origin": "ocr"}

    assert classify(scanned, 300) == ("review", "OCR не нашёл текста")
    assert classify(scanned | {"number": 1}, 300) == ("expected", "обложка или концевая")


def test_a_page_the_ocr_engine_failed_on_is_flagged() -> None:
    failed = page(chars=0, source=0) | {"origin": "ocr", "ocr_error": "RuntimeError: boom"}

    assert classify(failed, 300) == ("review", "OCR упал на странице")
