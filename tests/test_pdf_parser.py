"""The two repairs made on top of the layout parser, on synthetic PDFs.

Latin text only: PDF base-14 fonts cannot encode Cyrillic (see
test_inventory.make_pdf). Courier - also base-14 - stands in for a code font.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from nsu_rag_llm.ingest.cleaning import FORMULA, PICTURE
from nsu_rag_llm.ingest.parsers.pdf import code_block, glyph_map, parse_pdf, rebuild

LISTING = [
    (0, "int main(void) {"),
    (4, "pid_t pid = fork();"),
    (4, "if (pid == 0) {"),
    (8, 'execlp("ls", "ls", NULL);'),
    (4, "}"),
    (4, "return 0;"),
    (0, "}"),
]
PROSE = "A process is a program in execution; the kernel schedules its threads. "


@pytest.fixture
def page_with_listing(tmp_path: Path):
    """One page: a paragraph in Helvetica above a listing in Courier.

    Returns the open page and the rectangles of the two regions.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    prose_rect = pymupdf.Rect(72, 72, 520, 160)
    page.insert_textbox(prose_rect, PROSE * 4, fontname="helv", fontsize=10)

    size, top, left = 10, 200, 72
    char_width = pymupdf.get_text_length("x", fontname="cour", fontsize=size)
    for row, (indent, line) in enumerate(LISTING):
        page.insert_text(
            (left + indent * char_width, top + row * 14), line, fontname="cour", fontsize=size
        )
    code_rect = pymupdf.Rect(left - 2, top - 12, 520, top + len(LISTING) * 14)

    path = tmp_path / "listing.pdf"
    doc.save(path)
    doc.close()
    reopened = pymupdf.open(path)
    yield reopened[0], prose_rect, code_rect
    reopened.close()


def test_a_monospaced_region_becomes_a_fenced_listing(page_with_listing) -> None:
    page, _, code_rect = page_with_listing

    block = code_block(page, tuple(code_rect))

    assert block is not None
    lines = block.splitlines()
    assert lines[0] == "```" and lines[-1] == "```"
    body = lines[1:-1]
    assert len(body) == len(LISTING)  # one line per line, nothing joined


def test_indentation_is_recovered_from_line_positions(page_with_listing) -> None:
    page, _, code_rect = page_with_listing

    body = code_block(page, tuple(code_rect)).splitlines()[1:-1]

    indents = [len(line) - len(line.lstrip(" ")) for line in body]
    assert indents == [indent for indent, _ in LISTING]


def test_prose_is_not_mistaken_for_code(page_with_listing) -> None:
    page, prose_rect, _ = page_with_listing

    assert code_block(page, tuple(prose_rect)) is None


def _chunk(text: str, boxes: list[tuple[str, tuple, tuple[int, int]]]) -> dict:
    return {
        "text": text,
        "metadata": {"page_number": 1},
        "page_boxes": [
            {"class": kind, "bbox": bbox, "pos": pos, "index": i}
            for i, (kind, bbox, pos) in enumerate(boxes)
        ],
    }


def test_a_dropped_formula_leaves_a_marker_in_its_place(page_with_listing) -> None:
    page, prose_rect, _ = page_with_listing
    before, after = "By the convolution formula the density is\n\n", "\n\nwhich integrates to one."
    text = before + "\n\n" + after
    chunk = _chunk(
        text,
        [
            ("text", tuple(prose_rect), (0, len(before))),
            ("formula", (0, 0, 1, 1), (len(before), len(before) + 2)),
            ("text", tuple(prose_rect), (len(before) + 2, len(text))),
        ],
    )

    result = rebuild(page, chunk)

    assert result.formulas == 1
    assert result.markdown.index(FORMULA) > result.markdown.index("density is")
    assert result.markdown.index(FORMULA) < result.markdown.index("integrates")


def test_picture_text_is_kept_without_the_renderers_comments(page_with_listing) -> None:
    page, _, _ = page_with_listing
    inner = "\n\n<!-- Start of picture text -->\nCPU\nRAM\n<!-- End of picture text -->\n"
    chunk = _chunk(inner, [("picture", (0, 0, 1, 1), (0, len(inner)))])

    result = rebuild(page, chunk)

    assert result.pictures == 1
    assert PICTURE in result.markdown
    assert "CPU" in result.markdown and "RAM" in result.markdown
    assert "<!--" not in result.markdown


def test_running_headers_are_counted_not_rendered(page_with_listing) -> None:
    page, prose_rect, _ = page_with_listing
    chunk = _chunk(
        "Body text.", [("page-header", (0, 0, 1, 1), (0, 0)), ("text", tuple(prose_rect), (0, 10))]
    )

    result = rebuild(page, chunk)

    assert result.running_boxes == 1
    assert result.markdown == "Body text."


def test_a_flattened_listing_is_replaced_by_the_fenced_one(page_with_listing) -> None:
    page, _, code_rect = page_with_listing
    flattened = "`int main(void) { pid_t pid = fork(); if (pid == 0) { ... } return 0; }`"
    chunk = _chunk(flattened, [("text", tuple(code_rect), (0, len(flattened)))])

    result = rebuild(page, chunk)

    assert result.code_blocks == 1
    assert result.markdown.strip().startswith("```")
    assert "    pid_t pid = fork();" in result.markdown


def test_parse_pdf_numbers_pages_from_one(tmp_path: Path) -> None:
    """End to end through the real layout model, which ships in the wheel."""
    doc = pymupdf.open()
    for n in range(3):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 700), f"Page {n + 1}. " + PROSE * 8)
    path = tmp_path / "three.pdf"
    doc.save(path)
    doc.close()

    pages = parse_pdf(path)

    assert [p.number for p in pages] == [1, 2, 3]
    assert all("process" in p.markdown for p in pages)
    assert all(p.label is None for p in pages)


class FakePage:
    """Just enough of a PyMuPDF page for `glyph_map`: spans with font names.

    A real PDF set in Wingdings would need the font file shipped with the tests.
    """

    def __init__(self, spans: list[tuple[str, str]]) -> None:
        self.spans = spans

    def get_text(self, _kind: str, flags: int = 0) -> dict:
        return {
            "blocks": [
                {"lines": [{"spans": [{"font": font, "text": text} for font, text in self.spans]}]}
            ]
        }


def test_a_wingdings_check_mark_becomes_a_real_one() -> None:
    page = FakePage([("Helvetica", "fork()"), ("Wingdings-Regular", "")])

    assert glyph_map(page) == {0xF0FC: "✓"}


def test_the_same_code_point_in_another_font_is_not_guessed() -> None:
    """U+F0FC is a check mark in Wingdings and a brace piece in Symbol."""
    page = FakePage([("SymbolMT", "")])

    assert glyph_map(page) == {}


def test_a_conflict_between_fonts_on_one_page_is_left_alone() -> None:
    page = FakePage([("Wingdings-Regular", ""), ("MT-Extra", "")])

    assert glyph_map(page) == {}


def test_symbol_encoded_fonts_share_one_table() -> None:
    page = FakePage([("EuclidSymbol", "x  a"), ("SymbolMT", " item")])

    assert glyph_map(page) == {0xF0AE: "→", 0xF0B7: "•"}


def test_rebuild_records_the_size_of_the_pages_own_text_layer(page_with_listing) -> None:
    page, prose_rect, _ = page_with_listing
    chunk = _chunk("Body text.", [("text", tuple(prose_rect), (0, 10))])

    result = rebuild(page, chunk)

    assert result.source_chars > 200  # the paragraph and the listing on the page


def test_a_formula_region_holding_prose_keeps_its_text(page_with_listing) -> None:
    """The layout model sometimes calls a paragraph with inline maths a formula;
    replacing it with a marker would throw away a definition or a theorem."""
    page, prose_rect, _ = page_with_listing
    chunk = _chunk("\n\n", [("formula", tuple(prose_rect), (0, 2))])

    result = rebuild(page, chunk)

    assert result.formula_prose == 1
    assert result.formulas == 0
    assert FORMULA not in result.markdown
    assert "kernel schedules its threads" in result.markdown


def test_a_real_formula_region_still_becomes_a_marker(page_with_listing) -> None:
    """Few words of four letters or more: symbols, not prose."""
    page, _, code_rect = page_with_listing
    chunk = _chunk("\n\n", [("formula", tuple(code_rect), (0, 2))])

    result = rebuild(page, chunk)

    assert result.formulas == 1
    assert result.formula_prose == 0
    assert FORMULA in result.markdown
