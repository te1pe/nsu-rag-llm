"""PDF with a text layer -> Markdown, one record per page.

`pymupdf4llm` in layout mode does the heavy lifting: a small ONNX model
classifies every region of a page (text, section header, formula, picture,
table, running header, running footer) and renders them in reading order.
Measured against the older font-size heuristics on real pages of this corpus
it finds far more headings, drops running headers, and keeps paragraphs whole
(ADR-012). It gets two things wrong, and both are repaired here using the
character offsets (`pos`) it reports for every region:

* **Formulas** are treated as pictures and vanish from the text. A text layer
  cannot represent a display formula anyway - the older renderer produced one
  symbol per line - so the region becomes a `[формула]` marker: the answer can
  still say "the formula is on page 82".
* **Code listings** are flattened into one long line. A region whose text is
  mostly in a monospaced font is re-extracted line by line into a fenced
  block, with indentation recovered from the x-coordinate of each line.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

from nsu_rag_llm.ingest.cleaning import FORMULA, PICTURE

#: Bump when the output of this module changes for the same input: it is part
#: of the cache key, so a bump re-parses everything once.
PARSER_REVISION = 3

#: A region counts as code when at least this share of its characters are set
#: in a monospaced font. Inline `code` inside prose stays well below it.
CODE_SHARE = 0.6

#: Text extraction without image blocks. The default for "dict" includes every
#: image's pixels, and several MVP books are scans with a text layer under a
#: full-page picture: decoding it on every call made one region check cost
#: 80-150 ms instead of 6. Image blocks carry no lines, so the text that comes
#: back is identical either way.
TEXT_ONLY = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES

#: The layout model sometimes calls a whole paragraph a formula when it is
#: prose with inline maths - a definition, a theorem. Measured on the MVP: one
#: such region covered 52% of a page and held 50 Russian words; genuine display
#: formulas hold at most a handful ("при", "где"). A region with at least this
#: many words of four or more letters is kept as text instead of a marker.
PROSE_WORDS = 8
WORD = re.compile(r"[^\W\d_]{4,}")

#: Symbol fonts put their own glyphs at private-use code points - byte 0xNN of
#: the font at U+F0NN - so one code point means different things in different
#: fonts: U+F0FC is a check mark in Wingdings and a brace piece in Symbol. Only
#: glyphs measured in this corpus, in fonts recognised by name, are mapped;
#: anything else stays as it is and is counted as junk by the report.
#: Measured on the MVP: 1 480 Wingdings check marks in one book's support
#: tables (86% of all private-use characters), 41 arrows in EuclidSymbol, 15
#: bullets in SymbolMT. MT-Extra uses another encoding and is left alone.
FONT_GLYPHS = {
    "wingdings": {0xF0FC: "\u2713"},
    "symbol": {0xF0AE: "\u2192", 0xF0B7: "\u2022"},
}

MONOSPACE_FONT = re.compile(r"courier|consolas|mono|menlo|monaco|inconsolata|lucida console", re.I)
PICTURE_TEXT_MARKERS = re.compile(r"<!-- (Start|End) of picture text -->\n?")


@dataclass
class ParsedPage:
    """One page as the parser produced it, before cleaning."""

    number: int
    """1-based position in the file - the page number used everywhere (ADR-011)."""
    label: str | None
    """The page label the PDF itself defines, if any."""
    markdown: str
    formulas: int = 0
    pictures: int = 0
    code_blocks: int = 0
    running_boxes: int = 0
    """Page headers and footers the layout model recognised and dropped."""
    source_chars: int = 0
    """Non-space characters in the page's own text layer. Lets the report tell a
    page that is blank in the PDF from a page whose text the parser lost."""
    formula_prose: int = 0
    """Regions labelled as formulas that held prose, and were kept as text."""

    def to_json(self) -> dict:
        return asdict(self)


def parser_key() -> str:
    """Identifies everything that decides the parser's output."""
    return f"pymupdf4llm-layout/{pymupdf4llm.version}/r{PARSER_REVISION}"


def _is_monospace(span: dict) -> bool:
    # Bit 3 of the span flags is PyMuPDF's "monospaced"; plenty of PDFs leave
    # it unset, hence the font-name fallback.
    return bool(span["flags"] & 8) or bool(MONOSPACE_FONT.search(span["font"]))


def code_block(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> str | None:
    """Re-extract a region as a fenced listing, or None if it is not code."""
    content = page.get_text("dict", clip=pymupdf.Rect(bbox), flags=TEXT_ONLY)
    lines = []
    for block in content["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                lines.append((line["bbox"][1], line["bbox"][0], line["spans"], text))
    if not lines:
        return None

    total = sum(len(text) for *_, text in lines)
    mono = sum(len(s["text"]) for *_, spans, _ in lines for s in spans if _is_monospace(s))
    if mono / total < CODE_SHARE:
        return None

    # Indentation is not stored in a PDF - only where each line starts. One
    # character's width, measured on this listing, turns that back into spaces.
    widths = [
        (s["bbox"][2] - s["bbox"][0]) / len(s["text"])
        for *_, spans, _ in lines
        for s in spans
        if _is_monospace(s) and len(s["text"].strip()) > 3
    ]
    char_width = statistics.median(widths) if widths else 6.0
    lines.sort(key=lambda row: (round(row[0]), row[1]))
    left = min(x for _, x, _, _ in lines)
    body = "\n".join(
        " " * round((x - left) / char_width) + text.rstrip() for _, x, _, text in lines
    )
    return f"```\n{body}\n```"


def _is_private_use(char: str) -> bool:
    return "\ue000" <= char <= "\uf8ff"


def glyph_map(page: pymupdf.Page) -> dict[int, str]:
    """Translation table for the private-use characters on this page.

    Built from the fonts the characters are actually set in. A character seen
    in two fonts that disagree - or in a font we know nothing about - is left
    alone rather than guessed.
    """
    seen: dict[int, set[str | None]] = {}
    for block in page.get_text("dict", flags=TEXT_ONLY)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                font = span["font"].lower()
                family = next((name for name in FONT_GLYPHS if name in font), None)
                for char in span["text"]:
                    if _is_private_use(char):
                        meaning = FONT_GLYPHS.get(family, {}).get(ord(char)) if family else None
                        seen.setdefault(ord(char), set()).add(meaning)
    return {
        code: meanings.pop()
        for code, meanings in seen.items()
        if len(meanings) == 1 and None not in meanings
    }


def formula_prose(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> str | None:
    """The text of a "formula" region, if it is really prose; else None.

    The text is kept exactly as the PDF has it - inline symbols included, in
    whatever state the text layer left them - with only its lines joined into
    one paragraph.
    """
    text = page.get_text("text", clip=pymupdf.Rect(bbox), flags=TEXT_ONLY)
    if len(WORD.findall(text)) < PROSE_WORDS:
        return None
    return " ".join(text.split())


def rebuild(page: pymupdf.Page, chunk: dict) -> ParsedPage:
    """Repair one page of layout output using the offsets of its regions."""
    text = chunk["text"]
    result = ParsedPage(
        number=chunk["metadata"]["page_number"],
        label=page.get_label() or None,
        markdown="",
    )
    pieces: list[str] = []
    cursor = 0
    for box in sorted(chunk["page_boxes"], key=lambda b: b["pos"][0]):
        start, end = box["pos"]
        kind = box["class"]
        if kind in ("page-header", "page-footer"):
            result.running_boxes += 1
            continue
        if start < cursor:  # overlapping offsets would duplicate text
            continue
        if kind == "formula":
            if prose := formula_prose(page, box["bbox"]):
                replacement = f"\n\n{prose}\n\n"
                result.formula_prose += 1
            else:
                replacement = f"\n\n{FORMULA}\n\n"
                result.formulas += 1
        elif kind == "picture":
            # Keep any words printed inside the figure - axis labels, block
            # names - but without the HTML comments the renderer wraps them in.
            inner = PICTURE_TEXT_MARKERS.sub("", text[start:end]).strip()
            replacement = f"\n\n{PICTURE}\n\n{inner}\n\n" if inner else f"\n\n{PICTURE}\n\n"
            result.pictures += 1
        elif kind in ("text", "list-item") and (listing := code_block(page, box["bbox"])):
            replacement = f"\n\n{listing}\n\n"
            result.code_blocks += 1
        else:
            continue
        pieces += [text[cursor:start], replacement]
        cursor = end
    pieces.append(text[cursor:])
    markdown = "".join(pieces)
    if any(_is_private_use(char) for char in markdown):
        markdown = markdown.translate(glyph_map(page))
    result.markdown = markdown
    result.source_chars = len("".join(page.get_text(flags=TEXT_ONLY).split()))
    return result


def parse_pdf(path: Path) -> list[ParsedPage]:
    """Parse every page of a PDF that has a usable text layer."""
    with pymupdf.open(path) as doc:
        chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            # Running headers and footers are recognised by the layout model.
            header=False,
            footer=False,
            # OCR belongs to phase 4.5. Left on, the parser would quietly OCR
            # figure-only pages in English, its default language.
            use_ocr=False,
            show_progress=False,
        )
        return [rebuild(doc[chunk["metadata"]["page_number"] - 1], chunk) for chunk in chunks]
