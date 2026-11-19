from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

from nsu_rag_llm.ingest.cleaning import FORMULA, PICTURE

PARSER_REVISION = 3

CODE_SHARE = 0.6

# без картинок в разы быстрее
TEXT_ONLY = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES

PROSE_WORDS = 8
WORD = re.compile(r"[^\W\d_]{4,}")

FONT_GLYPHS = {
    "wingdings": {0xF0FC: "\u2713"},
    "symbol": {0xF0AE: "\u2192", 0xF0B7: "\u2022"},
}

MONOSPACE_FONT = re.compile(r"courier|consolas|mono|menlo|monaco|inconsolata|lucida console", re.I)
PICTURE_TEXT_MARKERS = re.compile(r"<!-- (Start|End) of picture text -->\n?")


@dataclass
class ParsedPage:
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
    return f"pymupdf4llm-layout/{pymupdf4llm.version}/r{PARSER_REVISION}"


def _is_monospace(span: dict) -> bool:
    return bool(span["flags"] & 8) or bool(MONOSPACE_FONT.search(span["font"]))


def code_block(page: pymupdf.Page, bbox: tuple[float, float, float, float]) -> str | None:
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
    text = page.get_text("text", clip=pymupdf.Rect(bbox), flags=TEXT_ONLY)
    if len(WORD.findall(text)) < PROSE_WORDS:
        return None
    return " ".join(text.split())


def rebuild(page: pymupdf.Page, chunk: dict) -> ParsedPage:
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
        if start < cursor:
            continue
        if kind == "formula":
            if prose := formula_prose(page, box["bbox"]):
                replacement = f"\n\n{prose}\n\n"
                result.formula_prose += 1
            else:
                replacement = f"\n\n{FORMULA}\n\n"
                result.formulas += 1
        elif kind == "picture":
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
    with pymupdf.open(path) as doc:
        chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            header=False,
            footer=False,
            use_ocr=False,
            show_progress=False,
        )
        return [rebuild(doc[chunk["metadata"]["page_number"] - 1], chunk) for chunk in chunks]
