from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from nsu_rag_llm.ingest.cleaning import PLACEHOLDERS, split_fenced
from nsu_rag_llm.ingest.inventory import text_stats

TokenCounter = Callable[[str], int]

SEPARATOR = " › "
MIN_BEFORE_BREAK = 100
# контекст не вытесняет сам текст
MAX_CONTEXT_TOKENS = 64
MODEL_MAX_TOKENS = 512
MODEL_RESERVED_TOKENS = 8

HEADING = re.compile(r"^(#{1,6}) +(.+?)\s*$")
MAX_HEADING_CHARS = 150
TABLE_LINE = re.compile(r"^\s*\|")
TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[A-ZА-ЯЁ0-9])")


@dataclass
class Block:
    kind: str
    text: str
    page_start: int
    page_end: int
    level: int = 0
    """Heading level, for headings."""
    path: tuple[str, ...] = ()
    """Section titles above this block, outermost first."""
    carried: bool = False
    """Repeated from the previous chunk as overlap."""
    split: bool = False
    """One piece of a listing or table too long to keep in one chunk."""
    tokens: int | None = None
    """Cached token count; packing asks for it many times."""


@dataclass
class DocInfo:
    course: str
    course_title: str
    doc_title: str
    file: str
    sha256: str
    semester: int | None
    origin: str


@dataclass
class Chunk:
    id: str
    course: str
    course_title: str
    doc_title: str
    file: str
    doc_sha256: str
    semester: int | None
    origin: str
    ordinal: int
    page_start: int
    page_end: int
    section_path: list[str]
    text: str
    context: str
    tokens: int
    lang: str
    has_code: bool = False
    has_table: bool = False
    split_structure: bool = False
    """A listing or table too long for one chunk was split along its structure."""

    @property
    def embedding_text(self) -> str:
        return f"{self.context}\n\n{self.text}"

    def to_json(self) -> dict:
        return dict(self.__dict__)


def clean_title(markdown: str) -> str:
    title = re.sub(r"[*_`]", "", markdown)
    title = re.sub(r"\s+", " ", title).strip(" .:")
    return title[:120]


def _prose_blocks(segment: str, page: int) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[str] = []

    def close_paragraph() -> None:
        if "".join(paragraph).strip():
            blocks.append(Block("text", "\n".join(paragraph).strip(), page, page))
        paragraph.clear()

    def close_table() -> None:
        if table:
            blocks.append(Block("table", "\n".join(table), page, page))
        table.clear()

    for line in segment.split("\n"):
        if TABLE_LINE.match(line):
            close_paragraph()
            table.append(line)
            continue
        close_table()
        heading = HEADING.match(line)
        if heading and len(heading.group(2)) > MAX_HEADING_CHARS:
            paragraph.append(heading.group(2))
        elif heading:
            close_paragraph()
            if clean_title(heading.group(2)):
                blocks.append(
                    Block("heading", line.strip(), page, page, level=len(heading.group(1)))
                )
        elif line.strip():
            paragraph.append(line)
        else:
            close_paragraph()
    close_paragraph()
    close_table()
    return blocks


def page_blocks(markdown: str, page: int) -> list[Block]:
    blocks: list[Block] = []
    for is_code, segment in split_fenced(markdown):
        if not is_code:
            blocks.extend(_prose_blocks(segment, page))
        elif segment.strip():
            blocks.append(Block("code", segment.strip(), page, page))
    return blocks


def join_across_pages(pages: Iterable[tuple[int, str]]) -> list[Block]:
    blocks: list[Block] = []
    for number, markdown in pages:
        current = page_blocks(markdown, number)
        if (
            current
            and blocks
            and current[0].kind == "text"
            and blocks[-1].kind == "text"
            and current[0].text[:1].islower()
        ):
            previous = blocks[-1]
            joiner = "" if previous.text.endswith("-") else " "
            previous.text = previous.text.rstrip("-") + joiner + current[0].text
            previous.page_end = current[0].page_end
            current = current[1:]
        blocks.extend(current)
    return blocks


def assign_sections(blocks: list[Block]) -> None:
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.kind == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, clean_title(HEADING.match(block.text).group(2))))
            block.path = tuple(title for _, title in stack)
        else:
            block.path = tuple(title for _, title in stack)


def _pack(pieces: list[str], budget: int, count: TokenCounter, joiner: str) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for piece in pieces:
        candidate = joiner.join([*current, piece])
        if current and count(candidate) > budget:
            groups.append(joiner.join(current))
            current = [piece]
        else:
            current.append(piece)
    if current:
        groups.append(joiner.join(current))
    return groups


def hard_split(text: str, budget: int, count: TokenCounter) -> list[str]:
    pieces: list[str] = []
    rest = text
    while rest and count(rest) > budget:
        low, high = 1, len(rest)
        while low < high:
            middle = (low + high + 1) // 2
            if count(rest[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        cut = rest.rfind(" ", 0, low)
        cut = cut if cut > low // 2 else low
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _fit(pieces: list[str], budget: int, count: TokenCounter) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        out.extend(hard_split(piece, budget, count) if count(piece) > budget else [piece])
    return out


def _split_words(text: str, budget: int, count: TokenCounter) -> list[str]:
    return _pack(_fit(text.split(" "), budget, count), budget, count, " ")


def split_prose(text: str, budget: int, count: TokenCounter) -> list[str]:
    sentences = SENTENCE_END.split(text)
    pieces: list[str] = []
    for sentence in sentences:
        if count(sentence) > budget:
            pieces.extend(_split_words(sentence, budget, count))
        else:
            pieces.append(sentence)
    return _pack(pieces, budget, count, " ")


def split_listing(text: str, budget: int, count: TokenCounter) -> list[str]:
    body = text.strip().removeprefix("```").removesuffix("```").strip("\n")
    fence_cost = count("```\n\n```")
    room = max(budget - fence_cost, 1)
    parts = re.split(r"\n\s*\n", body)
    lines_first: list[str] = []
    for part in parts:
        if count(part) > room:
            lines = _fit(part.split("\n"), room, count)
            lines_first.extend(_pack(lines, room, count, "\n"))
        else:
            lines_first.append(part)
    return [f"```\n{group}\n```" for group in _pack(lines_first, room, count, "\n\n")]


def split_table(text: str, budget: int, count: TokenCounter) -> list[str]:
    lines = text.split("\n")
    header = lines[:2] if len(lines) > 1 and TABLE_RULE.match(lines[1]) else lines[:1]
    rows = lines[len(header) :]
    if count("\n".join(header)) > budget // 2:
        rows, header = lines, []
    room = budget - count("\n".join(header)) if header else budget
    groups = _pack(_fit(rows, room, count), room, count, "\n")
    return ["\n".join([*header, group]) for group in groups]


def split_block(block: Block, budget: int, count: TokenCounter) -> list[Block]:
    if count(block.text) <= budget:
        return [block]
    splitter = {"code": split_listing, "table": split_table}.get(block.kind, split_prose)
    pieces = splitter(block.text, budget, count)
    structural = block.kind in ("code", "table") and len(pieces) > 1
    return [
        Block(
            block.kind,
            piece,
            block.page_start,
            block.page_end,
            block.level,
            block.path,
            split=structural,
        )
        for piece in pieces
    ]


def tail(text: str, tokens: int, count: TokenCounter) -> str:
    text = text.rstrip()
    starts = [0, *(match.end() for match in SENTENCE_END.finditer(text))]
    best = None
    for position in reversed(starts):
        if count(text[position:]) > tokens:
            break
        best = position
    if best is not None:
        return text[best:]
    words = [match.start() for match in re.finditer(r"(?<=\s)\S", text)]
    best = None
    for position in reversed(words):
        if count("\u2026" + text[position:]) > tokens:
            break
        best = position
    return "\u2026" + text[best:] if best is not None else ""


def _worth_carrying(text: str) -> bool:
    for placeholder in PLACEHOLDERS:
        text = text.replace(placeholder, "")
    return len(text.strip(" .\u2026")) >= 20


def common_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    prefix = list(paths[0])
    for path in paths[1:]:
        while prefix and tuple(prefix) != path[: len(prefix)]:
            prefix.pop()
    return tuple(prefix)


def context_line(document: DocInfo, path: tuple[str, ...], count: TokenCounter) -> str:
    parts = list(path)
    title = document.doc_title
    while True:
        line = SEPARATOR.join([document.course_title, title, *parts])
        if count(line) <= MAX_CONTEXT_TOKENS:
            return line
        if parts:
            parts.pop(0)
        elif " " in title:
            title = title.rsplit(" ", 1)[0]
        else:
            return line


def chunk_id(model: str, document: DocInfo, page_start: int, embedding_text: str) -> str:
    key = f"{model}|{document.sha256}|{page_start}|{embedding_text}"
    return str(uuid.UUID(bytes=hashlib.sha256(key.encode("utf-8")).digest()[:16]))


def chunk_document(
    document: DocInfo,
    pages: Iterable[tuple[int, str]],
    count: TokenCounter,
    model: str,
    budget: int = 400,
    overlap: int = 50,
    min_before_break: int = MIN_BEFORE_BREAK,
) -> list[Chunk]:
    budget = budget_for_model(budget)
    blocks = join_across_pages(pages)
    assign_sections(blocks)

    groups: list[list[Block]] = []
    current: list[Block] = []

    def size(items: list[Block]) -> int:
        total = 0
        for block in items:
            if block.tokens is None:
                block.tokens = count(block.text)
            total += block.tokens
        return total

    def has_content(items: list[Block]) -> bool:
        return any(
            b.kind != "heading" and not b.carried and b.text.strip() not in PLACEHOLDERS
            for b in items
        )

    def flush(with_overlap: bool) -> None:
        nonlocal current
        trailing: list[Block] = []
        if has_content(current):
            while current and current[-1].kind == "heading":
                trailing.insert(0, current.pop())
        if has_content(current) or size(current) >= budget // 2:
            groups.append(current)
            carried: list[Block] = []
            closing_run: list[Block] = []
            for block in reversed(current):
                if block.kind != "text" or block.carried:
                    break
                closing_run.insert(0, block)
            if with_overlap and overlap and closing_run:
                closing = tail("\n\n".join(b.text for b in closing_run), overlap, count)
                if _worth_carrying(closing):
                    last = closing_run[-1]
                    carried = [
                        Block(
                            "text",
                            closing,
                            last.page_end,
                            last.page_end,
                            path=last.path,
                            carried=True,
                        )
                    ]
            current = carried + trailing
        else:
            current = current + trailing

    for block in blocks:
        if block.kind == "heading":
            if all(b.carried for b in current):
                current = []
            if size(current) >= min_before_break:
                flush(with_overlap=False)
        room = budget - overlap if block.kind == "text" else budget
        for piece in split_block(block, room, count):
            if (
                current
                and size([*current, piece]) > budget
                and (has_content(current) or size(current) >= budget // 2)
            ):
                flush(with_overlap=True)
                if size([*current, piece]) > budget:
                    current = [b for b in current if not b.carried]
            current.append(piece)
    if has_content(current):
        groups.append(current)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for group in groups:
        own = [b for b in group if not b.carried]
        path = common_path([b.path for b in own]) or (own[0].path if own else ())
        text = "\n\n".join(b.text for b in group).strip()
        context = context_line(document, path, count)
        page_start = min(b.page_start for b in group)
        cid = chunk_id(model, document, page_start, f"{context}\n\n{text}")
        if cid in seen:
            continue
        seen.add(cid)
        _, language = text_stats(text)
        chunks.append(
            Chunk(
                id=cid,
                course=document.course,
                course_title=document.course_title,
                doc_title=document.doc_title,
                file=document.file,
                doc_sha256=document.sha256,
                semester=document.semester,
                origin=document.origin,
                ordinal=len(chunks),
                page_start=page_start,
                page_end=max(b.page_end for b in group),
                section_path=list(path),
                text=text,
                context=context,
                tokens=count(text),
                lang=language,
                has_code=any(b.kind == "code" for b in group),
                has_table=any(b.kind == "table" for b in group),
                split_structure=any(b.split for b in group),
            )
        )
    return chunks


def budget_for_model(requested: int) -> int:
    return min(requested, MODEL_MAX_TOKENS - MODEL_RESERVED_TOKENS - MAX_CONTEXT_TOKENS - 2)
