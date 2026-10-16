"""Split cleaned pages into retrieval chunks.

A chunk is what the retriever returns and what the model reads, so it must be
small enough to embed whole (multilingual-e5-base reads 512 tokens) and whole
enough to make sense on its own. The rules, measured on the MVP (ADR-020):

* **Headings prefer to start a chunk, but do not force one.** Heading levels
  in this corpus are relative to each book - one uses only level 1, another
  puts 1 605 run-in labels ("Объявление", "Результат") at level 6 - and 18% of
  sections are under 50 tokens. So a heading starts a new chunk only once the
  current one holds a reasonable amount; small sections are packed together.
* **A chunk's section is the common ancestor** of the headings its text sits
  under, so a chunk holding three small subsections is labelled with the
  section that contains all three.
* **Long sections are split between paragraphs**, up to `budget` tokens, with
  about `overlap` tokens of the previous chunk's closing sentences carried
  over, so that a sentence cut off at a boundary is still read in full once.
* **Listings and tables are not cut through the middle.** One that fits goes
  whole. One that cannot fit - 76 listings and 257 tables exceed the budget,
  one table runs to 5 656 tokens - is split where its structure allows it: a
  listing at blank lines between its parts, else between lines, never inside
  one; a table between rows, with its header repeated on every piece.
* **A paragraph that runs across a page break is joined back together** - 28%
  of pages start mid-sentence.

Counting is done by a function passed in, not by a tokenizer this module
loads: in production it is the embedding model's own tokenizer, in tests a
fake, so this module needs no model and no network.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from nsu_rag_llm.ingest.cleaning import PLACEHOLDERS, split_fenced
from nsu_rag_llm.ingest.inventory import text_stats

TokenCounter = Callable[[str], int]

#: Separator of the "course › document › section" context line.
SEPARATOR = " › "
#: A heading starts a new chunk only if the current one already holds at least
#: this many tokens; below it, the small section is packed with the next one.
MIN_BEFORE_BREAK = 100
#: The context line is capped so that it cannot crowd out the text it labels;
#: past this, the outermost section titles are dropped first.
MAX_CONTEXT_TOKENS = 64
#: multilingual-e5-base reads 512 tokens; two are its special tokens and the
#: "passage: " prefix costs a few more.
MODEL_MAX_TOKENS = 512
MODEL_RESERVED_TOKENS = 8

HEADING = re.compile(r"^(#{1,6}) +(.+?)\s*$")
#: The layout model occasionally marks a whole bold paragraph as a heading - one
#: ran to 884 characters. Real headings in the MVP are a few words long.
MAX_HEADING_CHARS = 150
TABLE_LINE = re.compile(r"^\s*\|")
TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[A-ZА-ЯЁ0-9])")


@dataclass
class Block:
    """An indivisible piece of a page, as far as packing is concerned."""

    kind: str  # "heading" | "text" | "code" | "table"
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
    """What the chunker needs to know about the document it is splitting."""

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
        """What gets embedded: where the text sits, then the text itself."""
        return f"{self.context}\n\n{self.text}"

    def to_json(self) -> dict:
        return dict(self.__dict__)


def clean_title(markdown: str) -> str:
    title = re.sub(r"[*_`]", "", markdown)
    title = re.sub(r"\s+", " ", title).strip(" .:")
    return title[:120]


def _prose_blocks(segment: str, page: int) -> list[Block]:
    """Headings, paragraphs and tables of a stretch of text outside code."""
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
    """Headings, paragraphs, listings and tables of one page, in order."""
    blocks: list[Block] = []
    for is_code, segment in split_fenced(markdown):
        if not is_code:
            blocks.extend(_prose_blocks(segment, page))
        elif segment.strip():
            blocks.append(Block("code", segment.strip(), page, page))
    return blocks


def join_across_pages(pages: Iterable[tuple[int, str]]) -> list[Block]:
    """All blocks of a document, with paragraphs broken by a page re-joined.

    A page whose first block is prose starting in lower case continues the
    previous page's last paragraph. Only prose is joined: a listing or table
    that continues is left as two blocks, each still whole in itself.
    """
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
    """Record, on every block, the headings it sits under."""
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
    """Greedily group pieces into runs of at most `budget` tokens."""
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
    """Cut a piece that has no structure left to cut along.

    The last resort, for things like a 629-character table cell without a
    single space: find the longest prefix that fits, back off to a space if
    there is one, repeat.
    """
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
    """Hard-split any piece that is still larger than the budget."""
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
    """Between the parts of a listing, then between lines - never inside one."""
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
    """Between rows, with the header row (and its rule) repeated on each piece."""
    lines = text.split("\n")
    header = lines[:2] if len(lines) > 1 and TABLE_RULE.match(lines[1]) else lines[:1]
    rows = lines[len(header) :]
    if count("\n".join(header)) > budget // 2:
        # A header this large would leave no room for rows; the pieces go
        # without it rather than repeating half a chunk of header each time.
        rows, header = lines, []
    room = budget - count("\n".join(header)) if header else budget
    groups = _pack(_fit(rows, room, count), room, count, "\n")
    return ["\n".join([*header, group]) for group in groups]


def split_block(block: Block, budget: int, count: TokenCounter) -> list[Block]:
    """A block that cannot fit in one chunk, cut where its structure allows."""
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
    """The closing sentences of a text, up to about `tokens` tokens.

    Returned as an exact suffix of `text` - line breaks and all - so that the
    overlap is literally the end of the previous chunk, and can be checked as
    such. Whole sentences when they fit. When the last sentence alone is
    longer - common, since Russian academic sentences run long - its closing
    words are carried instead, marked with an ellipsis as the fragment they are.
    """
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
    """A marker or a couple of words is no context for the next chunk."""
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
    """ "course › document › section", with outer sections dropped if too long."""
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
    """Identity of a chunk: the model, the document's bytes, where and what.

    Derived from content, so re-running the pipeline gives unchanged chunks
    the same id and indexing can skip them; a change to the text, the context
    line or the embedding model gives a new id.
    """
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
    """Turn one document's cleaned pages into chunks."""
    budget = budget_for_model(budget)
    blocks = join_across_pages(pages)
    assign_sections(blocks)

    groups: list[list[Block]] = []
    current: list[Block] = []

    def size(items: list[Block]) -> int:
        # Summed per block rather than counted on the joined text: packing asks
        # this once per candidate block, and re-tokenising the whole chunk each
        # time made chunking quadratic. The sum ignores the few tokens of the
        # separators; the exact count is taken once per finished chunk, and
        # the budget leaves room inside the model's window for the difference.
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
        # Headings at the very end belong to what follows them, not to this
        # chunk - but only when there is content to leave behind; a run of
        # nothing but headings is emitted as it is once it is large enough.
        trailing: list[Block] = []
        if has_content(current):
            while current and current[-1].kind == "heading":
                trailing.insert(0, current.pop())
        if has_content(current) or size(current) >= budget // 2:
            groups.append(current)
            carried: list[Block] = []
            # The closing prose of the chunk: the run of text blocks at its
            # end, so that a chunk ending "... the density is [формула]"
            # carries the sentence that introduces the formula, not the marker.
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
            # Overlap repeats the end of a section into its own continuation.
            # Carried text with nothing of its own after it would open the
            # next section instead, so it goes; but once the continuation has
            # started, the carried text is its context and stays.
            if all(b.carried for b in current):
                current = []
            if size(current) >= min_before_break:
                flush(with_overlap=False)
        # Prose is cut to leave room for the overlap the next chunk will carry;
        # cut to the full budget, carried text plus a piece would never fit,
        # and long paragraphs - the norm in a textbook - would lose all overlap.
        room = budget - overlap if block.kind == "text" else budget
        for piece in split_block(block, room, count):
            if (
                current
                and size([*current, piece]) > budget
                and (has_content(current) or size(current) >= budget // 2)
            ):
                flush(with_overlap=True)
                # The carried overlap plus a large piece may still not fit.
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
        if cid in seen:  # the same text, on the same page, of the same document
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
    """The largest text budget that keeps context line + text inside the model.

    The context line is capped at MAX_CONTEXT_TOKENS, so any budget up to this
    one fits whatever section a chunk ends up in.
    """
    return min(requested, MODEL_MAX_TOKENS - MODEL_RESERVED_TOKENS - MAX_CONTEXT_TOKENS - 2)
