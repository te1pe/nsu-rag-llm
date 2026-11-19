from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from nsu_rag_llm import jsonl

CONTEXT = 70


@dataclass(frozen=True)
class Match:
    file: str
    """Relative to data/raw/ - exactly what a gold entry's `file` takes."""
    page: int
    origin: str
    indexes: tuple[str, ...]
    """Which indexes hold the document: "mvp", "full"."""
    context: str


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def indexed_documents(chunks_file: Path) -> set[str]:
    if not chunks_file.exists():
        return set()
    with chunks_file.open(encoding="utf-8") as handle:
        return {json.loads(line)["doc_sha256"] for line in handle if line.strip()}


def relative_source(source: str, raw_dir: Path) -> str:
    path = Path(source)
    try:
        return str(path.relative_to(raw_dir))
    except ValueError:
        return str(path)


def find(
    phrase: str,
    parsed_dir: Path,
    raw_dir: Path,
    indexes: dict[str, set[str]],
    course: str | None = None,
    limit: int = 20,
) -> list[Match]:
    wanted = normalise(phrase)
    if not wanted:
        return []
    matches: list[Match] = []
    for folder in sorted(parsed_dir.iterdir()) if parsed_dir.exists() else []:
        meta_path, pages_path = folder / "meta.json", folder / "pages.jsonl"
        if not meta_path.exists() or not pages_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        file = relative_source(meta["source"], raw_dir)
        if course and file.split("/")[0] != course:
            continue
        held = tuple(name for name, shas in indexes.items() if folder.name in shas)
        for page in jsonl.read(pages_path):
            text = normalise(page["markdown"])
            at = text.find(wanted)
            if at < 0:
                continue
            start, end = max(0, at - CONTEXT), at + len(wanted) + CONTEXT
            matches.append(
                Match(file, page["number"], page.get("origin", "?"), held, text[start:end])
            )
            if len(matches) >= limit:
                return matches
    return matches
