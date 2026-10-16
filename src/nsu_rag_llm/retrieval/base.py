"""What every retriever returns, and the one filter they all support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Hit:
    """One retrieved chunk and the score its retriever gave it.

    `chunk` holds the fields written at ingest time (`data/chunks.jsonl`, and
    the same as the Qdrant payload): id, course, file, doc_sha256, page_start,
    page_end, section_path, doc_title, context, text, origin.
    """

    chunk: dict
    score: float

    @property
    def id(self) -> str:
        return self.chunk["id"]

    @property
    def passage(self) -> str:
        """The text as it was embedded: context line, then the chunk."""
        return f"{self.chunk['context']}\n\n{self.chunk['text']}"


class Retriever(Protocol):
    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]: ...
