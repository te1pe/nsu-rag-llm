from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Hit:
    chunk: dict
    score: float

    @property
    def id(self) -> str:
        return self.chunk["id"]

    @property
    def passage(self) -> str:
        return f"{self.chunk['context']}\n\n{self.chunk['text']}"


class Retriever(Protocol):
    def search(self, query: str, k: int, course: str | None = None) -> list[Hit]: ...
