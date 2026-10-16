"""What the OpenAI-compatible endpoint cannot tell: Ollama's native API.

The model comparison needs two things from it: how much memory a model takes
while it answers (`GET /api/ps`), and a clean start for every model - the
others unloaded (`POST /api/generate` with `keep_alive: 0` and no prompt), so
that one model's speed is not measured with another still resident.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class Memory(Protocol):
    def size_of(self, model: str) -> int | None: ...

    def unload_all(self) -> None: ...


class Ollama:
    def __init__(self, api_url: str, timeout: float = 30.0) -> None:
        self.client = httpx.Client(base_url=api_url, timeout=timeout)

    def loaded(self) -> list[dict]:
        response = self.client.get("/api/ps")
        response.raise_for_status()
        return response.json().get("models") or []

    def size_of(self, model: str) -> int | None:
        """Bytes the model takes now - weights, KV cache, buffers - or None."""
        names = {model, f"{model}:latest"}
        for loaded in self.loaded():
            if loaded.get("name") in names or loaded.get("model") in names:
                return loaded.get("size")
        return None

    def unload_all(self) -> None:
        for loaded in self.loaded():
            response = self.client.post(
                "/api/generate", json={"model": loaded["name"], "keep_alive": 0}
            )
            response.raise_for_status()
