"""Shared test fixtures.

Nothing here touches the network, a GPU, a running LLM or the real course
materials: the health probes are replaced wholesale via FastAPI's dependency
overrides.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nsu_rag_llm.api.deps import Probe, get_llm_probe, get_qdrant_probe
from nsu_rag_llm.api.main import create_app
from nsu_rag_llm.api.schemas import ComponentHealth


def fixed_probe(
    status: str = "ok",
    detail: str = "fake probe",
    delay: float = 0.0,
) -> Probe:
    """A probe that sleeps for `delay` and then reports `status`."""

    async def probe() -> ComponentHealth:
        if delay:
            await asyncio.sleep(delay)
        return ComponentHealth(status=status, latency_ms=delay * 1000, detail=detail)

    return probe


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def set_probes(app: FastAPI):
    """Install fake Qdrant and LLM probes for one test."""

    def install(qdrant: Probe, llm: Probe) -> None:
        app.dependency_overrides[get_qdrant_probe] = lambda: qdrant
        app.dependency_overrides[get_llm_probe] = lambda: llm

    yield install
    app.dependency_overrides.clear()
