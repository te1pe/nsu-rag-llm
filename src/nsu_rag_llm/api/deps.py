"""Dependency wiring for the HTTP API.

The health probes are handed to the endpoint as *callables* rather than as
already-awaited results. FastAPI resolves dependencies one after another, so
returning results here would serialise the probes; returning callables lets the
endpoint run them concurrently. It also gives tests a single seam: override the
factory, get a fake probe.

The answering pipeline is loaded on the first question, not at start-up: its
models take seconds to load and gigabytes of memory, and `/health` must answer
at once, including in a container that has no models at all.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from nsu_rag_llm.api.schemas import AskRequest, ComponentHealth
from nsu_rag_llm.config import Settings, get_settings
from nsu_rag_llm.generation.answer import Answerer

SettingsDep = Annotated[Settings, Depends(get_settings)]

#: A probe reports on one dependency and never raises.
Probe = Callable[[], Awaitable[ComponentHealth]]

#: Work that either returns a human-readable detail line or raises.
Check = Callable[[], Awaitable[str]]


async def measure(check: Check, timeout: float) -> ComponentHealth:
    """Run `check` under a deadline, turning any failure into a report."""
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 1)

    try:
        async with asyncio.timeout(timeout):
            detail = await check()
    except TimeoutError:
        return ComponentHealth(
            status="error",
            latency_ms=elapsed_ms(),
            detail=f"timed out after {timeout:g}s",
        )
    except Exception as exc:  # noqa: BLE001 - a probe must survive anything
        return ComponentHealth(
            status="error",
            latency_ms=elapsed_ms(),
            detail=f"{type(exc).__name__}: {exc}",
        )
    return ComponentHealth(status="ok", latency_ms=elapsed_ms(), detail=detail)


def get_qdrant_probe(settings: SettingsDep) -> Probe:
    """Build a probe that asks Qdrant to list its collections."""

    async def check() -> str:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=int(settings.health_timeout_seconds),
        )
        try:
            collections = await client.get_collections()
        finally:
            await client.close()
        names = sorted(c.name for c in collections.collections)
        return f"{len(names)} collection(s): {', '.join(names) or '-'}"

    async def probe() -> ComponentHealth:
        return await measure(check, settings.health_timeout_seconds)

    return probe


def get_llm_probe(settings: SettingsDep) -> Probe:
    """Build a probe that checks the configured model is actually served."""

    async def check() -> str:
        client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.health_timeout_seconds,
            max_retries=0,
        )
        try:
            served = [model.id async for model in client.models.list()]
        finally:
            await client.close()
        # Ollama reports tags, so a model configured as "qwen3.5-9b-8k" is
        # served under the id "qwen3.5-9b-8k:latest".
        wanted = settings.llm_model
        if wanted in served or f"{wanted}:latest" in served:
            return f"model '{wanted}' is served"
        raise RuntimeError(
            f"model '{wanted}' not among {len(served)} served model(s): "
            f"{', '.join(sorted(served)) or '-'}"
        )

    async def probe() -> ComponentHealth:
        return await measure(check, settings.health_timeout_seconds)

    return probe


QdrantProbeDep = Annotated[Probe, Depends(get_qdrant_probe)]
LlmProbeDep = Annotated[Probe, Depends(get_llm_probe)]


def load_answerer(settings: Settings) -> Answerer:
    answerer = Answerer.from_settings(settings)
    answerer.warm_up()
    return answerer


async def loaded_answerer(request: Request, settings: Settings) -> Answerer:
    """The process-wide pipeline, loaded by whichever question comes first."""
    state = request.app.state
    async with state.answerer_lock:
        if state.answerer is None:
            try:
                state.answerer = await asyncio.to_thread(load_answerer, settings)
            except Exception as exc:
                # Not cached: the next question tries again, e.g. once Qdrant is up.
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    f"поиск не готов: {type(exc).__name__}: {exc}",
                ) from exc
    return state.answerer


def loaded_agent(request: Request, settings: Settings, answerer: Answerer):
    """The agent, built around the loaded pipeline: it shares its models."""
    state = request.app.state
    if state.agent is None:
        from nsu_rag_llm.agent.graph import Agent
        from nsu_rag_llm.agent.llm import OpenAICompatibleTools
        from nsu_rag_llm.ingest.build import course_titles

        state.agent = Agent(
            answerer=answerer,
            model=OpenAICompatibleTools(settings),
            titles=course_titles(settings.courses_file),
            settings=settings,
        )
    return state.agent


@dataclass(frozen=True)
class Asking:
    """A question that passed every check, and what will answer it."""

    question: str
    course: str | None
    answerer: Answerer
    """The phase 3 pipeline - or the agent, which answers through the same calls."""


async def checked_request(request: Request, settings: SettingsDep, body: AskRequest) -> Asking:
    """Validate the question first, then load the pipeline, then check the course.

    The order is the point. FastAPI calls a dependency's own dependencies
    before it validates the request body, so the pipeline is loaded here, in
    the body of this function - which FastAPI calls only once `body` is valid.
    An empty question is a 422 at once, not after 20 s of loading models, nor
    a 503 when they cannot be loaded.

    And all of it happens in a dependency because a streaming response has
    sent its 200 by the time the endpoint runs: a refusal here is still a
    proper status code.
    """
    answerer = await loaded_answerer(request, settings)
    if body.course is not None and body.course not in answerer.courses:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"курса {body.course!r} нет в индексе; есть: {', '.join(sorted(answerer.courses))}",
        )
    system = loaded_agent(request, settings, answerer) if body.agent else answerer
    return Asking(body.question, body.course, system)


AskingDep = Annotated[Asking, Depends(checked_request)]
