"""FastAPI application."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import asdict
from typing import Any

import openai
from fastapi import APIRouter, FastAPI, HTTPException, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from nsu_rag_llm import __version__
from nsu_rag_llm.api.deps import AskingDep, LlmProbeDep, QdrantProbeDep
from nsu_rag_llm.api.schemas import HealthResponse
from nsu_rag_llm.config import configure_logging, get_settings
from nsu_rag_llm.generation.answer import Event, Result

log = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Report on every external dependency",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
)
async def health(
    response: Response,
    qdrant_probe: QdrantProbeDep,
    llm_probe: LlmProbeDep,
) -> HealthResponse:
    """Probe Qdrant and the LLM endpoint concurrently.

    Answers 503 when anything is down, so that a load balancer or `docker
    compose` healthcheck can act on it, while the body still says which part
    broke and how long each probe took.
    """
    qdrant, llm = await asyncio.gather(qdrant_probe(), llm_probe())
    healthy = qdrant.status == "ok" and llm.status == "ok"
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if healthy else "degraded",
        qdrant=qdrant,
        llm=llm,
    )


@router.post(
    "/ask",
    response_model=Result,
    summary="Answer a question from the handbooks, with citations",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Empty question or unknown course"},
        status.HTTP_502_BAD_GATEWAY: {"description": "The LLM endpoint failed"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Models or index not loaded"},
    },
)
async def ask(asking: AskingDep) -> Result:
    """Search, rerank, and answer with `[course · «document» · p. N]` after each claim.

    `sources` lists every fragment the model read, labelled as in the prompt;
    `cited` says which of them the answer cites; `citation_errors` lists
    citations of fragments that do not exist.
    """
    try:
        return await asking.answerer.ask(asking.question, asking.course)
    except openai.APIError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"LLM: {type(exc).__name__}: {exc}"
        ) from exc


def _sse(event: Event) -> ServerSentEvent:
    payload: Any
    if event.kind == "sources":
        payload = [asdict(source) for source in event.data]
    elif event.kind == "token":
        payload = {"text": event.data}
    else:
        payload = event.data.summary()
    # Serialised here rather than by FastAPI, to keep Cyrillic readable on the
    # wire: `curl -N` shows the answer, not a wall of \u escapes.
    return ServerSentEvent(event=event.kind, raw_data=json.dumps(payload, ensure_ascii=False))


@router.post(
    "/ask/stream",
    response_class=EventSourceResponse,
    summary="The same answer, streamed as Server-Sent Events",
)
async def ask_stream(asking: AskingDep) -> AsyncIterator[ServerSentEvent]:
    """Events, in order: `sources` (the fragments, before the model starts),
    `token` (`{"text": ...}`, pieces of the answer with citations already
    expanded), `done` (everything `/ask` returns except the sources), or
    `error` if something failed once the stream had begun.
    """
    try:
        async with aclosing(asking.answerer.events(asking.question, asking.course)) as events:
            async for event in events:
                yield _sse(event)
    except Exception as exc:  # the 200 is already sent: the error goes in the stream
        log.exception("answer stream failed")
        detail = {"detail": f"{type(exc).__name__}: {exc}"}
        yield ServerSentEvent(event="error", raw_data=json.dumps(detail, ensure_ascii=False))


def create_app() -> FastAPI:
    """Build the application. A factory keeps tests free of import-time state."""
    configure_logging(get_settings().log_level)
    app = FastAPI(
        title="NSU RAG LLM",
        version=__version__,
        summary="Question answering over NSU FIT course materials.",
    )
    app.include_router(router)
    app.state.answerer = None
    app.state.agent = None
    app.state.answerer_lock = asyncio.Lock()
    return app


app = create_app()
