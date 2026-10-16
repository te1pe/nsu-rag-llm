"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

ComponentStatus = Literal["ok", "error"]


class ComponentHealth(BaseModel):
    """Outcome of probing one dependency."""

    status: ComponentStatus
    latency_ms: float = Field(description="Wall-clock time the probe took.")
    detail: str | None = Field(
        default=None,
        description="What was found, or why the probe failed.",
    )


class HealthResponse(BaseModel):
    """Aggregate health. `degraded` means at least one dependency is down."""

    status: Literal["ok", "degraded"]
    qdrant: ComponentHealth
    llm: ComponentHealth


class AskRequest(BaseModel):
    """A question, optionally confined to one course."""

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
        Field(
            description="The question, in Russian.",
            examples=["Чем процесс отличается от потока?"],
        ),
    ]
    course: str | None = Field(
        default=None,
        description="A course slug, e.g. `os`: search only that course's handbooks.",
        examples=["os"],
    )
    agent: bool = Field(
        default=False,
        description="Answer with the agent (phase 5): routing, re-querying, tools.",
    )
