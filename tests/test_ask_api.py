"""`POST /ask` and `POST /ask/stream`, with the pipeline built from fakes."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from nsu_rag_llm.api import deps
from tests.conftest import fixed_probe
from tests.test_generation import TWO, ScriptedChat, cite, make


def events(body: str) -> list[tuple[str, object]]:
    """Parse an SSE body into (event, decoded data) pairs, skipping comments."""
    parsed = []
    for block in body.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if not line.startswith(":")
        )
        if fields:
            parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


@pytest.fixture
def answer_with(app):
    """Install a pipeline built from fakes, as if the first question had loaded it."""

    def install(chat) -> None:
        app.state.answerer = make(TWO, chat)

    return install


def test_ask_returns_the_answer_its_sources_and_timings(client, answer_with) -> None:
    answer_with(ScriptedChat("Процесс — программа в работе [Ф1]."))

    response = client.post("/ask", json={"question": "Что такое процесс?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == f"Процесс — программа в работе {cite('3')}."
    assert [s["number"] for s in body["sources"]] == [1, 2]
    assert body["cited"] == [1]
    assert body["citation_errors"] == []
    assert set(body["timings"]) >= {"search_ms", "rerank_ms", "first_token_ms", "total_ms"}


def test_the_stream_sends_sources_then_tokens_then_done(client, answer_with) -> None:
    answer_with(ScriptedChat("Процесс ", "[Ф", "1]", " и поток [Ф2]."))

    response = client.post("/ask/stream", json={"question": "Что такое процесс?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response.text)
    kinds = [kind for kind, _ in stream]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"}
    done = stream[-1][1]
    assert "".join(data["text"] for kind, data in stream if kind == "token") == done["answer"]
    assert "sources" not in done  # already sent first
    assert done["cited"] == [1, 2]
    # Cyrillic stays readable on the wire, not \\u-escaped.
    assert "Операционные системы" in response.text


@pytest.mark.parametrize(
    "body", [{"question": "   "}, {"question": ""}, {}, {"question": "x" * 2001}]
)
def test_a_missing_empty_or_huge_question_is_rejected(client, answer_with, body) -> None:
    answer_with(ScriptedChat("[Ф1]"))

    assert client.post("/ask", json=body).status_code == 422


def test_an_unknown_course_is_rejected_before_the_stream_starts(client, answer_with) -> None:
    answer_with(ScriptedChat("[Ф1]"))

    response = client.post("/ask/stream", json={"question": "?", "course": "chemistry"})

    assert response.status_code == 422
    assert "os" in response.json()["detail"]


def unreachable() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://llm/v1/chat"))


class DeadChat:
    async def stream(self, messages):
        raise unreachable()
        yield  # an async generator, like the real one


def test_an_unreachable_llm_is_a_502(client, answer_with) -> None:
    answer_with(DeadChat())

    response = client.post("/ask", json={"question": "?"})

    assert response.status_code == 502
    assert "APIConnectionError" in response.json()["detail"]


def test_an_llm_failure_mid_stream_ends_it_with_an_error_event(client, answer_with) -> None:
    answer_with(DeadChat())

    response = client.post("/ask/stream", json={"question": "?"})

    stream = events(response.text)
    assert stream[0][0] == "sources"
    assert stream[-1][0] == "error"
    assert "APIConnectionError" in stream[-1][1]["detail"]


def test_an_invalid_question_is_rejected_without_loading_the_pipeline(client, monkeypatch) -> None:
    """Regression: FastAPI resolves a dependency's own dependencies before it
    validates the body, so an empty question used to wait for the models to
    load - and would have been a 503, not a 422, had they failed to."""
    loads = []

    def load(settings):
        loads.append(settings)
        raise RuntimeError("the models must not be loaded for an invalid request")

    monkeypatch.setattr(deps, "load_answerer", load)

    for path in ("/ask", "/ask/stream"):
        assert client.post(path, json={"question": "  "}).status_code == 422
    assert loads == []


def test_the_pipeline_loads_on_the_first_question_and_only_once(
    client, monkeypatch, set_probes
) -> None:
    loads = []

    def load(settings):
        loads.append(settings)
        return make(TWO, ScriptedChat("[Ф1]"))

    monkeypatch.setattr(deps, "load_answerer", load)
    # Fake probes: the real ones would reach for Qdrant and Ollama on localhost.
    set_probes(fixed_probe(), fixed_probe())

    assert client.get("/health").status_code == 200
    assert loads == []
    client.post("/ask", json={"question": "?"})
    client.post("/ask", json={"question": "?"})
    assert len(loads) == 1


def test_a_failed_load_is_a_503_and_is_retried(client, monkeypatch) -> None:
    attempts = []

    def load(settings):
        attempts.append(1)
        if len(attempts) == 1:
            raise FileNotFoundError("data/chunks.jsonl")
        return make(TWO, ScriptedChat("[Ф1]"))

    monkeypatch.setattr(deps, "load_answerer", load)

    first = client.post("/ask", json={"question": "?"})
    second = client.post("/ask", json={"question": "?"})

    assert first.status_code == 503
    assert "FileNotFoundError" in first.json()["detail"]
    assert second.status_code == 200
