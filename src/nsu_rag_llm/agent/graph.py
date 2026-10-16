"""The agent (phase 5): the phase 3 pipeline, with a decision at every step.

    route ──► retrieve ──► strong enough? ── yes ──► generate ──► END
                 ▲              │ no                 (tools)
                 │              ├── tries left ──► rewrite ──┐
                 │              └── none left ───► refuse ──► END
                 └────────────────────────────────────────────┘

* **route** - the model names the course the question belongs to, from a
  JSON enum of the indexed courses plus "none"; the search is then filtered
  to it. A course the user gave is kept as given.
* **retrieve** - `search_course_materials`: hybrid search, reranked, exactly
  the phase 3 retrieval.
* **the check** - the reranker's best score. The cross-encoder reads question
  and passage together, so a low best score means nothing retrieved speaks to
  the question (ADR-031).
* **rewrite** - the model rephrases the question as a search query. The
  second rewrite also drops the router's filter: the router may have been
  wrong.
* **refuse** - "В методичках этого нет.", with no generation at all.
* **generate** - the phase 3 prompt and citations, plus two tools the model
  may call: `python_calc`, and `search_course_materials` for what the first
  search missed; fragments a tool finds get labels of their own.

The nodes are plain Python over this project's own retrieval and model
client; LangGraph only holds the state and the edges (ADR-031).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import TypedDict

from nsu_rag_llm.agent.calc import python_calc
from nsu_rag_llm.agent.llm import ToolCall, ToolModel
from nsu_rag_llm.config import Settings
from nsu_rag_llm.generation.answer import Answerer, Event, Result, Source, Timings
from nsu_rag_llm.generation.budget import fit
from nsu_rag_llm.generation.citations import Fragment, render
from nsu_rag_llm.generation.prompt import (
    REFUSAL,
    SYSTEM,
    fragment_block,
    is_refusal,
    user_message,
)
from nsu_rag_llm.retrieval.base import Hit

NO_COURSE = "none"

ROUTER_SYSTEM = """Определи, к какому курсу относится вопрос студента.

Курсы:
{listing}

Если вопрос не относится ни к одному курсу или относится к нескольким сразу, выбери «none». Ответь только JSON-объектом."""  # noqa: E501 - prose, not code

REWRITE_SYSTEM = """Перепиши вопрос студента в поисковый запрос к методичкам: раскрой сокращения, добавь термины и синонимы, которыми это называют в учебниках. Не отвечай на вопрос. Ответь только JSON-объектом."""  # noqa: E501 - prose, not code

AGENT_SYSTEM = (
    SYSTEM
    + """
6. Если для ответа нужно посчитать число, вызови python_calc, а не считай в уме.
7. Если во фрагментах не хватает сведений, вызови search_course_materials с другим запросом. Найденные фрагменты получат свои метки, на них тоже можно ссылаться."""  # noqa: E501 - prose, not code
)


def tools(courses: list[str]) -> list[dict]:
    """The tools, described as the model sees them (OpenAI function format)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "python_calc",
                "description": (
                    "Вычисляет арифметическое выражение: числа, + - * / // % **, скобки, "
                    "sqrt, exp, log, sin, cos, factorial, comb, perm, pi, e. Возвращает число."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Например: comb(10, 3) * 0.3**3 * 0.7**7",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_course_materials",
                "description": (
                    "Ищет в методичках фрагменты по запросу и возвращает новые фрагменты "
                    "с метками для ссылок."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Поисковый запрос."},
                        "course": {
                            "type": "string",
                            "enum": courses,
                            "description": "Искать только в этом курсе.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


class State(TypedDict, total=False):
    question: str
    given_course: str | None
    """A course the user asked for: never overridden."""
    course: str | None
    """The filter the next search uses."""
    routed: str | None
    query: str
    tries: int
    hits: list[Hit]
    best: float | None
    result: Result
    steps: list[str]
    timings: Timings


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def _schema(field: str, values: dict) -> dict:
    return {
        "type": "object",
        "properties": {field: values},
        "required": [field],
        "additionalProperties": False,
    }


class Agent:
    def __init__(
        self, *, answerer: Answerer, model: ToolModel, titles: dict[str, str], settings: Settings
    ) -> None:
        self.answerer = answerer
        self.model = model
        self.titles = {slug: titles.get(slug, slug) for slug in sorted(answerer.courses)}
        self.settings = settings
        self.tools = tools(list(self.titles))
        self.graph = self._build()

    def _build(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(State)
        graph.add_node("route", self.route)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("rewrite", self.rewrite)
        graph.add_node("generate", self.generate)
        graph.add_node("refuse", self.refuse)
        graph.add_edge(START, "route")
        graph.add_edge("route", "retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self.decide,
            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    async def ask(self, question: str, course: str | None = None) -> Result:
        started = time.perf_counter()
        question = question.strip()
        state = await self.graph.ainvoke(
            {
                "question": question,
                "given_course": course,
                "course": course,
                "routed": None,
                "query": question,
                "tries": 0,
                "steps": [],
                "timings": Timings(),
            }
        )
        result = state["result"]
        # No streaming here: the reader sees the answer when it is complete.
        result.timings.total_ms = result.timings.first_token_ms = _ms(started)
        return result

    async def events(self, question: str, course: str | None = None) -> AsyncIterator[Event]:
        """The agent's answer as the pipeline's events: it cannot stream tokens
        (a reply may be a tool call), so the answer comes as one token."""
        result = await self.ask(question, course)
        yield Event("sources", result.sources)
        yield Event("token", result.answer)
        yield Event("done", result)

    # --- nodes ------------------------------------------------------------------

    async def route(self, state: State) -> dict:
        if state.get("given_course"):
            return {"steps": [*state["steps"], f"курс задан: {state['given_course']}"]}
        started = time.perf_counter()
        listing = "\n".join(f"- {slug}: {title}" for slug, title in self.titles.items())
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM.format(listing=listing)},
            {"role": "user", "content": state["question"]},
        ]
        schema = _schema("course", {"type": "string", "enum": [*self.titles, NO_COURSE]})
        try:
            reply = await self.model.reply(messages, schema=("route", schema), max_tokens=64)
            picked = json.loads(reply.content)["course"]
        except json.JSONDecodeError, KeyError, TypeError:
            picked = NO_COURSE
        course = picked if picked in self.titles else None
        step = f"маршрут: {course or 'без фильтра'} ({_ms(started):.0f} мс)"
        # "none" is kept as said: the evaluation scores the router on it.
        routed = course or NO_COURSE
        return {"course": course, "routed": routed, "steps": [*state["steps"], step]}

    async def retrieve(self, state: State) -> dict:
        timings = state["timings"]
        stage = Timings()
        hits = await asyncio.to_thread(
            self.answerer.retrieve, state["query"], state.get("course"), stage
        )
        timings.search_ms += stage.search_ms
        timings.rerank_ms += stage.rerank_ms
        best = hits[0].score if hits else None
        where = state.get("course") or "все курсы"
        score = "—" if best is None else f"{best:.3f}"
        step = f"поиск ({where}, попытка {state['tries'] + 1}): лучший скор {score}"
        return {"hits": hits, "best": best, "steps": [*state["steps"], step]}

    def decide(self, state: State) -> str:
        if not state.get("hits"):
            return "rewrite" if state["tries"] < self.settings.agent_max_rewrites else "refuse"
        if self.answerer.rerank is None:
            # Without the cross-encoder there is no score that means anything
            # on its own: an RRF score only ranks. Answer from what was found.
            return "generate"
        if (state.get("best") or 0.0) >= self.settings.agent_min_score:
            return "generate"
        if state["tries"] < self.settings.agent_max_rewrites:
            return "rewrite"
        return "refuse"

    async def rewrite(self, state: State) -> dict:
        schema = _schema("query", {"type": "string"})
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": state["question"]},
        ]
        try:
            reply = await self.model.reply(messages, schema=("rewrite", schema), max_tokens=128)
            query = str(json.loads(reply.content)["query"]).strip() or state["question"]
        except json.JSONDecodeError, KeyError, TypeError:
            query = state["question"]
        tries = state["tries"] + 1
        course = state.get("course")
        # The second try searches everything: the router may have been wrong.
        if tries >= 2 and not state.get("given_course"):
            course = None
        step = f"переформулировка {tries}" + (" без фильтра курса" if course is None else "")
        return {"query": query, "tries": tries, "course": course, "steps": [*state["steps"], step]}

    async def refuse(self, state: State) -> dict:
        steps = [*state["steps"], "отказ: ничего достаточно близкого к вопросу"]
        return {"result": self._result(state, REFUSAL, [], None, steps=steps, dropped=0)}

    async def generate(self, state: State) -> dict:
        started = time.perf_counter()
        s = self.settings
        reserve = s.agent_tool_tokens
        fitted = fit(
            state["question"],
            state["hits"][: s.answer_fragments],
            self.answerer.count,
            s.llm_context_tokens,
            s.llm_max_output_tokens + reserve,
        )
        fragments = list(fitted.fragments)
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_message(state["question"], fragments)},
        ]
        steps = list(state["steps"])
        budget = {"tokens": reserve}
        reply = None
        for round_ in range(s.agent_tool_rounds + 1):
            last = round_ == s.agent_tool_rounds
            reply = await self.model.reply(messages, tools=None if last else self.tools)
            if not reply.tool_calls:
                break
            messages.append(reply.as_message())
            for call in reply.tool_calls:
                output = await self._tool(call, fragments, budget, state)
                steps.append(f"инструмент {call.name}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        assert reply is not None
        # The answer arrives whole, after every tool round.
        state["timings"].llm_first_output_ms = state["timings"].llm_first_token_ms = _ms(started)
        return {
            "result": self._result(
                state,
                reply.content.strip(),
                fragments,
                reply,
                steps=steps,
                dropped=len(fitted.dropped),
                estimated=fitted.prompt_tokens,
            )
        }

    # --- helpers ----------------------------------------------------------------

    async def _tool(self, call: ToolCall, fragments: list[Fragment], budget: dict, state) -> str:
        if call.name == "python_calc":
            return python_calc(str(call.arguments.get("expression", "")))
        if call.name != "search_course_materials":
            return f"Нет такого инструмента: {call.name}"
        query = str(call.arguments.get("query", "")).strip()
        if not query:
            return "Ошибка: пустой запрос."
        course = call.arguments.get("course")
        course = course if course in self.titles else None
        stage = Timings()
        hits = await asyncio.to_thread(self.answerer.retrieve, query, course, stage)
        state["timings"].search_ms += stage.search_ms
        state["timings"].rerank_ms += stage.rerank_ms
        seen = {f.chunk["id"] for f in fragments}
        blocks = []
        for hit in hits[: self.settings.agent_tool_fragments]:
            if hit.id in seen:
                continue
            fragment = Fragment(len(fragments) + 1, hit.chunk)
            block = fragment_block(fragment)
            cost = self.answerer.count(block) + 2
            if cost > budget["tokens"]:
                break  # the window is full: better fewer fragments than a cut prompt
            budget["tokens"] -= cost
            fragments.append(fragment)
            blocks.append(block)
        return "\n\n".join(blocks) or "Новых фрагментов не найдено или они не помещаются."

    def _result(
        self,
        state: State,
        raw: str,
        fragments: list[Fragment],
        reply,
        *,
        steps: list[str],
        dropped: int,
        estimated: int = 0,
    ) -> Result:
        rendered = render(raw, {f.number: f for f in fragments})
        refused = is_refusal(raw)
        errors = list(rendered.errors)
        if not refused and not rendered.cited and not errors:
            errors.append("в ответе нет ни одной ссылки")
        timings = state["timings"]
        return Result(
            answer=rendered.text,
            raw_answer=raw,
            refused=refused,
            sources=[Source.of(f) for f in fragments],
            cited=rendered.cited,
            citation_errors=errors,
            truncated=False,
            dropped=dropped,
            prompt_tokens_estimated=estimated,
            prompt_tokens=reply.prompt_tokens if reply else None,
            completion_tokens=reply.completion_tokens if reply else None,
            mode="agent",
            course=state.get("course"),
            timings=timings,
            routed_course=state.get("routed"),
            steps=steps,
        )
