"""The agent (phase 5): every branch of the graph, with a scripted fake model.

The retrieval is the real Answerer.retrieve over a fake search and a fake
reranker whose scores the test sets, so the graph's decisions - generate,
rewrite, refuse - are driven exactly as the real cross-encoder would drive
them.
"""

from __future__ import annotations

import json

import pytest

from nsu_rag_llm.agent.calc import calculate, python_calc
from nsu_rag_llm.agent.graph import Agent
from nsu_rag_llm.agent.llm import Reply, ToolCall
from nsu_rag_llm.config import Settings
from nsu_rag_llm.generation.answer import Answerer
from nsu_rag_llm.generation.prompt import REFUSAL
from nsu_rag_llm.retrieval.base import Hit
from tests.test_generation import COURSE, ScriptedChat, chunk, words


class ScriptedModel:
    """A fake model: replies from a script; records every request."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def reply(self, messages, tools=None, schema=None, max_tokens=None) -> Reply:
        self.calls.append({"messages": messages, "tools": tools, "schema": schema})
        if not self.replies:
            raise AssertionError(f"unexpected request: {schema[0] if schema else 'answer'}")
        return self.replies.pop(0)


def json_reply(**fields) -> Reply:
    return Reply(content=json.dumps(fields, ensure_ascii=False))


class Library:
    """Search and rerank: what is found for a query, and how good it looks."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.searches: list[tuple[str, str | None]] = []

    def search(self, query: str, k: int, course: str | None) -> list[Hit]:
        self.searches.append((query, course))
        return [
            Hit(chunk(f"{query}-{n}", f"Текст о «{query}», часть {n}.", start=n), 0.0)
            for n in range(1, 4)
        ]

    def rerank(self, query: str, hits) -> list[Hit]:
        score = self.scores.get(query, 0.0)
        return [Hit(h.chunk, score - i * 0.01) for i, h in enumerate(hits)]


def agent(model: ScriptedModel, library: Library, rerank: bool = True, **settings) -> Agent:
    config = Settings(_env_file=None, **settings)
    answerer = Answerer(
        search=library.search,
        rerank=library.rerank if rerank else None,
        chat=ScriptedChat(),
        count=words,
        settings=config,
        mode="hybrid+rerank",
        courses=frozenset({"os", "matan"}),
    )
    return Agent(
        answerer=answerer,
        model=model,
        titles={"os": COURSE, "matan": "Математический анализ"},
        settings=config,
    )


QUESTION = "Что такое семафор?"


async def test_the_router_picks_the_course_and_the_search_is_filtered_to_it() -> None:
    model = ScriptedModel(json_reply(course="os"), Reply("Семафор — счётчик [Ф1]."))
    library = Library({QUESTION: 0.9})

    result = await agent(model, library).ask(QUESTION)

    assert library.searches == [(QUESTION, "os")]
    assert result.routed_course == "os" and result.course == "os"
    route = model.calls[0]["schema"]
    assert route[0] == "route"
    assert route[1]["properties"]["course"]["enum"] == ["matan", "os", "none"]
    assert result.cited == [1] and not result.refused


async def test_a_course_the_user_gave_is_kept_and_the_router_not_asked() -> None:
    model = ScriptedModel(Reply("Ответ [Ф1]."))
    library = Library({QUESTION: 0.9})

    result = await agent(model, library).ask(QUESTION, course="matan")

    assert library.searches == [(QUESTION, "matan")]
    assert len(model.calls) == 1  # only the answer
    assert result.routed_course is None


async def test_a_router_that_answers_nonsense_means_no_filter() -> None:
    model = ScriptedModel(Reply("os, наверное"), Reply("Ответ [Ф1]."))
    library = Library({QUESTION: 0.9})

    await agent(model, library).ask(QUESTION)

    assert library.searches == [(QUESTION, None)]


async def test_a_weak_search_is_rewritten_and_the_second_rewrite_drops_the_filter() -> None:
    model = ScriptedModel(
        json_reply(course="os"),
        json_reply(query="семафор синхронизация"),
        json_reply(query="семафор Дейкстры"),
        Reply("Ответ [Ф1]."),
    )
    library = Library({QUESTION: 0.01, "семафор синхронизация": 0.02, "семафор Дейкстры": 0.8})

    result = await agent(model, library).ask(QUESTION)

    assert library.searches == [
        (QUESTION, "os"),
        ("семафор синхронизация", "os"),
        ("семафор Дейкстры", None),
    ]
    assert not result.refused
    assert sum("переформулировка" in step for step in result.steps) == 2


async def test_when_nothing_is_close_enough_the_agent_refuses_without_generating() -> None:
    model = ScriptedModel(
        json_reply(course="none"), json_reply(query="запрос 1"), json_reply(query="запрос 2")
    )
    library = Library({})

    result = await agent(model, library).ask("Кто выиграл чемпионат мира?")

    assert result.refused and result.answer == REFUSAL
    assert result.sources == []
    assert all(call["schema"] for call in model.calls)  # no answer was ever requested
    assert len(library.searches) == 3


async def test_without_a_reranker_there_is_no_score_to_refuse_on() -> None:
    model = ScriptedModel(json_reply(course="os"), Reply("Ответ [Ф1]."))

    result = await agent(model, Library({}), rerank=False).ask(QUESTION)

    assert not result.refused


async def test_the_calculator_tool_is_run_and_its_result_fed_back() -> None:
    model = ScriptedModel(
        json_reply(course="matan"),
        Reply("", [ToolCall("c1", "python_calc", {"expression": "comb(10, 3)"})]),
        Reply("Всего 120 сочетаний [Ф1]."),
    )
    library = Library({QUESTION: 0.9})

    result = await agent(model, library).ask(QUESTION)

    answer_call = model.calls[-1]["messages"]
    assert answer_call[-1] == {"role": "tool", "tool_call_id": "c1", "content": "120"}
    assert answer_call[-2]["tool_calls"][0]["function"]["name"] == "python_calc"
    assert "инструмент python_calc" in result.steps
    assert result.answer.startswith("Всего 120 сочетаний")


async def test_fragments_a_tool_search_finds_get_labels_of_their_own() -> None:
    model = ScriptedModel(
        json_reply(course="os"),
        Reply("", [ToolCall("s1", "search_course_materials", {"query": "мьютекс"})]),
        Reply("Семафор [Ф1], мьютекс [Ф11]."),
    )
    library = Library({QUESTION: 0.9, "мьютекс": 0.9})

    result = await agent(model, library, answer_fragments=10).ask(QUESTION)

    tool_output = model.calls[-1]["messages"][-1]["content"]
    assert tool_output.startswith("[Ф4]")  # three fragments found first, the next label
    assert result.citation_errors == ["ссылка [Ф11] на несуществующий фрагмент"]
    assert [s.number for s in result.sources] == [1, 2, 3, 4, 5, 6]


async def test_a_tool_search_never_overfills_the_prompt() -> None:
    model = ScriptedModel(
        json_reply(course="os"),
        Reply("", [ToolCall("s1", "search_course_materials", {"query": "мьютекс"})]),
        Reply("Ответ [Ф1]."),
    )
    library = Library({QUESTION: 0.9, "мьютекс": 0.9})

    result = await agent(model, library, agent_tool_tokens=5).ask(QUESTION)

    assert "не помещаются" in model.calls[-1]["messages"][-1]["content"]
    assert len(result.sources) == 3


async def test_tool_rounds_are_bounded_and_the_last_request_offers_no_tools() -> None:
    calc = Reply("", [ToolCall("c", "python_calc", {"expression": "1+1"})])
    model = ScriptedModel(json_reply(course="os"), calc, calc, Reply("Итог [Ф1]."))

    await agent(model, Library({QUESTION: 0.9}), agent_tool_rounds=2).ask(QUESTION)

    assert [bool(call["tools"]) for call in model.calls[1:]] == [True, True, False]


async def test_search_time_adds_up_over_every_search() -> None:
    model = ScriptedModel(
        json_reply(course="os"), json_reply(query="другой запрос"), Reply("Ответ [Ф1].")
    )
    library = Library({QUESTION: 0.0, "другой запрос": 0.9})

    result = await agent(model, library).ask(QUESTION)

    assert len(library.searches) == 2
    assert result.timings.total_ms >= result.timings.search_ms >= 0
    assert result.mode == "agent"


# --- python_calc ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("2 + 2 * 3", 8),
        ("(1 + 2) ** 2", 9),
        ("-3 // 2", -2),
        ("comb(10, 3)", 120),
        ("max(1, 5, 2)", 5),
        ("round(pi, 3)", 3.142),
        ("sqrt(16)", 4.0),
    ],
)
def test_arithmetic_is_computed(expression: str, value) -> None:
    assert calculate(expression) == pytest.approx(value)


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("true")',
        "(1).__class__",
        "open('x')",
        "[1, 2]",
        "'text'",
        "lambda: 1",
        "x + 1",
        "True + 1",
        "[x for x in range(3)]",
        "print(1)",
    ],
)
def test_anything_but_arithmetic_is_refused_before_it_runs(expression: str) -> None:
    assert python_calc(expression).startswith("Ошибка")


@pytest.mark.parametrize(
    "expression",
    ["9 ** 9 ** 9", "((10 ** 1000) ** 1000) ** 1000", "factorial(100000)", "comb(10**6, 5*10**5)"],
)
def test_arithmetic_that_would_hang_the_process_is_refused(expression: str) -> None:
    assert python_calc(expression).startswith("Ошибка")


def test_errors_come_back_as_text_for_the_model() -> None:
    assert python_calc("1 / 0") == "Ошибка: ZeroDivisionError: division by zero"
    assert python_calc("2 +").startswith("Ошибка: не арифметика")
    assert python_calc("4.0 / 2") == "2"
