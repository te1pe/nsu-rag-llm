"""Answer generation on synthetic fragments, with a scripted fake LLM.

No model, no network: the "LLM" replays pieces of text it was given, the
retriever returns fixed hits, and tokens are counted as words.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from nsu_rag_llm.config import Settings
from nsu_rag_llm.generation.answer import NO_CITATIONS, Answerer
from nsu_rag_llm.generation.budget import CHAT_OVERHEAD, MARGIN, fit
from nsu_rag_llm.generation.citations import Fragment, StreamingRenderer, render
from nsu_rag_llm.generation.llm import Delta
from nsu_rag_llm.generation.prompt import (
    REFUSAL,
    SYSTEM,
    fragment_block,
    is_refusal,
    messages,
    user_message,
)
from nsu_rag_llm.retrieval.base import Hit

COURSE = "Операционные системы"
DOC = "Синтетическое пособие"


def chunk(cid: str, text: str = "Синтетический текст.", start: int = 3, end: int | None = None):
    return {
        "id": cid,
        "course": "os",
        "course_title": COURSE,
        "doc_title": DOC,
        "file": "os/sem4/synthetic.pdf",
        "page_start": start,
        "page_end": end or start,
        "section_path": ["Процессы"],
        "origin": "digital",
        "context": f"{COURSE} › {DOC} › Процессы",
        "text": text,
    }


def fragments(*chunks: dict) -> dict[int, Fragment]:
    return {n: Fragment(n, c) for n, c in enumerate(chunks, start=1)}


def cite(page: str) -> str:
    return f"[{COURSE} · «{DOC}» · с. {page}]"


# --- citations -------------------------------------------------------------


def test_a_label_becomes_the_full_citation() -> None:
    rendered = render("Процесс — это программа в работе [Ф1].", fragments(chunk("a")))

    assert rendered.text == f"Процесс — это программа в работе {cite('3')}."
    assert rendered.cited == [1]
    assert rendered.errors == []


def test_a_fragment_across_pages_cites_the_range() -> None:
    rendered = render("[Ф1]", fragments(chunk("a", start=3, end=4)))

    assert rendered.text == cite("3–4")


def test_a_label_with_no_fragment_is_an_error() -> None:
    rendered = render("Утверждение [Ф7].", fragments(chunk("a")))

    assert rendered.text == "Утверждение [нет фрагмента Ф7]."
    assert rendered.errors == ["ссылка [Ф7] на несуществующий фрагмент"]
    assert rendered.cited == []


@pytest.mark.parametrize("label", ["[Ф1, Ф2]", "[Ф1; Ф2]", "[Ф1, 2]", "[ф1,ф2]", "[Ф 1, Ф 2]"])
def test_one_bracket_may_cite_several_fragments(label: str) -> None:
    rendered = render(label, fragments(chunk("a", start=3), chunk("b", start=5)))

    assert rendered.text == f"{cite('3')} {cite('5')}"
    assert rendered.cited == [1, 2]


def test_the_spelled_out_label_is_understood() -> None:
    assert render("[Фрагмент 1]", fragments(chunk("a"))).cited == [1]


@pytest.mark.parametrize(
    "prose",
    [
        "на отрезке [0, 1] функция непрерывна",
        "интервал [a, b] и точка [1]",
        "см. [формула] ниже",
        "подпись [Рис. 2]",
        "список [1, 2, 3]",
    ],
)
def test_brackets_that_are_not_labels_are_left_alone(prose: str) -> None:
    """Mathematics puts numbers in brackets; only [Ф…] is a citation."""
    rendered = render(prose, fragments(chunk("a"), chunk("b"), chunk("c")))

    assert rendered.text == prose
    assert rendered.cited == []
    assert rendered.errors == []


def test_two_fragments_of_one_page_are_cited_once() -> None:
    rendered = render("[Ф1, Ф2]", fragments(chunk("a", start=3), chunk("b", start=3)))

    assert rendered.text == cite("3")
    assert rendered.cited == [1, 2]


def stream(renderer: StreamingRenderer, pieces: list[str]) -> list[str]:
    out = [renderer.feed(piece) for piece in pieces]
    out.append(renderer.flush())
    return out


def test_a_label_split_across_tokens_is_held_back_then_expanded() -> None:
    renderer = StreamingRenderer(fragments(chunk("a")))

    out = stream(renderer, ["Ответ ", "[", "Ф", "1", "]", "."])

    assert out[:5] == ["Ответ ", "", "", "", cite("3")]
    assert "".join(out) == f"Ответ {cite('3')}."
    assert renderer.cited == [1]


TEXT = "Отрезок [0, 1] [Ф1]; поток [Ф2, Ф9] и [Рис. 2]."


@pytest.mark.parametrize("cut", range(1, len(TEXT)))
def test_streaming_matches_rendering_the_whole_text_at_any_split(cut: int) -> None:
    table = fragments(chunk("a", start=3), chunk("b", start=5))
    whole = render(TEXT, table)
    renderer = StreamingRenderer(table)

    out = stream(renderer, [TEXT[:cut], TEXT[cut:]])

    assert "".join(out) == whole.text
    assert renderer.cited == whole.cited
    assert renderer.errors == whole.errors


def test_streaming_character_by_character_matches_too() -> None:
    table = fragments(chunk("a", start=3), chunk("b", start=5))
    renderer = StreamingRenderer(table)

    assert "".join(stream(renderer, list(TEXT))) == render(TEXT, table).text


def test_an_unclosed_bracket_is_released_at_the_end() -> None:
    renderer = StreamingRenderer(fragments(chunk("a")))

    assert "".join(stream(renderer, ["текст ", "[Ф"])) == "текст [Ф"
    assert renderer.errors == []


def test_a_bracket_open_too_long_to_be_a_label_is_not_held() -> None:
    renderer = StreamingRenderer(fragments(chunk("a")))
    prose = "[" + "очень длинное пояснение в скобках " * 3

    assert renderer.feed(prose) == prose


# --- prompt ------------------------------------------------------------------


def test_a_fragment_is_shown_with_its_label_context_and_pages() -> None:
    block = fragment_block(Fragment(2, chunk("a", "Текст фрагмента.", start=3, end=4)))

    assert block == f"[Ф2] {COURSE} › {DOC} › Процессы (с. 3–4)\nТекст фрагмента."


def test_the_prompt_holds_the_rules_the_fragments_and_the_question() -> None:
    table = fragments(chunk("a", "Первый."), chunk("b", "Второй."))

    system, user = messages("Что такое процесс?", list(table.values()))

    assert system == {"role": "system", "content": SYSTEM}
    assert user["role"] == "user"
    assert user["content"].index("[Ф1]") < user["content"].index("[Ф2]")
    assert user["content"].endswith("Вопрос: Что такое процесс?")


def test_the_rules_teach_the_label_the_parser_reads_and_the_refusal() -> None:
    """The prompt and the citation parser must agree, or every answer fails."""
    assert "[Ф1]" in SYSTEM
    assert render("[Ф1]", fragments(chunk("a"))).cited == [1]
    assert f"«{REFUSAL}»" in SYSTEM


@pytest.mark.parametrize(
    "answer",
    [REFUSAL, f"«{REFUSAL}»", f"**{REFUSAL}**", "в методичках этого нет", f"{REFUSAL} Увы."],
)
def test_refusals_are_recognised(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer", ["В методичках сказано, что процесс — это… [Ф1]", "Процесс — это… [Ф1]"]
)
def test_answers_are_not_refusals(answer: str) -> None:
    assert not is_refusal(answer)


# --- budget ------------------------------------------------------------------


def words(text: str) -> int:
    return len(text.split())


def chars(text: str) -> int:
    return len(text)


def hits(*chunks: dict) -> list[Hit]:
    return [Hit(c, 1.0) for c in chunks]


def test_everything_fits_a_roomy_window() -> None:
    fitted = fit("вопрос", hits(chunk("a"), chunk("b")), words, 8192, 1024)

    assert [f.number for f in fitted.fragments] == [1, 2]
    assert fitted.dropped == []


def test_what_does_not_fit_is_skipped_logged_and_numbering_stays_dense(caplog) -> None:
    def count(text: str) -> int:
        return 10_000 if "ОГРОМНЫЙ" in text else words(text)

    with caplog.at_level(logging.WARNING, logger="nsu_rag_llm.generation.budget"):
        fitted = fit(
            "вопрос",
            hits(chunk("a"), chunk("big", "ОГРОМНЫЙ фрагмент"), chunk("c")),
            count,
            8192,
            1024,
        )

    assert [f.chunk["id"] for f in fitted.fragments] == ["a", "c"]
    assert [f.number for f in fitted.fragments] == [1, 2]
    assert [c["id"] for c in fitted.dropped] == ["big"]
    assert "dropped 1 of 3" in caplog.text


def test_the_estimate_covers_the_whole_prompt_and_stays_in_the_window() -> None:
    """Counted in characters, which add up exactly, the estimate must cover
    every character sent, and leave room for the answer."""
    text = "слово " * 300
    many = hits(*(chunk(str(i), text) for i in range(20)))
    context, output = 12_000, 1_000

    fitted = fit("вопрос", many, chars, context, output)
    sent = chars(SYSTEM) + chars(user_message("вопрос", fitted.fragments))

    assert 0 < len(fitted.fragments) < 20
    assert fitted.prompt_tokens >= sent + CHAT_OVERHEAD
    assert fitted.prompt_tokens + MARGIN + output <= context


# --- the pipeline ----------------------------------------------------------


class ScriptedChat:
    """A fake LLM that streams the pieces it was given, then usage."""

    def __init__(self, *pieces: str, finish: str = "stop", prompt_tokens: int = 700) -> None:
        self.pieces = pieces
        self.finish = finish
        self.prompt_tokens = prompt_tokens
        self.calls: list[list[dict]] = []

    async def stream(self, messages):
        self.calls.append(messages)
        for piece in self.pieces:
            yield Delta(text=piece)
        yield Delta(finish_reason=self.finish)
        yield Delta(prompt_tokens=self.prompt_tokens, completion_tokens=len(self.pieces))


class Search:
    def __init__(self, found: list[Hit], delay: float = 0.0) -> None:
        self.found = found
        self.delay = delay
        self.calls: list[tuple] = []
        self.running = 0
        self.most_at_once = 0

    def __call__(self, query: str, k: int, course: str | None) -> list[Hit]:
        self.calls.append((query, k, course))
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        time.sleep(self.delay)
        self.running -= 1
        return self.found[:k]


def make(found, chat, rerank=None, **settings) -> Answerer:
    return Answerer(
        search=found if isinstance(found, Search) else Search(found),
        rerank=rerank,
        chat=chat,
        count=words,
        settings=Settings(_env_file=None, **settings),
        mode="hybrid",
        courses=frozenset({"os"}),
    )


TWO = hits(chunk("a", "Процесс — программа в работе.", start=3), chunk("b", "Поток.", start=5))


async def collect(answerer: Answerer, question: str = "Что такое процесс?", course=None):
    return [event async for event in answerer.events(question, course)]


async def test_the_answer_streams_with_citations_expanded() -> None:
    chat = ScriptedChat("Процесс — программа ", "[Ф", "1]", ". Поток — его часть [Ф2].")

    events = await collect(make(TWO, chat))

    kinds = [e.kind for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done"
    assert set(kinds[1:-1]) == {"token"}
    result = events[-1].data
    assert "".join(e.data for e in events if e.kind == "token") == result.answer
    assert result.answer == f"Процесс — программа {cite('3')}. Поток — его часть {cite('5')}."
    assert result.cited == [1, 2]
    assert result.citation_errors == []
    assert not result.refused


async def test_the_model_reads_the_labelled_fragments_and_the_question() -> None:
    chat = ScriptedChat("Ответ [Ф1].")

    await make(TWO, chat).ask("  Что такое процесс?  ")

    user = chat.calls[0][1]["content"]
    assert "[Ф1]" in user and "[Ф2]" in user
    assert user.endswith("Вопрос: Что такое процесс?")


async def test_sources_carry_the_label_and_the_citation() -> None:
    result = await make(TWO, ScriptedChat("Ответ [Ф2].")).ask("?")

    assert [(s.number, s.citation) for s in result.sources] == [(1, cite("3")), (2, cite("5"))]
    assert result.cited == [2]


async def test_a_citation_of_a_missing_fragment_is_an_error() -> None:
    result = await make(TWO, ScriptedChat("Ответ [Ф9].")).ask("?")

    assert result.citation_errors == ["ссылка [Ф9] на несуществующий фрагмент"]
    assert "[нет фрагмента Ф9]" in result.answer


async def test_an_answer_without_any_citation_is_flagged() -> None:
    result = await make(TWO, ScriptedChat("Ответ без ссылок.")).ask("?")

    assert result.citation_errors == [NO_CITATIONS]


async def test_a_refusal_is_recognised_and_needs_no_citation() -> None:
    result = await make(TWO, ScriptedChat(REFUSAL)).ask("?")

    assert result.refused
    assert result.citation_errors == []


async def test_nothing_found_is_a_refusal_without_asking_the_model() -> None:
    chat = ScriptedChat("не должен быть вызван")

    result = await make([], chat).ask("?")

    assert chat.calls == []
    assert result.answer == REFUSAL
    assert result.refused
    assert result.sources == []


async def test_the_reranker_reads_a_longer_list_and_its_best_are_used() -> None:
    found = Search(hits(*(chunk(c) for c in "abcde")))

    def reverse(query, shortlist):
        return list(reversed(shortlist))

    answerer = make(found, ScriptedChat("[Ф1]"), reverse, answer_fragments=2, rerank_candidates=5)
    result = await answerer.ask("?")

    assert found.calls[0][1] == 5
    assert [s.number for s in result.sources] == [1, 2]
    assert [s.text for s in result.sources] == [chunk("e")["text"]] * 2
    assert [c["id"] for c in (h.chunk for h in reverse("?", found.found))][:2] == ["e", "d"]


async def test_without_a_reranker_the_retriever_top_is_used() -> None:
    found = Search(hits(*(chunk(c) for c in "abcde")))

    await make(found, ScriptedChat("[Ф1]"), answer_fragments=3).ask("?", course="os")

    assert found.calls == [("?", 3, "os")]


async def test_every_stage_is_timed() -> None:
    def slow_rerank(query, shortlist):
        time.sleep(0.02)
        return list(shortlist)

    result = await make(Search(TWO, delay=0.01), ScriptedChat("[Ф1]"), slow_rerank).ask("?")

    t = result.timings
    assert t.search_ms >= 10
    assert t.rerank_ms >= 20
    assert t.llm_first_token_ms is not None and t.first_token_ms is not None
    assert t.llm_first_output_ms == t.llm_first_token_ms  # no thinking here
    assert t.llm_first_token_ms <= t.first_token_ms <= t.total_ms
    assert t.first_token_ms >= t.search_ms + t.rerank_ms


class ThinkingChat(ScriptedChat):
    """Thinks - output nobody sees - for a while before it answers."""

    async def stream(self, messages):
        self.calls.append(messages)
        for _ in range(3):
            await asyncio.sleep(0.01)
            yield Delta(reasoning=True)
        for piece in self.pieces:
            yield Delta(text=piece)
        yield Delta(prompt_tokens=100, completion_tokens=10)


async def test_thinking_is_output_but_not_the_first_token_of_the_answer() -> None:
    result = await make(TWO, ThinkingChat("Ответ [Ф1].")).ask("?")

    t = result.timings
    assert t.llm_first_output_ms < t.llm_first_token_ms
    assert t.llm_first_token_ms - t.llm_first_output_ms >= 20  # two more sleeps
    assert result.answer.startswith("Ответ")  # no thinking in the answer


async def test_usage_and_a_cut_off_answer_are_reported(caplog) -> None:
    chat = ScriptedChat("Длинный ответ [Ф1]", finish="length", prompt_tokens=9000)

    with caplog.at_level(logging.INFO, logger="nsu_rag_llm.generation.answer"):
        result = await make(TWO, chat).ask("?")

    assert result.truncated
    assert result.prompt_tokens == 9000
    assert "search=" in caplog.text and "first_token=" in caplog.text
    assert "cut off" in caplog.text
    assert "may have cut it" in caplog.text  # 9000 > 8192 - 1024


async def test_fragments_over_the_budget_never_reach_the_model() -> None:
    found = hits(chunk("a"), chunk("big", "ОГРОМНЫЙ " * 9000), chunk("c"))
    chat = ScriptedChat("[Ф1]")

    result = await make(found, chat).ask("?")

    assert result.dropped == 1
    assert "ОГРОМНЫЙ" not in chat.calls[0][1]["content"]
    assert [s.number for s in result.sources] == [1, 2]


async def test_concurrent_questions_search_one_at_a_time() -> None:
    """The models behind search are not thread-safe."""
    found = Search(TWO, delay=0.05)
    answerer = make(found, ScriptedChat("[Ф1]"))

    await asyncio.gather(answerer.ask("один?"), answerer.ask("два?"))

    assert len(found.calls) == 2
    assert found.most_at_once == 1


async def test_a_reader_who_leaves_closes_the_model_stream_at_once() -> None:
    """Otherwise the connection to the LLM - and its generation - outlives the
    reader until the garbage collector happens to finalise the stream."""
    closed = []

    class Endless:
        async def stream(self, messages):
            try:
                while True:
                    yield Delta(text="слово ")
            finally:
                closed.append(True)

    events = make(TWO, Endless()).events("?")
    async for event in events:
        if event.kind == "token":
            break
    await events.aclose()

    assert closed == [True]


def test_warm_up_runs_search_and_rerank_once() -> None:
    reranked = []
    found = Search(TWO)
    answerer = make(found, ScriptedChat(), lambda q, h: reranked.append(q) or list(h))

    answerer.warm_up()

    assert len(found.calls) == 1
    assert len(reranked) == 1
