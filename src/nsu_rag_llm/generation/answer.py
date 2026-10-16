"""From a question to a cited answer: search, rerank, fit, generate, cite.

Every stage is timed - search, rerank, the first token, the whole answer - and
logged on one line per question, as CLAUDE.md asks. The same pipeline serves
the CLI, `POST /ask` and the streaming `POST /ask/stream`: it produces a
sequence of events, and callers either forward them as they come or wait for
the last one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from nsu_rag_llm.config import Settings
from nsu_rag_llm.generation.budget import fit
from nsu_rag_llm.generation.citations import Fragment, StreamingRenderer
from nsu_rag_llm.generation.llm import ChatModel
from nsu_rag_llm.generation.prompt import REFUSAL, is_refusal, messages
from nsu_rag_llm.retrieval.base import Hit

log = logging.getLogger(__name__)

Search = Callable[[str, int, str | None], list[Hit]]
Rerank = Callable[[str, Sequence[Hit]], list[Hit]]
Count = Callable[[str], int]

WARM_UP_QUERY = "прогрев"
NO_CITATIONS = "в ответе нет ни одной ссылки"


@dataclass
class Source:
    """A fragment the model was given, under the label it was given."""

    number: int
    citation: str
    course: str
    course_title: str
    doc_title: str
    file: str
    page_start: int
    page_end: int
    section_path: list[str]
    origin: str
    context: str
    """The "course › document › section" line the model saw above the text."""
    text: str

    @classmethod
    def of(cls, fragment: Fragment) -> Source:
        c = fragment.chunk
        return cls(
            number=fragment.number,
            citation=fragment.citation,
            course=c["course"],
            course_title=c["course_title"],
            doc_title=c["doc_title"],
            file=c["file"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            section_path=list(c.get("section_path", [])),
            origin=c.get("origin", "unknown"),
            context=c["context"],
            text=c["text"],
        )


@dataclass
class Timings:
    """Milliseconds: how long each stage took, and when the reader saw what."""

    search_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_first_output_ms: float | None = None
    """From the request to the model until it outputs anything, thinking included:
    the time it spent reading the prompt."""
    llm_first_token_ms: float | None = None
    """From the request to the model until the first token of the answer itself;
    with thinking on, later than the first output by the time spent thinking."""
    first_token_ms: float | None = None
    """From the question's arrival: what the reader waits before text appears."""
    total_ms: float = 0.0
    """From the question's arrival to the end of the answer."""


@dataclass
class Result:
    answer: str
    """With every [Фn] expanded into [course · «document» · p. N]."""
    raw_answer: str
    """What the model wrote, labels and all: what a judge checks citations on."""
    refused: bool
    sources: list[Source]
    cited: list[int]
    """Labels of the sources the answer cites, in order of first appearance."""
    citation_errors: list[str]
    truncated: bool
    """The answer ran into LLM_MAX_OUTPUT_TOKENS and is cut off."""
    dropped: int
    """Fragments retrieved but left out to fit the context budget."""
    prompt_tokens_estimated: int
    prompt_tokens: int | None
    completion_tokens: int | None
    mode: str
    course: str | None
    """The course filter the search used."""
    timings: Timings = field(default_factory=Timings)
    routed_course: str | None = None
    """The agent's router's pick (phase 5); None for the plain pipeline."""
    steps: list[str] = field(default_factory=list)
    """What the agent did, in order: routing, searches, rewrites, tools."""

    def summary(self) -> dict[str, Any]:
        """Everything but the sources, which a stream has already sent."""
        return {k: v for k, v in asdict(self).items() if k != "sources"}


@dataclass(frozen=True)
class Event:
    kind: str
    """"sources", then "token" for each piece of the answer, then "done"."""
    data: Any


def indexed_courses(chunks_file: Path) -> frozenset[str]:
    """Course slugs that have chunks in the index, and so can filter a search."""
    with chunks_file.open(encoding="utf-8") as handle:
        return frozenset(json.loads(line)["course"] for line in handle if line.strip())


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


class Answerer:
    def __init__(
        self,
        *,
        search: Search,
        rerank: Rerank | None,
        chat: ChatModel,
        count: Count,
        settings: Settings,
        mode: str,
        courses: frozenset[str],
    ) -> None:
        self.search = search
        self.rerank = rerank
        self.chat = chat
        self.count = count
        self.settings = settings
        self.mode = mode
        self.courses = courses
        # With a reranker the cross-encoder picks the best of a longer list;
        # without one the retriever's own top fragments are the answer's.
        self.candidates = settings.rerank_candidates if rerank else settings.answer_fragments
        # The embedder, the cross-encoder and their tokenizers are not safe to
        # call from two threads at once (a Hugging Face fast tokenizer raises
        # "Already borrowed"): concurrent questions search one at a time.
        self._models = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings, mode: str | None = None) -> Answerer:
        """The production wiring: every model loaded here, once."""
        from nsu_rag_llm.generation.llm import OpenAICompatibleChat, llm_token_counter
        from nsu_rag_llm.retrieval.modes import Retrievers
        from nsu_rag_llm.retrieval.rerank import rerank

        mode = mode or settings.retrieval_mode
        retrievers = Retrievers(settings)
        if retrievers.uses_reranker(mode):
            search, reranker = retrievers.hybrid.search, partial(rerank, retrievers.scorer)
        else:
            search, reranker = retrievers.get(mode).search, None
            if mode == "hybrid+rerank":
                mode = "hybrid"  # RERANKER_ENABLED=false
        return cls(
            search=search,
            rerank=reranker,
            chat=OpenAICompatibleChat(settings),
            count=llm_token_counter(settings.llm_tokenizer),
            settings=settings,
            mode=mode,
            courses=indexed_courses(settings.chunks_file),
        )

    def warm_up(self) -> None:
        """Run the models once, so the first real question is timed fairly.

        The first call on MPS compiles kernels and takes seconds longer than
        every later one; that is start-up cost, not the cost of a search.
        """
        with self._models:
            hits = self.search(WARM_UP_QUERY, 2, None)
            if self.rerank is not None:
                self.rerank(WARM_UP_QUERY, hits)

    def retrieve(
        self, question: str, course: str | None = None, timings: Timings | None = None
    ) -> list[Hit]:
        """The fragments an answer is built from: searched, reranked, cut to size."""
        timings = timings or Timings()
        with self._models:
            started = time.perf_counter()
            hits = self.search(question, self.candidates, course)
            timings.search_ms = _ms(started)
            if self.rerank is not None:
                started = time.perf_counter()
                hits = self.rerank(question, hits)
                timings.rerank_ms = _ms(started)
        return hits[: self.settings.answer_fragments]

    async def events(self, question: str, course: str | None = None) -> AsyncIterator[Event]:
        arrived = time.perf_counter()
        timings = Timings()
        # Off the event loop: a server keeps answering /health while it searches.
        hits = await asyncio.to_thread(self.retrieve, question.strip(), course, timings)
        async with aclosing(self.events_from(question, course, hits, timings, arrived)) as events:
            async for event in events:
                yield event

    async def events_from(
        self,
        question: str,
        course: str | None,
        hits: Sequence[Hit],
        timings: Timings | None = None,
        arrived: float | None = None,
    ) -> AsyncIterator[Event]:
        """Answer from fragments already retrieved.

        The evaluation retrieves once per question and hands the same
        fragments to every model it compares, so that only the model differs.
        """
        arrived = arrived if arrived is not None else time.perf_counter()
        question = question.strip()
        timings = timings or Timings()
        fitted = fit(
            question,
            hits,
            self.count,
            self.settings.llm_context_tokens,
            self.settings.llm_max_output_tokens,
        )
        result = Result(
            answer="",
            raw_answer="",
            refused=False,
            sources=[Source.of(f) for f in fitted.fragments],
            cited=[],
            citation_errors=[],
            truncated=False,
            dropped=len(fitted.dropped),
            prompt_tokens_estimated=fitted.prompt_tokens,
            prompt_tokens=None,
            completion_tokens=None,
            mode=self.mode,
            course=course,
            timings=timings,
        )
        yield Event("sources", result.sources)

        if not fitted.fragments:
            # Nothing was found: the honest answer needs no model.
            result.answer = result.raw_answer = REFUSAL
            result.refused = True
            timings.total_ms = _ms(arrived)
            yield Event("token", REFUSAL)
            self._log(result)
            yield Event("done", result)
            return

        renderer = StreamingRenderer({f.number: f for f in fitted.fragments})
        raw: list[str] = []
        rendered: list[str] = []
        asked = time.perf_counter()
        # aclosing: a reader who stops listening mid-answer closes the model's
        # stream at once, rather than whenever the garbage collector gets to it.
        async with aclosing(self.chat.stream(messages(question, fitted.fragments))) as deltas:
            async for delta in deltas:
                if (delta.text or delta.reasoning) and timings.llm_first_output_ms is None:
                    timings.llm_first_output_ms = _ms(asked)
                if delta.text:
                    if timings.first_token_ms is None:
                        timings.llm_first_token_ms = _ms(asked)
                        timings.first_token_ms = _ms(arrived)
                    raw.append(delta.text)
                    if piece := renderer.feed(delta.text):
                        rendered.append(piece)
                        yield Event("token", piece)
                if delta.finish_reason == "length":
                    result.truncated = True
                if delta.prompt_tokens is not None:
                    result.prompt_tokens = delta.prompt_tokens
                    result.completion_tokens = delta.completion_tokens
        if tail := renderer.flush():
            rendered.append(tail)
            yield Event("token", tail)

        timings.total_ms = _ms(arrived)
        result.answer = "".join(rendered).strip()
        result.raw_answer = "".join(raw).strip()
        result.refused = is_refusal(result.raw_answer)
        result.cited = renderer.cited
        result.citation_errors = renderer.errors
        if not result.refused and not result.cited and not result.citation_errors:
            result.citation_errors.append(NO_CITATIONS)
        self._log(result)
        yield Event("done", result)

    async def ask(self, question: str, course: str | None = None) -> Result:
        async with aclosing(self.events(question, course)) as events:
            async for event in events:
                if event.kind == "done":
                    return event.data
        raise RuntimeError("the answer stream ended without a result")

    def _log(self, result: Result) -> None:
        t = result.timings
        # Thinking tokens count as completion tokens: time them from the first
        # output of any kind, not from the first token of the answer. Both
        # marks are counted from the request to the model; total_ms from the
        # question's arrival, which came (first_token_ms - llm_first_token_ms)
        # earlier.
        generating = 0.0
        if None not in (t.first_token_ms, t.llm_first_token_ms, t.llm_first_output_ms):
            asked = t.first_token_ms - t.llm_first_token_ms
            generating = (t.total_ms - asked - t.llm_first_output_ms) / 1000
        speed = (
            f"{result.completion_tokens / generating:.1f}"
            if result.completion_tokens and generating > 0
            else "-"
        )
        log.info(
            "answer mode=%s course=%s fragments=%d dropped=%d "
            "prompt_tokens=%s estimated=%d completion_tokens=%s "
            "search=%.0fms rerank=%.0fms llm_first_token=%sms first_token=%sms total=%.0fms "
            "tokens_per_s=%s refused=%s truncated=%s citation_errors=%d",
            result.mode,
            result.course or "-",
            len(result.sources),
            result.dropped,
            result.prompt_tokens if result.prompt_tokens is not None else "-",
            result.prompt_tokens_estimated,
            result.completion_tokens if result.completion_tokens is not None else "-",
            t.search_ms,
            t.rerank_ms,
            f"{t.llm_first_token_ms:.0f}" if t.llm_first_token_ms is not None else "-",
            f"{t.first_token_ms:.0f}" if t.first_token_ms is not None else "-",
            t.total_ms,
            speed,
            result.refused,
            result.truncated,
            len(result.citation_errors),
        )
        for error in result.citation_errors:
            log.warning("citation: %s", error)
        if result.truncated:
            log.warning(
                "answer cut off at LLM_MAX_OUTPUT_TOKENS=%d", self.settings.llm_max_output_tokens
            )
        room = self.settings.llm_context_tokens - self.settings.llm_max_output_tokens
        if result.prompt_tokens is not None and result.prompt_tokens > room:
            log.warning(
                "the prompt took %d tokens, more than the %d the budget allows: "
                "the server may have cut it",
                result.prompt_tokens,
                room,
            )
