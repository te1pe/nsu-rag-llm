"""LLM-as-judge: is an answer faithful to its fragments, cited right, complete?

The judge is a model behind its own OpenAI-compatible endpoint (JUDGE_*
settings), asked two separate questions about every answer:

* **faithfulness and citations** - it reads the fragments the answering model
  read, labelled as that model saw them, and the answer with its labels;
* **completeness** - it compares the answer with the author's reference answer.

Every verdict is a JSON object constrained by a schema. The lists of problems
come before the yes/no fields: a small model decides better once it has
written down what it found, and the lists tell a reader of the run *why*.
A reply that does not parse is a judge failure, counted as such - never a
verdict (ADR-028).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from nsu_rag_llm.config import Settings
from nsu_rag_llm.generation.citations import Fragment
from nsu_rag_llm.generation.prompt import FRAGMENTS_HEADER, QUESTION_PREFIX, fragment_block

FAITHFULNESS_SYSTEM = """Ты — строгий проверяющий. Тебе дают фрагменты методичек, вопрос студента и ответ ассистента. Ассистент должен был отвечать только по этим фрагментам и ставить после каждого утверждения метку фрагмента, из которого оно взято, например [Ф2].

Проверь ответ и заполни поля:
1. unsupported — утверждения ответа, которых нет ни в одном фрагменте. Перефразирование — не нарушение. Сведения, которых во фрагментах нет, — нарушение, даже если они верны. Формула, которой во фрагменте нет (на её месте там «[формула]»), — тоже нарушение.
2. miscited — утверждения, после которых стоит метка фрагмента, не подтверждающего это утверждение. Утверждение из unsupported, после которого стоит метка, тоже сюда.
3. uncited — утверждения без метки фрагмента.
4. faithful — true, если список unsupported пуст.
5. citations_correct — true, если списки miscited и uncited пусты.

Утверждения в списках приводи коротко, своими словами. Ответь только JSON-объектом."""  # noqa: E501 - prose, not code

COMPLETENESS_SYSTEM = """Ты — строгий проверяющий. Тебе дают вопрос студента, эталонный ответ преподавателя и ответ ассистента.

Заполни поля:
1. missing — ключевые утверждения эталона, которых нет в ответе ассистента. Другая формулировка того же — не пропуск. Лишние сведения в ответе ассистента здесь не оцениваются.
2. complete — true, если список missing пуст.

Утверждения в списке приводи коротко, своими словами. Ответь только JSON-объектом."""  # noqa: E501 - prose, not code


class FaithfulnessVerdict(BaseModel):
    unsupported: list[str]
    miscited: list[str]
    uncited: list[str]
    faithful: bool
    citations_correct: bool

    @property
    def consistent(self) -> bool:
        """Do the yes/no fields agree with the lists the judge wrote first?

        An unsupported claim that carries a label is a citation error too;
        judges tend to record it under `unsupported` only, so "citations
        wrong" with nothing but unsupported claims listed is not a
        contradiction.
        """
        if self.faithful != (not self.unsupported):
            return False
        if self.miscited or self.uncited:
            return not self.citations_correct
        return self.citations_correct or bool(self.unsupported)


class CompletenessVerdict(BaseModel):
    missing: list[str]
    complete: bool

    @property
    def consistent(self) -> bool:
        return self.complete == (not self.missing)


class JudgeError(Exception):
    """The judge replied with something that is not a verdict."""


class JsonModel(Protocol):
    async def complete(self, messages: list[dict[str, str]], name: str, schema: dict) -> str: ...


class OpenAICompatibleJson:
    """The judge's model, asked for JSON that matches a schema."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.judge_base_url,
            api_key=settings.judge_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def complete(self, messages: list[dict[str, str]], name: str, schema: dict) -> str:
        s = self.settings
        response = await self.client.chat.completions.create(
            model=s.judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=s.judge_max_output_tokens,
            reasoning_effort=s.judge_reasoning_effort,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        )
        return response.choices[0].message.content or ""


def schema_of(model: type[BaseModel]) -> dict:
    """The JSON schema a verdict must match, closed to extra fields."""
    return model.model_json_schema() | {"additionalProperties": False}


def parse[V: (FaithfulnessVerdict, CompletenessVerdict)](text: str, model: type[V]) -> V:
    """Read a verdict, tolerating prose or a code fence around the JSON."""
    try:
        return model.model_validate_json(text)
    except ValidationError:
        pass
    found = re.search(r"\{.*\}", text, re.DOTALL)
    if found:
        try:
            return model.model_validate_json(found.group(0))
        except ValidationError as exc:
            raise JudgeError(f"не вердикт: {exc.errors()[0]['msg']}") from exc
    raise JudgeError(f"нет JSON в ответе судьи ({len(text)} симв.)")


class Judge:
    def __init__(self, model: JsonModel, name: str) -> None:
        self.model = model
        self.name = name

    async def faithfulness(
        self, question: str, fragments: Sequence[Fragment], raw_answer: str
    ) -> FaithfulnessVerdict:
        blocks = "\n\n".join(fragment_block(f) for f in fragments)
        user = (
            f"{FRAGMENTS_HEADER}\n\n{blocks}\n\n{QUESTION_PREFIX}{question}\n\n"
            f"Ответ ассистента:\n{raw_answer}"
        )
        return await self._ask(FAITHFULNESS_SYSTEM, user, FaithfulnessVerdict, "faithfulness")

    async def completeness(self, question: str, reference: str, answer: str) -> CompletenessVerdict:
        user = (
            f"{QUESTION_PREFIX}{question}\n\nЭталонный ответ:\n{reference}\n\n"
            f"Ответ ассистента:\n{answer}"
        )
        return await self._ask(COMPLETENESS_SYSTEM, user, CompletenessVerdict, "completeness")

    async def _ask[V: (FaithfulnessVerdict, CompletenessVerdict)](
        self, system: str, user: str, model: type[V], name: str
    ) -> V:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return parse(await self.model.complete(messages, name, schema_of(model)), model)
