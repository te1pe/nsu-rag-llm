"""The answering model, behind any OpenAI-compatible endpoint.

Sampling comes from settings on every request (ADR-002), thinking is switched
off with reasoning_effort (ADR-003), and the reply is streamed so that the time
to the first token can be measured and shown.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

from nsu_rag_llm.config import Settings


@dataclass(frozen=True)
class Delta:
    """A piece of a streamed reply: some text, why it stopped, or token usage."""

    text: str = ""
    reasoning: bool = False
    """A piece of the model's thinking: never shown, but it is output, and time."""
    finish_reason: str | None = None
    """"stop", or "length" when the answer ran into LLM_MAX_OUTPUT_TOKENS."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatModel(Protocol):
    def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[Delta]: ...


class OpenAICompatibleChat:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[Delta]:
        s = self.settings
        response = await self.client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            temperature=s.llm_temperature,
            top_p=s.llm_top_p,
            presence_penalty=s.llm_presence_penalty,
            max_tokens=s.llm_max_output_tokens,
            reasoning_effort=s.llm_reasoning_effort,
            stream=True,
            # The last event then carries prompt and completion token counts:
            # the proof that the prompt fitted the budget.
            stream_options={"include_usage": True},
        )
        # Leaving the block closes the connection, and Ollama stops generating
        # when its client goes away: a reader who disconnects mid-answer does
        # not keep the model busy for the next question.
        async with response:
            async for event in response:
                if event.choices:
                    choice = event.choices[0]
                    # Thinking arrives outside `content`: Ollama calls the field
                    # "reasoning", vLLM "reasoning_content".
                    extra = choice.delta.model_extra or {}
                    if extra.get("reasoning") or extra.get("reasoning_content"):
                        yield Delta(reasoning=True)
                    if choice.delta.content or choice.finish_reason:
                        yield Delta(
                            text=choice.delta.content or "",
                            finish_reason=choice.finish_reason,
                        )
                if event.usage is not None:
                    yield Delta(
                        prompt_tokens=event.usage.prompt_tokens,
                        completion_tokens=event.usage.completion_tokens,
                    )


def llm_token_counter(tokenizer_name: str):
    """Token counting with the answering model's tokenizer (ADR-024)."""
    import transformers
    from transformers import AutoTokenizer

    # It warns on every text over its nominal length; counting is the point.
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
