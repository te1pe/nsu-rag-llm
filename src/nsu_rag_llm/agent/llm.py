"""The agent's model: one reply at a time, with tools or a JSON schema.

Phase 3 streams its answer token by token; the agent cannot, because a reply
may turn out to be a tool call rather than text, and that is known only once
the reply is complete. Everything else - endpoint, sampling, thinking - comes
from the same settings as the answering model's (ADR-002, ADR-003).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from openai import AsyncOpenAI

from nsu_rag_llm.config import Settings


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def as_message(self) -> dict:
        """The reply as the next request must repeat it, tool calls included."""
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return message


class ToolModel(Protocol):
    async def reply(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        schema: tuple[str, dict] | None = None,
        max_tokens: int | None = None,
    ) -> Reply: ...


def _arguments(raw: str | None) -> dict:
    """Tool arguments as the model wrote them; a model can write bad JSON."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": raw or ""}
    return value if isinstance(value, dict) else {"_unparsed": raw}


class OpenAICompatibleTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    async def reply(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        schema: tuple[str, dict] | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        s = self.settings
        extra: dict = {}
        if tools:
            extra["tools"] = tools
        if schema:
            name, body = schema
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": body, "strict": True},
            }
        response = await self.client.chat.completions.create(
            model=s.llm_model,
            messages=messages,
            temperature=s.llm_temperature,
            top_p=s.llm_top_p,
            presence_penalty=s.llm_presence_penalty,
            max_tokens=max_tokens or s.llm_max_output_tokens,
            reasoning_effort=s.llm_reasoning_effort,
            **extra,
        )
        message = response.choices[0].message
        calls = [
            ToolCall(call.id, call.function.name, _arguments(call.function.arguments))
            for call in message.tool_calls or []
        ]
        usage = response.usage
        return Reply(
            content=message.content or "",
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
