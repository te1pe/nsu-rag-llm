"""Fit the fragments into the model's context window.

The window is LLM_CONTEXT_TOKENS (8192, pinned in the Modelfile); the answer
needs LLM_MAX_OUTPUT_TOKENS of it. Everything else - system prompt, question,
fragments, and the chat template's own tokens - must fit in the rest, or the
server silently cuts the prompt, which is the failure CLAUDE.md warns about.

Tokens are counted with the answering model's own tokenizer. Checked against
Ollama's reported prompt_tokens on ten prompts of real chunks: the count plus
CHAT_OVERHEAD matched exactly on all nine Russian ones and was one token
short on an English one, hence the margin (ADR-024).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from nsu_rag_llm.generation.citations import Fragment
from nsu_rag_llm.generation.prompt import FRAGMENTS_HEADER, QUESTION_PREFIX, SYSTEM, fragment_block
from nsu_rag_llm.retrieval.base import Hit

log = logging.getLogger(__name__)

#: Tokens the chat template adds around a system and a user message, and the
#: assistant turn's opening with thinking off. Measured against Ollama.
CHAT_OVERHEAD = 17
#: Slack for the separators between fragments and the one-token drift seen.
MARGIN = 32


@dataclass
class Fitted:
    fragments: list[Fragment]
    dropped: list[dict] = field(default_factory=list)
    """Chunks that were retrieved but did not fit."""
    prompt_tokens: int = 0
    """Estimated prompt size, overhead included."""
    budget: int = 0


def fit(
    question: str,
    hits: Sequence[Hit],
    count: Callable[[str], int],
    context_tokens: int,
    output_tokens: int,
) -> Fitted:
    """Take fragments in rank order while they fit; number them 1..n.

    A fragment that does not fit is skipped rather than ending the list: a
    shorter one further down may still fit, and every dropped fragment is
    logged, as CLAUDE.md requires.
    """
    budget = context_tokens - output_tokens - CHAT_OVERHEAD - MARGIN
    used = count(SYSTEM) + count(f"{FRAGMENTS_HEADER}\n\n\n\n{QUESTION_PREFIX}{question}")
    result = Fitted(fragments=[], budget=budget)
    for hit in hits:
        candidate = Fragment(len(result.fragments) + 1, hit.chunk)
        cost = count(fragment_block(candidate)) + 2  # the blank line between fragments
        if used + cost > budget:
            result.dropped.append(hit.chunk)
            continue
        result.fragments.append(candidate)
        used += cost
    result.prompt_tokens = used + CHAT_OVERHEAD
    if result.dropped:
        log.warning(
            "context budget: dropped %d of %d fragments (budget %d tokens, used %d): %s",
            len(result.dropped),
            len(hits),
            budget,
            used,
            ", ".join(c["id"][:8] for c in result.dropped),
        )
    return result
