from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from nsu_rag_llm.generation.citations import Fragment
from nsu_rag_llm.generation.prompt import FRAGMENTS_HEADER, QUESTION_PREFIX, SYSTEM, fragment_block
from nsu_rag_llm.retrieval.base import Hit

log = logging.getLogger(__name__)

CHAT_OVERHEAD = 17
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
    budget = context_tokens - output_tokens - CHAT_OVERHEAD - MARGIN
    used = count(SYSTEM) + count(f"{FRAGMENTS_HEADER}\n\n\n\n{QUESTION_PREFIX}{question}")
    result = Fitted(fragments=[], budget=budget)
    for hit in hits:
        candidate = Fragment(len(result.fragments) + 1, hit.chunk)
        cost = count(fragment_block(candidate)) + 2
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
