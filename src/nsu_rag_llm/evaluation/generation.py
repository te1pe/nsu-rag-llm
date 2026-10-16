"""Answer quality and speed, per generating model: the phase 4 metrics.

Every model answers the same questions from the same fragments, so a
difference between two rows of the report is a difference between models.

* **faithfulness** - share of answers (refusals aside) whose every claim the
  judge found in the fragments;
* **citations valid** - share of answers citing only fragments that exist, and
  citing at least one: checked by code, no judge involved;
* **citations supported** - share of answers whose every citation the judge
  found to support the claim it follows, with no claim left uncited;
* **completeness** - share of *answerable* questions whose answer covers the
  reference answer; a refusal there counts as incomplete;
* **refusal accuracy** - share of *unanswerable* questions refused;
* **false refusals** - share of answerable questions refused;
* **TTFT** and **tokens/s** - medians, from the model's side only: the time
  from the request to the first token, and the answer's tokens over the time
  after it;
* **peak memory** - the largest size Ollama reported for the model while it
  answered (`/api/ps`: weights, KV cache and buffers).

A metric over no questions is "—", never 0.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from nsu_rag_llm.evaluation.questions import Question


@dataclass(frozen=True)
class Variant:
    """One generating model as it is compared: the model and how it is run."""

    label: str
    """Its name in reports and run directories, unique."""
    model: str
    """The Ollama model (built from its Modelfile, num_ctx 8192)."""
    tokenizer: str
    """Hugging Face repository with the model's own tokenizer (ADR-024)."""
    reasoning_effort: str = "none"
    max_output_tokens: int = 1024


def load_variants(path: Path) -> list[Variant]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    variants = [Variant(**entry) for entry in raw]
    labels = [v.label for v in variants]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"повторяются label: {', '.join(duplicates)}")
    return variants


@dataclass
class Answered:
    """One question, answered by one variant - what is stored per answer."""

    id: str
    type: str
    course: str
    refused: bool
    raw_answer: str
    answer: str
    cited: list[int]
    citation_errors: list[str]
    truncated: bool
    dropped: int
    prompt_tokens: int | None
    prompt_tokens_estimated: int
    completion_tokens: int | None
    llm_first_output_ms: float | None
    llm_first_token_ms: float | None
    total_ms: float
    sources: list[dict]
    """Number, file and pages of each fragment the model read."""
    routed_course: str | None = None
    """The agent's router's pick, "none" included; None when nothing routed."""
    steps: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed. Thinking tokens are completion tokens too, so the
        clock starts at the first output of any kind; total_ms runs from the
        request to the model, like the first-output mark."""
        if not self.completion_tokens or self.llm_first_output_ms is None:
            return None
        seconds = (self.total_ms - self.llm_first_output_ms) / 1000
        return self.completion_tokens / seconds if seconds > 0 else None


@dataclass
class Judged:
    """The judge's verdicts on one answer; None where it was not asked or failed."""

    id: str
    faithful: bool | None = None
    citations_correct: bool | None = None
    complete: bool | None = None
    unsupported: list[str] = field(default_factory=list)
    miscited: list[str] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    inconsistent: bool = False
    """The judge's yes/no contradicted the list it wrote."""
    errors: list[str] = field(default_factory=list)


def _share(values: list[bool]) -> float:
    return sum(values) / len(values) if values else math.nan


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


@dataclass
class VariantResult:
    variant: Variant
    run_id: str
    answered: list[Answered] = field(default_factory=list)
    judged: dict[str, Judged] = field(default_factory=dict)
    peak_memory_bytes: int | None = None
    room: int = 0
    """Tokens the prompt may take: the window minus the answer's reserve."""

    def _verdicts(self, name: str, keep: Callable[[Answered], bool]) -> list[bool]:
        values = []
        for a in self.answered:
            if not keep(a):
                continue
            verdict = getattr(self.judged.get(a.id), name, None) if a.id in self.judged else None
            if verdict is not None:
                values.append(verdict)
        return values

    def faithfulness(self, keep: Callable[[Answered], bool] = lambda a: True) -> float:
        return _share(self._verdicts("faithful", lambda a: not a.refused and keep(a)))

    def citations_supported(self, keep: Callable[[Answered], bool] = lambda a: True) -> float:
        return _share(self._verdicts("citations_correct", lambda a: not a.refused and keep(a)))

    def citations_valid(self, keep: Callable[[Answered], bool] = lambda a: True) -> float:
        return _share([not a.citation_errors for a in self.answered if not a.refused and keep(a)])

    def completeness(self, keep: Callable[[Answered], bool] = lambda a: True) -> float:
        values = []
        for a in self.answered:
            if not a.answerable or not keep(a):
                continue
            if a.refused:
                values.append(False)
                continue
            verdict = self.judged[a.id].complete if a.id in self.judged else None
            if verdict is not None:
                values.append(verdict)
        return _share(values)

    def refusal_accuracy(self) -> float:
        return _share([a.refused for a in self.answered if not a.answerable])

    def false_refusals(self) -> float:
        return _share([a.refused for a in self.answered if a.answerable])

    def ttft_ms(self) -> float:
        return _median([a.llm_first_token_ms for a in self.answered if a.llm_first_token_ms])

    def tokens_per_second(self) -> float:
        return _median([a.tokens_per_second for a in self.answered if a.tokens_per_second])

    def count(self, keep: Callable[[Answered], bool]) -> int:
        return sum(keep(a) for a in self.answered)

    def truncated(self) -> int:
        return sum(a.truncated for a in self.answered)

    def over_budget(self) -> int:
        """Answers whose prompt, as the server counted it, did not fit the room."""
        return sum(
            a.prompt_tokens is not None and a.prompt_tokens > self.room for a in self.answered
        )

    def estimate_error(self) -> tuple[int, int] | None:
        """Least and greatest (estimated - reported) prompt tokens."""
        errors = [
            a.prompt_tokens_estimated - a.prompt_tokens
            for a in self.answered
            if a.prompt_tokens is not None and a.sources
        ]
        return (min(errors), max(errors)) if errors else None

    def routed(self) -> bool:
        return any(a.routed_course is not None for a in self.answered)

    def router_accuracy(self) -> float:
        """Share of answerable questions routed to the question's own course."""
        return _share(
            [
                a.routed_course == a.course
                for a in self.answered
                if a.answerable and a.routed_course is not None
            ]
        )

    def mean_searches(self) -> float:
        counts = [sum(s.startswith("поиск") for s in a.steps) for a in self.answered]
        return statistics.fmean(counts) if counts else math.nan

    def judge_failures(self) -> int:
        return sum(bool(j.errors) for j in self.judged.values())

    def inconsistent(self) -> int:
        return sum(j.inconsistent for j in self.judged.values())


def by_id(questions: list[Question]) -> dict[str, Question]:
    return {q.id: q for q in questions}


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_judged(path: Path) -> dict[str, Judged]:
    judged = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = Judged(**json.loads(line))
                judged[record.id] = record
    return judged
