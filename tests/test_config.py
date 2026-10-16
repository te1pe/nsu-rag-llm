"""Settings behave as documented, and `.env.example` documents all of them."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nsu_rag_llm.config import Settings, render_env_example

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def env_example_keys() -> set[str]:
    keys = set()
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            keys.add(line.split("=", 1)[0])
    return keys


def test_defaults_match_the_documented_stack() -> None:
    # _env_file=None so a developer's own .env cannot sway the assertions.
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "qwen3.5-9b-8k"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_model == "intfloat/multilingual-e5-base"


def test_sampling_defaults_are_the_deliberate_ones() -> None:
    """qwen3.5 ships temperature 1 and presence_penalty 1.5; we override both.

    A non-zero presence penalty punishes repeated tokens, and this service is
    meant to repeat a citation after every claim.
    """
    settings = Settings(_env_file=None)

    assert settings.llm_temperature == 0.0
    assert settings.llm_presence_penalty == 0.0
    assert settings.llm_reasoning_effort == "none"


def test_answer_budget_fits_inside_the_context_window() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_max_output_tokens < settings.llm_context_tokens


def test_an_unknown_retrieval_mode_is_refused_at_start_up(monkeypatch) -> None:
    """A typo in RETRIEVAL_MODE must fail loudly, not on the first question."""
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid-rerank")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_environment_overrides_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "gemma4-12b-8k")
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "16384")

    settings = Settings(_env_file=None)

    assert settings.llm_model == "gemma4-12b-8k"
    assert settings.llm_context_tokens == 16384


def test_env_example_covers_every_setting() -> None:
    """Guards against a new setting that nobody can discover."""
    declared = {name.upper() for name in Settings.model_fields}

    assert env_example_keys() == declared


def test_env_example_on_disk_is_the_generated_one() -> None:
    """Stronger than the key check: comments and defaults must match too.

    Regenerate with:
        uv run python -c "from nsu_rag_llm.config import write_env_example as w; w()"
    """
    assert ENV_EXAMPLE.read_text(encoding="utf-8") == render_env_example()


def test_env_example_values_are_the_actual_defaults(monkeypatch) -> None:
    """`.env.example` must be copyable to `.env` without changing behaviour."""
    pairs = _example_pairs()
    # Read the built-in defaults from a clean environment first, so that a
    # variable the developer happens to export cannot skew the comparison.
    for key, _ in pairs:
        monkeypatch.delenv(key, raising=False)
    defaults = Settings(_env_file=None).model_dump()

    for key, value in pairs:
        monkeypatch.setenv(key, value)
    from_example = Settings(_env_file=None).model_dump()

    assert from_example == defaults


def test_relative_paths_are_anchored_to_the_repository(monkeypatch) -> None:
    """`RAW_DIR=data/raw` must not depend on the working directory."""
    monkeypatch.setenv("RAW_DIR", "data/raw")

    settings = Settings(_env_file=None)

    assert settings.raw_dir.is_absolute()
    assert settings.raw_dir.parts[-2:] == ("data", "raw")


def _example_pairs() -> list[tuple[str, str]]:
    pairs = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            pairs.append((key, value))
    return pairs
