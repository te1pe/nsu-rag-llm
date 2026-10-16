"""Application settings.

Values come from environment variables, falling back to `.env`, falling back to
the defaults declared here. Field names map to env vars by upper-casing, so
`llm_model` is set by `LLM_MODEL` — the spelling used throughout the docs.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Device = Literal["auto", "mps", "cuda", "cpu"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]
RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid+rerank"]
CorpusScope = Literal["mvp", "full"]
OcrEngine = Literal["ocrmac", "rapidocr"]


class Settings(BaseSettings):
    """Everything the service needs to know that is not hard-coded."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Answering LLM: any OpenAI-compatible endpoint, Ollama by default ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5-9b-8k"
    # Ollama ignores the key, but the OpenAI client refuses to start without one.
    llm_api_key: str = "ollama"
    llm_timeout_seconds: float = 120.0

    # Sampling lives here rather than in ollama/Modelfile so that phase 4 can
    # compare different models under identical sampling. qwen3.5 ships with
    # temperature 1 and presence_penalty 1.5; the latter penalises repeated
    # tokens, which is exactly wrong for answers that repeat a citation after
    # every claim.
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_presence_penalty: float = 0.0
    # "none" switches Ollama's thinking off over the OpenAI-compatible endpoint.
    llm_reasoning_effort: ReasoningEffort = "none"

    # Prompt budget. The Modelfile pins num_ctx to the same number; keep them
    # in step or the prompt builder will plan for room the model does not have.
    llm_context_tokens: int = 8192
    llm_max_output_tokens: int = 1024
    # Hugging Face repository whose tokenizer is LLM_MODEL's own: prompts are
    # measured with it before they are sent (ADR-024). Only the tokenizer
    # files are downloaded, not the weights.
    llm_tokenizer: str = "Qwen/Qwen3.5-9B"

    # --- Judge LLM (phase 4), a deliberately separate endpoint ---
    judge_base_url: str = "http://localhost:11434/v1"
    judge_model: str = "qwen3.5-9b-8k"
    judge_api_key: str = "ollama"
    # A verdict is a short JSON object; thinking would make every one of them
    # several times slower, and its effect on agreement is measured, not assumed.
    judge_reasoning_effort: ReasoningEffort = "none"
    judge_max_output_tokens: int = 1024
    # Ollama's own API, for what the OpenAI-compatible one does not offer: the
    # memory a model takes (/api/ps) and unloading it between compared models.
    ollama_api_url: str = "http://localhost:11434"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nsu_methodicals"

    # --- Models loaded in-process (phases 1c and 2) ---
    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: Device = "auto"

    # --- Paths ---
    # Declared relative so that `.env.example` can be machine-independent; the
    # validator below anchors them, so consumers always get an absolute path.
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    courses_file: Path = Path("data/courses.yaml")
    # The nsumedia download manifest: the only record of which subject a file
    # was filed under, and of the downloads that failed.
    manifest_file: Path = Path("data/nsumedia-manifest.csv")
    inventory_file: Path = Path("data/inventory.md")
    # Benchmark results, kept so a plain inventory run does not discard them.
    inventory_costs_file: Path = Path("data/inventory-costs.json")
    parse_report_file: Path = Path("data/parse-report.md")
    chunks_file: Path = Path("data/chunks.jsonl")
    ingest_report_file: Path = Path("data/ingest-report.md")
    # The author's evaluation questions, and where reports go. Both live in the
    # repository: reports hold only question ids and metrics (CLAUDE.md).
    questions_file: Path = Path("eval/questions.jsonl")
    eval_reports_dir: Path = Path("eval/reports")
    # Phase 4: the author's verdicts on answers, the models compared, and each
    # run's answers and judgements - which quote the handbooks, hence data/.
    manual_labels_file: Path = Path("eval/manual_labels.jsonl")
    models_file: Path = Path("eval/models.yaml")
    eval_runs_dir: Path = Path("data/eval-runs")

    # --- Parsing (phase 1b) ---
    # Parallel processes. The layout model's ONNX runtime already spreads each
    # page over ~3 cores, so more processes fight over the same 8: measured on
    # 718 pages, 1 process took 64 s, 2 took 49 s, 4 took 55 s at twice the CPU.
    parse_workers: int = 2

    # --- Scaling to the full corpus (phase 4.5) ---
    # "mvp" parses and indexes what courses.yaml includes; "full" every unique
    # document, scans OCR'd, into an index of its own - corpus.full.env sets
    # this with the files and collection that go with it.
    corpus_scope: CorpusScope = "mvp"
    # Measured in phase 4.5 (ADR-030).
    ocr_engine: OcrEngine = "ocrmac"

    # --- Chunking and indexing (phase 1c) ---
    # Tokens of text per chunk, and of the previous chunk's closing sentences
    # carried over. Counted with the embedding model's own tokenizer.
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50
    # Measured on MPS: 32 gives 24 chunks/s, 64 gives 23, 128 gives 18.
    embed_batch_size: int = 32

    # --- Retrieval (phase 2) ---
    # Each retriever's shortlist before fusion, the RRF constant, and how many
    # fused candidates the cross-encoder re-reads.
    retrieval_candidates: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    # The reranker is the heaviest model in the pipeline; false turns it off
    # everywhere, and "hybrid+rerank" then means plain "hybrid".
    reranker_enabled: bool = True

    # --- Answering (phase 3) ---
    # The retrieval mode answers are built on, and how many of its best
    # fragments the model reads (ADR-026).
    retrieval_mode: RetrievalMode = "hybrid+rerank"
    answer_fragments: int = 10

    # --- The agent (phase 5) ---
    # Below this best reranker score the search is judged to have missed:
    # the agent rewrites the query, and after AGENT_MAX_REWRITES refuses
    # (ADR-031). Tools may add AGENT_TOOL_FRAGMENTS fragments per search,
    # within AGENT_TOOL_TOKENS of room kept free in the prompt.
    agent_min_score: float = 0.1
    agent_max_rewrites: int = 2
    agent_tool_rounds: int = 3
    agent_tool_fragments: int = 3
    agent_tool_tokens: int = 1500

    # --- Operational ---
    log_level: str = "INFO"
    health_timeout_seconds: float = Field(
        default=5.0,
        description="Per-dependency deadline for /health, so a hung backend "
        "cannot hang the probe itself.",
    )

    @field_validator("*")
    @classmethod
    def anchor_paths_to_project_root(cls, value: object) -> object:
        """Read a relative path as relative to the repository, not the cwd.

        Applied to every field rather than to a list of names: the list went
        stale twice while the settings were still being written, and a path
        that silently stays relative breaks only once something runs from a
        different working directory.
        """
        if isinstance(value, Path) and not value.is_absolute():
            return PROJECT_ROOT / value
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once."""
    return Settings()


def configure_logging(level: str) -> None:
    """Send the package's log records to stderr.

    Uvicorn configures only its own loggers, so without this the per-answer
    timing line of phase 3 would go nowhere. Idempotent: the app factory and
    the CLI may both call it.
    """
    logger = logging.getLogger("nsu_rag_llm")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


#: Section headers for `.env.example`, keyed by the field they precede.
_SECTIONS = {
    "llm_base_url": (
        "Answering LLM. Any OpenAI-compatible endpoint; Ollama by default.\n"
        "# From inside a container use http://host.docker.internal:11434/v1"
    ),
    "llm_temperature": (
        "Sampling. Kept here rather than in ollama/Modelfile so that phase 4\n"
        "# compares models under identical sampling. qwen3.5 defaults to\n"
        "# temperature 1 and presence_penalty 1.5; the latter punishes repeated\n"
        "# tokens, which is wrong for answers that repeat a citation per claim.\n"
        "# LLM_REASONING_EFFORT=none switches Ollama's thinking off."
    ),
    "llm_context_tokens": "Prompt budget. Keep in step with num_ctx in ollama/Modelfile.",
    "judge_base_url": "Judge LLM (phase 4). Deliberately a separate endpoint.",
    "qdrant_url": "Vector store. Inside docker compose this is http://qdrant:6333",
    "embedding_model": ("Models loaded in-process (phases 1c and 2). device: auto|mps|cuda|cpu"),
    "raw_dir": "Paths, read as relative to the repository root.",
    "parse_workers": "Parsing (phase 1b).",
    "corpus_scope": (
        "The full corpus (phase 4.5): see corpus.full.env.\n"
        "# CORPUS_SCOPE: mvp|full; OCR_ENGINE: ocrmac|rapidocr"
    ),
    "chunk_tokens": "Chunking and indexing (phase 1c).",
    "retrieval_candidates": "Retrieval (phase 2).",
    "retrieval_mode": "Answering (phase 3). RETRIEVAL_MODE: bm25|dense|hybrid|hybrid+rerank",
    "agent_min_score": "The agent (phase 5).",
    "log_level": "Operational.",
}


def render_env_example() -> str:
    """Render `.env.example` from the settings themselves.

    Regenerate with:
        uv run python -c "from nsu_rag_llm.config import write_env_example as w; w()"

    A test compares the file on disk against this, so a new setting cannot be
    added without also being documented.
    """
    defaults = Settings(_env_file=None)
    lines = [
        "# Copy to .env and edit. Every value below is the built-in default, so an",
        "# empty .env behaves identically to this file.",
        "# Generated from nsu_rag_llm.config.Settings - a test fails if they drift.",
    ]
    for name in Settings.model_fields:
        if name in _SECTIONS:
            lines += ["", f"# --- {_SECTIONS[name]}"]
        value = getattr(defaults, name)
        if isinstance(value, Path):
            value = value.relative_to(PROJECT_ROOT)
        lines.append(f"{name.upper()}={value}")
    return "\n".join(lines) + "\n"


def write_env_example(path: Path | None = None) -> Path:
    target = path or PROJECT_ROOT / ".env.example"
    target.write_text(render_env_example(), encoding="utf-8")
    return target


def resolve_device(preference: Device) -> str:
    """Turn `auto` into the best device actually available.

    Order is mps -> cuda -> cpu: this project indexes natively on an Apple
    silicon Mac, and inside the container there is no GPU at all.
    """
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
