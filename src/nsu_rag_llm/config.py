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
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5-9b-8k"
    # Ollama ключ не нужен, клиенту OpenAI нужен
    llm_api_key: str = "ollama"
    llm_timeout_seconds: float = 120.0

    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_presence_penalty: float = 0.0
    llm_reasoning_effort: ReasoningEffort = "none"

    # совпадает с num_ctx в Modelfile
    llm_context_tokens: int = 8192
    llm_max_output_tokens: int = 1024
    llm_tokenizer: str = "Qwen/Qwen3.5-9B"

    judge_base_url: str = "http://localhost:11434/v1"
    judge_model: str = "qwen3.5-9b-8k"
    judge_api_key: str = "ollama"
    judge_reasoning_effort: ReasoningEffort = "none"
    judge_max_output_tokens: int = 1024
    ollama_api_url: str = "http://localhost:11434"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nsu_methodicals"

    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: Device = "auto"

    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")
    courses_file: Path = Path("data/courses.yaml")
    manifest_file: Path = Path("data/nsumedia-manifest.csv")
    inventory_file: Path = Path("data/inventory.md")
    inventory_costs_file: Path = Path("data/inventory-costs.json")
    parse_report_file: Path = Path("data/parse-report.md")
    chunks_file: Path = Path("data/chunks.jsonl")
    ingest_report_file: Path = Path("data/ingest-report.md")
    questions_file: Path = Path("eval/questions.jsonl")
    eval_reports_dir: Path = Path("eval/reports")
    manual_labels_file: Path = Path("eval/manual_labels.jsonl")
    models_file: Path = Path("eval/models.yaml")
    eval_runs_dir: Path = Path("data/eval-runs")

    parse_workers: int = 2

    corpus_scope: CorpusScope = "mvp"
    ocr_engine: OcrEngine = "ocrmac"

    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50
    # на MPS 32 — самый быстрый
    embed_batch_size: int = 32

    retrieval_candidates: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    reranker_enabled: bool = True

    retrieval_mode: RetrievalMode = "hybrid+rerank"
    answer_fragments: int = 10

    agent_min_score: float = 0.1
    agent_max_rewrites: int = 2
    agent_tool_rounds: int = 3
    agent_tool_fragments: int = 3
    agent_tool_tokens: int = 1500

    log_level: str = "INFO"
    health_timeout_seconds: float = Field(
        default=5.0,
        description="Per-dependency deadline for /health, so a hung backend "
        "cannot hang the probe itself.",
    )

    @field_validator("*")
    @classmethod
    def anchor_paths_to_project_root(cls, value: object) -> object:
        if isinstance(value, Path) and not value.is_absolute():
            return PROJECT_ROOT / value
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str) -> None:
    logger = logging.getLogger("nsu_rag_llm")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


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
