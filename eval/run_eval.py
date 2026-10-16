"""Evaluate the pipeline against eval/questions.jsonl.

    uv run python eval/run_eval.py --stage retrieval
    uv run python eval/run_eval.py --stage retrieval --validate-only
    uv run python eval/run_eval.py --stage retrieval --modes bm25,hybrid

The same as `nsu-rag eval`; this script exists because CLAUDE.md names it.
Phase 4 adds `--stage generation`.
"""

from __future__ import annotations

import sys

from nsu_rag_llm.cli import app

if __name__ == "__main__":
    sys.exit(app(["eval", *sys.argv[1:]]))
