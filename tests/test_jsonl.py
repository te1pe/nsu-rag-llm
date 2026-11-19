from __future__ import annotations

import json
from pathlib import Path

from nsu_rag_llm import jsonl

SEPARATORS = "\u2028\u2029\x85"


def test_a_record_holding_a_line_separator_stays_whole(tmp_path: Path) -> None:
    records = [{"text": f"до{SEPARATORS}после"}, {"text": "вторая"}]
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )

    assert jsonl.read(path) == records


def test_blank_lines_are_skipped_and_numbers_kept_for_messages() -> None:
    text = '{"a": 1}\n\n  \n{"b": 2}\n'

    assert jsonl.lines(text) == ['{"a": 1}', '{"b": 2}']
    assert jsonl.numbered(text) == [(1, '{"a": 1}'), (4, '{"b": 2}')]
