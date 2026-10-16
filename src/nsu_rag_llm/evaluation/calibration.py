"""How far to trust the judge: its verdicts against the author's own labels.

For every answer the author labelled in `eval/manual_labels.jsonl`, the
judge's verdict on the same answer - same run, same question - is read from
that run's `judgements.jsonl`. Agreement and Cohen's kappa are computed for
faithfulness and for completeness separately; a judge can be good at one and
useless at the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nsu_rag_llm.evaluation.agreement import Agreement, agreement
from nsu_rag_llm.evaluation.generation import read_judged
from nsu_rag_llm.evaluation.labels import LabelFile, read_labels


@dataclass
class Calibration:
    labels: LabelFile
    faithful: Agreement | None = None
    complete: Agreement | None = None
    runs: list[str] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    """Which judge model produced the verdicts, per run - ideally just one."""
    unmatched: list[str] = field(default_factory=list)
    """Labels with no verdict to compare: not judged, or the judge failed."""
    report: Path | None = None


def calibrate(labels_file: Path, runs_dir: Path) -> Calibration:
    labels = read_labels(labels_file, runs_dir)
    result = Calibration(labels)
    if not labels.ok:
        return result
    verdicts: dict[str, dict] = {}
    pairs: dict[str, tuple[list[bool], list[bool]]] = {"faithful": ([], []), "complete": ([], [])}
    for label in labels.labels:
        if label.answer_run not in verdicts:
            path = runs_dir / label.answer_run / "judgements.jsonl"
            verdicts[label.answer_run] = read_judged(path) if path.exists() else {}
        judged = verdicts[label.answer_run].get(label.id)
        matched = False
        for name in pairs:
            verdict = getattr(judged, name) if judged else None
            if verdict is not None:
                pairs[name][0].append(getattr(label, name))
                pairs[name][1].append(verdict)
                matched = True
        if not matched:
            result.unmatched.append(f"{label.answer_run}/{label.id}")
    result.runs = sorted(verdicts)
    for run in result.runs:
        path = runs_dir / run / "judge.json"
        judge = json.loads(path.read_text(encoding="utf-8"))["judge"] if path.exists() else "?"
        if judge not in result.judges:
            result.judges.append(judge)
    result.faithful = agreement(*pairs["faithful"])
    result.complete = agreement(*pairs["complete"])
    return result


def _fmt(value: float, pct: bool = False) -> str:
    if value != value:  # NaN
        return "—"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def render(calibration: Calibration) -> str:
    judges = ", ".join(f"`{j}`" for j in calibration.judges) or "—"
    lines = [
        f"# Калибровка судьи — {date.today().isoformat()}",
        "",
        f"Судья {judges} против разметки автора (`eval/manual_labels.jsonl`). "
        f"Прогоны: {', '.join(f'`{r}`' for r in calibration.runs) or '—'}.",
        "",
        "| вердикт | n | совпадение | каппа Коэна | оба «да» | оба «нет» "
        "| только автор «да» | только судья «да» |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, title in (("faithful", "верность"), ("complete", "полнота")):
        a: Agreement | None = getattr(calibration, name)
        if a is None:
            continue
        lines.append(
            f"| {title} | {a.n} | {_fmt(a.observed, pct=True)} | {_fmt(a.kappa)} | "
            f"{a.both_yes} | {a.both_no} | {a.only_a} | {a.only_b} |"
        )
    lines += [
        "",
        "Каппа: 1 — полное согласие, 0 — не лучше случайного, < 0 — хуже случайного. "
        "Ориентир Лэндиса и Коха: выше 0,6 — существенное согласие, выше 0,8 — почти полное.",
        "",
    ]
    if calibration.unmatched:
        lines += ["Размеченные ответы без вердикта судьи:", ""]
        lines += [f"- `{u}`" for u in calibration.unmatched]
        lines.append("")
    return "\n".join(lines)
