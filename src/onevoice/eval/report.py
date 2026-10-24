"""Renders a ResultsTable to markdown and CSV.

The error count appears in both formats. A BLEU score printed without the number
of utterances that failed to produce one is a misleading report even when the
score itself is arithmetically correct.
"""

from __future__ import annotations

import csv
import io
import math
from pathlib import Path

from onevoice.eval.evaluator import ResultsTable


def _format(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    text = f"{value:.4g}"
    if "e" in text:
        # %g drops to scientific notation once the exponent reaches the
        # precision (e.g. 12345.6 -> "1.235e+04"), which is plausible but
        # unreadable for a latency in the tens of thousands of ms.
        text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


def _format_any(value: object) -> str:
    if isinstance(value, float):
        return _format(value)
    return str(value)


def to_markdown(table: ResultsTable) -> str:
    lines = [
        f"# Evaluation — {table.experiment or 'unnamed'}",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name, result in table.metrics.items():
        lines.append(f"| {name} | {_format(result.value)} |")
        for key, value in result.detail.items():
            lines.append(f"| ↳ {name}.{key} | {_format_any(value)} |")
    lines += ["", "| Count | Utterances |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in table.counts.items()]
    return "\n".join(lines) + "\n"


def to_csv(table: ResultsTable) -> str:
    # A "kind" column keeps counts (e.g. errored,1) distinguishable from a
    # metric that happened to be named the same thing, and detail rows are
    # keyed as "<metric>.<detail>" (e.g. end_to_speech_ms.p95) so p95 is no
    # longer thrown away at the render layer.
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["kind", "name", "value", "experiment"])
    for name, result in table.metrics.items():
        writer.writerow(["metric", name, _format(result.value), table.experiment])
        for key, value in result.detail.items():
            writer.writerow(
                ["detail", f"{name}.{key}", _format_any(value), table.experiment]
            )
    for name, count in table.counts.items():
        writer.writerow(["count", name, count, table.experiment])
    return buf.getvalue()


def write_report(table: ResultsTable, directory: str | Path) -> tuple[Path, Path]:
    """Write report.md and report.csv into directory. Returns both paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    csv_path = out / "report.csv"
    md_path.write_text(to_markdown(table), encoding="utf-8")
    csv_path.write_text(to_csv(table), encoding="utf-8")
    return md_path, csv_path
