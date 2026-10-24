"""Report rendering."""

import math

from onevoice.eval.evaluator import ResultsTable
from onevoice.eval.metrics.base import MetricResult
from onevoice.eval.report import to_csv, to_markdown, write_report


def _table() -> ResultsTable:
    return ResultsTable(
        experiment="whisper_small_opus_melo",
        metrics={
            "bleu": MetricResult("bleu", 38.5, {"tokenizer": "13a"}),
            "end_to_speech_ms": MetricResult(
                "end_to_speech_ms", 950.0, {"p95": 1400.0}
            ),
        },
        counts={"total": 3, "scored": 2, "errored": 1},
    )


def test_markdown_contains_the_experiment_name():
    assert "whisper_small_opus_melo" in to_markdown(_table())


def test_markdown_reports_every_metric():
    output = to_markdown(_table())
    assert "bleu" in output
    assert "38.5" in output
    assert "end_to_speech_ms" in output


def test_markdown_surfaces_the_error_count():
    # A report showing BLEU without showing that a third of the corpus failed
    # is misleading even when the number is correct.
    output = to_markdown(_table())
    assert "errored" in output
    assert "1" in output


def test_csv_has_a_header_and_one_row_per_metric_count_and_detail():
    # Updated for the kind column and per-detail rows (findings 3 and 5):
    # detail entries (e.g. p95) now get their own keyed row.
    table = _table()
    lines = to_csv(table).strip().split("\n")
    assert lines[0] == "kind,name,value,experiment"
    detail_rows = sum(len(m.detail) for m in table.metrics.values())
    assert len(lines) == 1 + len(table.metrics) + len(table.counts) + detail_rows


def test_csv_surfaces_the_error_count():
    assert "errored" in to_csv(_table())


def test_csv_distinguishes_counts_from_metrics_via_kind_column():
    # errored,1 (a count) must not be indistinguishable from a metric
    # literally named "errored".
    lines = to_csv(_table()).strip().split("\n")
    assert any(line.startswith("count,errored,") for line in lines)
    assert not any(line.startswith("metric,errored,") for line in lines)


def test_csv_includes_the_experiment_name():
    assert "whisper_small_opus_melo" in to_csv(_table())


def test_csv_renders_detail_with_a_keyed_metric_name():
    output = to_csv(_table())
    assert "end_to_speech_ms.p95" in output
    assert "1400" in output


def test_markdown_renders_detail_next_to_the_headline_value():
    # p95 must be visible, not thrown away at the render layer.
    output = to_markdown(_table())
    assert "p95" in output
    assert "1400" in output


def test_report_keeps_large_latencies_out_of_scientific_notation():
    table = ResultsTable(
        experiment="x",
        metrics={"end_to_speech_ms": MetricResult("end_to_speech_ms", 12345.6, {})},
        counts={"total": 1, "scored": 1, "errored": 0},
    )
    md = to_markdown(table)
    csv = to_csv(table)
    assert "12345.6" in md
    assert "e+" not in md.lower()
    assert "12345.6" in csv
    assert "e+" not in csv.lower()


def test_report_format_does_not_degrade_small_values():
    table = ResultsTable(
        experiment="x",
        metrics={
            "wer": MetricResult("wer", 0.0, {}),
            "cer": MetricResult("cer", 0.8, {}),
        },
        counts={"total": 1, "scored": 1, "errored": 0},
    )
    md = to_markdown(table)
    assert "| wer | 0 |" in md
    assert "| cer | 0.8 |" in md


def test_nan_renders_as_a_marker_not_as_the_string_nan():
    table = ResultsTable(
        experiment="empty",
        metrics={"bleu": MetricResult("bleu", math.nan, {})},
        counts={"total": 1, "scored": 0, "errored": 1},
    )
    assert "n/a" in to_markdown(table)


def test_write_report_creates_both_files(tmp_path):
    md_path, csv_path = write_report(_table(), tmp_path)
    assert md_path.is_file()
    assert csv_path.is_file()
    assert "whisper_small_opus_melo" in md_path.read_text()
