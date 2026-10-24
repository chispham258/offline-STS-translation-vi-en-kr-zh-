"""Phase 3 gate: pipeline -> manifest -> evaluator -> report, with no models."""

import onevoice.components.dummy  # noqa: F401 - registers vad/asr/mt/tts doubles
from onevoice.config import load_config
from onevoice.eval.evaluator import evaluate
from onevoice.eval.manifest import write_run
from onevoice.eval.report import to_markdown, write_report
from onevoice.pipeline import run_pipeline

REFS = "fixtures/mini_vi_en/refs.jsonl"
CONFIG = "configs/dummy_vi_en.yaml"


def _run(tmp_path):
    records = run_pipeline(load_config(CONFIG), REFS, tmp_path)
    manifest = tmp_path / "run.jsonl"
    write_run(manifest, records)
    return manifest


def test_the_pipeline_produces_a_manifest_the_evaluator_can_read(tmp_path):
    table = evaluate(_run(tmp_path), REFS, experiment="dummy")
    assert table.counts["total"] == 3
    assert table.counts["references"] == 3


def test_every_metric_is_produced(tmp_path):
    table = evaluate(_run(tmp_path), REFS, experiment="dummy")
    expected = {"bleu", "chrf", "wer", "cer", "end_to_speech_ms", "rtf"}
    assert expected <= set(table.metrics)


def test_no_utterance_errored_on_the_dummy_pipeline(tmp_path):
    table = evaluate(_run(tmp_path), REFS, experiment="dummy")
    assert table.counts["errored"] == 0
    assert table.counts["scored"] == 3


def test_latency_was_actually_measured(tmp_path):
    table = evaluate(_run(tmp_path), REFS, experiment="dummy")
    latency = table.metrics["end_to_speech_ms"]
    assert latency.detail["measured"] == 3
    assert latency.detail["unmeasurable"] == 0


def test_a_report_renders(tmp_path):
    table = evaluate(_run(tmp_path), REFS, experiment="dummy")
    markdown = to_markdown(table)
    assert "bleu" in markdown
    assert "errored" in markdown
    md_path, csv_path = write_report(table, tmp_path / "report")
    assert md_path.is_file()
    assert csv_path.is_file()


def test_the_dummy_pipeline_scores_poorly_and_that_is_correct(tmp_path):
    # D18: EchoASR describes the audio rather than transcribing it, so the
    # "translation" cannot match the reference. A high score here would mean
    # the harness was measuring something other than what it claims.
    table = evaluate(_run(tmp_path), REFS, experiment="dummy")
    assert table.metrics["bleu"].value < 5.0
