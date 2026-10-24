"""The Phase 4 benchmark runner."""

import onevoice.components.dummy  # noqa: F401  - registers the dummy components
from onevoice.benchmark import Q6_LIMITATION, render_report, run_benchmark

REFS = "fixtures/mini_vi_en/refs.jsonl"
CONFIG = "configs/dummy_vi_en.yaml"


def test_runs_a_config_end_to_end(tmp_path):
    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="smoke")
    assert result.table.counts["total"] == 3
    assert result.manifest_path.is_file()
    assert result.report_path.is_file()


def test_report_carries_the_q6_limitation(tmp_path):
    # Spec section 7.5: a report without this does not satisfy the Phase 4 gate.
    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="smoke")
    assert Q6_LIMITATION.strip() in result.report_path.read_text()


def test_q6_statement_names_what_it_does_and_does_not_affect():
    # A blanket "the metric is broken" warning is worse than useless - it trains
    # readers to discount figures that are actually sound. So the caveat must
    # say which metric is blind and which is not.
    #
    # This previously asserted the word "batch", because the statement claimed
    # every model was batch-shaped and therefore unaffected. That claim was true
    # when written and false from Task 11 onward (F2). The intent is unchanged;
    # what satisfies it is now the rtf / pipeline_rtf distinction.
    assert "push()" in Q6_LIMITATION
    assert "rtf" in Q6_LIMITATION and "pipeline_rtf" in Q6_LIMITATION


def test_q6_statement_measures_streaming_cost_rather_than_asserting_it():
    """D45: the caveat must be computed from the run, not hardcoded.

    A constant cannot notice when it stops being true, which is exactly how a
    false statement survived five published reports.
    """
    from onevoice.benchmark import q6_statement
    from onevoice.eval.manifest import RunRecord

    def record(stream_start, speech_end, utt):
        return RunRecord(
            utt_id=utt,
            hyp_text="x",
            audio_duration_s=10.0,
            timing={
                "stream_start": stream_start,
                "speech_end": speech_end,
                "tts_done": 5.0,
            },
        )

    batch_only = q6_statement([record(2.0, 2.0, "a"), record(2.0, 2.0, "b")])
    assert "nothing is hidden" in batch_only

    streaming = q6_statement([record(0.0, 2.0, "a"), record(0.0, 2.0, "b")])
    assert "hides" in streaming
    assert "0.2000" in streaming  # 2s hidden over 10s of audio


def test_q6_statement_is_honest_when_it_cannot_measure():
    from onevoice.benchmark import q6_statement

    assert "could not be measured" in q6_statement([])


def test_render_report_does_not_alter_the_evaluator_output(tmp_path):
    # D21/D24: the evaluator's rendering must survive verbatim inside ours.
    from onevoice.eval.evaluator import evaluate
    from onevoice.eval.report import to_markdown

    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="smoke")
    evaluator_markdown = to_markdown(evaluate(result.manifest_path, REFS, "smoke"))
    assert evaluator_markdown in render_report(result.table)


def test_manifest_is_retained_for_rescoring(tmp_path):
    # D5 decoupling is what makes the D21 freeze cheap to hold: a corrected
    # metric can re-score this manifest without re-running a model.
    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="smoke")
    assert result.manifest_path.read_text().strip()


def test_run_writes_component_provenance(tmp_path):
    """Voice and checkpoint selection are part of the experiment.

    A benchmark number without them cannot be reproduced or fairly compared
    later, which matters most for MOS in Phase 5 where the audio artifacts
    outlive the run that made them.
    """
    import json

    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="prov")
    assert result.metadata_path.is_file()

    meta = json.loads(result.metadata_path.read_text())
    assert meta["experiment"] == "prov"
    assert set(meta["components"]) == {"vad", "asr", "mt", "tts"}
    assert meta["config"].endswith("dummy_vi_en.yaml")


def test_provenance_falls_back_to_the_class_name(tmp_path):
    # A component that does not implement provenance() still gets identified,
    # so the record is never silently missing a stage.
    import json

    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="prov")
    meta = json.loads(result.metadata_path.read_text())
    assert meta["components"]["asr"]["component"] == "EchoASR"


# --- checkpointing -----------------------------------------------------------


def test_reports_complete_on_a_fresh_run(tmp_path):
    import json

    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")
    assert result.status == "complete"
    assert result.utterances_fresh == 3
    assert result.utterances_resumed == 0
    assert json.loads(result.metadata_path.read_text())["status"] == "complete"


def test_resumes_from_a_partial_manifest(tmp_path):
    """The case that matters: an interruption after some utterances finished.

    A run that silently restarts wastes the tens of minutes a real
    configuration takes, and looks identical to one that never crashed.
    """
    import json

    # Simulate a run interrupted after one utterance.
    first = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")
    lines = first.manifest_path.read_text().splitlines()
    first.manifest_path.write_text(lines[0] + "\n")

    second = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")

    assert second.utterances_resumed == 1
    assert second.utterances_fresh == 2
    assert second.status == "resumed"
    assert json.loads(second.metadata_path.read_text())["status"] == "resumed"
    # And the completed set is whole again.
    assert second.table.counts["total"] == 3
    assert second.table.counts["errored"] == 0


def test_resume_false_starts_clean(tmp_path):
    """Resuming across a configuration change would mix two experiments into
    one manifest, and the mixture would score as though it were one run."""
    run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")
    again = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck", resume=False)
    assert again.utterances_resumed == 0
    assert again.utterances_fresh == 3


def test_a_fully_complete_run_redoes_nothing(tmp_path):
    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")
    assert result.utterances_fresh == 3
    rerun = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")
    assert rerun.utterances_fresh == 0
    assert rerun.utterances_resumed == 3


def test_metadata_records_timing_and_host(tmp_path):
    import json

    result = run_benchmark(CONFIG, REFS, tmp_path, experiment="ck")
    meta = json.loads(result.metadata_path.read_text())

    assert meta["started_at"] < meta["finished_at"]
    assert meta["duration_seconds"] >= 0
    assert meta["utterances_total"] == 3
    for key in ("python", "platform", "machine"):
        assert meta["host"][key]


def test_published_results_carry_the_q6_statement():
    """The published results document must carry the Q6 caveat.

    Q6: RTF and end-of-speech latency are blind to per-frame work inside
    push(). Every published number quotes latency, so the caveat must travel
    with the number rather than living in a spec a reader may not open.

    The per-report glob was retired when the repo moved to a single
    `docs/RESULTS.md`; the requirement itself still holds.
    """
    from pathlib import Path

    from onevoice.benchmark import Q6_LIMITATION

    results = Path("docs/RESULTS.md")
    assert results.is_file(), "no published results document found"

    # Match on the distinguishing sentence, not the whole block.
    needle = "Known limitation (Q6)"
    assert needle in Q6_LIMITATION, "Q6_LIMITATION no longer contains the needle"
    assert needle in results.read_text(), "RESULTS.md is missing the Q6 statement"


def test_q6_always_warns_against_subtracting_the_two_percentiles():
    """The guidance is unconditional, not contingent on this run's sets.

    An earlier version warned only when the populations happened to differ.
    That teaches the reader the subtraction is fine by default and wrong
    occasionally; it is wrong in principle, because rtf is undefined for any
    utterance whose detector never fired.
    """
    assert "must not be subtracted" in Q6_LIMITATION
    assert "descriptive only" in Q6_LIMITATION
    assert "streaming_cost" in Q6_LIMITATION


def test_q6_leads_with_the_paired_statistic_and_its_sample_size():
    from onevoice.benchmark import q6_statement
    from onevoice.eval.manifest import RunRecord

    paired = RunRecord(
        utt_id="a",
        hyp_text="x",
        audio_duration_s=10.0,
        timing={"stream_start": 0.0, "speech_end": 2.0, "tts_done": 5.0},
    )
    unpairable = RunRecord(  # detector never fired: pipeline_rtf but no rtf
        utt_id="b",
        hyp_text="",
        audio_duration_s=10.0,
        timing={"stream_start": 0.0, "tts_done": 0.2},
    )
    text = q6_statement([paired, unpairable])
    assert "Measured (paired)" in text
    assert "paired_n=1" in text
    assert "1 could not be paired" in text
    assert "40.0% of real processing cost" in text


def test_q6_reports_a_batch_shaped_run_as_hiding_nothing():
    from onevoice.benchmark import q6_statement
    from onevoice.eval.manifest import RunRecord

    records = [
        RunRecord(
            utt_id=str(i),
            hyp_text="x",
            audio_duration_s=10.0,
            timing={"stream_start": 2.0, "speech_end": 2.0, "tts_done": 5.0},
        )
        for i in range(2)
    ]
    text = q6_statement(records)
    assert "nothing" in text and "paired_n=2" in text
