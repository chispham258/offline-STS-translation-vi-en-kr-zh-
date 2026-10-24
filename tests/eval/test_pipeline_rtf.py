"""F2: RTF must be able to see work done inside push().

`rtf` measures `speech_end -> tts_done`. Every Phase 4 model except Silero is
batch-shaped, so nothing significant happened before `speech_end` and the
exclusion was harmless. A streaming component breaks that: Silero runs a network
on every 512-sample window inside `push()`, entirely outside the measured
window, and the published Q6 statement asserted the exclusion was safe.

`pipeline_rtf` measures `stream_start -> tts_done` and therefore includes it.
Both are reported. The existing metric is not redefined — D45 is a correctness
exception scoped to *adding* a measurement, and these tests pin that scope.
"""

from __future__ import annotations

from onevoice.eval.manifest import RunRecord
from onevoice.eval.metrics.timing import (
    pipeline_rtf,
    pipeline_rtf_summary,
    rtf,
    rtf_summary,
)


def _record(stream_start, speech_end, tts_done, duration=10.0, utt="u1"):
    return RunRecord(
        utt_id=utt,
        hyp_text="x",
        status="ok",
        audio_duration_s=duration,
        timing={
            "stream_start": stream_start,
            "speech_end": speech_end,
            "tts_done": tts_done,
        },
    )


# --- the new metric ----------------------------------------------------------


def test_pipeline_rtf_spans_the_whole_run_not_just_the_tail():
    # 2s of pre-speech-end work + 3s after, over 10s of audio.
    record = _record(stream_start=0.0, speech_end=2.0, tts_done=5.0)
    assert rtf(record) == 0.3  # (5-2)/10 - unchanged
    assert pipeline_rtf(record) == 0.5  # (5-0)/10 - includes the 2s


def test_pipeline_rtf_equals_rtf_when_nothing_precedes_speech_end():
    """A purely batch-shaped configuration must show no gap.

    If this ever fails, the new metric is measuring something other than
    per-frame work and the comparison between the two is meaningless.
    """
    record = _record(stream_start=1.0, speech_end=1.0, tts_done=4.0)
    assert pipeline_rtf(record) == rtf(record)


def test_pipeline_rtf_is_never_smaller_than_rtf():
    """stream_start precedes speech_end by construction."""
    for start, end in ((0.0, 0.5), (0.0, 3.0), (2.0, 2.0)):
        record = _record(stream_start=start, speech_end=end, tts_done=9.0)
        assert pipeline_rtf(record) >= rtf(record)


def test_missing_stamps_are_unmeasurable_not_zero():
    # Same discipline as rtf(): a missing stamp must not collapse to a
    # fast-looking number and drag a percentile down.
    empty = RunRecord(utt_id="u", hyp_text="", audio_duration_s=1.0, timing={})
    assert pipeline_rtf(empty) is None
    record = _record(0.0, 1.0, 5.0)
    record.timing.pop("stream_start")
    assert pipeline_rtf(record) is None


def test_inconsistent_stamps_are_unmeasurable():
    assert pipeline_rtf(_record(stream_start=5.0, speech_end=1.0, tts_done=2.0)) is None


def test_zero_duration_is_unmeasurable():
    assert pipeline_rtf(_record(0.0, 1.0, 5.0, duration=0.0)) is None


def test_summary_reports_percentiles_and_unmeasurable_count():
    records = [
        _record(0.0, 1.0, 5.0, utt="a"),
        _record(0.0, 1.0, 9.0, utt="b"),
        RunRecord(utt_id="c", hyp_text="", audio_duration_s=1.0, timing={}),
    ]
    result = pipeline_rtf_summary(records)
    assert result.name == "pipeline_rtf"
    assert result.detail["measured"] == 2
    assert result.detail["unmeasurable"] == 1
    assert "p50" in result.detail and "p95" in result.detail


# --- D45 scope: the existing metric must not move ----------------------------


def test_existing_rtf_is_untouched_by_the_addition():
    """D45 permits adding a measurement, not redefining one.

    Every Phase 4 figure was recorded under this definition. If it changes,
    those figures silently stop meaning what their reports say.
    """
    record = _record(stream_start=0.0, speech_end=2.0, tts_done=5.0)
    assert rtf(record) == (5.0 - 2.0) / 10.0


def test_existing_rtf_summary_keeps_its_name_and_shape():
    records = [_record(0.0, 1.0, 5.0, utt="a"), _record(0.0, 2.0, 6.0, utt="b")]
    result = rtf_summary(records)
    assert result.name == "rtf"
    assert set(result.detail) >= {
        "measured",
        "unmeasurable",
        "p50",
        "p95",
        "max",
        "mean",
    }


# --- the paired statistic is the one that means something --------------------


def test_streaming_cost_is_paired_and_reports_its_sample_size():
    """Only utterances carrying both stamps can be paired.

    rtf and pipeline_rtf percentiles describe different populations whenever a
    VAD failed to fire, so the difference of their medians is not a quantity.
    The per-utterance difference is, and its n must travel with it.
    """
    from onevoice.eval.metrics.timing import streaming_cost_summary

    paired = _record(stream_start=0.0, speech_end=2.0, tts_done=5.0, utt="a")
    also = _record(stream_start=0.0, speech_end=3.0, tts_done=5.0, utt="b")
    unpairable = RunRecord(  # no speech_end: pipeline_rtf only
        utt_id="c",
        hyp_text="",
        audio_duration_s=10.0,
        timing={"stream_start": 0.0, "tts_done": 0.2},
    )

    result = streaming_cost_summary([paired, also, unpairable])
    assert result.name == "streaming_cost"
    assert result.detail["paired_n"] == 2
    assert result.detail["unpairable"] == 1
    # a hides 2s, b hides 3s, over 10s of audio -> 0.20 and 0.30
    assert result.value == 0.20  # nearest-rank p50
    assert result.detail["max"] == 0.30


def test_streaming_cost_reports_the_share_of_processing_that_rtf_cannot_see():
    """The percentage a reader needs: how much of real cost rtf misses."""
    from onevoice.eval.metrics.timing import streaming_cost_summary

    # stream_start=0, speech_end=2, tts_done=5 -> pipeline 0.5, rtf 0.3.
    # Hidden share of total processing = 2/5 = 40%.
    result = streaming_cost_summary([_record(0.0, 2.0, 5.0)])
    assert result.detail["p50_pct_of_pipeline"] == 40.0


def test_streaming_cost_is_zero_for_a_batch_shaped_run():
    from onevoice.eval.metrics.timing import streaming_cost_summary

    result = streaming_cost_summary([_record(2.0, 2.0, 5.0, utt="a")])
    assert result.value == 0.0
    assert result.detail["paired_n"] == 1


def test_streaming_cost_with_nothing_pairable_is_not_a_number():
    import math

    from onevoice.eval.metrics.timing import streaming_cost_summary

    unpairable = RunRecord(
        utt_id="c", hyp_text="", audio_duration_s=10.0, timing={"stream_start": 0.0}
    )
    result = streaming_cost_summary([unpairable])
    assert result.detail["paired_n"] == 0
    assert math.isnan(result.value)


# --- naming stays stable as the module grows ---------------------------------


def test_metric_naming():
    """Guard the convention documented in `timing.py`'s module docstring.

    `rtf` and `pipeline_rtf` are ratios over named windows; `streaming_cost` is
    a derived statistic. A name like `true_rtf` or `actual_rtf` implies the
    other ratios are wrong — they are not, they answer different questions —
    and readers who believe it will discard a sound figure.
    """
    import inspect

    from onevoice.eval.metrics import timing

    banned = ("true_", "actual_", "real_", "correct_")
    public = [
        name
        for name, obj in inspect.getmembers(timing, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == timing.__name__
    ]

    for name in public:
        assert not any(name.startswith(p) for p in banned), (
            f"{name!r} implies the other ratios are incorrect. Each is correct "
            "over its own window — name a new window `<window>_rtf` and a new "
            "derived quantity with a noun. See timing.py's docstring."
        )

    # A window metric is a primitive measurement; a derived metric compares
    # primitive measurements. A name carrying both concepts - pipeline_cost_rtf,
    # pipeline_difference_rtf - describes two things at once, and one of them
    # belongs at the other level. Keeping the levels disjoint is what makes the
    # taxonomy closed rather than merely conventional.
    comparison_words = ("cost", "diff", "difference", "delta", "gap", "ratio_of")
    for name in public:
        if name.endswith("rtf") or name.endswith("rtf_summary"):
            assert not any(w in name for w in comparison_words), (
                f"{name!r} mixes a window with a comparison. A *_rtf name is a "
                "primitive measurement over a window; a comparison between "
                "windows is a derived metric and takes a noun."
            )

    # The documented hierarchy must actually exist, so the docstring cannot
    # drift away from the code it describes.
    assert {"rtf", "pipeline_rtf", "streaming_cost_summary"} <= set(public)


def test_the_naming_convention_is_documented_where_it_is_used():
    from onevoice.eval.metrics import timing

    doc = timing.__doc__ or ""
    assert "Naming" in doc
    assert "true_" in doc and "actual_" in doc  # the banned forms are named
    assert "paired_n" in doc
