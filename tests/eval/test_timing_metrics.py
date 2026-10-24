"""Latency and RTF derived from run records."""

import pytest

from onevoice.eval.manifest import RunRecord
from onevoice.eval.metrics.timing import (
    end_to_speech_ms,
    latency_summary,
    rtf,
    rtf_summary,
)


def _record(utt_id: str, timing: dict, duration: float = 2.0) -> RunRecord:
    return RunRecord(
        utt_id=utt_id, hyp_text="x", audio_duration_s=duration, timing=timing
    )


def test_end_to_speech_is_first_audio_minus_speech_end():
    rec = _record("a", {"speech_end": 1.80, "tts_first_audio": 2.75})
    assert end_to_speech_ms(rec) == pytest.approx(950.0)


def test_missing_stamp_returns_none_not_zero():
    assert end_to_speech_ms(_record("a", {"speech_end": 1.0})) is None
    assert end_to_speech_ms(_record("a", {})) is None


def test_rtf_is_processing_time_over_audio_duration():
    rec = _record("a", {"speech_end": 1.0, "tts_done": 3.0}, duration=4.0)
    assert rtf(rec) == pytest.approx(0.5)


def test_rtf_returns_none_when_duration_is_zero():
    rec = _record("a", {"speech_end": 1.0, "tts_done": 3.0}, duration=0.0)
    assert rtf(rec) is None


def test_end_to_speech_returns_none_when_stamps_are_inverted():
    # A negative interval means the stamps are inconsistent, not that the
    # pipeline was instant; it must not drag the corpus mean down.
    rec = _record("a", {"speech_end": 2.0, "tts_first_audio": 1.5})
    assert end_to_speech_ms(rec) is None


def test_rtf_returns_none_when_stamps_are_inverted():
    rec = _record("a", {"speech_end": 2.0, "tts_done": 1.5}, duration=4.0)
    assert rtf(rec) is None


def test_inverted_stamp_counts_as_unmeasurable_in_summary():
    records = [
        _record("a", {"speech_end": 0.0, "tts_first_audio": 0.5}),
        _record("b", {"speech_end": 2.0, "tts_first_audio": 1.5}),
        _record("c", {"speech_end": 0.0, "tts_first_audio": 0.9}),
    ]
    result = latency_summary(records)
    assert result.detail["measured"] == 2
    assert result.detail["unmeasurable"] == 1


def test_latency_summary_reports_p50_and_p95():
    records = [
        _record(f"u{i}", {"speech_end": 0.0, "tts_first_audio": i / 1000.0})
        for i in range(1, 101)
    ]
    result = latency_summary(records)
    assert result.detail["p50"] == pytest.approx(50.0, abs=1.0)
    assert result.detail["p95"] == pytest.approx(95.0, abs=1.0)
    assert result.value == result.detail["p50"]


def test_summary_counts_records_it_could_not_measure():
    records = [
        _record("a", {"speech_end": 0.0, "tts_first_audio": 1.0}),
        _record("b", {"speech_end": 0.0}),
    ]
    result = latency_summary(records)
    assert result.detail["measured"] == 1
    assert result.detail["unmeasurable"] == 1


def test_summary_of_nothing_measurable_is_nan_not_zero():
    result = latency_summary([_record("a", {})])
    assert result.value != result.value  # NaN
    assert result.detail["measured"] == 0


def test_rtf_summary_reports_the_worst_case():
    records = [
        _record("a", {"speech_end": 0.0, "tts_done": 1.0}, duration=4.0),
        _record("b", {"speech_end": 0.0, "tts_done": 3.0}, duration=4.0),
    ]
    result = rtf_summary(records)
    assert result.detail["max"] == pytest.approx(0.75)
