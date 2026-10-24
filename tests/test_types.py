"""Clock injection and timing traces."""

import pytest

from onevoice.types import FakeClock, SystemClock, TimingTrace


def test_fake_clock_starts_at_zero_and_advances():
    clock = FakeClock()
    assert clock.now() == 0.0
    clock.advance(1.5)
    assert clock.now() == 1.5


def test_system_clock_is_monotonic():
    clock = SystemClock()
    first = clock.now()
    second = clock.now()
    assert second >= first


def test_trace_records_events_in_order():
    clock = FakeClock()
    trace = TimingTrace(clock)
    trace.stamp("speech_end")
    clock.advance(0.4)
    trace.stamp("tts_first_audio")
    assert trace.events == ("speech_end", "tts_first_audio")


def test_elapsed_ms_is_exact():
    clock = FakeClock()
    trace = TimingTrace(clock)
    trace.stamp("speech_end")
    clock.advance(1.234)
    trace.stamp("tts_first_audio")
    assert trace.elapsed_ms("speech_end", "tts_first_audio") == pytest.approx(1234.0)


def test_elapsed_ms_rejects_unknown_event():
    trace = TimingTrace(FakeClock())
    trace.stamp("speech_end")
    with pytest.raises(KeyError, match="tts_first_audio"):
        trace.elapsed_ms("speech_end", "tts_first_audio")


def test_duplicate_stamp_is_rejected():
    trace = TimingTrace(FakeClock())
    trace.stamp("speech_end")
    with pytest.raises(ValueError, match="speech_end"):
        trace.stamp("speech_end")


def test_roundtrips_through_dict():
    clock = FakeClock()
    trace = TimingTrace(clock)
    trace.stamp("speech_end")
    clock.advance(0.5)
    trace.stamp("tts_first_audio")
    restored = TimingTrace.from_dict(trace.to_dict())
    assert restored.to_dict() == trace.to_dict()
    assert restored.elapsed_ms("speech_end", "tts_first_audio") == pytest.approx(500.0)


def test_restored_trace_is_read_only_by_default():
    clock = FakeClock()
    trace = TimingTrace(clock)
    trace.stamp("speech_end")
    restored = TimingTrace.from_dict(trace.to_dict())
    with pytest.raises(ValueError, match="clock"):
        restored.stamp("tts_first_audio")


def test_restored_trace_accepts_an_explicit_clock_and_can_continue():
    clock = FakeClock()
    trace = TimingTrace(clock)
    trace.stamp("speech_end")
    restored = TimingTrace.from_dict(trace.to_dict(), clock=clock)
    clock.advance(0.25)
    restored.stamp("tts_first_audio")
    assert restored.elapsed_ms("speech_end", "tts_first_audio") == pytest.approx(250.0)


def test_record_stores_a_caller_supplied_value():
    trace = TimingTrace(FakeClock())
    trace.record("speech_end_audio", 1.8)
    assert trace.get("speech_end_audio") == pytest.approx(1.8)
    assert trace.to_dict()["speech_end_audio"] == pytest.approx(1.8)


def test_record_appends_duplicate_values():
    # Async pipeline records per-chunk/per-segment values under one key; the
    # trace keeps them as a list so the driver can collapse them later.
    trace = TimingTrace(FakeClock())
    trace.record("speech_end_audio", 1.8)
    trace.record("speech_end_audio", 2.0)
    assert trace.get("speech_end_audio") == [1.8, 2.0]


def test_record_works_on_a_read_only_restored_trace():
    # record() takes no clock reading, so the read-only guard must not block it.
    restored = TimingTrace.from_dict({"speech_end": 1.0})
    restored.record("speech_end_audio", 1.8)
    assert restored.get("speech_end_audio") == pytest.approx(1.8)
