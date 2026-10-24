"""Streaming faster-whisper ASR: regression tests that need no model weights."""

import sys
import types

import numpy as np
import pytest

from onevoice.components.faster_whisper_asr_streaming import (
    FasterWhisperASRStreaming,
)
from onevoice.types import AsrEvent, AudioChunk


def _component(monkeypatch) -> FasterWhisperASRStreaming:
    """Build the wrapper with a fake WhisperModel constructor."""
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return FasterWhisperASRStreaming(model_name="tiny", src_lang="en")


def _chunk(seconds: float) -> AudioChunk:
    return AudioChunk(
        pcm=np.zeros(int(seconds * 16000), dtype=np.float32),
        sample_rate=16000,
        t_start=0.0,
    )


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str = "hello") -> None:
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = -0.3
        self.compression_ratio = 1.1
        self.no_speech_prob = 0.01


class _FakeModel:
    """Records the audio length of every transcribe() call."""

    def __init__(self, emit_text: bool = True) -> None:
        self.calls: list[int] = []
        self.emit_text = emit_text

    def transcribe(self, audio, **kwargs):
        self.calls.append(len(audio))
        duration = len(audio) / 16000
        info = types.SimpleNamespace(
            duration=duration,
            duration_after_vad=duration,
            language_probability=1.0,
        )
        segments = [_FakeSegment(0.0, duration)] if self.emit_text else []
        return iter(segments), info


def test_pop_pending_marks_frozen_events(monkeypatch):
    """AsrEvent is frozen; marking is_final must copy, not mutate.

    Regression: assigning ev.is_final raised FrozenInstanceError inside the
    ASR worker thread, killing the task and deadlocking the pipeline.
    """
    asr = _component(monkeypatch)
    asr._pending_segments.append(
        AsrEvent(text="hello", t_audio_end=1.0, t_wall=0.0)
    )

    out = asr._pop_pending(is_final=True)

    assert len(out) == 1
    assert out[0].is_final is True
    assert out[0].text == "hello"
    assert asr._pending_segments == []


def test_pop_pending_keeps_partial_flag(monkeypatch):
    asr = _component(monkeypatch)
    asr._pending_segments.append(
        AsrEvent(text="partial", t_audio_end=2.0, t_wall=0.0, is_final=True)
    )

    out = asr._pop_pending(is_final=False)

    assert out[0].is_final is False


def test_window_is_trimmed_to_one_chunk(monkeypatch):
    """Window 5.06s must be sliced to 5.00s, not left as 5.00 + 0.06.

    Regression: the 0.06s remainder became a padded chunk that hallucinated
    and triggered temperature fallback, costing 6-12s per window.
    """
    asr = _component(monkeypatch)
    asr._chunk_batch_size = 1
    model = _FakeModel()
    asr._model = model

    asr.push(_chunk(5.06))

    assert model.calls == [int(5.0 * 16000)]
    assert asr._last_segment_end == pytest.approx(5.0)


def test_empty_window_advances_cursor(monkeypatch):
    """A window that yields no segment must not be decoded again."""
    asr = _component(monkeypatch)
    asr._chunk_batch_size = 1
    model = _FakeModel(emit_text=False)
    asr._model = model

    asr.push(_chunk(5.06))
    assert asr._last_segment_end == pytest.approx(5.0)

    asr.push(_chunk(0.5))
    assert len(model.calls) == 1


def test_flush_decodes_tail_as_one_window(monkeypatch):
    """Flush decodes the remaining <5s tail once, all events final."""
    asr = _component(monkeypatch)
    asr._chunk_batch_size = 1
    model = _FakeModel()
    asr._model = model

    asr.push(_chunk(5.4))
    out = asr.flush()

    assert model.calls == [int(5.0 * 16000), int(0.4 * 16000)]
    assert out
    assert all(event.is_final for event in out)
