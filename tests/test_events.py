"""Audio and event value types."""

import numpy as np
import pytest

from onevoice.types import AsrEvent, AudioChunk, MtEvent, TtsEvent, VadEvent


def _pcm(n: int = 512) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def test_audio_chunk_carries_pcm_rate_and_offset():
    chunk = AudioChunk(pcm=_pcm(), sample_rate=16000, t_start=1.5)
    assert chunk.sample_rate == 16000
    assert chunk.t_start == 1.5
    assert chunk.pcm.shape == (512,)


def test_audio_chunk_reports_its_duration():
    chunk = AudioChunk(pcm=_pcm(16000), sample_rate=16000, t_start=0.0)
    assert chunk.duration_s == pytest.approx(1.0)


def test_audio_chunk_is_frozen():
    chunk = AudioChunk(pcm=_pcm(), sample_rate=16000, t_start=0.0)
    with pytest.raises(AttributeError):
        chunk.sample_rate = 8000


def test_audio_chunk_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        AudioChunk(pcm=_pcm(), sample_rate=0, t_start=0.0)


def test_comparing_two_chunks_does_not_raise():
    # A generated __eq__ would compare the arrays elementwise and blow up on
    # "truth value of an array is ambiguous". Identity comparison is correct.
    a = AudioChunk(pcm=_pcm(), sample_rate=16000, t_start=0.0)
    b = AudioChunk(pcm=_pcm(), sample_rate=16000, t_start=0.0)
    assert a != b
    assert a == a
    assert a in [a, b]


def test_vad_event_kind_is_constrained():
    assert VadEvent(kind="speech_start", t_audio=0.5).kind == "speech_start"
    with pytest.raises(ValueError, match="speech_middle"):
        VadEvent(kind="speech_middle", t_audio=0.5)


def test_asr_event_defaults_to_non_final_with_no_confidence():
    event = AsrEvent(text="xin ch", t_audio_end=0.8, t_wall=1.0)
    assert event.is_final is False
    assert event.confidence is None


def test_asr_event_can_be_final_with_confidence():
    event = AsrEvent(
        text="xin chào", is_final=True, confidence=0.91, t_audio_end=1.2, t_wall=1.4
    )
    assert event.is_final is True
    assert event.confidence == pytest.approx(0.91)


def test_mt_event_mirrors_asr_event_shape():
    event = MtEvent(text="hello", is_final=True, t_wall=1.6)
    assert event.text == "hello"
    assert event.is_final is True


def test_tts_event_carries_audio_not_text():
    event = TtsEvent(pcm=_pcm(320), sample_rate=16000, is_final=False, t_wall=1.8)
    assert event.pcm.shape == (320,)
    assert event.sample_rate == 16000


def test_confidence_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        AsrEvent(text="x", confidence=1.5, t_audio_end=0.1, t_wall=0.2)
