"""Fixture audio is generated from the references, deterministically."""

import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.make_fixtures import speech_window, synthesize

FIXTURES = Path("fixtures/mini_vi_en")
SAMPLE_RATE = 16000


def test_speech_window_comes_from_the_reference_manifest():
    # The references are the contract; audio is generated to fit them.
    assert speech_window("utt-001") == pytest.approx((0.2, 1.8))
    assert speech_window("utt-002") == pytest.approx((0.3, 2.4))


def test_unknown_utterance_is_rejected():
    with pytest.raises(KeyError, match="utt-999"):
        speech_window("utt-999")


def test_synthesized_audio_is_silent_before_speech_starts():
    pcm = synthesize(2.0, 0.5, 1.5, SAMPLE_RATE)
    lead = pcm[: int(0.5 * SAMPLE_RATE)]
    assert np.abs(lead).max() == pytest.approx(0.0, abs=1e-9)


def test_synthesized_audio_is_silent_after_speech_ends():
    pcm = synthesize(2.0, 0.5, 1.5, SAMPLE_RATE)
    tail = pcm[int(1.5 * SAMPLE_RATE) :]
    assert np.abs(tail).max() == pytest.approx(0.0, abs=1e-9)


def test_synthesized_audio_is_loud_during_speech():
    pcm = synthesize(2.0, 0.5, 1.5, SAMPLE_RATE)
    speech = pcm[int(0.5 * SAMPLE_RATE) : int(1.5 * SAMPLE_RATE)]
    assert np.abs(speech).max() > 0.1


def test_synthesis_is_deterministic():
    # Regenerating must not drift from the committed fixtures.
    a = synthesize(1.0, 0.2, 0.8, SAMPLE_RATE)
    b = synthesize(1.0, 0.2, 0.8, SAMPLE_RATE)
    assert np.array_equal(a, b)


def test_generated_length_matches_the_requested_duration():
    pcm = synthesize(2.5, 0.1, 2.0, SAMPLE_RATE)
    assert len(pcm) == int(2.5 * SAMPLE_RATE)


@pytest.mark.parametrize("utt_id", ["utt-001", "utt-002", "utt-003"])
def test_the_committed_wav_exists_and_is_16k_mono(utt_id):
    path = FIXTURES / "audio" / f"{utt_id}.wav"
    assert path.is_file()
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getsampwidth() == 2


@pytest.mark.parametrize("utt_id", ["utt-001", "utt-002", "utt-003"])
def test_the_committed_wav_is_long_enough_for_its_speech_window(utt_id):
    _, end = speech_window(utt_id)
    path = FIXTURES / "audio" / f"{utt_id}.wav"
    with wave.open(str(path), "rb") as handle:
        duration = handle.getnframes() / handle.getframerate()
    assert duration >= end
