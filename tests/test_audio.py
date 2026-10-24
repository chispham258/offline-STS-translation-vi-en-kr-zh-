"""Reading wavs and slicing them into frames."""

import numpy as np
import pytest

from onevoice.audio import chunk_audio, read_wav, write_wav
from onevoice.types import AudioChunk

SAMPLE_RATE = 16000


def test_write_then_read_round_trips(tmp_path):
    original = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    path = tmp_path / "round.wav"
    write_wav(path, original, SAMPLE_RATE)
    pcm, rate = read_wav(path)
    assert rate == SAMPLE_RATE
    assert pcm.dtype == np.float32
    assert len(pcm) == len(original)
    # 16-bit quantisation, so approximate rather than exact.
    assert np.abs(pcm - original).max() < 1e-3


def test_read_normalises_into_minus_one_to_one(tmp_path):
    path = tmp_path / "loud.wav"
    write_wav(path, np.ones(400, dtype=np.float32), SAMPLE_RATE)
    pcm, _ = read_wav(path)
    assert np.abs(pcm).max() <= 1.0


def test_write_clips_out_of_range_input(tmp_path):
    # Values outside [-1, 1] would wrap around on int16 cast and turn a loud
    # sample into a loud sample of the opposite sign.
    path = tmp_path / "clip.wav"
    write_wav(path, np.array([2.0, -2.0], dtype=np.float32), SAMPLE_RATE)
    pcm, _ = read_wav(path)
    assert pcm[0] > 0.99
    assert pcm[1] < -0.99


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_wav(tmp_path / "nope.wav")


def test_chunks_are_audio_chunks_of_the_frame_size():
    pcm = np.zeros(SAMPLE_RATE, dtype=np.float32)
    chunks = list(chunk_audio(pcm, SAMPLE_RATE, frame_ms=32))
    assert all(isinstance(c, AudioChunk) for c in chunks)
    assert len(chunks[0].pcm) == 512


def test_chunk_t_start_advances_by_the_frame_duration():
    pcm = np.zeros(SAMPLE_RATE, dtype=np.float32)
    chunks = list(chunk_audio(pcm, SAMPLE_RATE, frame_ms=32))
    assert chunks[0].t_start == pytest.approx(0.0)
    assert chunks[1].t_start == pytest.approx(0.032)
    assert chunks[2].t_start == pytest.approx(0.064)


def test_the_final_partial_frame_is_kept():
    # Dropping it would truncate up to 32 ms, and a speech_end landing in that
    # window would silently vanish - corrupting the boundary D10 checks.
    pcm = np.zeros(700, dtype=np.float32)
    chunks = list(chunk_audio(pcm, SAMPLE_RATE, frame_ms=32))
    assert len(chunks) == 2
    assert len(chunks[1].pcm) == 700 - 512


def test_chunks_reconstruct_the_original_audio():
    pcm = np.arange(1000, dtype=np.float32) / 1000.0
    rebuilt = np.concatenate(
        [c.pcm for c in chunk_audio(pcm, SAMPLE_RATE, frame_ms=32)]
    )
    assert np.array_equal(rebuilt, pcm)


def test_empty_audio_yields_no_chunks():
    assert list(chunk_audio(np.zeros(0, dtype=np.float32), SAMPLE_RATE)) == []


def test_chunk_pcm_is_read_only():
    # FIX 6: chunk.pcm is a view into the caller's array. The frozen dataclass
    # blocks `chunk.pcm *= 0.5`, but an in-place write like `chunk.pcm[:] = 0`
    # would silently corrupt the caller's audio without this.
    pcm = np.zeros(SAMPLE_RATE, dtype=np.float32)
    chunk = next(chunk_audio(pcm, SAMPLE_RATE, frame_ms=32))
    with pytest.raises(ValueError):
        chunk.pcm[:] = 1.0
