"""Reading wav files and slicing them into frames for the pipeline."""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from onevoice.types import AudioChunk

DEFAULT_FRAME_MS = 32


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit mono wav as float32 in [-1, 1], with its sample rate."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no such audio file: {target}")
    with wave.open(str(target), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return samples / 32768.0, rate


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    """Write float32 [-1, 1] as 16-bit mono PCM, creating parent directories.

    Clips before casting. Without that, a sample above 1.0 wraps around on the
    int16 cast and a loud positive sample becomes a loud negative one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def chunk_audio(
    pcm: np.ndarray,
    sample_rate: int,
    frame_ms: int = DEFAULT_FRAME_MS,
) -> Iterator[AudioChunk]:
    """Slice audio into fixed-duration frames.

    Returns an iterator, not a sequence. D15 binds *components* - push and flush
    must return concrete sequences because a lazy return would defer state
    mutation. This is a pure generator over immutable input with no state to
    mutate, so streaming is correct and avoids materialising a whole utterance.

    The final partial frame is kept. Dropping it would truncate up to frame_ms
    of audio, and a speech_end landing in that window would silently vanish.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    for start in range(0, len(pcm), frame_len):
        frame = pcm[start : start + frame_len]
        # A view into the caller's array. `chunk.pcm[:] = 0` or a numpy
        # `out=` target would otherwise corrupt the caller's audio silently.
        frame.setflags(write=False)
        yield AudioChunk(
            pcm=frame,
            sample_rate=sample_rate,
            t_start=start / sample_rate,
        )
