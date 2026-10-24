"""Generate the deterministic audio fixtures the dummy pipeline runs on.

D19: stdlib `wave` plus numpy. soundfile and scipy both pull binary wheels for
a job that is a few dozen lines.

The speech windows are read from fixtures/mini_vi_en/refs.jsonl rather than
hardcoded here. The references are the contract (D10) and the audio is generated
to fit them; if the two could drift, the boundary assertion would measure
nothing.

Usage: uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from onevoice.audio import write_wav

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "mini_vi_en"
REFS = FIXTURES / "refs.jsonl"
AUDIO_DIR = FIXTURES / "audio"

SAMPLE_RATE = 16000
TONE_HZ = 220.0
AMPLITUDE = 0.5
TAIL_PAD_S = 0.2


@lru_cache(maxsize=1)
def _references() -> dict[str, dict]:
    lines = REFS.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["utt_id"]: row for row in rows}


def speech_window(utt_id: str) -> tuple[float, float]:
    """Declared speech boundaries for an utterance, from the reference manifest."""
    try:
        row = _references()[utt_id]
    except KeyError:
        known = ", ".join(sorted(_references()))
        raise KeyError(f"no reference for {utt_id!r}; known: {known}") from None
    return float(row["speech_start_s"]), float(row["speech_end_s"])


def synthesize(
    duration_s: float,
    speech_start_s: float,
    speech_end_s: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """A silent buffer with one tone burst occupying the speech window.

    Deterministic: a fixed-frequency tone, no randomness, so regenerating never
    drifts from the committed fixtures.
    """
    total = int(duration_s * sample_rate)
    pcm = np.zeros(total, dtype=np.float32)

    start = int(speech_start_s * sample_rate)
    end = min(int(speech_end_s * sample_rate), total)
    if end <= start:
        return pcm

    t = np.arange(end - start, dtype=np.float32) / sample_rate
    pcm[start:end] = AMPLITUDE * np.sin(2.0 * np.pi * TONE_HZ * t)
    return pcm


def main() -> int:
    for utt_id in sorted(_references()):
        start, end = speech_window(utt_id)
        pcm = synthesize(end + TAIL_PAD_S, start, end)
        path = AUDIO_DIR / f"{utt_id}.wav"
        write_wav(path, pcm, SAMPLE_RATE)
        print(f"wrote {path} ({len(pcm) / SAMPLE_RATE:.2f}s, speech {start}-{end}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
