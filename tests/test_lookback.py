"""Q7: a lookahead VAD must not lose the leading frames of every utterance.

`EnergyVAD` decides on the frame in front of it, so its `speech_start` carries
that frame's own `t_start` and the driver's gate happens to be correct. A real
VAD needs context before it will commit, so it reports `speech_start` while
`t_audio` points *backwards* at audio the driver has already consumed and
dropped. Every utterance then reaches ASR clipped, mid-word, and nothing in the
run record says so — the transcript is merely worse.

These tests pin the fix and, just as importantly, pin that the fix is a no-op
for a zero-lookahead detector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from onevoice.audio import DEFAULT_FRAME_MS
from onevoice.components.dummy import EchoASR, EnergyVAD, IdentityMT, SilenceTTS
from onevoice.eval.manifest import RefRecord
from onevoice.interfaces import BaseVAD
from onevoice.lookback import DEFAULT_LOOKBACK_MS, LookbackBuffer, LookbackUnderrun
from onevoice.pipeline import run_utterance
from onevoice.types import AudioChunk, FakeClock, VadEvent

RATE = 16000
FRAME_MS = 20


def _chunk(index: int, value: float = 0.5) -> AudioChunk:
    n = int(RATE * FRAME_MS / 1000)
    return AudioChunk(
        pcm=np.full(n, value, dtype=np.float32),
        sample_rate=RATE,
        t_start=index * FRAME_MS / 1000,
    )


# --- the buffer itself -------------------------------------------------------


def test_replays_the_frame_containing_the_requested_time():
    buf = LookbackBuffer()
    for i in range(5):
        buf.append(_chunk(i))
    replayed = buf.since(_chunk(3).t_start)
    assert [c.t_start for c in replayed] == [0.06, 0.08]


def test_replay_is_in_arrival_order():
    buf = LookbackBuffer()
    for i in range(6):
        buf.append(_chunk(i))
    replayed = buf.since(0.0)
    assert [c.t_start for c in replayed] == sorted(c.t_start for c in replayed)


def test_a_time_inside_a_frame_replays_that_whole_frame():
    """Half a frame of speech is still speech; ASR gets the frame."""
    buf = LookbackBuffer()
    for i in range(4):
        buf.append(_chunk(i))
    # 0.05 s falls inside frame 2, which spans [0.04, 0.06).
    assert buf.since(0.05)[0].t_start == pytest.approx(0.04)


def test_the_window_bounds_memory():
    """A streaming buffer that never evicts is a memory leak with extra steps."""
    buf = LookbackBuffer(window_ms=100)  # 5 frames at 20 ms
    for i in range(50):
        buf.append(_chunk(i))
    assert buf.retained_frames() <= 6


def test_asking_for_audio_older_than_the_window_raises():
    """The alternative is replaying a short prefix and reporting nothing.

    That yields a clipped utterance, a plausible transcript, and a WER nobody
    can explain. An utterance that raises is excluded and counted.
    """
    buf = LookbackBuffer(window_ms=100)
    for i in range(50):
        buf.append(_chunk(i))
    with pytest.raises(LookbackUnderrun, match="lookback window"):
        buf.since(0.0)


def test_no_underrun_before_anything_is_evicted():
    buf = LookbackBuffer(window_ms=1000)
    for i in range(3):
        buf.append(_chunk(i))
    assert len(buf.since(0.0)) == 3


def test_reset_clears_retained_audio():
    buf = LookbackBuffer()
    for i in range(4):
        buf.append(_chunk(i))
    buf.reset()
    assert buf.since(0.0) == []


def test_default_window_leaves_headroom_over_a_typical_vad_lookahead():
    assert DEFAULT_LOOKBACK_MS >= 300


# --- driver integration ------------------------------------------------------


class LookaheadVAD(BaseVAD):
    """Commits to speech only after `lookahead` frames, and says so honestly.

    Stands in for a real detector: `speech_start` is emitted late, with
    `t_audio` pointing back at where speech actually began.
    """

    def __init__(self, lookahead: int = 3, threshold: float = 0.05) -> None:
        self.lookahead = lookahead
        self.threshold = threshold
        self._pending: list[float] = []
        self._open = False
        self._last_t = 0.0

    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        self._check_not_flushed()
        self._last_t = chunk.t_start
        loud = float(np.abs(chunk.pcm).max(initial=0.0)) > self.threshold
        if loud and not self._open:
            self._pending.append(chunk.t_start)
            if len(self._pending) > self.lookahead:
                self._open = True
                started_at = self._pending[0]
                self._pending = []
                return [VadEvent(kind="speech_start", t_audio=started_at)]
            return []
        if not loud:
            self._pending = []
            if self._open:
                self._open = False
                return [VadEvent(kind="speech_end", t_audio=chunk.t_start)]
        return []

    def flush(self) -> Sequence[VadEvent]:
        self._mark_flushed()
        if not self._open:
            return []
        self._open = False
        return [VadEvent(kind="speech_end", t_audio=self._last_t)]

    def reset(self) -> None:
        super().reset()
        self._pending = []
        self._open = False
        self._last_t = 0.0


def _speech_wav(path, lead_silence_s=0.3, speech_s=1.0):
    silence = np.zeros(int(RATE * lead_silence_s), dtype=np.float32)
    t = np.arange(int(RATE * speech_s)) / RATE
    speech = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    from onevoice.audio import write_wav

    write_wav(path, np.concatenate([silence, speech]), RATE)
    return path


def _ref() -> RefRecord:
    return RefRecord(
        utt_id="utt-lookback",
        audio_path="audio/utt-lookback.wav",
        src_lang="vi",
        tgt_lang="en",
        src_text="x",
        ref_text="y",
        speech_start_s=0.3,
        speech_end_s=1.3,
    )


def _samples_reaching_asr(vad, tmp_path) -> int:
    root = tmp_path / "in"
    (root / "audio").mkdir(parents=True)
    _speech_wav(root / "audio" / "utt-lookback.wav")
    components = {
        "vad": vad,
        "asr": EchoASR(),
        "mt": IdentityMT(),
        "tts": SilenceTTS(),
    }
    record = run_utterance(
        _ref(), components, FakeClock(), tmp_path / "out", root
    )
    assert record.status == "ok", record.error
    # EchoASR reports what it consumed: "audio of N samples at R Hz".
    return int(record.asr_text.split()[2])


def test_lookahead_vad_keeps_the_frames_before_the_event(tmp_path):
    """The whole point of Q7.

    A 3-frame lookahead must not cost 3 frames of audio. Without the lookback
    buffer the late `speech_start` discards them and ASR starts mid-word.
    """
    lookahead = _samples_reaching_asr(LookaheadVAD(lookahead=3), tmp_path / "a")
    instant = _samples_reaching_asr(LookaheadVAD(lookahead=0), tmp_path / "b")
    frame = int(RATE * FRAME_MS / 1000)
    assert lookahead == instant, (
        f"lookahead VAD lost {(instant - lookahead) / frame:.0f} frames of audio"
    )


def test_zero_lookahead_detector_is_completely_unaffected(tmp_path):
    """The fix must not change `EnergyVAD`'s behaviour by a single sample.

    Every Phase 4 number so far was measured with `EnergyVAD`. If this moves,
    the fix silently invalidated the baseline.
    """
    before = 16192  # measured on this fixture with the pre-fix driver
    assert _samples_reaching_asr(EnergyVAD(), tmp_path) == before


class RecordingASR(EchoASR):
    """Remembers which frames it was handed, so double-pushes are visible."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[float] = []

    def push(self, chunk):
        self.seen.append(chunk.t_start)
        return super().push(chunk)

    def reset(self) -> None:
        super().reset()
        self.seen = []


def _seen_by_asr(vad, tmp_path) -> list[float]:
    root = tmp_path / "in"
    (root / "audio").mkdir(parents=True)
    _speech_wav(root / "audio" / "utt-lookback.wav")
    asr = RecordingASR()
    record = run_utterance(
        _ref(),
        {"vad": vad, "asr": asr, "mt": IdentityMT(), "tts": SilenceTTS()},
        FakeClock(),
        tmp_path / "out",
        root,
    )
    assert record.status == "ok", record.error
    return asr.seen


def test_no_frame_reaches_asr_twice(tmp_path):
    """Replaying `[t_audio, now]` and then also pushing `now` double-counts.

    Doubling the frame at every speech onset inflates the audio ASR sees and is
    nearly invisible in a transcript, so assert the property directly rather
    than reconstructing the detector's arithmetic.
    """
    for vad in (EnergyVAD(), LookaheadVAD(lookahead=3)):
        seen = _seen_by_asr(vad, tmp_path / f"{type(vad).__name__}")
        assert len(seen) == len(set(seen)), f"{type(vad).__name__} double-pushed"


def test_frames_reach_asr_in_order(tmp_path):
    """Replayed audio must not arrive after the frame that triggered it."""
    seen = _seen_by_asr(LookaheadVAD(lookahead=3), tmp_path)
    assert seen == sorted(seen)


def test_replayed_audio_starts_where_the_detector_said_speech_began(tmp_path):
    seen = _seen_by_asr(LookaheadVAD(lookahead=3), tmp_path)
    # Speech begins at 0.3 s; the frame containing it starts at or just below.
    assert seen[0] <= 0.3
    assert seen[0] > 0.3 - DEFAULT_FRAME_MS / 1000 - 1e-9
