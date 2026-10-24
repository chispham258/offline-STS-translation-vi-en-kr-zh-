"""A bounded replay buffer for audio a VAD has not yet committed to (Q7).

`EnergyVAD` decides on the frame in front of it, so its `speech_start` carries
that frame's own `t_start` and gating ASR on "is the detector open right now"
happens to be correct. Any detector that needs context before committing is
different: it reports `speech_start` several frames late, with `t_audio`
pointing *backwards* at where speech actually began. Those frames have already
been consumed by the loop and dropped, so ASR starts mid-word on every
utterance — and nothing says so. The transcript is merely worse.

This retains the recent past so the driver can replay `[t_audio, now]` when the
detector finally commits. It holds `AudioChunk`s by reference; frames from
`chunk_audio` are read-only views over the caller's array, so retaining them
costs bookkeeping rather than a copy.

Deliberately not a component: it has no `push`/`flush`/`reset` triple (D6), is
not registered, and is not swappable. It is driver plumbing.
"""

from __future__ import annotations

from collections import deque

from onevoice.types import AudioChunk

#: Retained audio, in milliseconds. Must exceed the largest lookahead of any
#: detector this framework runs, with headroom — the cost of being generous is
#: bytes, and the cost of being tight is a raised utterance mid-benchmark.
#: Silero decides on 32 ms windows and pads onsets by ~30 ms by default; 500 ms
#: is roughly an order of magnitude above that.
DEFAULT_LOOKBACK_MS = 500


class LookbackUnderrun(RuntimeError):
    """A detector asked for audio older than the window retained.

    Raised rather than replaying a short prefix. A truncated replay produces a
    clipped utterance, a plausible-looking transcript, and a WER nobody can
    attribute — the failure mode this project keeps finding. An utterance that
    raises is marked errored, excluded from scoring, and counted in the report.
    """


class LookbackBuffer:
    """Retains the most recent `window_ms` of audio for replay."""

    def __init__(self, window_ms: int = DEFAULT_LOOKBACK_MS) -> None:
        if window_ms <= 0:
            raise ValueError(f"window_ms must be positive, got {window_ms!r}")
        self.window_ms = window_ms
        self._frames: deque[AudioChunk] = deque()
        self._retained_s = 0.0
        # End time of the newest frame evicted so far. `since` uses it to tell
        # "we never had that audio" apart from "the stream started here".
        self._evicted_until: float | None = None

    def append(self, chunk: AudioChunk) -> None:
        """Retain a frame, evicting whatever has aged out of the window."""
        self._frames.append(chunk)
        self._retained_s += chunk.duration_s
        limit = self.window_ms / 1000.0
        while self._frames and self._retained_s - self._frames[0].duration_s > limit:
            dropped = self._frames.popleft()
            self._retained_s -= dropped.duration_s
            self._evicted_until = dropped.t_start + dropped.duration_s

    def since(self, t_audio: float) -> list[AudioChunk]:
        """Frames overlapping `[t_audio, now]`, in arrival order.

        A frame is included when it *ends* after `t_audio`, so a boundary
        falling mid-frame replays that whole frame: half a frame of speech is
        still speech, and handing ASR a partial frame to save 10 ms of audio
        would be a strange trade.

        "After" is measured with half a sample of tolerance. Both `t_start` and
        `duration_s` are quotients by the sample rate, so the previous frame's
        computed end lands a float whisker above the next frame's start
        (0.288 + 0.032 == 0.32000000000000004). Without the tolerance a
        zero-lookahead detector replays one frame more than it should, which
        would have silently shifted every measurement taken so far.
        """
        if self._evicted_until is not None and t_audio < self._evicted_until:
            raise LookbackUnderrun(
                f"speech_start reports t_audio={t_audio:.3f}s but the lookback "
                f"window retains only from {self._evicted_until:.3f}s "
                f"({self.window_ms} ms). Replaying what is left would hand ASR "
                "a clipped utterance and report it as a normal result; raise "
                "DEFAULT_LOOKBACK_MS above the detector's lookahead instead."
            )
        return [
            c
            for c in self._frames
            if c.t_start + c.duration_s > t_audio + 0.5 / c.sample_rate
        ]

    def retained_frames(self) -> int:
        """How many frames are currently held. For tests and diagnostics."""
        return len(self._frames)

    def reset(self) -> None:
        """Drop everything. Called per utterance."""
        self._frames.clear()
        self._retained_s = 0.0
        self._evicted_until = None
