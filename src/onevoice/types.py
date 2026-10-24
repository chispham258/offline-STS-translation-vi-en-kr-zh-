"""Core value types shared across the pipeline and the evaluator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# Timing event names shared between whatever stamps them (a Phase 3 driver)
# and whatever reads them (the evaluator's timing metrics). A driver that
# names an event differently would silently make every record unmeasurable,
# so both sides import the same constant instead of matching string literals.
SPEECH_END = "speech_end"
TTS_FIRST_AUDIO = "tts_first_audio"
TTS_DONE = "tts_done"

# Stage-attribution stamps, added so the ASR / MT / TTS split of end-to-speech
# latency is observable. Additive only: no metric reads them, no existing event
# moved, and an archived run that predates them stays scoreable because it
# simply lacks the keys. Same shape as pipeline_rtf being added beside an
# unchanged rtf (D45).
ASR_FIRST_PARTIAL = "asr_first_partial"
ASR_FINAL = "asr_final"
MT_DONE = "mt_done"

# Wall seconds actually spent inside the TTS component's own calls, summed
# across push() and flush(). Not a stamp: a duration, recorded rather than
# stamped. It exists because a synchronous TTS (Piper synthesizes in push())
# does its work inside the MT window, so mt_done -> tts_done reads ~0 and the
# stage split misattributes TTS time to MT. Observational only - no metric,
# delta or selection rule reads it. Absent when no synthesis ran.
TTS_SYNTHESIS_S = "tts_synthesis_s"


@runtime_checkable
class Clock(Protocol):
    """A source of monotonic time, in seconds."""

    def now(self) -> float: ...


class SystemClock:
    """Wraps time.perf_counter. Use in production."""

    def now(self) -> float:
        return time.perf_counter()


class FakeClock:
    """Deterministic clock for tests. Only advances when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class _ReadOnlyClock:
    """Placeholder clock for a trace restored without an explicit clock.

    `now()` is only ever called by `stamp()`. Raising here turns an attempt
    to stamp a read-only restored trace into an immediate, clear error
    instead of a silent mix of epochs.
    """

    def now(self) -> float:
        raise ValueError(
            "cannot stamp a trace restored by from_dict() without a clock: "
            "the default restore is read-only because a fresh SystemClock's "
            "epoch has no relation to whatever clock produced the stored "
            "values, and mixing clock epochs would silently produce a "
            "plausible but wrong elapsed_ms; pass from_dict(data, clock=...) "
            "with the clock that produced the data if you need to continue "
            "stamping"
        )


class TimingTrace:
    """Ordered, named timestamps taken from an injectable clock.

    Event names are recorded once each; re-stamping an event is a bug, not an
    update, so it raises rather than silently overwriting the first value.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._stamps: dict[str, float] = {}

    def stamp(self, event: str) -> float:
        if event in self._stamps:
            raise ValueError(f"event already stamped: {event}")
        value = self._clock.now()
        self._stamps[event] = value
        return value

    def record(self, event: str, value: float) -> None:
        """Store a caller-supplied value under `event`, no clock read involved.

        For values measured in a different timebase than this trace's clock
        (e.g. VAD boundaries in audio time, not wall-clock time), where
        `stamp()` would be the wrong tool. Takes no clock reading, so unlike
        `stamp()` it works on a trace restored read-only by `from_dict()`.

        If the same event is recorded multiple times (e.g., queue wait times
        for multiple items at the same stage), values are stored as a list.
        """
        if event in self._stamps:
            existing = self._stamps[event]
            if isinstance(existing, list):
                existing.append(value)
            else:
                self._stamps[event] = [existing, value]
        else:
            self._stamps[event] = value

    def get(self, event: str) -> float | None:
        return self._stamps.get(event)

    def elapsed_ms(self, start: str, end: str) -> float:
        for event in (start, end):
            if event not in self._stamps:
                raise KeyError(f"event not stamped: {event}")
        return (self._stamps[end] - self._stamps[start]) * 1000.0

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._stamps)

    def to_dict(self) -> dict[str, float]:
        return dict(self._stamps)

    @classmethod
    def from_dict(
        cls, data: dict[str, float], clock: Clock | None = None
    ) -> TimingTrace:
        """Restore a trace from a serialized dict of stamps.

        Read-only by default: without an explicit `clock`, the restored
        trace raises on `stamp()` rather than silently mixing epochs. A
        fresh default clock's epoch has no relation to whatever clock
        produced `data`, so a stamp taken after restore would land in a
        dict of old values and `elapsed_ms` would return a plausible but
        wrong number with no error. Pass `clock` only when it is the same
        clock (same epoch) that produced `data`.
        """
        trace = cls(clock if clock is not None else _ReadOnlyClock())
        trace._stamps = dict(data)
        return trace

    def record_queue_wait(self, stage: str, wait_ms: float) -> None:
        """Record time item spent waiting in queue before this stage."""
        event = f"{stage}_wait_ms"
        if event in self._stamps:
            # Multiple items can pass through the same stage; store as list
            existing = self._stamps[event]
            if isinstance(existing, list):
                existing.append(wait_ms)
            else:
                self._stamps[event] = [existing, wait_ms]
        else:
            self._stamps[event] = wait_ms


Language = Literal["vi", "en", "zh", "ko"]

VadEventKind = Literal["speech_start", "speech_end"]


@dataclass(frozen=True, eq=False)
class AudioChunk:
    """A slice of mono PCM audio.

    eq=False on purpose: pcm is a numpy array, and a generated __eq__ would
    compare elementwise and return an array, so any comparison of two chunks
    would raise "truth value of an array is ambiguous". Identity is the
    semantic we want.
    """

    pcm: np.ndarray
    sample_rate: int
    t_start: float

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(
                f"sample_rate must be positive, got {self.sample_rate!r}"
            )

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate


@dataclass(frozen=True)
class VadEvent:
    """A speech boundary detected in the audio stream."""

    kind: VadEventKind
    t_audio: float

    def __post_init__(self) -> None:
        if self.kind not in ("speech_start", "speech_end"):
            raise ValueError(
                f"unknown VAD event kind {self.kind!r}; "
                "expected 'speech_start' or 'speech_end'"
            )


@dataclass(frozen=True)
class AsrEvent:
    """A transcript hypothesis. Partial unless is_final."""

    text: str
    t_audio_end: float
    t_wall: float
    is_final: bool = False
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )


@dataclass(frozen=True)
class MtEvent:
    """A translation hypothesis. Partial unless is_final."""

    text: str
    t_wall: float
    is_final: bool = False


@dataclass(frozen=True, eq=False)
class TtsEvent:
    """A slice of synthesized audio. eq=False for the same reason as AudioChunk."""

    pcm: np.ndarray
    sample_rate: int
    t_wall: float
    is_final: bool = False
