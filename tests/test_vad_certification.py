"""F1: certification must reject deliberately broken detectors.

Before Phase 4.1 the VAD contract's only stimulus was silence. A correct VAD
emits nothing on silence, so every timestamp assertion skipped itself, and a
detector reporting `t_audio=0.0` for every boundary passed 14 of 16 tests.

Certification strength is not what a correct component passes — it is what a
broken one fails. Each class below is broken in a specific, plausible way, and
each must be rejected. If any of these ever passes, the contract has stopped
testing and the VAD guarantees D42's lookback relies on are void.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from onevoice.contracts import VADContract
from onevoice.interfaces import BaseVAD
from onevoice.types import AudioChunk, VadEvent

THRESHOLD = 0.05


class _Detector(BaseVAD):
    """Correct detection, correct state handling. Only `_stamp` is overridden.

    Isolating the defect to timestamping is the point: these must fail on the
    timestamp assertions specifically, not incidentally on some other contract.
    """

    def __init__(self) -> None:
        self._open = False
        self._last_end = 0.0
        self._n = 0

    def _stamp(self, chunk: AudioChunk) -> float:
        raise NotImplementedError

    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        self._check_not_flushed()
        self._last_end = chunk.t_start + chunk.duration_s
        loud = float(np.abs(chunk.pcm).max(initial=0.0)) > THRESHOLD
        if loud and not self._open:
            self._open = True
            self._n += 1
            return [VadEvent(kind="speech_start", t_audio=self._stamp(chunk))]
        if not loud and self._open:
            self._open = False
            self._n += 1
            return [VadEvent(kind="speech_end", t_audio=self._stamp(chunk))]
        return []

    def flush(self) -> Sequence[VadEvent]:
        self._mark_flushed()
        if not self._open:
            return []
        self._open = False
        return [VadEvent(kind="speech_end", t_audio=self._last_end)]

    def reset(self) -> None:
        super().reset()
        self._open = False
        self._last_end = 0.0
        self._n = 0


class ZeroTimestampVAD(_Detector):
    """Reports every boundary at 0.0 — the original F1 counterexample."""

    def _stamp(self, chunk: AudioChunk) -> float:
        return 0.0


class OffsetTimestampVAD(_Detector):
    """Correct shape, constant 2.5 s error. Plausible frame-indexing bug."""

    def _stamp(self, chunk: AudioChunk) -> float:
        return chunk.t_start + 2.5


class DriftingTimestampVAD(_Detector):
    """Error grows with stream position — a sample-rate or accumulator bug."""

    def _stamp(self, chunk: AudioChunk) -> float:
        return chunk.t_start * 1.15


class MonotonicButFabricatedVAD(_Detector):
    """Timestamps increase and never go backwards, but are invented.

    The nastiest of the four: it satisfies ordering and looks plausible in a
    log, while bearing no relation to the audio it was given.
    """

    def _stamp(self, chunk: AudioChunk) -> float:
        return self._n * 0.5


class IgnoresChunkTStartVAD(_Detector):
    """Times from its own sample counter, ignoring `AudioChunk.t_start`.

    Not hypothetical: `SileroVAD` shipped with exactly this defect, masked
    because the driver feeds every utterance from t=0. Found by this suite.
    """

    def __init__(self) -> None:
        super().__init__()
        self._elapsed = 0.0

    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        events = super().push(chunk)
        self._elapsed += chunk.duration_s
        return events

    def _stamp(self, chunk: AudioChunk) -> float:
        return self._elapsed

    def reset(self) -> None:
        super().reset()
        self._elapsed = 0.0


BROKEN = [
    ZeroTimestampVAD,
    OffsetTimestampVAD,
    DriftingTimestampVAD,
    MonotonicButFabricatedVAD,
    IgnoresChunkTStartVAD,
]


def _run_contract(factory) -> list[str]:
    """Run the VAD contract against `factory`; return names of failed tests."""

    class _Probe(VADContract):
        def make_component(self):
            return factory()

    failed = []
    probe = _Probe()
    for name in dir(probe):
        if not name.startswith("test_"):
            continue
        try:
            getattr(probe, name)()
        except AssertionError:
            failed.append(name)
        except pytest.skip.Exception:
            # A skipped test is neither a pass nor a rejection.
            continue
        except Exception:  # noqa: BLE001 - a raise is also a rejection
            failed.append(name)
    return failed


@pytest.mark.parametrize("factory", BROKEN, ids=lambda f: f.__name__)
def test_broken_detectors_are_rejected(factory):
    failed = _run_contract(factory)
    assert failed, (
        f"{factory.__name__} passed the entire VAD contract. Certification "
        "cannot be relied upon: this component reports timestamps unrelated "
        f"to the audio it was given.\n{factory.__doc__}"
    )


@pytest.mark.parametrize("factory", BROKEN, ids=lambda f: f.__name__)
def test_rejection_is_for_the_right_reason(factory):
    """A broken clock must fail a timestamp test, not something incidental.

    Failing for an unrelated reason would mean the timestamp assertions are
    still not running, and the suite would pass this file while remaining as
    permissive as before.
    """
    failed = _run_contract(factory)
    timestamp_tests = {
        "test_timestamps_fall_within_the_pushed_audio",
        "test_timestamps_shift_with_the_audio_timeline",
        "test_timestamps_are_monotonic_within_a_stream",
    }
    assert timestamp_tests & set(failed), (
        f"{factory.__name__} was rejected, but not by a timestamp assertion "
        f"(failed: {failed}). The timestamp checks may not be executing."
    )


def test_a_correct_detector_still_passes():
    """The counterweight: hardening must not reject a conforming component."""
    from onevoice.components.dummy import EnergyVAD

    assert _run_contract(EnergyVAD) == []
