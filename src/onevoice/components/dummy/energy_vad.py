"""Energy-threshold voice activity detection.

D20: this actually detects rather than passing through. D10 requires the
detected boundaries to be asserted against the fixture's declared ground truth,
and a passthrough reporting speech at t=0 would make that assertion vacuous.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.interfaces import BaseVAD
from onevoice.registry import register
from onevoice.types import AudioChunk, VadEvent

DEFAULT_THRESHOLD = 0.05


@register("vad", "energy")
class EnergyVAD(BaseVAD):
    """Opens speech when peak amplitude crosses a threshold, closes when it falls."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._open = False
        self._last_t = 0.0

    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        self._check_not_flushed()
        self._last_t = chunk.t_start
        loud = float(np.abs(chunk.pcm).max(initial=0.0)) > self.threshold
        if loud and not self._open:
            self._open = True
            return [VadEvent(kind="speech_start", t_audio=chunk.t_start)]
        if not loud and self._open:
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
        self._open = False
        self._last_t = 0.0
