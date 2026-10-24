"""A synthesis double that emits silence proportional to the text length."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.async_pipeline import FLUSH_MARKER, POISON_PILL
from onevoice.interfaces import BaseTTS
from onevoice.registry import register
from onevoice.types import TtsEvent

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MS_PER_CHAR = 60


@register("tts", "silence")
class SilenceTTS(BaseTTS):
    """Emits silent audio whose duration scales with the text."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        ms_per_char: int = DEFAULT_MS_PER_CHAR,
    ) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        # Handle sentinel values first
        if text is None or text is FLUSH_MARKER or text is POISON_PILL:
            return []
        # Handle MtEvent payload from MT flush
        if hasattr(text, 'text'):
            text = text.text
        # Handle sentinel values that might have been passed through
        if text is None or text is FLUSH_MARKER or text is POISON_PILL:
            return []
        if not text:
            return []
        samples = int(len(text) * self.ms_per_char / 1000 * self.sample_rate)
        if samples <= 0:
            return []
        # Ensure samples is a Python int and create 1D array
        samples = int(samples)
        pcm = np.zeros(samples, dtype=np.float32)
        # Ensure 1D array - use atleast_1d to guarantee at least 1D
        pcm = np.atleast_1d(pcm)
        # Ensure it's exactly 1D
        if pcm.ndim != 1:
            pcm = np.atleast_1d(pcm).flatten()
        return [
            TtsEvent(
                pcm=pcm,
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=is_final,
            )
        ]

    def flush(self) -> Sequence[TtsEvent]:
        self._mark_flushed()
        return []
