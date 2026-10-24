"""A batch-shaped ASR double that describes the audio it consumed.

D18: the text is derived from the audio rather than read from a script. A
scripted ASR would need a cursor advancing per utterance that reset() must not
rewind - exactly the state-leak shape the Phase 2 contract suite catches - so it
would fail its own certification suite.

The consequence is that BLEU on a dummy run is near zero. That is correct:
Phase 3 proves the plumbing, not the quality.
"""

from __future__ import annotations

from collections.abc import Sequence

from onevoice.interfaces import BaseASR
from onevoice.registry import register
from onevoice.types import AsrEvent, AudioChunk


@register("asr", "echo")
class EchoASR(BaseASR):
    """Buffers pushed audio and emits one final event on flush (D6)."""

    def __init__(self) -> None:
        self._samples = 0
        self._rate = 0
        self._t_audio_end = 0.0

    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        self._check_not_flushed()
        self._samples += len(chunk.pcm)
        self._rate = chunk.sample_rate
        self._t_audio_end = chunk.t_start + chunk.duration_s
        return []

    def flush(self) -> Sequence[AsrEvent]:
        self._mark_flushed()
        if self._samples == 0:
            return []
        return [
            AsrEvent(
                text=f"audio of {self._samples} samples at {self._rate} Hz",
                t_audio_end=self._t_audio_end,
                t_wall=0.0,
                is_final=True,
                confidence=1.0,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._samples = 0
        self._rate = 0
        self._t_audio_end = 0.0
