"""Streaming voice-activity-detection contract."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from onevoice.interfaces.base import StreamingComponent
from onevoice.types import AudioChunk, VadEvent


class BaseVAD(StreamingComponent):
    """Segments an audio stream into speech regions."""

    @abstractmethod
    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        """Consume one audio chunk. Returns boundaries detected, possibly empty."""

    @abstractmethod
    def flush(self) -> Sequence[VadEvent]:
        """End the stream, emitting a closing speech_end if one is open."""
