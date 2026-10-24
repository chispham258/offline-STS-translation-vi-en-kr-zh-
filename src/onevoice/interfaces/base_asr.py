"""Streaming ASR contract."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from onevoice.interfaces.base import StreamingComponent
from onevoice.types import AsrEvent, AudioChunk


class BaseASR(StreamingComponent):
    """Turns audio into transcript hypotheses.

    A batch model such as Whisper satisfies this by buffering pushed chunks and
    emitting one final event on flush(). Nothing is lost by the streaming shape.
    """

    @abstractmethod
    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        """Consume one audio chunk. Returns events emitted, possibly empty.

        D15: returns a concrete Sequence, never a generator. A lazy return would
        defer the state mutation until the caller iterated.
        """

    @abstractmethod
    def flush(self) -> Sequence[AsrEvent]:
        """End the stream and return any remaining events."""
