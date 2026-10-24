"""Streaming speech-synthesis contract."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from onevoice.interfaces.base import StreamingComponent
from onevoice.types import TtsEvent


class BaseTTS(StreamingComponent):
    """Turns target-language text into audio."""

    @abstractmethod
    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        """Consume a text segment. Returns audio emitted, possibly empty."""

    @abstractmethod
    def flush(self) -> Sequence[TtsEvent]:
        """End the stream and return any remaining audio."""
