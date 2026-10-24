"""Streaming machine-translation contract."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from onevoice.interfaces.base import StreamingComponent
from onevoice.types import MtEvent


class BaseMT(StreamingComponent):
    """Turns source-language text into target-language text.

    D12: a pivot (vi -> en -> zh) is implemented as a BaseMT composing two
    BaseMT instances, not as branching in the pipeline driver, so the driver
    never learns that a hop happened.
    """

    @abstractmethod
    def push(self, text: str, is_final: bool = False) -> Sequence[MtEvent]:
        """Consume a source segment. Returns translations emitted, possibly empty."""

    @abstractmethod
    def flush(self) -> Sequence[MtEvent]:
        """End the stream and return any remaining events."""
