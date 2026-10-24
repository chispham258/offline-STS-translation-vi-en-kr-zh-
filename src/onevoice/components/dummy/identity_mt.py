"""A translation double that returns its input unchanged."""

from __future__ import annotations

from collections.abc import Sequence

from onevoice.interfaces import BaseMT
from onevoice.registry import register
from onevoice.types import MtEvent


@register("mt", "identity")
class IdentityMT(BaseMT):
    """Passes source text through as if it were the translation."""

    def push(self, text: str, is_final: bool = False) -> Sequence[MtEvent]:
        self._check_not_flushed()
        if not text:
            return []
        return [MtEvent(text=text, t_wall=0.0, is_final=is_final)]

    def flush(self) -> Sequence[MtEvent]:
        self._mark_flushed()
        return []
