"""Streaming bookkeeping shared by every component kind."""

from __future__ import annotations

from abc import ABC


class StreamingComponent(ABC):  # noqa: B024
    """Flush-state bookkeeping so each subclass does not reimplement D16.

    D16: flush() ends the stream. push() afterwards raises until reset().
    Subclasses call _check_not_flushed() at the top of push() and
    _mark_flushed() inside flush().
    """

    _flushed: bool = False

    @property
    def is_flushed(self) -> bool:
        return self._flushed

    def _check_not_flushed(self) -> None:
        if self._flushed:
            raise RuntimeError(
                f"{type(self).__name__} was flushed; call reset() before "
                "pushing again. Pushing into a flushed component would splice "
                "two utterances into one."
            )

    def _mark_flushed(self) -> None:
        self._flushed = True

    def reset(self) -> None:
        """Return to the initial state. Subclasses extend with super().reset()."""
        self._flushed = False
