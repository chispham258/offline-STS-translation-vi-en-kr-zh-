"""The four component base classes share one streaming shape (D6)."""

import numpy as np
import pytest

from onevoice.interfaces import BaseASR, BaseMT, BaseTTS, BaseVAD
from onevoice.types import AsrEvent, AudioChunk

ALL_BASES = [BaseVAD, BaseASR, BaseMT, BaseTTS]


@pytest.mark.parametrize("base", ALL_BASES)
def test_base_cannot_be_instantiated(base):
    with pytest.raises(TypeError):
        base()


@pytest.mark.parametrize("base", ALL_BASES)
def test_base_declares_push_and_flush_abstract(base):
    assert "push" in base.__abstractmethods__
    assert "flush" in base.__abstractmethods__


@pytest.mark.parametrize("base", ALL_BASES)
def test_base_provides_reset_concretely(base):
    # reset() has a working default: subclasses extend it, they do not have to
    # reimplement flush-state bookkeeping.
    assert "reset" not in base.__abstractmethods__


class _MinimalASR(BaseASR):
    """The smallest thing that satisfies BaseASR."""

    def push(self, chunk: AudioChunk):
        self._check_not_flushed()
        return []

    def flush(self):
        self._mark_flushed()
        return [AsrEvent(text="done", is_final=True, t_audio_end=1.0, t_wall=1.0)]


def _chunk() -> AudioChunk:
    return AudioChunk(
        pcm=np.zeros(512, dtype=np.float32), sample_rate=16000, t_start=0.0
    )


def test_a_minimal_subclass_is_instantiable():
    assert _MinimalASR() is not None


def test_push_after_flush_raises_until_reset():
    # D16: flush() means end of stream. Accepting more audio afterwards would
    # produce a transcript spanning two utterances that looks well-formed.
    asr = _MinimalASR()
    asr.push(_chunk())
    asr.flush()
    with pytest.raises(RuntimeError, match="flush"):
        asr.push(_chunk())
    asr.reset()
    assert asr.push(_chunk()) == []


def test_is_flushed_tracks_state():
    asr = _MinimalASR()
    assert asr.is_flushed is False
    asr.flush()
    assert asr.is_flushed is True
    asr.reset()
    assert asr.is_flushed is False
