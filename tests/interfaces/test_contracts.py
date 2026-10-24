"""The contract suite, exercised against minimal reference implementations.

These stand-ins exist only to prove the suite itself works. Phase 3's dummy
components and Phase 4's real models subclass the same contracts.
"""

from collections.abc import Sequence

import numpy as np

from onevoice.contracts import ASRContract, MTContract, TTSContract, VADContract
from onevoice.interfaces import BaseASR, BaseMT, BaseTTS, BaseVAD
from onevoice.types import AsrEvent, AudioChunk, MtEvent, TtsEvent, VadEvent


class _RefVAD(BaseVAD):
    """A minimal conforming VAD, used to prove the contract accepts one.

    `flush()` originally reported `speech_end` at a hardcoded `t_audio=0.0`.
    That is non-monotonic against the `speech_start` values it had already
    emitted and lies outside the audio it was given — it survived only because
    the contract could not check timestamps at all before D44. The reference
    double must be the thing the contract is meant to accept, so it now tracks
    the timeline like a real detector.
    """

    def __init__(self) -> None:
        self._pending_end = False
        self._last_end = 0.0

    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        self._check_not_flushed()
        self._pending_end = True
        self._last_end = chunk.t_start + chunk.duration_s
        return [VadEvent(kind="speech_start", t_audio=chunk.t_start)]

    def flush(self) -> Sequence[VadEvent]:
        self._mark_flushed()
        if not self._pending_end:
            return []
        self._pending_end = False
        return [VadEvent(kind="speech_end", t_audio=self._last_end)]

    def reset(self) -> None:
        super().reset()
        self._pending_end = False
        self._last_end = 0.0


class _RefASR(BaseASR):
    def __init__(self) -> None:
        self._emitted = False

    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        self._check_not_flushed()
        return []

    def flush(self) -> Sequence[AsrEvent]:
        self._mark_flushed()
        if self._emitted:
            return []
        self._emitted = True
        return [AsrEvent(text="x", is_final=True, t_audio_end=1.0, t_wall=1.0)]

    def reset(self) -> None:
        super().reset()
        self._emitted = False


class _RefMT(BaseMT):
    def push(self, text: str, is_final: bool = False) -> Sequence[MtEvent]:
        self._check_not_flushed()
        return [MtEvent(text=text, is_final=is_final, t_wall=0.0)]

    def flush(self) -> Sequence[MtEvent]:
        self._mark_flushed()
        return []


class _RefTTS(BaseTTS):
    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        return [
            TtsEvent(
                pcm=np.zeros(160, dtype=np.float32),
                sample_rate=16000,
                is_final=is_final,
                t_wall=0.0,
            )
        ]

    def flush(self) -> Sequence[TtsEvent]:
        self._mark_flushed()
        return []


class TestRefVAD(VADContract):
    def make_component(self):
        return _RefVAD()


class TestRefASR(ASRContract):
    def make_component(self):
        return _RefASR()


class TestRefMT(MTContract):
    def make_component(self):
        return _RefMT()


class TestRefTTS(TTSContract):
    def make_component(self):
        return _RefTTS()
