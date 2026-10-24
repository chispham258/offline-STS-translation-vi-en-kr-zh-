"""Async driver contract: MT and TTS stages receive text, not event objects."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from onevoice.async_pipeline import (
    FLUSH_MARKER,
    POISON_PILL,
    AsyncComponent,
    QueueItem,
    run_utterance_async,
)
from onevoice.audio import write_wav
from onevoice.eval.manifest import RefRecord
from onevoice.interfaces import BaseASR, BaseMT, BaseTTS
from onevoice.types import (
    SPEECH_END,
    AsrEvent,
    AudioChunk,
    FakeClock,
    MtEvent,
    SystemClock,
    TimingTrace,
    TtsEvent,
    VadEvent,
)


class RecordingTTS(BaseTTS):
    def __init__(self) -> None:
        self.pushed: list[tuple[type, object]] = []

    def push(self, text: str, is_final: bool = False) -> list[TtsEvent]:
        self.pushed.append((type(text), text))
        return []

    def flush(self) -> list[TtsEvent]:
        self._mark_flushed()
        return []


class RecordingMT(BaseMT):
    def __init__(self) -> None:
        self.pushed: list[tuple[type, object]] = []

    def push(self, text: str, is_final: bool = False) -> list[MtEvent]:
        self.pushed.append((type(text), text))
        return []

    def flush(self) -> list[MtEvent]:
        self._mark_flushed()
        return []


def test_tts_stage_unwraps_mt_event_to_text():
    async def scenario() -> None:
        tts = RecordingTTS()
        in_queue: asyncio.Queue = asyncio.Queue()
        out_queue: asyncio.Queue = asyncio.Queue()
        stage = AsyncComponent(
            tts,
            "tts",
            in_queue,
            out_queue,
            TimingTrace(SystemClock()),
            "mt→tts",
        )
        stage.start()

        event = MtEvent(text="Xin chào", t_wall=1.0, is_final=True)
        await in_queue.put(
            QueueItem(
                payload=event,
                pushed_at=time.perf_counter(),
                utterance_id="u1",
            )
        )
        await in_queue.put(FLUSH_MARKER)

        while True:
            item = await asyncio.wait_for(out_queue.get(), timeout=5)
            if item is FLUSH_MARKER:
                break

        await in_queue.put(POISON_PILL)
        await asyncio.wait_for(stage.task, timeout=5)

        assert tts.pushed == [(str, "Xin chào")]

    asyncio.run(scenario())


def test_mt_stage_unwraps_asr_event_to_text():
    async def scenario() -> None:
        mt = RecordingMT()
        in_queue: asyncio.Queue = asyncio.Queue()
        out_queue: asyncio.Queue = asyncio.Queue()
        stage = AsyncComponent(
            mt,
            "mt",
            in_queue,
            out_queue,
            TimingTrace(SystemClock()),
            "asr→mt",
        )
        stage.start()

        event = AsrEvent(text="Xin chào", t_audio_end=1.0, t_wall=1.0, is_final=True)
        await in_queue.put(
            QueueItem(
                payload=event,
                pushed_at=time.perf_counter(),
                utterance_id="u1",
            )
        )
        await in_queue.put(FLUSH_MARKER)

        while True:
            item = await asyncio.wait_for(out_queue.get(), timeout=5)
            if item is FLUSH_MARKER:
                break

        await in_queue.put(POISON_PILL)
        await asyncio.wait_for(stage.task, timeout=5)

        assert mt.pushed == [(str, "Xin chào")]

    asyncio.run(scenario())


class ScriptedVAD:
    """Emit two speech bursts, leaving the second one open at EOF."""

    def __init__(self) -> None:
        self.calls = 0
        self.flush_called = False

    def reset(self) -> None:
        self.calls = 0
        self.flush_called = False

    def push(self, chunk: AudioChunk) -> list[VadEvent]:
        events = []
        if self.calls == 1:
            events.append(VadEvent("speech_start", chunk.t_start + 0.001))
        if self.calls == 3:
            end = chunk.t_start + chunk.duration_s - 0.001
            events.append(VadEvent("speech_end", end))
        if self.calls == 6:
            events.append(VadEvent("speech_start", chunk.t_start + 0.001))
        self.calls += 1
        return events

    def flush(self) -> list[VadEvent]:
        self.flush_called = True
        return [VadEvent("speech_end", 0.49)]


class FinalASR(BaseASR):
    def __init__(self) -> None:
        self.chunks: list[AudioChunk] = []

    def push(self, chunk: AudioChunk) -> list[AsrEvent]:
        self.chunks.append(chunk)
        return []

    def flush(self) -> list[AsrEvent]:
        self._mark_flushed()
        return [AsrEvent("asr text", 0.49, 0.0, True)]

    def reset(self) -> None:
        super().reset()
        self.chunks.clear()


class FinalMT(BaseMT):
    def push(self, text: str, is_final: bool = False) -> list[MtEvent]:
        return []

    def flush(self) -> list[MtEvent]:
        self._mark_flushed()
        return [MtEvent("translated", 0.0, True)]


def test_async_driver_uses_last_speech_end_and_flushes_vad(tmp_path):
    pcm = np.zeros(8192, dtype=np.float32)
    write_wav(tmp_path / "audio.wav", pcm, 16000)
    ref = RefRecord(
        utt_id="u1",
        audio_path="audio.wav",
        src_lang="vi",
        tgt_lang="en",
        src_text="",
        ref_text="",
        speech_start_s=0.0,
        speech_end_s=0.512,
    )
    vad = ScriptedVAD()
    trace = TimingTrace(FakeClock())
    asr = AsyncComponent(
        FinalASR(),
        "asr",
        asyncio.Queue(),
        asyncio.Queue(),
        trace,
        "vad→asr",
    )
    mt = AsyncComponent(
        FinalMT(),
        "mt",
        asr.out_queue,
        asyncio.Queue(),
        trace,
        "asr→mt",
    )
    tts = AsyncComponent(
        RecordingTTS(),
        "tts",
        mt.out_queue,
        asyncio.Queue(),
        trace,
        "mt→tts",
    )

    record = asyncio.run(
        run_utterance_async(
            ref,
            vad,
            {"asr": asr, "mt": mt, "tts": tts},
            trace,
            tmp_path,
            tmp_path,
        )
    )

    assert record.status == "ok"
    assert record.asr_text == "asr text"
    assert record.hyp_text == "translated"
    assert vad.flush_called is True
    assert trace.get("speech_start_audio") == pytest.approx(0.033)
    assert trace.get(SPEECH_END) is not None
    assert trace.get("speech_end_audio") == pytest.approx(0.49)
    assert asr.component.chunks[0].t_start == pytest.approx(0.032)
    assert asr.component.chunks.count(asr.component.chunks[0]) == 1
    assert len({id(c) for c in asr.component.chunks}) == len(asr.component.chunks)
