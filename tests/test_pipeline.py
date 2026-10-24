"""The sequential driver: audio in, RunRecords out."""

import numpy as np
import pytest

from onevoice.audio import read_wav, write_wav
from onevoice.components.dummy import EchoASR, EnergyVAD, IdentityMT, SilenceTTS
from onevoice.config import load_config
from onevoice.eval.manifest import RefRecord, RunRecord
from onevoice.eval.metrics.timing import end_to_speech_ms, pipeline_rtf, rtf
from onevoice.interfaces import BaseASR, BaseMT, BaseTTS
from onevoice.pipeline import run_pipeline, run_utterance
from onevoice.types import (
    ASR_FINAL,
    ASR_FIRST_PARTIAL,
    MT_DONE,
    SPEECH_END,
    TTS_DONE,
    TTS_FIRST_AUDIO,
    TTS_SYNTHESIS_S,
    AsrEvent,
    FakeClock,
    MtEvent,
    TtsEvent,
)

REFS = "fixtures/mini_vi_en/refs.jsonl"
CONFIG = "configs/dummy_vi_en.yaml"


def _components():
    return {
        "vad": EnergyVAD(),
        "asr": EchoASR(),
        "mt": IdentityMT(),
        "tts": SilenceTTS(),
    }


def _ref() -> RefRecord:
    return RefRecord(
        utt_id="utt-001",
        audio_path="audio/utt-001.wav",
        src_lang="vi",
        tgt_lang="en",
        src_text="Xin chào",
        ref_text="Hello",
        speech_start_s=0.2,
        speech_end_s=1.8,
    )


def test_run_pipeline_produces_one_record_per_reference(tmp_path):
    records = run_pipeline(load_config(CONFIG), REFS, tmp_path)
    assert len(records) == 3
    assert [r.utt_id for r in records] == ["utt-001", "utt-002", "utt-003"]


def test_every_record_succeeds_on_the_dummy_pipeline(tmp_path):
    records = run_pipeline(load_config(CONFIG), REFS, tmp_path)
    assert all(r.status == "ok" for r in records), [r.error for r in records]


def test_records_carry_transcript_translation_and_audio(tmp_path):
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path)[0]
    assert record.asr_text
    assert record.hyp_text
    assert (tmp_path / "audio" / "utt-001.wav").is_file()


def test_timing_events_are_all_present(tmp_path):
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path)[0]
    events = ("stream_start", "speech_start", SPEECH_END, TTS_FIRST_AUDIO, TTS_DONE)
    for event in events:
        assert event in record.timing, f"missing {event}"


def test_timing_events_are_monotonically_ordered(tmp_path):
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path)[0]
    order = ["stream_start", "speech_start", SPEECH_END, TTS_FIRST_AUDIO, TTS_DONE]
    stamps = [record.timing[e] for e in order]
    assert stamps == sorted(stamps), dict(zip(order, stamps, strict=True))


@pytest.mark.parametrize(
    ("utt_id", "declared_start", "declared_end"),
    [("utt-001", 0.2, 1.8), ("utt-002", 0.3, 2.4), ("utt-003", 0.1, 2.9)],
)
def test_detected_boundaries_match_the_declared_ones(
    tmp_path, utt_id, declared_start, declared_end
):
    # D10. A fixture with leading silence would otherwise corrupt speech_end and
    # make every latency number fiction while the tests stayed green.
    records = {r.utt_id: r for r in run_pipeline(load_config(CONFIG), REFS, tmp_path)}
    record = records[utt_id]
    tolerance = 0.05  # one and a half 32 ms frames
    assert record.timing["speech_start_audio"] == pytest.approx(
        declared_start, abs=tolerance
    )
    assert record.timing["speech_end_audio"] == pytest.approx(
        declared_end, abs=tolerance
    )


def test_latency_is_measured_against_the_injected_clock(tmp_path):
    # A FakeClock that is never advanced must produce exactly zero latency. If
    # the driver reached for perf_counter instead of the injected clock, this
    # would be some small nonzero number - which "assert latency >= 0" would
    # happily accept, proving nothing.
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path, clock=FakeClock())[0]
    latency_ms = (record.timing[TTS_FIRST_AUDIO] - record.timing[SPEECH_END]) * 1000
    assert latency_ms == pytest.approx(0.0, abs=1e-9)


def test_a_failing_component_marks_the_utterance_and_continues(tmp_path):
    class ExplodingMT(IdentityMT):
        def push(self, text, is_final=False):
            raise RuntimeError("simulated MT failure")

    components = _components()
    components["mt"] = ExplodingMT()
    record = run_utterance(
        _ref(), components, FakeClock(), tmp_path, "fixtures/mini_vi_en"
    )
    assert record.status == "error"
    assert "simulated MT failure" in record.error
    assert record.hyp_text == ""


def test_audio_duration_is_recorded(tmp_path):
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path)[0]
    assert record.audio_duration_s > 0


class BufferingMT(BaseMT):
    """Batch-shaped MT double: buffers on push, emits only on flush (D6).

    This is Opus-MT's shape. IdentityMT emits on push, so it never exercises
    the flush path the driver must also feed into TTS.
    """

    def __init__(self) -> None:
        self._buffer: list[tuple[str, bool]] = []

    def push(self, text, is_final=False):
        self._check_not_flushed()
        if text:
            self._buffer.append((text, is_final))
        return []

    def flush(self):
        self._mark_flushed()
        return [MtEvent(text=t, t_wall=0.0, is_final=f) for t, f in self._buffer]

    def reset(self) -> None:
        super().reset()
        self._buffer = []


def test_mt_flush_is_fed_to_tts(tmp_path):
    # FIX 1: a batch MT that only emits on flush() must still reach TTS -
    # otherwise hyp_text stays "" and a real Opus-MT would score BLEU 0 on a
    # run the harness reports as fully successful.
    components = _components()
    components["mt"] = BufferingMT()
    record = run_utterance(
        _ref(), components, FakeClock(), tmp_path, "fixtures/mini_vi_en"
    )
    assert record.status == "ok", record.error
    assert record.hyp_text != ""
    pcm, _ = read_wav(tmp_path / "audio" / "utt-001.wav")
    assert len(pcm) > 0


class BufferingTTS(BaseTTS):
    """Batch-shaped TTS double: buffers on push, emits only on flush (D6).

    This is MeloTTS's shape. SilenceTTS emits on push, so it never exercises
    the TTS_FIRST_AUDIO stamp on the flush path.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._buffer: list[str] = []

    def push(self, text, is_final=False):
        self._check_not_flushed()
        if text:
            self._buffer.append(text)
        return []

    def flush(self):
        self._mark_flushed()
        if not self._buffer:
            return []
        samples = max(sum(len(t) for t in self._buffer), 1) * 10
        return [
            TtsEvent(
                pcm=np.zeros(samples, dtype=np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=True,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._buffer = []


def test_tts_flush_audio_stamps_first_audio(tmp_path):
    # FIX 2: audio arriving only on tts.flush() must still stamp
    # TTS_FIRST_AUDIO, or end_to_speech_ms is computed over zero samples for
    # every utterance once TTS is batch-shaped (MeloTTS, at Phase 4).
    components = _components()
    components["tts"] = BufferingTTS()
    record = run_utterance(
        _ref(), components, FakeClock(), tmp_path, "fixtures/mini_vi_en"
    )
    assert record.status == "ok", record.error
    assert TTS_FIRST_AUDIO in record.timing
    latency_ms = (record.timing[TTS_FIRST_AUDIO] - record.timing[SPEECH_END]) * 1000
    assert latency_ms >= 0.0


class FixedRateTTS(BaseTTS):
    """A TTS double emitting audio at a fixed, non-input sample rate."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def push(self, text, is_final=False):
        self._check_not_flushed()
        if not text:
            return []
        return [
            TtsEvent(
                pcm=np.zeros(100, dtype=np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=is_final,
            )
        ]

    def flush(self):
        self._mark_flushed()
        return []


def test_output_wav_uses_the_tts_sample_rate(tmp_path):
    # FIX 3: a 24 kHz TTS output written at the input's 16 kHz is 2.04s of
    # audio labelled 3.06s, pitched 1.5x low.
    components = _components()
    components["tts"] = FixedRateTTS(24000)
    record = run_utterance(
        _ref(), components, FakeClock(), tmp_path, "fixtures/mini_vi_en"
    )
    assert record.status == "ok", record.error
    _, rate = read_wav(tmp_path / "audio" / "utt-001.wav")
    assert rate == 24000


class MixedRateTTS(BaseTTS):
    """A TTS double whose single push emits two different sample rates."""

    def push(self, text, is_final=False):
        self._check_not_flushed()
        if not text:
            return []
        return [
            TtsEvent(
                pcm=np.zeros(100, dtype=np.float32),
                sample_rate=16000,
                t_wall=0.0,
            ),
            TtsEvent(
                pcm=np.zeros(100, dtype=np.float32),
                sample_rate=24000,
                t_wall=0.0,
                is_final=is_final,
            ),
        ]

    def flush(self):
        self._mark_flushed()
        return []


def test_mismatched_tts_sample_rates_error_the_utterance(tmp_path):
    # FIX 3: silently concatenating two rates into one wav yields a file
    # that's wrong in a way nothing downstream can detect - raise instead.
    components = _components()
    components["tts"] = MixedRateTTS()
    record = run_utterance(
        _ref(), components, FakeClock(), tmp_path, "fixtures/mini_vi_en"
    )
    assert record.status == "error"
    assert "sample rate" in record.error


def test_a_second_speech_segment_does_not_error_the_utterance(tmp_path):
    # FIX 4: stamp() raises on a duplicate by design. Stamping directly from
    # the frame loop on a two-burst utterance hits "speech_start" (or
    # SPEECH_END) twice and errors an utterance whose only flaw is a pause.
    rate = 16000

    def _tone(seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * rate), dtype=np.float32) / rate
        return (0.5 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    def _silence(seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * rate), dtype=np.float32)

    pcm = np.concatenate(
        [_silence(0.1), _tone(0.3), _silence(0.3), _tone(0.3), _silence(0.1)]
    )
    write_wav(tmp_path / "utt-multi.wav", pcm, rate)
    ref = RefRecord(
        utt_id="utt-multi",
        audio_path="utt-multi.wav",
        src_lang="vi",
        tgt_lang="en",
        src_text="Xin chào",
        ref_text="Hello",
        speech_start_s=0.1,
        speech_end_s=1.0,
    )
    record = run_utterance(ref, _components(), FakeClock(), tmp_path, tmp_path)
    assert record.status == "ok", record.error
    # speech_end is the end of the *last* segment (~1.0s), not the first
    # (~0.4s) - that is what the latency metric wants.
    assert record.timing["speech_end_audio"] == pytest.approx(1.0, abs=0.1)


def test_a_missing_audio_file_does_not_abort_the_run(tmp_path):
    # FIX 5: read_wav sat outside the containment boundary, so one bad
    # reference unwound the whole run's list comprehension and discarded
    # every record already computed - contradicting run_utterance's own
    # docstring.
    rows = [
        RefRecord(
            utt_id="ok-1",
            audio_path="audio/utt-001.wav",
            src_lang="vi",
            tgt_lang="en",
            src_text="Xin chào",
            ref_text="Hello",
            speech_start_s=0.2,
            speech_end_s=1.8,
        ),
        RefRecord(
            utt_id="missing-audio",
            audio_path="audio/does-not-exist.wav",
            src_lang="vi",
            tgt_lang="en",
            src_text="Xin chào",
            ref_text="Hello",
            speech_start_s=0.2,
            speech_end_s=1.8,
        ),
        RefRecord(
            utt_id="ok-2",
            audio_path="audio/utt-002.wav",
            src_lang="vi",
            tgt_lang="en",
            src_text="Xin chào",
            ref_text="Hello",
            speech_start_s=0.3,
            speech_end_s=2.4,
        ),
    ]
    refs_path = tmp_path / "refs.jsonl"
    refs_path.write_text(
        "\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8"
    )
    records = run_pipeline(
        load_config(CONFIG),
        refs_path,
        tmp_path,
        fixtures_root="fixtures/mini_vi_en",
    )
    by_id = {r.utt_id: r for r in records}
    assert len(records) == 3
    assert by_id["ok-1"].status == "ok"
    assert by_id["ok-2"].status == "ok"
    assert by_id["missing-audio"].status == "error"
    assert "does-not-exist.wav" in by_id["missing-audio"].error
    assert by_id["missing-audio"].audio_duration_s == 0.0


# --- Stage-attribution instrumentation (additive, D50 exception) -------------
#
# Three stamps were added so the ASR / MT / TTS split of end-to-speech latency
# is observable. They add measurements; they redefine none. Every assertion
# about the pre-existing events above still holds unchanged, which is the point.


class StreamingASR(BaseASR):
    """Emits a partial from push() and a final from flush().

    EchoASR is batch-shaped (D6), so it can never exercise the partial path.
    This double exists to prove the partial *is* stamped when one arrives, and
    EchoASR proves it is not invented when one never does.
    """

    def __init__(self) -> None:
        self._pushes = 0

    def push(self, chunk):
        self._check_not_flushed()
        self._pushes += 1
        if self._pushes == 1:
            return [
                AsrEvent(
                    text="partial",
                    t_audio_end=chunk.t_start + chunk.duration_s,
                    t_wall=0.0,
                    is_final=False,
                    confidence=0.5,
                )
            ]
        return []

    def flush(self):
        self._mark_flushed()
        return [
            AsrEvent(
                text="final transcript",
                t_audio_end=1.0,
                t_wall=0.0,
                is_final=True,
                confidence=1.0,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._pushes = 0


class TwoFinalASR(BaseASR):
    """Flushes two final events, to prove asr_final is stamped exactly once."""

    def push(self, chunk):
        self._check_not_flushed()
        return []

    def flush(self):
        self._mark_flushed()
        return [
            AsrEvent(
                text=f"segment {i}",
                t_audio_end=float(i),
                t_wall=0.0,
                is_final=True,
                confidence=1.0,
            )
            for i in (1, 2)
        ]


class ExplodingMT(BaseMT):
    """Raises on flush, after ASR has completed."""

    def push(self, text, is_final=False):
        return []

    def flush(self):
        raise RuntimeError("mt exploded")


def _run_one(components, tmp_path, clock=None):
    return run_utterance(
        _ref(),
        components,
        clock if clock is not None else FakeClock(),
        tmp_path,
        "fixtures/mini_vi_en",
    )


def test_a_streaming_asr_stamps_the_first_partial(tmp_path):
    components = _components() | {"asr": StreamingASR()}
    record = _run_one(components, tmp_path)
    assert record.status == "ok", record.error
    assert ASR_FIRST_PARTIAL in record.timing


def test_a_batch_asr_does_not_fabricate_a_first_partial(tmp_path):
    record = _run_one(_components(), tmp_path)
    assert record.status == "ok", record.error
    assert ASR_FINAL in record.timing
    assert ASR_FIRST_PARTIAL not in record.timing


def test_asr_final_is_stamped_exactly_once_across_several_final_events(tmp_path):
    components = _components() | {"asr": TwoFinalASR()}
    record = _run_one(components, tmp_path)
    assert record.status == "ok", record.error
    assert ASR_FINAL in record.timing


def test_mt_done_is_stamped_after_mt_and_before_tts_completes(tmp_path):
    record = _run_one(_components(), tmp_path)
    assert MT_DONE in record.timing
    assert record.timing[ASR_FINAL] <= record.timing[MT_DONE]
    assert record.timing[MT_DONE] <= record.timing[TTS_DONE]


def test_mt_done_is_stamped_when_translation_arrives_only_on_flush(tmp_path):
    components = _components() | {"mt": BufferingMT()}
    record = _run_one(components, tmp_path)
    assert record.status == "ok", record.error
    assert MT_DONE in record.timing
    assert record.timing[MT_DONE] <= record.timing[TTS_DONE]


def test_stage_timestamps_are_non_decreasing(tmp_path):
    clock = FakeClock()
    components = _components() | {"asr": StreamingASR()}
    record = _run_one(components, tmp_path, clock=clock)
    order = [
        "stream_start",
        ASR_FIRST_PARTIAL,
        "speech_start",
        SPEECH_END,
        ASR_FINAL,
        MT_DONE,
        TTS_FIRST_AUDIO,
        TTS_DONE,
    ]
    stamps = [record.timing[e] for e in order if e in record.timing]
    assert stamps == sorted(stamps), record.timing


def test_a_stage_failure_leaves_only_completed_stage_timestamps(tmp_path):
    components = _components() | {"mt": ExplodingMT()}
    record = _run_one(components, tmp_path)
    assert record.status == "error"
    assert "mt exploded" in record.error
    assert ASR_FINAL in record.timing
    assert MT_DONE not in record.timing
    assert TTS_DONE not in record.timing


def test_the_pre_existing_timing_events_are_unchanged(tmp_path):
    """The additive change must leave every earlier event exactly as it was."""
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path)[0]
    for event in (
        "stream_start",
        "speech_start",
        "speech_start_audio",
        SPEECH_END,
        "speech_end_audio",
        TTS_FIRST_AUDIO,
        TTS_DONE,
    ):
        assert event in record.timing, f"missing {event}"
    order = ["stream_start", "speech_start", SPEECH_END, TTS_FIRST_AUDIO, TTS_DONE]
    stamps = [record.timing[e] for e in order]
    assert stamps == sorted(stamps)


def test_a_legacy_timing_dict_without_the_new_keys_still_scores():
    """Archived runs predate these stamps and must remain measurable."""
    legacy = RunRecord(
        utt_id="legacy",
        hyp_text="hello",
        status="ok",
        audio_duration_s=2.0,
        timing={
            "stream_start": 0.0,
            "speech_start": 0.1,
            "speech_end": 1.0,
            "tts_first_audio": 1.5,
            "tts_done": 1.8,
        },
    )
    assert ASR_FINAL not in legacy.timing
    assert end_to_speech_ms(legacy) == pytest.approx(500.0)
    assert rtf(legacy) == pytest.approx(0.4)
    assert pipeline_rtf(legacy) == pytest.approx(0.9)


class ChattyASR(BaseASR):
    """Emits a partial on every push, to prove the stamp is taken only once."""

    def push(self, chunk):
        self._check_not_flushed()
        return [
            AsrEvent(
                text="partial",
                t_audio_end=chunk.t_start + chunk.duration_s,
                t_wall=0.0,
                is_final=False,
                confidence=0.5,
            )
        ]

    def flush(self):
        self._mark_flushed()
        return [
            AsrEvent(
                text="final",
                t_audio_end=1.0,
                t_wall=0.0,
                is_final=True,
                confidence=1.0,
            )
        ]


class ExplodingASR(BaseASR):
    """Raises on flush, so the failure lands before asr_final is stamped."""

    def push(self, chunk):
        self._check_not_flushed()
        return []

    def flush(self):
        raise RuntimeError("asr exploded")


def test_repeated_partials_stamp_the_first_only(tmp_path):
    """stamp() raises on a duplicate, so an unguarded second stamp would error."""
    components = _components() | {"asr": ChattyASR()}
    record = _run_one(components, tmp_path)
    assert record.status == "ok", record.error
    assert ASR_FIRST_PARTIAL in record.timing


def test_a_failure_before_asr_final_leaves_it_absent(tmp_path):
    components = _components() | {"asr": ExplodingASR()}
    record = _run_one(components, tmp_path)
    assert record.status == "error"
    assert "asr exploded" in record.error
    assert ASR_FINAL not in record.timing
    assert MT_DONE not in record.timing
    assert TTS_DONE not in record.timing
    # The stages that did complete keep their stamps.
    assert "stream_start" in record.timing
    assert SPEECH_END in record.timing


def test_the_new_keys_do_not_change_any_existing_timing_metric(tmp_path):
    """Same trace, scored with and without the additive keys: identical."""
    record = run_pipeline(load_config(CONFIG), REFS, tmp_path)[0]
    assert ASR_FINAL in record.timing and MT_DONE in record.timing

    stripped = RunRecord(
        utt_id=record.utt_id,
        hyp_text=record.hyp_text,
        status=record.status,
        asr_text=record.asr_text,
        audio_duration_s=record.audio_duration_s,
        timing={
            k: v
            for k, v in record.timing.items()
            if k not in (ASR_FIRST_PARTIAL, ASR_FINAL, MT_DONE)
        },
    )
    assert end_to_speech_ms(record) == end_to_speech_ms(stripped)
    assert rtf(record) == rtf(stripped)
    assert pipeline_rtf(record) == pipeline_rtf(stripped)


# --- TTS synthesis accumulator (additive diagnostic) ------------------------
#
# PiperTTS synthesizes inside push(); BufferingTTS synthesizes inside flush().
# Wrapping both call sites measures either shape without the driver needing to
# know which it has. Observational only: no metric, delta or selection rule
# reads it.


class SlowSyncTTS(BaseTTS):
    """Synchronous shape - synthesizes in push(). Piper's shape."""

    def __init__(self, clock, cost: float = 0.5, sample_rate: int = 16000) -> None:
        self._clock = clock
        self._cost = cost
        self.sample_rate = sample_rate

    def push(self, text, is_final=False):
        self._check_not_flushed()
        if not text:
            return []
        self._clock.advance(self._cost)
        return [
            TtsEvent(
                pcm=np.zeros(100, dtype=np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=is_final,
            )
        ]

    def flush(self):
        self._mark_flushed()
        return []


class SlowBatchTTS(BaseTTS):
    """Batch shape - synthesizes in flush(). MeloTTS's shape."""

    def __init__(self, clock, cost: float = 0.7, sample_rate: int = 16000) -> None:
        self._clock = clock
        self._cost = cost
        self.sample_rate = sample_rate
        self._buffer: list[str] = []

    def push(self, text, is_final=False):
        self._check_not_flushed()
        if text:
            self._buffer.append(text)
        return []

    def flush(self):
        self._mark_flushed()
        if not self._buffer:
            return []
        self._clock.advance(self._cost)
        return [
            TtsEvent(
                pcm=np.zeros(100, dtype=np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=True,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._buffer = []


class SilentNoWorkTTS(BaseTTS):
    """Emits nothing and does no work, so no synthesis time should be recorded."""

    def push(self, text, is_final=False):
        self._check_not_flushed()
        return []

    def flush(self):
        self._mark_flushed()
        return []


class ExplodingTTS(BaseTTS):
    """Raises before doing any synthesis."""

    def push(self, text, is_final=False):
        raise RuntimeError("tts exploded")

    def flush(self):
        return []


def test_synchronous_tts_synthesis_time_is_accumulated(tmp_path):
    clock = FakeClock()
    components = _components() | {"tts": SlowSyncTTS(clock, cost=0.5)}
    record = _run_one(components, tmp_path, clock=clock)
    assert record.status == "ok", record.error
    assert record.timing[TTS_SYNTHESIS_S] == pytest.approx(0.5)


def test_batch_tts_synthesis_time_is_accumulated_from_flush(tmp_path):
    clock = FakeClock()
    components = _components() | {"tts": SlowBatchTTS(clock, cost=0.7)}
    record = _run_one(components, tmp_path, clock=clock)
    assert record.status == "ok", record.error
    assert record.timing[TTS_SYNTHESIS_S] == pytest.approx(0.7)
    # The existing events on the flush path are untouched.
    assert TTS_FIRST_AUDIO in record.timing
    assert TTS_DONE in record.timing
    assert record.timing[TTS_FIRST_AUDIO] <= record.timing[TTS_DONE]


def test_no_synthesis_records_no_value(tmp_path):
    clock = FakeClock()
    components = _components() | {"tts": SilentNoWorkTTS()}
    record = _run_one(components, tmp_path, clock=clock)
    assert record.status == "ok", record.error
    assert TTS_SYNTHESIS_S not in record.timing


def test_a_tts_failure_does_not_fabricate_synthesis_time(tmp_path):
    components = _components() | {"tts": ExplodingTTS()}
    record = _run_one(components, tmp_path)
    assert record.status == "error"
    assert "tts exploded" in record.error
    assert TTS_SYNTHESIS_S not in record.timing


def test_the_synthesis_accumulator_changes_no_existing_metric(tmp_path):
    clock = FakeClock()
    components = _components() | {"tts": SlowSyncTTS(clock, cost=0.5)}
    record = _run_one(components, tmp_path, clock=clock)
    stripped = RunRecord(
        utt_id=record.utt_id,
        hyp_text=record.hyp_text,
        status=record.status,
        asr_text=record.asr_text,
        audio_duration_s=record.audio_duration_s,
        timing={k: v for k, v in record.timing.items() if k != TTS_SYNTHESIS_S},
    )
    assert end_to_speech_ms(record) == end_to_speech_ms(stripped)
    assert rtf(record) == rtf(stripped)
    assert pipeline_rtf(record) == pipeline_rtf(stripped)


def test_an_archived_run_without_the_synthesis_key_still_scores():
    legacy = RunRecord(
        utt_id="legacy",
        hyp_text="hello",
        status="ok",
        audio_duration_s=2.0,
        timing={
            "stream_start": 0.0,
            "speech_start": 0.1,
            "speech_end": 1.0,
            "tts_first_audio": 1.5,
            "tts_done": 1.8,
        },
    )
    assert TTS_SYNTHESIS_S not in legacy.timing
    assert end_to_speech_ms(legacy) == pytest.approx(500.0)
    assert rtf(legacy) == pytest.approx(0.4)
    assert pipeline_rtf(legacy) == pytest.approx(0.9)
