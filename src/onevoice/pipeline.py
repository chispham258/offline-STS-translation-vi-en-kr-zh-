"""Sequential pipeline driver: audio in, RunRecords out.

D8: sequential now, async scheduler at Phase 6. The interfaces already permit
parallelism; this driver does not need it.

The dependency runs one way. This module imports onevoice.eval.manifest to build
RunRecords; onevoice.eval must never import this module (D5), and
tests/eval/test_no_model_imports.py enforces that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from onevoice.audio import chunk_audio, read_wav, write_wav
from onevoice.config import ExperimentConfig, build
from onevoice.eval.manifest import RefRecord, RunRecord, read_refs
from onevoice.lookback import LookbackBuffer
from onevoice.registry import KINDS
from onevoice.types import (
    ASR_FINAL,
    ASR_FIRST_PARTIAL,
    MT_DONE,
    SPEECH_END,
    TTS_DONE,
    TTS_FIRST_AUDIO,
    TTS_SYNTHESIS_S,
    Clock,
    SystemClock,
    TimingTrace,
)


def build_components(config: ExperimentConfig) -> dict[str, Any]:
    """Instantiate every component named by a config.

    `languages` is passed to every component; `build()` only forwards it to
    the ones whose `__init__` actually accepts `src_lang`/`tgt_lang` (named
    or via **kwargs), so a component that does not want them is unaffected.
    """
    return {
        kind: build(kind, getattr(config, kind), config.languages) for kind in KINDS
    }


def run_utterance(
    ref: RefRecord,
    components: dict[str, Any],
    clock: Clock,
    out_dir: str | Path,
    fixtures_root: str | Path,
) -> RunRecord:
    """Run one utterance end to end and return its record.

    A component raising marks this utterance errored and returns; one bad
    utterance must not abort a long run, but the failure is recorded because the
    evaluator excludes errored utterances and reports the count alongside every
    score.
    """
    for component in components.values():
        component.reset()

    trace = TimingTrace(clock)
    trace.stamp("stream_start")
    asr_text = ""
    hyp_text = ""
    duration = 0.0
    out_pcm: list[np.ndarray] = []
    out_rate: int | None = None
    tts_synthesis_s = 0.0

    def _accumulate(tts_event) -> None:
        nonlocal out_rate
        if out_rate is None:
            out_rate = tts_event.sample_rate
        elif out_rate != tts_event.sample_rate:
            raise ValueError(
                f"tts emitted mixed sample rates {out_rate} and "
                f"{tts_event.sample_rate} within one utterance; silently "
                "concatenating them would write a wav file wrong in a way "
                "nothing downstream can detect"
            )
        if trace.get(TTS_FIRST_AUDIO) is None:
            trace.stamp(TTS_FIRST_AUDIO)
        out_pcm.append(tts_event.pcm)

    def _note_partial(asr_events) -> None:
        """Stamp the first partial ASR text, if one ever arrives.

        The frame loop discards `push()`'s return value and still does - this
        only looks at it. A batch ASR (D6) returns nothing here, so no stamp is
        created rather than a zero being invented.
        """
        if trace.get(ASR_FIRST_PARTIAL) is not None:
            return
        for event in asr_events:
            if event.text:
                trace.stamp(ASR_FIRST_PARTIAL)
                return

    def _synthesize(mt_event) -> None:
        nonlocal hyp_text, tts_synthesis_s
        hyp_text = mt_event.text
        # Timed around the call itself, never around the iteration: D15 makes
        # push() return a concrete Sequence, so the synthesis has already
        # happened by the time it returns. A synchronous TTS (Piper) does its
        # work here; a batch one does it in flush(), which is timed the same
        # way. Neither the driver nor this accumulator needs to know which.
        started = clock.now()
        tts_events = tts.push(mt_event.text, is_final=mt_event.is_final)
        tts_synthesis_s += clock.now() - started
        for tts_event in tts_events:
            _accumulate(tts_event)

    try:
        audio_path = Path(fixtures_root) / ref.audio_path
        pcm, rate = read_wav(audio_path)
        duration = len(pcm) / rate

        vad, asr, mt, tts = (components[k] for k in ("vad", "asr", "mt", "tts"))
        speaking = False
        speech_start_audio: float | None = None
        speech_end_audio: float | None = None

        # Q7: gating on "is the detector open right now" drops every frame a
        # lookahead VAD needed as context, because it reports speech_start only
        # after that audio has gone past. Replay from the event's own t_audio
        # instead. For a zero-lookahead detector t_audio is the current frame's
        # t_start, so the replay is exactly [chunk] and nothing changes.
        lookback = LookbackBuffer()

        for chunk in chunk_audio(pcm, rate):
            lookback.append(chunk)
            replayed = False
            for event in vad.push(chunk):
                if event.kind == "speech_start":
                    if not speaking:
                        for buffered in lookback.since(event.t_audio):
                            _note_partial(asr.push(buffered))
                        replayed = True
                    speaking = True
                    if speech_start_audio is None:
                        speech_start_audio = event.t_audio
                else:
                    speaking = False
                    speech_end_audio = event.t_audio
            # `replayed` already delivered this frame; pushing it again would
            # double-count the audio at every onset.
            if speaking and not replayed:
                _note_partial(asr.push(chunk))

        for event in vad.flush():
            if event.kind == "speech_end":
                speech_end_audio = event.t_audio

        # Stamped once here, after the frame loop, rather than inline as each
        # VAD event arrives: a two-burst utterance with an internal pause
        # crosses speech_start/speech_end twice, and stamp() rightly raises
        # on a duplicate. speech_end is the end of the *last* segment, which
        # is what the latency metric wants.
        if speech_start_audio is not None:
            trace.stamp("speech_start")
            trace.record("speech_start_audio", speech_start_audio)
        if speech_end_audio is not None:
            trace.stamp(SPEECH_END)
            trace.record("speech_end_audio", speech_end_audio)

        for asr_event in asr.flush():
            # Stamped on receipt, before MT is fed: stamping after the loop
            # would put MT and TTS work inside the ASR window. Guarded because
            # flush() may emit several finals and stamp() raises on a repeat.
            if asr_event.is_final and trace.get(ASR_FINAL) is None:
                trace.stamp(ASR_FINAL)
            asr_text = asr_event.text
            for mt_event in mt.push(asr_event.text, is_final=asr_event.is_final):
                _synthesize(mt_event)

        for mt_event in mt.flush():
            _synthesize(mt_event)
        # MT has produced everything it is going to. asr_final -> mt_done is
        # MT alone only when TTS is batch-shaped (D6). Piper synthesizes inside
        # push(), so its work lands in this window and mt_done -> tts_done
        # reads ~0. The driver is deliberately not restructured to separate
        # them - buffering MT output and synthesizing afterwards would delay
        # tts_first_audio, an existing measurement. TTS_SYNTHESIS_S measures
        # the synthesis directly instead, so the attribution is recoverable:
        # MT alone = (mt_done - asr_final) - tts_synthesis_s spent so far.
        trace.stamp(MT_DONE)

        started = clock.now()
        tts_events = tts.flush()
        tts_synthesis_s += clock.now() - started
        for tts_event in tts_events:
            _accumulate(tts_event)
        trace.stamp(TTS_DONE)
        # Recorded after TTS_DONE so the stamp is not delayed by the write, and
        # only when synthesis actually took measurable time - a zero would be
        # indistinguishable from "this component was never called".
        if tts_synthesis_s > 0.0:
            trace.record(TTS_SYNTHESIS_S, tts_synthesis_s)

    except Exception as exc:  # noqa: BLE001 - containment is the point
        # Whatever synthesis genuinely completed before the failure is kept;
        # nothing is invented for the part that never ran.
        if tts_synthesis_s > 0.0 and trace.get(TTS_SYNTHESIS_S) is None:
            trace.record(TTS_SYNTHESIS_S, tts_synthesis_s)
        return RunRecord(
            utt_id=ref.utt_id,
            hyp_text="",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            asr_text=asr_text,
            audio_duration_s=duration,
            timing=trace.to_dict(),
        )

    hyp_audio_path = Path(out_dir) / "audio" / f"{ref.utt_id}.wav"
    joined = np.concatenate(out_pcm) if out_pcm else np.zeros(0, dtype=np.float32)
    write_wav(hyp_audio_path, joined, out_rate if out_rate is not None else rate)

    return RunRecord(
        utt_id=ref.utt_id,
        hyp_text=hyp_text,
        status="ok",
        asr_text=asr_text,
        hyp_audio_path=str(hyp_audio_path),
        audio_duration_s=duration,
        timing=trace.to_dict(),
    )


def run_pipeline(
    config: ExperimentConfig,
    refs_path: str | Path,
    out_dir: str | Path,
    clock: Clock | None = None,
    fixtures_root: str | Path | None = None,
) -> list[RunRecord]:
    """Run every utterance in a reference manifest."""
    the_clock = clock if clock is not None else SystemClock()
    root = fixtures_root if fixtures_root is not None else Path(refs_path).parent
    components = build_components(config)
    return [
        run_utterance(ref, components, the_clock, out_dir, root)
        for ref in read_refs(refs_path)
    ]
