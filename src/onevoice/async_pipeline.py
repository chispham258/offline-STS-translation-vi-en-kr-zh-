"""Async pipeline driver for overlapping VAD → ASR → MT → TTS stages.

Replaces the sequential run_utterance with an asyncio-based pipeline that
overlaps stages via bounded queues, targeting RTF < 0.6.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    AudioChunk,
    SystemClock,
    TimingTrace,
)

logger = logging.getLogger(__name__)


# Sentinels for flow control
FLUSH_MARKER = object()      # End of 1 utterance → call component.flush()
POISON_PILL = None           # Shutdown entire pipeline → break loop


# Diagnostic clock shared by all stages; reset at the start of each utterance.
_UTT_T0 = 0.0


def _rel() -> float:
    """Seconds since the current utterance started (diagnostic logs only)."""
    return time.perf_counter() - _UTT_T0


@dataclass
class QueueItem:
    """Wraps component I/O with timing metadata."""
    payload: Any                    # AudioChunk | str | MtEvent | TtsEvent
    pushed_at: float                # wall-clock when enqueued
    utterance_id: str               # for tracing


class AsyncComponent:
    """Wraps a sync component (ASR/MT/TTS) in an async task with I/O queues.

    VAD is handled separately - called directly by the main loop.
    """

    def __init__(
        self,
        component: Any,               # sync component instance
        name: str,                    # "asr" | "mt" | "tts"
        in_queue: asyncio.Queue,      # input queue
        out_queue: asyncio.Queue,     # output queue (None for TTS)
        trace: TimingTrace,           # shared timing trace
        queue_name: str,              # e.g. "asr→mt"
    ) -> None:
        self.component = component
        self.name = name
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.trace = trace
        self.queue_name = queue_name
        self.task: asyncio.Task | None = None
        self.emitted: list[Any] = []       # results this stage produced
        self._tts_synthesis_s = 0.0        # accumulated TTS work, seconds

    def reset_utterance(self) -> None:
        """Clear per-utterance state; components refuse pushes once flushed."""
        self.component.reset()
        self.emitted.clear()
        self._tts_synthesis_s = 0.0

    def _note_emitted(self, results: Sequence[Any]) -> None:
        for r in results:
            self.emitted.append(r)
        if (
            self.name == "tts"
            and results
            and self.trace.get(TTS_FIRST_AUDIO) is None
        ):
            self.trace.stamp(TTS_FIRST_AUDIO)

    def _stamp_stage_end(self) -> None:
        """Stamp the stage's completion once its flush has returned."""
        if self.name == "asr":
            if self.trace.get(ASR_FINAL) is None:
                self.trace.stamp(ASR_FINAL)
        elif self.name == "mt":
            if self.trace.get(MT_DONE) is None:
                self.trace.stamp(MT_DONE)
        elif self.name == "tts":
            if self.trace.get(TTS_DONE) is None:
                self.trace.stamp(TTS_DONE)
            if (
                self._tts_synthesis_s > 0.0
                and self.trace.get(TTS_SYNTHESIS_S) is None
            ):
                self.trace.record(TTS_SYNTHESIS_S, self._tts_synthesis_s)

    async def run(self) -> None:
        """Main loop: pull from in_queue, process, push to out_queue."""
        logger.debug(
            "stage %s started in=%s out=%s t=%.3f",
            self.name,
            id(self.in_queue),
            id(self.out_queue),
            _rel(),
        )
        try:
            while True:
                item = await self.in_queue.get()
                if item is POISON_PILL:              # Shutdown signal
                    logger.debug("stage %s recv POISON_PILL", self.name)
                    await self.out_queue.put(POISON_PILL)
                    break

                if item is FLUSH_MARKER:             # End of utterance
                    utt_id = getattr(self, "_last_utterance_id", "unknown")
                    logger.debug(
                        "stage %s recv FLUSH_MARKER utt=%s qsize=%s t=%.3f",
                        self.name,
                        utt_id,
                        self.in_queue.qsize(),
                        _rel(),
                    )
                    started = self.trace._clock.now()
                    results = await asyncio.to_thread(self.component.flush)
                    elapsed = self.trace._clock.now() - started
                    self.trace.record(f"{self.name}_flush_ms", elapsed * 1000)
                    if self.name == "tts":
                        self._tts_synthesis_s += elapsed
                    logger.debug(
                        "stage %s flush -> %s results utt=%s flush_ms=%.0f "
                        "t=%.3f",
                        self.name,
                        len(results),
                        utt_id,
                        elapsed * 1000,
                        _rel(),
                    )
                    self._note_emitted(results)
                    self._stamp_stage_end()
                    now = self.trace._clock.now()
                    # Use the last seen utterance_id for flush results
                    utt_id = getattr(self, "_last_utterance_id", "unknown")
                    for r in results:
                        await self.out_queue.put(QueueItem(
                            payload=r,
                            pushed_at=now,
                            utterance_id=utt_id,
                        ))
                    await self.out_queue.put(FLUSH_MARKER)
                    logger.debug("stage %s forwarded FLUSH_MARKER", self.name)
                    continue

                # Regular item: record queue wait time
                popped_at = self.trace._clock.now()
                self.trace.record(f"{self.queue_name}_wait_ms",
                                (popped_at - item.pushed_at) * 1000)

                # Store utterance_id for flush
                self._last_utterance_id = item.utterance_id

                logger.debug(
                    "stage %s recv %s utt=%s qsize=%s t=%.3f",
                    self.name,
                    type(item.payload).__name__,
                    item.utterance_id,
                    self.in_queue.qsize(),
                    _rel(),
                )
                # Process in thread pool to avoid blocking event loop (CPU-bound!)
                # MT and TTS implement push(str); ASR emits AsrEvent and MT
                # emits MtEvent. Unwrap here so every backend works on the
                # async path, not only the ones that happen to tolerate an
                # event object.
                payload = item.payload
                if self.name in ("mt", "tts") and not isinstance(payload, str):
                    payload = getattr(payload, "text", payload)
                started = self.trace._clock.now()
                results = await asyncio.to_thread(self.component.push, payload)
                elapsed = self.trace._clock.now() - started
                self.trace.record(f"{self.name}_process_ms", elapsed * 1000)
                if elapsed >= 0.5:
                    logger.debug(
                        "slow %s push %.0fms utt=%s t=%.3f",
                        self.name,
                        elapsed * 1000,
                        item.utterance_id,
                        _rel(),
                    )
                if self.name == "tts":
                    self._tts_synthesis_s += elapsed
                self._note_emitted(results)
                if (
                    self.name == "asr"
                    and results
                    and self.trace.get(ASR_FIRST_PARTIAL) is None
                ):
                    self.trace.stamp(ASR_FIRST_PARTIAL)

                if results:
                    logger.debug(
                        "stage %s push -> %s results", self.name, len(results)
                    )

                now = self.trace._clock.now()
                for r in results:
                    await self.out_queue.put(QueueItem(
                        payload=r,
                        pushed_at=now,
                        utterance_id=item.utterance_id,
                    ))
        except Exception:
            logger.exception("stage %s failed", self.name)
            raise

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name=f"stage-{self.name}")
        self.task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("task death %s: %r", self.name, exc, exc_info=exc)

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


def _collapse_timing(raw: dict[str, Any]) -> dict[str, float]:
    """Flatten repeated per-item timings into one number per key.

    TimingTrace.record() appends to a list when called more than once (one
    entry per chunk or segment), but RunRecord.timing is float-only. Waits are
    averaged; stage work is summed, because the total is what the stage cost.
    """
    timing: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            total = float(sum(value))
            if key.endswith(("_process_ms", "_flush_ms")):
                timing[key] = total
            else:
                timing[key] = total / len(value)
            timing[f"{key}_n"] = float(len(value))
        else:
            timing[key] = float(value)
    return timing


async def _feed_vad_to_asr(
    pcm: Any,
    rate: int,
    vad_component: Any,
    asr_queue: asyncio.Queue,
    trace: TimingTrace,
    utterance_id: str,
) -> tuple[float | None, float | None, dict[str, int]]:
    """Gate audio to ASR across every speech segment in one utterance.

    Mirrors the sequential driver: replay from the VAD event's own timestamp,
    push each frame at most once, and keep the last speech_end for the trace.
    """
    speaking = False
    speech_start_audio: float | None = None
    speech_end_audio: float | None = None
    lookback = LookbackBuffer()
    stats = {"fed": 0, "gated": 0, "replayed": 0, "events": 0}

    def item(chunk: AudioChunk) -> QueueItem:
        return QueueItem(
            payload=chunk,
            pushed_at=trace._clock.now(),
            utterance_id=utterance_id,
        )

    for chunk in chunk_audio(pcm, rate):
        stats["fed"] += 1
        lookback.append(chunk)
        replayed = False
        for event in vad_component.push(chunk):
            stats["events"] += 1
            if event.kind == "speech_start":
                if not speaking:
                    for buffered in lookback.since(event.t_audio):
                        await asr_queue.put(item(buffered))
                        stats["replayed"] += 1
                    replayed = True
                speaking = True
                if speech_start_audio is None:
                    speech_start_audio = event.t_audio
            elif event.kind == "speech_end":
                speaking = False
                speech_end_audio = event.t_audio
        if speaking and not replayed:
            await asr_queue.put(item(chunk))
            stats["gated"] += 1
        await asyncio.sleep(0)

    for event in vad_component.flush():
        stats["events"] += 1
        if event.kind == "speech_end":
            speech_end_audio = event.t_audio
    return speech_start_audio, speech_end_audio, stats


async def run_utterance_async(
    ref: RefRecord,
    vad_component: Any,
    components: dict[str, AsyncComponent],
    trace: TimingTrace,
    out_dir: Path,
    fixtures_root: Path,
) -> RunRecord:
    """Run one utterance through async pipeline.
    
    VAD is called directly (synchronously) to gate audio chunks to ASR queue.
    ASR/MT/TTS run as async tasks.
    """
    global _UTT_T0
    _UTT_T0 = time.perf_counter()

    # Components are flushed at the end of each utterance, and a flushed
    # component refuses further pushes. Reset before starting the next one.
    # Point every stage at this utterance's trace, otherwise stage stamps land
    # in the trace created by build_async_components and never reach the record.
    for c in components.values():
        c.trace = trace
        c.reset_utterance()

    # Start all async tasks (ASR, MT, TTS)
    for c in components.values():
        c.start()
    
    tts_queue = components["tts"].out_queue
    asr_queue = components["asr"].in_queue
    
    logger.debug(
        "utterance %s start q_asr=%s q_mt=%s q_tts=%s",
        ref.utt_id,
        asr_queue.qsize(),
        components["mt"].in_queue.qsize(),
        tts_queue.qsize(),
    )

    try:
        # 1. Process audio through VAD and gate chunks to ASR queue
        audio_path = Path(fixtures_root) / ref.audio_path
        pcm, rate = read_wav(audio_path)
        duration = len(pcm) / rate
        logger.debug("audio duration %.2fs", duration)

        # Reset VAD state
        vad_component.reset()
        
        speech_start_audio, speech_end_audio, stats = await _feed_vad_to_asr(
            pcm, rate, vad_component, asr_queue, trace, ref.utt_id
        )
        if speech_start_audio is not None and trace.get("speech_start") is None:
            trace.stamp("speech_start")
            trace.record("speech_start_audio", speech_start_audio)
        if speech_end_audio is not None and trace.get(SPEECH_END) is None:
            trace.stamp(SPEECH_END)
            trace.record("speech_end_audio", speech_end_audio)
        logger.debug(
            "utterance %s feed done fed=%s gated=%s replay=%s vad_events=%s "
            "q_asr=%s t=%.3f",
            ref.utt_id,
            stats["fed"],
            stats["gated"],
            stats["replayed"],
            stats["events"],
            asr_queue.qsize(),
            _rel(),
        )

        # 2. Signal end of input (flush ASR)
        await asr_queue.put(FLUSH_MARKER)
        logger.debug("sent FLUSH_MARKER to ASR queue")
        
        # Allow MT task to process the FLUSH_MARKER
        await asyncio.sleep(0.1)
        
        # 3. Collect TTS output from TTS queue
        tts_queue = components["tts"].out_queue
        audio_chunks = []
        sample_rate = 16000  # default for SilenceTTS
        while True:
            item = await tts_queue.get()
            logger.debug("got TTS item %s", type(item).__name__)
            if item is POISON_PILL:
                break
            if item is FLUSH_MARKER:
                break
            audio_chunks.append(item.payload.pcm)
            # Track sample rate from first TTS event
            if sample_rate == 16000 and hasattr(item.payload, "sample_rate"):
                sample_rate = item.payload.sample_rate

        logger.debug(
            "utterance %s TTS done chunks=%s q_tts=%s t=%.3f",
            ref.utt_id,
            len(audio_chunks),
            tts_queue.qsize(),
            _rel(),
        )
        
        # 4. Write audio, build RunRecord
        hyp_audio_path = out_dir / "audio" / f"{ref.utt_id}.wav"
        if audio_chunks:
            combined_pcm = __import__("numpy").concatenate(audio_chunks)
            write_wav(hyp_audio_path, combined_pcm, sample_rate)
        
        # Build RunRecord from trace
        asr_text = " ".join(
            getattr(e, "text", "") for e in components["asr"].emitted
        ).strip()
        mt_events = components["mt"].emitted
        hyp_text = (
            getattr(mt_events[-1], "text", "").strip() if mt_events else ""
        )
        return RunRecord(
            utt_id=ref.utt_id,
            hyp_text=hyp_text,
            status="ok",
            error="",
            asr_text=asr_text,
            hyp_audio_path=(
                str(hyp_audio_path.relative_to(out_dir)) if audio_chunks else ""
            ),
            audio_duration_s=duration,
            timing=_collapse_timing(trace.to_dict()),
        )
    
    except Exception:
        logger.exception("utterance %s failed", ref.utt_id)
        # Cancel all tasks, wait for cancellation to complete
        for c in components.values():
            if c.task:
                c.task.cancel()
        await asyncio.gather(
            *(c.task for c in components.values() if c.task),
            return_exceptions=True,
        )
        raise
    finally:
        # Ensure cleanup: stop all components
        logger.debug("utterance %s stop begin t=%.3f", ref.utt_id, _rel())
        for c in components.values():
            await c.stop()
        logger.debug(
            "utterance %s stop end t=%.3f q_asr=%s q_mt=%s q_tts=%s",
            ref.utt_id,
            _rel(),
            asr_queue.qsize(),
            components["mt"].in_queue.qsize(),
            tts_queue.qsize(),
        )


def build_async_components(
    config: ExperimentConfig,
) -> tuple[Any, dict[str, AsyncComponent], asyncio.Queue]:
    """Instantiate components and wrap ASR/MT/TTS in AsyncComponent with queues.

    VAD is returned directly (called synchronously by main loop).
    Returns: (vad_component, async_components_dict, vad_queue)
    """

    # Build sync components first
    sync_components = {
        kind: build(kind, getattr(config, kind), config.languages)
        for kind in KINDS
    }
    
    # Get queue sizes from config (with defaults)
    pipeline = getattr(config, "pipeline", None)
    queue_sizes = pipeline.queue_sizes if pipeline else {}
    vad_asr_size = queue_sizes.get("vad_asr", 16)
    asr_mt_size = queue_sizes.get("asr_mt", 4)
    mt_tts_size = queue_sizes.get("mt_tts", 4)
    
    # Create queues
    vad_queue = asyncio.Queue(maxsize=vad_asr_size)
    asr_queue = asyncio.Queue(maxsize=asr_mt_size)
    mt_queue = asyncio.Queue(maxsize=mt_tts_size)
    tts_queue = asyncio.Queue(maxsize=0)  # unbounded for output
    
    # Create trace
    trace = TimingTrace(SystemClock())
    
    # Wrap ASR, MT, TTS as async components. VAD is used directly by the
    # caller (no async task), so it is returned unwrapped below.
    components = {
        "asr": AsyncComponent(
            sync_components["asr"], "asr", vad_queue, asr_queue, trace, "vad→asr"
        ),
        "mt": AsyncComponent(
            sync_components["mt"], "mt", asr_queue, mt_queue, trace, "asr→mt"
        ),
        "tts": AsyncComponent(
            sync_components["tts"], "tts", mt_queue, tts_queue, trace, "mt→tts"
        ),
    }
    
    return sync_components["vad"], components, vad_queue


async def run_benchmark_async(
    config_path: str | Path,
    refs_path: str | Path,
    out_dir: str | Path,
    experiment: str = "",
    resume: bool = True,
) -> Any:
    """Run a config over a reference manifest asynchronously."""
    import json
    import shutil
    from datetime import UTC, datetime

    from onevoice.benchmark import _completed_utterance_ids, collect_provenance
    from onevoice.config import load_config
    
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run.jsonl"
    
    config = load_config(config_path)
    shutil.copyfile(config_path, out / "config.snapshot.yaml")
    
    refs = read_refs(refs_path)
    fixtures_root = Path(refs_path).parent
    
    if not resume and manifest_path.exists():
        manifest_path.unlink()
    already_done = _completed_utterance_ids(manifest_path)
    
    started_at = datetime.now(UTC)
    fresh = 0
    
    vad_component, components, vad_queue = build_async_components(config)
    
    with manifest_path.open("a", encoding="utf-8") as sink:
        for ref in refs:
            if ref.utt_id in already_done:
                continue
            
            trace = TimingTrace(SystemClock())
            trace.stamp("stream_start")
            
            record = await run_utterance_async(
                ref, vad_component, components, trace, out, fixtures_root
            )
            
            sink.write(record.model_dump_json() + "\n")
            sink.flush()
            fresh += 1
    
    finished_at = datetime.now(UTC)
    
    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "config": str(config_path),
                "refs": str(refs_path),
                "components": collect_provenance(
                    {k: v.component for k, v in components.items()}
                ),
                "host": _host_info(),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": round(
                    (finished_at - started_at).total_seconds(), 3
                ),
                **run_identity(config_path, refs_path),
                "config_snapshot": "config.snapshot.yaml",
                "utterances_total": len(refs),
                "utterances_fresh": fresh,
                "utterances_resumed": len(already_done),
                "status": "resumed" if already_done else "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    
    from onevoice.benchmark import BenchmarkResult, evaluate
    table = evaluate(manifest_path, refs_path, experiment=experiment)
    report_path = out / "report.md"
    report_path.write_text(
        render_report(table, read_run(manifest_path), config), encoding="utf-8"
    )
    
    return BenchmarkResult(
        table=table,
        manifest_path=manifest_path,
        report_path=report_path,
        metadata_path=metadata_path,
        utterances_fresh=fresh,
        utterances_resumed=len(already_done),
        status="resumed" if already_done else "complete",
    )


def _host_info() -> dict[str, str]:
    import platform
    return {
        "cpu": platform.processor(),
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }


def run_identity(config_path: str | Path, refs_path: str | Path) -> dict[str, Any]:
    from onevoice.provenance import _git, file_hash
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "refs_path": str(refs_path),
        "refs_hash": file_hash(refs_path),
        "framework_commit": commit or "unknown",
        "framework_dirty": True if status is None else bool(status.strip()),
    }


def render_report(
    table: Any, run_records: Sequence[RunRecord], config: ExperimentConfig
) -> str:
    from onevoice.benchmark import Q6_LIMITATION

    streaming = table.metrics.get("streaming_cost")
    streaming_detail = streaming.detail if streaming is not None else {}
    lines = [
        f"# Evaluation — {table.experiment or 'unnamed'}",
        "",
        f"> **{config.experiment.classification.capitalize()} benchmark.** "
        "Tracks regressions and guides engineering. Not sufficient for final "
        "model selection: a difference here is not a ranking unless an interval "
        "says so. See `docs/INTERPRETATION.md`.",
        "",
        "> **Known limitation (Q6).** `rtf` and `end_to_speech_ms` measure only",
        "> work performed after the final speech frame, so neither captures",
        "> per-frame work inside `push()`. `pipeline_rtf` spans the whole",
        "> utterance and does capture it.",
        "",
        "> Quote **`streaming_cost`**: it is the per-utterance difference between",
        "> the two, which is what `rtf` under-reports. The `rtf` and `pipeline_rtf`",
        "> percentiles are descriptive only and must not be subtracted from one",
        "> another - when a detector fails to fire, that utterance has a",
        "> `pipeline_rtf` but no `rtf`, so the two percentiles describe different",
        "> populations.",
        "",
        "> These are host-machine figures, not device predictions.",
        "",
        "> **Measured (paired).** `rtf` hides a median of "
        f"{streaming_detail.get('p50', 0):.4f} of audio duration per "
        f"utterance (worst {streaming_detail.get('max', 0):.4f}), which is "
        f"{streaming_detail.get('p50_pct_of_pipeline', 0):.1f}% of real "
        f"processing cost (worst "
        f"{streaming_detail.get('max_pct_of_pipeline', 0):.1f}%). Paired "
        f"over paired_n={streaming_detail.get('paired_n', 0)} utterances; "
        f"{streaming_detail.get('unpairable', 0)} could not be paired.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name, result in table.metrics.items():
        lines.append(f"| {name} | {result.value:.4g} |")
        for key, value in result.detail.items():
            lines.append(f"| ↳ {name}.{key} | {value} |")
    lines += ["", "| Count | Utterances |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in table.counts.items()]
    lines.append("")
    lines.append(Q6_LIMITATION.strip())
    lines.append("")
    return "\n".join(lines)


def read_run(path: str | Path) -> list[RunRecord]:
    import json

    from onevoice.eval.manifest import RunRecord
    with open(path, encoding="utf-8") as f:
        return [RunRecord(**json.loads(line)) for line in f if line.strip()]


def collect_provenance(components: dict[str, Any]) -> dict[str, Any]:
    """Collect provenance from sync components."""
    return {name: comp.provenance() for name, comp in components.items()}