"""TTS component benchmark — backend-agnostic by construction (D35).

Nothing in this module names a backend. Every implementation is resolved
through the registry from a config, given the same input text, timed the same
way, and archived identically. Piper's baseline status (D23) governs which
component ships by default, never how the comparison runs — so if a challenger
wins on latency or footprint, this makes it visible rather than burying it.

Deliberately does NOT score quality. No automatic metric in this framework
evaluates synthesis naturalness; that is the Phase 5 MOS panel's job (Q4). What
is measured here is what can be measured honestly: synthesis latency, the
duration of audio produced, and the ratio between them.

Archives every waveform (D31) and the backend's provenance (D33) so a later MOS
round has artifacts without re-running inference.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from onevoice.audio import write_wav
from onevoice.config import load_config
from onevoice.eval.manifest import read_refs
from onevoice.pipeline import build_components
from onevoice.provenance import run_identity
from onevoice.types import Clock, SystemClock

TTS_ONLY_CAVEAT = """\
> **Scope.** These figures measure the TTS component alone, synthesizing text
> taken directly from the reference manifest. No VAD, ASR or MT is involved, and
> **no quality metric is reported** — nothing in this framework scores synthesis
> naturalness, which is the Phase 5 MOS panel's job.
>
> Latency here is synthesis time for one utterance. It is not the competition's
> end-of-speech-to-first-audio metric and must not be compared against it.
>
> Every backend below was given the same input text, timed the same way, and
> archived identically. These are host-machine figures, not device predictions.
"""


@dataclass(frozen=True)
class Synthesis:
    """One utterance's TTS-only result, archived verbatim (D31)."""

    utt_id: str
    backend: str
    text: str
    wav_path: str
    latency_ms: float
    audio_seconds: float
    sample_rate: int
    synthesis_rtf: float


@dataclass(frozen=True)
class TtsBenchmarkResult:
    """Everything one TTS-only benchmark run produced."""

    experiment: str
    backend: str
    syntheses: list[Synthesis] = field(default_factory=list)
    syntheses_path: Path = Path()
    metadata_path: Path = Path()
    report_path: Path = Path()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, int(len(ordered) * fraction + 0.999999) - 1)
    return ordered[index]


def _summary_markdown(title: str, values: list[float], unit: str) -> str:
    if not values:
        return f"| {title} | n/a | n/a | n/a | n/a |\n"
    return (
        f"| {title} ({unit}) "
        f"| {_percentile(values, 0.50):.4g} "
        f"| {_percentile(values, 0.95):.4g} "
        f"| {max(values):.4g} "
        f"| {sum(values) / len(values):.4g} |\n"
    )


def render_report(
    experiment: str,
    backend: str,
    syntheses: list[Synthesis],
    notes: str = "",
) -> str:
    """Caveat, then the config's own notes, then measurements."""
    sections = [
        f"# TTS component benchmark - {experiment or 'unnamed'}",
        "",
        TTS_ONLY_CAVEAT,
    ]
    if notes.strip():
        sections += [f"> **This configuration.** {notes.strip()}", ""]

    latencies = [s.latency_ms for s in syntheses]
    durations = [s.audio_seconds for s in syntheses]
    rtfs = [s.synthesis_rtf for s in syntheses]

    sections += [
        f"## Measurements - backend `{backend}`",
        "",
        "| Measure | p50 | p95 | max | mean |",
        "|---|---|---|---|---|",
        _summary_markdown("Synthesis latency", latencies, "ms").rstrip(),
        _summary_markdown("Audio produced", durations, "s").rstrip(),
        _summary_markdown("Synthesis RTF", rtfs, "ratio").rstrip(),
        "",
        f"Utterances synthesized: {len(syntheses)}",
        "",
    ]
    return "\n".join(sections)


def run_tts_benchmark(
    config_path: str | Path,
    refs_path: str | Path,
    out_dir: str | Path,
    experiment: str = "",
    text_field: str = "ref_text",
    clock: Clock | None = None,
) -> TtsBenchmarkResult:
    """Synthesize one manifest field with the config's TTS component.

    `text_field` selects which manifest column is spoken — `ref_text` (target
    language) or `src_text` (source language). Whichever is chosen applies to
    every backend equally; that symmetry is the point.
    """
    the_clock = clock if clock is not None else SystemClock()
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    # Archive the config beside the run: a hash detects that the file changed,
    # only a copy lets the experiment be re-run afterwards (F6).
    shutil.copyfile(config_path, out / "config.snapshot.yaml")
    backend = config.tts.model
    components = build_components(config)
    tts = components["tts"]

    syntheses: list[Synthesis] = []
    for ref in read_refs(refs_path):
        text = getattr(ref, text_field)
        tts.reset()

        started = the_clock.now()
        events = list(tts.push(text, is_final=True)) + list(tts.flush())
        latency_ms = (the_clock.now() - started) * 1000.0

        if not events:
            continue

        rate = events[0].sample_rate
        pcm = events[0].pcm if len(events) == 1 else _concatenate(events)
        wav_path = out / "audio" / f"{ref.utt_id}.wav"
        write_wav(wav_path, pcm, rate)

        audio_seconds = len(pcm) / rate
        syntheses.append(
            Synthesis(
                utt_id=ref.utt_id,
                backend=backend,
                text=text,
                wav_path=str(wav_path),
                latency_ms=latency_ms,
                audio_seconds=audio_seconds,
                sample_rate=rate,
                synthesis_rtf=(
                    latency_ms / 1000.0 / audio_seconds
                    if audio_seconds
                    else float("nan")
                ),
            )
        )

    syntheses_path = out / "syntheses.jsonl"
    syntheses_path.write_text(
        "\n".join(json.dumps(asdict(s)) for s in syntheses) + "\n", encoding="utf-8"
    )

    from onevoice.benchmark import collect_provenance

    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "benchmark": "tts_component",
                "config": str(config_path),
                **run_identity(config_path, refs_path),
                "config_snapshot": "config.snapshot.yaml",
                "refs": str(refs_path),
                "text_field": text_field,
                "components": collect_provenance({"tts": tts}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = out / "report.md"
    report_path.write_text(
        render_report(experiment, backend, syntheses, config.experiment.notes),
        encoding="utf-8",
    )

    return TtsBenchmarkResult(
        experiment=experiment,
        backend=backend,
        syntheses=syntheses,
        syntheses_path=syntheses_path,
        metadata_path=metadata_path,
        report_path=report_path,
    )


def _concatenate(events):
    import numpy as np

    rates = {e.sample_rate for e in events}
    if len(rates) > 1:
        raise ValueError(
            f"backend emitted mixed sample rates {sorted(rates)} within one "
            "utterance; concatenating them would produce a file that is wrong "
            "in a way nothing downstream can detect"
        )
    return np.concatenate([e.pcm for e in events])
