"""MT component benchmark: gold (src_text, ref_text) pairs, no pipeline.

Why this exists. The pipeline feeds MT whatever ASR emitted, and EchoASR
derives its text from audio rather than a script (D18), so pipeline BLEU has
been 0 for every MT backend integrated so far - it measures the dummy ASR,
not translation. This module bypasses the pipeline entirely: it reads
src_text straight from the reference manifest, pushes it through the MT
component alone, and scores the hypothesis against ref_text.

It calls bleu()/chrf() from onevoice.eval.metrics.text unchanged, the same
functions the evaluator calls, so this number and a future end-to-end number
are computed identically (D21: src/onevoice/eval/** shows zero diff for this
module).

The ceiling framing. Feeding src_text means feeding a *gold* transcript, so
this measures MT quality given perfect ASR - a ceiling on end-to-end quality,
not a prediction of it. MT_ONLY_CAVEAT (below) says so on every report this
produces, and D28 archives every raw hypothesis alongside the metrics so a
qualitative read of a translation needs no re-inference.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from onevoice.config import build, load_config
from onevoice.eval.evaluator import ResultsTable
from onevoice.eval.manifest import RefRecord, read_refs
from onevoice.eval.metrics.text import bleu, chrf
from onevoice.eval.report import to_markdown
from onevoice.provenance import run_identity
from onevoice.types import Clock, SystemClock

MT_ONLY_CAVEAT = """\
> **Scope.** These figures score the MT component alone, on gold source text
> taken from the reference manifest. No ASR, VAD or TTS is involved. Because
> the input is a perfect transcript, this is a **ceiling** on end-to-end
> quality, not a prediction of it. Compare against the end-to-end run to see
> what the ASR stage costs. Latency here is MT-stage only and is not the
> competition's end-of-speech-to-first-audio metric.
>
> **Sample size.** The fixture manifest has three utterances. This is a
> smoke-level baseline, not a corpus result - do not quote it as a model
> comparison. The real corpus arrives at Task 8.
"""


@dataclass(frozen=True)
class Hypothesis:
    """One utterance's MT-only result, archived verbatim (D28)."""

    utt_id: str
    backend: str
    src_text: str
    ref_text: str
    hyp_text: str
    latency_ms: float


@dataclass(frozen=True)
class MtBenchmarkResult:
    """Everything one MT-only benchmark run produced."""

    table: ResultsTable
    hypotheses: list[Hypothesis] = field(default_factory=list)
    hyps_path: Path = Path()
    report_path: Path = Path()
    metadata_path: Path = Path()


def _target_language(refs: list[RefRecord]) -> str:
    languages = {r.tgt_lang for r in refs}
    if len(languages) != 1:
        raise ValueError(
            f"reference manifest spans multiple target languages "
            f"{sorted(languages)}; BLEU tokenization is per target language "
            "(D14), so benchmark each direction separately"
        )
    return languages.pop()


def _translate(mt: object, src_text: str, clock: Clock) -> tuple[str, float]:
    """push()+flush() one utterance through mt. Returns (hyp_text, latency_ms).

    Real backends (OpusMT, M2M100MT, NLLBMT, PivotMT) buffer on push() and
    emit their translation on flush(); IdentityMT emits immediately on
    push(). Collecting events from both and keeping the last one's text
    covers either shape without caring which it is.
    """
    mt.reset()
    start = clock.now()
    hyp_text = ""
    for event in mt.push(src_text, is_final=True):
        hyp_text = event.text
    for event in mt.flush():
        hyp_text = event.text
    latency_ms = (clock.now() - start) * 1000.0
    return hyp_text, latency_ms


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Values must be sorted."""
    if not values:
        return math.nan
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def _latency_markdown(latencies: list[float]) -> str:
    ordered = sorted(latencies)
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)
    worst = ordered[-1] if ordered else math.nan
    lines = [
        "## Latency - MT stage only",
        "",
        "One push()+flush() cycle per utterance. Not the competition's "
        "end-of-speech-to-first-audio metric - MT alone, never interleaved "
        "with the quality table above.",
        "",
        "| Stat | ms |",
        "|---|---|",
        f"| p50 | {p50:.3f} |",
        f"| p95 | {p95:.3f} |",
        f"| max | {worst:.3f} |",
        f"| n | {len(latencies)} |",
    ]
    return "\n".join(lines) + "\n"


def render_report(
    experiment: str,
    quality: ResultsTable,
    latencies: list[float],
    notes: str = "",
) -> str:
    """Caveat, then quality (evaluator's own renderer), then latency - in that
    order and in separate sections, so a reader cannot mistake a stage timing
    for the competition's end-to-end metric.

    A config's `notes` are rendered immediately after the scope caveat and
    before any number, so a qualification cannot be read separately from the
    figure it qualifies.
    """
    sections = [
        f"# MT component benchmark - {experiment or 'unnamed'}",
        "",
        MT_ONLY_CAVEAT,
    ]
    if notes.strip():
        sections += [f"> **This configuration.** {notes.strip()}", ""]
    sections += [
        "## Translation quality - MT-only, gold source text",
        "",
        to_markdown(quality),
        _latency_markdown(latencies),
    ]
    return "\n".join(sections)


def run_mt_benchmark(
    config_path: str | Path,
    refs_path: str | Path,
    out_dir: str | Path,
    experiment: str = "",
    clock: Clock | None = None,
) -> MtBenchmarkResult:
    """Feed every reference's src_text through the config's MT component alone.

    No audio, no VAD, no ASR, no TTS: this never imports onevoice.pipeline.
    """
    the_clock = clock if clock is not None else SystemClock()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    refs = read_refs(refs_path)
    backend = config.mt.model
    mt = build("mt", config.mt, languages=config.languages)

    hypotheses = []
    for ref in refs:
        hyp_text, latency_ms = _translate(mt, ref.src_text, the_clock)
        hypotheses.append(
            Hypothesis(
                utt_id=ref.utt_id,
                backend=backend,
                src_text=ref.src_text,
                ref_text=ref.ref_text,
                hyp_text=hyp_text,
                latency_ms=latency_ms,
            )
        )

    hyps_path = out / "hypotheses.jsonl"
    body = "\n".join(json.dumps(asdict(h)) for h in hypotheses)
    hyps_path.write_text(body + "\n", encoding="utf-8")

    tgt_lang = _target_language(refs)
    hyp_texts = [h.hyp_text for h in hypotheses]
    ref_texts = [h.ref_text for h in hypotheses]
    quality = ResultsTable(
        experiment=experiment,
        metrics={
            "bleu": bleu(hyp_texts, ref_texts, tgt_lang),
            "chrf": chrf(hyp_texts, ref_texts),
        },
        counts={"utterances": len(hypotheses)},
    )

    # F6: this benchmark previously recorded no metadata at all, so the MT
    # ceiling figures Task 9's correction rests on had no provenance - not the
    # model, not the config, not the code that produced them.
    from onevoice.benchmark import collect_provenance

    shutil.copyfile(config_path, out / "config.snapshot.yaml")
    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "benchmark": "mt_component",
                **run_identity(config_path, refs_path),
                "config_snapshot": "config.snapshot.yaml",
                "components": collect_provenance({"mt": mt}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    latencies = [h.latency_ms for h in hypotheses]
    report_path = out / "report.md"
    report_path.write_text(
        render_report(
            experiment, quality, latencies, config.experiment.notes
        ),
        encoding="utf-8",
    )

    return MtBenchmarkResult(
        table=quality,
        hypotheses=hypotheses,
        hyps_path=hyps_path,
        report_path=report_path,
        metadata_path=metadata_path,
    )
