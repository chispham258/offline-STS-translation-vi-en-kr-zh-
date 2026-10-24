"""Joins a run manifest against references and computes every metric.

Imports no pipeline and no model code (D5). Given the same manifests it produces
the same numbers, which is what makes a metric change re-runnable without
re-running a pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from onevoice.eval.manifest import RefRecord, RunRecord, read_refs, read_run
from onevoice.eval.metrics.base import MetricResult
from onevoice.eval.metrics.error_rate import (
    cer,
    cer_normalized,
    wer,
    wer_normalized,
)
from onevoice.eval.metrics.text import bleu, chrf
from onevoice.eval.metrics.timing import (
    latency_summary,
    pipeline_rtf_summary,
    rtf_summary,
    streaming_cost_summary,
)


# COMET is optional (extra) - import lazily to avoid hard dependency
def _comet_available() -> bool:
    try:
        import comet  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class ResultsTable:
    """Every metric for one experiment, plus the counts needed to read them."""

    experiment: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


def _target_language(refs: list[RefRecord]) -> str:
    languages = {r.tgt_lang for r in refs}
    if len(languages) != 1:
        raise ValueError(
            f"corpus spans multiple target languages {sorted(languages)}; BLEU "
            "tokenization is per target language (D14), so evaluate each "
            "direction separately"
        )
    return languages.pop()


def _nan_metric(name: str) -> MetricResult:
    return MetricResult(name=name, value=math.nan, detail={"measured": 0})


def evaluate(
    run_path: str | Path,
    refs_path: str | Path,
    experiment: str = "",
) -> ResultsTable:
    """Score a run manifest against a reference manifest."""
    run_records = read_run(run_path)
    ref_by_id = {r.utt_id: r for r in read_refs(refs_path)}

    run_ids = {r.utt_id for r in run_records}
    for record in run_records:
        if record.utt_id not in ref_by_id:
            raise KeyError(
                f"run contains {record.utt_id!r} with no matching reference; "
                "scoring a subset is not comparable to scoring the full set"
            )

    missing = sorted(set(ref_by_id) - run_ids)
    if missing:
        raise KeyError(
            f"run does not cover reference utt_id(s) {missing}; a partial run "
            "scored against its subset would look like a complete, clean run"
        )

    scored: list[RunRecord] = [r for r in run_records if r.status == "ok"]
    errored = len(run_records) - len(scored)
    counts = {
        "total": len(run_records),
        "scored": len(scored),
        "errored": errored,
        # D49/F5: "scored" counts records the evaluator could score and
        # "errored" counts records that raised. Neither notices an utterance
        # that ran cleanly and produced nothing - which is how a run reporting
        # errored: 0 concealed a 24% dropout. An empty hypothesis is not an
        # error, but it is not a result either, and a reader deserves to see
        # the difference without recomputing it.
        "empty_outputs": sum(1 for r in scored if not (r.hyp_text or "").strip()),
        "references": len(ref_by_id),
    }

    refs_for_scored = [ref_by_id[r.utt_id] for r in scored]
    tgt_lang = _target_language(list(ref_by_id.values()))

    if scored:
        hyps = [r.hyp_text for r in scored]
        gold = [ref_by_id[r.utt_id].ref_text for r in scored]
        asr_hyps = [r.asr_text for r in scored]
        asr_gold = [r.src_text for r in refs_for_scored]
        srcs = [ref_by_id[r.utt_id].src_text for r in scored]
        metrics = {
            "bleu": bleu(hyps, gold, tgt_lang),
            "chrf": chrf(hyps, gold),
            "wer": wer(asr_hyps, asr_gold),
            "cer": cer(asr_hyps, asr_gold),
            # Reported alongside, never instead of: raw wer/cer are what every
            # recorded Phase 4 figure means (D49).
            "wer_normalized": wer_normalized(asr_hyps, asr_gold),
            "cer_normalized": cer_normalized(asr_hyps, asr_gold),
        }
        # COMET is optional (D55): competition scores accuracy as BLEU + COMET.
        # Only compute if the extra is installed; never block on missing COMET.
        if _comet_available():
            try:
                from onevoice.eval.metrics.comet import comet
                metrics["comet"] = comet(srcs, hyps, gold)
            except Exception:
                # COMET may fail for various reasons (model download, GPU OOM, etc.)
                # Don't let it break the whole evaluation.
                pass
    else:
        metrics = {
            name: _nan_metric(name)
            for name in (
                "bleu",
                "chrf",
                "wer",
                "cer",
                "wer_normalized",
                "cer_normalized",
                "comet",
            )
        }

    metrics["end_to_speech_ms"] = latency_summary(scored)
    metrics["rtf"] = rtf_summary(scored)
    # D45/F2: rtf cannot see work done inside push(). pipeline_rtf spans
    # stream_start -> tts_done, so the gap between the two is the
    # streaming cost. Added, never substituted - see docs/KNOWLEDGE.md.
    metrics["pipeline_rtf"] = pipeline_rtf_summary(scored)
    # The paired difference is the statistic that means something: the two RTF
    # percentiles above are computed over different populations whenever a
    # detector failed to fire, so subtracting them is not a quantity.
    metrics["streaming_cost"] = streaming_cost_summary(scored)

    return ResultsTable(experiment=experiment, metrics=metrics, counts=counts)
