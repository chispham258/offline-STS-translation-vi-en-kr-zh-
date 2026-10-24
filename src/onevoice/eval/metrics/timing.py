"""Latency and real-time-factor metrics derived from a run's timing stamps.

Summaries report percentiles rather than only a mean. The competition's 2000 ms
figure is a ceiling, not an average, and a mean of 1.2 s comfortably hides a p95
of 2.4 s.

Naming
------

Three names sit at two different levels, and keeping them straight is what stops
this module accumulating `true_rtf`, `actual_rtf` and `real_rtf`.

**Level 1 — ratios over a named window.** A metric ending in `rtf` is processing
wall time divided by audio duration, over a window its prefix names. Neither is
more correct than the other; they answer different questions.

`rtf` spans `speech_end -> tts_done`. It answers: what does the user wait
through after they stop speaking? This models the competition's requirement.

`pipeline_rtf` spans `stream_start -> tts_done`. It answers: what does this
configuration cost to run in total? It includes per-frame work inside `push()`.

**Level 2 — statistics derived from those ratios.** `streaming_cost` is the
paired per-utterance difference `pipeline_rtf - rtf`: the work `rtf` cannot see.
It is a derived quantity, not a window, so it takes a noun naming what it
measures rather than an adjective attached to `rtf`.

The two levels are what keeps the taxonomy closed: **a window metric is a
primitive measurement and never depends on another metric; a derived metric
compares primitive measurements and never defines a window of its own.** So
there is no such thing as a `pipeline_cost_rtf` or a `pipeline_difference_rtf` —
a name carrying both a window and a comparison is describing two things at once,
and one of them belongs at the other level.

Rules for anything added later:

1. A new window gets `<window>_rtf`. A new derived quantity gets a noun. Never
   both in one name.
2. **Never** `true_`, `actual_`, `real_` or `correct_`. Each ratio is correct
   over its own window; a name implying otherwise mislabels the others as wrong
   and invites readers to discard a figure that is sound. `test_metric_naming`
   in `tests/eval/test_pipeline_rtf.py` enforces this, and was verified by
   adding a `true_rtf` and watching it fail.
3. Percentiles of two `*_rtf` metrics are **not** comparable to each other
   unless both are measurable on the same utterances, which is not guaranteed:
   an utterance whose detector never fired has a `pipeline_rtf` but no
   `speech_end`, so no `rtf`. Compare by pairing per utterance and report
   `paired_n` — that is what `streaming_cost` exists to do (D45).
"""

from __future__ import annotations

import math

from onevoice.eval.manifest import RunRecord
from onevoice.eval.metrics.base import MetricResult
from onevoice.types import SPEECH_END, TTS_DONE, TTS_FIRST_AUDIO


def end_to_speech_ms(record: RunRecord) -> float | None:
    """Milliseconds from end of speech to first synthesized audio.

    Returns None when either stamp is absent, or when the interval is
    negative. A negative interval means the stamps are inconsistent, not
    that the pipeline was instant, so it must not collapse to a fast-looking
    number and drag a percentile down.
    """
    start = record.timing.get(SPEECH_END)
    end = record.timing.get(TTS_FIRST_AUDIO)
    if start is None or end is None or end < start:
        return None
    return (end - start) * 1000.0


def rtf(record: RunRecord) -> float | None:
    """Post-speech processing time divided by audio duration.

    Measures `speech_end -> tts_done`, so it excludes anything a component does
    inside `push()` as audio arrives. That exclusion is deliberate — it models
    the competition's end-of-speech-to-audio requirement — and it is why
    `pipeline_rtf` exists alongside it. Unchanged since Phase 1; every recorded
    Phase 4 figure uses this definition (D45).
    """
    start = record.timing.get(SPEECH_END)
    end = record.timing.get(TTS_DONE)
    if start is None or end is None or end < start or record.audio_duration_s <= 0:
        return None
    return (end - start) / record.audio_duration_s


def pipeline_rtf(record: RunRecord) -> float | None:
    """Whole-utterance processing time divided by audio duration (D45, F2).

    Measures `stream_start -> tts_done`, so unlike `rtf` it includes per-frame
    work done inside `push()`. A batch-shaped configuration does nothing before
    `speech_end` and its two figures coincide; the gap between them *is* the
    streaming cost, which `rtf` alone cannot see.

    Both stamps predate this metric, so every archived run can be re-scored
    without being re-run — which is what let the historical record be corrected
    rather than restarted.
    """
    start = record.timing.get("stream_start")
    end = record.timing.get(TTS_DONE)
    if start is None or end is None or end < start or record.audio_duration_s <= 0:
        return None
    return (end - start) / record.audio_duration_s


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Values must be sorted."""
    if not values:
        return math.nan
    index = max(0, math.ceil(fraction * len(values)) - 1)
    return values[index]


def _summarize(name: str, measured: list[float], unmeasurable: int) -> MetricResult:
    values = sorted(measured)
    p50 = _percentile(values, 0.50)
    detail: dict[str, object] = {
        "measured": len(values),
        "unmeasurable": unmeasurable,
        "p50": p50,
        "p95": _percentile(values, 0.95),
        "max": values[-1] if values else math.nan,
        "mean": sum(values) / len(values) if values else math.nan,
    }
    return MetricResult(name=name, value=p50, detail=detail)


def latency_summary(records: list[RunRecord]) -> MetricResult:
    """End-to-speech latency across a run. Headline value is p50."""
    values = [v for v in (end_to_speech_ms(r) for r in records) if v is not None]
    return _summarize("end_to_speech_ms", values, len(records) - len(values))


def rtf_summary(records: list[RunRecord]) -> MetricResult:
    """Real-time factor across a run. Headline value is p50, but max is the risk."""
    values = [v for v in (rtf(r) for r in records) if v is not None]
    return _summarize("rtf", values, len(records) - len(values))


def pipeline_rtf_summary(records: list[RunRecord]) -> MetricResult:
    """Whole-pipeline real-time factor across a run. Headline value is p50."""
    values = [v for v in (pipeline_rtf(r) for r in records) if v is not None]
    return _summarize("pipeline_rtf", values, len(records) - len(values))


def streaming_cost_summary(records: list[RunRecord]) -> MetricResult:
    """Per-utterance cost `rtf` cannot see, paired (D45, F2).

    **This is the statistic to quote.** `rtf` and `pipeline_rtf` percentiles are
    descriptive only: they are computed over different populations whenever a
    detector failed to fire, because such an utterance has a `pipeline_rtf` but
    no `speech_end` and therefore no `rtf`. The difference of two medians over
    two populations is not a quantity. The per-utterance difference is, and it
    is well defined exactly on the subset where both exist — so `paired_n`
    travels with it and is never implicit.

    Headline value is the median hidden share of audio duration.
    `p50_pct_of_pipeline` expresses the same thing as a fraction of real
    processing cost, which is what tells a reader how much `rtf` under-reports.
    """
    pairs = [
        (pipeline_rtf(r), rtf(r))
        for r in records
        if pipeline_rtf(r) is not None and rtf(r) is not None
    ]
    hidden = sorted(full - post for full, post in pairs)
    percentages = sorted(
        100.0 * (full - post) / full for full, post in pairs if full > 0
    )

    detail: dict[str, object] = {
        "paired_n": len(pairs),
        "unpairable": len(records) - len(pairs),
        "p50": _percentile(hidden, 0.50),
        "p95": _percentile(hidden, 0.95),
        "max": hidden[-1] if hidden else math.nan,
        "p50_pct_of_pipeline": _percentile(percentages, 0.50),
        "max_pct_of_pipeline": percentages[-1] if percentages else math.nan,
    }
    return MetricResult(
        name="streaming_cost", value=_percentile(hidden, 0.50), detail=detail
    )
