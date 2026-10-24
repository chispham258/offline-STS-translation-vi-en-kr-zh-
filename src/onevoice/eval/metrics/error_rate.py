"""Word and character error rates.

Diagnostic metrics: they isolate ASR failures from MT failures. The competition
does not score them, but a bad BLEU with a good WER means something different
from a bad BLEU with a bad WER.
"""

from __future__ import annotations

import jiwer

from onevoice.eval.metrics.base import MetricResult


def _check_lengths(hyps: list[str], refs: list[str]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(
            f"hypothesis/reference length mismatch: {len(hyps)} vs {len(refs)}"
        )


def wer(hyps: list[str], refs: list[str]) -> MetricResult:
    """Corpus word error rate. 0.0 is perfect; values above 1.0 are possible."""
    _check_lengths(hyps, refs)
    return MetricResult(name="wer", value=float(jiwer.wer(refs, hyps)))


def cer(hyps: list[str], refs: list[str]) -> MetricResult:
    """Corpus character error rate. Matters more than WER for zh and ko."""
    _check_lengths(hyps, refs)
    return MetricResult(name="cer", value=float(jiwer.cer(refs, hyps)))


#: Punctuation stripped by `normalize_transcript`. Deliberately not a
#: `str.isalnum()` filter: that would also strip Vietnamese combining marks in
#: decomposed text and silently change what is being measured.
_PUNCTUATION = ".,!?;:\"'`()[]{}<>-–—…«»„“”‘’/\\|@#$%^&*_+=~"


def normalize_transcript(text: str) -> str:
    """Casing- and punctuation-insensitive form of a transcript (D49, F5).

    Whisper emits capitals and punctuation; FLEURS transcripts are lowercase and
    unpunctuated. Scoring one against the other charges the model for a
    formatting convention it was never asked to match.

    **This does not expand numbers.** Whisper writes `2` where FLEURS writes
    `hai`, and expanding digits is language-specific enough to be a methodology
    change rather than a normalisation. The normalised figure therefore still
    contains that mismatch, and is a *lower bound* on formatting effects, not a
    clean model-error measurement.
    """
    lowered = text.lower()
    stripped = "".join(" " if c in _PUNCTUATION else c for c in lowered)
    return " ".join(stripped.split())


def wer_normalized(hyps: list[str], refs: list[str]) -> MetricResult:
    """WER after casing/punctuation normalisation. Reported ALONGSIDE `wer`.

    Never a replacement: `wer` is what the competition-facing figure has always
    meant, and every recorded Phase 4 number uses it.
    """
    _check_lengths(hyps, refs)
    return MetricResult(
        name="wer_normalized",
        value=float(
            jiwer.wer(
                [normalize_transcript(r) for r in refs],
                [normalize_transcript(h) for h in hyps],
            )
        ),
    )


def cer_normalized(hyps: list[str], refs: list[str]) -> MetricResult:
    """CER after casing/punctuation normalisation. Reported ALONGSIDE `cer`."""
    _check_lengths(hyps, refs)
    return MetricResult(
        name="cer_normalized",
        value=float(
            jiwer.cer(
                [normalize_transcript(r) for r in refs],
                [normalize_transcript(h) for h in hyps],
            )
        ),
    )
