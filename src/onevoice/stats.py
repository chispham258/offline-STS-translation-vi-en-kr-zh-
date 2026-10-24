"""Interval evidence for comparisons between runs (F3, D48).

Policy: `docs/INTERPRETATION.md`. A point estimate is never a conclusion, so a
comparison between two configurations reports an interval or it does not claim a
ranking.

Deliberately **outside** `onevoice/eval/`. This computes no metric and changes
no metric's meaning — it consumes metric values and describes how certain their
difference is. Interpretation is a reporting concern, so D21 is untouched and no
exception is needed.

Pairing is per utterance because that is what removes corpus difficulty from the
comparison: two configurations scored over the same utterances differ only by
the thing that changed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0
DEFAULT_LEVEL = 0.95


@dataclass(frozen=True)
class Comparison:
    """The evidence for a difference between two configurations."""

    name: str
    delta: float
    low: float
    high: float
    p_greater: float
    n: int
    resamples: int
    level: float = DEFAULT_LEVEL

    @property
    def resolves(self) -> bool:
        """Whether the interval excludes zero.

        The bar for claiming a direction. Deliberately not called
        `significant`: this is a bootstrap interval, not a hypothesis test, and
        borrowing that word imports machinery this framework does not use.
        """
        return (self.low > 0.0) or (self.high < 0.0)

    def sentence(self, higher_is_better: bool = True) -> str:
        """One line stating the observation, the evidence, and the conclusion.

        The wording rule from `docs/INTERPRETATION.md` is applied here rather
        than left to whoever writes the report, because "no comparative language
        without an interval" only holds if the interval writes the language.
        """
        evidence = (
            f"Δ={self.delta:+.4g}, {self.level:.4g} CI "
            f"[{self.low:+.4g}, {self.high:+.4g}], "
            f"P(B>A)={self.p_greater:.3f}, n={self.n}"
        )
        if not self.resolves:
            return (
                f"**{self.name}** — {evidence}. No statistically supported "
                f"difference at n={self.n}."
            )
        direction = "higher" if self.delta > 0 else "lower"
        better = (self.delta > 0) == higher_is_better
        return (
            f"**{self.name}** — {evidence}. B is {direction} than A, interval "
            f"excludes zero ({'favours B' if better else 'favours A'})."
        )


def paired_bootstrap(
    ids: Sequence[str],
    score_a: Callable[[Sequence[str]], float],
    score_b: Callable[[Sequence[str]], float],
    *,
    name: str = "metric",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Comparison:
    """Bootstrap the difference B − A over the same utterances.

    `score_a` / `score_b` take a list of utterance ids and return a corpus-level
    score, so this works for any metric that is not a simple per-utterance mean
    — BLEU in particular, which cannot be averaged.

    `level` is the two-sided interval level. It exists because a family of
    comparisons needs a correction stated in advance — Phase 5.0 pre-registers
    three comparisons at Bonferroni-corrected 98.33%, and a tool that could only
    produce 95% would have silently contradicted its own specification.

    Deterministic given `seed`, because a confidence interval that moves between
    runs of the same data is not evidence.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    if not ids:
        raise ValueError("cannot bootstrap an empty utterance set")

    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sample = [rng.choice(ids) for _ in ids]
        deltas.append(score_b(sample) - score_a(sample))
    deltas.sort()

    k = int(((1.0 - level) / 2.0) * resamples)
    return Comparison(
        name=name,
        delta=score_b(list(ids)) - score_a(list(ids)),
        low=deltas[k],
        high=deltas[resamples - 1 - k],
        p_greater=sum(1 for d in deltas if d > 0) / resamples,
        n=len(ids),
        resamples=resamples,
        level=level,
    )
