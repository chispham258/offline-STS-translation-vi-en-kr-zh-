"""The selection hierarchy (D60).

    Gate D -> eligibility/coverage -> quality -> non-inferiority -> tie-breakers

Two things are deliberately kept apart:

**Statistical resolution** — how small a difference the experiment can reliably
distinguish. A property of the corpus and the metric.

**Engineering acceptance tolerance (δ)** — how much quality we are willing to
give up for speed, size or deployability. A judgement, owner-set.

The noise floor is *evidence about* δ, never a derivation of it: an experiment
becoming more sensitive does not make us more willing to lose quality.

**δ is a ceiling on acceptable regression, not a budget to spend.** The rule
below is written so that "choose the fastest candidate within δ" cannot happen:
performance breaks ties *only* between candidates that are statistically
indistinguishable on quality. A candidate that is genuinely worse — even by less
than δ — never beats a better one on speed.

Candidates are classified individually rather than collapsed into one global
escalate flag: `superior` / `non_inferior` / `inferior` /
`unresolved_decision_relevant` / `unresolved_decision_irrelevant`.

Both unresolved statuses mean *insufficient evidence for a classification*.
Neither means "probably inferior", neither is ever imputed to a resolved
verdict, and neither enters a quality tier. They differ only in whether the
undecidedness could still change the outcome (D66):

    decision-relevant   <=> some admissible value inside its own uncertainty
                            interval would make this candidate the selected one
    decision-irrelevant <=> no such value exists

That question is answered by re-applying this same hierarchy to the
hypothetical, not by a hand-written condition, so it cannot drift away from the
rule it is meant to be about. See `_decision_relevant`.

Three claims that are routinely conflated and are kept apart here:

    "demonstrated superior to the incumbent" — an interval, versus one
        reference. Several candidates can hold it simultaneously.
    "selected" — the output of this whole hierarchy.
    "pairwise superior to another challenger" — requires a pairwise comparison
        that this module never invents. `_equivalence_tier` only consults one
        when the caller supplies it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from onevoice.stats import Comparison

#: Provisional verdict from `classify`, which sees one comparison and therefore
#: cannot know whether the undecidedness matters. `select` refines it into one
#: of the two decision statuses below.
UNRESOLVED = "unresolved"

#: An unresolved comparison that could still change the selection: some value
#: inside its interval would make this candidate the selected one.
UNRESOLVED_RELEVANT = "unresolved_decision_relevant"

#: An unresolved comparison that cannot change the selection: no value inside
#: its interval would make this candidate the selected one. Still undecided —
#: this is *not* a statement that the candidate is inferior.
UNRESOLVED_IRRELEVANT = "unresolved_decision_irrelevant"

VERDICTS = (
    "superior",
    "non_inferior",
    "inferior",
    UNRESOLVED_RELEVANT,
    UNRESOLVED_IRRELEVANT,
)


@dataclass(frozen=True)
class Margin:
    """Per-slot, per-metric acceptance tolerance (D60).

    Never global. ASR `wer_normalized`, MT COMET and TTS MOS live on different
    scales with different semantics, so one absolute δ cannot apply to all of
    them — 0.018 means something very different on a 0–1 error rate than on a
    1–5 opinion score.

    `resolution` is the measured bootstrap half-width of the **incumbent's**
    metric, computed before any candidate is scored so the derivation cannot be
    circular. `k` is the owner-approved judgement.
    """

    slot: str
    metric: str
    k: float
    resolution: float
    lower_is_better: bool = True

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.resolution <= 0:
            raise ValueError(
                f"resolution must be positive, got {self.resolution}. It is the "
                "measured incumbent half-width and must be computed before "
                "candidates are scored."
            )

    @property
    def delta(self) -> float:
        """The acceptance ceiling. Not a target."""
        return self.k * self.resolution


def classify(comparison: Comparison, margin: Margin) -> str:
    """Verdict for one candidate against the incumbent.

    `comparison` is candidate − incumbent on the primary metric, at the
    multiplicity-corrected level.
    """
    lo, hi = comparison.low, comparison.high
    if not margin.lower_is_better:
        lo, hi = -hi, -lo  # re-orient so "more negative is better"

    if hi < 0.0:
        return "superior"
    if hi < margin.delta:
        return "non_inferior"
    if lo > margin.delta:
        return "inferior"
    return UNRESOLVED


@dataclass(frozen=True)
class Selection:
    """The outcome of applying the hierarchy. Mechanical, not a judgement."""

    slot: str
    chosen: str
    reason: str
    verdicts: dict[str, str]
    quality_tier: tuple[str, ...]
    escalate: bool = False
    #: Candidates that could not be run, mapped to why. Deliberately kept apart
    #: from `verdicts`: an execution failure is not a quality measurement, and
    #: filing it as `inferior` would let a deployment fact be cited later as
    #: "we measured it and it was worse".
    errored: dict[str, str] = field(default_factory=dict)

    @property
    def tie_broken_on_performance(self) -> bool:
        return len(self.quality_tier) > 1


def select(
    *,
    slot: str,
    incumbent: str,
    comparisons: dict[str, Comparison],
    margin: Margin,
    pairwise: Callable[[str, str], Comparison] | None = None,
    performance_rank: Callable[[str], float] | None = None,
    errored: dict[str, str] | None = None,
) -> Selection:
    """Apply the hierarchy. Quality first; performance only breaks true ties.

    `pairwise(a, b)` compares two *candidates* on the primary metric — needed to
    establish genuine quality-equivalence. Without it, no tie can be declared
    and the single best point estimate wins, which is the conservative default.

    `performance_rank` is consulted **only** among candidates already shown
    indistinguishable on quality. Lower is preferred (smaller/faster).

    `errored` maps candidates that could not be executed to the reason. They
    are carried through untouched and never receive a verdict — a model that
    OOMs or has no compatible processor has said nothing about its quality.
    """
    failures = dict(errored or {})
    both = sorted(set(failures) & set(comparisons))
    if both:
        raise ValueError(
            f"candidates {both} are both scored and errored; a candidate that "
            "produced a comparison did execute"
        )
    verdicts = {n: classify(c, margin) for n, c in comparisons.items()}

    # Q8/D66. An unresolved comparison blocks a selection only while it could
    # still change one. Each unresolved candidate is asked a counterfactual:
    # is there any admissible value inside its own interval under which this
    # hierarchy would select it? If not, escalating on its account would block
    # a decision the evidence already supports.
    resolved = {n: c for n, c in comparisons.items() if verdicts[n] != UNRESOLVED}
    for name in [n for n, v in verdicts.items() if v == UNRESOLVED]:
        relevant = _decision_relevant(
            name,
            comparisons[name],
            resolved,
            slot=slot,
            incumbent=incumbent,
            margin=margin,
            pairwise=pairwise,
            performance_rank=performance_rank,
        )
        verdicts[name] = UNRESOLVED_RELEVANT if relevant else UNRESOLVED_IRRELEVANT

    blocking = sorted(n for n, v in verdicts.items() if v == UNRESOLVED_RELEVANT)
    if blocking:
        return Selection(
            slot=slot,
            chosen=incumbent,
            reason=(
                f"escalate: {blocking} unresolved against the incumbent at this "
                "n and still able to lead; the comparison cannot yet support a "
                "selection"
            ),
            verdicts=verdicts,
            quality_tier=(),
            escalate=True,
            errored=failures,
        )

    superior = [n for n, v in verdicts.items() if v == "superior"]

    if superior:
        # Quality wins outright. Rank by point estimate, best first.
        ordered = sorted(superior, key=lambda n: comparisons[n].delta)
        if not margin.lower_is_better:
            ordered = sorted(superior, key=lambda n: -comparisons[n].delta)
        best = ordered[0]
        tier = _equivalence_tier(best, ordered, pairwise)
        chosen = _break_tie(tier, performance_rank)
        reason = (
            f"{chosen} is superior to the incumbent on quality"
            if len(tier) == 1
            else (
                f"{sorted(tier)} are superior and mutually indistinguishable on "
                f"quality; {chosen} chosen on performance"
            )
        )
        return Selection(slot, chosen, reason, verdicts, tuple(tier), errored=failures)

    # No superior candidate. The incumbent's quality stands; a non-inferior
    # challenger may still displace it, but only if it is genuinely equivalent.
    eligible = [n for n, v in verdicts.items() if v == "non_inferior"]
    if not eligible:
        return Selection(
            slot=slot,
            chosen=incumbent,
            reason=(
                "no candidate is superior or non-inferior; incumbent retained. "
                "Note delta is a ceiling on acceptable regression, not a budget"
            ),
            verdicts=verdicts,
            quality_tier=(incumbent,),
            errored=failures,
        )

    tier = tuple([incumbent, *eligible])
    chosen = _break_tie(tier, performance_rank)
    reason = (
        f"no candidate beats the incumbent on quality; {sorted(tier)} are "
        f"within delta={margin.delta:.4g}; {chosen} chosen on performance"
    )
    return Selection(slot, chosen, reason, verdicts, tier, errored=failures)


def _outcome_grid(comparison: Comparison, others: dict[str, Comparison],
                  margin: Margin) -> list[float]:
    """Admissible values inside a candidate's interval worth testing.

    The hierarchy is piecewise in the candidate's true value, and every piece
    boundary is one of: the interval endpoints, the verdict thresholds (0 and
    ±δ), or another candidate's point estimate — the value at which the
    best-point-estimate ordering flips. Testing those covers every distinct
    behaviour of the rule, so an exhaustive sweep of a continuous interval is
    unnecessary.

    Derived from the interval, the orientation and δ. No metric, slot or
    candidate is named.
    """
    lo, hi = comparison.low, comparison.high
    breakpoints = [lo, hi, 0.0, margin.delta, -margin.delta]
    breakpoints += [c.delta for c in others.values()]
    return sorted({t for t in breakpoints if lo <= t <= hi})


def _decision_relevant(
    name: str,
    comparison: Comparison,
    resolved: dict[str, Comparison],
    *,
    slot: str,
    incumbent: str,
    margin: Margin,
    pairwise: Callable[[str, str], Comparison] | None,
    performance_rank: Callable[[str], float] | None,
) -> bool:
    """Could any admissible value inside this interval make `name` selected?

    The counterfactual is asked against the *already-fixed* hierarchy rather
    than against a hand-written condition: each admissible value is substituted
    in as if the comparison had resolved exactly there, `select` is re-applied,
    and the question is simply whether it returns this candidate. So the
    definition of "could win" is always whatever the selection rule currently
    is, and cannot drift away from it.

    Other *unresolved* candidates are held out. Each is assessed independently;
    letting two undecided comparisons block each other would make both look
    irrelevant, or both relevant, for reasons unrelated to either one.

    Values that are themselves undecidable are skipped — an outcome landing
    exactly on a verdict threshold is one the rule cannot classify, so it cannot
    make this candidate selected. That also guarantees termination: every
    substituted comparison the inner `select` sees is resolved, so it finds
    nothing unresolved and never recurses.
    """
    for value in _outcome_grid(comparison, resolved, margin):
        hypothetical = dict(resolved)
        hypothetical[name] = replace(comparison, delta=value, low=value, high=value)
        if classify(hypothetical[name], margin) == UNRESOLVED:
            continue
        outcome = select(
            slot=slot,
            incumbent=incumbent,
            comparisons=hypothetical,
            margin=margin,
            pairwise=pairwise,
            performance_rank=performance_rank,
        )
        if outcome.chosen == name:
            return True
    return False


def _equivalence_tier(
    best: str,
    ordered: list[str],
    pairwise: Callable[[str, str], Comparison] | None,
) -> tuple[str, ...]:
    """Candidates statistically indistinguishable from the best.

    Without `pairwise` the tier is just the best candidate — the conservative
    default, because declaring a tie we did not measure is how a worse model
    wins on speed.
    """
    if pairwise is None:
        return (best,)
    tier = [best]
    for other in ordered[1:]:
        if not pairwise(best, other).resolves:
            tier.append(other)
    return tuple(tier)


def _break_tie(
    tier: tuple[str, ...], performance_rank: Callable[[str], float] | None
) -> str:
    """Choose within a quality-equivalent tier. Performance only decides here."""
    if len(tier) == 1 or performance_rank is None:
        return tier[0]
    return min(tier, key=performance_rank)
