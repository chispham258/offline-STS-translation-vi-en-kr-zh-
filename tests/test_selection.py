"""The selection hierarchy (D60).

The property that matters most: **"choose the fastest candidate within δ" must
be structurally impossible.** Performance may only break ties between
candidates already shown indistinguishable on quality.
"""

from __future__ import annotations

import pytest

from onevoice.selection import (
    UNRESOLVED_IRRELEVANT,
    UNRESOLVED_RELEVANT,
    VERDICTS,
    Margin,
    classify,
    select,
)
from onevoice.stats import Comparison


def _cmp(name, delta, lo, hi, resolves_override=None):
    return Comparison(name, delta, lo, hi, 0.5, 200, 1000, 0.95)


ASR = Margin(slot="asr", metric="wer_normalized", k=1.5, resolution=0.0117)
#: vi->en MT, approved 2026-08-15. COMET is higher-is-better.
MT_EN = Margin(
    slot="mt", metric="comet", k=1.5, resolution=0.0117, lower_is_better=False
)


# --- margin semantics --------------------------------------------------------


def test_delta_is_k_times_the_measured_resolution():
    assert ASR.delta == pytest.approx(1.5 * 0.0117)


def test_margins_are_per_slot_and_per_metric_not_global():
    mos = Margin(slot="tts", metric="mos", k=1.5, resolution=0.2,
                 lower_is_better=False)
    assert mos.delta != ASR.delta
    assert mos.lower_is_better is False


def test_resolution_must_be_measured_not_assumed():
    with pytest.raises(ValueError, match="resolution"):
        Margin(slot="asr", metric="w", k=1.5, resolution=0.0)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        Margin(slot="asr", metric="w", k=0.0, resolution=0.01)


# --- verdicts ----------------------------------------------------------------


def test_superior_requires_the_interval_to_exclude_zero():
    assert classify(_cmp("a", -0.05, -0.08, -0.02), ASR) == "superior"


def test_non_inferior_is_below_delta_but_not_better():
    assert classify(_cmp("a", 0.005, -0.002, 0.010), ASR) == "non_inferior"


def test_inferior_is_worse_than_delta():
    assert classify(_cmp("a", 0.06, 0.04, 0.08), ASR) == "inferior"


def test_unresolved_straddles_delta():
    assert classify(_cmp("a", 0.015, -0.01, 0.05), ASR) == "unresolved"


def test_higher_is_better_metrics_are_reoriented():
    """MOS rises when quality improves; the verdict must not invert."""
    mos = Margin(slot="tts", metric="mos", k=1.5, resolution=0.2,
                 lower_is_better=False)
    assert classify(_cmp("a", +0.5, +0.3, +0.7), mos) == "superior"
    assert classify(_cmp("a", -0.9, -1.1, -0.7), mos) == "inferior"


# --- the anti-"fastest within delta" property -------------------------------


def test_a_worse_candidate_never_wins_on_speed():
    """The core property.

    `slow_good` is superior. `fast_ok` is merely non-inferior and much faster.
    Quality must win: delta is a ceiling on acceptable loss, not a budget to
    spend on speed.
    """
    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={
            "slow_good": _cmp("slow_good", -0.04, -0.06, -0.02),   # superior
            "fast_ok": _cmp("fast_ok", 0.010, 0.005, 0.015),       # non-inferior
        },
        margin=ASR,
        performance_rank=lambda n: {"slow_good": 100.0, "fast_ok": 1.0}[n],
    )
    assert result.chosen == "slow_good"
    assert not result.tie_broken_on_performance


def test_performance_breaks_only_a_measured_tie():
    """Two superior candidates, indistinguishable from each other."""
    def pairwise(a, b):
        return _cmp(f"{a}-{b}", 0.001, -0.01, 0.012)  # does not resolve

    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={
            "big": _cmp("big", -0.05, -0.07, -0.03),
            "small": _cmp("small", -0.048, -0.068, -0.028),
        },
        margin=ASR,
        pairwise=pairwise,
        performance_rank=lambda n: {"big": 800.0, "small": 40.0}[n],
    )
    assert result.chosen == "small"
    assert result.tie_broken_on_performance
    assert set(result.quality_tier) == {"big", "small"}


def test_without_pairwise_evidence_no_tie_is_declared():
    """Declaring an unmeasured tie is how a worse model wins on speed."""
    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={
            "big": _cmp("big", -0.05, -0.07, -0.03),
            "small": _cmp("small", -0.030, -0.050, -0.010),
        },
        margin=ASR,
        performance_rank=lambda n: {"big": 800.0, "small": 40.0}[n],
    )
    assert result.chosen == "big"          # best point estimate
    assert not result.tie_broken_on_performance


def test_a_resolved_pairwise_difference_keeps_the_better_candidate():
    def pairwise(a, b):
        return _cmp(f"{a}-{b}", -0.02, -0.03, -0.01)  # resolves: a is better

    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={
            "big": _cmp("big", -0.05, -0.07, -0.03),
            "small": _cmp("small", -0.030, -0.050, -0.010),
        },
        margin=ASR,
        pairwise=pairwise,
        performance_rank=lambda n: {"big": 800.0, "small": 40.0}[n],
    )
    assert result.chosen == "big"


# --- incumbent handling ------------------------------------------------------


def test_incumbent_retained_when_nothing_is_superior_or_non_inferior():
    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={"worse": _cmp("worse", 0.09, 0.07, 0.11)},
        margin=ASR,
    )
    assert result.chosen == "incumbent"
    assert "ceiling" in result.reason


def test_a_non_inferior_smaller_candidate_may_displace_the_incumbent():
    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={"smaller": _cmp("smaller", 0.008, 0.003, 0.013)},
        margin=ASR,
        performance_rank=lambda n: {"incumbent": 800.0, "smaller": 40.0}[n],
    )
    assert result.chosen == "smaller"
    assert result.tie_broken_on_performance


def test_unresolved_escalates_and_selects_nothing():
    result = select(
        slot="asr",
        incumbent="incumbent",
        comparisons={"maybe": _cmp("maybe", 0.015, -0.01, 0.05)},
        margin=ASR,
    )
    assert result.escalate
    assert result.chosen == "incumbent"
    assert "escalate" in result.reason


# --- execution failure is not a quality verdict ------------------------------


def test_a_candidate_that_cannot_execute_gets_no_quality_verdict():
    """An execution failure is not a measurement.

    A model that OOMs, has no compatible processor, or crashes on the corpus
    has told us nothing about its translation quality. Recording it as
    `inferior` would convert a deployment fact into a quality fact and let it
    be cited later as "we measured it and it was worse". We did not measure it.
    """
    sel = select(
        slot="mt",
        incumbent="nllb",
        comparisons={"envit5": _cmp("envit5", -0.03, -0.05, -0.01)},
        margin=MT_EN,
        errored={"madlad": "OOM at 3B on 16 GB host"},
    )
    assert "madlad" not in sel.verdicts
    assert sel.errored == {"madlad": "OOM at 3B on 16 GB host"}
    assert "madlad" not in sel.quality_tier


def test_a_candidate_cannot_be_both_scored_and_errored():
    """The two are mutually exclusive; overlap means the caller lost track."""
    with pytest.raises(ValueError, match="both scored and errored"):
        select(
            slot="mt",
            incumbent="nllb",
            comparisons={"madlad": _cmp("madlad", 0.01, -0.01, 0.03)},
            margin=MT_EN,
            errored={"madlad": "OOM"},
        )

# --- D66: unresolved is decision-relevant or decision-irrelevant -------------
#
# The formal definition: an unresolved candidate is decision-RELEVANT iff there
# exists at least one outcome inside its own uncertainty interval under which
# the already-fixed hierarchy would select it. Otherwise decision-IRRELEVANT.
#
# Both remain UNRESOLVED. Neither is a claim about quality.


def test_unresolved_that_can_still_reach_a_superior_outcome_is_relevant():
    """Interval reaches past the incumbent and past the best rival's point
    estimate, so an admissible outcome exists in which it is selected."""
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "wide": _cmp("wide", -0.012, -0.045, +0.025),
            "clear": _cmp("clear", -0.030, -0.041, -0.018),
        },
        margin=ASR,
    )
    assert sel.verdicts["wide"] == UNRESOLVED_RELEVANT
    assert sel.escalate
    assert sel.chosen == "inc"


def test_unresolved_that_can_at_most_become_non_inferior_is_irrelevant():
    """Its whole interval sits on the acceptable side of the incumbent but
    behind a demonstrated-superior rival, so no outcome selects it.

    Non-inferiority never displaces a superior candidate, so "could become
    non_inferior" is not "could be selected".
    """
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "close": _cmp("close", +0.008, -0.002, +0.021),
            "clear": _cmp("clear", -0.030, -0.041, -0.018),
        },
        margin=ASR,
    )
    assert sel.verdicts["close"] == UNRESOLVED_IRRELEVANT
    assert not sel.escalate
    assert sel.chosen == "clear"


def test_unresolved_behind_the_incumbent_that_cannot_reach_the_boundary():
    """Every admissible value leaves it behind the incumbent, so no outcome
    selects it while another candidate is demonstrated superior."""
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "behind": _cmp("behind", +0.016, +0.003, +0.030),
            "clear": _cmp("clear", -0.030, -0.041, -0.018),
        },
        margin=ASR,
    )
    assert sel.verdicts["behind"] == UNRESOLVED_IRRELEVANT
    assert not sel.escalate
    assert sel.chosen == "clear"


def test_unresolved_whose_interval_reaches_the_winning_boundary_is_relevant():
    """The boundary case that must not be rounded away: the interval only just
    extends past the best rival's point estimate, and that is enough."""
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "reaches": _cmp("reaches", -0.014, -0.035, +0.020),
            "clear": _cmp("clear", -0.030, -0.041, -0.018),
        },
        margin=ASR,
    )
    assert sel.verdicts["reaches"] == UNRESOLVED_RELEVANT
    assert sel.escalate


def test_lone_unresolved_that_could_prove_superior_still_blocks():
    """Nothing is demonstrated superior and this candidate's interval reaches
    past the incumbent, so an outcome selecting it exists."""
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={"only": _cmp("only", -0.004, -0.026, +0.019)},
        margin=ASR,
    )
    assert sel.verdicts["only"] == UNRESOLVED_RELEVANT
    assert sel.escalate
    assert sel.chosen == "inc"


def test_lone_unresolved_that_can_at_most_tie_the_incumbent_is_irrelevant():
    """A consequence of the formal definition, pinned because it CHANGES
    behaviour: with no performance tie-break supplied, a `non_inferior`
    challenger never displaces the incumbent. So a lone unresolved candidate
    whose interval cannot reach superiority can never be selected, and
    escalating on its account would buy nothing — the decision is already
    determined. It stays unresolved; the incumbent is retained, not preferred
    over it on quality.
    """
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={"only": _cmp("only", +0.016, +0.003, +0.030)},
        margin=ASR,
    )
    assert sel.verdicts["only"] == UNRESOLVED_IRRELEVANT
    assert not sel.escalate
    assert sel.chosen == "inc"


def test_a_lone_unresolved_candidate_blocks_once_a_tie_break_could_pick_it():
    """The same candidate becomes relevant when a performance rank exists that
    could select it, which is what makes the calculation a property of the
    hierarchy rather than a hand-written condition."""
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={"only": _cmp("only", +0.016, +0.003, +0.030)},
        margin=ASR,
        performance_rank=lambda n: 0.0 if n == "only" else 99.0,
    )
    assert sel.verdicts["only"] == UNRESOLVED_RELEVANT
    assert sel.escalate


def test_decision_relevance_respects_metric_orientation():
    """On a higher-is-better metric "worse" is a *lower* score. Getting this
    backwards would discard a genuine contender, or keep a hopeless one."""
    sel = select(
        slot="mt",
        incumbent="inc",
        comparisons={
            "behind": _cmp("behind", -0.016, -0.030, -0.003),
            "clear": _cmp("clear", +0.030, +0.018, +0.041),
        },
        margin=MT_EN,
    )
    assert sel.verdicts["behind"] == UNRESOLVED_IRRELEVANT
    assert sel.chosen == "clear"


def test_unresolved_is_never_converted_to_a_resolved_verdict():
    """"Unresolved" means insufficient evidence for a classification. Both
    decision statuses stay unresolved; neither is a quality claim."""
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "behind": _cmp("behind", +0.016, +0.003, +0.030),
            "clear": _cmp("clear", -0.030, -0.041, -0.018),
        },
        margin=ASR,
    )
    assert sel.verdicts["behind"].startswith("unresolved")
    assert sel.verdicts["behind"] not in ("inferior", "non_inferior", "superior")
    assert sel.verdicts["behind"] in VERDICTS
    assert "behind" not in sel.quality_tier


def test_several_superior_candidates_do_not_trigger_a_pairwise_comparison():
    """Quality-first, unchanged. Multiple candidates independently superior to
    the incumbent are ranked by their pre-registered point estimates; no
    pairwise test is introduced merely because there is more than one, and no
    tie is declared without pairwise evidence.
    """
    sel = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "good": _cmp("good", -0.020, -0.031, -0.009),
            "best": _cmp("best", -0.030, -0.041, -0.018),
        },
        margin=ASR,
    )
    assert sel.verdicts == {"good": "superior", "best": "superior"}
    assert sel.chosen == "best"
    assert sel.quality_tier == ("best",)
    assert not sel.tie_broken_on_performance


def test_performance_cannot_resolve_or_rescue_quality_uncertainty():
    """Efficiency evidence neither settles an unresolved comparison nor
    re-admits a candidate excluded on quality."""
    blocked = select(
        slot="asr",
        incumbent="inc",
        comparisons={"only": _cmp("only", +0.016, +0.003, +0.030)},
        margin=ASR,
        performance_rank=lambda n: 0.0 if n == "only" else 99.0,
    )
    assert blocked.escalate and blocked.chosen == "inc"

    excluded = select(
        slot="asr",
        incumbent="inc",
        comparisons={
            "behind_but_tiny": _cmp("behind_but_tiny", +0.016, +0.003, +0.030),
            "clear": _cmp("clear", -0.030, -0.041, -0.018),
        },
        margin=ASR,
        performance_rank=lambda n: 0.0 if n == "behind_but_tiny" else 99.0,
    )
    assert excluded.chosen == "clear"


def test_the_rule_names_no_candidate_metric_or_observed_result():
    """The correction must be generic. Scoped to identifiers and literals: a
    candidate or model name anywhere in the module, or any observed value or
    delta from the MT track, would mean the rule had been fitted to its data.
    """
    from pathlib import Path

    import onevoice.selection as module

    source = Path(module.__file__).read_text().lower()
    for name in (
        "envit5", "madlad", "nllb", "opus", "m2m100", "marian",
        "seamless", "phowhisper", "whisper", "silero", "vits",
    ):
        assert name not in source, f"candidate name {name!r} leaked into the rule"
    for literal in ("0.0176", "2.351", "0.0263", "0.01755", "1.5675", "0.8029"):
        assert literal not in source, f"observed value {literal!r} leaked in"
