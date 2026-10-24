"""MOS panel analysis (TTS pre-registration §8, §9A, §10; D77 short sessions).

Order is fixed: validate each result against its rater's plan, exclude raters
(§8) before any system score exists, then aggregate and bootstrap (§9A). Reads
ratings only; imports no model, pipeline or evaluator code.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from onevoice.panel import HP_PASS, Clip, RaterPlan, huggins_pool
from onevoice.stats import Comparison

RESAMPLES = 1000
SEED = 0
RESOLUTION_LEVEL = 0.95
COMPARISON_LEVEL = 0.975  # Bonferroni over exactly 2 planned challengers

UPPER_ANCHOR_MIN = 3.5
LOWER_ANCHOR_MAX = 3.0
DUPLICATE_MAX_MEAN_ABS_DIFF = 1.0

Matrix = dict[int, dict[str, float]]  # rater -> utt_id -> rating


@dataclass(frozen=True)
class RaterResult:
    rater: int
    headphone_correct: int
    ratings: tuple[dict[str, Any], ...]
    complete: bool


def load_result(
    result: dict[str, Any], plan: RaterPlan, session_id: str
) -> RaterResult:
    """Validate a returned results JSON against the plan its page was built from.

    The headphone score is recomputed from the frozen pool; the page's own
    `headphone_passed` is not trusted.
    """
    if result.get("session_id") != session_id:
        raise ValueError(f"rater {plan.rater}: session id does not match the package")
    if result.get("rater") != plan.rater:
        raise ValueError(f"result names rater {result.get('rater')}, not {plan.rater}")
    ratings = tuple(result["ratings"])
    if [r["clip_id"] for r in ratings] != list(plan.trial_order[: len(ratings)]):
        raise ValueError(f"rater {plan.rater}: ratings do not follow the plan order")
    pool = huggins_pool()
    correct = sum(
        h["choice"] == pool[h["trial"]][1]
        for h, planned in zip(result["headphone"], plan.headphone_trials, strict=False)
        if h["trial"] == planned
    )
    return RaterResult(
        rater=plan.rater,
        headphone_correct=correct,
        ratings=ratings,
        complete=len(ratings) == len(plan.trial_order),
    )


def exclusion_reasons(result: RaterResult, clips: list[Clip]) -> list[str]:
    """§8, applied before any system score. An incomplete session is excluded
    too: the crossed design needs every valid rater to rate every clip."""
    if result.headphone_correct < HP_PASS:
        return ["headphone"]
    if not result.complete:
        return ["incomplete"]
    by_id = {c.clip_id: c for c in clips}
    rated = [(by_id[r["clip_id"]], r) for r in result.ratings]
    reasons = []

    def mean_of(role: str) -> float:
        return statistics.fmean(r["rating"] for c, r in rated if c.role == role)

    if mean_of("upper_anchor") < UPPER_ANCHOR_MIN or (
        mean_of("lower_anchor") > LOWER_ANCHOR_MAX
    ):
        reasons.append("anchors")

    original = {c.clip_id: r["rating"] for c, r in rated if c.duplicate_of is None}
    diffs = [
        abs(r["rating"] - original[c.duplicate_of]) for c, r in rated if c.duplicate_of
    ]
    if diffs and statistics.fmean(diffs) > DUPLICATE_MAX_MEAN_ABS_DIFF:
        reasons.append("duplicates")

    # Rule 4, read per trial: the median of (trial time - clip duration) is
    # negative. The page locks rating until playback ends, so this can only
    # fire on a result the page did not produce.
    lead = statistics.median(r["trial_ms"] / 1000 - c.duration_s for c, r in rated)
    if lead < 0:
        reasons.append("too_fast")

    if len({r["rating"] for _, r in rated}) == 1:
        reasons.append("zero_variance")
    return reasons


def system_matrix(results: list[RaterResult], clips: list[Clip], role: str) -> Matrix:
    """Ratings of one system by the given (already valid) raters: anchors and
    duplicates excluded (§7, §9 step 1)."""
    by_id = {c.clip_id: c for c in clips}
    matrix: Matrix = {}
    for res in results:
        row = {}
        for r in res.ratings:
            c = by_id[r["clip_id"]]
            if c.role == role and c.duplicate_of is None:
                row[c.utt_id] = r["rating"]
        matrix[res.rater] = row
    return matrix


BlockOf = dict[str, int]  # utt_id -> block


def _utts(matrix: Matrix) -> list[str]:
    return sorted({u for row in matrix.values() for u in row})


def _score(
    matrix: Matrix,
    rater_draws: dict[int, list[int]],
    utts: list[str],
    block_of: BlockOf,
) -> float:
    """§9A aggregation on a (re)sample: per utterance, the mean over the drawn
    raters of its block; then the mean over utterances."""
    return statistics.fmean(
        statistics.fmean(matrix[i][u] for i in rater_draws[block_of[u]]) for u in utts
    )


def mos(matrix: Matrix) -> float:
    """Per utterance, mean over the raters who rated it; then over utterances."""
    return statistics.fmean(
        statistics.fmean(row[u] for row in matrix.values() if u in row)
        for u in _utts(matrix)
    )


def _block_raters(matrix: Matrix, block_of: BlockOf) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for r in sorted(matrix):
        for g in sorted({block_of[u] for u in matrix[r]}):
            groups.setdefault(g, []).append(r)
    return dict(sorted(groups.items()))


def _bounds(values: list[float], level: float) -> tuple[float, float]:
    values = sorted(values)
    k = int(((1.0 - level) / 2.0) * len(values))
    return values[k], values[len(values) - 1 - k]


def _replicates(matrices: list[Matrix], block_of: BlockOf) -> list[list[float]]:
    """§9A: utterances first, then each block's raters in block order; the same
    draws scored for every matrix."""
    utts = _utts(matrices[0])
    groups = _block_raters(matrices[0], block_of)
    rng = random.Random(SEED)
    out = []
    for _ in range(RESAMPLES):
        us = [rng.choice(utts) for _ in utts]
        rs = {g: [rng.choice(raters) for _ in raters] for g, raters in groups.items()}
        out.append([_score(m, rs, us, block_of) for m in matrices])
    return out


def nested_interval(
    matrix: Matrix, block_of: BlockOf, level: float = RESOLUTION_LEVEL
) -> tuple[float, float]:
    """Block-nested bootstrap interval for one system (§9A)."""
    return _bounds([s for (s,) in _replicates([matrix], block_of)], level)


def resolution(incumbent: Matrix, block_of: BlockOf) -> float:
    """§10: half-width of the incumbent's own 95% interval."""
    low, high = nested_interval(incumbent, block_of, RESOLUTION_LEVEL)
    return (high - low) / 2


def paired_interval(
    incumbent: Matrix,
    candidate: Matrix,
    block_of: BlockOf,
    level: float = COMPARISON_LEVEL,
) -> Comparison:
    """Paired block-nested bootstrap (§9A): the same draws for both systems,
    differenced inside the replicate."""
    if sorted(incumbent) != sorted(candidate) or _utts(incumbent) != _utts(candidate):
        raise ValueError("paired systems need the same raters and utterances")
    diffs = [c - i for i, c in _replicates([incumbent, candidate], block_of)]
    low, high = _bounds(diffs, level)
    return Comparison(
        name="mos",
        delta=mos(candidate) - mos(incumbent),
        low=low,
        high=high,
        p_greater=sum(d > 0 for d in diffs) / RESAMPLES,
        n=len(_utts(incumbent)),
        resamples=RESAMPLES,
        level=level,
    )


# --- the two-stage runner (§7 gate, §10 ordering, §15 selection) -------------

INCUMBENT = "system:piper-vi-vais1000"
CHALLENGERS = ("system:mms-tts-vie", "system:f5")
# Gate D for the Vietnamese target is {vits, f5}, carried solely by F5 (§4).
GATE_D_CARRIER = "system:f5"
K = 1.5
MIN_VALID_RATERS_PER_BLOCK = 8
ANCHOR_GATE = 1.5


def load_key(
    key: dict[str, Any],
) -> tuple[list[Clip], dict[int, tuple[RaterPlan, str]], BlockOf]:
    """Clips (session repeats included), plans and utterance blocks."""
    clips = [Clip(**c) for c in key["clips"]]
    plans = {}
    for n, p in key["raters"].items():
        plan = RaterPlan(
            rater=p["rater"],
            seed=p["seed"],
            block=p["block"],
            headphone_trials=tuple(p["headphone_trials"]),
            duplicate=Clip(**p["duplicate"]),
            trial_order=tuple(p["trial_order"]),
        )
        plans[int(n)] = (plan, p["session_id"])
    clips += [plan.duplicate for plan, _ in plans.values()]
    block_of = {u: g for g, block in enumerate(key["blocks"]) for u in block}
    return clips, plans, block_of


def results_digest(results: list[dict[str, Any]]) -> str:
    """Identity of the exact set of returned results, independent of order."""
    parts = sorted(
        hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
        for r in results
    )
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _screen(
    key: dict[str, Any], results: list[dict[str, Any]], pilot: bool
) -> tuple[list[Clip], list[RaterResult], dict[str, list[str]], BlockOf]:
    """Validate every result, then apply §8 before any score exists."""
    if key.get("pilot") and not pilot:
        raise ValueError("key was built for a pilot; pass pilot=True to analyse it")
    clips, plans, block_of = load_key(key)
    loaded = []
    for r in results:
        if r.get("rater") not in plans:
            raise ValueError(f"result for unknown rater {r.get('rater')!r}")
        plan, session_id = plans[r["rater"]]
        loaded.append(load_result(r, plan, session_id))
    if len({r.rater for r in loaded}) != len(loaded):
        raise ValueError("more than one result for the same rater")
    excluded, valid = {}, []
    for res in sorted(loaded, key=lambda x: x.rater):
        reasons = exclusion_reasons(res, clips)
        if reasons:
            excluded[str(res.rater)] = reasons
        else:
            valid.append(res)
    return clips, valid, excluded, block_of


def _without(results: list[RaterResult], clips: list[Clip], roles: tuple[str, ...]):
    by_id = {c.clip_id: c for c in clips}
    return [
        RaterResult(
            r.rater,
            r.headphone_correct,
            tuple(x for x in r.ratings if by_id[x["clip_id"]].role not in roles),
            r.complete,
        )
        for r in results
    ]


def resolution_stage(
    key: dict[str, Any], results: list[dict[str, Any]], *, pilot: bool = False
) -> dict[str, Any]:
    """Stage 1: exclusions, anchor gate, incumbent-only resolution and delta.

    Challenger ratings are removed after §8 and before any score, so nothing
    computed here can depend on a candidate mean.
    """
    clips, valid, excluded, block_of = _screen(key, results, pilot)
    valid = _without(valid, clips, CHALLENGERS)
    record: dict[str, Any] = {
        "stage": "resolution",
        "pilot": bool(key.get("pilot")),
        "results_digest": results_digest(results),
        "returned": len(results),
        "excluded": excluded,
        "valid_raters": [r.rater for r in valid],
    }
    blocks = sorted(set(block_of.values()))
    per_block = {
        str(g): sum(1 for r in valid if plans_block(key, r.rater) == g) for g in blocks
    }
    record["valid_per_block"] = per_block
    if min(per_block.values()) < MIN_VALID_RATERS_PER_BLOCK:
        return {**record, "status": "insufficient_raters"}

    upper = mos(system_matrix(valid, clips, "upper_anchor"))
    lower = mos(system_matrix(valid, clips, "lower_anchor"))
    record |= {
        "anchor_upper_mos": upper,
        "anchor_lower_mos": lower,
        "anchor_separation": upper - lower,
    }
    if upper - lower < ANCHOR_GATE:
        return {**record, "status": "invalid_panel"}

    incumbent = system_matrix(valid, clips, INCUMBENT)
    low, high = nested_interval(incumbent, block_of, RESOLUTION_LEVEL)
    res = (high - low) / 2
    return {
        **record,
        "status": "resolved",
        "incumbent": INCUMBENT,
        "incumbent_mos": mos(incumbent),
        "incumbent_interval_95": [low, high],
        "resolution": res,
        "k": K,
        "delta": K * res,
    }


def plans_block(key: dict[str, Any], rater: int) -> int:
    return key["raters"][str(rater)]["block"]


def _common(a: Matrix, b: Matrix) -> tuple[Matrix, Matrix]:
    """Restrict two matrices to the utterances both systems were presented
    for; an errored cell removes that utterance from the pair only (§8)."""
    utts = set(_utts(a)) & set(_utts(b))
    return (
        {r: {u: v for u, v in row.items() if u in utts} for r, row in a.items()},
        {r: {u: v for u, v in row.items() if u in utts} for r, row in b.items()},
    )


def comparison_stage(
    key: dict[str, Any],
    results: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    pilot: bool = False,
) -> dict[str, Any]:
    """Stage 2: paired comparisons and the frozen hierarchy, only on the exact
    results stage 1 resolved."""
    from onevoice.selection import Margin, select

    if record.get("status") != "resolved":
        raise ValueError(f"stage 1 did not resolve: {record.get('status')}")
    if record["results_digest"] != results_digest(results):
        raise ValueError("results differ from those stage 1 recorded")
    clips, valid, _, block_of = _screen(key, results, pilot)
    if [r.rater for r in valid] != record["valid_raters"]:
        raise ValueError("valid raters differ from those stage 1 recorded")

    presented = {c.role for c in clips}
    errored = {c: "no stimuli presented" for c in CHALLENGERS if c not in presented}
    incumbent = system_matrix(valid, clips, INCUMBENT)
    comparisons = {}
    for cand in CHALLENGERS:
        if cand in errored:
            continue
        inc, cm = _common(incumbent, system_matrix(valid, clips, cand))
        comparisons[cand] = paired_interval(inc, cm, block_of, COMPARISON_LEVEL)

    out: dict[str, Any] = {
        "stage": "comparison",
        "pilot": record["pilot"],
        "results_digest": record["results_digest"],
        "delta": record["delta"],
        "levels": {
            role: mos(system_matrix(valid, clips, role))
            for role in (INCUMBENT, *comparisons)
        },
        "comparisons": {n: asdict(c) for n, c in comparisons.items()},
        "errored": errored,
    }
    if GATE_D_CARRIER not in comparisons:
        return {
            **out,
            "gate_d": "fail",
            "chosen": None,
            "verdicts": {},
            "reason": "Gate D fails over executed candidates: scoped/unresolved, "
            "not a slot selection and not an incumbent win",
        }
    margin = Margin(
        slot="tts",
        metric="mos",
        k=K,
        resolution=record["resolution"],
        lower_is_better=False,
    )
    sel = select(
        slot="tts",
        incumbent=INCUMBENT,
        comparisons=comparisons,
        margin=margin,
        errored=errored,
    )
    return {
        **out,
        "gate_d": "pass",
        "chosen": sel.chosen,
        "verdicts": sel.verdicts,
        "reason": sel.reason,
        "escalate": sel.escalate,
    }
