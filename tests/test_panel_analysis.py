"""MOS panel analysis (protocol §8, §9A, §10; D77 short sessions).

Exclusions are applied before any system score exists; aggregation and the
block-nested bootstrap are checked against the frozen §9A pseudo-code written
out literally, draw order included.
"""

from __future__ import annotations

import random

import pytest

from onevoice import panel
from onevoice import panel_analysis as pa

SYSTEMS = ("system:inc", "system:cand")
IDS = [str(2000 + i) for i in range(20)]  # 4 blocks of 5
BLOCK_OF = {u: g for g, b in enumerate(panel.blocks(IDS)) for u in b}


def _clips():
    records = [
        {
            "role": role,
            "utt_id": u,
            "output_path": f"{role}/{u}.wav",
            "output_sha256": f"{role}{u}",
            "duration_s": 4.0,
        }
        for role in ("upper_anchor", "lower_anchor", *SYSTEMS)
        for u in IDS
    ]
    return panel.build_clips(records, IDS)


def _plan(clips, rater):
    return panel.rater_plan(clips, IDS, rater)


def _result(clips, rater, score, *, hp_ok=6, trial_s=6.0, n=None):
    """A rater result where `score(clip)` gives each rating."""
    plan = _plan(clips, rater)
    pool = panel.huggins_pool()
    by_id = {c.clip_id: c for c in [*clips, plan.duplicate]}
    order = plan.trial_order[: n if n is not None else len(plan.trial_order)]
    return {
        "session_id": f"s{rater}",
        "rater": rater,
        "headphone": [
            {"trial": t, "choice": pool[t][1] if k < hp_ok else (pool[t][1] + 1) % 3}
            for k, t in enumerate(plan.headphone_trials)
        ],
        "ratings": [
            {
                "index": i,
                "clip_id": cid,
                "rating": score(by_id[cid]),
                "replays": 0,
                "trial_ms": trial_s * 1000,
            }
            for i, cid in enumerate(order)
        ],
    }


def _good(c):
    return {"upper_anchor": 5, "lower_anchor": 1}.get(c.role, 3 + int(c.utt_id) % 2)


def _load(clips, result):
    return pa.load_result(result, _plan(clips, result["rater"]), result["session_id"])


def _reasons(clips, result):
    plan = _plan(clips, result["rater"])
    return pa.exclusion_reasons(_load(clips, result), [*clips, plan.duplicate])


# --- ingest -------------------------------------------------------------------


def test_a_result_from_another_session_is_refused():
    clips = _clips()
    with pytest.raises(ValueError, match="session"):
        pa.load_result(_result(clips, 1, _good), _plan(clips, 1), "other")


def test_a_result_whose_clips_do_not_follow_the_plan_is_refused():
    clips = _clips()
    result = _result(clips, 1, _good)
    r = result["ratings"]
    r[0], r[1] = r[1], r[0]
    with pytest.raises(ValueError, match="plan"):
        _load(clips, result)


def test_headphone_score_is_recomputed_from_the_pool_not_trusted():
    clips = _clips()
    result = _result(clips, 1, _good, hp_ok=5)
    result["headphone_passed"] = True  # the page's own claim is ignored
    assert _load(clips, result).headphone_correct == 5


# --- exclusions (§8), each rule in isolation ---------------------------------


def test_a_good_rater_is_kept():
    clips = _clips()
    assert _reasons(clips, _result(clips, 1, _good)) == []


def test_rule1_one_missed_headphone_trial_excludes():
    clips = _clips()
    assert _reasons(clips, _result(clips, 1, _good, hp_ok=5)) == ["headphone"]


@pytest.mark.parametrize(
    ("upper", "lower", "reasons"),
    [
        ((4, 3), (1, 1), []),  # upper mean 3.5: boundary passes
        ((3, 3), (1, 1), ["anchors"]),  # 3.0 < 3.5
        ((5, 5), (3, 3), []),  # lower mean 3.0: boundary passes
        ((5, 5), (3, 4), ["anchors"]),  # 3.5 > 3.0
    ],
)
def test_rule2_anchor_means_over_two_plus_two_anchors(upper, lower, reasons):
    clips = _clips()
    plan = _plan(clips, 1)
    by_id = {c.clip_id: c for c in clips}
    presented = [by_id[c] for c in plan.trial_order if c in by_id]
    ups = sorted(c.utt_id for c in presented if c.role == "upper_anchor")
    lows = sorted(c.utt_id for c in presented if c.role == "lower_anchor")

    def score(c):
        if c.role == "upper_anchor":
            return upper[ups.index(c.utt_id)]
        if c.role == "lower_anchor":
            return lower[lows.index(c.utt_id)]
        return _good(c)

    assert _reasons(clips, _result(clips, 1, score)) == reasons


@pytest.mark.parametrize(("shift", "reasons"), [(1, []), (2, ["duplicates"])])
def test_rule3_the_one_repeat_may_differ_by_at_most_one(shift, reasons):
    clips = _clips()

    def score(c):
        base = _good(c)
        return max(1, base - shift) if c.duplicate_of else base

    assert _reasons(clips, _result(clips, 1, score)) == reasons


def test_rule4_answering_before_clips_end_excludes():
    clips = _clips()
    assert _reasons(clips, _result(clips, 1, _good, trial_s=3.9)) == ["too_fast"]
    assert _reasons(clips, _result(clips, 1, _good, trial_s=4.0)) == []


def test_rule5_zero_variance_excludes():
    clips = _clips()
    assert "zero_variance" in _reasons(clips, _result(clips, 1, lambda c: 3))


def test_an_incomplete_session_is_excluded():
    clips = _clips()
    assert _reasons(clips, _result(clips, 1, _good, n=10)) == ["incomplete"]


# --- matrix, aggregation, block-nested bootstrap -------------------------------


def _panel(per_block=3):
    clips = _clips()
    results = []
    for r in range(1, 4 * per_block + 1):

        def score(c, r=r):
            if not c.role.startswith("system"):
                return _good(c)
            jitter = random.Random(f"{r}|{c.role}|{c.utt_id}").choice((-1, 0, 1))
            bias = 1 if c.role == "system:cand" else 0
            return min(5, max(1, _good(c) + jitter + bias))

        results.append(_load(clips, _result(clips, r, score)))
    # As in the key: every session's repeat is a clip of its own.
    repeats = [_plan(clips, r.rater).duplicate for r in results]
    return [*clips, *repeats], results


def test_matrix_rows_hold_each_raters_own_block_only():
    clips, results = _panel()
    matrix = pa.system_matrix(results, clips, "system:inc")
    for res in results:
        block = _plan(clips, res.rater).block
        assert sorted(matrix[res.rater]) == sorted(panel.blocks(IDS)[block])


def test_mos_is_mean_over_its_raters_then_over_utterances():
    matrix = {1: {"a": 5, "b": 1}, 2: {"a": 3}, 3: {"b": 3}}
    assert pa.mos(matrix) == pytest.approx(((5 + 3) / 2 + (1 + 3) / 2) / 2)


def _reference(matrices, level, block_of):
    """§9A pseudo-code, literally, for one matrix or an (inc, cand) pair."""
    first = matrices[0]
    utts = sorted({u for row in first.values() for u in row})
    groups = sorted({block_of[u] for u in utts})
    raters_g = {
        g: sorted(r for r, row in first.items() if any(block_of[u] == g for u in row))
        for g in groups
    }
    rng = random.Random(0)
    values = []
    for _ in range(1000):
        us = [rng.choice(utts) for _ in utts]
        rs = {g: [rng.choice(raters_g[g]) for _ in raters_g[g]] for g in groups}

        def score(m, us=us, rs=rs):
            cells = [
                sum(m[i][u] for i in rs[block_of[u]]) / len(rs[block_of[u]]) for u in us
            ]
            return sum(cells) / len(cells)

        values.append(
            score(first) if len(matrices) == 1 else score(matrices[1]) - score(first)
        )
    values.sort()
    k = int((1 - level) / 2 * 1000)
    return values[k], values[999 - k]


def test_single_system_interval_matches_the_frozen_algorithm():
    clips, results = _panel()
    matrix = pa.system_matrix(results, clips, "system:inc")
    low, high = pa.nested_interval(matrix, BLOCK_OF)
    assert (low, high) == pytest.approx(_reference([matrix], 0.95, BLOCK_OF))
    assert pa.resolution(matrix, BLOCK_OF) == pytest.approx((high - low) / 2)


def test_paired_interval_matches_the_frozen_algorithm_at_the_comparison_level():
    clips, results = _panel()
    inc = pa.system_matrix(results, clips, "system:inc")
    cand = pa.system_matrix(results, clips, "system:cand")
    comparison = pa.paired_interval(inc, cand, BLOCK_OF)
    assert comparison.level == 0.975
    assert (comparison.low, comparison.high) == pytest.approx(
        _reference([inc, cand], 0.975, BLOCK_OF)
    )
    assert comparison.delta == pytest.approx(pa.mos(cand) - pa.mos(inc))
    assert comparison.n == 20


def test_a_system_paired_with_itself_has_a_zero_interval():
    clips, results = _panel()
    inc = pa.system_matrix(results, clips, "system:inc")
    comparison = pa.paired_interval(inc, inc, BLOCK_OF)
    assert comparison.low == comparison.high == 0.0


def test_paired_systems_must_share_raters():
    with pytest.raises(ValueError, match="same raters"):
        pa.paired_interval({1: {"a": 3}}, {2: {"a": 3}}, {"a": 0})
