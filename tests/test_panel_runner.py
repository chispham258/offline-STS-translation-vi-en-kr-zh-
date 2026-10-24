"""MOS analysis runner (protocol §7, §10, §15).

The ordering §10 requires - incumbent-only resolution committed before any
candidate mean is examined - is enforced structurally: stage 1 strips every
challenger rating before computing anything, and stage 2 refuses to run on a
panel whose results differ from the ones stage 1 recorded.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict

import pytest

from onevoice import panel
from onevoice import panel_analysis as pa

INC, MMS, F5 = pa.INCUMBENT, pa.CHALLENGERS[0], pa.CHALLENGERS[1]
IDS = [str(3000 + i) for i in range(20)]  # 4 blocks of 5
FULL = 4 * pa.MIN_VALID_RATERS_PER_BLOCK  # 32 sessions: exactly 8 per block


def _key(roles=("upper_anchor", "lower_anchor", INC, MMS, F5), raters=FULL):
    records = [
        {
            "role": role,
            "utt_id": u,
            "output_path": f"{role}/{u}.wav",
            "output_sha256": f"{role}{u}",
            "duration_s": 4.0,
        }
        for role in roles
        for u in IDS
    ]
    clips = panel.build_clips(records, IDS)
    plans = {
        str(n): {"session_id": f"s{n}", **asdict(panel.rater_plan(clips, IDS, n))}
        for n in range(1, raters + 1)
    }
    return {
        "pilot": False,
        "blocks": panel.blocks(IDS),
        "clips": [asdict(c) for c in clips],
        "raters": plans,
    }


def _results(key, score, raters=None, hp_ok=6):
    clips = {c["clip_id"]: c for c in key["clips"]}
    pool = panel.huggins_pool()
    rng = random.Random(3)
    out = []
    for n, plan in key["raters"].items():
        if raters is not None and int(n) not in raters:
            continue
        clips = {**clips, plan["duplicate"]["clip_id"]: plan["duplicate"]}
        out.append(
            {
                "session_id": plan["session_id"],
                "rater": int(n),
                "headphone": [
                    {"trial": t, "choice": pool[t][1] if k < hp_ok else 9}
                    for k, t in enumerate(plan["headphone_trials"])
                ],
                "ratings": [
                    {
                        "index": i,
                        "clip_id": cid,
                        "rating": score(clips[cid], int(n), rng),
                        "replays": 0,
                        "trial_ms": 6000.0,
                    }
                    for i, cid in enumerate(plan["trial_order"])
                ],
            }
        )
    return out


def _score(inc=3.0, mms=3.0, f5=3.0, upper=5, lower=1):
    means = {INC: inc, MMS: mms, F5: f5}

    def score(c, rater, rng):
        if c["role"] == "upper_anchor":
            return upper
        if c["role"] == "lower_anchor":
            return lower
        # Noise fixed per (rater, clip content), so a duplicate repeats its
        # original's rating and rule 3 never depends on the test's randomness.
        jitter = random.Random(f"{rater}|{c['role']}|{c['utt_id']}")
        base = means[c["role"]] + jitter.choice((-1, 0, 1))
        return int(min(5, max(1, round(base))))

    return score


# --- stage 1 ------------------------------------------------------------------


def test_stage1_scores_nothing_from_challenger_ratings():
    """Challenger ratings reach stage 1 only through §8's per-rater pass/fail
    (rule 3 spans every system's duplicates); no score is computed from them."""
    key = _key()
    a = pa.resolution_stage(key, _results(key, _score(mms=1, f5=1)))
    b = pa.resolution_stage(key, _results(key, _score(mms=5, f5=5)))
    # Identical except for the digest of the (different) results files.
    a.pop("results_digest"), b.pop("results_digest")
    assert a == b
    assert "mms" not in json.dumps(a).lower() and "f5" not in json.dumps(a).lower()


def test_stage1_records_resolution_and_delta_from_the_incumbent():
    key = _key()
    rec = pa.resolution_stage(key, _results(key, _score()))
    assert rec["status"] == "resolved"
    assert rec["valid_raters"] == list(range(1, FULL + 1))
    assert rec["valid_per_block"] == {str(g): 8 for g in range(4)}
    assert rec["delta"] == pytest.approx(pa.K * rec["resolution"])
    assert rec["resolution"] > 0
    assert rec["anchor_separation"] == pytest.approx(4.0)


def test_panel_failing_the_anchor_gate_is_invalid_and_yields_no_resolution():
    key = _key()
    # Raters pass every §8 rule (upper 4 >= 3.5, lower 3 <= 3.0) but the
    # separation 1.0 is under the 1.5 gate.
    rec = pa.resolution_stage(key, _results(key, _score(upper=4, lower=3)))
    assert rec["status"] == "invalid_panel"
    assert "resolution" not in rec and "delta" not in rec


def test_too_few_valid_raters_yields_no_resolution():
    key = _key()
    # Drop one session from block 3 only: 31 valid overall, but 7 in block 3.
    results = _results(key, _score(), raters=range(1, FULL))
    rec = pa.resolution_stage(key, results)
    assert rec["status"] == "insufficient_raters"
    assert rec["valid_per_block"]["3"] == 7
    assert "resolution" not in rec


def test_excluded_raters_are_reported_with_reasons():
    key = _key(raters=FULL + 1)  # a ninth session in block 0
    results = _results(key, _score())
    results[0]["headphone"][0]["choice"] = 9
    rec = pa.resolution_stage(key, results)
    assert rec["excluded"] == {"1": ["headphone"]}
    assert rec["valid_raters"] == list(range(2, FULL + 2))
    assert rec["status"] == "resolved"


def test_a_pilot_key_is_refused_unless_pilot_is_asked_for():
    key = {**_key(), "pilot": True}
    with pytest.raises(ValueError, match="pilot"):
        pa.resolution_stage(key, _results(key, _score()))
    assert pa.resolution_stage(key, _results(key, _score()), pilot=True)["pilot"]


# --- stage 2 ------------------------------------------------------------------


def test_stage2_refuses_results_other_than_those_stage1_recorded():
    key = _key(raters=FULL + 1)
    results = _results(key, _score())
    rec = pa.resolution_stage(key, results[:FULL])
    with pytest.raises(ValueError, match="results"):
        pa.comparison_stage(key, results, rec)


def test_stage2_refuses_a_panel_stage1_did_not_resolve():
    key = _key()
    results = _results(key, _score(upper=4, lower=3))
    rec = pa.resolution_stage(key, results)
    with pytest.raises(ValueError, match="invalid_panel"):
        pa.comparison_stage(key, results, rec)


def test_stage2_selects_through_the_frozen_hierarchy():
    key = _key()
    results = _results(key, _score(inc=2.0, mms=2.0, f5=4.5))
    rec = pa.resolution_stage(key, results)
    out = pa.comparison_stage(key, results, rec)
    assert out["gate_d"] == "pass"
    assert out["verdicts"][F5] == "superior"
    assert out["chosen"] == F5
    assert out["comparisons"][F5]["level"] == 0.975
    assert out["delta"] == rec["delta"]


def test_without_f5_gate_d_fails_and_nothing_is_selected():
    key = _key(roles=("upper_anchor", "lower_anchor", INC, MMS))
    results = _results(key, _score())
    rec = pa.resolution_stage(key, results)
    out = pa.comparison_stage(key, results, rec)
    assert out["gate_d"] == "fail"
    assert out["chosen"] is None
    assert out["errored"] == {F5: "no stimuli presented"}


def test_no_challenger_rating_reaches_any_stage1_score(monkeypatch):
    """Guards future edits: whatever stage 1 scores, it is handed results with
    every challenger rating already removed."""
    key = _key()
    challenger_ids = {
        c["clip_id"]
        for c in [*key["clips"], *(p["duplicate"] for p in key["raters"].values())]
        if c["role"] in pa.CHALLENGERS
    }
    real = pa.system_matrix

    def checked(results, clips, role):
        for res in results:
            assert not challenger_ids & {r["clip_id"] for r in res.ratings}
        return real(results, clips, role)

    monkeypatch.setattr(pa, "system_matrix", checked)
    assert pa.resolution_stage(key, _results(key, _score()))["status"] == "resolved"
