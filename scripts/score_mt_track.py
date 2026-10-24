"""Score the MT track against the pre-registered δ (2026-08-15).

Reads archived `hypotheses.jsonl` files only. Nothing here re-runs a model, so
the scoring can be repeated without touching the runs — the same separation the
evaluator keeps (D5).

δ is read from the constants below, which were approved before any candidate
ran. They are not computed here and must not be.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from onevoice.eval.metrics.comet import DEFAULT_MODEL, _load
from onevoice.eval.metrics.text import bleu, chrf
from onevoice.selection import Margin, select
from onevoice.stats import paired_bootstrap

#: Approved 2026-08-15, fixed for the whole track. Resolution measured on the
#: incumbent before any candidate ran; k=1.5. Never recomputed from candidates.
RESOLUTION = {"en": 0.0117, "zh": 1.5675, "ko": 0.0176}
PRIMARY = {"en": "comet", "zh": "chrf", "ko": "comet"}
K = 1.5

INCUMBENT = "nllb"
CHALLENGERS = {
    "en": ["opus", "envit5", "m2m100", "madlad"],
    "zh": ["m2m100", "madlad"],
    "ko": ["m2m100", "madlad"],
}
#: Gate D depends on these families being present among candidates that ran.
FAMILY = {
    "nllb": "m2m_100",
    "m2m100": "m2m_100",
    "opus": "marian",
    "envit5": "t5",
    "madlad": "t5",
}


def _rows(name: str, tgt: str) -> dict[str, dict] | None:
    d = "mt_incumbent" if name == INCUMBENT else f"mt_{name}"
    path = Path(f"outputs/{d}_vi_{tgt}/hypotheses.jsonl")
    if not path.exists():
        return None
    return {
        r["utt_id"]: r
        for r in (json.loads(line) for line in path.read_text().splitlines())
    }


def _error_reason(name: str, tgt: str) -> str:
    path = Path(f"outputs/mt_{name}_vi_{tgt}.error.json")
    if path.exists():
        return json.loads(path.read_text())["error"]
    return "run absent: no hypotheses and no recorded error"


def score_direction(model, tgt: str) -> dict[str, object]:
    """One direction, end to end. A function rather than a loop body so the
    scorer closures bind parameters instead of loop variables."""
    inc = _rows(INCUMBENT, tgt)
    assert inc is not None, f"incumbent missing for vi->{tgt}"

    scored, errored = {}, {}
    for name in CHALLENGERS[tgt]:
        got = _rows(name, tgt)
        if got is None:
            errored[name] = _error_reason(name, tgt)
        else:
            scored[name] = got

    # Gate D is re-checked over candidates that actually EXECUTED. A family
    # present only via a candidate that failed to run is not diversity.
    families = {FAMILY[n] for n in [INCUMBENT, *scored]}
    outside = families - {FAMILY[INCUMBENT]}
    gate_d = bool(outside)

    ids = sorted(inc)
    cache: dict[str, dict[str, float]] = {}

    def comet_seg(name: str, rows: dict[str, dict]) -> dict[str, float]:
        if name not in cache:
            order = [rows[i] for i in ids]
            out = model.predict(
                [
                    {
                        "src": r["src_text"],
                        "mt": r["hyp_text"],
                        "ref": r["ref_text"],
                    }
                    for r in order
                ],
                progress_bar=False,
            )
            cache[name] = dict(zip(ids, (float(x) for x in out.scores), strict=True))
        return cache[name]

    def scorer(name: str, rows: dict[str, dict]):
        metric = PRIMARY[tgt]
        if metric == "comet":
            seg = comet_seg(name, rows)
            return lambda s: statistics.fmean(seg[i] for i in s)
        return lambda s: chrf(
            [rows[i]["hyp_text"] for i in s], [rows[i]["ref_text"] for i in s]
        ).value

    # Bonferroni within direction, over challengers actually compared.
    n_comp = max(len(scored), 1)
    level = 1.0 - 0.05 / n_comp

    margin = Margin(
        slot="mt",
        metric=PRIMARY[tgt],
        k=K,
        resolution=RESOLUTION[tgt],
        lower_is_better=False,
    )

    comparisons = {}
    for name, rows in scored.items():
        comparisons[name] = paired_bootstrap(
            ids,
            scorer(INCUMBENT, inc),
            scorer(name, rows),
            name=f"{name} vs {INCUMBENT} (vi->{tgt}, {PRIMARY[tgt]})",
            level=level,
        )

    secondary = {}
    for name, rows in {INCUMBENT: inc, **scored}.items():
        hyp = [rows[i]["hyp_text"] for i in ids]
        ref = [rows[i]["ref_text"] for i in ids]
        secondary[name] = {
            "bleu": bleu(hyp, ref, tgt).value,
            "chrf": chrf(hyp, ref).value,
            "comet": statistics.fmean(comet_seg(name, rows).values()),
        }

    sel = None
    if gate_d and comparisons:
        sel = select(
            slot="mt",
            incumbent=INCUMBENT,
            comparisons=comparisons,
            margin=margin,
            errored=errored,
        )

    result = {
        "primary": PRIMARY[tgt],
        "delta": margin.delta,
        "level": level,
        "gate_d": gate_d,
        "families_executed": sorted(families),
        "errored": errored,
        "comparisons": {
            n: {
                "delta": c.delta,
                "low": c.low,
                "high": c.high,
                "p_greater": c.p_greater,
                "paired_n": c.n,
            }
            for n, c in comparisons.items()
        },
        "levels": secondary,
        "selection": None
        if sel is None
        else {
            "chosen": sel.chosen,
            "reason": sel.reason,
            "verdicts": sel.verdicts,
            "quality_tier": list(sel.quality_tier),
            "escalate": sel.escalate,
            "errored": sel.errored,
        },
    }

    print(f"\n=== vi->{tgt}  primary={PRIMARY[tgt]}  delta={margin.delta:.4g} ===")
    print(f"Gate D over EXECUTED candidates: {'PASS' if gate_d else 'FAIL'} "
          f"{sorted(families)}")
    if errored:
        print(f"errored (no quality verdict): {errored}")
    for n, c in comparisons.items():
        print(f"  {n:<8} d={c.delta:+.4f} CI[{c.low:+.4f},{c.high:+.4f}] "
              f"P={c.p_greater:.3f} n={c.n}")
    if sel:
        print(f"  -> {sel.chosen}: {sel.reason}")
    else:
        print("  -> NOT A SELECTION (Gate D failed on executed candidates)")
    return result


def main() -> None:
    model = _load(DEFAULT_MODEL)
    report = {f"vi->{t}": score_direction(model, t) for t in ("en", "zh", "ko")}
    Path("outputs/mt_track_result.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()