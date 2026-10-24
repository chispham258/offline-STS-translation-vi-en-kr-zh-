"""Score the en→vi track against the committed pre-registration (d37e443).

Reads archived `hypotheses.jsonl` only and re-runs no translation model, so the
candidate archives cannot move.

Every constant below is READ from the pre-registration, not derived here.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from onevoice.eval.metrics.comet import DEFAULT_MODEL, _load
from onevoice.eval.metrics.text import bleu, chrf
from onevoice.selection import Margin, select
from onevoice.stats import paired_bootstrap

INCUMBENT = "nllb"
TGT = "vi"

#: Pre-registered and approved before any candidate ran. Never recomputed.
RESOLUTION = 0.0122
K = 1.5
PRIMARY = "comet"

#: Bonferroni was fixed in advance over the PLANNED family of 3 challengers.
#: `madlad` errored, so only 2 are compared -- but the level is deliberately NOT
#: recomputed to 1 - 0.05/2. Loosening a pre-registered correction because a
#: candidate failed to execute would make the test easier in response to an
#: outcome, and in the direction that favours declaring a winner.
LEVEL = 1.0 - 0.05 / 3

CHALLENGERS = ["m2m100", "envit5", "madlad"]


def _rows(name: str) -> dict[str, dict] | None:
    d = "mt_incumbent" if name == INCUMBENT else f"mt_{name}"
    path = Path(f"outputs/{d}_en_vi/hypotheses.jsonl")
    if not path.exists():
        return None
    return {
        r["utt_id"]: r
        for r in (json.loads(line) for line in path.read_text().splitlines())
    }


def _error_reason(name: str) -> str:
    path = Path(f"outputs/mt_{name}_en_vi.error.json")
    if path.exists():
        rec = json.loads(path.read_text())
        return f"{rec['error']} (elapsed {rec['evidence']['elapsed']})"
    return "run absent: no hypotheses and no recorded error"


def main() -> None:
    inc = _rows(INCUMBENT)
    assert inc is not None, "incumbent archive missing"
    ids = sorted(inc)

    scored, errored = {}, {}
    for name in CHALLENGERS:
        got = _rows(name)
        if got is None:
            errored[name] = _error_reason(name)
        else:
            scored[name] = got

    # Gate D over candidates that actually EXECUTED.
    family = {"nllb": "m2m_100", "m2m100": "m2m_100", "envit5": "t5",
              "madlad": "t5"}
    families = {family[n] for n in [INCUMBENT, *scored]}
    gate_d = bool(families - {family[INCUMBENT]})

    model = _load(DEFAULT_MODEL)
    seg: dict[str, dict[str, float]] = {}

    def comet_seg(name: str, rows: dict[str, dict]) -> dict[str, float]:
        if name not in seg:
            order = [rows[i] for i in ids]
            out = model.predict(
                [
                    {"src": r["src_text"], "mt": r["hyp_text"], "ref": r["ref_text"]}
                    for r in order
                ],
                progress_bar=False,
            )
            seg[name] = dict(zip(ids, (float(x) for x in out.scores), strict=True))
        return seg[name]

    def primary(name: str, rows: dict[str, dict]):
        s = comet_seg(name, rows)
        return lambda sample: statistics.fmean(s[i] for i in sample)

    margin = Margin(
        slot="mt", metric=PRIMARY, k=K, resolution=RESOLUTION, lower_is_better=False
    )
    assert abs(margin.delta - 0.0183) < 5e-5, "delta moved from the approved value"

    comparisons = {
        name: paired_bootstrap(
            ids,
            primary(INCUMBENT, inc),
            primary(name, rows),
            name=f"{name} vs {INCUMBENT} (en->vi, {PRIMARY})",
            level=LEVEL,
        )
        for name, rows in scored.items()
    }

    levels = {}
    for name, rows in {INCUMBENT: inc, **scored}.items():
        hyp = [rows[i]["hyp_text"] for i in ids]
        ref = [rows[i]["ref_text"] for i in ids]
        levels[name] = {
            "comet": statistics.fmean(comet_seg(name, rows).values()),
            "bleu": bleu(hyp, ref, TGT).value,
            "chrf": chrf(hyp, ref).value,
        }

    sel = select(
        slot="mt",
        incumbent=INCUMBENT,
        comparisons=comparisons,
        margin=margin,
        errored=errored,
    )

    print(f"=== en->vi  primary={PRIMARY}  delta={margin.delta:.4g}  "
          f"level={LEVEL:.5f} ===")
    print(f"Gate D over EXECUTED candidates: {'PASS' if gate_d else 'FAIL'} "
          f"{sorted(families)}")
    print(f"\n{'model':<10}{'COMET':>9}{'BLEU':>8}{'chrF':>8}   status")
    print(f"  {INCUMBENT:<8}{levels[INCUMBENT]['comet']:>9.4f}"
          f"{levels[INCUMBENT]['bleu']:>8.2f}{levels[INCUMBENT]['chrf']:>8.2f}"
          f"   (incumbent)")
    for name in CHALLENGERS:
        if name in errored:
            print(f"  {name:<8}{'--':>9}{'--':>8}{'--':>8}   ERRORED (no quality "
                  f"verdict): {errored[name]}")
            continue
        m = levels[name]
        print(f"  {name:<8}{m['comet']:>9.4f}{m['bleu']:>8.2f}{m['chrf']:>8.2f}"
              f"   {sel.verdicts[name]}")
    print()
    for name, c in comparisons.items():
        print(f"  {name:<8} d={c.delta:+.4f} CI[{c.low:+.4f},{c.high:+.4f}] "
              f"P={c.p_greater:.3f} paired_n={c.n}")
    print(f"\n-> {sel.chosen}: {sel.reason}")
    print(f"   quality_tier={list(sel.quality_tier)}  escalate={sel.escalate}")
    print(f"   errored (excluded from ranking and tiering)={sel.errored}")

    Path("outputs/en_vi_track_result.json").write_text(
        json.dumps(
            {
                "primary": PRIMARY,
                "delta": margin.delta,
                "level": LEVEL,
                "gate_d": gate_d,
                "families_executed": sorted(families),
                "errored": errored,
                "levels": levels,
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
                "selection": {
                    "chosen": sel.chosen,
                    "reason": sel.reason,
                    "verdicts": sel.verdicts,
                    "quality_tier": list(sel.quality_tier),
                    "escalate": sel.escalate,
                    "errored": sel.errored,
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
