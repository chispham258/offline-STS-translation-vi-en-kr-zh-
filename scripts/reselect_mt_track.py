"""Re-apply the selection logic to archived MT comparisons (D66).

Recomputes **nothing**. It reads the stored comparison numbers -- delta,
interval, p_greater, paired_n -- from the archived result and re-runs the
decision logic over them. No model inference, no metric, no bootstrap, so the
measurements cannot move.
"""

from __future__ import annotations

import json
from pathlib import Path

from onevoice.selection import Margin, select
from onevoice.stats import Comparison

RESULT = Path("outputs/mt_track_result.json")
INCUMBENT = "nllb"
#: Approved before any candidate ran. Read, never derived.
RESOLUTION = {"vi->en": 0.0117, "vi->zh": 1.5675, "vi->ko": 0.0176}
K = 1.5


def main() -> None:
    report = json.loads(RESULT.read_text())
    for direction, block in report.items():
        comparisons = {
            name: Comparison(
                name=name,
                delta=c["delta"],
                low=c["low"],
                high=c["high"],
                p_greater=c["p_greater"],
                n=c["paired_n"],
                resamples=1000,
                level=block["level"],
            )
            for name, c in block["comparisons"].items()
        }
        margin = Margin(
            slot="mt",
            metric=block["primary"],
            k=K,
            resolution=RESOLUTION[direction],
            lower_is_better=False,
        )
        assert margin.delta == block["delta"], "delta moved; it must not"

        sel = select(
            slot="mt",
            incumbent=INCUMBENT,
            comparisons=comparisons,
            margin=margin,
            errored=block["errored"],
        )
        block["selection"] = {
            "chosen": sel.chosen,
            "reason": sel.reason,
            "verdicts": sel.verdicts,
            "quality_tier": list(sel.quality_tier),
            "escalate": sel.escalate,
            "errored": sel.errored,
        }
        print(f"\n{direction}  primary={block['primary']} delta={margin.delta:.4g}")
        for name, c in comparisons.items():
            print(f"  {name:<8} d={c.delta:+.4f} CI[{c.low:+.4f},{c.high:+.4f}]"
                  f"  {sel.verdicts[name]}")
        print(f"  -> {sel.chosen}: {sel.reason}")

    RESULT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
