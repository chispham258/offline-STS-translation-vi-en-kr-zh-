"""Score an existing run manifest with COMET (Gate 0, D55).

    uv run python scripts/score_comet.py <run.jsonl> <refs.jsonl> [out.json]

COMET is deliberately **not** wired into `evaluate()`. D7 keeps it behind an
optional extra because it pulls torch, pytorch-lightning and a ~2.3 GB
checkpoint; making it unconditional would break the framework's "installs and
tests with no model present" property, which every CI run depends on. It is also
slow enough that it should not run on every scoring pass.

So it is an opt-in step over an archived manifest instead — which is exactly the
decoupling D5 exists for: metrics recomputed from a run without re-running any
model. Nothing in `src/onevoice/eval/` changes, so no D21 exception is needed.

COMET is reference-based: it needs the source, the hypothesis and the reference.
Scoring vi->en means src = Vietnamese source text, mt = our English hypothesis,
ref = the FLEURS English reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2

    run_path, refs_path = Path(argv[1]), Path(argv[2])
    out_path = Path(argv[3]) if len(argv) > 3 else run_path.parent / "comet.json"

    from onevoice.eval.metrics.comet import DEFAULT_MODEL, comet, comet_available

    if not comet_available():
        print(
            "COMET is not available. Install it with: uv sync --extra comet\n"
            "If it is installed and still unavailable, check that setuptools<81 "
            "is present — torchmetrics 0.10.x imports pkg_resources, which "
            "setuptools>=81 no longer ships."
        )
        return 1

    refs = {json.loads(x)["utt_id"]: json.loads(x) for x in refs_path.open()}
    runs = {json.loads(x)["utt_id"]: json.loads(x) for x in run_path.open()}
    ids = sorted(set(refs) & set(runs))
    if not ids:
        print("no overlapping utt_ids between run and refs")
        return 1

    # Skip records that errored: the evaluator excludes them from every other
    # metric, so including them here would make COMET the one metric scored
    # over a different population.
    scored = [i for i in ids if runs[i].get("status", "ok") == "ok"]

    result = comet(
        srcs=[refs[i]["src_text"] for i in scored],
        hyps=[runs[i]["hyp_text"] for i in scored],
        refs=[refs[i]["ref_text"] for i in scored],
    )

    payload = {
        "metric": "comet",
        "value": result.value,
        "model": DEFAULT_MODEL,
        "run": str(run_path),
        "refs": str(refs_path),
        "utterances_scored": len(scored),
        "utterances_excluded": len(ids) - len(scored),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
