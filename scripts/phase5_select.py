"""Apply the Phase 5.0 selection rule mechanically.

Written and committed BEFORE any arm finished, so the rule cannot be bent to
the data it will be shown. Spec:
`docs/superpowers/specs/2026-08-06-phase-5-0-whisper-checkpoint-selection.md`

Usage: uv run python scripts/phase5_select.py

Prints the three comparisons, the classification of each challenger, and the
decision the rule produces. It does not decide anything a human then
reinterprets - §7 of the spec is implemented here verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

from onevoice.eval.metrics.error_rate import normalize_transcript
from onevoice.eval.metrics.text import bleu
from onevoice.stats import paired_bootstrap

REFS = Path("corpora/fleurs_vi_en_200/refs.jsonl")
INCUMBENT = "small"

#: Arm -> (output dir, parameter count in millions). Size is the deterministic
#: proxy the rule uses; latency is reported but is not a gate (spec §7).
ARMS = {
    "tiny": ("outputs/phase5_tiny", 37.8),
    "base": ("outputs/phase5_base", 72.6),
    "small": ("outputs/phase5_small", 241.7),
    "turbo": ("outputs/phase5_turbo", 808.9),
}

#: Pre-registered, spec §6. Derived from the n=200 bootstrap resolution.
MARGIN = 0.05
#: Bonferroni over three comparisons against the fixed incumbent, spec §6.
COMPARISONS = 3
CI_LEVEL = 1 - 0.05 / COMPARISONS  # 0.98333...


def _load(path: Path) -> dict[str, dict]:
    return {json.loads(line)["utt_id"]: json.loads(line) for line in path.open()}


def _wer_normalized(records: dict[str, dict], refs: dict[str, dict]):
    """Corpus wer_normalized over a resampled id list."""
    import jiwer

    def score(ids):
        hyp = [normalize_transcript(records[i]["asr_text"]) for i in ids]
        gold = [normalize_transcript(refs[i]["src_text"]) for i in ids]
        return float(jiwer.wer(gold, hyp))

    return score


def _bleu(records: dict[str, dict], refs: dict[str, dict]):
    def score(ids):
        return bleu(
            [records[i]["hyp_text"] for i in ids],
            [refs[i]["ref_text"] for i in ids],
            "en",
        ).value

    return score


def classify(low: float, high: float) -> str:
    """Spec §7, rules 1-4. Lower wer_normalized is better."""
    if high < 0.0:
        return "superior"
    if high < MARGIN:
        return "non-inferior"
    if low > MARGIN:
        return "inferior"
    return "unresolved"


def main() -> int:
    refs = _load(REFS)
    available = {}
    for arm, (out, params) in ARMS.items():
        manifest = Path(out) / "run.jsonl"
        if not manifest.is_file():
            print(f"arm {arm}: NOT RUN")
            continue
        records = _load(manifest)
        if len(records) < len(refs):
            print(f"arm {arm}: INCOMPLETE ({len(records)}/{len(refs)})")
            continue
        available[arm] = (records, params)

    if INCUMBENT not in available:
        print(f"\nincumbent '{INCUMBENT}' has not completed; cannot compare.")
        return 1

    base_records = available[INCUMBENT][0]
    ids = sorted(set(base_records) & set(refs))
    print(f"\nn = {len(ids)}   margin δ = {MARGIN}   CI level = {CI_LEVEL:.4f}")
    print(f"incumbent = whisper-{INCUMBENT}\n")

    verdicts = {}
    for arm, (records, params) in sorted(available.items(), key=lambda kv: kv[1][1]):
        if arm == INCUMBENT:
            continue
        cmp = paired_bootstrap(
            ids,
            _wer_normalized(base_records, refs),
            _wer_normalized(records, refs),
            name=f"wer_normalized {arm} - {INCUMBENT}",
            resamples=1000,
            level=CI_LEVEL,
        )
        verdict = classify(cmp.low, cmp.high)
        verdicts[arm] = (cmp, verdict, params)
        print(f"  {arm:6} Δ={cmp.delta:+.4f}  CI [{cmp.low:+.4f}, {cmp.high:+.4f}]  "
              f"P(worse)={cmp.p_greater:.3f}  -> {verdict}")

    if any(v == "unresolved" for _, v, _ in verdicts.values()):
        print("\nDECISION: ESCALATE to n=400 (spec §8) - at least one comparison "
              "is unresolved.")
        return 0

    superior = {a: v for a, v in verdicts.items() if v[1] == "superior"}
    if superior:
        pick = min(superior, key=lambda a: superior[a][0].delta)
        print(f"\nDECISION: select whisper-{pick} (superior; quality wins "
              "outright regardless of size).")
    else:
        eligible = {INCUMBENT: available[INCUMBENT][1]}
        eligible.update(
            {a: p for a, (_, v, p) in verdicts.items() if v == "non-inferior"}
        )
        pick = min(eligible, key=lambda a: eligible[a])
        if pick == INCUMBENT:
            print(f"\nDECISION: no change - whisper-{INCUMBENT} retained "
                  "(no smaller checkpoint is non-inferior).")
        else:
            print(f"\nDECISION: select whisper-{pick} (smallest non-inferior, "
                  f"{eligible[pick]}M vs {eligible[INCUMBENT]}M).")

    if pick != INCUMBENT:
        conf = paired_bootstrap(
            ids,
            _bleu(base_records, refs),
            _bleu(available[pick][0], refs),
            name=f"bleu {pick} - {INCUMBENT}",
            resamples=1000,
            level=CI_LEVEL,
        )
        print(f"\nconfirmatory: {conf.sentence()}")
        if conf.high < 0:
            print("CONFIRMATORY CONSTRAINT VIOLATED -> selection SUSPENDED, "
                  "escalate to n=400 (spec §7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
