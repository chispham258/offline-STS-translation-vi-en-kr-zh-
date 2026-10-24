"""Build the frozen Vietnamese MOS evaluation manifest (Phase 6B).

Text only. **Synthesises nothing, collects no rating, computes no MOS and no
delta.** Its whole job is to fix *which 40 sentences* the panel will hear,
before any audio exists, so the evaluation set cannot be adjusted once results
are visible.

Selection rule, approved 2026-09-03 before the manifest was built:

    ids = select_ids(parent_ids, limit=N_UTTERANCES, seed=SEED)

`onevoice.corpus.select_ids` is the project's existing deterministic subsetter —
the same function and seed that produced every other corpus subset here,
including the 200-utterance parent. Reusing it keeps one selection idiom in the
project rather than inventing a second, and it spreads the sample across the
parent's whole id range instead of taking a contiguous block, which on FLEURS
would risk topic and speaker clustering.

The chosen ids are written into the sidecar regardless, so the manifest stays
reproducible from its own metadata even if `select_ids` ever changes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from onevoice.corpus import select_ids

#: Frozen by the TTS pre-registration (Phase 6A).
PARENT = Path("corpora/fleurs_en_vi_200")
PARENT_REFS_SHA256 = (
    "ec2324551590af7b030b739f8863b710a9a3800853c3ce3d0ec27a6415411f10"
)
TGT_LANG = "vi"
N_UTTERANCES = 40
SEED = 0
#: Duplicate control items, frozen: a fixed stride over the manifest order.
DUPLICATE_STRIDE = 10

OUT = Path("corpora/tts_mos_vi_40")


def _framework_commit() -> tuple[str, bool]:
    """(commit, dirty). Provenance must never break a build, hence the guards."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except Exception:  # noqa: BLE001
        return "unknown", True


def build() -> dict:
    refs_path = PARENT / "refs.jsonl"
    actual = hashlib.sha256(refs_path.read_bytes()).hexdigest()
    if actual != PARENT_REFS_SHA256:
        raise RuntimeError(
            f"parent corpus hash mismatch: expected {PARENT_REFS_SHA256}, got "
            f"{actual}. The pre-registration froze the parent; a changed parent "
            "means the frozen manifest would not be the one that was registered."
        )

    rows = {
        r["utt_id"]: r
        for r in (json.loads(line) for line in refs_path.read_text().splitlines())
    }
    meta = json.loads((PARENT / "refs.meta.json").read_text())
    if meta["tgt_lang"] != TGT_LANG:
        raise RuntimeError(
            f"parent target language is {meta['tgt_lang']!r}, expected "
            f"{TGT_LANG!r}: the MOS text must be in the language being spoken."
        )

    chosen = select_ids(list(rows), limit=N_UTTERANCES, seed=SEED)
    if len(chosen) != N_UTTERANCES:
        raise RuntimeError(f"expected {N_UTTERANCES} ids, selected {len(chosen)}")

    # The synthesis text is the parent's TARGET-language reference. Using the
    # reference rather than any model's output keeps TTS isolated from MT error,
    # the same separation the MT track used to isolate MT from ASR.
    items = [
        {"utt_id": i, "text": rows[i]["ref_text"], "tgt_lang": TGT_LANG}
        for i in chosen
    ]
    duplicates = chosen[::DUPLICATE_STRIDE]

    OUT.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n"
    (OUT / "manifest.jsonl").write_text(body, encoding="utf-8")

    commit, dirty = _framework_commit()
    sidecar = {
        "built_by": "scripts/build_tts_mos_manifest.py",
        "purpose": "Vietnamese TTS MOS evaluation set (Phase 6B). Text only.",
        "tgt_lang": TGT_LANG,
        "utterances": len(items),
        "selection_rule": (
            "onevoice.corpus.select_ids(parent_ids, limit=40, seed=0) — "
            "sorted(Random(seed).sample(sorted(parent_ids), limit))"
        ),
        "seed": SEED,
        "limit": N_UTTERANCES,
        "parent_corpus": str(PARENT),
        "parent_refs_sha256": PARENT_REFS_SHA256,
        "parent_utterances": len(rows),
        "text_field": "ref_text (parent target language)",
        "utterance_ids": chosen,
        "duplicate_rule": (
            f"manifest_ids[0::{DUPLICATE_STRIDE}] — every system's clip for each "
            "selected utterance is repeated once. Quality-control statistic only: "
            "excluded from system MOS, aggregation and every bootstrap."
        ),
        "duplicate_ids": duplicates,
        "manifest_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "framework_commit": commit,
        "framework_dirty": dirty,
        "not_included": (
            "No audio, no ratings, no MOS, no delta. This artifact fixes the "
            "evaluation text set only."
        ),
    }
    (OUT / "manifest.meta.json").write_text(
        json.dumps(sidecar, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sidecar


if __name__ == "__main__":
    s = build()
    print(f"utterances      : {s['utterances']}")
    print(f"manifest sha256 : {s['manifest_sha256']}")
    print(f"duplicates ({len(s['duplicate_ids'])})  : {s['duplicate_ids']}")
    print(
        f"framework       : {s['framework_commit'][:12]} "
        f"dirty={s['framework_dirty']}"
    )
