"""Phase 6C: build the blinding key and one offline package per session (D77).

    .venv-tts-prep/bin/python scripts/build_mos_panel.py --sessions 80 [--pilot]

Session n covers block (n - 1) mod 8, so 80 sessions give 10 per block; building
more later appends sessions without changing any already built.

Reads the prepared stimuli (outputs/mos_vi_40/stimuli.jsonl) and the rater text
(configs/panel_text_vi.json). Writes outputs/mos_vi_40/panel/key.json - which
maps opaque ids back to systems and must never be sent to a rater - and
outputs/mos_vi_40/panel/rater_NNN/ for each session. Without native-speaker
approval of the text only --pilot packages build.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from onevoice import panel
from onevoice.provenance import file_hash, framework_identity

OUT = Path("outputs/mos_vi_40")
MANIFEST = Path("corpora/tts_mos_vi_40/manifest.jsonl")
TEXT = Path("configs/panel_text_vi.json")


def main() -> None:
    import soundfile

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=int, default=80)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    ids = [json.loads(line)["utt_id"] for line in MANIFEST.read_text().splitlines()]
    records = [
        {**r, "duration_s": soundfile.info(r["output_path"]).duration}
        for r in map(json.loads, (OUT / "stimuli.jsonl").read_text().splitlines())
    ]
    clips = panel.build_clips(records, ids)
    text = json.loads(TEXT.read_text(encoding="utf-8"))

    root = OUT / "panel"
    raters = {}
    for n in range(1, args.sessions + 1):
        plan = panel.rater_plan(clips, ids, n)
        pkg = panel.build_package(
            clips, plan, text, root / f"rater_{n:03d}", pilot=args.pilot
        )
        session = panel.read_page_session(pkg / "index.html")
        raters[n] = {"session_id": session["session_id"], **asdict(plan)}

    key = {
        "pilot": args.pilot,
        "manifest_sha256": file_hash(MANIFEST),
        "stimuli_sha256": file_hash(OUT / "stimuli.jsonl"),
        "text_sha256": file_hash(TEXT),
        "text_approved_by": text.get("approved_by"),
        **framework_identity(),
        "blocks": panel.blocks(ids),
        "clips": [asdict(c) for c in clips],
        "raters": raters,
    }
    (root / "key.json").write_text(
        json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    kind = "PILOT " if args.pilot else ""
    print(f"{kind}{args.sessions} session packages, 20 trials each -> {root}")


if __name__ == "__main__":
    main()
