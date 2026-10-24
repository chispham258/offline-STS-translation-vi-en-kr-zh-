"""Phase 6C: package the built MOS sessions for sending to raters (D77).

    python scripts/distribute_mos_panel.py pack [--return-method TEXT]
    python scripts/distribute_mos_panel.py check <path/to/tracker.csv>

`pack` zips every outputs/mos_vi_40/panel/rater_NNN folder into
outputs/mos_vi_40/distribution/, writes tracker.csv (one row per session) and
message_vi.md (one rater message that fits every session). key.json is never
packed. `check` flags anyone given two sessions from the same block, which the
key cannot detect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

PANEL = Path("outputs/mos_vi_40/panel")
OUT = Path("outputs/mos_vi_40/distribution")
MESSAGE = Path("configs/panel_message_vi.md")
TRACKER_FIELDS = [
    "session",
    "block",
    "zip",
    "zip_sha256",
    "pilot",
    "person",
    "contact",
    "sent_date",
    "results_received",
    "notes",
]


def check_tracker(path: str | Path) -> list[str]:
    """One problem line per person holding two sessions of the same block."""
    seen: dict[tuple[str, str], str] = {}
    problems = []
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            person = " ".join(row["person"].split()).casefold()
            if not person:
                continue
            key = (person, row["block"])
            if key in seen:
                problems.append(
                    f"{row['person'].strip()} has sessions {seen[key]} and "
                    f"{row['session']} in block {row['block']}"
                )
            else:
                seen[key] = row["session"]
    return problems


def cmd_pack(args: argparse.Namespace) -> None:
    key = json.loads((PANEL / "key.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, plan in sorted(key["raters"].items(), key=lambda kv: int(kv[0])):
        folder = PANEL / f"rater_{int(n):03d}"
        zip_path = OUT / f"OneVoice_listening_{int(n):03d}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(PANEL).as_posix())
        rows.append(
            {
                "session": n,
                "block": plan["block"],
                "zip": zip_path.name,
                "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                "pilot": key["pilot"],
            }
        )
    with (OUT / "tracker.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRACKER_FIELDS})
    message = (
        MESSAGE.read_text(encoding="utf-8")
        .replace("{return_method}", args.return_method)
        .replace("{zip_name}", "OneVoice_listening_<số>.zip")
        .replace("{session}", "<số>")
    )
    (OUT / "message_vi.md").write_text(message, encoding="utf-8")
    kind = "PILOT " if key["pilot"] else ""
    print(f"{kind}{len(rows)} zips, tracker.csv and message_vi.md -> {OUT}")


def cmd_check(args: argparse.Namespace) -> None:
    problems = check_tracker(args.tracker)
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        raise SystemExit(1)
    print("ok: nobody has two sessions in the same block")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--return-method", default="{return_method}")
    pack.set_defaults(func=cmd_pack)
    check = sub.add_parser("check")
    check.add_argument("tracker")
    check.set_defaults(func=cmd_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
