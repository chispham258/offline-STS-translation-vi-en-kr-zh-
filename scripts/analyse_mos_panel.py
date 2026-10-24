"""Phase 6C: analyse returned MOS results in the two stages §10 requires.

    # Stage 1: exclusions, anchor gate, incumbent-only resolution and delta.
    python scripts/analyse_mos_panel.py resolution --results DIR
    # ... review and COMMIT the resolution file, then:
    # Stage 2: paired comparisons and the frozen selection hierarchy.
    python scripts/analyse_mos_panel.py compare --results DIR

Stage 2 refuses to run unless the stage-1 file is committed and unmodified, and
unless DIR holds exactly the results stage 1 recorded. --pilot analyses a pilot
key into outputs/ and skips the commit gate; its output is never evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from onevoice import panel_analysis as pa

KEY = Path("outputs/mos_vi_40/panel/key.json")
# Historical archive paths: MOS is paused (D79) and its artifacts are not
# shipped in this checkout.
RESOLUTION = Path("docs/benchmarks/tts-mos-vi-40-resolution.json")
RESULT = Path("docs/benchmarks/tts-mos-vi-40-result.json")
PILOT_DIR = Path("outputs/mos_vi_40/panel/pilot_analysis")


def require_committed(path: Path) -> None:
    """Refuse unless `path` is tracked by git and identical to HEAD."""
    path = Path(path)

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=path.parent, capture_output=True, text=True
        )

    if git("ls-files", "--error-unmatch", path.name).returncode != 0:
        raise SystemExit(f"{path} is not committed; commit stage 1 before stage 2")
    if git("status", "--porcelain", "--", path.name).stdout.strip():
        raise SystemExit(f"{path} is modified since its commit; refusing stage 2")


def _results(directory: str) -> list[dict]:
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        raise SystemExit(f"no results JSON in {directory}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def cmd_resolution(args: argparse.Namespace) -> None:
    out = PILOT_DIR / RESOLUTION.name if args.pilot else RESOLUTION
    if out.exists() and not args.pilot:
        raise SystemExit(f"{out} exists; a committed resolution is never rewritten")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    record = pa.resolution_stage(key, _results(args.results), pilot=args.pilot)
    _write(out, record)
    print(
        f"{record['status']}: {len(record['valid_raters'])} valid raters, "
        f"excluded {record['excluded'] or 'none'}"
    )
    if record["status"] == "resolved":
        print(
            f"incumbent MOS {record['incumbent_mos']:.3f}, resolution "
            f"{record['resolution']:.4f}, delta {record['delta']:.4f}"
        )
    print(f"-> {out}" + ("" if args.pilot else "  (commit it before stage 2)"))


def cmd_compare(args: argparse.Namespace) -> None:
    res_path = PILOT_DIR / RESOLUTION.name if args.pilot else RESOLUTION
    if not args.pilot:
        require_committed(res_path)
    record = json.loads(res_path.read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    out = pa.comparison_stage(key, _results(args.results), record, pilot=args.pilot)
    _write(PILOT_DIR / RESULT.name if args.pilot else RESULT, out)
    print(f"Gate D: {out['gate_d']}; delta {out['delta']:.4f}")
    for name, c in out["comparisons"].items():
        print(
            f"  {name}: d={c['delta']:+.3f} {c['level']:.3f} CI "
            f"[{c['low']:+.3f}, {c['high']:+.3f}] -> {out['verdicts'].get(name)}"
        )
    for name, why in out["errored"].items():
        print(f"  {name}: ERRORED ({why}), no quality verdict")
    print(f"-> {out['chosen']}: {out['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    for name, func in (("resolution", cmd_resolution), ("compare", cmd_compare)):
        p = sub.add_parser(name)
        p.add_argument("--results", required=True)
        p.add_argument("--pilot", action="store_true")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
