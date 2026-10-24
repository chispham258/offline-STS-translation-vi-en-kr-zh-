"""Phase 6C: synthesis inputs and prepared MOS stimuli for the Vietnamese panel.

    # 1. Synthesis inputs: the parent RefRecords for the 40 frozen ids.
    uv run python scripts/prepare_mos_stimuli.py refs

    # 2. Synthesise each system in its own environment. Resumable: a rerun
    #    skips clips already recorded in <out>/syntheses.jsonl.
    .venv-tts-f5/bin/python scripts/prepare_mos_stimuli.py synth \
        --config configs/mos_vi_f5.yaml --out outputs/mos_vi_40/synth/f5

    # 3. Prepare anchors and system outputs (in .venv-tts-prep only).
    .venv-tts-prep/bin/python scripts/prepare_mos_stimuli.py prepare \
        --system piper-vi-vais1000=outputs/mos_vi_40/synth/piper ...

A cell the protocol cannot prepare is written to errored.jsonl with its reason
and gets no stimulus (§8). Nothing is limited or repaired to make it pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from onevoice.provenance import file_hash
from onevoice.stimulus import StimulusError, mos_synthesis_refs, prepare_stimulus

MANIFEST = Path("corpora/tts_mos_vi_40/manifest.jsonl")
SIDECAR = Path("corpora/tts_mos_vi_40/manifest.meta.json")
PARENT = Path("corpora/fleurs_en_vi_200/refs.jsonl")
# The Vietnamese FLEURS recordings: vi is the SOURCE language of this corpus.
ANCHOR_AUDIO = Path("corpora/fleurs_vi_en_200/audio")
OUT = Path("outputs/mos_vi_40")
# Components register on import; a TTS-only run needs the dummies plus one TTS.
TTS_MODULES = {
    "piper": "onevoice.components.piper_tts",
    "mms": "onevoice.components.mms_tts",
    "f5": "onevoice.components.cloning_tts",
    "xtts": "onevoice.components.cloning_tts",
    "silence": "onevoice.components.dummy.silence_tts",
}
DUMMY_MODULES = (
    "onevoice.components.dummy.energy_vad",
    "onevoice.components.dummy.echo_asr",
    "onevoice.components.dummy.identity_mt",
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _manifest() -> tuple[list[dict], str]:
    digest = file_hash(MANIFEST)
    frozen = json.loads(SIDECAR.read_text())["manifest_sha256"]
    if digest != frozen:
        raise SystemExit(f"manifest digest {digest} != frozen {frozen}; refusing")
    return _jsonl(MANIFEST), digest


def cmd_refs(_: argparse.Namespace) -> None:
    items, _digest = _manifest()
    refs = mos_synthesis_refs(items, _jsonl(PARENT))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "refs.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in refs),
        encoding="utf-8",
    )
    print(f"{len(refs)} synthesis refs -> {path}")


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def synthesise(config_path: str | Path, refs_path: str | Path, out: str | Path) -> int:
    """Synthesise every ref not yet recorded in `out/syntheses.jsonl`.

    Each clip's record is appended the moment its WAV is written, and each call
    that does work appends one provenance line to `invocations.jsonl`, so a
    stalled run loses at most the clip in progress. Returns clips synthesised.
    """
    import numpy as np

    from onevoice.audio import write_wav
    from onevoice.config import load_config
    from onevoice.eval.manifest import read_refs
    from onevoice.pipeline import build_components
    from onevoice.provenance import run_identity

    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    index, invocations = out / "syntheses.jsonl", out / "invocations.jsonl"
    done = {r["utt_id"] for r in _jsonl(index)} if index.is_file() else set()
    todo = [r for r in read_refs(refs_path) if r.utt_id not in done]
    if not todo:
        return 0

    config = load_config(config_path)
    for module in (*DUMMY_MODULES, TTS_MODULES[config.tts.model]):
        importlib.import_module(module)
    tts = build_components(config)["tts"]
    invocation = len(_jsonl(invocations)) if invocations.is_file() else 0
    _append(
        invocations,
        {
            "invocation": invocation,
            "started_at": datetime.now(UTC).isoformat(),
            **run_identity(config_path, refs_path),
            # Called directly, not via onevoice.benchmark.collect_provenance:
            # that module loads the evaluator, whose dependencies (jiwer) model
            # environments do not have.
            "components": {
                "tts": tts.provenance()
                if callable(getattr(tts, "provenance", None))
                else {"component": type(tts).__name__}
            },
        },
    )

    for ref in todo:
        tts.reset()
        started = time.perf_counter()
        events = list(tts.push(ref.ref_text, is_final=True)) + list(tts.flush())
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not events:
            continue  # no audio: prepare records the cell as errored
        rate = events[0].sample_rate
        pcm = np.concatenate([e.pcm for e in events])
        wav = out / "audio" / f"{ref.utt_id}.wav"
        write_wav(wav, pcm, rate)
        _append(
            index,
            {
                "utt_id": ref.utt_id,
                "wav_path": str(wav),
                "sample_rate": rate,
                "audio_seconds": len(pcm) / rate,
                "latency_ms": latency_ms,
                "invocation": invocation,
            },
        )
    return len(todo)


def cmd_synth(args: argparse.Namespace) -> None:
    n = synthesise(args.config, args.refs, args.out)
    print(f"synthesised {n} -> {args.out}")


def _system_provenance(synth_dir: Path) -> dict[str, dict]:
    """Provenance per utt_id: per invocation when resumable, else per run."""
    rows = _jsonl(synth_dir / "syntheses.jsonl")
    if (synth_dir / "invocations.jsonl").is_file():
        runs = {i["invocation"]: i for i in _jsonl(synth_dir / "invocations.jsonl")}
        return {r["utt_id"]: runs[r["invocation"]] for r in rows}
    meta = json.loads((synth_dir / "metadata.json").read_text())
    return {r["utt_id"]: meta for r in rows}


def cmd_prepare(args: argparse.Namespace) -> None:
    items, digest = _manifest()
    ids = [i["utt_id"] for i in items]

    # (role, utt_id, source wav or None, system provenance)
    jobs = [("upper_anchor", u, ANCHOR_AUDIO / f"{u}.wav", None) for u in ids]
    jobs += [("lower_anchor", u, ANCHOR_AUDIO / f"{u}.wav", None) for u in ids]
    for spec in args.system:
        name, synth_dir = spec.split("=", 1)
        synth_dir = Path(synth_dir)
        runs = _system_provenance(synth_dir)
        syntheses = _jsonl(synth_dir / "syntheses.jsonl")
        wavs = {s["utt_id"]: s["wav_path"] for s in syntheses}
        for u in ids:
            src = Path(wavs[u]) if u in wavs else None
            run = runs.get(u, {})
            provenance = {
                k: run[k]
                for k in (
                    "components",
                    "config_hash",
                    "refs_hash",
                    "framework_commit",
                    "framework_dirty",
                    "invocation",
                    "recovered",
                )
                if k in run
            }
            jobs.append((f"system:{name}", u, src, provenance))

    records, errored = [], []
    for role, utt_id, src, provenance in jobs:
        cell = {"role": role, "utt_id": utt_id}
        if src is None or not src.is_file():
            errored.append({**cell, "reason": "no audio produced"})
            continue
        dst = OUT / "stimuli" / role.replace(":", "_") / f"{utt_id}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            records.append(
                prepare_stimulus(
                    src,
                    dst,
                    utt_id=utt_id,
                    manifest_sha256=digest,
                    role=role,
                    system_provenance=provenance,
                )
            )
        except StimulusError as exc:
            errored.append({**cell, "reason": str(exc)})

    for name, rows in (("stimuli.jsonl", records), ("errored.jsonl", errored)):
        (OUT / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
    print(f"prepared {len(records)}, errored {len(errored)} -> {OUT}")
    for e in errored:
        print(f"  ERRORED {e['role']} {e['utt_id']}: {e['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    sub.add_parser("refs").set_defaults(func=cmd_refs)
    synth = sub.add_parser("synth")
    synth.add_argument("--config", required=True)
    synth.add_argument("--refs", default=str(OUT / "refs.jsonl"))
    synth.add_argument("--out", required=True)
    synth.set_defaults(func=cmd_synth)
    prep = sub.add_parser("prepare")
    prep.add_argument("--system", action="append", default=[], metavar="NAME=DIR")
    prep.set_defaults(func=cmd_prepare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
