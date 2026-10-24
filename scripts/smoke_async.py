"""Run utterances through the async pipeline under hard time limits.

This is the reproducible replacement for the unsaved inline smoke harness.
It always builds queues through build_async_components(), so the queue objects
under test are the ones the pipeline actually uses.

Usage:
    uv run python scripts/smoke_async.py \
        --config configs/full_en_vi_vieneu_async_silence.yaml \
        --refs corpora/fleurs_en_vi_200/refs_1utt.jsonl \
        --out runs/smoke_async_1utt \
        --asr-model faster_whisper_streaming \
        --timeout 120
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Registry entries are created by importing component modules. The normal
# benchmark entrypoint imports these elsewhere; a standalone smoke script must
# import the ones named in its config or resolve() sees an empty registry.
import onevoice.components.dummy.silence_tts  # noqa: F401
import onevoice.components.faster_whisper_asr_streaming  # noqa: F401
import onevoice.components.language_router_tts  # noqa: F401
import onevoice.components.m2m_mt  # noqa: F401
import onevoice.components.piper_tts  # noqa: F401
import onevoice.components.silero_vad  # noqa: F401
import onevoice.components.t5_mt  # noqa: F401
from onevoice.async_pipeline import build_async_components, run_utterance_async
from onevoice.config import load_config
from onevoice.eval.manifest import read_refs
from onevoice.types import SystemClock, TimingTrace


def _wiring_report(components) -> bool:
    asr = components["asr"]
    mt = components["mt"]
    tts = components["tts"]
    pairs = (
        ("asr.out -> mt.in", asr.out_queue is mt.in_queue),
        ("mt.out -> tts.in", mt.out_queue is tts.in_queue),
    )
    print(
        "[WIRE] "
        f"asr_in={id(asr.in_queue)} asr_out={id(asr.out_queue)} "
        f"mt_in={id(mt.in_queue)} mt_out={id(mt.out_queue)} "
        f"tts_in={id(tts.in_queue)} tts_out={id(tts.out_queue)}",
        flush=True,
    )
    ok = True
    for label, same in pairs:
        print(f"[WIRE] {label}: {'OK' if same else 'MISMATCH'}", flush=True)
        ok = ok and same
    return ok


async def _run(args, config, refs, out_dir: Path, fixtures_root: Path) -> int:
    vad, components, _ = build_async_components(config)
    if not _wiring_report(components):
        print("[SMOKE] aborting: queue wiring mismatch", flush=True)
        return 3

    for ref in refs:
        trace = TimingTrace(SystemClock())
        trace.stamp("stream_start")
        print(f"[SMOKE] utterance {ref.utt_id}", flush=True)
        record = await asyncio.wait_for(
            run_utterance_async(
                ref, vad, components, trace, out_dir, fixtures_root
            ),
            timeout=args.timeout,
        )
        print(f"[SMOKE] record {record.model_dump_json()}", flush=True)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--asr-model",
        default="faster_whisper_streaming",
        help="registry name for the ASR component (default: streaming)",
    )
    parser.add_argument(
        "--asr-model-name",
        default=None,
        help="override asr.params.model_name (e.g. tiny for a fast smoke run)",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable per-stage debug logs",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = load_config(args.config)
    config.asr.model = args.asr_model
    if args.asr_model_name:
        config.asr.params["model_name"] = args.asr_model_name
    if args.asr_model.endswith("_streaming"):
        config.asr.params.setdefault("chunk_batch_size", 2)
        config.asr.params.setdefault("stream_chunk_length", 5)
    print(
        f"[SMOKE] asr={config.asr.model} timeout={args.timeout}s "
        f"refs={args.refs} limit={args.limit}",
        flush=True,
    )

    refs = read_refs(args.refs)[: args.limit]
    if not refs:
        print("[SMOKE] no references found", flush=True)
        return 1

    out_dir = Path(args.out)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    fixtures_root = Path(args.refs).parent

    try:
        return asyncio.run(_run(args, config, refs, out_dir, fixtures_root))
    except TimeoutError:
        print(
            f"[SMOKE] TIMEOUT after {args.timeout}s - pipeline did not finish",
            flush=True,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
