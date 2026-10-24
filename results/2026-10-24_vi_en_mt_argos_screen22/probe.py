#!/usr/bin/env python3
"""Re-run the Argos Translate vi->en MT-only fixed-22 probe.

This is the archived reproduction path for `report.md`. It uses the official
Argos model package (`.argosmodel`) directly through CTranslate2, without the
`argostranslate` wrapper and without Argos sentence-boundary detection. The 22
screen rows are single sentences, so SBD is not part of this probe.

Example (the archived int8 ASR-source arm):

    uv run python runs/bakeoff_vi_en/mt_argos_screen22/probe.py \
        --package /path/to/translate-vi_en-1_9.argosmodel \
        --source runs/bakeoff_vi_en/asr_wav2vec2_lm_screen22/run.jsonl \
        --source-kind asr \
        --compute-type int8 \
        --out runs/bakeoff_vi_en/mt_argos_screen22/argos_vi_en_int8_screen22.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import zipfile
from pathlib import Path

import ctranslate2
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[3]


def extract_package(package: Path, work: Path) -> Path:
    """Extract an Argos package and return its single model root."""
    roots = [
        p
        for p in work.iterdir()
        if p.is_dir() and (p / "model" / "model.bin").is_file()
    ]
    if roots:
        return roots[0]
    with zipfile.ZipFile(package) as archive:
        archive.extractall(work)
    roots = [
        p
        for p in work.iterdir()
        if p.is_dir() and (p / "model" / "model.bin").is_file()
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one model root, found {roots}")
    return roots[0]


def load_source(kind: str, path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    key = "asr_text" if kind == "asr" else "src_text"
    missing = [r.get("utt_id") for r in rows if not r.get(key)]
    if missing:
        raise RuntimeError(f"missing {key} for {missing}")
    return [{"utt_id": r["utt_id"], "src_text": r[key]} for r in rows]


def decode(processor: spm.SentencePieceProcessor, tokens: list[str]) -> str:
    return (
        processor.decode_pieces(tokens)
        .replace("▁", " ")
        .replace("_", " ")
        .lstrip()
    )


def translate_rows(
    root: Path,
    rows: list[dict[str, str]],
    *,
    compute_type: str,
    beam_size: int,
    length_penalty: float,
    intra_threads: int,
    inter_threads: int,
    warmup: int,
) -> list[dict[str, object]]:
    processor = spm.SentencePieceProcessor(
        model_file=str(root / "sentencepiece.model")
    )
    translator = ctranslate2.Translator(
        str(root / "model"),
        device="cpu",
        compute_type=compute_type,
        inter_threads=inter_threads,
        intra_threads=intra_threads,
    )

    def infer(text: str) -> tuple[str, float]:
        tokens = processor.encode(text, out_type=str)
        started = time.perf_counter()
        result = translator.translate_batch(
            [tokens],
            beam_size=beam_size,
            num_hypotheses=1,
            length_penalty=length_penalty,
            replace_unknowns=True,
            max_batch_size=32,
            batch_type="tokens",
            return_scores=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return decode(processor, result[0].hypotheses[0]), elapsed_ms

    for _ in range(warmup):
        infer(rows[0]["src_text"])

    outputs: list[dict[str, object]] = []
    for row in rows:
        hypothesis, elapsed_ms = infer(row["src_text"])
        outputs.append(
            {
                "utt_id": row["utt_id"],
                "src_text": row["src_text"],
                "hyp_text": hypothesis,
                "wall_ms": elapsed_ms,
                "status": "ok",
                "compute_type": compute_type,
                "beam_size": beam_size,
                "length_penalty": length_penalty,
                "intra_threads": intra_threads,
                "inter_threads": inter_threads,
            }
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-kind",
        choices=("asr", "refs"),
        default="asr",
        help="`asr` reads asr_text from a run.jsonl; `refs` reads src_text.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.2)
    parser.add_argument("--intra-threads", type=int, default=4)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    rows = load_source(args.source_kind, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="argos-vi-en-") as tmp:
        root = extract_package(args.package, Path(tmp))
        outputs = translate_rows(
            root,
            rows,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            intra_threads=args.intra_threads,
            inter_threads=args.inter_threads,
            warmup=args.warmup,
        )

    with args.out.open("w") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    times = [float(row["wall_ms"]) for row in outputs]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n": len(outputs),
                "compute_type": args.compute_type,
                "wall_ms_mean": round(statistics.mean(times), 1),
                "wall_ms_p50": round(statistics.median(times), 1),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
