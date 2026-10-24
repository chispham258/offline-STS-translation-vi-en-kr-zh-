"""Build a reference manifest from a real speech corpus.

Phase 4 uses FLEURS (`google/fleurs`, CC-BY-4.0), chosen on evidence: CoVoST2
has no Vietnamese at all (D39). FLEURS is FLoRes-derived, so an utterance's `id`
identifies the same sentence in every language — which is what turns Vietnamese
audio plus the English row's transcription into a translation reference.

**Boundaries.** FLEURS ships one utterance per audio file and no timing
annotation, so the manifest's `speech_start_s`/`speech_end_s` are the file
extent. Those bound the *file*, not the speech: they include lead-in and
trailing silence. Per D38 that makes the D10 accuracy assertion **not
applicable** here, and the sidecar says so in the manifest's own metadata rather
than leaving a reader to infer it. See `docs/BOUNDARY_METHODOLOGY.md`.

Deriving boundaries with an energy detector instead is explicitly forbidden:
asserting an energy VAD against energy-derived boundaries passes by
construction.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from onevoice.audio import write_wav

DEFAULT_CORPUS = "google/fleurs"
DEFAULT_LIMIT = 50
DEFAULT_SEED = 0

_INSTALL_HINT = (
    "datasets is not installed. Install it with: uv sync --extra corpus. "
    "It is kept out of the default environment so CI stays lightweight."
)

#: FLEURS config per language code we care about.
FLEURS_CONFIGS = {
    "vi": "vi_vn",
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
}


def fleurs_config(lang: str) -> str:
    """FLEURS config name for a language code."""
    try:
        return FLEURS_CONFIGS[lang]
    except KeyError:
        known = ", ".join(sorted(FLEURS_CONFIGS))
        raise ValueError(
            f"no FLEURS config mapped for language {lang!r}; known: {known}"
        ) from None


@dataclass(frozen=True)
class CorpusManifest:
    """Where a built manifest and its sidecar live."""

    refs_path: Path
    meta_path: Path
    audio_dir: Path
    utterance_ids: list[str]


def _framework_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break a build
        return "unknown"


def select_ids(available: list[str], limit: int | None, seed: int) -> list[str]:
    """Choose a deterministic subset of utterance ids.

    Deterministic given (available, limit, seed) — and the chosen ids are
    recorded in the sidecar regardless, so a run is reproducible from its own
    metadata even if this function ever changes.
    """
    ordered = sorted(available)
    if limit is None or limit >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, limit))


def build_refs(
    out_dir: str | Path,
    src_lang: str = "vi",
    tgt_lang: str = "en",
    split: str = "test",
    limit: int | None = DEFAULT_LIMIT,
    seed: int = DEFAULT_SEED,
    corpus: str = DEFAULT_CORPUS,
) -> CorpusManifest:
    """Write a RefManifest plus its boundary sidecar from FLEURS.

    `limit=None` uses the whole split. The default is a bounded subset: an
    end-to-end run puts ASR, MT and TTS on every utterance, and at measured
    CPU latencies the full 857-utterance vi split takes hours per configuration.
    A benchmark nobody re-runs stops being a benchmark.
    """
    try:
        import io

        import soundfile as sf
        from datasets import Audio, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(_INSTALL_HINT) from exc

    out = Path(out_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    src_config = fleurs_config(src_lang)
    tgt_config = fleurs_config(tgt_lang)

    # decode=False keeps the raw bytes: datasets>=5 would otherwise pull in
    # torchcodec to decode, and soundfile - already needed here - does the job.
    src_rows = load_dataset(corpus, src_config, split=split).cast_column(
        "audio", Audio(decode=False)
    )
    tgt_rows = load_dataset(corpus, tgt_config, split=split)

    # FLoRes ids are the join key: the same id is the same sentence in every
    # language, which is what makes the target row a translation reference.
    tgt_text = {
        str(row["id"]): row["transcription"]
        for row in tgt_rows.remove_columns(["audio"])
    }
    src_by_id = {
        str(row["id"]): row for row in src_rows if str(row["id"]) in tgt_text
    }

    chosen = select_ids(list(src_by_id), limit, seed)

    records = []
    for utt_id in chosen:
        row = src_by_id[utt_id]
        decoded, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        pcm = np.asarray(decoded, dtype=np.float32)
        if pcm.ndim > 1:  # defensive: FLEURS is mono, but do not assume
            pcm = pcm.mean(axis=1).astype(np.float32)
        rate = int(rate)
        rel = f"audio/{utt_id}.wav"
        write_wav(audio_dir / f"{utt_id}.wav", pcm, rate)

        duration = len(pcm) / rate
        records.append(
            {
                "utt_id": utt_id,
                "audio_path": rel,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src_text": row["transcription"],
                "ref_text": tgt_text[utt_id],
                # D38: file extent, NOT speech boundaries.
                "speech_start_s": 0.0,
                "speech_end_s": duration,
            }
        )

    refs_path = out / "refs.jsonl"
    refs_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    meta_path = out / "refs.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "corpus": corpus,
                "corpus_revision": str(
                    getattr(src_rows.info, "version", None) or "unknown"
                ),
                "src_config": src_config,
                "tgt_config": tgt_config,
                "split": split,
                "utterances": len(records),
                "subset_limit": limit,
                "subset_seed": seed,
                "utterance_ids": chosen,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "boundary_source": "file_extent",
                "boundary_note": (
                    "FLEURS ships one utterance per audio file with no timing "
                    "annotation. Boundaries are the file extent and therefore "
                    "include lead-in and trailing silence. They are NOT speech "
                    "boundaries and must not be used to score VAD accuracy."
                ),
                "d10_assertion": "not_applicable",
                "built_by": "onevoice.corpus",
                "framework_commit": _framework_commit(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return CorpusManifest(
        refs_path=refs_path,
        meta_path=meta_path,
        audio_dir=audio_dir,
        utterance_ids=chosen,
    )


def read_manifest_meta(refs_path: str | Path) -> dict:
    """Load the sidecar for a manifest, if one exists.

    Returns a `declared`/unknown default when absent, so a caller can always ask
    a manifest what its boundaries mean rather than assuming.
    """
    sidecar = Path(refs_path).with_suffix("").with_suffix(".meta.json")
    if not sidecar.is_file():
        sidecar = Path(refs_path).parent / "refs.meta.json"
    if not sidecar.is_file():
        return {"boundary_source": "unknown", "d10_assertion": "unknown"}
    return json.loads(sidecar.read_text(encoding="utf-8"))
