"""Manifest records and JSONL IO.

A RefManifest is ground truth. A RunManifest is what a pipeline produced. The
evaluator joins them on utt_id. Neither knows anything about models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RefRecord(BaseModel):
    """One ground-truth utterance."""

    utt_id: str
    audio_path: str
    src_lang: str
    tgt_lang: str
    src_text: str
    ref_text: str
    speech_start_s: float
    speech_end_s: float


class RunRecord(BaseModel):
    """One utterance as produced by a pipeline run."""

    utt_id: str
    hyp_text: str
    status: Literal["ok", "error"] = "ok"
    error: str = ""
    asr_text: str = ""
    hyp_audio_path: str = ""
    audio_duration_s: float
    timing: dict[str, float] = Field(default_factory=dict)


def _read_jsonl(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _check_no_duplicates(utt_ids: list[str], manifest: str) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for utt_id in utt_ids:
        (dupes if utt_id in seen else seen).add(utt_id)
    if dupes:
        raise ValueError(
            f"{manifest} contains duplicate utt_id(s) {sorted(dupes)}; a "
            "repeated utt_id would silently inflate or overwrite a scored "
            "utterance"
        )


def read_refs(path: str | Path) -> list[RefRecord]:
    refs = [RefRecord(**row) for row in _read_jsonl(path)]
    _check_no_duplicates([r.utt_id for r in refs], "reference manifest")
    return refs


def read_run(path: str | Path) -> list[RunRecord]:
    run = [RunRecord(**row) for row in _read_jsonl(path)]
    _check_no_duplicates([r.utt_id for r in run], "run manifest")
    return run


def write_run(path: str | Path, records: list[RunRecord]) -> None:
    body = "\n".join(r.model_dump_json() for r in records)
    Path(path).write_text(body + "\n", encoding="utf-8")
