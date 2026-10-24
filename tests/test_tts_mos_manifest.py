"""The frozen Vietnamese MOS manifest (Phase 6B).

The manifest fixes which 40 sentences the panel hears, **before any audio or
rating exists**. These tests pin the properties that make that freeze mean
something: it is reproducible from its recorded inputs, it matches the parent
the pre-registration froze, and it contains text only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

MANIFEST = Path("corpora/tts_mos_vi_40/manifest.jsonl")
SIDECAR = Path("corpora/tts_mos_vi_40/manifest.meta.json")

pytestmark = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="MOS manifest not built"
)


@pytest.fixture
def meta():
    return json.loads(SIDECAR.read_text())


@pytest.fixture
def items():
    return [json.loads(line) for line in MANIFEST.read_text().splitlines()]


def test_manifest_holds_exactly_the_pre_registered_count(items, meta):
    assert len(items) == 40 == meta["utterances"]
    assert len({i["utt_id"] for i in items}) == 40


def test_manifest_is_reproducible_from_its_own_recorded_inputs(meta):
    """Seed, limit and parent hash are recorded, so the selection can be redone
    without this script. A manifest that cannot be rebuilt is an assertion."""
    from onevoice.corpus import select_ids

    parent = json.loads(Path(meta["parent_corpus"], "refs.meta.json").read_text())
    redone = select_ids(
        list(parent["utterance_ids"]), limit=meta["limit"], seed=meta["seed"]
    )
    assert redone == meta["utterance_ids"]


def test_manifest_bytes_match_the_recorded_digest(meta):
    """A hash in the sidecar that nothing checks is decoration."""
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == meta["manifest_sha256"]


def test_parent_corpus_is_the_one_the_preregistration_froze(meta):
    """If the parent moved, the frozen manifest is not the registered one."""
    parent_refs = Path(meta["parent_corpus"], "refs.jsonl")
    assert (
        hashlib.sha256(parent_refs.read_bytes()).hexdigest()
        == meta["parent_refs_sha256"]
        == "ec2324551590af7b030b739f8863b710a9a3800853c3ce3d0ec27a6415411f10"
    )


def test_text_is_the_target_language_reference_not_a_model_output(items, meta):
    """Evaluating TTS on reference text isolates it from MT error, the same
    separation the MT track used to isolate MT from ASR."""
    assert meta["text_field"].startswith("ref_text")
    parent_rows = {
        r["utt_id"]: r
        for r in (
            json.loads(line)
            for line in Path(meta["parent_corpus"], "refs.jsonl")
            .read_text()
            .splitlines()
        )
    }
    for item in items:
        assert item["tgt_lang"] == "vi"
        assert item["text"] == parent_rows[item["utt_id"]]["ref_text"]
        assert item["text"].strip()


def test_duplicate_items_follow_the_frozen_stride(items, meta):
    ids = [i["utt_id"] for i in items]
    assert meta["duplicate_ids"] == ids[0::10]
    assert len(meta["duplicate_ids"]) == 4


def test_manifest_carries_no_audio_rating_or_score(items, meta):
    """6B is text only. A rating or score reaching this artifact would mean the
    evaluation set could be tuned against results it already contains."""
    forbidden = {"mos", "rating", "score", "audio", "wav", "delta", "verdict"}
    for item in items:
        assert not (forbidden & set(item)), f"manifest item carries {item.keys()}"
    assert not (forbidden & set(meta))
