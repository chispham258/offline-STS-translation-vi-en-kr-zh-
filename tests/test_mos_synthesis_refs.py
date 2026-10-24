"""Bridge from the text-only MOS manifest to synthesis inputs (6C).

The TTS benchmark reads full RefRecords. The MOS manifest is text only by
design (D70), so synthesis uses the parent's records for the 40 ids - and must
refuse if any text differs from what the frozen manifest says the panel hears.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onevoice.stimulus import mos_synthesis_refs

MANIFEST = Path("corpora/tts_mos_vi_40/manifest.jsonl")
PARENT = Path("corpora/fleurs_en_vi_200/refs.jsonl")


def _parent(utt_id, text, tgt="vi"):
    return {"utt_id": utt_id, "tgt_lang": tgt, "ref_text": text, "src_text": "x"}


def _item(utt_id, text, tgt="vi"):
    return {"utt_id": utt_id, "text": text, "tgt_lang": tgt}


def test_returns_parent_records_in_manifest_order():
    parent = [_parent("1", "một"), _parent("2", "hai"), _parent("3", "ba")]
    out = mos_synthesis_refs([_item("3", "ba"), _item("1", "một")], parent)
    assert [r["utt_id"] for r in out] == ["3", "1"]
    assert out[0] is not parent[2]  # a copy, the parent is never mutated


@pytest.mark.parametrize(
    "item",
    [
        _item("1", "một khác"),  # text drifted from the frozen manifest
        _item("9", "chín"),  # id absent from the parent
        _item("1", "một", tgt="en"),  # wrong target language
    ],
)
def test_refuses_any_mismatch_with_the_frozen_manifest(item):
    with pytest.raises(ValueError):
        mos_synthesis_refs([item], [_parent("1", "một")])


@pytest.mark.skipif(not MANIFEST.is_file(), reason="MOS manifest not built")
def test_the_real_manifest_bridges_to_forty_parent_records():
    items = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    parent = [json.loads(line) for line in PARENT.read_text().splitlines()]
    out = mos_synthesis_refs(items, parent)
    assert len(out) == 40
    assert [r["ref_text"] for r in out] == [i["text"] for i in items]
