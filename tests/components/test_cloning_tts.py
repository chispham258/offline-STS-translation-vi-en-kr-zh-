"""Contract tests for the voice-cloning TTS wrappers.

**No synthesis happens here and none is claimed.** Every test below exercises
the parts of the contract that run *before* the heavy library is imported —
language resolution and the speaker-reference requirement — which is exactly
the Gate-D / preprocessing validation this stage calls for. Whether either model
produces usable audio is `output_tested` evidence and needs an authorised run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from onevoice.components.cloning_tts import (
    F5_LANGS,
    F5TTS,
    XTTS_LANGS,
    ViXttsTTS,
    _commit_from_snapshot,
)
from onevoice.provenance import ProvenanceError


@pytest.fixture
def reference(tmp_path):
    clip = tmp_path / "speaker.wav"
    clip.write_bytes(b"RIFF....WAVEfmt ")  # content is irrelevant; identity is not
    return clip


# --- the speaker reference is part of the experiment -------------------------


@pytest.mark.parametrize("cls", [ViXttsTTS, F5TTS])
def test_a_cloning_model_refuses_to_run_without_a_reference(cls):
    """A silent default speaker would make two runs incomparable while looking
    like a clean run. Refusal happens before any library import."""
    kwargs = {"reference_text": "xin chào"} if cls is F5TTS else {}
    with pytest.raises(ValueError, match="requires `reference_audio`"):
        cls(tgt_lang="vi", reference_audio=None, **kwargs)


@pytest.mark.parametrize("cls", [ViXttsTTS, F5TTS])
def test_a_missing_reference_file_raises_rather_than_being_skipped(cls, tmp_path):
    kwargs = {"reference_text": "xin chào"} if cls is F5TTS else {}
    with pytest.raises(FileNotFoundError):
        cls(tgt_lang="vi", reference_audio=str(tmp_path / "absent.wav"), **kwargs)


def test_f5_requires_the_reference_transcript_too(reference):
    """F5 conditions on the reference TRANSCRIPT as well as its audio. A wrong
    or empty transcript degrades output without raising, so it is required."""
    with pytest.raises(ValueError, match="reference_text"):
        F5TTS(tgt_lang="vi", reference_audio=str(reference))


# --- language contract -------------------------------------------------------


def test_xtts_maps_the_pipeline_code_to_the_checkpoint_code(reference):
    """XTTS speaks `zh-cn`, not `zh`. Passing our own code straight through
    would either raise deep in the library or be accepted as something else."""
    assert XTTS_LANGS["zh"] == "zh-cn"
    assert XTTS_LANGS["ko"] == "ko"
    assert XTTS_LANGS["vi"] == "vi"
    assert XTTS_LANGS["en"] == "en"


def test_xtts_refuses_a_language_it_cannot_express(reference):
    """Refusal, not fluent audio in the wrong language — which a listener panel
    would score as a bad model rather than a bad harness."""
    with pytest.raises(ValueError, match="cannot express target language"):
        ViXttsTTS(tgt_lang="th", reference_audio=str(reference))


def test_f5_refuses_any_language_but_the_one_it_declares(reference):
    with pytest.raises(ValueError, match="declares only"):
        F5TTS(
            tgt_lang="en",
            reference_audio=str(reference),
            reference_text="hello",
        )


def test_f5_pinyin_vocabulary_is_not_treated_as_chinese_coverage(reference):
    """Its vocab file carries pinyin tokens inherited from the base F5-TTS
    model. That is an artefact of the base vocabulary, not a capability of this
    Vietnamese fine-tune, and must not become a coverage claim."""
    assert "zh" not in F5_LANGS
    with pytest.raises(ValueError, match="pinyin"):
        F5TTS(
            tgt_lang="zh",
            reference_audio=str(reference),
            reference_text="ni hao",
        )


# --- declared coverage is config evidence, not verified behaviour ------------


def test_xtts_declares_the_four_languages_the_pipeline_needs():
    """Read from `capleaf/viXTTS` config.json. This is `config_inspected`
    evidence that the direction is EXPRESSIBLE — never that the fine-tune
    produces usable speech in it."""
    assert {"vi", "en", "zh", "ko"} <= set(XTTS_LANGS)


def test_declared_languages_are_not_an_output_tested_claim():
    """The registry separates evidence classes for a reason: viXTTS inherits 18
    languages from XTTS-v2 but was fine-tuned on Vietnamese, so declaring `ko`
    says nothing about Korean quality."""
    from onevoice.components import cloning_tts

    source = cloning_tts.__doc__ or ""
    assert "output_tested" in source
    assert "config_inspected" in (ViXttsTTS.declared_languages.__doc__ or source)


# --- the checkpoint anchor is the Hub commit, never a stand-in ---------------

COMMIT = "50228ccc563853f0ac628f49ed99a11f653d9ebe"


def _hub_cache(tmp_path):
    """The Hub cache layout: snapshot files are symlinks into `blobs/`."""
    repo = tmp_path / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    blob = repo / "blobs" / "5ae8293dd098"
    blob.write_bytes(b"weights")
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "model_last.pt").symlink_to(Path("../../blobs") / blob.name)
    return snapshot


def test_the_anchor_of_a_symlinked_snapshot_file_is_its_commit(tmp_path):
    """Following the symlink lands in `blobs/`, which names no commit; the
    first F5 MOS run recorded sha256 of the path string instead."""
    snapshot = _hub_cache(tmp_path)
    assert _commit_from_snapshot(snapshot / "model_last.pt") == COMMIT


def test_the_anchor_of_a_snapshot_directory_is_its_commit(tmp_path):
    assert _commit_from_snapshot(_hub_cache(tmp_path)) == COMMIT


def test_a_path_outside_a_snapshot_is_refused_not_hashed(tmp_path):
    weights = tmp_path / "model_last.pt"
    weights.write_bytes(b"weights")
    with pytest.raises(ProvenanceError, match="snapshot"):
        _commit_from_snapshot(weights)
