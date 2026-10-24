"""Corpus manifest builder — boundary methodology and reproducibility (D38, D39)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onevoice.corpus import (
    fleurs_config,
    read_manifest_meta,
    select_ids,
)


def _datasets_available() -> bool:
    try:
        import datasets  # noqa: F401
    except Exception:
        return False
    return True


requires_corpus = pytest.mark.skipif(
    not _datasets_available(), reason="corpus extra not installed"
)


# --- structural: run in CI with nothing installed ----------------------------


def test_module_imports_without_datasets():
    import onevoice.corpus  # noqa: F401


def test_language_configs_cover_the_competition_languages():
    assert fleurs_config("vi") == "vi_vn"
    assert fleurs_config("en") == "en_us"
    assert fleurs_config("zh") == "cmn_hans_cn"
    assert fleurs_config("ko") == "ko_kr"


def test_unmapped_language_raises_rather_than_guessing():
    # A wrong config silently benchmarks the wrong language.
    with pytest.raises(ValueError, match="fr"):
        fleurs_config("fr")


def test_building_without_the_extra_raises_a_clear_error(tmp_path):
    from onevoice import corpus

    if _datasets_available():
        pytest.skip("corpus extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra corpus"):
        corpus.build_refs(tmp_path)


# --- subset determinism ------------------------------------------------------


def test_subset_selection_is_deterministic():
    available = [f"{i:04d}" for i in range(200)]
    first = select_ids(available, limit=50, seed=0)
    second = select_ids(available, limit=50, seed=0)
    assert first == second
    assert len(first) == 50


def test_a_different_seed_selects_a_different_subset():
    available = [f"{i:04d}" for i in range(200)]
    assert select_ids(available, 50, seed=0) != select_ids(available, 50, seed=1)


def test_selection_is_sorted_so_manifests_are_stable():
    available = [f"{i:04d}" for i in range(200)]
    chosen = select_ids(available, 50, seed=0)
    assert chosen == sorted(chosen)


def test_no_limit_takes_everything():
    available = [f"{i:04d}" for i in range(20)]
    assert select_ids(available, None, seed=0) == sorted(available)


def test_a_limit_beyond_the_split_takes_everything():
    available = [f"{i:04d}" for i in range(20)]
    assert len(select_ids(available, 999, seed=0)) == 20


# --- sidecar -----------------------------------------------------------------


def test_missing_sidecar_reports_unknown_rather_than_assuming(tmp_path):
    """A caller must always be able to ask a manifest what its boundaries mean.

    Silently defaulting to `declared` would let a file_extent manifest be
    scored for VAD accuracy, which D38 forbids.
    """
    refs = tmp_path / "refs.jsonl"
    refs.write_text("")
    meta = read_manifest_meta(refs)
    assert meta["boundary_source"] == "unknown"
    assert meta["d10_assertion"] == "unknown"


def test_sidecar_is_read_when_present(tmp_path):
    refs = tmp_path / "refs.jsonl"
    refs.write_text("")
    (tmp_path / "refs.meta.json").write_text(
        json.dumps(
            {
                "boundary_source": "file_extent",
                "d10_assertion": "not_applicable",
                "utterance_ids": ["a", "b"],
            }
        )
    )
    meta = read_manifest_meta(refs)
    assert meta["boundary_source"] == "file_extent"
    assert meta["d10_assertion"] == "not_applicable"


def test_the_shipped_fixture_has_declared_boundaries():
    """The synthetic fixtures keep the D10 accuracy check alive.

    If this ever became file_extent, the Phase 3 gate's boundary assertion
    would silently stop measuring anything.
    """
    meta = read_manifest_meta("fixtures/mini_vi_en/refs.jsonl")
    assert meta["boundary_source"] == "declared"
    assert meta["d10_assertion"] == "valid"


# --- real corpus -------------------------------------------------------------


@requires_corpus
@pytest.mark.slow
def test_builds_a_manifest_from_fleurs(tmp_path):
    from onevoice.corpus import build_refs

    built = build_refs(tmp_path, src_lang="vi", tgt_lang="en", limit=3)

    assert built.refs_path.is_file()
    assert built.meta_path.is_file()
    assert len(built.utterance_ids) == 3

    rows = [json.loads(line) for line in built.refs_path.read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["src_lang"] == "vi"
        assert row["tgt_lang"] == "en"
        assert row["src_text"] and row["ref_text"]
        assert row["src_text"] != row["ref_text"]
        assert (tmp_path / row["audio_path"]).is_file()


@requires_corpus
@pytest.mark.slow
def test_sidecar_records_provenance_and_declares_the_boundary_source(tmp_path):
    from onevoice.corpus import build_refs

    built = build_refs(tmp_path, limit=3)
    meta = json.loads(built.meta_path.read_text())

    assert meta["corpus"] == "google/fleurs"
    assert meta["boundary_source"] == "file_extent"
    assert meta["d10_assertion"] == "not_applicable"
    assert meta["subset_seed"] == 0
    assert meta["utterance_ids"] == built.utterance_ids
    assert len(meta["framework_commit"]) >= 7
    assert "not speech boundaries" in meta["boundary_note"].lower()


@requires_corpus
@pytest.mark.slow
def test_rebuilding_with_the_same_seed_selects_the_same_utterances(tmp_path):
    from onevoice.corpus import build_refs

    first = build_refs(tmp_path / "a", limit=3, seed=7)
    second = build_refs(tmp_path / "b", limit=3, seed=7)
    assert first.utterance_ids == second.utterance_ids


@requires_corpus
@pytest.mark.slow
def test_audio_is_written_at_the_corpus_sample_rate(tmp_path):
    import wave

    from onevoice.corpus import build_refs

    built = build_refs(tmp_path, limit=2)
    for wav in sorted(Path(built.audio_dir).glob("*.wav")):
        with wave.open(str(wav)) as handle:
            assert handle.getframerate() == 16000
            assert handle.getnchannels() == 1
