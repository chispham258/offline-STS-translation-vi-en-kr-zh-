"""F4: every production component satisfies the provenance contract (D46).

The conformance test enumerates the **registry** rather than a hand-written
list, so registering a component is what enrols it in the contract. A wrapper
added later cannot ship without provenance unless someone deliberately adds it
to the dummy allow-list, which is a visible act in review.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from onevoice.provenance import (
    IDENTITY_FIELDS,
    IMMUTABLE_ANCHORS,
    REQUIRED_FIELDS,
    ProvenanceError,
    params_hash,
    validate_provenance,
)

#: Test doubles identify nothing and produce no number anyone would reproduce.
#: Exempt by explicit name, never by "it has no provenance() so presumably it
#: is fine" - that reasoning is what let six production wrappers ship without.
DUMMY_BACKENDS = {"echo", "identity", "silence", "energy"}

#: Backends whose constructor requires arguments. Smallest checkpoints, because
#: this test constructs every one of them and identity does not depend on size.
CONSTRUCTOR_ARGS: dict[str, dict] = {
    "opus": {"model_name": "Helsinki-NLP/opus-mt-vi-en"},
    "m2m100": {
        "model_name": "facebook/m2m100_418M",
        "src_lang": "vi",
        "tgt_lang": "en",
    },
    "nllb": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "src_lang": "vi",
        "tgt_lang": "en",
    },
    # The cloning TTS backends REFUSE to construct without an explicit speaker
    # reference - that guard is the point of them, so it is not relaxed to suit
    # this test. The reference below is the one the TTS pre-registration freezes.
    "xtts": {
        "tgt_lang": "vi",
        "reference_audio": "corpora/fleurs_vi_en_200/audio/1664.wav",
    },
    "f5": {
        "tgt_lang": "vi",
        "reference_audio": "corpora/fleurs_vi_en_200/audio/1664.wav",
        "reference_text": (
            "vặn hai cục bột khô với nhau rồi bóp chúng thành một viên cầu "
            "bằng tay ướt sạch"
        ),
    },
    # A composite: its identity is both legs, so it is constructed with two.
    "pivot": {
        "first": {
            "model": "opus",
            "params": {"model_name": "Helsinki-NLP/opus-mt-vi-en"},
        },
        "second": {"model": "identity"},
    },
    # A language router is a composite too. Use the Vietnamese Piper voice:
    # in CI the missing tts-piper extra is a skip, and this avoids loading
    # VieNeu just to check provenance.
    "language_router": {
        "tgt_lang": "vi",
        "backends": {
            "vi": {
                "module": "onevoice.components.piper_tts",
                "model": "piper",
                "params": {"voice": "vi_VN-vais1000-medium"},
            },
        },
    },
}


def _valid(**overrides):
    record = {
        "component": "whisper",
        "implementation": "onevoice.components.whisper_asr.WhisperASR",
        "model_id": "openai/whisper-small",
        "revision": "default",
        "commit_hash": "abc123def456",
        "params_hash": "0" * 64,
        "invocation": {"src_lang": "vi"},
    }
    record.update(overrides)
    return record


# --- the validator itself ----------------------------------------------------


def test_a_complete_record_passes():
    validate_provenance(_valid())


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_field_is_actually_required(field):
    record = _valid()
    del record[field]
    with pytest.raises(ProvenanceError, match=field):
        validate_provenance(record)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_placeholders_do_not_satisfy_a_required_field(field):
    """'unknown' passes a presence check and reproduces nothing."""
    with pytest.raises(ProvenanceError, match="placeholder"):
        validate_provenance(_valid(**{field: "unknown"}))


def test_an_immutable_anchor_is_required():
    record = _valid()
    del record["commit_hash"]
    with pytest.raises(ProvenanceError, match="immutable anchor"):
        validate_provenance(record)


def test_either_anchor_satisfies_the_rule():
    """A Hub model has a commit; a file-based voice has only bytes."""
    hub = _valid()
    assert "commit_hash" in hub
    validate_provenance(hub)

    file_based = _valid()
    del file_based["commit_hash"]
    file_based["weight_hash"] = "f" * 64
    validate_provenance(file_based)


def test_an_unknown_anchor_does_not_count_as_an_anchor():
    with pytest.raises(ProvenanceError, match="immutable anchor"):
        validate_provenance(_valid(commit_hash="unknown"))


def test_revision_equal_to_commit_hash_is_rejected():
    """MMSTTS shipped exactly this, setting both to the resolved commit.

    It discards what the experiment asked for, which is the question that
    matters when a later run silently differs.
    """
    with pytest.raises(ProvenanceError, match="revision equals commit_hash"):
        validate_provenance(_valid(revision="abc123def456"))


def test_anchor_list_is_what_the_spec_says():
    assert set(IMMUTABLE_ANCHORS) == {"commit_hash", "weight_hash"}


# --- invocation identity ----------------------------------------------------


def test_invocation_is_required():
    """Identity says what ran; invocation says what it was asked to do.

    A run reproducible in implementation but not in request is not
    reproducible: the same Whisper checkpoint asked for a different source
    language produces a different transcript.
    """
    record = _valid()
    del record["invocation"]
    with pytest.raises(ProvenanceError, match="invocation"):
        validate_provenance(record)


def test_an_empty_invocation_is_rejected():
    # Every production component here is asked for something. A wrapper
    # claiming otherwise has usually forgotten a parameter.
    with pytest.raises(ProvenanceError, match="invocation"):
        validate_provenance(_valid(invocation={}))


def test_invocation_must_be_a_dict_not_a_string():
    with pytest.raises(ProvenanceError, match="invocation"):
        validate_provenance(_valid(invocation="vi->en"))


def test_identity_and_invocation_are_disjoint_concepts():
    """The split is structural, so a request parameter cannot pose as identity."""
    assert "invocation" not in IDENTITY_FIELDS
    assert set(IDENTITY_FIELDS) == set(REQUIRED_FIELDS)


# --- params_hash -------------------------------------------------------------


def test_params_hash_is_stable_and_order_independent():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_params_hash_distinguishes_different_configurations():
    assert params_hash({"model": "small"}) != params_hash({"model": "tiny"})


def test_params_hash_never_raises_on_an_awkward_value():
    """A provenance helper must not be able to break a benchmark."""
    assert params_hash({"device": object()})


# --- conformance over the registry -------------------------------------------


def _production_backends():
    """Every registered non-dummy backend, as (kind, name) pairs."""
    import importlib
    import pkgutil

    import onevoice.components as pkg
    from onevoice.registry import KINDS, registered_names

    # Import every component module so the registry is populated. Walking the
    # package rather than listing modules is deliberate: a new wrapper is
    # enrolled in this contract by existing, not by being remembered here.
    # The wrappers' heavy imports are inside __init__, so importing the module
    # is safe without any extra installed.
    for info in pkgutil.iter_modules(pkg.__path__):
        if not info.ispkg:
            importlib.import_module(f"{pkg.__name__}.{info.name}")

    # Dummies must be *registered* - PivotMT composes one as a no-op second leg
    # - but they are exempt from the contract, which is what DUMMY_BACKENDS
    # below expresses. Registration and conformance are separate questions.
    importlib.import_module("onevoice.components.dummy")

    return [
        (kind, name)
        for kind in KINDS
        for name in registered_names(kind)
        if name not in DUMMY_BACKENDS
    ]


def test_the_registry_exposes_production_backends_to_check():
    # If this ever returns nothing, the conformance test below is passing
    # vacuously and proving nothing.
    assert _production_backends()


@pytest.mark.slow
@pytest.mark.parametrize(
    "kind_name", _production_backends(), ids=lambda kn: f"{kn[0]}/{kn[1]}"
)
def test_production_components_declare_conforming_provenance(kind_name):
    """Constructing every real backend is slow; skipping it is not an option.

    A schema nothing is validated against is documentation, not a contract.
    """
    from onevoice.registry import resolve

    kind, name = kind_name
    cls = resolve(kind, name)
    # Corpus audio is gitignored (rebuilt from refs.meta.json), so a cloning
    # backend's speaker reference is absent on a fresh checkout such as CI.
    # Like a missing extra, that is a missing prerequisite, not a defect.
    reference = CONSTRUCTOR_ARGS.get(name, {}).get("reference_audio")
    if reference is not None and not Path(reference).is_file():
        pytest.skip(f"{name} speaker reference not present: {reference}")
    try:
        component = cls(**CONSTRUCTOR_ARGS.get(name, {}))
    except RuntimeError as exc:  # missing optional extra
        pytest.skip(f"{name} requires an extra that is not installed: {exc}")

    describe = getattr(component, "provenance", None)
    assert callable(describe), (
        f"{kind}/{name} has no provenance(). Every production component must "
        "declare one - see docs/PROVENANCE.md."
    )
    validate_provenance(describe(), label=f"{kind}/{name}")


# --- run identity (F6) -------------------------------------------------------


def test_run_identity_records_content_not_just_paths(tmp_path):
    """A filename identifies a run only until someone edits the file."""
    from onevoice.provenance import run_identity

    config = tmp_path / "c.yaml"
    config.write_text("vad:\n  model: energy\n")
    refs = tmp_path / "refs.jsonl"
    refs.write_text("{}\n")

    first = run_identity(config, refs)
    assert len(first["config_hash"]) == 64
    assert len(first["refs_hash"]) == 64

    config.write_text("vad:\n  model: silero\n")
    assert run_identity(config, refs)["config_hash"] != first["config_hash"]


def test_run_identity_is_stable_when_nothing_changes(tmp_path):
    from onevoice.provenance import run_identity

    config = tmp_path / "c.yaml"
    config.write_text("x: 1\n")
    refs = tmp_path / "r.jsonl"
    refs.write_text("{}\n")
    assert run_identity(config, refs)["config_hash"] == (
        run_identity(config, refs)["config_hash"]
    )


def test_a_missing_input_is_reported_not_silently_hashed(tmp_path):
    from onevoice.provenance import file_hash

    assert file_hash(tmp_path / "nope.yaml") == "missing"


def test_run_identity_validates(tmp_path):
    from onevoice.provenance import run_identity, validate_run_identity

    config = tmp_path / "c.yaml"
    config.write_text("x: 1\n")
    refs = tmp_path / "r.jsonl"
    refs.write_text("{}\n")
    validate_run_identity(run_identity(config, refs))


def test_a_missing_config_hash_fails_validation(tmp_path):
    from onevoice.provenance import run_identity, validate_run_identity

    refs = tmp_path / "r.jsonl"
    refs.write_text("{}\n")
    record = run_identity(tmp_path / "gone.yaml", refs)
    with pytest.raises(ProvenanceError, match="config_hash"):
        validate_run_identity(record)


def test_framework_dirty_is_recorded_as_a_bool():
    """A commit hash from a dirty tree names code that is not what ran.

    Recording the commit without the flag would claim reproducibility the run
    does not have.
    """
    from onevoice.provenance import framework_identity

    identity = framework_identity()
    assert isinstance(identity["framework_dirty"], bool)
    assert identity["framework_commit"]


def test_unknown_git_state_counts_as_dirty(monkeypatch):
    """Failure to determine cleanliness is not cleanliness."""
    from onevoice import provenance

    monkeypatch.setattr(provenance, "_git", lambda *a: None)
    assert provenance.framework_identity()["framework_dirty"] is True


def test_component_and_run_identity_stay_separate():
    """params_hash identifies one component; config_hash identifies the run.

    Merging them would let a component duplicate run-level configuration, or a
    run stand in for component provenance. Neither is recoverable later.
    """
    from onevoice.provenance import IDENTITY_FIELDS, RUN_IDENTITY_FIELDS

    assert "config_hash" not in IDENTITY_FIELDS
    assert "params_hash" not in RUN_IDENTITY_FIELDS
    assert not set(IDENTITY_FIELDS) & set(RUN_IDENTITY_FIELDS)


@pytest.mark.parametrize(
    "runner",
    ["benchmark", "mt_benchmark", "tts_benchmark"],
)
def test_every_runner_records_conforming_run_identity(tmp_path, runner):
    """F6 conformance, enumerated over runners rather than spot-checked.

    `mt_benchmark` wrote no metadata at all before this, so the MT ceiling
    figures Task 9's correction rests on had no provenance whatsoever.
    """
    import json

    import onevoice.components.dummy  # noqa: F401
    from onevoice.provenance import validate_run_identity

    refs = "fixtures/mini_vi_en/refs.jsonl"
    config = "configs/dummy_vi_en.yaml"
    out = tmp_path / runner

    if runner == "benchmark":
        from onevoice.benchmark import run_benchmark

        result = run_benchmark(config, refs, out, experiment="f6")
    elif runner == "mt_benchmark":
        from onevoice.mt_benchmark import run_mt_benchmark

        result = run_mt_benchmark(config, refs, out, experiment="f6")
    else:
        from onevoice.tts_benchmark import run_tts_benchmark

        result = run_tts_benchmark(config, refs, out, experiment="f6")

    meta = json.loads(result.metadata_path.read_text())
    validate_run_identity(meta)
    # The archived config is what makes the run re-runnable after the original
    # file changes; the hash only proves it changed.
    assert (out / "config.snapshot.yaml").is_file()
