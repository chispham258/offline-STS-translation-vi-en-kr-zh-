"""The candidate registry and Gate D (D56, D57).

Registration is not selection. This registry records what a candidate *is* —
family, coverage, interface fit, deployment metadata — and deliberately holds no
ranking and no measurement. Scores live in run manifests, which are already
provenanced; copying them here would create a second source of truth that can
drift from the archives.
"""

from __future__ import annotations

import pytest

from onevoice.candidates import (
    Candidate,
    CandidateRegistry,
    DeploymentMeta,
    GateDError,
)


def _asr(name, family, langs=("vi",), **kw):
    return Candidate(
        slot="asr",
        name=name,
        family=family,
        checkpoint=f"org/{name}",
        languages=frozenset(langs),
        **kw,
    )


def _tts(name, family, out=("vi",), **kw):
    return Candidate(
        slot="tts",
        name=name,
        family=family,
        checkpoint=f"org/{name}",
        languages=frozenset(out),
        **kw,
    )


# --- registration is not selection ------------------------------------------


def test_a_candidate_records_identity_not_rank():
    c = _asr("turbo", "whisper")
    assert not hasattr(c, "score")
    assert not hasattr(c, "rank")
    assert not hasattr(c, "bleu")


def test_registry_preserves_insertion_order_and_does_not_sort_by_merit():
    r = CandidateRegistry()
    r.add(_asr("b", "whisper"))
    r.add(_asr("a", "wav2vec2"))
    assert [c.name for c in r.for_slot("asr")] == ["b", "a"]


def test_duplicate_registration_raises():
    # Same discipline as the component registry (D17): a silent overwrite
    # would mean the matrix does not contain what someone believes it does.
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    with pytest.raises(KeyError, match="turbo"):
        r.add(_asr("turbo", "whisper"))


def test_slots_are_independent():
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_tts("piper-vi", "vits"))
    assert [c.name for c in r.for_slot("asr")] == ["turbo"]
    assert [c.name for c in r.for_slot("tts")] == ["piper-vi"]


def test_unknown_slot_is_rejected():
    with pytest.raises(ValueError, match="punctuation"):
        Candidate(slot="punctuation", name="x", family="y", checkpoint="z")


# --- G1: Vietnamese capability is exclusionary, not a low score -------------


def test_a_tts_candidate_without_vietnamese_is_refused_registration():
    """Vietnamese speech output is a hard product requirement.

    Such a candidate is excluded from the matrix rather than scored poorly —
    scoring it would imply it could win.
    """
    with pytest.raises(ValueError, match="Vietnamese"):
        _tts("piper-en", "vits", out=("en", "de", "it"))


def test_an_asr_candidate_without_vietnamese_is_refused_registration():
    with pytest.raises(ValueError, match="Vietnamese"):
        _asr("whisper-en-only", "whisper", langs=("en",))


def test_vad_is_exempt_because_it_is_language_agnostic():
    Candidate(slot="vad", name="silero", family="silero", checkpoint="snakers4/x")


def test_mt_capability_is_checked_by_direction_not_language_set():
    with pytest.raises(ValueError, match="direction"):
        Candidate(slot="mt", name="en-de", family="marian", checkpoint="x",
                  directions=frozenset({"en->de"}))
    Candidate(slot="mt", name="vi-en", family="marian", checkpoint="x",
              directions=frozenset({"vi->en"}))


# --- Gate D (D56) ------------------------------------------------------------


def test_gate_d_fails_on_a_single_family():
    """Phase 5.0's matrix: four candidates, one family."""
    r = CandidateRegistry()
    for n in ("tiny", "base", "small", "turbo"):
        r.add(_asr(n, "whisper"))
    result = r.gate_d("asr", incumbent_family="whisper")
    assert not result.passes
    assert result.families == {"whisper"}
    assert "best within family" in result.explain()


def test_gate_d_fails_when_the_only_other_candidate_is_the_incumbent_family():
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_asr("phowhisper", "whisper"))
    assert not r.gate_d("asr", incumbent_family="whisper").passes


def test_gate_d_passes_with_a_distinct_family_outside_the_incumbent():
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_asr("w2v2-vi", "wav2vec2"))
    result = r.gate_d("asr", incumbent_family="whisper")
    assert result.passes
    assert result.families == {"whisper", "wav2vec2"}


def test_gate_d_is_evaluated_per_slot_independently():
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_asr("w2v2-vi", "wav2vec2"))
    r.add(_tts("piper-vi", "vits"))
    r.add(_tts("mms-vie", "vits"))
    assert r.gate_d("asr", "whisper").passes
    assert not r.gate_d("tts", "vits").passes  # both VITS


def test_a_finetune_does_not_count_as_a_distinct_family():
    """PhoWhisper is a Whisper fine-tune: it feels like diversity and is not."""
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_asr("phowhisper-large", "whisper"))
    r.add(_asr("w2v2-vi", "wav2vec2"))
    result = r.gate_d("asr", incumbent_family="whisper")
    assert result.passes
    # ...but it is the wav2vec2 entry carrying the gate, not PhoWhisper.
    assert result.outside_incumbent == {"wav2vec2"}


def test_asserting_a_selection_on_a_failing_matrix_raises():
    """The gate has to bite, not merely report."""
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_asr("small", "whisper"))
    with pytest.raises(GateDError, match="best within family"):
        r.require_gate_d("asr", incumbent_family="whisper")


def test_require_gate_d_is_silent_when_the_matrix_is_diverse():
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper"))
    r.add(_asr("w2v2-vi", "wav2vec2"))
    r.require_gate_d("asr", incumbent_family="whisper")


# --- deployment metadata is recorded, never gating --------------------------


def test_a_candidate_with_no_ai_hub_artifact_still_registers_and_counts():
    """A model must not lose because Qualcomm has not published an artifact."""
    r = CandidateRegistry()
    r.add(_asr("turbo", "whisper", deployment=DeploymentMeta(ai_hub_available=True)))
    r.add(
        _asr("w2v2-vi", "wav2vec2", deployment=DeploymentMeta(ai_hub_available=False))
    )
    result = r.gate_d("asr", incumbent_family="whisper")
    assert result.passes
    assert {c.name for c in r.for_slot("asr")} == {"turbo", "w2v2-vi"}


def test_deployment_status_defaults_to_unevaluated_not_ok():
    """'We have not checked' must never read as 'it works'."""
    assert DeploymentMeta().status == "unevaluated"


def test_deployment_status_must_be_a_known_value():
    with pytest.raises(ValueError, match="deployment status"):
        DeploymentMeta(status="probably fine")


def test_deployment_metadata_is_separate_from_model_identity():
    c = _asr("turbo", "whisper", deployment=DeploymentMeta(ai_hub_available=True))
    assert c.deployment.ai_hub_available is True
    # Identity fields carry no deployment information.
    assert "ai_hub" not in c.checkpoint
    assert not hasattr(c, "ai_hub_available")


# --- extensibility -----------------------------------------------------------


def test_the_minimum_matrix_is_a_floor_not_a_ceiling():
    r = CandidateRegistry()
    for n, f in (
        ("turbo", "whisper"), ("w2v2", "wav2vec2"), ("conformer", "conformer"),
    ):
        r.add(_asr(n, f))
    r.add(_asr("seamless", "seamless"))
    assert len(r.for_slot("asr")) == 4
    assert r.gate_d("asr", "whisper").passes


# --- the shipped matrix ------------------------------------------------------


def test_the_shipped_matrix_passes_gate_d_in_every_selectable_slot():
    """The whole point of D56: no slot's conclusion is predetermined.

    If this fails, some slot's experiment would produce a result that was
    available before it ran.
    """
    from onevoice.candidate_matrix import INCUMBENT_FAMILY, build_matrix

    r = build_matrix()
    for slot in ("asr", "mt", "tts"):
        result = r.gate_d(slot, INCUMBENT_FAMILY[slot])
        assert result.passes, result.explain()


def test_phowhisper_is_registered_but_does_not_carry_gate_d_for_asr():
    """It is Vietnamese-specialised and still a Whisper fine-tune."""
    from onevoice.candidate_matrix import build_matrix

    r = build_matrix()
    names = {c.name: c for c in r.for_slot("asr")}
    assert names["phowhisper-large"].family == "whisper"
    assert r.gate_d("asr", "whisper").outside_incumbent == {
        "wav2vec2",
        "seamless_m4t_v2",
    }


def test_every_tts_candidate_speaks_vietnamese():
    from onevoice.candidate_matrix import build_matrix

    for c in build_matrix().for_slot("tts"):
        assert "vi" in c.languages, c.name


def test_the_matrix_records_deployment_metadata_without_gating_on_it():
    """Candidates with no AI Hub artifact are present and count for Gate D."""
    from onevoice.candidate_matrix import build_matrix

    asr = {c.name: c for c in build_matrix().for_slot("asr")}
    assert asr["whisper-large-v3-turbo"].deployment.ai_hub_available is True
    assert asr["wav2vec2-vi-250h"].deployment.ai_hub_available is False
    # ...and it is the one WITHOUT an artifact that carries Gate D.
    assert asr["wav2vec2-vi-250h"].family == "wav2vec2"


def test_no_candidate_carries_a_score_or_rank():
    from onevoice.candidate_matrix import build_matrix

    r = build_matrix()
    for slot in ("vad", "asr", "mt", "tts"):
        for c in r.for_slot(slot):
            assert not any(hasattr(c, a) for a in ("score", "rank", "bleu", "wer"))


def test_every_deployment_status_starts_unevaluated():
    from onevoice.candidate_matrix import build_matrix

    r = build_matrix()
    for slot in ("vad", "asr", "mt", "tts"):
        for c in r.for_slot(slot):
            assert c.deployment.status == "unevaluated", c.name


# --- language-coverage evidence (D58) ---------------------------------------


def test_evidence_method_must_be_known():
    from onevoice.candidates import Evidence

    with pytest.raises(ValueError, match="evidence method"):
        Evidence(language_method="someone said so")


def test_a_model_name_is_weaker_evidence_than_a_hub_tag():
    from onevoice.candidates import Evidence

    assert not Evidence(language_method="model_name").is_verified
    assert not Evidence(language_method="model_card_prose").is_verified
    assert Evidence(language_method="hub_metadata").is_verified
    assert Evidence(language_method="output_tested").is_verified


def test_evidence_defaults_to_unverified():
    from onevoice.candidates import Evidence

    assert Evidence().language_method == "unverified"
    assert not Evidence().is_verified


def test_every_tts_candidate_records_how_vietnamese_was_established():
    """'Vietnamese supported' must be checkable, not assumed."""
    from onevoice.candidate_matrix import build_matrix

    for c in build_matrix().for_slot("tts"):
        assert c.evidence.language_method != "unverified", c.name
        assert c.evidence.language_source, c.name
        assert c.evidence.verified_on, c.name


def test_the_one_weakly_evidenced_tts_candidate_is_recorded_as_such():
    """facebook/mms-tts-vie carries no Hub language tag.

    Vietnamese is inferred from the checkpoint name. That is weaker than a
    structured tag and is recorded as such rather than rounded up.
    """
    from onevoice.candidate_matrix import build_matrix

    mms = {c.name: c for c in build_matrix().for_slot("tts")}["mms-tts-vie"]
    assert mms.evidence.language_method == "model_name"
    assert not mms.evidence.is_verified
    assert "NO language" in mms.evidence.language_source


def test_mt_families_reflect_verified_architecture_not_brand():
    """NLLB's Hub architecture is m2m_100, and MADLAD's is t5.

    Counting them as distinct brands inflated the MT family count from 3 to 5
    in Step 1. Gate D is only meaningful if family means a checkable thing.
    """
    from onevoice.candidate_matrix import build_matrix

    fams = {c.name: c.family for c in build_matrix().for_slot("mt")}
    assert fams["nllb-200-distilled-600M"] == fams["m2m100-418M"] == "m2m_100"
    assert fams["madlad400-3b-mt"] == fams["envit5-translation"] == "t5"


# --- coverage is a capability dimension, never a quality one (D59) ----------


def test_the_three_coverage_layers_are_distinct():
    from onevoice.candidates import Coverage

    c = Coverage(claimed=frozenset({"vi->en", "vi->zh"}))
    assert c.evaluable == frozenset()      # nothing evaluated yet
    assert c.deployable == frozenset()     # unevaluated, NOT "none"


def test_evaluable_is_a_property_of_our_corpus_not_of_the_model():
    from onevoice.candidates import Coverage

    broad = Coverage(claimed=frozenset({"vi->en", "vi->zh", "vi->ko"}))
    only_en = frozenset({"vi->en"})
    narrowed = broad.evaluable_within(only_en)
    assert narrowed.evaluable == {"vi->en"}
    # The claim is untouched: our gap is not the candidate's shortfall.
    assert narrowed.claimed == {"vi->en", "vi->zh", "vi->ko"}
    assert narrowed.unevaluable == {"vi->zh", "vi->ko"}


def test_a_narrow_candidate_is_not_penalised_by_the_coverage_model():
    """envit5 is en<->vi only. That is a capability fact, not a demerit."""
    from onevoice.candidates import Coverage

    narrow = Coverage(claimed=frozenset({"vi->en", "en->vi"}))
    corpus = frozenset({"vi->en", "vi->zh", "vi->ko"})
    result = narrow.evaluable_within(corpus)
    assert result.evaluable == {"vi->en"}
    # en->vi is claimed and NOT evaluable: our FLEURS corpora are vi-source
    # only, so every reverse direction is a gap in OUR substrate, not a
    # shortfall in the candidate. Recorded rather than silently dropped.
    assert result.unevaluable == {"en->vi"}


def test_coverage_carries_no_quality_information():
    from onevoice.candidates import Coverage

    c = Coverage(claimed=frozenset({"vi->en"}))
    for attr in ("score", "rank", "bleu", "comet", "quality"):
        assert not hasattr(c, attr)
