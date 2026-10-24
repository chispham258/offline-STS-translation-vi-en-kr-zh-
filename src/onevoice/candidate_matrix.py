"""The initial candidate matrix (D56).

**A floor, not a ceiling.** These are the minimum candidates needed to keep each
slot's matrix from being predetermined. Adding a materially different
architecture, or a stronger Vietnamese-capable option, is expected — append and
re-check Gate D.

**Registration is not selection.** Order is registration order and carries no
merit. `strengths`/`weaknesses` are expectations to be tested, not findings, and
several are recorded precisely so they can be falsified.

Deployment metadata is recorded and **never gating**: a candidate with no
Qualcomm artifact competes on equal terms (D54).
"""

from __future__ import annotations

from onevoice.candidates import (
    Candidate,
    CandidateRegistry,
    Coverage,
    DeploymentMeta,
    Evidence,
)

#: Directions our corpora can actually exercise today. **A property of our
#: substrate, not of any candidate.** FLEURS gives us Vietnamese *source* audio,
#: so every reverse direction (en->vi, zh->vi, ko->vi) is currently
#: unevaluable - a gap in what we built, recorded rather than hidden.
EVALUABLE_DIRECTIONS = frozenset({"vi->en", "vi->zh", "vi->ko"})

#: The competition's full requirement, for contrast with the line above.
REQUIRED_DIRECTIONS = frozenset({
    "vi->en", "en->vi", "vi->zh", "zh->vi", "vi->ko", "ko->vi",
})

#: Incumbent family per slot, used only to evaluate Gate D.
INCUMBENT_FAMILY = {
    "vad": "silero",
    "asr": "whisper",
    # NLLB's architecture class on the Hub is `m2m_100` -- NLLB and M2M-100 are
    # the SAME family, not two. Step 1 counted them separately; verified
    # metadata says otherwise, and Gate D only means something if `family` is a
    # checkable architecture rather than a brand.
    "mt": "m2m_100",
    "tts": "vits",
}

#: Every language and architecture claim below was read from Hub metadata on
#: this date rather than from prose. Six Step-1 entries were wrong.
_VERIFIED_ON = "2026-08-14"


def _hub(source: str, arch: str = "") -> Evidence:
    return Evidence(
        language_method="hub_metadata",
        language_source=source,
        architecture_source=arch or source,
        verified_on=_VERIFIED_ON,
    )


def build_matrix() -> CandidateRegistry:
    """The initial matrix. Extend here; Gate D is re-checked by test."""
    r = CandidateRegistry()

    # --- VAD -----------------------------------------------------------------
    r.add(Candidate(
        slot="vad", name="silero", family="silero",
        checkpoint="snakers4/silero-vad", license="MIT", interface="BaseVAD",
        strengths="incumbent (D43); halved WER vs EnergyVAD, P=1.000",
        weaknesses="boundary accuracy on real speech unmeasured (D38)",
        deployment=DeploymentMeta(runtime="torch jit", status="unevaluated"),
    ))
    r.add(Candidate(
        slot="vad", name="energy", family="threshold",
        checkpoint="onevoice/EnergyVAD", license="internal", interface="BaseVAD",
        strengths="only detector certified against declared boundaries (D43)",
        weaknesses="fixed threshold dropped 24% of one corpus (Task 9)",
    ))
    r.add(Candidate(
        slot="vad", name="webrtc", family="webrtc",
        checkpoint="wiseman/py-webrtcvad", license="BSD-3-Clause",
        interface="BaseVAD",
        strengths="tiny, no model download, widely deployed on edge",
        weaknesses="older GMM approach; expected weaker in noise",
    ))

    # --- ASR -----------------------------------------------------------------
    r.add(Candidate(
        slot="asr", name="whisper-large-v3-turbo", family="whisper",
        checkpoint="openai/whisper-large-v3-turbo",
        evidence=_hub("hf openai card; multilingual", "architecture:whisper"),
        params_m=808.9, languages=frozenset({"vi", "en", "zh", "ko"}),
        license="MIT", interface="BaseASR",
        strengths="quality reference (D51); COMET 0.7220 on FLEURS-200",
        weaknesses="808.9M; 200-token cap in the Qualcomm artifact",
        deployment=DeploymentMeta(
            deployment_reference="qualcomm/Whisper-Large-V3-Turbo v0.59.0",
            ai_hub_available=True,
            export_path="QNN context binary / precompiled QNN-ONNX",
            runtime="QAIRT 2.45", precision="float", status="unevaluated",
        ),
    ))
    r.add(Candidate(
        slot="asr", name="whisper-small", family="whisper",
        checkpoint="openai/whisper-small", params_m=241.7,
        languages=frozenset({"vi", "en", "zh", "ko"}), license="MIT",
        interface="BaseASR",
        evidence=_hub("hf openai card; multilingual", "architecture:whisper"),
        strengths="size/latency anchor; prior reference (D27)",
        weaknesses="wer_normalized 0.2245 vs turbo 0.0908 (Phase 5.0)",
        deployment=DeploymentMeta(ai_hub_available=True, status="unevaluated"),
    ))
    r.add(Candidate(
        slot="asr", name="phowhisper-large", family="whisper",
        checkpoint="vinai/PhoWhisper-large",
        evidence=_hub("hf language:vi", "architecture:whisper"),
        languages=frozenset({"vi"}), license="BSD-3-Clause", interface="BaseASR",
        strengths="Vietnamese-specialised, 844h, SOTA on Vietnamese benchmarks",
        weaknesses=(
            "WHISPER FINE-TUNE: does NOT satisfy Gate D. Vietnamese-only, so it "
            "cannot serve zh/ko source directions"
        ),
        deployment=DeploymentMeta(ai_hub_available=False, status="unevaluated"),
    ))
    r.add(Candidate(
        slot="asr", name="wav2vec2-vi-250h", family="wav2vec2",
        checkpoint="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        evidence=_hub(
            "hf language:vi", "architecture:wav2vec2 class:AutoModelForCTC"
        ),
        languages=frozenset({"vi"}), license="CC-BY-NC-4.0", interface="BaseASR",
        strengths=(
            "CTC: no autoregressive decode, so no decode-length cap and no "
            "per-token latency term - both on the competition latency path. "
            "Far smaller than Whisper"
        ),
        weaknesses="Vietnamese-only; CTC output has no punctuation or casing",
        deployment=DeploymentMeta(ai_hub_available=False, status="unevaluated"),
    ))
    r.add(Candidate(
        slot="asr", name="seamless-m4t-v2-large", family="seamless_m4t_v2",
        checkpoint="facebook/seamless-m4t-v2-large", params_m=2309.2,
        evidence=_hub(
            "hf language: includes vi,zh,ko", "architecture:seamless_m4t_v2"
        ),
        languages=frozenset({"vi", "en", "zh", "ko"}), license="CC-BY-NC-4.0",
        interface="BaseASR",
        strengths="third family; covers all four languages; can also do MT",
        weaknesses="large; NC licence recorded but non-gating per D22",
        deployment=DeploymentMeta(ai_hub_available=False, status="unevaluated"),
    ))

    # --- MT ------------------------------------------------------------------
    r.add(Candidate(
        slot="mt", name="nllb-200-distilled-600M", family="m2m_100",
        checkpoint="facebook/nllb-200-distilled-600M", params_m=600.0,
        evidence=_hub(
            "hf language: includes vi,zh,ko",
            "architecture:m2m_100 - SAME family as m2m100",
        ),
        directions=frozenset({
            "vi->en", "en->vi", "vi->zh", "zh->vi", "vi->ko", "ko->vi",
        }),
        license="CC-BY-NC-4.0", interface="BaseMT",
        strengths="baseline; covers all six directions in one model",
        weaknesses="NOT a commitment - no MT model selected on adequate evidence (D48)",
        coverage=Coverage(claimed=frozenset({
            "vi->en", "en->vi", "vi->zh", "zh->vi", "vi->ko", "ko->vi",
        })),
    ))
    r.add(Candidate(
        slot="mt", name="opus-mt-vi-en", family="marian",
        checkpoint="Helsinki-NLP/opus-mt-vi-en",
        directions=frozenset({"vi->en"}), license="Apache-2.0", interface="BaseMT",
        strengths="small and fast; integrated",
        weaknesses="bilingual; zh/ko need PivotMT, whose compounding is unmeasured",
        coverage=Coverage(claimed=frozenset({"vi->en"})),
    ))
    r.add(Candidate(
        slot="mt", name="m2m100-418M", family="m2m_100",
        checkpoint="facebook/m2m100_418M", params_m=418.0,
        directions=frozenset({"vi->en", "en->vi", "vi->zh", "vi->ko"}),
        license="MIT", interface="BaseMT",
        strengths="one model, all directions; integrated",
        weaknesses="indistinguishable from opus at n=50 (D48)",
        coverage=Coverage(claimed=frozenset({
            "vi->en", "en->vi", "vi->zh", "vi->ko",
        })),
    ))
    r.add(Candidate(
        slot="mt", name="envit5-translation", family="t5",
        checkpoint="VietAI/envit5-translation",
        evidence=_hub("hf language:vi,en", "architecture:t5"),
        directions=frozenset({"vi->en", "en->vi"}), license="OpenRAIL",
        interface="BaseMT",
        strengths="T5 family - satisfies Gate D. Vietnamese-specialised",
        weaknesses="en<->vi only; no zh/ko",
        coverage=Coverage(claimed=frozenset({"vi->en", "en->vi"})),
    ))
    r.add(Candidate(
        slot="mt", name="madlad400-3b-mt", family="t5",
        checkpoint="google/madlad400-3b-mt", params_m=2940.4,
        evidence=_hub(
            "hf language: includes vi,zh,ko",
            "architecture:t5 - SAME family as envit5",
        ),
        directions=frozenset({"vi->en", "en->vi", "vi->zh", "vi->ko"}),
        license="Apache-2.0", interface="BaseMT",
        strengths="fourth family; 400+ languages",
        weaknesses="3B - almost certainly too large for the target; Track B will say",
        coverage=Coverage(claimed=frozenset({
            "vi->en", "en->vi", "vi->zh", "vi->ko",
        })),
    ))

    # --- TTS -- every candidate must speak Vietnamese (G1, exclusionary) -----
    r.add(Candidate(
        slot="tts", name="piper-vi-vais1000", family="vits",
        checkpoint="rhasspy/piper-voices:vi_VN-vais1000-medium",
        evidence=Evidence(
            language_method="output_tested",
            language_source=(
                "voice id vi_VN-*; synthesised in Phase 4 TTS benchmark and "
                "archived (D31). Strongest evidence in the matrix: we ran it"
            ),
            architecture_source=(
                "VITS via piper ONNX; not a Hub transformers repo"
            ),
            verified_on=_VERIFIED_ON,
        ),
        languages=frozenset({"vi"}), license="MIT", interface="BaseTTS",
        strengths="incumbent (D23); 241ms p50, RTF 0.034; ONNX already",
        weaknesses="naturalness NEVER measured - MOS outstanding (Q4)",
        deployment=DeploymentMeta(
            deployment_reference="AI Hub Piper is EN/DE/IT only - NOT this voice",
            ai_hub_available=False, export_path="ONNX (native)", status="unevaluated",
        ),
    ))
    r.add(Candidate(
        slot="tts", name="mms-tts-vie", family="vits",
        checkpoint="facebook/mms-tts-vie", params_m=36.3, languages=frozenset({"vi"}),
        evidence=Evidence(
            language_method="model_name",
            language_source=(
                "hf:facebook/mms-tts-vie carries NO language: tag. Vietnamese "
                "is inferred from the ISO-639-3 suffix 'vie' plus the MMS paper"
            ),
            architecture_source="architecture:vits (hub metadata)",
            verified_on=_VERIFIED_ON,
        ),
        license="CC-BY-NC-4.0", interface="BaseTTS",
        strengths="integrated; Vietnamese",
        weaknesses="2904ms p50 vs Piper 241ms; naturalness unmeasured",
    ))
    r.add(Candidate(
        slot="tts", name="vixtts", family="xtts",
        checkpoint="capleaf/viXTTS", languages=frozenset({"vi"}),
        evidence=_hub("hf language:vi", "card:XTTS-v2 fine-tune on viVoice"),
        license="other (see card)", interface="BaseTTS",
        strengths="XTTS-v2 family - satisfies Gate D. Voice cloning; viVoice fine-tune",
        weaknesses="autoregressive - expected far slower; large",
    ))
    r.add(Candidate(
        slot="tts", name="f5-tts-vietnamese-vivoice", family="f5",
        checkpoint="hynt/F5-TTS-Vietnamese-ViVoice", languages=frozenset({"vi"}),
        evidence=_hub("hf language:vi", "card:F5-TTS flow-matching DiT"),
        license="CC-BY-NC-SA-4.0", interface="BaseTTS",
        strengths="flow-matching DiT - third family; 1000h Vietnamese",
        weaknesses="diffusion-style sampling; latency unknown",
    ))

    return r


def coverage_report(registry: CandidateRegistry, slot: str) -> list[dict]:
    """Per-candidate coverage, with our substrate's gaps attributed to us.

    Deliberately returns capability facts only — no score, no rank, no
    judgement about whether breadth is good (D59). A candidate covering one
    direction is not thereby worse than one covering six; they answer different
    questions, and Track A answers the quality one.
    """
    rows = []
    for c in registry.for_slot(slot):
        cov = c.coverage.evaluable_within(EVALUABLE_DIRECTIONS)
        rows.append({
            "name": c.name,
            "family": c.family,
            "claimed": sorted(cov.claimed),
            "evaluable_here": sorted(cov.evaluable),
            "claimed_but_not_evaluable": sorted(cov.unevaluable),
            "missing_from_required": sorted(REQUIRED_DIRECTIONS - cov.claimed),
        })
    return rows
