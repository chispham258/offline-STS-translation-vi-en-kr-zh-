# MT component screen — Argos Translate vi→en 1.9

> **Classification: fixed-22 screen / development evidence. Not a selection ranking.**
> n = 22; one sentence per utterance; no paired bootstrap CI. This run is enough
> to reject a candidate that misses a pre-registered gate, not enough to rank
> two candidates that are close.

## Scope

This is an **MT-only** probe. It does not run VAD, ASR or TTS. The source text is
the archived `asr_text` from the D80 fixed-22 pipeline run
(`runs/bakeoff_vi_en/asr_wav2vec2_lm_screen22/run.jsonl`), so it measures what
the MT stage receives in the cascade. The gold-source arm is a ceiling check.

The candidate is the official Argos Translate `translate-vi_en` 1.9 package,
run directly through CTranslate2 4.8.3. The `argostranslate` wrapper and its
sentence-boundary detector were bypassed: every screen row is one sentence.
This is not a full-pipeline run and not a Snapdragon measurement.

## Candidate and provenance

| Item | Value |
|---|---|
| Package | `translate-vi_en-1_9.argosmodel` |
| URL | `https://argos-net.com/v1/translate-vi_en-1_9.argosmodel` |
| Compressed / uncompressed | 65,895,840 B / 79,738,259 B |
| `model.bin` | 76,857,078 B; SHA-256 `63aa5bdb…fbcb96` |
| Package SHA-256 | `ebd51b71…4433953` |
| Model family | OPUS-MT vi→en, OpenNMT/CTranslate2; package 1.9, original model 1.0 |
| Code license | MIT |
| Model license | CC-BY 4.0; attribution required |
| Tokenizer | SentencePiece; package also ships a Stanza vi tokenizer |
| Native streaming | No; sentence/segment level |

## Method

- 22 utterances; references `corpora/fleurs_vi_en_200/refs_screen22.jsonl`.
- Source: D80 `asr_text` (cascade arm) or gold `src_text` (ceiling arm).
- CTranslate2 CPU, `beam_size=4`, `num_hypotheses=1`, `length_penalty=0.2`,
  `replace_unknowns=True`, `max_batch_size=32`, `inter_threads=1`,
  `intra_threads=4`, 3 warmup calls.
- Time is `translate_batch()` wall time only. It excludes tokenization and
  excludes VAD/ASR/TTS. It is not pipeline RTF.
- COMET uses `scripts/score_comet.py` and `Unbabel/wmt22-comet-da`; the script
  scores with the gold source text, matching the archived baseline reports.
- Repro script: `probe.py` in this directory.

## Measured values

### Cascade ASR-source arm (the decision arm)

| MT | BLEU | chrF | COMET | MT wall p50 |
|---|---:|---:|---:|---:|
| envit5 baseline, D80 | 14.55 | 49.23 | 0.7781 | 709.0 ms |
| Argos int8 | 14.60 | 45.01 | **0.6903** | **330.4 ms** |
| Argos float32 | 14.68 | 44.47 | 0.6878 | 784.4 ms |

Argos int8: 22/22 outputs, no empty/errored rows. Float32: 22/22.

### Gold-source ceiling arm

| MT | BLEU | chrF | COMET | MT wall p50 |
|---|---:|---:|---:|---:|
| envit5, archived gold22 | 23.98 | 58.11 | not recorded | 1280.3 ms |
| Argos int8, gold22 | 19.77 | 52.55 | 0.7837 | 306.8 ms |

Argos clean text is weaker than envit5 on the only archived gold-22 comparison,
and its cascade COMET is far below the 0.75 quality gate.

## Supported findings

1. **Argos int8 misses the quality gate on the cascade arm.** Observed COMET
   `0.6903 < 0.75`; observed chrF `45.01` vs baseline `49.23`. This is a gate
   failure, not a ranking claim.
2. **BLEU alone would have hidden the failure.** BLEU is `14.60` vs baseline
   `14.55`, while chrF and COMET are lower. The qualitative output contains
   semantic substitutions and malformed phrases on ASR-noisy input.
3. **The clean-source arm is better but still below envit5.** Argos gold22:
   BLEU `19.77`, chrF `52.55`, COMET `0.7837`; envit5 gold22: BLEU `23.98`,
   chrF `58.11`. No COMET was archived for envit5 gold22, so no COMET delta is
   claimed.
4. **Argos is much smaller and faster at the MT stage.** `63 MB` package,
   `76.9 MB` model; int8 MT p50 `330 ms` vs envit5 `709 ms` on this host.
   This is a stage observation, not a pipeline or device claim.
5. **The candidate is not robust enough for the ASR-noise cascade.** The
   observed clean→ASR COMET drop is `0.7837 → 0.6903`; the observed chrF drop
   is `52.55 → 45.01`. This is descriptive; no CI was computed.

## Decision

**Reject Argos as the primary vi→en MT for the quality-first configuration.**
Do not change `configs/full_vi_en_hybrid_async.yaml`; do not run full 200 for
this candidate; do not treat this as a D60 selection.

Keep it only as a **latency/fallback candidate** if the owner later accepts a
quality trade-off. The attractive properties are real: 63 MB package, MIT code,
CC-BY 4.0 model, direct vi→en (no pivot), CPU-only, and lower MT-stage latency.
They do not compensate for a cascade COMET below the gate.

Why envit5 remains the choice for this slot:

- it clears the cascade COMET gate where Argos does not;
- it has higher chrF on both the cascade and gold arms;
- it is already integrated and measured end to end;
- Argos adds a second MT implementation and an older OPUS-MT model without
  enough quality evidence to justify a swap.

## Open questions

- Would the official `argostranslate` path with Stanza/MiniSBD change the
  result? The 22 rows are single sentences, so the expected difference is small,
  but this probe did not test it.
- Does the result hold on the full 200? Not run; the fixed-22 gate failure makes
  full-200 unjustified unless the owner asks for a latency-mode study.
- Is a confidence-gated hybrid worth testing? It would need a calibrated
  confidence signal and a fallback to envit5; not part of the current phase.
- What is the device MT latency? Host M1 CPU only; not a Snapdragon prediction.

## Artifacts

All paths are relative to the repository root.

| Artifact | Purpose |
|---|---|
| `probe.py` | Reproducible direct-CTranslate2 probe |
| `argos_vi_en_int8_screen22.jsonl` | Cascade arm, int8, 22 rows |
| `argos_vi_en_float32_screen22.jsonl` | Cascade arm, float32, 22 rows |
| `argos_vi_en_gold22.jsonl` | Gold-source ceiling arm, int8 |
| `comet_int8.json` | COMET for the int8 cascade arm |
| `comet_float32.json` | COMET for the float32 cascade arm |
| `comet_gold22.json` | COMET for the gold-source arm |
| `metadata.json` | Hashes, source, environment, invocation, scores |
| `report.md` | This report |

Reproduction (int8 cascade arm):

```bash
uv run python runs/bakeoff_vi_en/mt_argos_screen22/probe.py \
  --package /path/to/translate-vi_en-1_9.argosmodel \
  --source runs/bakeoff_vi_en/asr_wav2vec2_lm_screen22/run.jsonl \
  --source-kind asr \
  --compute-type int8 \
  --out runs/bakeoff_vi_en/mt_argos_screen22/argos_vi_en_int8_screen22.jsonl
```

Score the archived int8 arm with the project's standard COMET script:

```bash
uv run python scripts/score_comet.py \
  runs/bakeoff_vi_en/mt_argos_screen22/argos_vi_en_int8_screen22.jsonl \
  corpora/fleurs_vi_en_200/refs_screen22.jsonl \
  runs/bakeoff_vi_en/mt_argos_screen22/comet_int8.json
```

## Caveats

- Host: Apple M1 CPU, macOS 15.7.1, Python 3.11.14; not a Snapdragon and not a
  device prediction.
- MT-only; no pipeline RTF, no VAD/ASR/TTS interaction.
- No paired bootstrap CI; fixed-22 screen, not a selection ranking.
- COMET is scored with the gold source text by the project's standard script;
  it is not an ASR-conditioned COMET.
- The package was run directly through CTranslate2; this bypasses Argos SBD and
  the `argostranslate` wrapper. All 22 rows are single sentences.
- Model package: code MIT; weights CC-BY 4.0; attribution is required if used.

## Decision record

`docs/KNOWLEDGE.md` D85.
