# RESULTS — run parameters and measurements

Record of benchmarks used by the project README. Every number below was measured on
**Apple M1 (arm64), macOS, Python 3.11.14, CPU** — not Snapdragon, and not a
device prediction. Dates are next to each table.

Conventions:

- **Bold** = module selected as main / deployment candidate.
- `RTF` = time after the final speech frame / audio duration; it **misses** work
  inside `push()`.
- `pipeline_rtf` = `stream_start` → `tts_done` / audio duration; this is the
  latency gate.
- `WER` / `CER` are **normalized** (punctuation, case, whitespace).
- `COMET` = `Unbabel/wmt22-comet-da`. `BLEU` uses tokenizer `13a` for vi/en.
- A point estimate is not a conclusion. A "selection" needs a paired bootstrap
  CI or an explicitly recorded owner decision.

Scope note: the framework targets the fixed set `L = {vi, en, zh, ko}` and any
`src != tgt` pair, but the tables below only cover the pairs actually measured.
Other pairs are configurable, not validated.

---

## 0. Shared configuration

| Item | Value |
|---|---|
| Corpus vi→en | FLEURS 200 utterances, seed 0 (`corpora/fleurs_vi_en_200/refs.jsonl`) |
| Screen 22 | every 10th row of the 200 set + 1660, 1805, 1669 |
| VAD | Silero, `threshold=0.5`, `min_silence_duration_ms=700`, `speech_pad_ms=30` |
| Async queues | `vad_asr=128`, `asr_mt=64`, `mt_tts=64` |
| MT vi→en | `VietAI/envit5-translation`, prefix `vi: `, direct route |
| TTS en | Piper `en_US-lessac-medium`, 22.05 kHz |
| TTS vi | Piper `vi_VN-vais1000-medium` (baseline); VieNeu-TTS v3 Turbo, voice `Mai Anh` (owner decision) |
| Deployment config | `configs/full_vi_en_hybrid_async.yaml` |
| Runner | `onevoice.benchmark.run_benchmark` or `run_benchmark_async`; manifest + report per run |

---

## 1. Full 200 vi→en — complete pipeline

2026-10-23. Same VAD, MT and TTS; only ASR changes.

| ASR | WER_norm | CER_norm | BLEU | chrF | COMET | pipeline RTF p50 | p95 | Errors / empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **wav2vec2-250h + 4-gram KenLM** | 0.1373 | 0.0838 | 17.51 | 50.59 | 0.7545 | **0.1960** | **0.3267** | 0 / 0 |
| Whisper-large-v3-turbo (quality ref) | **0.0981** | **0.0517** | 17.14 | 50.41 | 0.7602 | 0.8795 | 3.5716 | 0 / 0 |
| PhoWhisper-small CT2 INT8 | 0.1603 | 0.1117 | 16.09 | 49.88 | 0.7485 | 0.6300 | 1.8784 | 0 / 0 |
| **PhoWhisper-large CT2 INT8** (quality leader) | **0.0912** | **0.0614** | **20.26** | **53.42** | **0.7907** | 1.0780 | — | 0 / 0 |

Reading:

- **wav2vec2 + LM** is the only arm meeting WER < 0.15, COMET > 0.75 and
  pipeline RTF < 0.60 simultaneously. It misses BLEU > 20. Deployment candidate
  (D80).
- **PhoWhisper-large** is the first full-200 arm with BLEU > 20 and the highest
  COMET, but RTF 1.078 means it cannot be deployed on this host. The absolute
  BLEU CI `[18.34, 22.30]` still contains 20, so the > 20 target is **not
  statistically resolved**.
- Turbo remains the quality reference/fallback: best WER and CER in the
  deployable-speed group, but 4.5× the RTF of wav2vec2.
- No paired bootstrap was computed for this table, so it is not a ranking.

**ASR parameters:**

| Arm | Model / artifact | Decode |
|---|---|---|
| **wav2vec2 + LM** | `nguyenvulebinh/wav2vec2-base-vietnamese-250h`; KenLM 4-gram 1.36 GB, SHA-256 `d76758da…`, CC-BY-NC-4.0 | `pyctcdecode`, `beam_width=100`, `alpha=0.5`, `beta=1.5` |
| turbo | `models/whisper-large-v3-turbo-ct2-int8` | faster-whisper CT2 INT8, greedy defaults |
| PhoWhisper-small | `models/phowhisper-small-ct2-int8` | faster-whisper CT2 INT8 |
| **PhoWhisper-large** | `vinai/PhoWhisper-large` rev `b9136a44…`, CT2 INT8 1.5 GB, BSD-3-Clause | faster-whisper CT2 INT8, greedy defaults |

---

## 2. ASR checkpoint selection (Phase 5.0, n=200)

2026-08-06. Silero + Opus-MT + Piper held constant; only the Whisper checkpoint
changes. Different 200 draw from §1 (the draws are not nested).

| Arm | Params | WER_norm | BLEU | chrF | COMET | end-to-speech p50 (ms) | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny | 37.8M | 0.9811 | 0.96 | 21.43 | 0.3986 | 2646 | 0.219 |
| base | 72.6M | 0.4243 | 3.56 | 30.21 | 0.4716 | 3073 | 0.253 |
| small | 241.7M | 0.2245 | 8.87 | 39.05 | 0.5992 | 5210 | 0.418 |
| **turbo** | 808.9M | **0.0908** | **13.54** | **45.42** | **0.7220** | 11211 | 0.934 |

Paired bootstrap, 1000 resamples, Bonferroni 98.33%: turbo superior to small
(`-0.1338 [-0.1508, -0.1175]`). Turbo COMET 0.7220 reproduced bit-identically.

---

## 3. ASR track n=200 (2026-08-14) — no winner

Pre-registered, D60 hierarchy, paired bootstrap, Bonferroni 98.75%,
delta = 0.0175.

| Challenger | WER_norm | Δ vs turbo | 98.75% CI | Verdict |
|---|---:|---:|---|---|
| whisper-small | 0.2245 | +0.1338 | [+0.1170, +0.1512] | INFERIOR |
| wav2vec2-vi-250h | 0.1880 | +0.0973 | [+0.0755, +0.1218] | INFERIOR |
| phowhisper-large | 0.0946 | +0.0038 | [-0.0128, +0.0213] | **UNRESOLVED** |
| seamless-m4t-v2 | 0.1063 | +0.0156 | [-0.0032, +0.0360] | **UNRESOLVED** |
| **turbo** | **0.0908** | reference | — | kept as working reference |

Conclusion: **D60 ASR selection remains UNRESOLVED**; the n=400 escalation was
not run. wav2vec2 + LM was later chosen under the deployment gate (D80), not
D60.

---

## 4. Screen 22 — ASR and MT (2026-10-23)

Same Silero 700 ms, envit5, Piper, async pipeline. Screen for fast ranking, not a
final quality claim.

### ASR

| Arm | WER_norm | CER_norm | BLEU | chrF | COMET | ASR RTF p50 | pipeline RTF p50 | p95 | Empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faster-whisper base | 0.4587 | 0.2581 | 4.073 | 30.77 | 0.4843 | 0.0878 | 0.2076 | 0.3225 | 1 |
| PhoWhisper-base | 0.2108 | 0.1526 | 14.47 | 46.84 | 0.7167 | 0.0697 | 0.1438 | 0.2373 | 0 |
| PhoWhisper-small | 0.1906 | 0.1329 | 14.85 | 50.49 | 0.7444 | 0.2361 | 0.3368 | 2.5162 | 0 |
| Whisper-small | 0.2884 | 0.1572 | 11.78 | 42.05 | 0.6394 | 0.2043 | 0.2972 | 0.4284 | 0 |
| turbo | 0.1484 | 0.0738 | 14.54 | 48.87 | 0.7388 | 0.4611 | 0.5594 | 0.7614 | 0 |
| **wav2vec2 + LM** | **0.1366** | 0.0758 | **14.55** | **49.23** | **0.7781** | **0.0550** | **0.1747** | **0.2275** | 0 |

### PhoWhisper-large / medium (screen 22, 2026-10-23)

| Arm | WER_norm | BLEU | chrF | COMET | pipeline RTF p50 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **PhoWhisper-large** | 0.1046 | 17.40 | 53.08 | 0.7796 | 1.7890 | **speed_fallback** |
| PhoWhisper-medium | 0.1720 | 18.39 | 53.20 | 0.7874 | 0.9204 | reject |
| D80 baseline | 0.1366 | 14.55 | 49.23 | 0.7781 | 0.1747 | reference |

Large passes BLEU/COMET/WER but fails RTF; medium fails WER (0.1720 > 0.15) and
RTF, with repeated-token hallucinations. The default config was not changed.

### MT (ASR = turbo)

| MT | WER_norm | BLEU | chrF | COMET | pipeline RTF p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| m2m100 418M | 0.1484 | 12.92 | 45.01 | 0.7026 | 0.7160 | 1.0903 |
| **envit5-translation** | 0.1484 | **14.54** | **48.87** | **0.7388** | **0.5594** | **0.7614** |

---

## 5. MT Vietnamese → English

### Gold source, n=200 (2026-10-23)

| MT | BLEU | chrF | COMET | MT p50 / p95 / max (ms) |
|---|---:|---:|---:|---:|
| **envit5-translation** | **23.86** | **57.11** | **0.8322** | 768.7 / 1301.1 / 1953.8 |

This is a **ceiling** with perfect ASR, not a pipeline result. Against D80
(BLEU 17.51, COMET 0.7545), ASR costs 6.35 BLEU and 0.0777 COMET; 98.6% of the
BLEU gap comes from the ASR model, 1.4% from VAD/streaming, 0% from integration
(D81).

### Other MT candidates (fixed 22, ASR source)

| MT | BLEU | chrF | COMET | MT wall p50 (ms) | Decision |
|---|---:|---:|---:|---:|---|
| **envit5** | 14.55 | **49.23** | **0.7781** | 709 | **keep** |
| Argos 1.9 int8 | **14.60** | 45.01 | 0.6903 | 330 | reject (COMET < 0.75) |
| Argos 1.9 float32 | 14.68 | 44.47 | 0.6878 | 784 | reject |

Argos gold-source: 19.77 BLEU / 52.55 chrF / 0.7837 COMET — below envit5
(23.98 BLEU / 58.11 chrF on gold-22; no COMET was archived for envit5 gold-22)
and less robust to ASR noise. Keep only as a latency fallback if the owner
relaxes the quality gate.

### Multilingual pairs

`madlad400-3b-mt` was selected for vi→zh and vi→ko (D65); en→vi uses envit5
(madlad errored and could not be scored — D68). zh→vi and ko→vi are blocked:
no corpus. These are MT-component selections, not end-to-end validation for all
pairs in `L`.

---

## 6. TTS

### Vietnamese text, n=50 (2026-08-04, CPU)

| Backend | p50 / p95 / max (ms) | Synthesis RTF p50 | Audio p50 (s) |
|---|---:|---:|---:|
| **Piper** `vi_VN-vais1000-medium` | **241 / 455 / 538** | **0.034** | 6.46 |
| MMS `mms-tts-vie` | 2904 / 6867 / 8844 | 0.293 | 10.51 |

Piper is 12× faster at p50; its synthesis RTF is 0.034 against 0.293 for MMS.
There is no MOS, so this table says nothing about naturalness.

### VieNeu-TTS v3 Turbo — owner decision (D79)

| Item | Value |
|---|---|
| Model | `pnnbao-ump/VieNeu-TTS-v3-Turbo` @ `61b85e3d`, Apache-2.0 |
| Runtime | `vieneu` 3.8.3, ONNX/CPU fp32, 48 kHz, `onnxruntime<1.23` |
| Voice | `Mai Anh` (female, Northern, news style) |
| Smoke n=5 | load 8.7 s; RTF 0.78–1.26 on M1; WER 0.011 / CER 0.005 (Whisper-turbo round trip) |
| Presets | 15 Northern voices, CER 0.000–0.010 on n=5 → not enough to rank |
| MOS | **none** |

Selected by the owner, **not** by MOS or a benchmark win. The n=5 CER/RTF smoke
is intelligibility evidence only, not naturalness evidence.

### MOS (Phase 6C) — no valid result

- 200 stimuli prepared, loudness -26 LUFS, protocol frozen (D69–D72).
- Pilot 80 sessions (2026-10-19) rated by **one person**: compressed ratings
  (2–3), natural speech 2.59 < f5 2.98 < mms 2.68; anchor separation 0.41 vs a
  gate of 1.5. 78/80 sessions excluded by the anchor rule →
  `insufficient_raters`.
- D78: the owner accepts a single-rater panel as the official result, but the
  single-rater analysis path was never built; **no MOS has been published**.
- D79 paused the MOS track. Any future report must say "single rater".

---

## 7. VAD — why Silero (n=50, 2026-08-04)

| | EnergyVAD | **SileroVAD** |
|---|---:|---:|
| BLEU / chrF | 5.261 / 24.05 | **9.459 / 39.32** |
| WER / CER | 0.6094 / 0.4666 | **0.2929 / 0.1336** |
| Utterances with no output | 12 | **0** |
| end-to-speech p50 (ms) | 4774 | **3962** |
| Latency measured | 38 / 50 | **50 / 50** |

On the 38 utterances EnergyVAD did keep, Silero is still better: WER 0.2968 vs
0.4840, BLEU 10.38 vs 6.00. Measured lookahead 62 ms against the D42 500 ms
window → 8× headroom.

---

## 8. en→vi full 200 (2026-10-21)

Whisper-base + envit5; different pipeline from vi→en.

| TTS | BLEU | chrF | COMET | WER_norm | pipeline RTF p50 |
|---|---:|---:|---:|---:|---:|
| SilenceTTS (isolates ASR/MT) | 26.44 | 48.29 | 0.7692 | 0.1805 | 0.4645* |
| **Piper vi** | **26.60** | **48.42** | **0.7691** | **0.1753** | **0.2425** |

*The SilenceTTS run was measured while other CPU work was running, so its RTF is
not a clean comparison. The Piper vi run has p90 0.329, max 0.663; p50 is under
the 0.6 gate.

---

## 9. Quality-first gates

| Gate | Threshold | D80 wav2vec2+LM | PhoWhisper-large |
|---|---:|---:|---:|
| WER_norm | < 0.15 | **0.1373 PASS** | **0.0912 PASS** |
| BLEU | > 20 | 17.51 FAIL | **20.26 PASS\*** |
| COMET | > 0.75 | **0.7545 PASS** | **0.7907 PASS** |
| pipeline RTF p50 | < 0.60 | **0.1960 PASS** | 1.0780 FAIL |
| errors / empty | 0 / 0 | **0 / 0 PASS** | **0 / 0 PASS** |

\* Absolute BLEU CI `[18.34, 22.30]` contains 20 → not resolved.

Deployment conclusion: **D80 wav2vec2 + LM**. **PhoWhisper-large** is the quality
leader but speed-blocked. The next path is backend/device work (ONNX Runtime,
whisper.cpp/GGML, QNN/NPU), not another model family (D84).

---

## 10. Known limitation (Q6)

> **Known limitation (Q6).** `rtf` and `end_to_speech_ms` measure only work
> performed after the final speech frame, so neither captures per-frame work
> inside `push()`. `pipeline_rtf` spans the whole utterance and does capture it.
>
> Quote **`streaming_cost`**: it is the per-utterance difference between the two,
> which is what `rtf` under-reports. The `rtf` and `pipeline_rtf` percentiles are
> descriptive only and must not be subtracted from one another — when a detector
> fails to fire, that utterance has a `pipeline_rtf` but no `rtf`, so the two
> percentiles describe different populations.
>
> These are host-machine figures, not device predictions.

---

## 11. Sources and limits

- Full decision ledger D1–D85: `docs/KNOWLEDGE.md`.
- Pipeline stages and async mechanism: `docs/ARCHITECTURE.md`.
- Run archive (config + provenance + manifest + report): `results/` — see
  `results/README.md`.
- Design spec, detailed investigations and the old per-run benchmark reports are
  **not** in this checkout. Their numbers are preserved in this file and in
  `results/`; paths inside `KNOWLEDGE.md` are historical references.
- No Snapdragon device: every RTF is host M1, not a device prediction.
- No valid MOS: no naturalness conclusion for any TTS.
- D60 ASR selection is UNRESOLVED; D80 is a deployment decision, not a selection.
- Only pairs marked in [`README.md` § Language coverage](../README.md#language-coverage)
  have measured end-to-end evidence.
