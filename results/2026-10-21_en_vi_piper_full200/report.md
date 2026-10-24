# 200-utterance en→vi CPU arm — whisper-base + Piper vi via language router

**Date:** 2026-10-21
**Config:** `configs/full_en_vi_cpu_hybrid_async.yaml`
**ASR:** faster-whisper `base`, CT2 INT8
**MT:** `VietAI/envit5-translation`
**TTS:** `language_router` → Piper `vi_VN-vais1000-medium` (22.05 kHz)
**Corpus:** `corpora/fleurs_en_vi_200/refs.jsonl` (200 utterances)
**Raw log:** `run.log`
**Run manifest:** `runs/bench_200_cpu_piper/run_manifest.jsonl`

## Result

**200/200 `status=ok`**, 0 errors, 0 empty hypotheses, 0 silent WAVs.

| Metric | Value |
|---|---|
| BLEU | **26.60** |
| chrF | **48.42** |
| COMET | **0.7691** |
| WER (raw) | 0.3571 |
| WER (normalized) | **0.1753** |
| CER (raw) | 0.1337 |
| CER (normalized) | **0.0896** |
| end-to-speech | 2301.7 ms |
| RTF (p50) | **0.2384** |
| pipeline RTF (p50) | **0.2425** |
| pipeline RTF (mean) | 0.2549 |
| pipeline RTF (p90) | 0.329 |
| pipeline RTF (max) | 0.663 |
| streaming cost (paired p50) | 0.0034 |

`pipeline_rtf` is wall time from `stream_start` to `tts_done` divided by
source audio duration. The evaluator reports its p50; mean/p90/max are
computed directly from the same 200 records.

## Reading

- **The CPU arm meets the RTF target.** p50 0.2425, mean 0.2549, p90 0.329;
  one outlier at 0.663. The target is < 0.6.
- TTS is no longer the bottleneck: Piper vi synthesises in ~0.45 s per
  utterance. ASR/MT dominate the remaining latency.
- All 200 output WAVs are non-silent and 22.05 kHz; total output audio 1316.5 s.
- Text metrics use the fixed envit5 wrapper, which strips the model's `vi: `
  tag before the event reaches TTS/scoring. The earlier
  `2026-10-21-200-base-silence` run started before that fix and its manifest
  was stripped post hoc; its RTF was also measured while other CPU work was
  running, so its 0.4645 p50 is **not** a clean comparison for this run.
  This run is the CPU arm's clean measurement.

## What this does not claim

- **VieNeu Turbo quality/latency:** on this Mac CPU Turbo took 10.56 s for
  7.36 s of audio (pipeline RTF 1.116). It remains the NPU/quality arm
  (`configs/full_en_vi_vieneu_async_hybrid.yaml`), not the CPU arm.
- **vi→en/zh/ko quality:** the one-utterance runs used the correct Piper
  target voices, but Silero ended a 13.86 s Vietnamese utterance at 1.534 s,
  so ASR/MT quality for those directions is not measured (open D7/D8).
- **MOS / naturalness:** no MOS for Piper vi, Piper en/zh/ko, or the hybrid
  composition. The licences of the Piper voices also need review before a
  commercial release (`ko_KR-kss-medium` is KSS, commonly CC-BY-NC-SA).
