# Evaluation — asr_wav2vec2_lm_full200_pipeline

> **Development benchmark.** Tracks regressions and guides engineering. Not sufficient for final model selection: a difference here is not a ranking unless an interval says so. See `docs/INTERPRETATION.md`.

> **Known limitation (Q6).** `rtf` and `end_to_speech_ms` measure only
> work performed after the final speech frame, so neither captures
> per-frame work inside `push()`. `pipeline_rtf` spans the whole
> utterance and does capture it.

> Quote **`streaming_cost`**: it is the per-utterance difference between
> the two, which is what `rtf` under-reports. The `rtf` and `pipeline_rtf`
> percentiles are descriptive only and must not be subtracted from one
> another - when a detector fails to fire, that utterance has a
> `pipeline_rtf` but no `rtf`, so the two percentiles describe different
> populations.

> These are host-machine figures, not device predictions.

> **Measured (paired).** `rtf` hides a median of 0.0124 of audio duration per utterance (worst 0.0504), which is 6.5% of real processing cost (worst 12.5%). Paired over paired_n=200 utterances; 0 could not be paired.

| Metric | Value |
|---|---|
| bleu | 17.51 |
| ↳ bleu.tokenizer | 13a |
| ↳ bleu.signature | BLEU = 17.51 45.8/22.7/12.6/7.2 (BP = 1.000 ratio = 1.128 hyp_len = 5096 ref_len = 4516) |
| chrf | 50.59 |
| wer | 0.1388 |
| cer | 0.08486 |
| wer_normalized | 0.1373 |
| cer_normalized | 0.0838 |
| comet | 0.7545 |
| ↳ comet.model | Unbabel/wmt22-comet-da |
| end_to_speech_ms | 2275 |
| ↳ end_to_speech_ms.measured | 200 |
| ↳ end_to_speech_ms.unmeasurable | 0 |
| ↳ end_to_speech_ms.p50 | 2274.8567919479683 |
| ↳ end_to_speech_ms.p95 | 5794.465875020251 |
| ↳ end_to_speech_ms.max | 11516.48179104086 |
| ↳ end_to_speech_ms.mean | 2644.079170192126 |
| rtf | 0.1825 |
| ↳ rtf.measured | 200 |
| ↳ rtf.unmeasurable | 0 |
| ↳ rtf.p50 | 0.18253230963814895 |
| ↳ rtf.p95 | 0.3105943198010665 |
| ↳ rtf.max | 0.6641923346609896 |
| ↳ rtf.mean | 0.19574960177034678 |
| pipeline_rtf | 0.196 |
| ↳ pipeline_rtf.measured | 200 |
| ↳ pipeline_rtf.unmeasurable | 0 |
| ↳ pipeline_rtf.p50 | 0.19603185594178768 |
| ↳ pipeline_rtf.p95 | 0.32674617762876373 |
| ↳ pipeline_rtf.max | 0.6748382425010355 |
| ↳ pipeline_rtf.mean | 0.20910964815885777 |
| streaming_cost | 0.01244 |
| ↳ streaming_cost.paired_n | 200 |
| ↳ streaming_cost.unpairable | 0 |
| ↳ streaming_cost.p50 | 0.012437261198622873 |
| ↳ streaming_cost.p95 | 0.018294517889109074 |
| ↳ streaming_cost.max | 0.05035392327414351 |
| ↳ streaming_cost.p50_pct_of_pipeline | 6.462532117668858 |
| ↳ streaming_cost.max_pct_of_pipeline | 12.54820433548482 |

| Count | Utterances |
|---|---|
| total | 200 |
| scored | 200 |
| errored | 0 |
| empty_outputs | 0 |
| references | 200 |

> **Known limitation (Q6).** `rtf` and `end_to_speech_ms` measure only work
> performed after the final speech frame, so neither captures per-frame work
> inside `push()`. `pipeline_rtf` spans the whole utterance and does capture it.
>
> Quote **`streaming_cost`**: it is the per-utterance difference between the two,
> which is what `rtf` under-reports. The `rtf` and `pipeline_rtf` percentiles are
> descriptive only and must not be subtracted from one another - when a detector
> fails to fire, that utterance has a `pipeline_rtf` but no `rtf`, so the two
> percentiles describe different populations.
>
> These are host-machine figures, not device predictions.
