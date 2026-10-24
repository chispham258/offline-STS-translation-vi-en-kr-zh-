# Evaluation — asr_phowhisper_large_full200_pipeline

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

> **Measured (paired).** `rtf` hides a median of 0.0060 of audio duration per utterance (worst 0.0352), which is 0.6% of real processing cost (worst 3.1%). Paired over paired_n=200 utterances; 0 could not be paired.

| Metric | Value |
|---|---|
| bleu | 20.26 |
| ↳ bleu.tokenizer | 13a |
| ↳ bleu.signature | BLEU = 20.26 49.5/25.7/14.8/8.9 (BP = 1.000 ratio = 1.092 hyp_len = 4931 ref_len = 4516) |
| chrf | 53.42 |
| wer | 0.1214 |
| cer | 0.06975 |
| wer_normalized | 0.09124 |
| cer_normalized | 0.06136 |
| comet | 0.7907 |
| ↳ comet.model | Unbabel/wmt22-comet-da |
| end_to_speech_ms | 1.298e+04 |
| ↳ end_to_speech_ms.measured | 200 |
| ↳ end_to_speech_ms.unmeasurable | 0 |
| ↳ end_to_speech_ms.p50 | 12983.345916960388 |
| ↳ end_to_speech_ms.p95 | 20924.035832984373 |
| ↳ end_to_speech_ms.max | 31785.891792038456 |
| ↳ end_to_speech_ms.mean | 13750.515243750415 |
| rtf | 1.072 |
| ↳ rtf.measured | 200 |
| ↳ rtf.unmeasurable | 0 |
| ↳ rtf.p50 | 1.0721533622144956 |
| ↳ rtf.p95 | 1.5580380539569523 |
| ↳ rtf.max | 2.7708808288246627 |
| ↳ rtf.mean | 1.094667404526913 |
| pipeline_rtf | 1.078 |
| ↳ pipeline_rtf.measured | 200 |
| ↳ pipeline_rtf.unmeasurable | 0 |
| ↳ pipeline_rtf.p50 | 1.0784051636422445 |
| ↳ pipeline_rtf.p95 | 1.5639659239949122 |
| ↳ pipeline_rtf.max | 2.781125922134538 |
| ↳ pipeline_rtf.mean | 1.1012414349968958 |
| streaming_cost | 0.006036 |
| ↳ streaming_cost.paired_n | 200 |
| ↳ streaming_cost.unpairable | 0 |
| ↳ streaming_cost.p50 | 0.006036261788865183 |
| ↳ streaming_cost.p95 | 0.008653176232548132 |
| ↳ streaming_cost.max | 0.035237550348629076 |
| ↳ streaming_cost.p50_pct_of_pipeline | 0.5700781699452775 |
| ↳ streaming_cost.max_pct_of_pipeline | 3.0581588932494643 |

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
