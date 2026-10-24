# Evaluation — final-vi-en-200-turbo-envit5

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

> **Measured (paired).** `rtf` hides a median of 0.0122 of audio duration per utterance (worst 0.2187), which is 1.1% of real processing cost (worst 5.0%). Paired over paired_n=200 utterances; 0 could not be paired.

| Metric | Value |
|---|---|
| bleu | 17.14 |
| ↳ bleu.tokenizer | 13a |
| ↳ bleu.signature | BLEU = 17.14 45.4/22.4/12.2/6.9 (BP = 1.000 ratio = 1.115 hyp_len = 5037 ref_len = 4516) |
| chrf | 50.41 |
| wer | 0.2312 |
| cer | 0.09361 |
| wer_normalized | 0.09806 |
| cer_normalized | 0.05167 |
| comet | 0.7602 |
| ↳ comet.model | Unbabel/wmt22-comet-da |
| end_to_speech_ms | 1.306e+04 |
| ↳ end_to_speech_ms.measured | 200 |
| ↳ end_to_speech_ms.unmeasurable | 0 |
| ↳ end_to_speech_ms.p50 | 13064.048000029288 |
| ↳ end_to_speech_ms.p95 | 34081.17604104336 |
| ↳ end_to_speech_ms.max | 64185.72500010487 |
| ↳ end_to_speech_ms.mean | 14377.728209371562 |
| rtf | 0.8645 |
| ↳ rtf.measured | 200 |
| ↳ rtf.unmeasurable | 0 |
| ↳ rtf.p50 | 0.8644685185031166 |
| ↳ rtf.p95 | 3.458381642764113 |
| ↳ rtf.max | 5.693892187501654 |
| ↳ rtf.mean | 1.1789748829635318 |
| pipeline_rtf | 0.8795 |
| ↳ pipeline_rtf.measured | 200 |
| ↳ pipeline_rtf.unmeasurable | 0 |
| ↳ pipeline_rtf.p50 | 0.8794629948012431 |
| ↳ pipeline_rtf.p95 | 3.5716114175939486 |
| ↳ pipeline_rtf.max | 5.760328579344787 |
| ↳ pipeline_rtf.mean | 1.1965586231709557 |
| streaming_cost | 0.01217 |
| ↳ streaming_cost.paired_n | 200 |
| ↳ streaming_cost.unpairable | 0 |
| ↳ streaming_cost.p50 | 0.012169179581966949 |
| ↳ streaming_cost.p95 | 0.056706741791761495 |
| ↳ streaming_cost.max | 0.21865181505828257 |
| ↳ streaming_cost.p50_pct_of_pipeline | 1.126773366023858 |
| ↳ streaming_cost.max_pct_of_pipeline | 4.960634980253888 |

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
