# MT component benchmark - mt-envit5-gold200

> **Scope.** These figures score the MT component alone, on gold source text
> taken from the reference manifest. No ASR, VAD or TTS is involved. Because
> the input is a perfect transcript, this is a **ceiling** on end-to-end
> quality, not a prediction of it. Compare against the end-to-end run to see
> what the ASR stage costs. Latency here is MT-stage only and is not the
> competition's end-of-speech-to-first-audio metric.
>
> **Sample size.** This run scores 200 utterances from the fixed FLEURS
> vi->en subset. It is a corpus-level MT-only ceiling, not a pipeline result.

> **This configuration.** MT candidate run for vi->en, text-to-text, same corpus as the incumbent.

## Translation quality - MT-only, gold source text

# Evaluation — mt-envit5-gold200

| Metric | Value |
|---|---|
| bleu | 23.86 |
| ↳ bleu.tokenizer | 13a |
| ↳ bleu.signature | BLEU = 23.86 53.7/29.9/18.0/11.2 (BP = 1.000 ratio = 1.068 hyp_len = 4822 ref_len = 4516) |
| chrf | 57.11 |

| Count | Utterances |
|---|---|
| utterances | 200 |

## Latency - MT stage only

One push()+flush() cycle per utterance. Not the competition's end-of-speech-to-first-audio metric - MT alone, never interleaved with the quality table above.

| Stat | ms |
|---|---|
| p50 | 768.708 |
| p95 | 1301.125 |
| max | 1953.786 |
| n | 200 |
