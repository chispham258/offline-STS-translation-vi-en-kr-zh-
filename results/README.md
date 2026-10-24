# Run archive — parameters for the key measurements

Each directory is a completed run. It keeps the **config, provenance and
manifest** (plus a report where one was archived) so results can be inspected
or re-scored without model weights. Output audio (`audio/`) is omitted to keep
the repo small; the manifest still retains each utterance's path and timing.

Re-score from a manifest (does not run any model):

```bash
uv sync --extra dev --extra comet
uv run python -c "
from onevoice.eval.manifest import read_run
from onevoice.eval.evaluator import evaluate
records = read_run('results/<run>/run.jsonl')
# pass the matching refs from corpora/ when BLEU/chrF/WER are needed
"
```

## Index

| Directory | Date | N | Arm | Source |
|---|---|---:|---|---|
| `2026-10-23_vi_en_d80_wav2vec2_lm_full200` | 2026-10-23 | 200 | **D80: wav2vec2-250h + 4-gram KenLM + envit5 + Piper** | `runs/bakeoff_vi_en/asr_wav2vec2_lm_full200_pipeline` |
| `2026-10-23_vi_en_turbo_full200` | 2026-10-23 | 200 | Whisper-large-v3-turbo + envit5 + Piper | `runs/bakeoff_vi_en/final_200` |
| `2026-10-23_vi_en_phowhisper_large_full200` | 2026-10-23 | 200 | PhoWhisper-large CT2 INT8 + envit5 + Piper | `runs/bakeoff_vi_en/asr_phowhisper_large_full200_pipeline` |
| `2026-10-23_vi_en_mt_envit5_gold200` | 2026-10-23 | 200 | envit5, gold source (MT-only ceiling) | `runs/bakeoff_vi_en/mt_envit5_gold200` |
| `2026-10-24_vi_en_mt_argos_screen22` | 2026-10-24 | 22 | Argos 1.9 (int8 + float32) and gold-source | `runs/bakeoff_vi_en/mt_argos_screen22` |
| `2026-10-21_en_vi_piper_full200` | 2026-10-21 | 200 | Whisper-base + envit5 + Piper vi (en→vi) | `runs/bench_200_cpu_piper` |

## Directory contents

- `config.used.yaml` — the exact config that ran (may differ from the current
  config).
- `metadata.json` — provenance: model id, revision, hash, host, versions.
- `run.jsonl` or `run_manifest.jsonl` — one line per utterance: hypothesis,
  reference id, timing, status.
- `report.md` — the published report where one was archived; the consolidated
  Q6 statement is in `docs/RESULTS.md`.
- `comet.json`, `hypotheses.jsonl`, `bootstrap.json` — only in the runs that
  produced them.

## Limits

- Host: Apple M1, CPU, Python 3.11.14. **Not Snapdragon.**
- These are `development` / `screen` runs, not a D60 selection.
- No MOS; no naturalness conclusion.
- `2026-10-21_en_vi_piper_full200` lacks the original `metadata.json`; its config
  and report were taken from the repo at archive time.
