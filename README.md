# OneVoice

Speech-to-speech translation framework for **offline deployment**. The design
target is Snapdragon-class edge devices, but **edge suitability is not yet
measured** — every number in this repository is host Apple M1 CPU. The language
set is fixed:

`L = {vi, en, zh, ko}`

Any language in `L` can be the source; any other can be the target. The pair is
selected per run with `languages.src` / `languages.tgt` (`src != tgt`).

The pipeline is a cascade:

![OneVoice cascade: audio in → VAD → ASR → MT → TTS → audio out](docs/diagrams/architecture.svg)

Mechanism, queues, and timing: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Language coverage

`✓` measured end-to-end, `~` partial, `—` not measured, `B` blocked.

| src \ tgt | vi | en | zh | ko |
|---|---|---|---|---|
| **vi** | n/a | **✓** | ~ | ~ |
| **en** | **✓** | n/a | — | — |
| **zh** | B | — | n/a | — |
| **ko** | B | — | — | n/a |

- **Measured:** vi→en and en→vi, 200 utterances each.
- **Partial:** vi→zh and vi→ko — MT selected, end-to-end not validated.
- **Blocked:** zh→vi and ko→vi — no corpus.
- **Other pairs:** configurable, not measured. Do not quote a number for them.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

Dummy benchmark, no model download:

```bash
uv run python -c "
from onevoice.benchmark import run_benchmark
run_benchmark(
    'configs/dummy_vi_en.yaml',
    'fixtures/mini_vi_en/refs.jsonl',
    'runs/dummy',
    experiment='dummy',
)
"
```

One real utterance through the async pipeline:

```bash
uv sync --extra models
uv run python scripts/smoke_async.py \
  --config configs/full_vi_en_hybrid_async.yaml \
  --refs examples/refs.jsonl \
  --out runs/example-1660 \
  --limit 1
```

Model weights live in `models/` (gitignored). Extras are in `pyproject.toml`.

## Results at a glance

All numbers are host Apple M1 CPU, not edge-device measurements.

| Direction | N | Arm | WER_norm | BLEU | COMET | pipeline RTF p50 | Note |
|---|---:|---|---:|---:|---:|---:|---|
| vi→en | 200 | D80 wav2vec2 + 4-gram LM | 0.1373 | 17.51 | 0.7545 | **0.196** | host deployment candidate |
| vi→en | 200 | PhoWhisper-large | 0.0912 | **20.26** | **0.7907** | 1.078 | quality leader; RTF fail |
| en→vi | 200 | whisper-base + envit5 + Piper vi | 0.1753 | **26.60** | 0.7691 | **0.2425** | measured |

Parameters, tables and caveats: [`docs/RESULTS.md`](docs/RESULTS.md).
Host metrics and missing edge evidence: [`docs/RESULTS.md` § Edge-device readiness](docs/RESULTS.md#10-edge-device-readiness).

## Repository

```text
src/onevoice/    pipeline, components, eval
configs/         one YAML per arm and language pair
tests/           unit + end-to-end
examples/        3 FLEURS clips + refs
results/         archived runs (config, provenance, manifest)
docs/            architecture, results, decision ledger
```

## Docs

| File | Role |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Stages, async queues, timing |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Run parameters and result tables |
| [`docs/KNOWLEDGE.md`](docs/KNOWLEDGE.md) | Decision ledger D1–D85 |
| [`results/README.md`](results/README.md) | Archived run index |
| [`docs/diagrams/`](docs/diagrams) | Mermaid source + rendered SVG |

## Status

- vi→en host deployment candidate: D80 wav2vec2 + 4-gram LM.
- Edge target is Snapdragon-class; no named device, no NPU run, and no device
  RTF/RSS/power/thermal measurement. Host M1 results do not establish fit.
- No valid MOS: VieNeu is an owner decision, not a benchmark win.
- D60 ASR selection is unresolved; D80 is a host deployment choice.
