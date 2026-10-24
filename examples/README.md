# Example audio (vi → en)

Three real FLEURS Vietnamese utterances, kept so the pipeline can be run without
downloading the whole corpus. Paths in `refs.jsonl` are relative to this
directory.

| utt_id | seconds | note |
|---|---:|---|
| 1660 | 13.86 | long utterance with an internal pause; used by the Silero VAD test |
| 1676 | 12.36 | named entities (`sachin tendulkar`) |
| 1692 | 9.06 | numbers / date (`nhật bản`, caesium, iodine) |

Run one utterance through the async pipeline:

```bash
uv run python scripts/smoke_async.py \
  --config configs/full_vi_en_hybrid_async.yaml \
  --refs examples/refs.jsonl \
  --out runs/example-1660 \
  --limit 1
```

The full-corpus manifests are in `corpora/fleurs_vi_en_200/`; audio is not
tracked because it is rebuildable from `refs.meta.json`.
