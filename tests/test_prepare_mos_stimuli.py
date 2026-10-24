"""Resumable MOS synthesis (6C).

The first F5 run stalled on one long utterance after 33 of 40 clips, and
because run_tts_benchmark writes its index only at the end, the 33 finished
clips had no record. `synthesise` writes each clip's record the moment its WAV
exists and skips clips already recorded, so a stall loses at most one clip.
"""

from __future__ import annotations

import json

from scripts.prepare_mos_stimuli import synthesise

CONFIG = """\
experiment:
  name: t
languages:
  src: en
  tgt: vi
vad:
  model: energy
asr:
  model: echo
mt:
  model: identity
tts:
  model: silence
"""


def _refs(tmp_path, ids):
    path = tmp_path / "refs.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "utt_id": u,
                    "audio_path": "a.wav",
                    "src_lang": "en",
                    "tgt_lang": "vi",
                    "src_text": "x",
                    "ref_text": f"câu {u}",
                    "speech_start_s": 0.0,
                    "speech_end_s": 1.0,
                }
            )
            + "\n"
            for u in ids
        )
    )
    return path


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_each_clip_is_recorded_with_its_invocation(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG)
    out = tmp_path / "synth"
    assert synthesise(cfg, _refs(tmp_path, ["1", "2"]), out) == 2
    rows = _lines(out / "syntheses.jsonl")
    assert [r["utt_id"] for r in rows] == ["1", "2"]
    assert all((out / "audio" / f"{r['utt_id']}.wav").is_file() for r in rows)
    (inv,) = _lines(out / "invocations.jsonl")
    assert {r["invocation"] for r in rows} == {inv["invocation"]} == {0}
    assert inv["components"]["tts"]
    assert inv["refs_hash"] and inv["config_hash"]


def test_a_rerun_skips_recorded_clips_and_only_does_the_rest(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG)
    out = tmp_path / "synth"
    synthesise(cfg, _refs(tmp_path, ["1"]), out)
    assert synthesise(cfg, _refs(tmp_path, ["1", "2", "3"]), out) == 2
    rows = _lines(out / "syntheses.jsonl")
    assert [(r["utt_id"], r["invocation"]) for r in rows] == [
        ("1", 0),
        ("2", 1),
        ("3", 1),
    ]
    assert len(_lines(out / "invocations.jsonl")) == 2


def test_synthesis_never_loads_the_evaluator(tmp_path):
    # Model environments (e.g. .venv-tts-f5) lack evaluator dependencies such
    # as jiwer; importing onevoice.benchmark there crashes synthesis.
    import subprocess
    import sys

    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG)
    code = (
        "import sys; from scripts.prepare_mos_stimuli import synthesise; "
        f"synthesise({str(cfg)!r}, {str(_refs(tmp_path, ['1']))!r}, "
        f"{str(tmp_path / 'synth')!r}); "
        "loaded = [m for m in sys.modules if m.startswith("
        "('onevoice.benchmark', 'onevoice.eval.evaluator', 'jiwer'))]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_nothing_left_to_do_builds_no_model_and_logs_no_invocation(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(CONFIG)
    out = tmp_path / "synth"
    refs = _refs(tmp_path, ["1"])
    synthesise(cfg, refs, out)
    assert synthesise(cfg, refs, out) == 0
    assert len(_lines(out / "invocations.jsonl")) == 1
