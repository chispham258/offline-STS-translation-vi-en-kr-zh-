"""The TTS component benchmark must be backend-agnostic (D35)."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

import onevoice.components.dummy  # noqa: F401  - registers SilenceTTS
from onevoice.tts_benchmark import TTS_ONLY_CAVEAT, run_tts_benchmark

REFS = "fixtures/mini_vi_en/refs.jsonl"
CONFIG = "configs/dummy_vi_en.yaml"


def test_runs_a_backend_end_to_end(tmp_path):
    result = run_tts_benchmark(CONFIG, REFS, tmp_path, experiment="dummy")
    assert len(result.syntheses) == 3
    assert result.report_path.is_file()
    assert result.syntheses_path.is_file()
    assert result.metadata_path.is_file()


def test_archives_a_wav_per_utterance(tmp_path):
    """D31: MOS in Phase 5 needs artifacts, not a promise to regenerate them."""
    result = run_tts_benchmark(CONFIG, REFS, tmp_path, experiment="dummy")
    for synthesis in result.syntheses:
        path = Path(synthesis.wav_path)
        assert path.is_file()
        with wave.open(str(path)) as handle:
            assert handle.getframerate() == synthesis.sample_rate


def test_report_carries_the_scope_caveat(tmp_path):
    result = run_tts_benchmark(CONFIG, REFS, tmp_path, experiment="dummy")
    assert TTS_ONLY_CAVEAT.strip() in result.report_path.read_text()


def test_report_states_that_no_quality_metric_is_produced(tmp_path):
    # A TTS report without that sentence invites a reader to assume the
    # latency table says something about how the voice sounds.
    text = result_text = run_tts_benchmark(
        CONFIG, REFS, tmp_path, experiment="dummy"
    ).report_path.read_text()
    assert "no quality metric" in text.lower()
    assert "MOS" in result_text


def test_metadata_records_provenance_and_the_text_field(tmp_path):
    result = run_tts_benchmark(
        CONFIG, REFS, tmp_path, experiment="dummy", text_field="src_text"
    )
    meta = json.loads(result.metadata_path.read_text())
    assert meta["benchmark"] == "tts_component"
    assert meta["text_field"] == "src_text"
    assert "tts" in meta["components"]


def test_text_field_selects_which_column_is_spoken(tmp_path):
    """Whichever field is chosen applies to every backend equally — that
    symmetry is what keeps the comparison neutral."""
    src = run_tts_benchmark(
        CONFIG, REFS, tmp_path / "src", experiment="d", text_field="src_text"
    )
    ref = run_tts_benchmark(
        CONFIG, REFS, tmp_path / "ref", experiment="d", text_field="ref_text"
    )
    assert [s.text for s in src.syntheses] != [s.text for s in ref.syntheses]


def test_harness_contains_no_backend_specific_code():
    """The harness must contain no backend-specific branch.

    A comparison whose harness knows which implementation it is running is not
    a neutral comparison, whatever its intent. Docstrings may name backends —
    explaining *why* the harness is neutral is not the same as branching on
    one — so only executable lines are inspected.
    """
    source = Path("src/onevoice/tts_benchmark.py").read_text()

    code_lines = []
    in_docstring = False
    for raw in source.splitlines():
        line = raw.strip()
        if line.count('"""') == 1:
            in_docstring = not in_docstring
            continue
        if in_docstring or line.startswith("#") or line.startswith('"""'):
            continue
        code_lines.append(line.lower())

    executable = "\n".join(code_lines)
    for name in ("piper", "melo", "mms", "vits", "silence", "onnx"):
        assert name not in executable, (
            f"harness references backend {name!r} in executable code:\n"
            + "\n".join(ln for ln in code_lines if name in ln)
        )


def test_mixed_sample_rates_within_one_utterance_raise(tmp_path):
    import numpy as np

    from onevoice.tts_benchmark import _concatenate
    from onevoice.types import TtsEvent

    events = [
        TtsEvent(pcm=np.zeros(10, dtype=np.float32), sample_rate=16000, t_wall=0.0),
        TtsEvent(pcm=np.zeros(10, dtype=np.float32), sample_rate=22050, t_wall=0.0),
    ]
    with pytest.raises(ValueError, match="mixed sample rates"):
        _concatenate(events)
