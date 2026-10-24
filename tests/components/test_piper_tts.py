"""Piper TTS wrapper — the Vietnamese baseline (D23).

Structural tests run in CI with no extra installed. Model tests are slow-marked
and skip without `tts-piper`.
"""

from __future__ import annotations

import numpy as np
import pytest

from onevoice.contracts import TTSContract
from onevoice.registry import resolve

VOICE = "vi_VN-vais1000-medium"


def _piper_available() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return True


def _voice_present() -> bool:
    from onevoice.components.piper_tts import voice_paths

    try:
        onnx, config = voice_paths(VOICE)
    except Exception:
        return False
    return onnx.is_file() and config.is_file()


requires_piper = pytest.mark.skipif(
    not _piper_available(), reason="tts-piper extra not installed"
)
requires_voice = pytest.mark.skipif(
    not (_piper_available() and _voice_present()),
    reason="piper voice not downloaded; run scripts/fetch_piper_voice.py",
)


# --- structural: run in CI with nothing installed ----------------------------


def test_module_imports_without_piper():
    """The registry must resolve `piper` on a machine that cannot run it."""
    import onevoice.components.piper_tts  # noqa: F401


def test_importing_the_module_does_not_import_piper():
    import sys

    import onevoice.components.piper_tts  # noqa: F401

    if _piper_available():
        pytest.skip("tts-piper installed; cannot distinguish lazy from eager")
    assert "piper" not in sys.modules


def test_registered_under_its_config_name():
    import onevoice.components.piper_tts as module

    assert resolve("tts", "piper") is module.PiperTTS


def test_constructing_without_the_extra_raises_a_clear_error():
    import onevoice.components.piper_tts as module

    if _piper_available():
        pytest.skip("tts-piper extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra tts-piper"):
        module.PiperTTS()


def test_voice_paths_are_derived_from_the_voice_name():
    from onevoice.components.piper_tts import voice_paths

    onnx, config = voice_paths(VOICE)
    assert onnx.name == f"{VOICE}.onnx"
    assert config.name == f"{VOICE}.onnx.json"
    assert onnx.parent == config.parent


# --- contract suite ----------------------------------------------------------


@requires_voice
@pytest.mark.slow
class TestPiperTTSContract(TTSContract):
    """Piper against the full Phase 2 contract suite.

    `deterministic = False` under the D21 correctness exception granted
    2026-08-03: Piper is a VITS model and samples its duration predictor, so
    the same text yields a different sample count on every call. The suite's
    reset check falls back to structural invariants — same event count, same
    types, same emptiness — which a leaked-state bug would still violate.
    Every other guarantee still applies unchanged.
    """

    deterministic = False

    def make_component(self):
        from onevoice.components.piper_tts import PiperTTS

        return PiperTTS(voice=VOICE, tgt_lang="vi")


# --- behaviour ---------------------------------------------------------------


@requires_voice
@pytest.mark.slow
def test_synthesizes_vietnamese_audio():
    from onevoice.components.piper_tts import PiperTTS

    tts = PiperTTS(voice=VOICE, tgt_lang="vi")
    events = list(tts.push("Xin chào, anh khỏe không?", is_final=True))
    assert len(events) == 1
    assert len(events[0].pcm) > 0
    assert np.abs(events[0].pcm).max() > 0.01


@requires_voice
@pytest.mark.slow
def test_sample_rate_comes_from_the_voice_config_not_a_constant():
    """The voice's own JSON declares 22050. Hardcoding 16000 would silently
    stretch every Vietnamese utterance by 1.38x and drop its pitch."""
    from onevoice.components.piper_tts import PiperTTS

    tts = PiperTTS(voice=VOICE, tgt_lang="vi")
    assert tts.sample_rate == 22050
    events = list(tts.push("Xin chào", is_final=True))
    assert events[0].sample_rate == 22050


@requires_voice
@pytest.mark.slow
def test_empty_text_emits_nothing():
    from onevoice.components.piper_tts import PiperTTS

    tts = PiperTTS(voice=VOICE, tgt_lang="vi")
    assert tts.push("", is_final=True) == []


@requires_voice
@pytest.mark.slow
def test_pcm_is_float32_in_range():
    from onevoice.components.piper_tts import PiperTTS

    tts = PiperTTS(voice=VOICE, tgt_lang="vi")
    pcm = list(tts.push("Xin chào", is_final=True))[0].pcm
    assert pcm.dtype == np.float32
    assert np.abs(pcm).max() <= 1.0


@requires_voice
@pytest.mark.slow
def test_reset_allows_a_second_utterance():
    """Piper is stochastic — VITS samples duration, so the same text yields a
    different length every call (measured: 8704/9984/9472/9728 for one phrase).
    Assert the second utterance is produced and plausible, never that it is
    identical."""
    from onevoice.components.piper_tts import PiperTTS

    tts = PiperTTS(voice=VOICE, tgt_lang="vi")
    first = list(tts.push("Xin chào", is_final=True))[0].pcm
    tts.flush()
    tts.reset()
    second = list(tts.push("Xin chào", is_final=True))[0].pcm

    assert len(second) > 0
    assert np.abs(second).max() > 0.01
    # Same text, so lengths should be the same order of magnitude.
    assert 0.5 < len(second) / len(first) < 2.0


@requires_voice
@pytest.mark.slow
def test_synthesis_is_stochastic():
    """Pins the property that makes the contract suite's reset test fail.

    This is a fact about VITS, not a defect in the wrapper. Recorded as a test
    so a future reader does not 'fix' the nondeterminism by silencing noise.
    """
    from onevoice.components.piper_tts import PiperTTS

    tts = PiperTTS(voice=VOICE, tgt_lang="vi")
    lengths = set()
    for _ in range(4):
        lengths.add(len(list(tts.push("Xin chào", is_final=True))[0].pcm))
        tts.reset()
    assert len(lengths) > 1, f"expected stochastic output, got {lengths}"
