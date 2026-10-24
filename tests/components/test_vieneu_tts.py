"""VieNeu-TTS v3 Turbo - the Vietnamese voice the owner chose directly (D79).

Not a Phase 6C winner: it was adopted without MOS. The tests pin what the
wrapper promises - a selectable preset voice with a Northern default, 48 kHz
read from the model, and provenance that names the exact Hub commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from onevoice.contracts import TTSContract
from onevoice.registry import resolve


def _vieneu_available() -> bool:
    try:
        import vieneu  # noqa: F401
    except Exception:
        return False
    return True


requires_vieneu = pytest.mark.skipif(
    not _vieneu_available(), reason="tts-vieneu extra not installed"
)


# --- structural: run in CI with nothing installed ----------------------------


def test_module_imports_without_vieneu():
    import onevoice.components.vieneu_tts  # noqa: F401


def test_registered_under_its_config_name():
    import onevoice.components.vieneu_tts as module

    assert resolve("tts", "vieneu") is module.VieneuTTS


def test_default_voice_is_northern():
    """Owner's rule: the default is a Northern (Hanoi) voice."""
    import onevoice.components.vieneu_tts as module

    assert module.DEFAULT_VOICE in module.NORTHERN_VOICES


def test_constructing_without_the_extra_raises_a_clear_error():
    import onevoice.components.vieneu_tts as module

    if _vieneu_available():
        pytest.skip("vieneu is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra tts-vieneu"):
        module.VieneuTTS()


# --- contract suite ----------------------------------------------------------


@requires_vieneu
@pytest.mark.slow
class TestVieneuTTSContract(TTSContract):
    """Sampled decoding (temperature 0.8), so not sample-exact across calls."""

    deterministic = False

    def make_component(self):
        from onevoice.components.vieneu_tts import VieneuTTS

        return VieneuTTS()


# --- behaviour ---------------------------------------------------------------


@pytest.fixture(scope="module")
def tts():
    if not _vieneu_available():
        pytest.skip("tts-vieneu extra not installed")
    from onevoice.components.vieneu_tts import VieneuTTS

    return VieneuTTS()


@requires_vieneu
@pytest.mark.slow
def test_synthesizes_vietnamese_audio_at_the_model_rate(tts):
    events = list(tts.push("Xin chào, anh khỏe không?", is_final=True))
    assert len(events) == 1
    assert events[0].sample_rate == tts.sample_rate == 48000
    pcm = events[0].pcm
    assert pcm.dtype == np.float32
    assert 0.01 < np.abs(pcm).max() <= 1.0


@requires_vieneu
@pytest.mark.slow
def test_empty_text_emits_nothing(tts):
    assert tts.push("   ", is_final=True) == []


@requires_vieneu
@pytest.mark.slow
def test_lists_voices_and_every_northern_voice_exists(tts):
    from onevoice.components.vieneu_tts import NORTHERN_VOICES

    assert set(NORTHERN_VOICES) <= set(tts.available_voices())


@requires_vieneu
@pytest.mark.slow
def test_unknown_voice_is_refused_at_construction():
    from onevoice.components.vieneu_tts import VieneuTTS

    with pytest.raises(ValueError, match="voice"):
        VieneuTTS(voice="Không Tồn Tại")


@requires_vieneu
@pytest.mark.slow
def test_provenance_records_commit_and_voice(tts):
    from onevoice.provenance import validate_provenance

    prov = tts.provenance()
    validate_provenance(prov)
    assert prov["component"] == "vieneu"
    assert prov["model_id"] == "pnnbao-ump/VieNeu-TTS-v3-Turbo"
    assert len(str(prov["commit_hash"])) == 40
    assert prov["invocation"]["voice"] == tts.voice
    assert prov["sample_rate"] == 48000
