"""MMS-TTS — the Vietnamese challenger to the Piper baseline.

D35: this is benchmarked by the same instrument and protocol as Piper. The
baseline governs which component ships by default, never how the comparison
runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from onevoice.contracts import TTSContract
from onevoice.registry import resolve

MODEL = "facebook/mms-tts-vie"


def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


requires_mms = pytest.mark.skipif(
    not _transformers_available(), reason="mt/asr extra not installed"
)


# --- structural: run in CI with nothing installed ----------------------------


def test_module_imports_without_transformers():
    import onevoice.components.mms_tts  # noqa: F401


def test_registered_under_its_config_name():
    import onevoice.components.mms_tts as module

    assert resolve("tts", "mms") is module.MmsTTS


def test_constructing_without_the_extra_raises_a_clear_error():
    import onevoice.components.mms_tts as module

    if _transformers_available():
        pytest.skip("transformers is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra"):
        module.MmsTTS()


# --- contract suite ----------------------------------------------------------


@requires_mms
@pytest.mark.slow
class TestMmsTTSContract(TTSContract):
    """MMS is VITS, so stochastic for the same reason Piper is (D32).

    Measured: 20736/20224/19968/20480 samples across four runs of one phrase.
    """

    deterministic = False

    def make_component(self):
        from onevoice.components.mms_tts import MmsTTS

        return MmsTTS(model_name=MODEL, tgt_lang="vi")


# --- behaviour ---------------------------------------------------------------


@requires_mms
@pytest.mark.slow
def test_synthesizes_vietnamese_audio():
    from onevoice.components.mms_tts import MmsTTS

    tts = MmsTTS(model_name=MODEL, tgt_lang="vi")
    events = list(tts.push("Xin chào, anh khỏe không?", is_final=True))
    assert len(events) == 1
    assert len(events[0].pcm) > 0
    assert np.abs(events[0].pcm).max() > 0.01


@requires_mms
@pytest.mark.slow
def test_sample_rate_comes_from_the_model_config():
    """MMS is 16 kHz where Piper is 22.05 kHz. Both must be read, not assumed —
    a hardcoded rate would be right for one backend and wrong for the other,
    which is exactly how a comparison gets silently biased."""
    from onevoice.components.mms_tts import MmsTTS

    tts = MmsTTS(model_name=MODEL, tgt_lang="vi")
    assert tts.sample_rate == 16000
    assert list(tts.push("Xin chào", is_final=True))[0].sample_rate == 16000


@requires_mms
@pytest.mark.slow
def test_empty_text_emits_nothing():
    from onevoice.components.mms_tts import MmsTTS

    tts = MmsTTS(model_name=MODEL, tgt_lang="vi")
    assert tts.push("", is_final=True) == []


@requires_mms
@pytest.mark.slow
def test_pcm_is_float32_in_range():
    from onevoice.components.mms_tts import MmsTTS

    tts = MmsTTS(model_name=MODEL, tgt_lang="vi")
    pcm = list(tts.push("Xin chào", is_final=True))[0].pcm
    assert pcm.dtype == np.float32
    assert np.abs(pcm).max() <= 1.0


@requires_mms
@pytest.mark.slow
def test_provenance_records_model_and_revision():
    """D33/D34: reproducible from metadata alone.

    D46/D73 validator: revision (what was requested) != commit_hash (what it
    resolved to). Collapsing them discards the request. "default" means nothing
    was pinned.
    """
    from onevoice.components.mms_tts import MmsTTS

    prov = MmsTTS(model_name=MODEL, tgt_lang="vi").provenance()
    assert prov["component"] == "mms"
    assert prov["model_id"] == MODEL
    # revision is what was requested ("default" = nothing pinned)
    # commit_hash is the actual Hub commit it resolved to
    assert prov["revision"] == "default"
    assert prov["commit_hash"] != "default"
    assert len(prov["commit_hash"]) >= 7
    assert prov["commit_hash"] != prov["revision"]  # D46/D73: must not collapse
    assert prov["sample_rate"] == 16000
