"""Whisper ASR wrapper: contract conformance plus structural checks."""

import numpy as np
import pytest

from onevoice.contracts import ASRContract
from onevoice.registry import resolve
from onevoice.types import AudioChunk

SAMPLE_RATE = 16000
FIXTURE = "fixtures/mini_vi_en/audio/utt-001.wav"


def _whisper_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


requires_whisper = pytest.mark.skipif(
    not _whisper_available(), reason="asr extra not installed"
)


# --- structural: these run in CI with no model installed ---------------------


def test_wrapper_module_imports_without_torch():
    # The module must be importable so the registry can resolve its name even
    # when the extra is absent; the heavy import belongs inside __init__.
    import onevoice.components.whisper_asr  # noqa: F401


def test_registered_under_its_config_name():
    import onevoice.components.whisper_asr as module

    assert resolve("asr", "whisper") is module.WhisperASR


def test_constructing_without_the_extra_raises_a_clear_error():
    import onevoice.components.whisper_asr as module

    if _whisper_available():
        pytest.skip("asr extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra asr"):
        module.WhisperASR()


# --- contract suite: needs the real model ------------------------------------


@requires_whisper
@pytest.mark.slow
class TestWhisperASRContract(ASRContract):
    def make_component(self):
        from onevoice.components.whisper_asr import WhisperASR

        return WhisperASR(model_name="openai/whisper-tiny", src_lang="vi")


# --- behaviour: needs the real model -----------------------------------------


@requires_whisper
@pytest.mark.slow
def test_push_buffers_and_flush_emits_one_final_event():
    from onevoice.audio import chunk_audio, read_wav
    from onevoice.components.whisper_asr import WhisperASR

    pcm, rate = read_wav(FIXTURE)
    asr = WhisperASR(model_name="openai/whisper-tiny", src_lang="vi")
    for chunk in chunk_audio(pcm, rate):
        assert asr.push(chunk) == []
    events = asr.flush()
    assert len(events) == 1
    assert events[0].is_final is True
    assert isinstance(events[0].text, str)


@requires_whisper
@pytest.mark.slow
def test_confidence_is_none_rather_than_fabricated():
    # A made-up confidence is worse than an absent one: Phase 7's confidence
    # logic needs a real number, and a placeholder would silently pass as one.
    from onevoice.components.whisper_asr import WhisperASR

    asr = WhisperASR(model_name="openai/whisper-tiny", src_lang="vi")
    asr.push(
        AudioChunk(
            pcm=np.zeros(SAMPLE_RATE, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            t_start=0.0,
        )
    )
    assert asr.flush()[0].confidence is None


@requires_whisper
@pytest.mark.slow
def test_reset_clears_the_buffer_between_utterances():
    from onevoice.audio import chunk_audio, read_wav
    from onevoice.components.whisper_asr import WhisperASR

    pcm, rate = read_wav(FIXTURE)
    asr = WhisperASR(model_name="openai/whisper-tiny", src_lang="vi")
    for chunk in chunk_audio(pcm, rate):
        asr.push(chunk)
    first = asr.flush()[0].text
    asr.reset()
    for chunk in chunk_audio(pcm, rate):
        asr.push(chunk)
    assert asr.flush()[0].text == first
