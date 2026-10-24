"""The dummy components, and their conformance to the Phase 2 contract suite."""

import numpy as np
import pytest

from onevoice.components.dummy import EchoASR, EnergyVAD, IdentityMT, SilenceTTS
from onevoice.contracts import ASRContract, MTContract, TTSContract, VADContract
from onevoice.registry import resolve
from onevoice.types import AudioChunk

SAMPLE_RATE = 16000


def _chunk(amplitude: float, t_start: float = 0.0, n: int = 512) -> AudioChunk:
    pcm = np.full(n, amplitude, dtype=np.float32)
    return AudioChunk(pcm=pcm, sample_rate=SAMPLE_RATE, t_start=t_start)


# --- contract conformance: the whole suite, unmodified -----------------------


class TestEnergyVADContract(VADContract):
    def make_component(self):
        return EnergyVAD()


class TestEchoASRContract(ASRContract):
    def make_component(self):
        return EchoASR()


class TestIdentityMTContract(MTContract):
    def make_component(self):
        return IdentityMT()


class TestSilenceTTSContract(TTSContract):
    def make_component(self):
        return SilenceTTS()


# --- registration ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "name", "cls"),
    [
        ("vad", "energy", EnergyVAD),
        ("asr", "echo", EchoASR),
        ("mt", "identity", IdentityMT),
        ("tts", "silence", SilenceTTS),
    ],
)
def test_component_is_registered_under_its_config_name(kind, name, cls):
    assert resolve(kind, name) is cls


# --- EnergyVAD ---------------------------------------------------------------


def test_vad_stays_silent_on_quiet_audio():
    vad = EnergyVAD(threshold=0.1)
    assert vad.push(_chunk(0.0)) == []


def test_vad_opens_speech_when_energy_crosses_the_threshold():
    vad = EnergyVAD(threshold=0.1)
    events = vad.push(_chunk(0.5, t_start=0.32))
    assert [e.kind for e in events] == ["speech_start"]
    assert events[0].t_audio == pytest.approx(0.32)


def test_vad_does_not_reopen_speech_while_already_open():
    vad = EnergyVAD(threshold=0.1)
    vad.push(_chunk(0.5, t_start=0.0))
    assert vad.push(_chunk(0.5, t_start=0.032)) == []


def test_vad_closes_speech_when_energy_falls_back():
    vad = EnergyVAD(threshold=0.1)
    vad.push(_chunk(0.5, t_start=0.0))
    events = vad.push(_chunk(0.0, t_start=0.032))
    assert [e.kind for e in events] == ["speech_end"]
    assert events[0].t_audio == pytest.approx(0.032)


def test_vad_closes_open_speech_on_flush():
    vad = EnergyVAD(threshold=0.1)
    vad.push(_chunk(0.5, t_start=0.0))
    assert [e.kind for e in vad.flush()] == ["speech_end"]


def test_vad_flush_emits_nothing_when_speech_is_already_closed():
    vad = EnergyVAD(threshold=0.1)
    vad.push(_chunk(0.0))
    assert vad.flush() == []


# --- EchoASR -----------------------------------------------------------------


def test_asr_emits_nothing_until_flush():
    # A batch model buffers and emits one final event on flush (D6).
    asr = EchoASR()
    assert asr.push(_chunk(0.5)) == []


def test_asr_final_text_describes_the_audio_it_consumed():
    # D18: derived from the audio, not scripted.
    asr = EchoASR()
    asr.push(_chunk(0.5, n=512))
    asr.push(_chunk(0.5, n=512))
    events = asr.flush()
    assert len(events) == 1
    assert events[0].is_final is True
    assert "1024" in events[0].text


def test_asr_reset_clears_the_buffer():
    asr = EchoASR()
    asr.push(_chunk(0.5, n=512))
    asr.flush()
    asr.reset()
    asr.push(_chunk(0.5, n=512))
    assert "512" in asr.flush()[0].text


# --- IdentityMT --------------------------------------------------------------


def test_mt_returns_its_input_unchanged():
    events = IdentityMT().push("xin chào", is_final=True)
    assert [e.text for e in events] == ["xin chào"]


def test_mt_preserves_the_final_flag():
    assert IdentityMT().push("a", is_final=True)[0].is_final is True
    assert IdentityMT().push("a")[0].is_final is False


# --- SilenceTTS --------------------------------------------------------------


def test_tts_emits_silence_proportional_to_the_text():
    events = SilenceTTS(ms_per_char=10).push("hello", is_final=True)
    assert len(events) == 1
    assert len(events[0].pcm) == int(5 * 0.010 * SAMPLE_RATE)
    assert np.abs(events[0].pcm).max() == pytest.approx(0.0)


def test_tts_emits_nothing_for_empty_text():
    assert SilenceTTS().push("", is_final=True) == []
