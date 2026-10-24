"""Silero VAD — the first detector with real lookahead (Task 11, D42).

Integrated last on purpose: before the lookback buffer existed, a detector that
confirms onsets retroactively would have produced a working-looking pipeline
with a systematically truncated first word.
"""

from __future__ import annotations

import numpy as np
import pytest

from onevoice.audio import chunk_audio, read_wav
from onevoice.contracts import VADContract
from onevoice.types import AudioChunk

RATE = 16000


def _vad_available() -> bool:
    try:
        import silero_vad  # noqa: F401
    except Exception:
        return False
    return True


requires_vad = pytest.mark.skipif(not _vad_available(), reason="vad extra missing")


def _speech():
    """Real speech, because Silero is trained on speech and a tone is not."""
    pcm, rate = read_wav("corpora/fleurs_vi_en/audio/1660.wav")
    assert rate == RATE
    return pcm, rate


# --- structural: these run in CI with nothing installed ----------------------


def test_module_imports_without_the_extra():
    import onevoice.components.silero_vad  # noqa: F401


def test_registered_under_its_own_name():
    import onevoice.components.silero_vad  # noqa: F401
    from onevoice.registry import resolve

    assert resolve("vad", "silero").__name__ == "SileroVAD"


def test_constructing_without_the_extra_names_the_extra():
    from onevoice.components.silero_vad import SileroVAD

    if _vad_available():
        pytest.skip("vad extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra vad"):
        SileroVAD()


# --- behaviour ---------------------------------------------------------------


@requires_vad
def test_detects_speech_in_real_audio():
    from onevoice.components.silero_vad import SileroVAD

    vad = SileroVAD()
    pcm, rate = _speech()
    events = [e for chunk in chunk_audio(pcm, rate) for e in vad.push(chunk)]
    events += list(vad.flush())
    assert any(e.kind == "speech_start" for e in events)
    assert any(e.kind == "speech_end" for e in events)


@requires_vad
def test_reports_onsets_in_the_past_which_is_the_whole_point():
    """Silero confirms an onset only after seeing what follows it.

    If `t_audio` were never behind the frame that produced the event, this
    detector would not exercise D42 and Task 10 would have been unnecessary.
    """
    from onevoice.components.silero_vad import SileroVAD

    vad = SileroVAD()
    pcm, rate = _speech()
    lag = []
    for chunk in chunk_audio(pcm, rate):
        for event in vad.push(chunk):
            if event.kind == "speech_start":
                lag.append(chunk.t_start + chunk.duration_s - event.t_audio)
    assert lag, "no speech_start detected"
    assert max(lag) > 0.032, "t_audio never pointed behind the current frame"


@requires_vad
def test_lookahead_fits_inside_the_lookback_window():
    """If this ever fails, D42's buffer is too small and utterances will raise."""
    from onevoice.components.silero_vad import SileroVAD
    from onevoice.lookback import DEFAULT_LOOKBACK_MS

    vad = SileroVAD()
    pcm, rate = _speech()
    worst = 0.0
    for chunk in chunk_audio(pcm, rate):
        for event in vad.push(chunk):
            if event.kind == "speech_start":
                worst = max(worst, chunk.t_start + chunk.duration_s - event.t_audio)
    assert worst * 1000 < DEFAULT_LOOKBACK_MS


@requires_vad
def test_an_unsupported_sample_rate_raises():
    """Silero accepts 8k and 16k. Anything else must not be silently resampled."""
    from onevoice.components.silero_vad import SileroVAD

    vad = SileroVAD()
    chunk = AudioChunk(
        pcm=np.zeros(512, dtype=np.float32), sample_rate=22050, t_start=0.0
    )
    with pytest.raises(ValueError, match="22050"):
        vad.push(chunk)


@requires_vad
def test_silence_produces_no_speech():
    from onevoice.components.silero_vad import SileroVAD

    vad = SileroVAD()
    silence = np.zeros(RATE, dtype=np.float32)
    events = [e for chunk in chunk_audio(silence, RATE) for e in vad.push(chunk)]
    events += list(vad.flush())
    assert [e for e in events if e.kind == "speech_start"] == []


@requires_vad
def test_flush_closes_an_utterance_that_never_went_quiet():
    """Audio ending mid-speech must still yield a speech_end.

    Without it the driver never stamps speech_end and the latency metric
    silently reports the utterance unmeasurable.
    """
    from onevoice.components.silero_vad import SileroVAD

    vad = SileroVAD()
    pcm, rate = _speech()
    opens = 0
    for chunk in chunk_audio(pcm[: int(rate * 1.2)], rate):
        for event in vad.push(chunk):
            opens += 1 if event.kind == "speech_start" else -1
    if opens > 0:
        assert any(e.kind == "speech_end" for e in vad.flush())


@requires_vad
def test_reset_returns_it_to_a_fresh_state():
    from onevoice.components.silero_vad import SileroVAD

    vad = SileroVAD()
    pcm, rate = _speech()
    first = [e for chunk in chunk_audio(pcm, rate) for e in vad.push(chunk)]
    vad.reset()
    again = [e for chunk in chunk_audio(pcm, rate) for e in vad.push(chunk)]
    assert [(e.kind, e.t_audio) for e in first] == [(e.kind, e.t_audio) for e in again]


# --- certification -----------------------------------------------------------


@requires_vad
@pytest.mark.slow
class TestSileroVADContract(VADContract):
    """The Phase 2 suite — gate item 2.

    `speech_chunks` is overridden because Silero is trained on speech and does
    not fire on the contract's default tone. That override is required, not
    optional: `test_speech_fixture_triggers_detection` fails loudly rather than
    skipping, so a detector whose stimulus does not suit it cannot pass
    vacuously (F1).
    """

    def make_component(self):
        from onevoice.components.silero_vad import SileroVAD

        return SileroVAD()

    def speech_chunks(self, t0: float = 0.0):
        from onevoice.types import AudioChunk

        pcm, rate = _speech()
        return [
            AudioChunk(pcm=c.pcm, sample_rate=c.sample_rate, t_start=t0 + c.t_start)
            for c in chunk_audio(pcm[: rate * 4], rate)
        ]


# --- D10: what this detector can and cannot be certified against -------------


@requires_vad
def test_tone_fixtures_are_outside_this_detectors_domain():
    """Silero must NOT be scored for boundary accuracy on the tone fixtures.

    The fixtures are tone bursts (`scripts/make_fixtures.py`); Silero is trained
    on speech. Measured against the declared boundaries it gets one onset right
    (-6 ms), misses another utterance entirely, and closes 1.5-2.5 s early,
    because a sustained pure tone stops looking like speech almost immediately.

    That is a stimulus mismatch, not a defect, and the honest response is to
    record it rather than widen D10's tolerance until it passes — a tolerance
    loosened to accommodate a result is a test that has stopped testing. D10
    accuracy therefore continues to be certified with `EnergyVAD`, whose
    measured margin on these same fixtures is within 24 ms at both ends.

    This test exists so that conclusion cannot be quietly reversed.
    """
    import json

    from onevoice.components.silero_vad import SileroVAD

    rows = [json.loads(line) for line in open("fixtures/mini_vi_en/refs.jsonl")]
    undetected = 0
    for row in rows:
        vad = SileroVAD()
        pcm, rate = read_wav("fixtures/mini_vi_en/" + row["audio_path"])
        events = [e for c in chunk_audio(pcm, rate) for e in vad.push(c)]
        events += list(vad.flush())
        if not [e for e in events if e.kind == "speech_start"]:
            undetected += 1
    assert undetected > 0, (
        "Silero now detects every tone fixture. If the fixtures gained real "
        "speech, re-measure its D10 margin and certify it properly; do not "
        "simply delete this test."
    )
