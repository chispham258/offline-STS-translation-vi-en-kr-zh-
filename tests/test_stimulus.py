"""MOS stimulus preparation (protocol §6-§7, D71/D72).

These tests pin the frozen parameters to the libraries the protocol names, so a
change to any of them fails here rather than silently changing what the panel
hears. They need the `tts-prep` extra and run in `.venv-tts-prep`.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

soxr = pytest.importorskip("soxr")
pyln = pytest.importorskip("pyloudnorm")
sf = pytest.importorskip("soundfile")
signal = pytest.importorskip("scipy.signal")

from onevoice import stimulus  # noqa: E402
from onevoice.stimulus import StimulusError  # noqa: E402


def _speechlike(sr: int, seconds: float = 3.0, level: float = 0.1) -> np.ndarray:
    """Amplitude-modulated tone complex: loud enough to pass loudness gating."""
    t = np.arange(int(sr * seconds)) / sr
    carrier = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t)
    return level * carrier * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))


def _write(path, y, sr, subtype="PCM_16"):
    sf.write(path, y, sr, subtype=subtype)
    return path


# --- frozen constants -------------------------------------------------------


def test_frozen_parameters_match_the_amended_protocol():
    assert stimulus.TARGET_SR == 16_000
    assert stimulus.TARGET_LUFS == -26.0
    assert stimulus.PEAK_CEILING_DBFS == -1.0
    assert stimulus.OVERSAMPLE == 4
    assert stimulus.ANCHOR_SNR_DB == 10.0
    assert stimulus.NOISE_SEED == 0
    assert stimulus.SOXR_QUALITY == "VHQ"


# --- resample ---------------------------------------------------------------


def test_resample_is_soxr_at_the_named_quality_not_the_default():
    y = _speechlike(22_050)
    out = stimulus.resample(y, 22_050)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(
        out, soxr.resample(y, 22_050, 16_000, quality="VHQ")
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((100, 2)),  # stereo is refused, never silently downmixed
        np.array([]),
        np.array([0.0, np.nan, 0.0]),
    ],
)
def test_resample_refuses_audio_the_protocol_cannot_prepare(bad):
    with pytest.raises(StimulusError):
        stimulus.resample(bad, 16_000)


# --- lower-anchor degradation ----------------------------------------------


def test_anchor_noise_hits_exactly_ten_db_over_the_full_waveform():
    y = _speechlike(16_000)
    noisy, snr = stimulus.add_anchor_noise(y)
    noise = noisy - y
    rms = lambda x: np.sqrt(np.mean(x**2))  # noqa: E731
    assert snr == pytest.approx(10.0, abs=1e-9)
    assert 20 * np.log10(rms(y) / rms(noise)) == pytest.approx(10.0, abs=1e-9)


def test_anchor_noise_is_the_frozen_generator_mean_removed_and_unit_rms():
    y = _speechlike(16_000)
    noisy, _ = stimulus.add_anchor_noise(y)
    n = np.random.default_rng(0).standard_normal(len(y))
    n -= n.mean()
    n /= np.sqrt(np.mean(n**2))
    expected = y + n * np.sqrt(np.mean(y**2)) / 10 ** (10 / 20)
    # Same construction, different float operation order: 1-ulp differences.
    np.testing.assert_allclose(noisy, expected, rtol=1e-12, atol=1e-15)


def test_anchor_noise_does_not_depend_on_processing_order():
    a, b = _speechlike(16_000), _speechlike(16_000, seconds=2.0, level=0.2)
    first, _ = stimulus.add_anchor_noise(a)
    stimulus.add_anchor_noise(b)
    again, _ = stimulus.add_anchor_noise(a)
    np.testing.assert_array_equal(first, again)


# --- peak and level ---------------------------------------------------------


def test_peak_is_measured_on_the_4x_polyphase_oversampled_signal():
    # A tone at fs/4 sampled 45 degrees off its crest: every sample peak reads
    # 0.707 of the true peak, which only the oversampled estimate recovers.
    t = np.arange(16_000)
    y = 0.5 * np.sin(2 * np.pi * t / 4 + np.pi / 4)
    expected = 20 * np.log10(np.max(np.abs(signal.resample_poly(y, 4, 1))))
    assert stimulus.oversampled_peak_dbfs(y) == expected
    assert expected > 20 * np.log10(np.max(np.abs(y)))


def test_normalise_reaches_the_target_loudness_and_records_the_gain():
    y = _speechlike(16_000, level=0.05)
    out, rec = stimulus.normalise(y)
    meter = pyln.Meter(16_000)
    assert meter.integrated_loudness(out) == pytest.approx(-26.0, abs=1e-6)
    assert rec["loudness_in_lufs"] == pytest.approx(meter.integrated_loudness(y))
    assert out == pytest.approx(y * 10 ** (rec["gain_db"] / 20))
    assert rec["peak_dbfs"] <= -1.0


def test_normalise_reaches_the_target_when_gain_moves_blocks_across_the_gate():
    # BS.1770's -70 LUFS absolute gate is not scale-invariant: here the quiet
    # tail passes the gate before the gain and falls below it after, so a
    # single gain computed from the input loudness lands 0.21 LU off target.
    y = np.concatenate(
        [_speechlike(16_000, 2.0, level=0.5), _speechlike(16_000, 2.0, level=1e-3)]
    )
    out, rec = stimulus.normalise(y)
    measured = pyln.Meter(16_000).integrated_loudness(out)
    assert measured == pytest.approx(-26.0, abs=stimulus.LOUDNESS_TOL_LU)
    assert rec["loudness_out_lufs"] == measured
    assert rec["gain_iterations"] > 1
    assert out == pytest.approx(y * 10 ** (rec["gain_db"] / 20))


def test_normalise_changes_level_only_so_the_anchor_snr_survives():
    y = _speechlike(16_000)
    noisy, _ = stimulus.add_anchor_noise(y)
    out, rec = stimulus.normalise(noisy)
    g = 10 ** (rec["gain_db"] / 20)
    rms = lambda x: np.sqrt(np.mean(x**2))  # noqa: E731
    assert 20 * np.log10(rms(y * g) / rms(out - y * g)) == pytest.approx(10.0)


def test_a_stimulus_over_the_ceiling_is_errored_never_limited():
    # Quiet tone plus one full-scale click: normalising the tone to -26 LUFS
    # drives the click far over -1 dBFS.
    y = _speechlike(16_000, level=0.001)
    y[8_000] = 0.9
    with pytest.raises(StimulusError, match="ceiling"):
        stimulus.normalise(y)


def test_unmeasurable_loudness_is_errored():
    with pytest.raises(StimulusError, match="loudness"):
        stimulus.normalise(np.zeros(16_000))


# --- end to end -------------------------------------------------------------

REQUIRED_KEYS = {
    "utt_id",
    "manifest_sha256",
    "source_path",
    "source_sha256",
    "role",
    "input_sr",
    "output_sr",
    "soxr_version",
    "soxr_quality",
    "loudness_in_lufs",
    "target_lufs",
    "gain_db",
    "peak_dbfs",
    "peak_ceiling_dbfs",
    "oversample",
    "dtype",
    "encoding",
    "library_versions",
    "framework_commit",
    "framework_dirty",
    "output_path",
    "output_sha256",
}


def test_prepare_writes_16k_mono_pcm16_and_records_provenance(tmp_path):
    src = _write(tmp_path / "src.wav", _speechlike(22_050), 22_050, "FLOAT")
    dst = tmp_path / "out.wav"
    rec = stimulus.prepare_stimulus(
        src,
        dst,
        utt_id="1664",
        manifest_sha256="m" * 64,
        role="system:piper-vi-vais1000",
        system_provenance={"model_id": "piper"},
    )
    info = sf.info(dst)
    assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
    assert REQUIRED_KEYS <= rec.keys()
    assert rec["input_sr"] == 22_050
    assert rec["system_provenance"] == {"model_id": "piper"}
    assert rec["output_sha256"] == hashlib.sha256(dst.read_bytes()).hexdigest()
    assert "noise_seed" not in rec


def test_lower_anchor_records_seed_and_both_snrs(tmp_path):
    src = _write(tmp_path / "src.wav", _speechlike(16_000), 16_000)
    rec = stimulus.prepare_stimulus(
        src,
        tmp_path / "low.wav",
        utt_id="1664",
        manifest_sha256="m" * 64,
        role="lower_anchor",
    )
    assert rec["noise_seed"] == 0
    assert rec["target_snr_db"] == 10.0
    assert rec["realised_snr_db"] == pytest.approx(10.0, abs=1e-9)


def test_prepare_is_byte_deterministic(tmp_path):
    src = _write(tmp_path / "src.wav", _speechlike(24_000), 24_000, "FLOAT")
    a = stimulus.prepare_stimulus(
        src, tmp_path / "a.wav", utt_id="1", manifest_sha256="m", role="upper_anchor"
    )
    b = stimulus.prepare_stimulus(
        src, tmp_path / "b.wav", utt_id="1", manifest_sha256="m", role="upper_anchor"
    )
    assert a["output_sha256"] == b["output_sha256"]


def test_an_errored_stimulus_writes_no_file(tmp_path):
    src = _write(tmp_path / "src.wav", np.zeros(16_000), 16_000)
    dst = tmp_path / "out.wav"
    with pytest.raises(StimulusError):
        stimulus.prepare_stimulus(
            src, dst, utt_id="1", manifest_sha256="m", role="upper_anchor"
        )
    assert not dst.exists()


@pytest.mark.parametrize(
    ("role", "system_provenance"),
    [
        ("anchor", None),
        ("system:", {"model_id": "x"}),
        ("system:piper", None),  # system output without model provenance
        ("upper_anchor", {"model_id": "x"}),  # anchors carry no model provenance
    ],
)
def test_prepare_refuses_malformed_roles(tmp_path, role, system_provenance):
    src = _write(tmp_path / "src.wav", _speechlike(16_000), 16_000)
    with pytest.raises(ValueError):
        stimulus.prepare_stimulus(
            src,
            tmp_path / "out.wav",
            utt_id="1",
            manifest_sha256="m",
            role=role,
            system_provenance=system_provenance,
        )
