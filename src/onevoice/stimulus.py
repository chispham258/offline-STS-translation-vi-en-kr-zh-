"""MOS stimulus preparation (TTS pre-registration §6-§7, D71/D72).

Runs after synthesis, in `.venv-tts-prep` only, and alters no model or
configuration provenance. Order is fixed: resample -> degrade (lower anchor
only) -> level-normalise -> encode. Arithmetic is float64 throughout.

A stimulus the protocol cannot prepare raises `StimulusError`; the caller
records that (system, utterance) cell as `errored` (§8). Nothing here limits,
compresses or otherwise repairs audio to make it pass.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from onevoice.provenance import file_hash, framework_identity

TARGET_SR = 16_000
TARGET_LUFS = -26.0  # D72; owner-set, not P.56 -26 dBov
PEAK_CEILING_DBFS = -1.0  # project-defined oversampled peak, never "dBTP"
OVERSAMPLE = 4
ANCHOR_SNR_DB = 10.0
NOISE_SEED = 0
SOXR_QUALITY = "VHQ"
# Integrated loudness is not linear in gain (the -70 LUFS absolute gate), so
# the single gain is refined until the measured output is within this tolerance.
LOUDNESS_TOL_LU = 0.001
MAX_GAIN_ITERATIONS = 10
WAV_SUBTYPE = "PCM_16"

ROLES = ("upper_anchor", "lower_anchor")
SYSTEM_PREFIX = "system:"


class StimulusError(ValueError):
    """The stimulus cannot be prepared under the protocol: an errored cell."""


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y**2)))


def resample(y: np.ndarray, sr: int) -> np.ndarray:
    """Mono float64 audio at `sr` -> `TARGET_SR` via soxr at the named quality."""
    import soxr

    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise StimulusError(f"expected mono audio, got shape {y.shape}")
    if y.size == 0:
        raise StimulusError("empty audio")
    if not np.all(np.isfinite(y)):
        raise StimulusError("non-finite samples")
    return soxr.resample(y, sr, TARGET_SR, quality=SOXR_QUALITY)


def add_anchor_noise(y: np.ndarray) -> tuple[np.ndarray, float]:
    """White Gaussian noise at exactly `ANCHOR_SNR_DB` over the full waveform.

    A fresh generator per call makes each anchor reproducible from its own
    source, independent of processing order. Returns the realised SNR.
    """
    noise = np.random.default_rng(NOISE_SEED).standard_normal(len(y))
    noise -= noise.mean()
    noise /= _rms(noise)
    noise *= _rms(y) / 10 ** (ANCHOR_SNR_DB / 20)
    return y + noise, 20 * np.log10(_rms(y) / _rms(noise))


def oversampled_peak_dbfs(y: np.ndarray) -> float:
    """Peak of the 4x polyphase-oversampled signal in dBFS (not BS.1770 dBTP)."""
    from scipy.signal import resample_poly

    return float(20 * np.log10(np.max(np.abs(resample_poly(y, OVERSAMPLE, 1)))))


def normalise(y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one gain to reach `TARGET_LUFS`; error if the peak then breaks the
    ceiling. Changes absolute level only.

    The gain is one scalar, refined by re-measuring the output: a gain changes
    which blocks pass BS.1770's absolute gate, so the first estimate can miss.
    """
    import pyloudnorm

    meter = pyloudnorm.Meter(TARGET_SR)

    def measure(x: np.ndarray) -> float:
        try:
            loudness = meter.integrated_loudness(x)
        except ValueError as exc:  # shorter than one gating block
            raise StimulusError(f"loudness unmeasurable: {exc}") from exc
        if not np.isfinite(loudness):
            raise StimulusError(f"loudness unmeasurable: {loudness} LUFS")
        return loudness

    loudness = measure(y)
    gain_db = TARGET_LUFS - loudness
    for iteration in range(1, MAX_GAIN_ITERATIONS + 1):  # noqa: B007 - recorded
        out = y * 10 ** (gain_db / 20)
        loudness_out = measure(out)
        if abs(loudness_out - TARGET_LUFS) <= LOUDNESS_TOL_LU:
            break
        gain_db += TARGET_LUFS - loudness_out
    else:
        raise StimulusError(
            f"loudness did not converge: {loudness_out:.4f} LUFS after "
            f"{MAX_GAIN_ITERATIONS} gain refinements"
        )
    peak = oversampled_peak_dbfs(out)
    if peak > PEAK_CEILING_DBFS:
        raise StimulusError(
            f"peak {peak:.2f} dBFS over ceiling {PEAK_CEILING_DBFS} dBFS "
            f"at {TARGET_LUFS} LUFS"
        )
    return out, {
        "loudness_in_lufs": loudness,
        "loudness_out_lufs": loudness_out,
        "gain_db": gain_db,
        "gain_iterations": iteration,
        "peak_dbfs": peak,
    }


def mos_synthesis_refs(
    manifest: list[dict[str, Any]], parent: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Parent RefRecords for the MOS manifest's ids, in manifest order.

    Refuses any id whose parent `ref_text` or `tgt_lang` differs from the frozen
    manifest: what is synthesised must be exactly what the panel was frozen on.
    """
    by_id = {r["utt_id"]: r for r in parent}
    out = []
    for item in manifest:
        ref = by_id.get(item["utt_id"])
        if ref is None:
            raise ValueError(f"utt {item['utt_id']} absent from parent refs")
        if (ref["ref_text"], ref["tgt_lang"]) != (item["text"], item["tgt_lang"]):
            raise ValueError(f"utt {item['utt_id']} differs from the MOS manifest")
        out.append(dict(ref))
    return out


def _check_role(role: str, system_provenance: dict[str, Any] | None) -> None:
    if role in ROLES:
        if system_provenance is not None:
            raise ValueError(f"{role} carries no model provenance")
        return
    if role.startswith(SYSTEM_PREFIX) and role != SYSTEM_PREFIX:
        if not system_provenance:
            raise ValueError(f"{role} requires its model/config provenance")
        return
    raise ValueError(f"unknown role {role!r}")


def prepare_stimulus(
    src: str | Path,
    dst: str | Path,
    *,
    utt_id: str,
    manifest_sha256: str,
    role: str,
    system_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare one stimulus and return its provenance record (§6).

    Raises `StimulusError` before writing anything if the stimulus is errored.
    """
    import soundfile

    _check_role(role, system_provenance)
    y, sr = soundfile.read(src, dtype="float64")
    y = resample(y, sr)

    record: dict[str, Any] = {}
    if role == "lower_anchor":
        y, snr = add_anchor_noise(y)
        record |= {
            "noise_seed": NOISE_SEED,
            "target_snr_db": ANCHOR_SNR_DB,
            "realised_snr_db": snr,
        }
    y, level = normalise(y)

    dst = Path(dst)
    soundfile.write(dst, y, TARGET_SR, subtype=WAV_SUBTYPE)

    record |= {
        "utt_id": utt_id,
        "manifest_sha256": manifest_sha256,
        "source_path": str(src),
        "source_sha256": file_hash(src),
        "role": role,
        "input_sr": sr,
        "output_sr": TARGET_SR,
        "soxr_version": version("soxr"),
        "soxr_quality": SOXR_QUALITY,
        **level,
        "target_lufs": TARGET_LUFS,
        "loudness_tol_lu": LOUDNESS_TOL_LU,
        "peak_ceiling_dbfs": PEAK_CEILING_DBFS,
        "oversample": OVERSAMPLE,
        "dtype": "float64",
        "encoding": {
            "format": "WAV",
            "subtype": WAV_SUBTYPE,
            "channels": 1,
            "dither": False,
        },
        "library_versions": {
            lib: version(lib)
            for lib in ("numpy", "scipy", "soxr", "pyloudnorm", "soundfile")
        },
        **framework_identity(),
        "output_path": str(dst),
        "output_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
    }
    if system_provenance is not None:
        record["system_provenance"] = system_provenance
    return record
