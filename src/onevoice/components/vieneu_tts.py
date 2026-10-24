"""VieNeu-TTS v3 Turbo - the Vietnamese voice adopted by owner decision (D79).

Chosen directly, not by the Phase 6C MOS selection, so nothing here claims it
beat Piper, MMS or F5 on naturalness. Apache-2.0, 48 kHz, torch-free on CPU
through ONNX Runtime, with 25 preset voices; `voice` picks one and the
default is a Northern (Hanoi) voice, per the owner.

Runs in its own environment (`.venv-tts-vieneu`, extra `tts-vieneu`).
onnxruntime is held below 1.23: 1.23+ rejects ONNX external-data files that
sit behind Hugging Face cache symlinks, which is exactly how the Hub stores
these graphs.

Decoding samples (temperature 0.8), so the same text gives a different
waveform on each call, as with the VITS backends (D32).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.components.cloning_tts import _commit_from_snapshot
from onevoice.interfaces import BaseTTS
from onevoice.provenance import params_hash
from onevoice.registry import register
from onevoice.types import TtsEvent

DEFAULT_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
CODEC_MODEL = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

#: Northern preset voices in vieneu 3.8.3, as its catalogue labels them.
NORTHERN_VOICES = (
    "Adam bựa",
    "Trúc Ly",
    "Thiện Minh",
    "Mai Anh",
    "Hải Đăng",
    "Thiền Tâm Đức",
    "Ngọc Huyền",
    "Minh Đức",
    "Phạm Tuyên",
    "Xuân Vĩnh",
    "Thanh Bình",
    "Ngọc Linh",
    "Đoan Trang",
    "Quỳnh Anh",
    "Quốc Tuấn",
)

DEFAULT_VOICE = "Mai Anh"

_SUBFOLDERS = {"fp32": "onnx_update", "int8": "onnx_int8"}

_INSTALL_HINT = (
    "vieneu is not installed. Install it into its own environment with: "
    "uv sync --extra tts-vieneu (or pip install 'vieneu' 'onnxruntime<1.23')."
)


@register("tts", "vieneu")
class VieneuTTS(BaseTTS):
    """Synthesizes Vietnamese with a VieNeu-TTS v3 Turbo preset voice."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        model_name: str = DEFAULT_MODEL,
        revision: str = "default",
        precision: str = "fp32",
        tgt_lang: str = "vi",
    ) -> None:
        try:
            from huggingface_hub import snapshot_download
            from vieneu import Vieneu
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(_INSTALL_HINT) from exc

        if precision not in _SUBFOLDERS:
            raise ValueError(f"precision must be one of {sorted(_SUBFOLDERS)}")
        subfolder = _SUBFOLDERS[precision]
        pinned = None if revision == "default" else revision

        # The backbone snapshot is materialised here and handed over as
        # `onnx_dir`, so the commit recorded is the one loaded. vieneu offers
        # no codec path on this route (it fetches the codec from main itself),
        # so the codec commit is main as resolved at construction.
        model_dir = snapshot_download(
            model_name, revision=pinned, allow_patterns=[f"{subfolder}/*"]
        )
        codec_dir = snapshot_download(
            CODEC_MODEL, allow_patterns=["*.onnx", "*.data", "*.json"]
        )

        self.voice = voice
        self.model_name = model_name
        self.revision = revision
        self.precision = precision
        self.tgt_lang = tgt_lang
        self._commit = _commit_from_snapshot(model_dir)
        self._codec_commit = _commit_from_snapshot(codec_dir)
        self._tts = Vieneu(
            backbone_repo=model_name,
            backend="onnx",
            precision=precision,
            onnx_dir=f"{model_dir}/{subfolder}",
        )
        if voice not in self.available_voices():
            raise ValueError(
                f"unknown voice {voice!r}; available: {self.available_voices()}"
            )
        self.sample_rate = int(self._tts.sample_rate)

    def available_voices(self) -> list[str]:
        """Preset voice names this checkpoint ships."""
        return [name for _label, name in self._tts.list_preset_voices()]

    def provenance(self) -> dict[str, object]:
        """D46: backbone commit, codec commit and the voice asked for."""
        params = {
            "model_name": self.model_name,
            "revision": self.revision,
            "precision": self.precision,
            "voice": self.voice,
            "tgt_lang": self.tgt_lang,
        }
        return {
            "component": "vieneu",
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_id": self.model_name,
            "revision": self.revision,
            "commit_hash": self._commit,
            "codec_model_id": CODEC_MODEL,
            "codec_commit_hash": self._codec_commit,
            "params_hash": params_hash(params),
            "invocation": {
                "tgt_lang": self.tgt_lang,
                "voice": self.voice,
                "watermark": True,
            },
            "sample_rate": self.sample_rate,
        }

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        if not text.strip():
            return []

        pcm = np.asarray(self._tts.infer(text, voice=self.voice), dtype=np.float32)
        pcm = pcm.squeeze()
        if pcm.ndim == 0 or len(pcm) == 0:
            return []

        return [
            TtsEvent(
                pcm=np.clip(pcm, -1.0, 1.0).astype(np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=is_final,
            )
        ]

    def flush(self) -> Sequence[TtsEvent]:
        self._mark_flushed()
        return []
