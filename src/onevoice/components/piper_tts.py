"""Piper TTS — the primary Vietnamese baseline (D23).

Piper is the only Vietnamese TTS candidate that ships ONNX, the format
Qualcomm's runtime consumes, so it needs no conversion step before Phase 6.
`piper-tts` bundles its own espeak-ng data and native bridge, so no system
`espeak-ng` install is required.

Batch-shaped in the sense that matters: one `push` synthesizes that text and
emits one event. `flush()` has nothing left to do. The driver stamps
TTS_FIRST_AUDIO on whichever path emits first, so either shape works — this one
gets audio out earlier.

The sample rate is read from the voice's own JSON, never hardcoded. The
Vietnamese baseline runs at 22050 Hz while the fixtures are 16000 Hz; assuming
16000 would stretch every utterance by 1.38x and drop its pitch, and nothing
downstream would flag it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from onevoice.interfaces import BaseTTS
from onevoice.provenance import params_hash
from onevoice.registry import register
from onevoice.types import TtsEvent

DEFAULT_VOICE = "vi_VN-vais1000-medium"

#: Where voice files live. Override with ONEVOICE_PIPER_VOICES.
VOICES_DIR = Path(
    os.environ.get("ONEVOICE_PIPER_VOICES", Path.home() / ".cache/onevoice/piper")
)

_INSTALL_HINT = (
    "piper-tts is not installed. Install it with: uv sync --extra tts-piper. "
    "It bundles its own espeak-ng data, so no system espeak-ng is needed."
)


def voice_paths(voice: str, voices_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Return the (onnx, config) paths for a voice name.

    Pure path arithmetic — safe to call with nothing installed, which is why the
    structural tests can use it.
    """
    root = Path(voices_dir) if voices_dir is not None else VOICES_DIR
    return root / f"{voice}.onnx", root / f"{voice}.onnx.json"


@register("tts", "piper")
class PiperTTS(BaseTTS):
    """Synthesizes with a Piper ONNX voice."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        tgt_lang: str = "vi",
        voices_dir: str | Path | None = None,
    ) -> None:
        try:
            from piper import PiperVoice
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(_INSTALL_HINT) from exc

        self.voice = voice
        self.tgt_lang = tgt_lang
        self._voices_dir = voices_dir

        onnx_path, config_path = voice_paths(voice, voices_dir)
        if not onnx_path.is_file() or not config_path.is_file():
            raise RuntimeError(
                f"Piper voice {voice!r} not found at {onnx_path.parent}. "
                f"Fetch it with: uv run python scripts/fetch_piper_voice.py {voice}"
            )

        # The voice declares its own rate; never assume the pipeline's.
        self.sample_rate = int(
            json.loads(config_path.read_text(encoding="utf-8"))["audio"]["sample_rate"]
        )
        self._voice = PiperVoice.load(onnx_path, config_path)

    def provenance(self) -> dict[str, object]:
        """D46: exactly which voice produced this run's audio.

        Piper loads a file, not a Hub checkpoint, so there is no library-
        reported commit. The weight hash is the anchor the schema requires and
        is the stronger identity: it pins the exact bytes rather than a ref.

        Voice selection is part of the experiment as much as the model is - two
        Piper voices differ more than two checkpoints of one voice would.
        """
        onnx_path, _ = voice_paths(self.voice, self._voices_dir)
        return {
            "component": "piper",
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_id": "rhasspy/piper-voices",
            "revision": "default",
            "weight_hash": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "params_hash": params_hash(
                {"voice": self.voice, "tgt_lang": self.tgt_lang}
            ),
            "invocation": {
                "voice": self.voice,
                "tgt_lang": self.tgt_lang,
                "sample_rate": self.sample_rate,
            },
            "weight_bytes": onnx_path.stat().st_size,
        }

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        if not text.strip():
            return []

        parts = [
            np.asarray(chunk.audio_float_array, dtype=np.float32)
            for chunk in self._voice.synthesize(text)
        ]
        if not parts:
            return []

        pcm = np.concatenate(parts) if len(parts) > 1 else parts[0]
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
