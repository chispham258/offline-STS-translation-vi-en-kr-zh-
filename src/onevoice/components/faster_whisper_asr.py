"""faster-whisper ASR — CTranslate2 backend with INT8 quantization (D7).

faster-whisper uses CTranslate2 for fast CPU/GPU inference with native
INT8 quantization. It is significantly faster than transformers-based
Whisper while maintaining comparable quality.

Install: uv sync --extra asr-faster-whisper
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.interfaces import BaseASR
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import AsrEvent, AudioChunk


@register("asr", "faster_whisper")
class FasterWhisperASR(BaseASR):
    """Batch-shaped per D6: push() buffers, flush() runs the model once.

    Uses faster-whisper (CTranslate2) with INT8 quantization for speed.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        src_lang: str = "vi",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper requires the asr-faster-whisper extra, "
                "which is not installed. Install it with: "
                "uv sync --extra asr-faster-whisper"
            ) from exc

        self._model_name = model_name
        self._src_lang = src_lang
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._num_workers = num_workers

        # faster-whisper handles device mapping internally
        device_arg = device if device != "cpu" else "cpu"
        self._model = WhisperModel(
            model_name,
            device=device_arg,
            compute_type=compute_type,
            num_workers=num_workers,
        )
        self._chunks: list[AudioChunk] = []

    def provenance(self) -> dict[str, object]:
        """D46: which checkpoint transcribed this run."""
        # faster-whisper uses CTranslate2 model format; record the source HF repo
        return describe_hub_model(
            component="faster_whisper",
            implementation=type(self),
            model_id=self._model_name,
            model=None,  # No HF model object to inspect
            tokenizer_revision="same-repo-as-model",
            params={
                "model_name": self._model_name,
                "src_lang": self._src_lang,
                "device": self._device,
                "compute_type": self._compute_type,
                "beam_size": self._beam_size,
                "num_workers": self._num_workers,
            },
            invocation={"src_lang": self._src_lang, "task": "transcribe"},
        )

    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        self._check_not_flushed()
        self._chunks.append(chunk)
        return []

    def flush(self) -> Sequence[AsrEvent]:
        self._mark_flushed()
        if not self._chunks:
            return []

        pcm = np.concatenate([chunk.pcm for chunk in self._chunks])
        rate = self._chunks[0].sample_rate
        last = self._chunks[-1]
        t_audio_end = last.t_start + last.duration_s

        # faster-whisper expects 16kHz; resample if needed
        if rate != 16000:
            import resampy
            pcm = resampy.resample(pcm, rate, 16000)
            rate = 16000

        segments, info = self._model.transcribe(
            pcm,
            language=self._src_lang,
            task="transcribe",
            beam_size=self._beam_size,
            vad_filter=False,  # We use our own VAD (Silero)
            word_timestamps=False,
        )

        # Concatenate all segment texts
        text = " ".join(segment.text for segment in segments).strip()

        return [
            AsrEvent(
                text=text,
                t_audio_end=t_audio_end,
                t_wall=0.0,
                is_final=True,
                confidence=None,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._chunks = []