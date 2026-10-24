"""Whisper ASR, behind an optional extra with a lazy import (D7).

transformers and torch pull in a large native dependency stack and a model
checkpoint. Importing them at module level would make this module
unimportable without the asr extra, and the registry would then be unable to
resolve the "whisper" name at all - so the heavy import lives inside
__init__, same pattern as src/onevoice/eval/metrics/comet.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.interfaces import BaseASR
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import AsrEvent, AudioChunk


@register("asr", "whisper")
class WhisperASR(BaseASR):
    """Batch-shaped per D6: push() buffers, flush() runs the model once."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        src_lang: str = "vi",
        device: str = "cpu",
        num_beams: int = 5,
        torch_dtype: str = "float32",
    ) -> None:
        try:
            import torch
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Whisper requires the asr extra, which is not installed. "
                "Install it with: uv sync --extra asr"
            ) from exc

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(torch_dtype, torch.float32)

        self._torch = torch
        self._model_name = model_name
        self._src_lang = src_lang
        self._device = device
        self._num_beams = num_beams
        self._torch_dtype = torch_dtype
        self._processor = WhisperProcessor.from_pretrained(model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self._model.to(device)
        self._model.eval()
        self._chunks: list[AudioChunk] = []

    def provenance(self) -> dict[str, object]:
        """D46: which checkpoint transcribed this run.

        The processor is loaded from the same repo as the model, so its
        revision is the model's; recorded explicitly because a tokenizer that
        could vary independently must be pinned separately.
        """
        return describe_hub_model(
            component="whisper",
            implementation=type(self),
            model_id=self._model_name,
            model=self._model,
            tokenizer_revision="same-repo-as-model",
            params={
                "model_name": self._model_name,
                "src_lang": self._src_lang,
                "device": self._device,
                "num_beams": self._num_beams,
                "torch_dtype": (
                    self._torch_dtype
                    if hasattr(self, "_torch_dtype")
                    else "float32"
                ),
            },
            # `device` is environment, not a request: it does not change what
            # Whisper was asked to transcribe.
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

        inputs = self._processor(pcm, sampling_rate=rate, return_tensors="pt")
        with self._torch.no_grad():
            generated = self._model.generate(
                inputs.input_features.to(self._device),
                language=self._src_lang,
                task="transcribe",
                num_beams=self._num_beams,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]

        return [
            AsrEvent(
                text=text.strip(),
                t_audio_end=t_audio_end,
                t_wall=0.0,
                is_final=True,
                confidence=None,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._chunks = []
