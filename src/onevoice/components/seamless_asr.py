"""SeamlessM4T-v2 ASR, behind the asr extra with a lazy import (D7).

Third family in the ASR matrix (D56), alongside `whisper` and `wav2vec2`.

**ASR-only by construction.** SeamlessM4T can also translate directly, and using
that ability here would make it a different experiment: the track compares ASR
components with MT held constant, so this wrapper calls it with
`tgt_lang == src_lang` and takes the transcript. That constraint is recorded in
the track's pre-registration, before any result.

Batch-shaped per D6: push() buffers, flush() runs the model once.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.interfaces import BaseASR
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import AsrEvent, AudioChunk

DEFAULT_MODEL = "facebook/seamless-m4t-v2-large"

#: SeamlessM4T uses ISO 639-3. Raising beats guessing: a wrong code silently
#: transcribes the wrong language, which looks like a bad model rather than a
#: bad config (same discipline as the NLLB FLORES mapping).
_LANG3 = {"vi": "vie", "en": "eng", "zh": "cmn", "ko": "kor"}


def seamless_lang(code: str) -> str:
    try:
        return _LANG3[code]
    except KeyError:
        raise ValueError(
            f"no SeamlessM4T language code mapped for {code!r}; "
            f"known: {sorted(_LANG3)}"
        ) from None


@register("asr", "seamless")
class SeamlessASR(BaseASR):
    """SeamlessM4T-v2 used strictly as a transcriber."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        src_lang: str = "vi",
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
        except ImportError as exc:
            raise RuntimeError(
                "SeamlessM4T requires the asr extra, which is not installed. "
                "Install it with: uv sync --extra asr"
            ) from exc

        self._torch = torch
        self._model_name = model_name
        self._src_lang = src_lang
        self._lang3 = seamless_lang(src_lang)
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()
        self._chunks: list[AudioChunk] = []

    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        self._check_not_flushed()
        self._chunks.append(chunk)
        return []

    def flush(self) -> Sequence[AsrEvent]:
        self._mark_flushed()
        if not self._chunks:
            return []

        pcm = np.concatenate([c.pcm for c in self._chunks])
        rate = self._chunks[0].sample_rate
        last = self._chunks[-1]

        inputs = self._processor(
            audios=pcm, sampling_rate=rate, return_tensors="pt"
        )
        with self._torch.no_grad():
            # tgt_lang == source language: transcription, not translation.
            generated = self._model.generate(
                **{k: v.to(self._device) for k, v in inputs.items()},
                tgt_lang=self._lang3,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]

        return [
            AsrEvent(
                text=text,
                t_audio_end=last.t_start + last.duration_s,
                t_wall=0.0,
                is_final=True,
                confidence=None,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._chunks = []

    def provenance(self) -> dict[str, object]:
        """D46/D57."""
        return describe_hub_model(
            component="seamless",
            implementation=type(self),
            model_id=self._model_name,
            model=self._model,
            tokenizer_revision="same-repo-as-model",
            params={
                "model_name": self._model_name,
                "src_lang": self._src_lang,
                "device": self._device,
            },
            invocation={
                "src_lang": self._src_lang,
                "task": "transcribe",
                "tgt_lang": self._lang3,
                "mode": "asr_only",
            },
        )
