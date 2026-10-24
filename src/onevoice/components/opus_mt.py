"""Opus-MT (Helsinki-NLP MarianMT), behind an optional extra with a lazy import (D7).

transformers and torch pull in a large native dependency stack and a model
checkpoint. Importing them at module level would make this module
unimportable without the mt extra, and the registry would then be unable to
resolve the "opus" name at all - so the heavy import lives inside __init__,
same pattern as src/onevoice/components/whisper_asr.py.
"""

from __future__ import annotations

from collections.abc import Sequence

from onevoice.interfaces import BaseMT
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import MtEvent


@register("mt", "opus")
class OpusMT(BaseMT):
    """One direct Marian pair per instance (D13): push() buffers, flush() translates.

    Batch-shaped per D6, same as WhisperASR. src_lang/tgt_lang are accepted
    so config.build() can forward them from a config's languages block; each
    Opus checkpoint is a fixed-direction model, so they are not fed into
    translation itself, but they let this component be constructed the same
    way as every other component.
    """

    def __init__(
        self,
        model_name: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Opus-MT requires the mt extra, which is not installed. "
                "Install it with: uv sync --extra mt"
            ) from exc

        self._model_name = model_name
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)
        self._model.eval()
        self._segments: list[str] = []

    def provenance(self) -> dict[str, object]:
        """D46: which Marian checkpoint translated this run."""
        return describe_hub_model(
            component="opus",
            implementation=type(self),
            model_id=self._model_name,
            model=self._model,
            tokenizer_revision="same-repo-as-model",
            params={
                "model_name": self._model_name,
                "src_lang": self._src_lang,
                "tgt_lang": self._tgt_lang,
            },
            invocation={
                "src_lang": self._src_lang,
                "tgt_lang": self._tgt_lang,
                "route": "direct",
            },
        )

    def push(self, text: str, is_final: bool = False) -> Sequence[MtEvent]:
        self._check_not_flushed()
        if text:
            self._segments.append(text)
        return []

    def flush(self) -> Sequence[MtEvent]:
        self._mark_flushed()
        if not self._segments:
            return []

        source = " ".join(self._segments)
        batch = self._tokenizer([source], return_tensors="pt", padding=True)
        generated = self._model.generate(**batch)
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return [MtEvent(text=text.strip(), t_wall=0.0, is_final=True)]

    def reset(self) -> None:
        super().reset()
        self._segments = []
