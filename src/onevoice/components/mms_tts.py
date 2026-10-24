"""MMS-TTS — Meta's VITS voices, the Vietnamese challenger to Piper (D23).

Benchmarked by the same instrument and protocol as the baseline (D35). Its one
clear structural advantage over Piper is size: 36.3M parameters against a 63 MB
ONNX. Whether that buys anything is a measurement, not an argument.

Like Piper it is VITS and therefore stochastic (D32): the same text yields a
different sample count on every call. Unlike Piper it runs at 16 kHz, which
matches the pipeline's native rate — which is why the rate is read from the
model config rather than assumed. A hardcoded rate would be correct for one
backend and wrong for the other, and that is how a comparison gets silently
biased.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.interfaces import BaseTTS
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import TtsEvent

DEFAULT_MODEL = "facebook/mms-tts-vie"

_INSTALL_HINT = (
    "transformers is not installed. Install it with: uv sync --extra mt "
    "(or --extra asr; MMS-TTS uses the same transformers stack)."
)


@register("tts", "mms")
class MmsTTS(BaseTTS):
    """Synthesizes with a Meta MMS VITS voice."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tgt_lang: str = "vi",
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(_INSTALL_HINT) from exc

        self.model_name = model_name
        self.tgt_lang = tgt_lang
        self._torch = torch
        self._model = VitsModel.from_pretrained(model_name)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The model declares its own rate; Piper's is different (22050).
        self.sample_rate = int(self._model.config.sampling_rate)

    def provenance(self) -> dict[str, object]:
        """D46: which VITS checkpoint synthesised this run.

        Previously set `revision` and `commit_hash` to the same resolved
        commit, which discarded what the experiment asked for. The shared
        helper keeps the two questions apart.
        """
        record = describe_hub_model(
            component="mms",
            implementation=type(self),
            model_id=self.model_name,
            model=self._model,
            params={"model_name": self.model_name, "tgt_lang": self.tgt_lang},
            invocation={"tgt_lang": self.tgt_lang, "voice": self.model_name},
        )
        record["sample_rate"] = self.sample_rate
        return record

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        if not text.strip():
            return []

        inputs = self._tokenizer(text, return_tensors="pt")
        with self._torch.no_grad():
            waveform = self._model(**inputs).waveform

        pcm = np.asarray(waveform.squeeze().cpu().numpy(), dtype=np.float32)
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
