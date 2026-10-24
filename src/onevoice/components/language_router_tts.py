"""Language-routed TTS: one config, one backend per target language.

The project targets en/vi/zh/ko, but no single offline model covers all four
with acceptable quality. The CPU/edge arm uses Piper for all four; the
quality/NPU arm can use VieNeu for Vietnamese (D79). This wrapper picks the
backend for the experiment's target language at construction time, so a run
still has exactly one TTS component and no per-utterance language switching.

Backends are imported lazily from their module path. That keeps the two
runtime environments isolated: a VieNeu run needs only `vieneu`, a Piper run
needs only `piper`, and neither has to install the other.

Config shape (CPU arm)::

    tts:
      model: language_router
      params:
        backends:
          vi:
            module: onevoice.components.piper_tts
            model: piper
            params: {voice: "vi_VN-vais1000-medium"}
          en:
            module: onevoice.components.piper_tts
            model: piper
            params: {voice: "en_US-lessac-medium"}
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from onevoice.interfaces import BaseTTS
from onevoice.registry import register, resolve
from onevoice.types import TtsEvent


@register("tts", "language_router")
class LanguageRouterTTS(BaseTTS):
    """Delegate synthesis to the backend registered for ``tgt_lang``."""

    def __init__(
        self,
        backends: dict[str, dict[str, Any]],
        tgt_lang: str,
        fallback: str | None = None,
    ) -> None:
        if not backends:
            raise ValueError("language_router needs at least one backend")
        key = self._select_key(backends, tgt_lang, fallback)
        spec = backends[key]
        module = spec.get("module")
        if module:
            importlib.import_module(module)
        model = spec["model"]
        params = dict(spec.get("params", {}))
        if "tgt_lang" in params:
            raise ValueError(
                "language_router backend params must not set tgt_lang; "
                "it comes from languages.tgt so the two cannot disagree"
            )
        params["tgt_lang"] = tgt_lang
        cls = resolve("tts", model)
        self._backend = cls(**params)
        self._backend_key = key
        self._backend_model = model
        self.tgt_lang = tgt_lang
        self.fallback = fallback

    @staticmethod
    def _select_key(
        backends: dict[str, dict[str, Any]],
        tgt_lang: str,
        fallback: str | None,
    ) -> str:
        if tgt_lang in backends:
            return tgt_lang
        family = tgt_lang.replace("-", "_").split("_")[0]
        if family in backends:
            return family
        if fallback is not None and fallback in backends:
            return fallback
        raise ValueError(
            f"language_router has no backend for {tgt_lang!r}; "
            f"available: {sorted(backends)}"
        )

    def _unwrap(self, text: object) -> str:
        # The async driver forwards MtEvent; the sequential driver forwards
        # str. Accept both so the router works on either path.
        if isinstance(text, str):
            return text
        value = getattr(text, "text", None)
        if isinstance(value, str):
            return value
        raise TypeError(
            f"expected str or an object with .text, got {type(text).__name__}"
        )

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        return self._backend.push(self._unwrap(text), is_final=is_final)

    def flush(self) -> Sequence[TtsEvent]:
        self._mark_flushed()
        return self._backend.flush()

    def reset(self) -> None:
        super().reset()
        self._backend.reset()

    def provenance(self) -> dict[str, object]:
        data = dict(self._backend.provenance())
        data["router"] = "language_router"
        data["language"] = self.tgt_lang
        data["backend_key"] = self._backend_key
        data["backend_model"] = self._backend_model
        return data
