"""T5-family MT wrappers: envit5 and MADLAD (D7 lazy import, D56 Gate D carriers).

Both are `t5` architecture and both need a **prefix on the source text** to
select the translation direction. That prefix is the danger: omit it and the
model still emits fluent text, just not the translation you asked for. The
failure is silent and looks like a bad model rather than a bad harness, which
is why the formats were pre-registered before scoring and why each wrapper
asserts its direction is expressible rather than defaulting.

- **envit5** (`VietAI/envit5-translation`) prefixes `"<src>: "`, e.g. `"vi: "`.
- **MADLAD** (`google/madlad400-*`) prefixes a target token `"<2xx> "`, e.g.
  `"<2en> "`.

Batch-shaped per D6, like every other MT wrapper: push() buffers, flush()
translates once.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from onevoice.interfaces import BaseMT
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import MtEvent

logger = logging.getLogger(__name__)

#: envit5 is trained on exactly this pair; anything else is out of scope.
ENVIT5_LANGS = ("vi", "en")

#: MADLAD target tokens. Raising beats guessing: a wrong or missing token
#: silently produces the wrong direction (same discipline as NLLB's FLORES map).
MADLAD_TARGET = {"en": "<2en>", "vi": "<2vi>", "zh": "<2zh>", "ko": "<2ko>"}


class _T5MT(BaseMT):
    """Shared buffering and generation; subclasses supply the prefix."""

    def __init__(self, model_name: str, src_lang: str, tgt_lang: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                f"{type(self).__name__} requires the mt extra, which is not "
                "installed. Install it with: uv sync --extra mt"
            ) from exc

        self._torch = torch
        self._model_name = model_name
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._prefix = self._build_prefix(src_lang, tgt_lang)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.eval()
        self._segments: list[str] = []

    def _build_prefix(self, src_lang: str, tgt_lang: str) -> str:
        raise NotImplementedError

    def push(self, text: str, is_final: bool = False) -> Sequence[MtEvent]:
        self._check_not_flushed()
        # Handle AsrEvent payload from ASR flush
        if hasattr(text, 'text'):
            text = text.text
        if text.strip():
            logger.debug("push: %.50r is_final=%s", text, is_final)
            self._segments.append(text)
        return []

    def flush(self) -> Sequence[MtEvent]:
        self._mark_flushed()
        logger.debug("flush: %s segments buffered", len(self._segments))
        if not self._segments:
            return []
        source = " ".join(self._segments)
        logger.debug("translating: %.50r", source)
        inputs = self._tokenizer(
            self._prefix + source, return_tensors="pt", truncation=True
        )
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=512)
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        logger.debug("translated: %.50r", text)
        return [MtEvent(text=self._clean(text), is_final=True, t_wall=0.0)]

    def _clean(self, text: str) -> str:
        """Strip any output-format tag the model emits.

        Default: nothing. Subclasses override where the model echoes a language
        tag into its output — that is a formatting convention, not translation
        content, and scoring it would penalise the model for its own output
        format exactly as raw WER penalised Whisper for punctuation (D49).

        The fix belongs in the wrapper, never in the corpus: the corpus stays
        identical for every candidate.
        """
        return text.strip()

    def reset(self) -> None:
        super().reset()
        self._segments = []

    def _provenance(self, component: str) -> dict[str, object]:
        return describe_hub_model(
            component=component,
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
                # Recorded because it is the silent-failure surface: a run whose
                # prefix was wrong is indistinguishable from a bad model unless
                # the prefix itself is in the record.
                "prefix": self._prefix,
            },
        )


@register("mt", "envit5")
class EnViT5MT(_T5MT):
    """`VietAI/envit5-translation` — en<->vi only."""

    def __init__(
        self,
        model_name: str = "VietAI/envit5-translation",
        src_lang: str = "vi",
        tgt_lang: str = "en",
    ) -> None:
        super().__init__(model_name, src_lang, tgt_lang)

    def _clean(self, text: str) -> str:
        """envit5 echoes the target tag, e.g. `"en: Hello"`. Verified by smoke
        test 2026-08-15 before any scoring."""
        out = text.strip()
        for code in ENVIT5_LANGS:
            tag = f"{code}: "
            if out.startswith(tag):
                return out[len(tag) :].strip()
        return out

    def _build_prefix(self, src_lang: str, tgt_lang: str) -> str:
        for code, role in ((src_lang, "source"), (tgt_lang, "target")):
            if code not in ENVIT5_LANGS:
                raise ValueError(
                    f"envit5 supports only {list(ENVIT5_LANGS)}; got {role} "
                    f"{code!r}. It cannot serve this direction, and producing "
                    "fluent output for it would be worse than refusing."
                )
        return f"{src_lang}: "

    def provenance(self) -> dict[str, object]:
        return self._provenance("envit5")


@register("mt", "madlad")
class MadladMT(_T5MT):
    """`google/madlad400-*` — target-token prefixed."""

    def __init__(
        self,
        model_name: str = "google/madlad400-3b-mt",
        src_lang: str = "vi",
        tgt_lang: str = "en",
    ) -> None:
        super().__init__(model_name, src_lang, tgt_lang)

    def _build_prefix(self, src_lang: str, tgt_lang: str) -> str:
        try:
            token = MADLAD_TARGET[tgt_lang]
        except KeyError:
            raise ValueError(
                f"no MADLAD target token mapped for {tgt_lang!r}; known: "
                f"{sorted(MADLAD_TARGET)}. A missing token silently translates "
                "into the wrong language."
            ) from None
        return f"{token} "

    def provenance(self) -> dict[str, object]:
        return self._provenance("madlad")
