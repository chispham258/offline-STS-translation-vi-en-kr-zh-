"""m2m100 and NLLB (Branch B, many-to-many MT), behind an optional extra
with a lazy import (D7), same pattern as opus_mt.py.

The language-code trap: m2m100 uses ISO-639-1 codes (vi/en/zh/ko), which are
exactly this framework's `Language` values, so `M2M100MT` needs no mapping -
`src_lang`/`tgt_lang` are forwarded to the tokenizer as-is. NLLB instead uses
FLORES-200 codes (vie_Latn, eng_Latn, zho_Hans, kor_Hang), so `NLLBMT` maps
through `nllb_flores_code()`, which raises `ValueError` naming the language
for anything not in the four-entry table - never a fallback default, since a
wrong forced_bos_token_id produces fluent output in the wrong language,
which scores near zero and looks like a bad model rather than a bad config.
"""

from __future__ import annotations

from collections.abc import Sequence

from onevoice.interfaces import BaseMT
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import MtEvent

NLLB_FLORES_200 = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}


def nllb_flores_code(language: str) -> str:
    """Map a framework `Language` code to its FLORES-200 code for NLLB.

    Raises ValueError naming the language if it has no mapping, rather than
    falling back to a default - see module docstring.
    """
    try:
        return NLLB_FLORES_200[language]
    except KeyError:
        raise ValueError(
            f"NLLBMT has no FLORES-200 mapping for language {language!r}; "
            f"expected one of {sorted(NLLB_FLORES_200)}"
        ) from None


@register("mt", "m2m100")
class M2M100MT(BaseMT):
    """Batch-shaped per D6, same as OpusMT: push() buffers, flush() translates.

    src_lang/tgt_lang are m2m100's own ISO-639-1 codes (ports directly from
    this framework's Language values, no mapping needed). The target
    language is set via forced_bos_token_id, not a tokenizer argument -
    tokenizer.get_lang_id(tgt_lang) resolves the token id.
    """

    def __init__(
        self,
        model_name: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> None:
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "m2m100 requires the mt extra, which is not installed. "
                "Install it with: uv sync --extra mt"
            ) from exc

        self._tgt_lang = tgt_lang
        # Recorded as requested, for provenance (D46): the tokenizer stores a
        # resolved form, and reproducing a run means re-issuing the request.
        self._src_lang = src_lang
        self._tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        if src_lang is not None:
            self._tokenizer.src_lang = src_lang
        self._model_name = model_name
        self._model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self._model.eval()
        self._segments: list[str] = []

    def provenance(self) -> dict[str, object]:
        """D46: which multilingual checkpoint translated this run."""
        return describe_hub_model(
            component="m2m100",
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
        forced_bos_token_id = self._tokenizer.get_lang_id(self._tgt_lang)
        generated = self._model.generate(
            **batch, forced_bos_token_id=forced_bos_token_id
        )
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return [MtEvent(text=text.strip(), t_wall=0.0, is_final=True)]

    def reset(self) -> None:
        super().reset()
        self._segments = []


@register("mt", "nllb")
class NLLBMT(BaseMT):
    """Batch-shaped per D6, same as OpusMT: push() buffers, flush() translates.

    src_lang/tgt_lang arrive as this framework's Language codes and are
    mapped to FLORES-200 via nllb_flores_code() in __init__, before any
    tokenizer or model is loaded - an unmapped language fails fast with no
    download attempted. The target language is set via forced_bos_token_id
    (tokenizer.convert_tokens_to_ids on the FLORES code), not a plain
    tokenizer argument.
    """

    def __init__(
        self,
        model_name: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLLB requires the mt extra, which is not installed. "
                "Install it with: uv sync --extra mt"
            ) from exc

        self._tgt_flores = (
            nllb_flores_code(tgt_lang) if tgt_lang is not None else None
        )
        src_flores = nllb_flores_code(src_lang) if src_lang is not None else None

        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if src_flores is not None:
            self._tokenizer.src_lang = src_flores
        self._model_name = model_name
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.eval()
        self._segments: list[str] = []

    def provenance(self) -> dict[str, object]:
        """D46: which multilingual checkpoint translated this run."""
        return describe_hub_model(
            component="nllb",
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
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(
            self._tgt_flores
        )
        generated = self._model.generate(
            **batch, forced_bos_token_id=forced_bos_token_id
        )
        text = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        return [MtEvent(text=text.strip(), t_wall=0.0, is_final=True)]

    def reset(self) -> None:
        super().reset()
        self._segments = []
