"""BLEU and chrF.

D14: the BLEU tokenizer is chosen by target language. sacreBLEU's default 13a
splits on whitespace, which is meaningless for unsegmented Chinese and wrong for
Korean morphology. A wrong tokenizer does not fail loudly - it returns a
plausible, stable, fabricated number - so an unknown language raises instead.
"""

from __future__ import annotations

import sacrebleu

from onevoice.eval.metrics.base import MetricResult

_TOKENIZERS = {
    "vi": "13a",
    "en": "13a",
    "zh": "zh",
    "ko": "ko-mecab",
}


def tokenizer_for(tgt_lang: str) -> str:
    """Return the sacreBLEU tokenizer for a target language."""
    try:
        return _TOKENIZERS[tgt_lang]
    except KeyError:
        supported = ", ".join(sorted(_TOKENIZERS))
        raise ValueError(
            f"no BLEU tokenizer configured for target language {tgt_lang!r}; "
            f"supported: {supported}"
        ) from None


def _check_lengths(hyps: list[str], refs: list[str]) -> None:
    if len(hyps) != len(refs):
        raise ValueError(
            f"hypothesis/reference length mismatch: {len(hyps)} vs {len(refs)}"
        )


_OPTIONAL_TOKENIZER_EXTRAS = {"ko-mecab": "ko"}


def bleu(hyps: list[str], refs: list[str], tgt_lang: str) -> MetricResult:
    """Corpus BLEU via sacreBLEU, tokenized per D14."""
    _check_lengths(hyps, refs)
    tokenizer = tokenizer_for(tgt_lang)
    try:
        score = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenizer)
    except Exception as exc:
        # sacreBLEU's own tokenizers (e.g. TokenizerKoMecab) catch ImportError
        # internally and raise a bare RuntimeError with a pip-install message,
        # so a handler scoped to (ImportError, ModuleNotFoundError) never
        # fires for the case it exists for. Catch broadly instead, and
        # re-raise with this repo's install command while keeping sacreBLEU's
        # own diagnosis visible via chaining.
        extra = _OPTIONAL_TOKENIZER_EXTRAS.get(tokenizer)
        if extra is not None:
            remedy = f"Install it with: uv sync --extra {extra}."
        else:
            remedy = (
                "Falling back to a different tokenizer would produce a "
                "plausible but meaningless score, so this raises instead."
            )
        raise RuntimeError(
            f"BLEU for {tgt_lang!r} using the {tokenizer!r} tokenizer failed: "
            f"{exc}. {remedy}"
        ) from exc
    return MetricResult(
        name="bleu",
        value=score.score,
        detail={"tokenizer": tokenizer, "signature": str(score)},
    )


def chrf(hyps: list[str], refs: list[str]) -> MetricResult:
    """Corpus chrF. Character-based, so it needs no tokenizer."""
    _check_lengths(hyps, refs)
    score = sacrebleu.corpus_chrf(hyps, [refs])
    return MetricResult(name="chrf", value=score.score, detail={})
