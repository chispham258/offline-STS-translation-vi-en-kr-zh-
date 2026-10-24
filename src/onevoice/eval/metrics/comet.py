"""COMET, behind an optional extra with a lazy import (D7).

unbabel-comet pulls torch, pytorch-lightning, and a ~2.3 GB checkpoint. Importing
it at module level would make the entire eval package unimportable without them,
so every import here is inside a function.
"""

from __future__ import annotations

from functools import lru_cache

from onevoice.eval.metrics.base import MetricResult

DEFAULT_MODEL = "Unbabel/wmt22-comet-da"


def comet_available() -> bool:
    """True when the comet extra is installed. Never raises."""
    try:
        import comet  # noqa: F401
    except Exception:
        # A broken install (incompatible torch/pytorch-lightning pairing,
        # a corrupt native extension, ...) is as unusable as an absent one.
        # This is an availability probe: callers branch on the bool result,
        # so it must never raise, regardless of how comet fails to import.
        return False
    return True


@lru_cache(maxsize=1)
def _load(model_name: str):
    from comet import download_model, load_from_checkpoint

    return load_from_checkpoint(download_model(model_name))


def comet(
    srcs: list[str],
    hyps: list[str],
    refs: list[str],
    model_name: str = DEFAULT_MODEL,
) -> MetricResult:
    """Corpus COMET. Raises if the optional extra is not installed."""
    if not (len(srcs) == len(hyps) == len(refs)):
        raise ValueError(
            f"length mismatch: srcs={len(srcs)} hyps={len(hyps)} refs={len(refs)}"
        )
    if not comet_available():
        raise RuntimeError(
            "COMET is not installed. Install it with: uv sync --extra comet. "
            "It is optional because it pulls torch and a ~2.3 GB checkpoint."
        )
    model = _load(model_name)
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(srcs, hyps, refs, strict=True)
    ]
    output = model.predict(data, progress_bar=False)
    return MetricResult(
        name="comet", value=float(output.system_score), detail={"model": model_name}
    )
