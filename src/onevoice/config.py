"""Typed experiment configuration loaded from YAML.

Parsing and resolution are deliberately separate. load_config validates
structure only and never touches the registry, so a config can be checked in CI
without importing a single model wrapper. build() is the explicit second step.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from onevoice.registry import resolve
from onevoice.types import Language


class _Strict(BaseModel):
    """Rejects unknown keys. A typo'd section must not be silently ignored."""

    model_config = ConfigDict(extra="forbid")


#: What a benchmark's results may be used to claim (F3, D48). See
#: docs/INTERPRETATION.md - the classification governs what may be concluded,
#: not how the benchmark is run.
CLASSIFICATIONS = ("smoke", "development", "selection")


def _check_classification(value: str) -> str:
    if value not in CLASSIFICATIONS:
        raise ValueError(
            f"unknown classification {value!r}; expected one of "
            f"{list(CLASSIFICATIONS)}. See docs/INTERPRETATION.md."
        )
    return value


class ExperimentMeta(_Strict):
    name: str
    classification: str = "development"
    """Smoke, development or selection.

    Defaults to `development`: the honest default is the middle one. Defaulting
    to `smoke` would let a real comparison disclaim itself, and defaulting to
    `selection` would let any config claim ranking authority it has not earned.

    A `selection` report must carry interval evidence for every comparison it
    makes - that is what distinguishes it from `development`, where a bolded
    winning column would be a ranking claim made typographically.
    """

    @field_validator("classification")
    @classmethod
    def _validate_classification(cls, value: str) -> str:
        return _check_classification(value)

    notes: str = ""
    """A caveat that must travel with this config's numbers.

    Rendered into every benchmark report produced from this config. A config
    whose results are not what a reader would assume - a degenerate pivot, a
    development-only model size - says so here, so the qualification cannot be
    separated from the figure it qualifies.
    """


class LanguagePair(_Strict):
    src: Language
    tgt: Language


class ComponentConfig(_Strict):
    model: str
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(_Strict):
    """Optional async pipeline configuration."""
    enabled: bool = False
    queue_sizes: dict[str, int] = Field(default_factory=dict)


class ExperimentConfig(_Strict):
    experiment: ExperimentMeta
    languages: LanguagePair
    vad: ComponentConfig
    asr: ComponentConfig
    mt: ComponentConfig
    tts: ComponentConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate a YAML experiment config."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config must be a YAML mapping, got {data!r}")
    return ExperimentConfig(**data)


def build(
    kind: str, cfg: ComponentConfig, languages: LanguagePair | None = None
) -> object:
    """Instantiate the component named by a config section.

    Pass `languages` for components that need the pair directly (Opus-MT
    needs src and tgt; MeloTTS needs tgt for voice selection), rather than
    duplicating the codes into `cfg.params` where they could disagree with
    `languages` and nothing would catch it.
    """
    cls = resolve(kind, cfg.model)
    kwargs = dict(cfg.params)
    if languages is not None:
        for key in ("src_lang", "tgt_lang"):
            if key in kwargs:
                raise ValueError(
                    f"{kind} component {cfg.model!r} params already set "
                    f"{key!r}; a component must not receive the language "
                    "twice from two sources that could disagree"
                )
        accepted = _accepted_language_kwargs(cls)
        if "src_lang" in accepted:
            kwargs["src_lang"] = languages.src
        if "tgt_lang" in accepted:
            kwargs["tgt_lang"] = languages.tgt
    return cls(**kwargs)


def _accepted_language_kwargs(cls: type) -> set[str]:
    """Which of src_lang/tgt_lang cls's constructor accepts, named or **kwargs.

    Each is forwarded independently: a Whisper wrapper taking only src_lang
    must still receive it, rather than getting neither because it doesn't
    also accept tgt_lang.

    Inspecting the class itself, rather than cls.__init__ directly, matters
    for a class with no __init__ of its own: cls.__init__ then resolves to
    object.__init__, whose signature spuriously reports (*args, **kwargs).
    """
    params = inspect.signature(cls).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return {"src_lang", "tgt_lang"}
    return {name for name in ("src_lang", "tgt_lang") if name in params}
