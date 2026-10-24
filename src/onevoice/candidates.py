"""Candidate registry and Gate D (D56, D57).

**Registration is not selection.** This registry records what a candidate *is* —
family, coverage, interface fit, deployment metadata — and holds no ranking and
no measurement. Scores live in run manifests, which are already provenanced
(D46/D47); copying them here would create a second source of truth that can
drift from the archives, and the archives win.

Deliberately separate from `onevoice.registry`, which resolves `kind -> name ->
class` so the pipeline can build components. That one is runtime machinery; this
one is an evaluation artifact. Merging them would make the pipeline depend on
evaluation bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SLOTS = ("vad", "asr", "mt", "tts")

#: Deployment progress. `unevaluated` is the default and the point: "we have not
#: checked" must never read as "it works".
DEPLOYMENT_STATUSES = (
    "unevaluated",
    "export_untested",
    "export_ok",
    "runtime_ok",
    "device_validated",
)

#: Slots whose candidates must handle Vietnamese to be in the matrix at all.
#: VAD is exempt: it detects speech, not language.
_VIETNAMESE_REQUIRED = ("asr", "tts")
_VI = "vi"


@dataclass(frozen=True)
class DeploymentMeta:
    """What we know about deploying a candidate. **Never a selection criterion.**

    Recorded so a later deployment measurement is reproducible (D57), and kept
    apart from model identity so that a model cannot lose merely because
    Qualcomm has not published an optimised artifact for it (D54).
    """

    deployment_reference: str = ""
    ai_hub_available: bool | None = None
    export_path: str = ""
    runtime: str = ""
    precision: str = ""
    status: str = "unevaluated"

    def __post_init__(self) -> None:
        if self.status not in DEPLOYMENT_STATUSES:
            raise ValueError(
                f"unknown deployment status {self.status!r}; expected one of "
                f"{list(DEPLOYMENT_STATUSES)}"
            )


#: How a language-coverage claim was established, weakest to strongest.
EVIDENCE_METHODS = (
    "unverified",          # nothing checked
    "model_name",          # inferred from the checkpoint name, e.g. "mms-tts-vie"
    "model_card_prose",    # claimed in prose on the card
    "hub_metadata",        # structured `language:` tag on the Hub
    "config_inspected",    # read out of the model config / tokenizer
    "output_tested",       # we ran it and listened to / read the output
)


@dataclass(frozen=True)
class Evidence:
    """Why we believe a candidate's language and architecture claims (D58).

    "Vietnamese supported" has to mean something checkable. A model *name*
    ending in `-vie` is weaker evidence than a structured Hub `language:` tag,
    which is weaker than having run the thing. Recording which one we have stops
    a card assumption from hardening into a fact.
    """

    language_method: str = "unverified"
    language_source: str = ""
    architecture_source: str = ""
    verified_on: str = ""

    def __post_init__(self) -> None:
        if self.language_method not in EVIDENCE_METHODS:
            raise ValueError(
                f"unknown evidence method {self.language_method!r}; expected one "
                f"of {list(EVIDENCE_METHODS)}"
            )

    @property
    def is_verified(self) -> bool:
        """Anything at or above a structured Hub tag counts as verified."""
        return self.language_method in (
            "hub_metadata",
            "config_inspected",
            "output_tested",
        )


@dataclass(frozen=True)
class Coverage:
    """Three different questions, deliberately not merged (D59).

    - `claimed` — what the candidate says it supports. A capability, recorded
      at registration.
    - `evaluable` — what our corpus can actually test. A property of *us*, not
      of the model: a direction we cannot evaluate is our gap, not the
      candidate's fault.
    - `deployable` — what may eventually run on target. Empty until Track C
      measures it; empty means unevaluated, never "none".

    Conflating these is how a model gets eliminated for breadth it was never
    asked for, or selected for breadth nobody measured.
    """

    claimed: frozenset[str] = frozenset()
    evaluable: frozenset[str] = frozenset()
    deployable: frozenset[str] = frozenset()

    def evaluable_within(self, corpus_directions: frozenset[str]) -> Coverage:
        """What of this candidate's claim our corpus can actually exercise."""
        return Coverage(
            claimed=self.claimed,
            evaluable=self.claimed & corpus_directions,
            deployable=self.deployable,
        )

    @property
    def unevaluable(self) -> frozenset[str]:
        """Claimed but not testable here — recorded, never silently dropped."""
        return self.claimed - self.evaluable


@dataclass(frozen=True)
class Candidate:
    """One model under consideration for one slot.

    `family` is the field Gate D operates on and the one most easily got wrong:
    it means an architecture, not a checkpoint. A fine-tune or a size variant is
    the *same* family — `PhoWhisper` is a Whisper fine-tune, so it belongs to
    `whisper` however Vietnamese-specialised it is.
    """

    slot: str
    name: str
    family: str
    checkpoint: str
    revision: str = "default"
    params_m: float | None = None
    #: ASR: input languages. TTS: output languages. Unused for vad/mt.
    languages: frozenset[str] = frozenset()
    #: MT only, e.g. {"vi->en"}.
    directions: frozenset[str] = frozenset()
    license: str = ""
    interface: str = ""
    strengths: str = ""
    weaknesses: str = ""
    deployment: DeploymentMeta = field(default_factory=DeploymentMeta)
    evidence: Evidence = field(default_factory=Evidence)
    #: Capability dimension, kept separate from quality (D59). Never a
    #: selection criterion on its own: breadth is not merit, and narrowness is
    #: not a defect.
    coverage: Coverage = field(default_factory=Coverage)

    def __post_init__(self) -> None:
        if self.slot not in SLOTS:
            raise ValueError(
                f"unknown slot {self.slot!r}; expected one of {list(SLOTS)}"
            )

        # G1 is exclusionary, not a low score. A candidate that cannot handle
        # Vietnamese is not in the matrix, because scoring it would imply it
        # could win.
        if self.slot in _VIETNAMESE_REQUIRED and _VI not in self.languages:
            direction = "transcribe" if self.slot == "asr" else "speak"
            raise ValueError(
                f"{self.slot} candidate {self.name!r} cannot {direction} "
                f"Vietnamese (languages={sorted(self.languages)}). Vietnamese is "
                "a hard product requirement, so such a candidate is excluded "
                "from the matrix rather than scored poorly."
            )

        if self.slot == "mt" and self.directions:
            if not any(_VI in d.split("->") for d in self.directions):
                raise ValueError(
                    f"mt candidate {self.name!r} covers no Vietnamese direction "
                    f"({sorted(self.directions)}); Vietnamese is a hard product "
                    "requirement"
                )


class GateDError(AssertionError):
    """A slot selection was asserted on a matrix that fails Gate D."""


@dataclass(frozen=True)
class GateDResult:
    """Whether a slot's matrix can support a selection claim at all."""

    slot: str
    incumbent_family: str
    families: set[str]
    outside_incumbent: set[str]

    @property
    def passes(self) -> bool:
        """Two distinct families, at least one outside the incumbent's."""
        return len(self.families) >= 2 and bool(self.outside_incumbent)

    def explain(self) -> str:
        if self.passes:
            return (
                f"Gate D passes for {self.slot}: families "
                f"{sorted(self.families)}, with {sorted(self.outside_incumbent)} "
                f"outside the incumbent family {self.incumbent_family!r}."
            )
        return (
            f"Gate D FAILS for {self.slot}: families {sorted(self.families)}, "
            f"incumbent {self.incumbent_family!r}. A result from this matrix is "
            f"'best within family {self.incumbent_family}' and must not be "
            "recorded as a slot selection (D56)."
        )


class CandidateRegistry:
    """Candidates per slot, in registration order.

    Insertion order is preserved and never sorted: any ordering here would read
    as a ranking, and this registry deliberately expresses none.
    """

    def __init__(self) -> None:
        self._by_slot: dict[str, list[Candidate]] = {s: [] for s in SLOTS}

    def add(self, candidate: Candidate) -> Candidate:
        existing = self._by_slot[candidate.slot]
        if any(c.name == candidate.name for c in existing):
            raise KeyError(
                f"candidate {candidate.name!r} is already registered for slot "
                f"{candidate.slot!r}; a silent overwrite would mean the matrix "
                "does not contain what someone believes it does"
            )
        existing.append(candidate)
        return candidate

    def for_slot(self, slot: str) -> list[Candidate]:
        if slot not in self._by_slot:
            raise ValueError(f"unknown slot {slot!r}; expected one of {list(SLOTS)}")
        return list(self._by_slot[slot])

    def gate_d(self, slot: str, incumbent_family: str) -> GateDResult:
        """Evaluate Gate D for one slot. Slots are independent (D56)."""
        families = {c.family for c in self.for_slot(slot)}
        return GateDResult(
            slot=slot,
            incumbent_family=incumbent_family,
            families=families,
            outside_incumbent={f for f in families if f != incumbent_family},
        )

    def require_gate_d(self, slot: str, incumbent_family: str) -> GateDResult:
        """Raise unless the matrix can support a selection claim.

        The gate has to bite rather than merely report: a warning nobody reads
        is how a single-family matrix produces a conclusion that was available
        before the experiment ran.
        """
        result = self.gate_d(slot, incumbent_family)
        if not result.passes:
            raise GateDError(result.explain())
        return result
