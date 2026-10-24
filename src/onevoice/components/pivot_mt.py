"""PivotMT: a BaseMT composing two BaseMT legs (D12).

D12: a pivot (vi -> en -> zh) is implemented as a BaseMT composing two BaseMT
instances, not as branching in the pipeline driver, so the driver never
learns that a hop happened - it just sees one mt component with the usual
push()/flush()/reset() shape.

Config design decision: `first`/`second` each accept either an
already-constructed BaseMT (what the dummy-leg tests below pass directly) or
a `{"model": ..., "params": {...}}` mapping - exactly what a YAML config's
nested `params` deserialize to, since onevoice.config.ComponentConfig.params
is a plain `dict[str, Any]`. A leg spec is built via onevoice.config.build(),
the same function that builds every other component, rather than duplicating
its resolve()-plus-construct logic here. This was chosen over "two model
names plus a shared class" because the two legs of a pivot are not
guaranteed to be the same wrapper class - D13 lists m2m100 as a competing
candidate for pivot directions, so a leg must be able to name any registered
mt component, not just another OpusMT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from onevoice.interfaces import BaseMT
from onevoice.provenance import params_hash
from onevoice.registry import register
from onevoice.types import MtEvent


def _build_leg(spec: BaseMT | Mapping[str, Any]) -> BaseMT:
    if isinstance(spec, BaseMT):
        return spec
    from onevoice.config import ComponentConfig, build

    return build("mt", ComponentConfig(**spec))


@register("mt", "pivot")
class PivotMT(BaseMT):
    """Chains two MT legs: every event the first leg emits feeds the second.

    push() forwards straight through the first leg and, for any event that
    produces (a streaming leg may emit immediately), on into the second leg.
    flush() must do the same for whatever the first leg was still holding -
    flush the first leg, feed its output into the second leg, then flush the
    second leg too and return what it produced. Skipping that second flush
    reproduces, one level down, the driver-forgets-mt.flush() bug the Phase 3
    final review caught: a batch-shaped first leg would hold everything
    until flush() and the pivot would silently emit nothing.
    """

    def __init__(
        self,
        first: BaseMT | Mapping[str, Any],
        second: BaseMT | Mapping[str, Any],
    ) -> None:
        self._first = _build_leg(first)
        self._second = _build_leg(second)

    def provenance(self) -> dict[str, object]:
        """D46: a pivot run's identity is BOTH legs, not the composite.

        `PivotMT` owns no weights, so it has no anchor of its own. Its anchor
        is derived from its legs': a sha256 over their immutable anchors, which
        changes if either leg changes. Recording only "pivot" would identify
        nothing, and recording one leg would silently hide the other - the
        exact failure D13's routing makes easy to introduce.
        """
        import hashlib

        legs = {}
        for label, leg in (("first", self._first), ("second", self._second)):
            describe = getattr(leg, "provenance", None)
            legs[label] = (
                describe()
                if callable(describe)
                else {"component": type(leg).__name__}
            )

        anchors = "".join(
            str(legs[label].get("commit_hash") or legs[label].get("weight_hash") or "")
            for label in ("first", "second")
        )
        return {
            "component": "pivot",
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "model_id": (
                f"pivot({legs['first'].get('model_id', '?')}"
                f" -> {legs['second'].get('model_id', '?')})"
            ),
            "revision": "composite",
            # Derived from the legs: changes whenever either leg does.
            "weight_hash": hashlib.sha256(anchors.encode("utf-8")).hexdigest(),
            "params_hash": params_hash(
                {
                    "first": legs["first"].get("params_hash"),
                    "second": legs["second"].get("params_hash"),
                }
            ),
            "invocation": {
                "route": "pivot",
                "first": legs["first"].get("invocation", {}),
                "second": legs["second"].get("invocation", {}),
            },
            "legs": legs,
        }

    def push(self, text: str, is_final: bool = False) -> Sequence[MtEvent]:
        self._check_not_flushed()
        events: list[MtEvent] = []
        for event in self._first.push(text, is_final=is_final):
            events.extend(self._second.push(event.text, is_final=event.is_final))
        return events

    def flush(self) -> Sequence[MtEvent]:
        self._mark_flushed()
        events: list[MtEvent] = []
        for event in self._first.flush():
            events.extend(self._second.push(event.text, is_final=event.is_final))
        events.extend(self._second.flush())
        return events

    def reset(self) -> None:
        super().reset()
        self._first.reset()
        self._second.reset()
