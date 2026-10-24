"""PivotMT: composition of two BaseMT legs (D12).

The two tests below use dummy legs on purpose: a pivot is a BaseMT composing
two BaseMT instances, and proving the composition with dummies means it runs
in CI, not only when a multi-gigabyte model happens to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence

from onevoice.contracts import MTContract
from onevoice.interfaces import BaseMT
from onevoice.registry import resolve
from onevoice.types import MtEvent


class BatchLeg(BaseMT):
    """A batch-shaped MT double: buffers on push, emits TAG[...] on flush."""

    def __init__(self, tag):
        self.tag, self._buf = tag, []

    def push(self, text, is_final=False) -> Sequence[MtEvent]:
        self._check_not_flushed()
        self._buf.append(text)
        return []

    def flush(self) -> Sequence[MtEvent]:
        self._mark_flushed()
        out = " ".join(self._buf)
        self._buf = []
        if not out:
            return []
        return [MtEvent(text=f"{self.tag}[{out}]", t_wall=0.0, is_final=True)]

    def reset(self) -> None:
        super().reset()
        self._buf = []


# --- structural: these run in CI with no model installed ---------------------


def test_wrapper_module_imports_without_transformers():
    import onevoice.components.pivot_mt  # noqa: F401


def test_registered_under_its_config_name():
    import onevoice.components.pivot_mt as module

    assert resolve("mt", "pivot") is module.PivotMT


# --- composition logic: dummy legs, no model needed ---------------------------


def test_pivot_composes_two_legs_without_any_model():
    # D12: a pivot is a BaseMT composing two BaseMT instances. Proving that
    # with dummies means the composition logic is tested in CI, not only
    # when a multi-gigabyte model happens to be installed.
    from onevoice.components.dummy import IdentityMT
    from onevoice.components.pivot_mt import PivotMT

    pivot = PivotMT(first=IdentityMT(), second=IdentityMT())
    events = list(pivot.push("xin chào", is_final=True)) + list(pivot.flush())
    assert [e.text for e in events] == ["xin chào"]


def test_pivot_flushes_both_legs():
    # The Phase 3 final review caught the driver never calling mt.flush(); a
    # pivot that forgets its second leg reproduces that bug one level down.
    from onevoice.components.pivot_mt import PivotMT

    pivot = PivotMT(first=BatchLeg("A"), second=BatchLeg("B"))
    pivot.push("x", is_final=True)
    events = list(pivot.flush())
    assert len(events) == 1
    assert events[0].text == "B[A[x]]"


def test_pivot_reset_resets_both_legs():
    from onevoice.components.pivot_mt import PivotMT

    pivot = PivotMT(first=BatchLeg("A"), second=BatchLeg("B"))
    pivot.push("x", is_final=True)
    pivot.flush()
    pivot.reset()

    pivot.push("y", is_final=True)
    events = list(pivot.flush())
    assert len(events) == 1
    assert events[0].text == "B[A[y]]"


def test_pivot_builds_legs_from_nested_config_dicts():
    # The config-driven path: build() forwards raw {"model", "params"} specs
    # (what a YAML config's nested params deserialize to) rather than
    # already-built instances. Uses "identity" so this needs no model extra.
    from onevoice.components.pivot_mt import PivotMT

    pivot = PivotMT(
        first={"model": "identity", "params": {}},
        second={"model": "identity", "params": {}},
    )
    events = list(pivot.push("xin chào", is_final=True)) + list(pivot.flush())
    assert [e.text for e in events] == ["xin chào"]


class TestPivotMTContract(MTContract):
    # D12: PivotMT must pass the same streaming contract as any other MT
    # component. Dummy legs keep this running with no model installed.
    def make_component(self):
        from onevoice.components.dummy import IdentityMT
        from onevoice.components.pivot_mt import PivotMT

        return PivotMT(first=IdentityMT(), second=IdentityMT())
