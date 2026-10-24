"""The knowledge ledger must keep the sections the phase-gate process relies on."""

import re
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "docs" / "KNOWLEDGE.md"

REQUIRED_SECTIONS = [
    "## Current status",
    "## Decision ledger",
    "## Assumed claims",
    "## Verified measurements",
    "## Open questions",
]


def _ids(text: str, prefix: str) -> list[int]:
    """IDs like D3 or Q3 found as the first cell of a ledger table row."""
    return sorted(int(n) for n in re.findall(rf"\| {prefix}(\d+) \|", text))


def test_ledger_exists():
    assert LEDGER.is_file()


def test_ledger_has_required_sections():
    text = LEDGER.read_text()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"missing sections: {missing}"


def test_every_decision_is_recorded():
    ids = _ids(LEDGER.read_text(), "D")
    assert ids, "no decisions found in ledger"
    expected = list(range(1, ids[-1] + 1))
    assert ids == expected, f"decision numbering has gaps: {ids}"


def test_every_open_question_is_recorded():
    ids = _ids(LEDGER.read_text(), "Q")
    assert ids, "no open questions found in ledger"
    expected = list(range(1, ids[-1] + 1))
    assert ids == expected, f"question numbering has gaps: {ids}"
