"""The phase gate must know what each phase's checks are.

Only the pure command-builder is tested. main() runs pytest, and invoking it
from inside pytest would recurse.
"""

import pytest

from scripts.phase_gate import gate_commands, unchecked_criteria


def test_phase_0_checks_lint_and_tests():
    flat = [" ".join(cmd) for cmd in gate_commands(0)]
    assert any("ruff check" in c for c in flat)
    assert any("pytest" in c for c in flat)


def test_phase_0_checks_the_package_imports():
    flat = [" ".join(cmd) for cmd in gate_commands(0)]
    assert any("import onevoice" in c for c in flat)


def test_every_command_is_a_list_of_strings():
    for cmd in gate_commands(0):
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)


def test_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="99"):
        gate_commands(99)


def test_phase_0_declares_its_unchecked_criteria():
    unchecked = unchecked_criteria(0)
    assert any("CI green on push" in c for c in unchecked)


def test_phase_2_has_a_gate():
    flat = [" ".join(cmd) for cmd in gate_commands(2)]
    assert any("ruff check" in c for c in flat)
    assert any("pytest" in c for c in flat)


def test_phase_2_has_no_unchecked_criteria():
    # Unlike Phase 0, every Phase 2 criterion is runnable locally.
    assert unchecked_criteria(2) == []


def test_phase_3_has_a_gate():
    flat = [" ".join(cmd) for cmd in gate_commands(3)]
    assert any("ruff check" in c for c in flat)
    assert any("pytest" in c for c in flat)


def test_phase_3_has_no_unchecked_criteria():
    assert unchecked_criteria(3) == []


def test_phase_4_has_a_gate():
    flat = [" ".join(cmd) for cmd in gate_commands(4)]
    assert any("ruff check" in c for c in flat)
    assert any("pytest" in c for c in flat)


def test_phase_4_has_no_unchecked_criteria():
    # Phase 0's git-remote criterion is long since satisfied, and every Phase 4
    # criterion is runnable locally.
    assert unchecked_criteria(4) == []
