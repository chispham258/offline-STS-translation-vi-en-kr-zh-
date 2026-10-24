"""Run the gate checks for a project phase.

Usage: uv run python scripts/phase_gate.py <phase>

Exits 0 only if every check for that phase passes. Phase definitions come from
docs/superpowers/specs/2026-08-01-onevoice-benchmark-framework-design.md and
their status is tracked in docs/KNOWLEDGE.md.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GATES: dict[int, list[list[str]]] = {
    0: [
        ["uv", "run", "python", "-c", "import onevoice"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pytest"],
    ],
    1: [
        ["uv", "run", "python", "-c", "import onevoice"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pytest"],
    ],
    2: [
        ["uv", "run", "python", "-c", "import onevoice"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pytest"],
    ],
    3: [
        ["uv", "run", "python", "-c", "import onevoice"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pytest"],
    ],
    # Phase 4 runs the same three checks deliberately. The phase integrated real
    # models, and the criterion it must meet is that the framework still imports,
    # lints and tests *without* the models extra - a wrapper that cannot be
    # skipped when its dependency is absent breaks CI for everyone.
    4: [
        ["uv", "run", "python", "-c", "import onevoice"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pytest"],
    ],
}

# Gate criteria from the design spec that this script has no way to check
# itself (e.g. they depend on infrastructure that doesn't exist yet, like a
# git remote). Future phases add their own entry here rather than touching
# main().
UNCHECKED: dict[int, list[str]] = {
    0: ["CI green on push (no git remote)"],
}


def gate_commands(phase: int) -> list[list[str]]:
    """Return the shell commands that make up a phase's gate.

    Raises ValueError if the phase has no defined gate.
    """
    if phase not in GATES:
        raise ValueError(f"no gate defined for phase {phase}")
    return GATES[phase]


def unchecked_criteria(phase: int) -> list[str]:
    """Return this phase's gate criteria that the script cannot check."""
    return UNCHECKED.get(phase, [])


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: phase_gate.py <phase>", file=sys.stderr)
        return 2

    try:
        phase = int(argv[1])
        commands = gate_commands(phase)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for cmd in commands:
        print(f"\n=== {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            print(f"\nGATE FAILED: {' '.join(cmd)}", file=sys.stderr)
            return 1

    message = f"\nGATE PASSED ({len(commands)} checks)"
    unchecked = unchecked_criteria(phase)
    if unchecked:
        message += f"; NOT CHECKED: {'; '.join(unchecked)}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
