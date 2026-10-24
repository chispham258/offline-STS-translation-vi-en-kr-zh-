"""The CI workflow must actually run the Phase 0 gate commands."""

from pathlib import Path

import yaml

from scripts.phase_gate import gate_commands

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _run_commands() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["test"]["steps"]
    return "\n".join(step.get("run", "") for step in steps)


def test_workflow_file_exists():
    assert WORKFLOW.is_file()


def test_workflow_runs_ruff():
    assert "ruff check" in _run_commands()


def test_workflow_runs_pytest():
    assert "pytest" in _run_commands()


def test_workflow_python_matches_pinned_version():
    pinned = (WORKFLOW.parents[2] / ".python-version").read_text().strip()
    assert pinned in _run_commands()


def test_gate_commands_have_a_ci_counterpart():
    """Every phase-0 gate command must be represented in the CI workflow.

    The gate and the workflow encode the same checks independently, so they
    can drift: a command added to one and not the other would go unnoticed.
    Comparing the joined gate command against the workflow's `run:` text
    catches that drift.

    Exempt: `import onevoice`. It's a local dev-shell smoke check with no
    workflow counterpart by design — `uv sync` + `uv run pytest` already
    exercise the package import in CI.
    """
    run_text = _run_commands()
    for cmd in gate_commands(0):
        joined = " ".join(cmd)
        if "import onevoice" in joined:
            continue
        assert joined in run_text, f"gate command has no CI counterpart: {joined}"
