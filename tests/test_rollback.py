"""D26: every shipped config must actually run.

A rollback path that has never been executed is a claim, not a capability.
"""

from pathlib import Path

import pytest

import onevoice.components.dummy  # noqa: F401
from onevoice.benchmark import run_benchmark

REFS = "fixtures/mini_vi_en/refs.jsonl"

ROLLBACK_CONFIGS = sorted(str(p) for p in Path("configs").glob("rollback_*.yaml"))


def test_at_least_one_rollback_config_exists():
    assert ROLLBACK_CONFIGS


@pytest.mark.parametrize("config", ROLLBACK_CONFIGS)
def test_rollback_config_runs_and_scores(config, tmp_path):
    # Each config here names only dummy components, so this runs in CI with no
    # model extras installed. When a real component is integrated, its task adds
    # a mixed config that this same test picks up automatically.
    result = run_benchmark(config, REFS, tmp_path, experiment="rollback")
    assert result.table.counts["errored"] == 0
    assert result.table.counts["scored"] == 3
