"""The stage-2 commit gate (§10): the incumbent-only resolution must be committed,
unmodified, before any candidate comparison runs."""

from __future__ import annotations

import subprocess

import pytest

from scripts.analyse_mos_panel import require_committed


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    return tmp_path


def test_an_uncommitted_resolution_is_refused(repo):
    (repo / "res.json").write_text("{}")
    with pytest.raises(SystemExit, match="not committed"):
        require_committed(repo / "res.json")


def test_a_committed_resolution_passes(repo):
    (repo / "res.json").write_text("{}")
    _git(repo, "add", "res.json")
    _git(repo, "commit", "-qm", "r")
    require_committed(repo / "res.json")


def test_a_modified_resolution_is_refused(repo):
    (repo / "res.json").write_text("{}")
    _git(repo, "add", "res.json")
    _git(repo, "commit", "-qm", "r")
    (repo / "res.json").write_text('{"resolution": 0.01}')
    with pytest.raises(SystemExit, match="modified"):
        require_committed(repo / "res.json")
