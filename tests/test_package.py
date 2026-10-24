"""Phase 0 smoke test: the package exists and is importable."""

import re

import onevoice


def test_package_imports():
    assert onevoice is not None


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", onevoice.__version__)
