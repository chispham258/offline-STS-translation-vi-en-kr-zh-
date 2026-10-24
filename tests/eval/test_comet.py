"""COMET is optional and lazily imported (D7)."""

import pytest

from onevoice.eval.metrics.comet import comet, comet_available


def test_availability_check_does_not_raise():
    assert isinstance(comet_available(), bool)


def test_importing_our_module_does_not_import_comet_itself():
    # The lazy import is the whole point of D7. Importing our wrapper must not
    # drag in comet (and therefore torch, and therefore a 2.3 GB checkpoint).
    import sys

    if comet_available():
        pytest.skip("comet extra installed; cannot distinguish lazy from eager here")
    assert "comet" not in sys.modules


def test_raises_a_clear_error_when_unavailable():
    if comet_available():
        pytest.skip("comet extra is installed")
    with pytest.raises(RuntimeError, match="uv sync --extra comet"):
        comet(["nguồn"], ["hyp"], ["ref"])


@pytest.mark.slow
def test_scores_identical_text_highly():
    if not comet_available():
        pytest.skip("comet extra not installed")
    result = comet(
        ["Xin chào, anh khỏe không?"],
        ["Hello, how are you?"],
        ["Hello, how are you?"],
    )
    assert 0.0 <= result.value <= 1.0


def test_length_mismatch_is_rejected_before_availability_is_checked():
    # The guard must fire regardless of whether the extra is installed —
    # a caller with mismatched inputs has a bug worth reporting even on a
    # machine that could not have run COMET anyway.
    with pytest.raises(ValueError, match="length"):
        comet(["a", "b"], ["x"], ["y"])


def test_availability_probe_returns_false_for_a_broken_install(monkeypatch):
    # A comet that imports but explodes is as unusable as one that is absent.
    import builtins

    real_import = builtins.__import__

    def exploding_import(name, *args, **kwargs):
        if name == "comet":
            raise OSError("simulated broken native extension")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", exploding_import)
    assert comet_available() is False
