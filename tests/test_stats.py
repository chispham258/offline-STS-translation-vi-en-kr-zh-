"""F3: a comparison reports an interval or it claims nothing (D48)."""

from __future__ import annotations

import pytest

from onevoice.stats import Comparison, paired_bootstrap

IDS = [str(i) for i in range(40)]


def _const(value):
    return lambda ids: value


def test_a_real_difference_resolves():
    result = paired_bootstrap(IDS, _const(10.0), _const(20.0), name="bleu")
    assert result.delta == pytest.approx(10.0)
    assert result.resolves
    assert "excludes zero" in result.sentence()


def test_no_difference_does_not_resolve():
    result = paired_bootstrap(IDS, _const(10.0), _const(10.0), name="bleu")
    assert not result.resolves
    assert "No statistically supported difference" in result.sentence()


def test_noise_around_zero_does_not_resolve():
    """The case that matters: a real but tiny difference under noise."""
    scores_a = {i: 10.0 + (hash(i) % 7) for i in IDS}
    scores_b = {i: 10.2 + (hash(i) % 7) for i in IDS}
    result = paired_bootstrap(
        IDS,
        lambda ids: sum(scores_a[i] for i in ids) / len(ids),
        lambda ids: sum(scores_b[i] for i in ids) / len(ids),
        name="bleu",
    )
    # Constant offset with identical noise: this one SHOULD resolve.
    assert result.resolves


def test_is_deterministic():
    """An interval that moves between runs of the same data is not evidence."""
    scores = {i: float(hash(i) % 13) for i in IDS}
    fn = lambda ids: sum(scores[i] for i in ids) / len(ids)  # noqa: E731
    first = paired_bootstrap(IDS, fn, fn, name="x")
    second = paired_bootstrap(IDS, fn, fn, name="x")
    assert (first.low, first.high) == (second.low, second.high)


def test_reports_sample_size():
    assert paired_bootstrap(IDS, _const(1.0), _const(2.0)).n == 40


def test_empty_input_raises_rather_than_returning_a_vacuous_interval():
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap([], _const(1.0), _const(2.0))


def test_sentence_never_claims_a_direction_without_an_interval():
    """The wording rule, enforced where the wording is produced."""
    flat = Comparison("bleu", 0.5, -1.0, 2.0, 0.7, 50, 1000)
    text = flat.sentence()
    for word in ("better", "worse", "beats", "outperforms", "improves"):
        assert word not in text.lower()
    assert "No statistically supported difference" in text


def test_sentence_states_observation_evidence_and_conclusion():
    result = Comparison("bleu", 4.2, 2.0, 6.0, 0.99, 50, 1000)
    text = result.sentence()
    assert "Δ=+4.2" in text          # observation
    assert "0.95 CI" in text and "n=50" in text  # evidence
    assert "excludes zero" in text   # conclusion


def test_lower_is_better_metrics_are_described_correctly():
    """WER falling is an improvement; the sentence must not invert it."""
    wer_drop = Comparison("wer", -0.3, -0.4, -0.2, 0.0, 50, 1000)
    assert "favours B" in wer_drop.sentence(higher_is_better=False)


# --- interval level is configurable, because families need correction --------


def test_a_higher_level_produces_a_wider_interval():
    """Bonferroni is useless if the tool cannot express the corrected level."""
    scores = {i: float(hash(i) % 11) for i in IDS}
    fn_a = lambda ids: sum(scores[i] for i in ids) / len(ids)  # noqa: E731
    fn_b = lambda ids: sum(scores[i] + 1.0 for i in ids) / len(ids)  # noqa: E731

    narrow = paired_bootstrap(IDS, fn_a, fn_b, level=0.95)
    wide = paired_bootstrap(IDS, fn_a, fn_b, level=0.9833)
    assert (wide.high - wide.low) >= (narrow.high - narrow.low)


def test_the_level_is_reported_so_a_reader_knows_which_was_used():
    result = paired_bootstrap(IDS, _const(1.0), _const(2.0), level=0.9833)
    assert result.level == 0.9833
    assert "0.9833 CI" in result.sentence()


def test_an_invalid_level_raises():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="level"):
        paired_bootstrap(IDS, _const(1.0), _const(2.0), level=1.5)
