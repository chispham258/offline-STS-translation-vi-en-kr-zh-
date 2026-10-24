"""WER and CER against hand-computed values."""

import pytest

from onevoice.eval.metrics.error_rate import cer, wer


def test_identical_text_has_zero_wer():
    assert wer(["the cat sat on the mat"], ["the cat sat on the mat"]).value == 0.0


def test_one_substitution_in_six_words():
    # "the" -> "a" is 1 substitution; reference has 6 words.
    result = wer(["the cat sat on a mat"], ["the cat sat on the mat"])
    assert result.value == pytest.approx(1 / 6)


def test_one_deletion_in_six_words():
    result = wer(["the cat sat on mat"], ["the cat sat on the mat"])
    assert result.value == pytest.approx(1 / 6)


def test_one_insertion_in_six_words():
    result = wer(["the big cat sat on the mat"], ["the cat sat on the mat"])
    assert result.value == pytest.approx(1 / 6)


def test_identical_text_has_zero_cer():
    assert cer(["abc"], ["abc"]).value == 0.0


def test_one_character_substitution_in_three():
    assert cer(["abd"], ["abc"]).value == pytest.approx(1 / 3)


def test_wer_can_exceed_one():
    # Three insertions against a one-word reference.
    result = wer(["a b c d"], ["a"])
    assert result.value == pytest.approx(3.0)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length"):
        wer(["a", "b"], ["a"])
