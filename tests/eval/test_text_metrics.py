"""BLEU and chrF, with the per-language tokenizer rule from D14."""

import pytest

from onevoice.eval.metrics.text import bleu, chrf, tokenizer_for


def test_identical_text_scores_100_bleu():
    hyps = ["Hello, how are you?"]
    refs = ["Hello, how are you?"]
    assert bleu(hyps, refs, "en").value == pytest.approx(100.0)


def test_completely_different_text_scores_zero_bleu():
    result = bleu(["aaa bbb ccc ddd"], ["www xxx yyy zzz"], "en")
    assert result.value == pytest.approx(0.0)


def test_identical_text_scores_100_chrf():
    result = chrf(["the conveyor belt"], ["the conveyor belt"])
    assert result.value == pytest.approx(100.0)


def test_partial_match_scores_between_zero_and_100():
    result = bleu(["The conveyor is broken."], ["The conveyor belt is broken."], "en")
    assert 0.0 < result.value < 100.0


def test_result_records_the_tokenizer_used():
    assert bleu(["hi"], ["hi"], "en").detail["tokenizer"] == "13a"


@pytest.mark.parametrize(
    ("lang", "expected"),
    [("vi", "13a"), ("en", "13a"), ("zh", "zh"), ("ko", "ko-mecab")],
)
def test_tokenizer_selected_by_target_language(lang, expected):
    assert tokenizer_for(lang) == expected


def test_unknown_language_is_rejected():
    with pytest.raises(ValueError, match="fr"):
        tokenizer_for("fr")


def test_chinese_uses_the_zh_tokenizer_not_whitespace():
    # Unsegmented Chinese under 13a would be one token and score ~0 or ~100
    # with no gradation. The zh tokenizer segments into characters.
    result = bleu(["我们需要安全帽"], ["我们需要安全帽"], "zh")
    assert result.value == pytest.approx(100.0)
    assert result.detail["tokenizer"] == "zh"


def test_empty_hypothesis_scores_zero_rather_than_crashing():
    assert bleu([""], ["Hello, how are you?"], "en").value == pytest.approx(0.0)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length"):
        bleu(["a", "b"], ["a"], "en")


def test_korean_bleu_raises_naming_the_uv_extra_when_mecab_is_absent():
    # D14: never silently fall back to 13a. The message must name this repo's
    # install command, not sacrebleu's pip instruction.
    try:
        import mecab_ko  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("sacrebleu[ko] is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra ko"):
        bleu(["안녕하세요"], ["안녕하세요"], "ko")
