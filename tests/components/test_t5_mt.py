"""envit5 and MADLAD: prefix correctness and direction integrity.

The prefix is a **silent-failure surface**. Omit it or get it wrong and the
model still emits fluent text — just not the translation that was asked for.
These tests exist because a wrong prefix is indistinguishable from a bad model
unless something checks it.
"""

from __future__ import annotations

import pytest


def _mt_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


requires_mt = pytest.mark.skipif(not _mt_available(), reason="mt extra missing")


# --- prefix construction: no model needed ------------------------------------


def test_envit5_prefix_names_the_source_language():
    from onevoice.components.t5_mt import EnViT5MT

    assert EnViT5MT._build_prefix(None, "vi", "en") == "vi: "
    assert EnViT5MT._build_prefix(None, "en", "vi") == "en: "


def test_envit5_refuses_a_direction_it_cannot_serve():
    """Fluent Chinese from an en<->vi model would be worse than an error."""
    from onevoice.components.t5_mt import EnViT5MT

    with pytest.raises(ValueError, match="envit5 supports only"):
        EnViT5MT._build_prefix(None, "vi", "zh")
    with pytest.raises(ValueError, match="source"):
        EnViT5MT._build_prefix(None, "zh", "en")


def test_madlad_prefix_is_the_TARGET_token_not_the_source():
    """MADLAD selects direction by target token; using the source would
    translate the wrong way while looking perfectly fluent."""
    from onevoice.components.t5_mt import MadladMT

    assert MadladMT._build_prefix(None, "vi", "en") == "<2en> "
    assert MadladMT._build_prefix(None, "en", "vi") == "<2vi> "
    assert MadladMT._build_prefix(None, "vi", "zh") == "<2zh> "
    assert MadladMT._build_prefix(None, "vi", "ko") == "<2ko> "


def test_madlad_refuses_an_unmapped_target():
    from onevoice.components.t5_mt import MadladMT

    with pytest.raises(ValueError, match="no MADLAD target token"):
        MadladMT._build_prefix(None, "vi", "th")


def test_prefix_is_recorded_in_invocation_provenance():
    """A run whose prefix was wrong must be diagnosable from its record."""
    import inspect

    from onevoice.components import t5_mt

    src = inspect.getsource(t5_mt._T5MT._provenance)
    assert '"prefix": self._prefix' in src


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _FakeTokenizer:
    def __call__(self, *args, **kwargs):
        return {}

    def batch_decode(self, *args, **kwargs):
        return ["vi: Xin chào"]


class _FakeModel:
    def generate(self, **kwargs):
        return []


def test_flush_strips_the_target_tag_before_it_reaches_tts():
    """envit5 echoes `vi: `; that tag must not be spoken or scored."""
    from onevoice.components.t5_mt import EnViT5MT

    mt = object.__new__(EnViT5MT)
    mt._flushed = False
    mt._segments = ["hello"]
    mt._prefix = "en: "
    mt._tokenizer = _FakeTokenizer()
    mt._model = _FakeModel()
    mt._torch = _FakeTorch()

    events = mt.flush()

    assert [event.text for event in events] == ["Xin chào"]


def test_push_does_not_write_to_stdout(capsys):
    """Diagnostics must go through logging, never unconditional print()."""
    from onevoice.components.t5_mt import EnViT5MT

    mt = object.__new__(EnViT5MT)
    mt._flushed = False
    mt._segments = []

    mt.push("Xin chào", is_final=True)

    assert capsys.readouterr().out == ""


def test_flush_with_no_segments_does_not_write_to_stdout(capsys):
    from onevoice.components.t5_mt import EnViT5MT

    mt = object.__new__(EnViT5MT)
    mt._flushed = False
    mt._segments = []

    assert list(mt.flush()) == []
    assert capsys.readouterr().out == ""


# --- direction integrity: needs the real models ------------------------------


@requires_mt
@pytest.mark.slow
def test_envit5_translates_vi_to_en_and_not_the_reverse():
    from onevoice.components.t5_mt import EnViT5MT

    mt = EnViT5MT(src_lang="vi", tgt_lang="en")
    mt.push("Xin chào, bạn khỏe không?", is_final=True)
    out = list(mt.flush())[0].text.lower()
    # English output contains ASCII words and no Vietnamese diacritics.
    assert any(w in out for w in ("hello", "hi", "how are you")), out
    assert "xin chào" not in out, out


@requires_mt
@pytest.mark.slow
def test_madlad_target_token_actually_switches_language():
    """The decisive check: same source, two targets, two different outputs."""
    from onevoice.components.t5_mt import MadladMT

    src = "Xin chào, bạn khỏe không?"
    outs = {}
    for tgt in ("en", "zh"):
        mt = MadladMT(model_name="google/madlad400-3b-mt", src_lang="vi", tgt_lang=tgt)
        mt.push(src, is_final=True)
        outs[tgt] = list(mt.flush())[0].text
    assert outs["en"] != outs["zh"], outs
    # Chinese output should contain CJK; English should not.
    assert any("一" <= c <= "鿿" for c in outs["zh"]), outs["zh"]
    assert not any("一" <= c <= "鿿" for c in outs["en"]), outs["en"]


# --- output-format handling (found by smoke test, before scoring) ------------


def test_envit5_output_tag_is_stripped():
    """envit5 echoes its target tag: 'en: Hello, how are you?'.

    Scoring that against a clean reference would penalise the model for its
    own output format -- the same class of error as raw WER penalising Whisper
    for punctuation (D49). Fixed in the wrapper; the corpus is untouched.
    """
    from onevoice.components.t5_mt import EnViT5MT

    assert EnViT5MT._clean(None, "en: Hello, how are you?") == "Hello, how are you?"
    assert EnViT5MT._clean(None, "vi: Chào bạn") == "Chào bạn"


def test_stripping_does_not_eat_legitimate_content():
    """A translation that merely begins with those letters must survive."""
    from onevoice.components.t5_mt import EnViT5MT

    assert EnViT5MT._clean(None, "entire village was flooded") == (
        "entire village was flooded"
    )
    assert EnViT5MT._clean(None, "vietnam is large") == "vietnam is large"


def test_madlad_does_not_strip_by_default():
    """MADLAD is not known to echo a tag; stripping blindly could remove
    content. Default behaviour is to leave output alone."""
    from onevoice.components.t5_mt import MadladMT

    assert MadladMT._clean(None, "<2en> something") == "<2en> something"
