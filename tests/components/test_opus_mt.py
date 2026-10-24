"""Opus-MT wrapper: contract conformance plus structural checks."""

import pytest

from onevoice.contracts import MTContract
from onevoice.registry import resolve


def _mt_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


requires_mt = pytest.mark.skipif(not _mt_available(), reason="mt extra not installed")


# --- structural: these run in CI with no model installed ---------------------


def test_wrapper_module_imports_without_transformers():
    # The module must be importable so the registry can resolve its name even
    # when the extra is absent; the heavy import belongs inside __init__.
    import onevoice.components.opus_mt  # noqa: F401


def test_registered_under_its_config_name():
    import onevoice.components.opus_mt as module

    assert resolve("mt", "opus") is module.OpusMT


def test_constructing_without_the_extra_raises_a_clear_error():
    import onevoice.components.opus_mt as module

    if _mt_available():
        pytest.skip("mt extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra mt"):
        module.OpusMT(model_name="Helsinki-NLP/opus-mt-vi-en")


# --- contract suite: needs the real model ------------------------------------


@requires_mt
@pytest.mark.slow
class TestOpusMTContract(MTContract):
    def make_component(self):
        from onevoice.components.opus_mt import OpusMT

        return OpusMT(
            model_name="Helsinki-NLP/opus-mt-vi-en", src_lang="vi", tgt_lang="en"
        )


# --- behaviour: needs the real model -----------------------------------------


@requires_mt
@pytest.mark.slow
def test_translates_vietnamese_to_english():
    from onevoice.components.opus_mt import OpusMT

    mt = OpusMT(model_name="Helsinki-NLP/opus-mt-vi-en", src_lang="vi", tgt_lang="en")
    events = mt.push("Xin chào, anh khỏe không?", is_final=True)
    events = list(events) + list(mt.flush())
    text = " ".join(e.text for e in events).lower()
    assert text.strip()
    assert "hello" in text or "how are you" in text


@requires_mt
@pytest.mark.slow
def test_push_buffers_and_flush_emits_one_final_event():
    from onevoice.components.opus_mt import OpusMT

    mt = OpusMT(model_name="Helsinki-NLP/opus-mt-vi-en", src_lang="vi", tgt_lang="en")
    assert mt.push("Xin chào", is_final=True) == []
    events = mt.flush()
    assert len(events) == 1
    assert events[0].is_final is True
    assert isinstance(events[0].text, str)


@requires_mt
@pytest.mark.slow
def test_reset_clears_the_buffer_between_utterances():
    from onevoice.components.opus_mt import OpusMT

    mt = OpusMT(model_name="Helsinki-NLP/opus-mt-vi-en", src_lang="vi", tgt_lang="en")
    mt.push("Xin chào, anh khỏe không?", is_final=True)
    first = mt.flush()[0].text
    mt.reset()
    mt.push("Xin chào, anh khỏe không?", is_final=True)
    assert mt.flush()[0].text == first
