"""m2m100 and NLLB wrappers: contract conformance plus structural checks.

The language-code trap (task brief): m2m100 uses ISO-639-1 codes (vi/en/zh/ko)
that happen to equal this framework's `Language` values directly, so
`M2M100MT` needs no mapping. NLLB uses FLORES-200 codes (vie_Latn, eng_Latn,
zho_Hans, kor_Hang) instead, so `NLLBMT` must translate between them - and an
unmapped language must raise rather than silently fall back to a default,
since a wrong forced_bos_token_id produces fluent output in the wrong
language, which scores near zero and looks like a bad model rather than a
bad config. The tests pinning that mapping and its failure mode need no
model and must not be slow-marked - they are what actually catches the trap.
"""

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
    # The module must be importable so the registry can resolve both names
    # even when the extra is absent; heavy imports belong inside __init__.
    import onevoice.components.m2m_mt  # noqa: F401


def test_m2m100_registered_under_its_config_name():
    import onevoice.components.m2m_mt as module

    assert resolve("mt", "m2m100") is module.M2M100MT


def test_nllb_registered_under_its_config_name():
    import onevoice.components.m2m_mt as module

    assert resolve("mt", "nllb") is module.NLLBMT


def test_constructing_m2m100_without_the_extra_raises_a_clear_error():
    import onevoice.components.m2m_mt as module

    if _mt_available():
        pytest.skip("mt extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra mt"):
        module.M2M100MT(model_name="facebook/m2m100_418M")


def test_constructing_nllb_without_the_extra_raises_a_clear_error():
    import onevoice.components.m2m_mt as module

    if _mt_available():
        pytest.skip("mt extra is installed")
    with pytest.raises(RuntimeError, match=r"uv sync --extra mt"):
        module.NLLBMT(model_name="facebook/nllb-200-distilled-600M")


# --- the language-code trap: no model needed, must not be slow ---------------


def test_nllb_flores_mapping_is_pinned():
    # Pins the four mappings the brief calls out explicitly. If any of these
    # drifts, NLLB silently targets the wrong language.
    import onevoice.components.m2m_mt as module

    assert module.nllb_flores_code("vi") == "vie_Latn"
    assert module.nllb_flores_code("en") == "eng_Latn"
    assert module.nllb_flores_code("zh") == "zho_Hans"
    assert module.nllb_flores_code("ko") == "kor_Hang"


def test_nllb_unmapped_language_raises_naming_the_language():
    import onevoice.components.m2m_mt as module

    with pytest.raises(ValueError, match="fr"):
        module.nllb_flores_code("fr")


@requires_mt
def test_constructing_nllb_with_an_unmapped_target_language_raises():
    # The mapping is validated in __init__ before any tokenizer/model is
    # loaded, so this fails fast even with the extra installed - no download,
    # not slow.
    from onevoice.components.m2m_mt import NLLBMT

    with pytest.raises(ValueError, match="fr"):
        NLLBMT(
            model_name="facebook/nllb-200-distilled-600M",
            src_lang="vi",
            tgt_lang="fr",
        )


# --- contract suite: needs the real models -----------------------------------


@requires_mt
@pytest.mark.slow
class TestM2M100MTContract(MTContract):
    def make_component(self):
        from onevoice.components.m2m_mt import M2M100MT

        return M2M100MT(model_name="facebook/m2m100_418M", src_lang="vi", tgt_lang="en")


@requires_mt
@pytest.mark.slow
class TestNLLBMTContract(MTContract):
    def make_component(self):
        from onevoice.components.m2m_mt import NLLBMT

        return NLLBMT(
            model_name="facebook/nllb-200-distilled-600M",
            src_lang="vi",
            tgt_lang="en",
        )


# --- behaviour: needs the real models -----------------------------------------


@requires_mt
@pytest.mark.slow
def test_m2m100_translates_vietnamese_to_english():
    from onevoice.components.m2m_mt import M2M100MT

    mt = M2M100MT(model_name="facebook/m2m100_418M", src_lang="vi", tgt_lang="en")
    events = list(mt.push("Xin chào, anh khỏe không?", is_final=True))
    events += list(mt.flush())
    text = " ".join(e.text for e in events).lower()
    assert text.strip()
    assert "hello" in text or "how are you" in text


@requires_mt
@pytest.mark.slow
def test_nllb_translates_vietnamese_to_english():
    from onevoice.components.m2m_mt import NLLBMT

    mt = NLLBMT(
        model_name="facebook/nllb-200-distilled-600M", src_lang="vi", tgt_lang="en"
    )
    events = list(mt.push("Xin chào, anh khỏe không?", is_final=True))
    events += list(mt.flush())
    text = " ".join(e.text for e in events).lower()
    assert text.strip()
    assert "hello" in text or "how are you" in text


@requires_mt
@pytest.mark.slow
def test_m2m100_push_buffers_and_flush_emits_one_final_event():
    from onevoice.components.m2m_mt import M2M100MT

    mt = M2M100MT(model_name="facebook/m2m100_418M", src_lang="vi", tgt_lang="en")
    assert mt.push("Xin chào", is_final=True) == []
    events = mt.flush()
    assert len(events) == 1
    assert events[0].is_final is True
    assert isinstance(events[0].text, str)


@requires_mt
@pytest.mark.slow
def test_nllb_push_buffers_and_flush_emits_one_final_event():
    from onevoice.components.m2m_mt import NLLBMT

    mt = NLLBMT(
        model_name="facebook/nllb-200-distilled-600M", src_lang="vi", tgt_lang="en"
    )
    assert mt.push("Xin chào", is_final=True) == []
    events = mt.flush()
    assert len(events) == 1
    assert events[0].is_final is True
    assert isinstance(events[0].text, str)


@requires_mt
@pytest.mark.slow
def test_m2m100_reset_clears_the_buffer_between_utterances():
    from onevoice.components.m2m_mt import M2M100MT

    mt = M2M100MT(model_name="facebook/m2m100_418M", src_lang="vi", tgt_lang="en")
    mt.push("Xin chào, anh khỏe không?", is_final=True)
    first = mt.flush()[0].text
    mt.reset()
    mt.push("Xin chào, anh khỏe không?", is_final=True)
    assert mt.flush()[0].text == first


@requires_mt
@pytest.mark.slow
def test_nllb_reset_clears_the_buffer_between_utterances():
    from onevoice.components.m2m_mt import NLLBMT

    mt = NLLBMT(
        model_name="facebook/nllb-200-distilled-600M", src_lang="vi", tgt_lang="en"
    )
    mt.push("Xin chào, anh khỏe không?", is_final=True)
    first = mt.flush()[0].text
    mt.reset()
    mt.push("Xin chào, anh khỏe không?", is_final=True)
    assert mt.flush()[0].text == first
