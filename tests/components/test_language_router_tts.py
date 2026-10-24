"""Language router: one TTS config, one backend per target language."""

from __future__ import annotations

import pytest

from onevoice.config import ComponentConfig, LanguagePair, build
from onevoice.interfaces import BaseTTS
from onevoice.registry import resolve
from onevoice.types import MtEvent, TtsEvent


class RecordingTTS(BaseTTS):
    """Minimal backend that records what the router forwarded."""

    def __init__(self, tgt_lang: str, tag: str = "fake") -> None:
        self.tgt_lang = tgt_lang
        self.tag = tag
        self.pushed: list[tuple[str, bool]] = []
        self.flushes = 0
        self.resets = 0

    def push(self, text: str, is_final: bool = False) -> list[TtsEvent]:
        self._check_not_flushed()
        self.pushed.append((text, is_final))
        return []

    def flush(self) -> list[TtsEvent]:
        self._mark_flushed()
        self.flushes += 1
        return []

    def reset(self) -> None:
        super().reset()
        self.resets += 1

    def provenance(self) -> dict[str, object]:
        return {"model": "fake", "tag": self.tag}


def _backends(*languages: str) -> dict[str, dict[str, object]]:
    return {
        lang: {
            "module": __name__,
            "model": "router_test_fake",
            "params": {"tag": lang},
        }
        for lang in languages
    }


@pytest.fixture(autouse=True)
def _fake_backend_resolution(monkeypatch):
    """Keep the fake out of the global registry.

    Registering a test backend would enrol it in the production provenance
    conformance test, which enumerates the registry and constructs every
    entry. Patch the router's resolver instead.
    """
    import onevoice.components.language_router_tts as router
    from onevoice.registry import resolve as real_resolve

    def resolve(kind: str, name: str):
        if name == "router_test_fake":
            return RecordingTTS
        return real_resolve(kind, name)

    monkeypatch.setattr(router, "resolve", resolve)


def test_registered_under_its_config_name():
    import onevoice.components.language_router_tts as module

    assert resolve("tts", "language_router") is module.LanguageRouterTTS


def test_selects_the_exact_language():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    router = LanguageRouterTTS(_backends("vi", "en"), tgt_lang="vi")

    assert router.provenance()["backend_key"] == "vi"
    assert router.provenance()["tag"] == "vi"
    assert router.tgt_lang == "vi"


def test_selects_the_language_family_for_a_region_tag():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    router = LanguageRouterTTS(_backends("en", "vi"), tgt_lang="en_US")

    assert router.provenance()["backend_key"] == "en"
    assert router.provenance()["language"] == "en_US"


def test_falls_back_when_the_language_is_absent():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    router = LanguageRouterTTS(
        _backends("vi", "en"), tgt_lang="ko", fallback="en"
    )

    assert router.provenance()["backend_key"] == "en"


def test_unknown_language_without_a_fallback_raises():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    with pytest.raises(ValueError, match=r"no backend for 'ko'"):
        LanguageRouterTTS(_backends("vi", "en"), tgt_lang="ko")


def test_empty_backends_raises():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    with pytest.raises(ValueError, match="at least one backend"):
        LanguageRouterTTS({}, tgt_lang="vi")


def test_only_the_selected_backend_is_imported():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    backends = _backends("vi")
    backends["en"] = {
        "module": "module_that_does_not_exist",
        "model": "router_test_fake",
        "params": {"tag": "en"},
    }

    router = LanguageRouterTTS(backends, tgt_lang="vi")

    assert router.provenance()["backend_key"] == "vi"


def test_nested_tgt_lang_is_rejected():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    backends = _backends("vi")
    backends["vi"]["params"] = {"tag": "vi", "tgt_lang": "en"}

    with pytest.raises(ValueError, match="must not set tgt_lang"):
        LanguageRouterTTS(backends, tgt_lang="vi")


def test_unwraps_mt_event_from_the_async_driver():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    router = LanguageRouterTTS(_backends("vi"), tgt_lang="vi")
    event = MtEvent(text="Xin chào", t_wall=1.0, is_final=True)

    router.push(event, is_final=True)

    assert router._backend.pushed == [("Xin chào", True)]


def test_rejects_payloads_without_text():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    router = LanguageRouterTTS(_backends("vi"), tgt_lang="vi")

    with pytest.raises(TypeError, match="expected str"):
        router.push(object())


def test_flush_and_reset_delegate():
    from onevoice.components.language_router_tts import LanguageRouterTTS

    router = LanguageRouterTTS(_backends("vi"), tgt_lang="vi")
    backend = router._backend

    router.push("một")
    router.flush()

    assert backend.flushes == 1
    assert router.is_flushed
    with pytest.raises(RuntimeError):
        router.push("hai")

    router.reset()

    assert backend.resets == 1
    assert not router.is_flushed
    assert router.push("ba") == []


def test_hybrid_config_lists_all_four_targets():
    from onevoice.config import load_config

    cfg = load_config("configs/full_en_vi_vieneu_async_hybrid.yaml")

    assert set(cfg.tts.params["backends"]) == {"vi", "en", "zh", "ko"}


def test_cpu_config_uses_piper_for_vietnamese():
    from onevoice.config import load_config

    cfg = load_config("configs/full_en_vi_cpu_hybrid_async.yaml")

    assert cfg.tts.params["backends"]["vi"]["model"] == "piper"
    voice = cfg.tts.params["backends"]["vi"]["params"]["voice"]
    assert voice == "vi_VN-vais1000-medium"


def test_build_passes_the_experiment_target_language():
    cfg = ComponentConfig(
        model="language_router",
        params={"backends": _backends("vi", "en")},
    )

    component = build("tts", cfg, LanguagePair(src="en", tgt="vi"))

    assert component.provenance()["tag"] == "vi"
    assert component.tgt_lang == "vi"
