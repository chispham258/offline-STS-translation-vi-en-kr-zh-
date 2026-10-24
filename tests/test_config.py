"""YAML experiment configuration."""

import pytest
from pydantic import ValidationError

from onevoice.config import (
    ComponentConfig,
    ExperimentConfig,
    LanguagePair,
    build,
    load_config,
)
from onevoice.registry import clear_registry, register, restore, snapshot

VALID = """
experiment:
  name: dummy_vi_en
languages:
  src: vi
  tgt: en
vad:
  model: passthrough
asr:
  model: echo
mt:
  model: identity
tts:
  model: silence
"""


@pytest.fixture(autouse=True)
def _isolate_registry():
    # Restoring the pre-test snapshot (rather than clearing) on teardown keeps
    # this isolation from leaking outward and emptying the registry for every
    # test file that runs afterwards in the same process.
    state = snapshot()
    clear_registry()
    yield
    restore(state)


def _write(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, VALID))
    assert cfg.experiment.name == "dummy_vi_en"
    assert cfg.languages.src == "vi"
    assert cfg.languages.tgt == "en"
    assert cfg.asr.model == "echo"


def test_the_shipped_example_config_is_valid():
    cfg = load_config("configs/dummy_vi_en.yaml")
    assert cfg.experiment.name


def test_component_params_default_to_empty(tmp_path):
    cfg = load_config(_write(tmp_path, VALID))
    assert cfg.asr.params == {}


def test_component_params_are_carried_through(tmp_path):
    text = VALID.replace(
        "asr:\n  model: echo", "asr:\n  model: echo\n  params:\n    beam_size: 4"
    )
    cfg = load_config(_write(tmp_path, text))
    assert cfg.asr.params == {"beam_size": 4}


def test_unsupported_language_is_rejected(tmp_path):
    text = VALID.replace("tgt: en", "tgt: fr")
    with pytest.raises(ValidationError) as excinfo:
        load_config(_write(tmp_path, text))
    assert "tgt" in str(excinfo.value)


def test_missing_required_section_names_the_key(tmp_path):
    text = VALID.replace("asr:\n  model: echo\n", "")
    with pytest.raises(ValidationError) as excinfo:
        load_config(_write(tmp_path, text))
    assert "asr" in str(excinfo.value)


def test_unknown_top_level_key_is_rejected(tmp_path):
    # A typo'd section would otherwise be silently ignored, and the experiment
    # would run with defaults nobody chose.
    text = VALID + "\nquantisation:\n  asr: int8\n"
    with pytest.raises(ValidationError) as excinfo:
        load_config(_write(tmp_path, text))
    assert "quantisation" in str(excinfo.value)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_empty_file_is_rejected(tmp_path):
    path = _write(tmp_path, "")
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    assert str(path) in str(excinfo.value)
    assert "mapping" in str(excinfo.value)


def test_bare_list_is_rejected(tmp_path):
    path = _write(tmp_path, "- a\n- b\n")
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    assert str(path) in str(excinfo.value)
    assert "mapping" in str(excinfo.value)


def test_build_instantiates_from_the_registry():
    @register("asr", "echo")
    class EchoASR:
        def __init__(self, **params):
            self.params = params

    component = build("asr", ComponentConfig(model="echo", params={"beam_size": 4}))
    assert isinstance(component, EchoASR)
    assert component.params == {"beam_size": 4}


def test_build_reports_an_unregistered_model():
    with pytest.raises(KeyError, match="whisper"):
        build("asr", ComponentConfig(model="whisper"))


def test_build_passes_languages_through():
    @register("mt", "opus")
    class OpusMT:
        def __init__(self, **params):
            self.params = params

    languages = LanguagePair(src="vi", tgt="en")
    component = build("mt", ComponentConfig(model="opus"), languages)
    assert component.params == {"src_lang": "vi", "tgt_lang": "en"}


def test_build_without_languages_omits_src_and_tgt():
    @register("mt", "opus")
    class OpusMT:
        def __init__(self, **params):
            self.params = params

    component = build("mt", ComponentConfig(model="opus"))
    assert component.params == {}


def test_build_passes_languages_to_named_parameters():
    @register("tts", "melo")
    class MeloTTS:
        def __init__(self, src_lang, tgt_lang):
            self.src_lang = src_lang
            self.tgt_lang = tgt_lang

    languages = LanguagePair(src="vi", tgt="en")
    component = build("tts", ComponentConfig(model="melo"), languages)
    assert component.src_lang == "vi"
    assert component.tgt_lang == "en"


def test_build_does_not_pass_languages_to_a_component_that_does_not_want_them():
    @register("vad", "energy")
    class EnergyVAD:
        def __init__(self):
            pass

    languages = LanguagePair(src="vi", tgt="en")
    # Must not raise TypeError for an unexpected keyword argument.
    component = build("vad", ComponentConfig(model="energy"), languages)
    assert isinstance(component, EnergyVAD)


def test_build_rejects_conflicting_language_keys():
    @register("mt", "opus")
    class OpusMT:
        def __init__(self, **params):
            self.params = params

    languages = LanguagePair(src="vi", tgt="en")
    cfg = ComponentConfig(model="opus", params={"src_lang": "en"})
    with pytest.raises(ValueError, match="src_lang"):
        build("mt", cfg, languages)


def test_build_passes_src_lang_to_a_component_that_only_wants_src_lang():
    # FIX 7: a Whisper wrapper takes only src_lang. The old all-or-nothing
    # check forwarded neither language, silently.
    @register("asr", "whisper")
    class WhisperASR:
        def __init__(self, src_lang):
            self.src_lang = src_lang

    languages = LanguagePair(src="vi", tgt="en")
    component = build("asr", ComponentConfig(model="whisper"), languages)
    assert component.src_lang == "vi"


def test_build_passes_tgt_lang_to_a_component_that_only_wants_tgt_lang():
    @register("tts", "melo_tgt_only")
    class TgtOnlyTTS:
        def __init__(self, tgt_lang):
            self.tgt_lang = tgt_lang

    languages = LanguagePair(src="vi", tgt="en")
    component = build("tts", ComponentConfig(model="melo_tgt_only"), languages)
    assert component.tgt_lang == "en"


def test_parsing_does_not_touch_the_registry(tmp_path):
    # Nothing is registered here, yet the config must still parse. Validation
    # in CI must not require importing every model wrapper.
    cfg = load_config(_write(tmp_path, VALID))
    assert isinstance(cfg, ExperimentConfig)
