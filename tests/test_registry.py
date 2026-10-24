"""Component registry: config strings resolve to classes."""

import pytest

from onevoice.registry import (
    clear_registry,
    register,
    registered_names,
    resolve,
    restore,
    snapshot,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    # Module-level mutable state: without this, tests leak into each other and
    # pass or fail depending on execution order. Restoring the pre-test
    # snapshot (rather than clearing) on teardown keeps this isolation from
    # leaking outward and emptying the registry for every test file that
    # runs afterwards in the same process.
    state = snapshot()
    clear_registry()
    yield
    restore(state)


def test_registered_class_resolves_by_name():
    @register("asr", "echo")
    class EchoASR:
        pass

    assert resolve("asr", "echo") is EchoASR


def test_decorator_returns_the_class_unchanged():
    @register("mt", "identity")
    class IdentityMT:
        pass

    assert IdentityMT.__name__ == "IdentityMT"
    assert resolve("mt", "identity") is IdentityMT


def test_same_name_in_different_kinds_does_not_collide():
    @register("asr", "dummy")
    class DummyASR:
        pass

    @register("tts", "dummy")
    class DummyTTS:
        pass

    assert resolve("asr", "dummy") is DummyASR
    assert resolve("tts", "dummy") is DummyTTS


def test_duplicate_registration_raises():
    # D17: silent overwrite would make the benchmarked model depend on import
    # order, which is unreproducible in exactly the way this framework prevents.
    @register("asr", "echo")
    class First:
        pass

    with pytest.raises(ValueError, match="echo"):

        @register("asr", "echo")
        class Second:
            pass


def test_unknown_name_lists_the_valid_ones():
    @register("asr", "echo")
    class EchoASR:
        pass

    @register("asr", "whisper")
    class WhisperASR:
        pass

    with pytest.raises(KeyError) as excinfo:
        resolve("asr", "conformer")
    message = str(excinfo.value)
    assert "conformer" in message
    assert "echo" in message
    assert "whisper" in message


def test_unknown_kind_is_rejected():
    with pytest.raises(KeyError, match="punctuation"):
        resolve("punctuation", "anything")


def test_registered_names_is_sorted():
    @register("asr", "whisper")
    class W:
        pass

    @register("asr", "echo")
    class E:
        pass

    assert registered_names("asr") == ["echo", "whisper"]


def test_registered_names_of_an_empty_kind_is_empty():
    assert registered_names("asr") == []


def test_restore_undoes_registrations_made_after_the_snapshot():
    @register("asr", "echo")
    class EchoASR:
        pass

    state = snapshot()

    @register("asr", "whisper")
    class WhisperASR:
        pass

    assert registered_names("asr") == ["echo", "whisper"]

    restore(state)

    assert registered_names("asr") == ["echo"]
    assert resolve("asr", "echo") is EchoASR


def test_restore_gives_back_exactly_the_original_names():
    @register("asr", "echo")
    class EchoASR:
        pass

    @register("tts", "silence")
    class SilenceTTS:
        pass

    state = snapshot()
    clear_registry()

    @register("mt", "identity")
    class IdentityMT:
        pass

    restore(state)

    assert registered_names("asr") == ["echo"]
    assert registered_names("tts") == ["silence"]
    assert registered_names("mt") == []
