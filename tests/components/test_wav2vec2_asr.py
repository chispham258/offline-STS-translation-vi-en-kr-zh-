"""wav2vec2 ASR wrapper: greedy contract plus optional 4-gram LM decoding.

These tests use fake transformers/pyctcdecode objects, so they do not download
the 95M acoustic model or the 1.36 GB KenLM.
"""

from __future__ import annotations

import hashlib
import types

import numpy as np
import pytest

from onevoice.types import AudioChunk


def _fake_transformers(monkeypatch, *, model_vocab_size: int = 3):
    """Install fake AutoProcessor/AutoModelForCTC classes."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class FakeTokenizer:
        def __init__(self, vocab):
            self._vocab = vocab

        def get_vocab(self):
            return dict(self._vocab)

    class FakeProcessor:
        def __init__(self, vocab):
            self.tokenizer = FakeTokenizer(vocab)

        def __call__(self, pcm, sampling_rate, return_tensors, padding):
            return types.SimpleNamespace(input_values=torch.zeros(1, 1600))

        def batch_decode(self, ids):
            return ["greedy text"]

    class FakeModel:
        config = types.SimpleNamespace(
            vocab_size=model_vocab_size,
            _commit_hash="deadbeef",
        )

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_values):
            return types.SimpleNamespace(
                logits=torch.zeros(1, 4, model_vocab_size)
            )

    vocab = {"a": 0, "b": 1, "<pad>": 2, "<s>": 3}
    processor = FakeProcessor(vocab)
    model = FakeModel()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: processor),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCTC,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: model),
    )
    return processor, model


def _chunk(seconds: float = 0.1) -> AudioChunk:
    return AudioChunk(
        pcm=np.zeros(int(seconds * 16000), dtype=np.float32),
        sample_rate=16000,
        t_start=0.0,
    )


def test_registered_under_its_config_name():
    import onevoice.components.wav2vec2_asr as module
    from onevoice.registry import resolve

    assert resolve("asr", "wav2vec2") is module.Wav2Vec2ASR


def test_ctc_labels_are_truncated_to_model_head(monkeypatch):
    # Tokenizer exposes <s> beyond the CTC head. Keeping it would shift labels.
    _fake_transformers(monkeypatch, model_vocab_size=3)
    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    asr = Wav2Vec2ASR()

    assert asr._ctc_labels() == ["a", "b", "<pad>"]


def test_ctc_labels_reject_a_short_tokenizer(monkeypatch):
    _fake_transformers(monkeypatch, model_vocab_size=5)
    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    asr = Wav2Vec2ASR()

    with pytest.raises(ValueError, match="tokenizer vocabulary is smaller"):
        asr._ctc_labels()


def test_greedy_flush_is_unchanged(monkeypatch, capsys):
    _fake_transformers(monkeypatch)
    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    asr = Wav2Vec2ASR()
    asr.push(_chunk())
    events = asr.flush()

    assert len(events) == 1
    assert events[0].text == "greedy text"
    assert events[0].is_final is True
    assert events[0].confidence is None
    assert capsys.readouterr().out == ""


def test_lm_decoder_is_used_with_requested_settings(monkeypatch, tmp_path, capsys):
    pytest.importorskip("pyctcdecode")
    import pyctcdecode

    _fake_transformers(monkeypatch)
    calls = {}

    class FakeDecoder:
        def decode(self, logits, beam_width):
            calls["logits_shape"] = logits.shape
            calls["beam_width"] = beam_width
            return "lm text"

    def fake_build(labels, path, *, unigrams, alpha, beta):
        calls.update(
            labels=labels,
            path=path,
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
        )
        return FakeDecoder()

    monkeypatch.setattr(pyctcdecode, "build_ctcdecoder", fake_build)
    lm = tmp_path / "lm.bin"
    lm.write_bytes(b"fake lm")

    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    asr = Wav2Vec2ASR(
        lm_path=str(lm),
        beam_width=50,
        alpha=0.7,
        beta=2.0,
        unigrams=["a"],
    )
    asr.push(_chunk())
    events = asr.flush()

    assert events[0].text == "lm text"
    assert calls["labels"] == ["a", "b", "<pad>"]
    assert calls["path"] == str(lm)
    assert calls["unigrams"] == ["a"]
    assert calls["alpha"] == 0.7
    assert calls["beta"] == 2.0
    assert calls["beam_width"] == 50
    assert calls["logits_shape"] == (4, 3)
    assert capsys.readouterr().out == ""


def test_missing_lm_file_fails_loudly(monkeypatch, tmp_path):
    _fake_transformers(monkeypatch)
    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    missing = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError, match="LM file not found"):
        Wav2Vec2ASR(lm_path=str(missing))


def test_provenance_records_lm_invocation(monkeypatch, tmp_path):
    pytest.importorskip("pyctcdecode")
    import pyctcdecode

    _fake_transformers(monkeypatch)
    monkeypatch.setattr(
        pyctcdecode,
        "build_ctcdecoder",
        lambda *args, **kwargs: types.SimpleNamespace(decode=lambda *a, **k: "x"),
    )
    lm = tmp_path / "lm.bin"
    lm.write_bytes(b"fake lm")

    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    asr = Wav2Vec2ASR(lm_path=str(lm), beam_width=25, alpha=0.4, beta=1.2)
    invocation = asr.provenance()["invocation"]

    assert invocation["decode"] == "ctc_beam_lm"
    assert invocation["lm_path"] == str(lm)
    assert invocation["lm_size_bytes"] == len(b"fake lm")
    assert invocation["lm_sha256"] == hashlib.sha256(b"fake lm").hexdigest()
    assert invocation["beam_width"] == 25
    assert invocation["alpha"] == 0.4
    assert invocation["beta"] == 1.2


def test_provenance_records_greedy_invocation(monkeypatch):
    _fake_transformers(monkeypatch)
    from onevoice.components.wav2vec2_asr import Wav2Vec2ASR

    invocation = Wav2Vec2ASR().provenance()["invocation"]

    assert invocation["decode"] == "ctc_greedy"
    assert invocation["lm_path"] is None
    assert invocation["lm_size_bytes"] is None
    assert invocation["lm_sha256"] is None
