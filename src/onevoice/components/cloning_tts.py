"""Voice-cloning TTS wrappers — viXTTS (xtts) and F5-TTS (f5).

These are the two families that make the TTS matrix satisfy Gate D. Both differ
from Piper and MMS in a way that matters more than architecture: **they are
zero-shot voice cloning models and synthesise in the voice of a reference clip.**

That single fact creates three silent-failure surfaces, and every design choice
below exists to close one of them:

**The reference clip is part of the experiment.** Two runs with different
references are not comparable, and a wrapper that quietly falls back to some
default speaker would make an unrepeatable run look like a clean one. So the
reference is **required**, never defaulted, and its path *and content hash* go
into provenance. A clip that is swapped on disk between runs is then visible.

**Language codes differ between checkpoints.** XTTS speaks `zh-cn`, not `zh`.
Passing the pipeline's own code straight through would either raise deep inside
the library or, worse, be accepted and synthesised in the wrong language. Each
wrapper maps explicitly and **raises on anything it cannot express**, because a
TTS component that emits fluent audio in the wrong language is exactly the
failure a listener panel would score as "bad model".

**A fine-tune's declared languages are not its verified languages.** viXTTS
inherits 18 languages from XTTS-v2 but was fine-tuned on Vietnamese. The wrapper
will happily accept `en`, and whether that output is *good* is a measurement
nobody has taken. `declared_languages()` reports what the checkpoint claims so a
coverage claim can be recorded as `config_inspected` — deliberately not as
`output_tested`, which requires reading real output.

Imports are lazy (D7). Neither library can share an environment with the other —
coqui pins `numpy 1.26` and f5-tts wants `numpy 2.x` — so both are absent from
the default environment and each needs its own isolated one (D64).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from onevoice.interfaces import BaseTTS
from onevoice.provenance import ProvenanceError, params_hash
from onevoice.registry import register
from onevoice.types import TtsEvent

VIXTTS_MODEL = "capleaf/viXTTS"
F5_MODEL = "hynt/F5-TTS-Vietnamese-ViVoice"
#: Architecture named by the checkpoint card. The api defaults to `F5TTS_v1_Base`,
#: which loads this checkpoint without error and produces wrong speech.
F5_ARCH = "F5TTS_Base"

#: XTTS-v2 language codes, read from `capleaf/viXTTS` config.json 2026-08-17.
#: Mapped from the pipeline's codes, which are not identical — `zh` vs `zh-cn`
#: is the whole reason this table exists rather than a pass-through.
XTTS_LANGS = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it", "pt": "pt",
    "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "zh": "zh-cn", "hu": "hu", "ko": "ko", "ja": "ja", "hi": "hi", "vi": "vi",
}

#: `hynt/F5-TTS-Vietnamese-ViVoice` declares `language: vi` only. Its vocab file
#: carries pinyin tokens inherited from the base F5-TTS vocabulary; that is an
#: artefact of the base model, NOT evidence of Chinese capability in this
#: fine-tune, and it is not treated as coverage.
F5_LANGS = ("vi",)


def _reference_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit_from_snapshot(path: str | Path) -> str:
    """The Hub caches a snapshot under `.../snapshots/<commit>/...`.

    That directory name IS the resolved commit, so the anchor comes for free
    without re-hashing gigabytes of weights. The path is made absolute but NOT
    resolved: snapshot files are symlinks into `blobs/`, which names no commit.
    A path outside a snapshot is refused - a stand-in hash would look like an
    anchor while anchoring nothing.
    """
    parts = Path(path).absolute().parts
    if "snapshots" not in parts or parts[-1] == "snapshots":
        raise ProvenanceError(f"{path} is not inside a Hub snapshot; no commit")
    return parts[parts.index("snapshots") + 1]


class _CloningTTS(BaseTTS):
    """Shared contract: a required, content-hashed speaker reference."""

    def __init__(self, model_name: str, tgt_lang: str, reference_audio: str | None):
        if reference_audio is None:
            raise ValueError(
                f"{type(self).__name__} clones a voice and therefore requires "
                "`reference_audio`. It is not defaulted: two runs with different "
                "references are not comparable, and a silent fallback speaker "
                "would make an unrepeatable run look clean."
            )
        ref = Path(reference_audio)
        if not ref.is_file():
            raise FileNotFoundError(f"reference_audio not found: {ref}")

        self.model_name = model_name
        self.tgt_lang = tgt_lang
        self.reference_audio = str(ref)
        self.reference_sha256 = _reference_digest(ref)
        self._model_lang = self._resolve_language(tgt_lang)
        #: Resolved checkpoint commit. Set by each subclass once the Hub
        #: snapshot is materialised; it is the immutable anchor D46 requires.
        self._commit_hash = "unresolved"

    def _resolve_language(self, tgt_lang: str) -> str:
        raise NotImplementedError

    def declared_languages(self) -> frozenset[str]:
        """Languages the checkpoint declares. Evidence class `config_inspected`.

        Not `output_tested`: declaring a language is not producing usable
        speech in it.
        """
        raise NotImplementedError

    def _base_provenance(self, component: str) -> dict[str, object]:
        params = {
            "model_name": self.model_name,
            "tgt_lang": self.tgt_lang,
            "reference_audio": self.reference_audio,
        }
        return {
            "component": component,
            "implementation": f"{type(self).__module__}.{type(self).__name__}",
            "model_id": self.model_name,
            # Requested revision. Neither wrapper pins one, and saying so is the
            # point of D46's requested-vs-effective split: `commit_hash` below
            # carries what was actually resolved.
            "revision": "default",
            "commit_hash": self._commit_hash,
            "params_hash": params_hash(params),
            "params": {
                "model_name": self.model_name,
                "tgt_lang": self.tgt_lang,
                "reference_audio": self.reference_audio,
            },
            "invocation": {
                "tgt_lang": self.tgt_lang,
                # The code actually handed to the model, which may differ from
                # the one the pipeline asked for.
                "model_language": self._model_lang,
                "reference_audio": self.reference_audio,
                # Hash, not just path: a clip swapped on disk between runs is
                # otherwise invisible in the record.
                "reference_sha256": self.reference_sha256,
            },
            "declared_languages": sorted(self.declared_languages()),
            "language_evidence": "config_inspected",
        }

    def flush(self) -> Sequence[TtsEvent]:
        self._mark_flushed()
        return []


@register("tts", "xtts")
class ViXttsTTS(_CloningTTS):
    """`capleaf/viXTTS` — XTTS-v2 fine-tuned on viVoice."""

    def __init__(
        self,
        model_name: str = VIXTTS_MODEL,
        tgt_lang: str = "vi",
        reference_audio: str | None = None,
    ) -> None:
        super().__init__(model_name, tgt_lang, reference_audio)
        try:
            import torch
            from huggingface_hub import snapshot_download
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as exc:  # pragma: no cover - exercised without extra
            # Name the ACTUAL failure. This message used to say only "coqui TTS
            # is not installed", which sent a debugging session after the wrong
            # package when the real cause was transformers 5.x removing
            # `BeamSearchScorer`, which coqui's XTTS imports.
            raise RuntimeError(
                f"viXTTS could not load its dependencies: {exc}. coqui TTS "
                "needs numpy 1.26 and transformers < 5 (its XTTS imports "
                "`BeamSearchScorer`, removed in 5.x), which conflicts with "
                "f5-tts. Install it into its OWN isolated environment (D64), "
                "never the benchmark one."
            ) from exc

        # NOT `TTS.api.TTS(name)`: that parses its argument as a four-part
        # coqui model id (`tts_models/<lang>/<dataset>/<model>`) and a Hub repo
        # id fails with an unpacking error. viXTTS is loaded from its checkpoint
        # directory instead.
        ckpt_dir = snapshot_download(model_name)
        self._commit_hash = _commit_from_snapshot(ckpt_dir)
        vocab = Path(ckpt_dir) / "vocab.json"
        if not vocab.is_file():
            raise RuntimeError(
                f"{model_name} ships no vocab.json. viXTTS EXPANDS the XTTS-v2 "
                "tokenizer for Vietnamese, so substituting the base vocabulary "
                "would mis-tokenise Vietnamese and degrade output silently "
                "rather than raise."
            )
        self._torch = torch
        self._config = XttsConfig()
        self._config.load_json(str(Path(ckpt_dir) / "config.json"))
        self._model = Xtts.init_from_config(self._config)
        self._model.load_checkpoint(
            self._config,
            checkpoint_dir=ckpt_dir,
            vocab_path=str(vocab),
            use_deepspeed=False,
        )
        self._model.eval()
        self.sample_rate = int(self._config.audio.output_sample_rate)

    def _resolve_language(self, tgt_lang: str) -> str:
        try:
            return XTTS_LANGS[tgt_lang]
        except KeyError:
            raise ValueError(
                f"viXTTS cannot express target language {tgt_lang!r}; it "
                f"declares {sorted(XTTS_LANGS)}. Refusing rather than "
                "synthesising fluent audio in the wrong language."
            ) from None

    def declared_languages(self) -> frozenset[str]:
        return frozenset(XTTS_LANGS)

    def provenance(self) -> dict[str, object]:
        record = self._base_provenance("xtts")
        record["sample_rate"] = self.sample_rate
        return record

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        if not text.strip():
            return []
        with self._torch.no_grad():
            result = self._model.synthesize(
                text,
                self._config,
                speaker_wav=self.reference_audio,
                language=self._model_lang,
            )
        pcm = np.asarray(result["wav"], dtype=np.float32).squeeze()
        if pcm.ndim == 0 or len(pcm) == 0:
            return []
        return [
            TtsEvent(
                pcm=np.clip(pcm, -1.0, 1.0).astype(np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=is_final,
            )
        ]


@register("tts", "f5")
class F5TTS(_CloningTTS):
    """`hynt/F5-TTS-Vietnamese-ViVoice` — flow-matching DiT, Vietnamese only."""

    def __init__(
        self,
        model_name: str = F5_MODEL,
        tgt_lang: str = "vi",
        reference_audio: str | None = None,
        reference_text: str | None = None,
    ) -> None:
        if reference_text is None:
            raise ValueError(
                "F5-TTS conditions on the TRANSCRIPT of the reference clip as "
                "well as its audio; `reference_text` is required. A wrong or "
                "empty transcript degrades output without raising."
            )
        super().__init__(model_name, tgt_lang, reference_audio)
        self.reference_text = reference_text
        try:
            from f5_tts.api import F5TTS as F5Api
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(
                f"F5-TTS could not load its dependencies: {exc}. f5-tts wants "
                "numpy 2.x, which conflicts with coqui TTS. Install it into "
                "its OWN isolated environment (D64), never the benchmark one."
            ) from exc

        # `ckpt_file` is a local path, not a Hub id — passing the repo id makes
        # the api silently fall back to its own default base checkpoint, which
        # would benchmark the wrong model entirely. The vocabulary ships as
        # `config.json` despite not being JSON: it is the token list.
        ckpt = hf_hub_download(model_name, "model_last.pt")
        self._commit_hash = _commit_from_snapshot(ckpt)
        vocab = hf_hub_download(model_name, "config.json")
        # `F5TTS_Base`, NOT the api default `F5TTS_v1_Base`. The checkpoint card
        # specifies the v0 architecture, and the mismatch does not raise: the
        # weights load and synthesis emits fluent Vietnamese-sounding speech
        # that says something else entirely. Caught by the smoke test, which is
        # the only thing that could have caught it.
        self._api = F5Api(model=F5_ARCH, ckpt_file=ckpt, vocab_file=vocab)
        self.sample_rate = int(getattr(self._api, "target_sample_rate", 24000))

    def _resolve_language(self, tgt_lang: str) -> str:
        if tgt_lang not in F5_LANGS:
            raise ValueError(
                f"this F5-TTS checkpoint declares only {list(F5_LANGS)}; got "
                f"{tgt_lang!r}. Its vocabulary carries pinyin tokens inherited "
                "from the base F5-TTS model, which is not Chinese capability in "
                "this fine-tune and is not treated as coverage."
            )
        return tgt_lang

    def declared_languages(self) -> frozenset[str]:
        return frozenset(F5_LANGS)

    def provenance(self) -> dict[str, object]:
        record = self._base_provenance("f5")
        record["sample_rate"] = self.sample_rate
        record["invocation"]["reference_text"] = self.reference_text
        return record

    def push(self, text: str, is_final: bool = False) -> Sequence[TtsEvent]:
        self._check_not_flushed()
        if not text.strip():
            return []
        wav, sr, _ = self._api.infer(
            ref_file=self.reference_audio,
            ref_text=self.reference_text,
            gen_text=text,
        )
        pcm = np.asarray(wav, dtype=np.float32)
        if pcm.ndim == 0 or len(pcm) == 0:
            return []
        self.sample_rate = int(sr)
        return [
            TtsEvent(
                pcm=np.clip(pcm, -1.0, 1.0).astype(np.float32),
                sample_rate=self.sample_rate,
                t_wall=0.0,
                is_final=is_final,
            )
        ]
