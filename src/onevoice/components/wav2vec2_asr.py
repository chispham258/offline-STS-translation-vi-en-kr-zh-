"""wav2vec2 CTC ASR, behind the asr extra with a lazy import (D7).

The first non-Whisper ASR in the matrix, and the candidate that carries Gate D
for the slot (D56). Architecturally distinct in ways that matter to this
project's metrics:

- **CTC, not autoregressive.** No decode-length cap and no per-token latency
  term -- both of which sit on the competition's End-of-Speech -> Start-of-Audio
  path.
- **No punctuation, no casing.** A CTC head over a character/word-piece
  vocabulary emits neither. Raw `wer` therefore penalises it for formatting it
  structurally cannot produce; `wer_normalized` (D49), the pre-registered
  primary metric, removes casing and punctuation and is the fair comparison.
  This is recorded in the track's pre-registration, before any result.

Batch-shaped per D6, exactly like the Whisper wrapper: push() buffers, flush()
runs the model once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from onevoice.interfaces import BaseASR
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import AsrEvent, AudioChunk

DEFAULT_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"


@register("asr", "wav2vec2")
class Wav2Vec2ASR(BaseASR):
    """CTC ASR. Greedy by default; optional beam search with an n-gram LM."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        src_lang: str = "vi",
        device: str = "cpu",
        lm_path: str | None = None,
        beam_width: int = 100,
        alpha: float = 0.5,
        beta: float = 1.5,
        unigrams: list[str] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "wav2vec2 requires the asr extra, which is not installed. "
                "Install it with: uv sync --extra asr"
            ) from exc

        self._torch = torch
        self._model_name = model_name
        self._src_lang = src_lang
        self._device = device
        self._lm_path = lm_path
        self._beam_width = beam_width
        self._alpha = alpha
        self._beta = beta
        self._lm_size_bytes: int | None = None
        self._lm_sha256: str | None = None
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForCTC.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()
        self._decoder = self._build_decoder(lm_path, unigrams)
        self._chunks: list[AudioChunk] = []

    def _ctc_labels(self) -> list[str]:
        """Tokenizer labels ordered by logit index.

        The tokenizer may expose added tokens (``<s>``/``</s>``) that the CTC
        head does not. Truncate to ``config.vocab_size`` so the label list is
        aligned with the logits; a mismatch would silently shift every label.
        """
        vocab = self._processor.tokenizer.get_vocab()
        labels = [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        labels = labels[: self._model.config.vocab_size]
        if len(labels) != self._model.config.vocab_size:
            raise ValueError(
                "tokenizer vocabulary is smaller than the CTC head: "
                f"{len(labels)} labels vs {self._model.config.vocab_size} logits"
            )
        return labels

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _build_decoder(self, lm_path: str | None, unigrams: list[str] | None):
        if lm_path is None:
            return None
        path = Path(lm_path)
        if not path.is_file():
            raise FileNotFoundError(f"wav2vec2 LM file not found: {path}")
        self._lm_size_bytes = path.stat().st_size
        self._lm_sha256 = self._file_sha256(path)
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError as exc:
            raise RuntimeError(
                "wav2vec2 LM decoding requires pyctcdecode and kenlm, which "
                "are not installed. Install them with: uv sync --extra asr"
            ) from exc
        return build_ctcdecoder(
            self._ctc_labels(),
            str(path),
            unigrams=unigrams,
            alpha=self._alpha,
            beta=self._beta,
        )

    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        self._check_not_flushed()
        self._chunks.append(chunk)
        return []

    def flush(self) -> Sequence[AsrEvent]:
        self._mark_flushed()
        if not self._chunks:
            return []

        pcm = np.concatenate([c.pcm for c in self._chunks])
        rate = self._chunks[0].sample_rate
        last = self._chunks[-1]

        inputs = self._processor(
            pcm, sampling_rate=rate, return_tensors="pt", padding=True
        )
        with self._torch.no_grad():
            logits = self._model(inputs.input_values.to(self._device)).logits
        if self._decoder is None:
            ids = self._torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(ids)[0]
        else:
            text = self._decoder.decode(
                logits[0].float().cpu().numpy(),
                beam_width=self._beam_width,
            )

        return [
            AsrEvent(
                text=text,
                t_audio_end=last.t_start + last.duration_s,
                t_wall=0.0,
                is_final=True,
                # Deliberately None, as for Whisper: CTC logits are not a
                # calibrated utterance confidence, and inventing one would be
                # worse than omitting it.
                confidence=None,
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._chunks = []

    def provenance(self) -> dict[str, object]:
        """D46/D57."""
        return describe_hub_model(
            component="wav2vec2",
            implementation=type(self),
            model_id=self._model_name,
            model=self._model,
            tokenizer_revision="same-repo-as-model",
            params={
                "model_name": self._model_name,
                "src_lang": self._src_lang,
                "device": self._device,
            },
            invocation={
                "src_lang": self._src_lang,
                "task": "transcribe",
                "decode": "ctc_beam_lm" if self._decoder is not None else "ctc_greedy",
                "lm_path": self._lm_path,
                "lm_size_bytes": self._lm_size_bytes,
                "lm_sha256": self._lm_sha256,
                "beam_width": self._beam_width,
                "alpha": self._alpha,
                "beta": self._beta,
            },
        )
