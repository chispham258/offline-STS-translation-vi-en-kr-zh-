"""Silero VAD, behind an optional extra with a lazy import (D7).

The first detector in this framework with genuine lookahead: it confirms an
onset only after observing what follows, so `speech_start` arrives late and its
`t_audio` points backwards. That is the condition Q7 described and D42 fixed —
integrating this before the lookback buffer existed would have produced a
working-looking pipeline with a systematically truncated first word.

Silero decides on fixed windows (512 samples at 16 kHz, 256 at 8 kHz) and
nothing else. `chunk_audio` frames happen to be 512 samples at 16 kHz, but that
is a coincidence of the default frame size, not a contract, so this buffers to
window size rather than assuming.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from onevoice.interfaces import BaseVAD
from onevoice.provenance import params_hash
from onevoice.registry import register
from onevoice.types import AudioChunk, VadEvent

#: Window Silero requires, per sample rate. It supports no others, and
#: resampling to reach one would change what is being measured.
WINDOW_SAMPLES = {16000: 512, 8000: 256}

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_MS = 100
DEFAULT_SPEECH_PAD_MS = 30


@register("vad", "silero")
class SileroVAD(BaseVAD):
    """Streaming wrapper over `silero_vad.VADIterator`.

    `speech_pad_ms` widens each detected segment outward at both ends. It is
    what makes `t_audio` land before the frame that produced the event, and it
    is left on: trimming a speech onset to the exact frame Silero became
    confident costs the leading consonant, which is precisely the loss Q7 was
    about.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_MS,
        speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS,
    ) -> None:
        try:
            import torch
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD requires the vad extra, which is not installed. "
                "Install it with: uv sync --extra vad"
            ) from exc

        self._torch = torch
        self._make_iterator = VADIterator
        self._model = load_silero_vad()
        self._threshold = threshold
        self._min_silence_duration_ms = min_silence_duration_ms
        self._speech_pad_ms = speech_pad_ms

        self._iterator = None
        self._rate: int | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._open = False
        # Where this stream sits on the caller's timeline. VADIterator counts
        # samples from zero regardless of AudioChunk.t_start, so without this
        # anchor every boundary is reported relative to the start of the push
        # sequence rather than to the audio timeline (F1). The driver feeds
        # each utterance from t=0, which masked it; the hardened VAD contract
        # does not.
        self._t0: float | None = None

    def _iterator_for(self, rate: int):
        if rate not in WINDOW_SAMPLES:
            raise ValueError(
                f"Silero VAD supports {sorted(WINDOW_SAMPLES)} Hz, got {rate}. "
                "Resampling to reach a supported rate would change the audio "
                "the detector is scored on, so it is not done implicitly."
            )
        if self._iterator is None or self._rate != rate:
            self._rate = rate
            self._iterator = self._make_iterator(
                self._model,
                threshold=self._threshold,
                sampling_rate=rate,
                min_silence_duration_ms=self._min_silence_duration_ms,
                speech_pad_ms=self._speech_pad_ms,
            )
        return self._iterator

    def push(self, chunk: AudioChunk) -> Sequence[VadEvent]:
        self._check_not_flushed()
        iterator = self._iterator_for(chunk.sample_rate)
        window = WINDOW_SAMPLES[chunk.sample_rate]
        if self._t0 is None:
            self._t0 = chunk.t_start

        self._pending = np.concatenate([self._pending, np.asarray(chunk.pcm)])

        events: list[VadEvent] = []
        while len(self._pending) >= window:
            block = self._pending[:window]
            self._pending = self._pending[window:]
            result = iterator(self._torch.from_numpy(np.ascontiguousarray(block)))
            self._consumed += window
            if not result:
                continue
            # VADIterator reports a sample index into the stream it has been
            # fed, counted from zero. _t0 re-anchors that onto the caller's
            # timeline; without it the index IS the timestamp, which is only
            # correct when the stream happens to start at t=0.
            if "start" in result:
                self._open = True
                events.append(
                    VadEvent(
                        kind="speech_start",
                        t_audio=self._t0 + result["start"] / chunk.sample_rate,
                    )
                )
            if "end" in result:
                self._open = False
                events.append(
                    VadEvent(
                        kind="speech_end",
                        t_audio=self._t0 + result["end"] / chunk.sample_rate,
                    )
                )
        return events

    def flush(self) -> Sequence[VadEvent]:
        """Close a segment the audio ended in the middle of.

        The trailing partial window is dropped rather than zero-padded: padding
        invents audio, and any boundary inside it is covered by the speech_end
        emitted here.
        """
        self._mark_flushed()
        if not self._open:
            return []
        self._open = False
        rate = self._rate or 16000
        return [
            VadEvent(
                kind="speech_end",
                t_audio=(self._t0 or 0.0)
                + (self._consumed + len(self._pending)) / rate,
            )
        ]

    def reset(self) -> None:
        super().reset()
        if self._iterator is not None:
            self._iterator.reset_states()
        self._pending = np.zeros(0, dtype=np.float32)
        self._consumed = 0
        self._open = False
        self._t0 = None

    def provenance(self) -> dict[str, object]:
        """D46: which detector, at which weights, configured how.

        Silero ships as a TorchScript file inside the `silero-vad` package
        rather than a Hub checkpoint, so there is no library-reported commit.
        The anchor is the sha256 of the weight file actually loaded, which pins
        the exact bytes; the package version is recorded as the mutable
        `revision` it was requested through.
        """
        import hashlib
        import importlib.metadata as metadata
        from pathlib import Path as _Path

        import silero_vad

        weights = _Path(silero_vad.__file__).parent / "data" / "silero_vad.jit"
        return {
            "component": "silero",
            "implementation": (
                f"{type(self).__module__}.{type(self).__qualname__}"
            ),
            "model_id": "snakers4/silero-vad",
            "revision": metadata.version("silero-vad"),
            "weight_hash": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "params_hash": params_hash(
                {
                    "threshold": self._threshold,
                    "min_silence_duration_ms": self._min_silence_duration_ms,
                    "speech_pad_ms": self._speech_pad_ms,
                }
            ),
            "invocation": {
                "threshold": self._threshold,
                "min_silence_duration_ms": self._min_silence_duration_ms,
                "speech_pad_ms": self._speech_pad_ms,
            },
        }
