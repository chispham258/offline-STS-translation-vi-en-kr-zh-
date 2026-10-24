"""Streaming faster-whisper ASR — chunked transcription with partial results (D6, D7).

Unlike batch-shaped ASR (push buffers, flush once), this:
- Accumulates chunks into small batches (2-4 chunks = 64-128ms)
- Runs transcribe() with chunk_length for streaming
- Yields partial AsrEvent as segments arrive
- Flush emits remaining segments with is_final=True
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from onevoice.interfaces import BaseASR
from onevoice.provenance import describe_hub_model
from onevoice.registry import register
from onevoice.types import AsrEvent, AudioChunk


@register("asr", "faster_whisper_streaming")
class FasterWhisperASRStreaming(BaseASR):
    """Streaming faster-whisper: chunked transcription with partial results.

    Unlike batch-shaped ASR (push buffers, flush once), this:
    - Accumulates chunks into small batches (default 2 = 64ms @ 32ms/chunk)
    - Runs transcribe() with chunk_length for streaming
    - Yields partial AsrEvent as segments arrive
    - Flush emits remaining segments with is_final=True
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        src_lang: str = "vi",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        num_workers: int = 4,
        chunk_batch_size: int = 2,      # chunks per batch (2 = 64ms @ 32ms/chunk)
        stream_chunk_length: int = 5,   # seconds per faster-whisper chunk
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper requires the asr-faster-whisper extra, "
                "which is not installed. Install it with: "
                "uv sync --extra asr-faster-whisper"
            ) from exc

        self._model_name = model_name
        self._src_lang = src_lang
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._num_workers = num_workers
        self._chunk_batch_size = chunk_batch_size
        self._stream_chunk_length = stream_chunk_length

        # faster-whisper handles device mapping internally
        device_arg = device if device != "cpu" else "cpu"
        self._model = WhisperModel(
            model_name,
            device=device_arg,
            compute_type=compute_type,
            num_workers=num_workers,
        )

        # Diagnostic: expose faster-whisper's temperature-fallback decisions.
        import logging

        fw_logger = logging.getLogger("faster_whisper")
        fw_logger.setLevel(logging.DEBUG)
        if not any(
            isinstance(h, logging.StreamHandler) for h in fw_logger.handlers
        ):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[FW-DEBUG] %(message)s"))
            fw_logger.addHandler(handler)

        # Streaming state
        self._buffer: list[AudioChunk] = []           # current batch
        self._stream_pcm = np.array([], dtype=np.float32)  # accumulated PCM
        self._stream_gen = None                       # faster-whisper generator
        self._gen_offset_s = 0.0                      # window start in stream time
        self._gen_window_end_s = 0.0                  # window end in stream time
        self._pending_segments: list = []             # decoded, not yet yielded
        self._last_segment_end = 0.0                  # track segment boundaries
        self._flushed = False

    def provenance(self) -> dict[str, object]:
        """D46: which checkpoint transcribed this run."""
        return describe_hub_model(
            component="faster_whisper_streaming",
            implementation=type(self),
            model_id=self._model_name,
            model=None,
            tokenizer_revision="same-repo-as-model",
            params={
                "model_name": self._model_name,
                "src_lang": self._src_lang,
                "device": self._device,
                "compute_type": self._compute_type,
                "beam_size": self._beam_size,
                "num_workers": self._num_workers,
                "chunk_batch_size": self._chunk_batch_size,
                "stream_chunk_length": self._stream_chunk_length,
            },
            invocation={"src_lang": self._src_lang, "task": "transcribe"},
        )

    def push(self, chunk: AudioChunk) -> Sequence[AsrEvent]:
        self._check_not_flushed()
        self._buffer.append(chunk)

        # When batch full -> run streaming transcribe
        if len(self._buffer) >= self._chunk_batch_size:
            return self._process_batch()
        return []

    def _process_batch(self) -> Sequence[AsrEvent]:
        """Accumulate batch PCM and run streaming transcribe."""
        if not self._buffer:
            return []

        # Concatenate batch
        pcm = np.concatenate([c.pcm for c in self._buffer])
        rate = self._buffer[0].sample_rate
        self._buffer.clear()

        # Resample if needed
        if rate != 16000:
            import resampy
            pcm = resampy.resample(pcm, rate, 16000)

        # Append to streaming buffer
        self._stream_pcm = np.concatenate([self._stream_pcm, pcm])
        stream_dur = len(self._stream_pcm) / 16000
        print(
            f"[DEBUG ASR] batch: {len(pcm)/16000:.2f}s, "
            f"total stream: {stream_dur:.2f}s",
            flush=True,
        )

        # Start a new window once the untranscribed tail is long enough.
        if self._stream_gen is None:
            tail_s = stream_dur - self._last_segment_end
            if tail_s >= self._stream_chunk_length:
                print(
                    f"[DEBUG ASR] Starting window at {self._last_segment_end:.1f}s "
                    f"(tail={tail_s:.1f}s >= chunk_len={self._stream_chunk_length}s)",
                    flush=True,
                )
                self._stream_gen = self._make_generator(
                    self._last_segment_end,
                    max_len_s=self._stream_chunk_length,
                )
                print("[DEBUG ASR] Streaming generator started", flush=True)

        # Pull new segments from generator into pending list
        if self._stream_gen is not None:
            self._drain_generator(is_final=False)
        return self._pop_pending(is_final=False)

    def _make_generator(self, start_s: float, max_len_s: float | None = None):
        """Transcribe one window of captured PCM, timestamped by offset.

        The generator is finite: it covers at most ``max_len_s`` seconds and is
        discarded once drained. Capping the slice to a whole ``chunk_length``
        matters: a 5.06s slice with ``chunk_length=5`` becomes a 5.00s chunk
        plus a 0.06s sliver, and that padded sliver hallucinates and can run
        the full temperature fallback (6-12s for 60ms of audio).
        """
        start_sample = int(start_s * 16000)
        end_sample = len(self._stream_pcm)
        if max_len_s is not None:
            end_sample = min(
                end_sample, start_sample + int(max_len_s * 16000)
            )
        self._gen_offset_s = start_s
        self._gen_window_end_s = end_sample / 16000
        window_s = (end_sample - start_sample) / 16000
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            self._stream_pcm[start_sample:end_sample],
            language=self._src_lang,
            task="transcribe",
            beam_size=self._beam_size,
            vad_filter=False,
            chunk_length=self._stream_chunk_length,
        )
        create_ms = (time.perf_counter() - started) * 1000
        print(
            f"[ASR-TIME] make_gen start={start_s:.2f}s window={window_s:.2f}s "
            f"create_ms={create_ms:.0f} info_dur={info.duration:.2f}s "
            f"info_vad={info.duration_after_vad:.2f}s "
            f"lang_prob={info.language_probability:.2f}",
            flush=True,
        )
        return segments

    def _drain_generator(self, is_final: bool) -> None:
        """Pull segments from generator into pending list."""
        if self._stream_gen is None:
            return
        offset = self._gen_offset_s
        started = time.perf_counter()
        n_segments = 0
        progressed = False
        for segment in self._stream_gen:
            n_segments += 1
            start = segment.start + offset
            end = segment.end + offset
            print(
                f"[ASR-TIME] segment yield#{n_segments} "
                f"at={(time.perf_counter() - started) * 1000:.0f}ms "
                f"span={start:.2f}-{end:.2f}s "
                f"logprob={segment.avg_logprob:.3f} "
                f"compression={segment.compression_ratio:.2f} "
                f"no_speech={segment.no_speech_prob:.3f}",
                flush=True,
            )
            if end <= self._last_segment_end:
                continue  # skip overlap
            self._last_segment_end = end
            progressed = True
            text = segment.text.strip()
            if text:  # skip empty segments
                self._pending_segments.append(AsrEvent(
                    text=text,
                    t_audio_end=end,
                    t_wall=0.0,
                    is_final=is_final,
                    confidence=None,
                ))
                print(f"[DEBUG ASR] segment: {start:.1f}-{end:.1f}: {text}", flush=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not progressed:
            # The whole window was decoded and produced no usable segment.
            # Advance to its end so the next push does not decode it again.
            self._last_segment_end = max(
                self._last_segment_end, self._gen_window_end_s
            )
        print(
            f"[ASR-TIME] drain done segments={n_segments} "
            f"decode_ms={elapsed_ms:.0f} final={is_final} "
            f"last_end={self._last_segment_end:.2f}",
            flush=True,
        )
        # Exhausted; the next window starts where this one ended.
        self._stream_gen = None

    def _pop_pending(self, is_final: bool) -> Sequence[AsrEvent]:
        """Pop and return pending segments, marking is_final."""
        if not self._pending_segments:
            return []
        # AsrEvent is frozen: build marked copies instead of mutating in place.
        results = [
            replace(ev, is_final=is_final) for ev in self._pending_segments
        ]
        self._pending_segments.clear()
        return results

    def flush(self) -> Sequence[AsrEvent]:
        self._mark_flushed()
        results = []

        # Process remaining buffer
        if self._buffer:
            results.extend(self._process_batch())

        # Drain any remaining segments from generator
        if self._stream_gen is not None:
            self._drain_generator(is_final=True)

        # Decode the remaining tail even when shorter than chunk_length,
        # otherwise the last <stream_chunk_length seconds are never emitted.
        tail_s = len(self._stream_pcm) / 16000 - self._last_segment_end
        if tail_s >= 0.3:
            self._stream_gen = self._make_generator(self._last_segment_end)
            self._drain_generator(is_final=True)

        results.extend(self._pop_pending(is_final=True))

        return results

    def reset(self) -> None:
        super().reset()
        self._buffer = []
        self._stream_pcm = np.array([], dtype=np.float32)
        self._stream_gen = None
        self._gen_offset_s = 0.0
        self._gen_window_end_s = 0.0
        self._pending_segments = []
        self._last_segment_end = 0.0
        self._flushed = False