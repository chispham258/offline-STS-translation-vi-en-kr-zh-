"""Reusable contract tests every component implementation must pass.

Lives in src/ rather than tests/ because Phase 4 imports it to run real Whisper,
Opus-MT and MeloTTS wrappers through the same suite. That reuse is the whole
reason this is a suite and not a pile of per-class tests.

Usage:

    class TestEchoASR(ASRContract):
        def make_component(self):
            return EchoASR()
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np
import pytest

from onevoice.interfaces import BaseASR, BaseMT, BaseTTS, BaseVAD
from onevoice.types import AsrEvent, AudioChunk, MtEvent, TtsEvent, VadEvent


def _chunk(t_start: float = 0.0) -> AudioChunk:
    return AudioChunk(
        pcm=np.zeros(512, dtype=np.float32), sample_rate=16000, t_start=t_start
    )


class _StreamingContract(ABC):
    """Behaviour every streaming component shares, regardless of kind."""

    event_type: type
    base_class: type

    #: Whether a component of this kind is expected to emit at least one
    #: event for the suite's fixture input. False for VAD: the fixture chunk
    #: is silent PCM, and a VAD that correctly detects no speech in silence
    #: emits nothing - that is documented-correct behaviour, not a bug.
    emits_on_silent_input: bool = True

    #: Whether the component under test produces identical output for identical
    #: input. False for inherently stochastic generators — a VITS TTS such as
    #: Piper or MMS samples its duration predictor, so the same text yields a
    #: different sample count on every call (measured on Piper: 8704, 9984,
    #: 9472, 9728 for one phrase).
    #:
    #: **D21 correctness exception, granted 2026-08-03.** This is not a
    #: relaxation of the benchmark freeze: no evaluator behaviour, metric,
    #: latency figure or report changes. It fixes a defect in the certification
    #: suite itself, which assumed determinism the `BaseTTS`/`BaseMT` interfaces
    #: never required and therefore rejected a conforming implementation. A
    #: contract that cannot certify a correct component is wrong.
    #:
    #: The exception is scoped to output-equality comparison in
    #: `test_reset_restores_fresh_behaviour` only. Stochastic components are
    #: still held to every other guarantee: event types, concrete sequences,
    #: flush-then-push raising, reset clearing flush state, and — checked here —
    #: that reset restores the *shape* of a fresh component's behaviour.
    deterministic: bool = True

    @abstractmethod
    def make_component(self):
        """Return a fresh instance under test."""

    def _push(self, component):
        """Push one unit of whatever this component consumes."""
        raise NotImplementedError

    def _push_final(self, component):
        """Push a unit with is_final=True where the component supports it.

        VAD and ASR take an AudioChunk with no is_final parameter, so the
        default here is a no-op alias for _push. MTContract and TTSContract
        override this to actually exercise the is_final=True path.
        """
        return self._push(component)

    @staticmethod
    def _describe(events) -> list[str]:
        """A comparable summary of a result sequence.

        Built from every dataclass field rather than a hand-picked subset:
        a hand-picked subset silently drops whichever numeric or boolean
        field the picker forgot to name, and that field is exactly where a
        broken component leaks state. Avoids == on events: TtsEvent holds a
        numpy array and is eq=False, so direct comparison is identity and
        would never match across instances.

        t_wall is excluded: it is a wall-clock stamp, so two runs of a
        correct component may legitimately differ there even when nothing
        is wrong.
        """
        described = []
        for event in events:
            parts = []
            for field in dataclasses.fields(event):
                if field.name == "t_wall":
                    continue
                value = getattr(event, field.name)
                if isinstance(value, np.ndarray):
                    value = len(value)
                parts.append(f"{field.name}={value!r}")
            described.append(f"{type(event).__name__}({','.join(parts)})")
        return described

    #: How far a stochastic component's generated length may vary between two
    #: runs of the same input before it is treated as leaked state. Piper's
    #: measured spread for one phrase was 8704..9984, a factor of 1.15. One
    #: leaked utterance doubles the output, so this separates jitter from a
    #: bug with room to spare.
    stochastic_length_ratio: float = 1.6

    def _assert_reset_matches_fresh(self, after_reset, fresh, label):
        """Compare a post-reset result against a fresh component's.

        Deterministic components must match exactly.

        For a stochastic one (`deterministic = False`) exact equality is not a
        property the interface ever promised — but almost everything else still
        is. Every non-array field is still compared exactly, so a latched
        `is_final`, a drifting sample rate or a leaked text buffer is caught
        just as before. Only generated-array *lengths* are relaxed, and only to
        a bounded ratio: a component that leaks one utterance across `reset()`
        roughly doubles its output, which this still rejects.
        """
        if self.deterministic:
            assert self._describe(after_reset) == self._describe(fresh), label
            return

        assert len(after_reset) == len(fresh), f"{label}: event count differs"
        assert [type(e) for e in after_reset] == [type(e) for e in fresh], (
            f"{label}: event types differ"
        )

        for got, want in zip(after_reset, fresh, strict=True):
            for field in dataclasses.fields(got):
                if field.name == "t_wall":
                    continue
                a = getattr(got, field.name)
                b = getattr(want, field.name)
                if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                    lo, hi = len(a), len(b)
                    if lo > hi:
                        lo, hi = hi, lo
                    assert lo > 0 and hi / lo <= self.stochastic_length_ratio, (
                        f"{label}: {field.name} length {len(a)} vs {len(b)} "
                        f"exceeds the stochastic tolerance "
                        f"({self.stochastic_length_ratio}x) - this is state "
                        f"leaking across reset(), not sampling jitter"
                    )
                else:
                    assert a == b, f"{label}: {field.name} differs ({a!r} vs {b!r})"

    def test_component_inherits_the_shared_base_class(self):
        # D16's flush bookkeeping lives on the Base* class. A duck-typed
        # class can reimplement that behaviour and pass every test below
        # without actually sharing it.
        assert isinstance(self.make_component(), self.base_class)

    def test_push_returns_a_concrete_sequence(self):
        # D15: never a generator. A lazy return would defer the state mutation
        # until the caller iterated, and the bug would surface far downstream.
        result = self._push(self.make_component())
        assert isinstance(result, list | tuple)

    def test_flush_returns_a_concrete_sequence(self):
        assert isinstance(self.make_component().flush(), list | tuple)

    def test_is_flushed_starts_false(self):
        assert self.make_component().is_flushed is False

    def test_flush_marks_the_component_flushed(self):
        component = self.make_component()
        component.flush()
        assert component.is_flushed is True

    def test_push_after_flush_raises(self):
        # D16: pushing into a flushed component would splice two utterances.
        component = self.make_component()
        component.flush()
        with pytest.raises(RuntimeError, match="flush"):
            self._push(component)

    def test_reset_clears_flush_state(self):
        component = self.make_component()
        component.flush()
        component.reset()
        assert component.is_flushed is False

    def test_push_works_again_after_reset(self):
        component = self.make_component()
        component.flush()
        component.reset()
        assert isinstance(self._push(component), list | tuple)

    def test_reset_on_a_fresh_component_is_harmless(self):
        component = self.make_component()
        component.reset()
        assert component.is_flushed is False

    def test_repeated_flush_is_allowed(self):
        # A driver that flushes twice has a bug, but flush is a read of
        # remaining state - it must not corrupt anything, and a component
        # with nothing new to report must not re-emit what it already gave.
        component = self.make_component()
        component.flush()
        second = component.flush()
        assert isinstance(second, list | tuple)
        assert len(second) == 0

    def test_reset_restores_fresh_behaviour(self):
        # A component that leaks state across reset() would splice two
        # utterances together - the bug D16 exists to prevent, arriving by
        # another route. The pipeline calls reset() between utterances.
        #
        # Compares a full push, push, flush cycle rather than only the first
        # push: a batch component (the canonical Whisper wrapper shape,
        # documented on BaseASR) returns [] from push() and only emits on
        # flush(), so comparing first pushes alone compares [] to [] and
        # proves nothing.
        used = self.make_component()
        self._push(used)
        self._push(used)
        used.flush()
        used.reset()

        after_reset_push_1 = self._push(used)
        after_reset_push_2 = self._push(used)
        after_reset_flush = used.flush()

        fresh = self.make_component()
        fresh_push_1 = self._push(fresh)
        fresh_push_2 = self._push(fresh)
        fresh_flush = fresh.flush()

        self._assert_reset_matches_fresh(
            after_reset_push_1, fresh_push_1, "after_reset_push_1"
        )
        self._assert_reset_matches_fresh(
            after_reset_push_2, fresh_push_2, "after_reset_push_2"
        )
        self._assert_reset_matches_fresh(
            after_reset_flush, fresh_flush, "after_reset_flush"
        )

        # Same comparison again, but routed through an explicit is_final=True
        # push (via _push_final) rather than two plain pushes. A component
        # that latches state on is_final=True and never clears it in reset()
        # (MTContract/TTSContract's push takes is_final; VAD/ASR's _push_final
        # is a no-op alias for _push and so this is equivalent to the cycle
        # above for them) would otherwise never be exercised through this
        # path and could leak that latch across reset() undetected.
        used_final = self.make_component()
        self._push(used_final)
        self._push_final(used_final)
        used_final.flush()
        used_final.reset()

        after_reset_final_push_1 = self._push(used_final)
        after_reset_final_push_2 = self._push_final(used_final)
        after_reset_final_flush = used_final.flush()

        fresh_final = self.make_component()
        fresh_final_push_1 = self._push(fresh_final)
        fresh_final_push_2 = self._push_final(fresh_final)
        fresh_final_flush = fresh_final.flush()

        self._assert_reset_matches_fresh(
            after_reset_final_push_1, fresh_final_push_1, "after_reset_final_push_1"
        )
        self._assert_reset_matches_fresh(
            after_reset_final_push_2, fresh_final_push_2, "after_reset_final_push_2"
        )
        self._assert_reset_matches_fresh(
            after_reset_final_flush, fresh_final_flush, "after_reset_final_flush"
        )

    def test_push_returns_events_of_the_right_type(self):
        for event in self._push(self.make_component()):
            assert isinstance(event, self.event_type)

    def test_flush_returns_events_of_the_right_type(self):
        for event in self.make_component().flush():
            assert isinstance(event, self.event_type)

    def test_push_final_returns_a_concrete_sequence_of_the_right_type(self):
        # MT/TTS push() takes is_final; a bug in that branch is invisible if
        # the suite only ever calls push() with the default is_final=False.
        result = self._push_final(self.make_component())
        assert isinstance(result, list | tuple)
        for event in result:
            assert isinstance(event, self.event_type)

    def test_push_then_flush_yields_at_least_one_event(self):
        # A wrapper that swallows a model-load failure and emits nothing
        # from push or flush passes every assertion above vacuously, since
        # they all iterate an empty list. But for a component where silence
        # on the fixture input is legitimate (VAD), emitting nothing is not
        # itself evidence of a swallowed failure - just confirm it ran.
        component = self.make_component()
        events = (
            list(self._push(component))
            + list(self._push(component))
            + list(component.flush())
        )
        if not events and not self.emits_on_silent_input:
            pytest.skip(
                "component emitted nothing for silent input, which is "
                "legitimate for this kind of component"
            )
        assert len(events) >= 1


class VADContract(_StreamingContract):
    event_type = VadEvent
    base_class = BaseVAD
    emits_on_silent_input = False

    def _push(self, component, t_start: float = 0.0):
        return component.push(_chunk(t_start))

    # --- stimulus -----------------------------------------------------------
    #
    # F1: every timestamp assertion below used to be guarded by "if the
    # component emitted nothing, skip", and the only fixture was silence. A
    # correct VAD emits nothing on silence, so the guards always fired and a
    # VAD hardcoding t_audio=0.0 passed the whole suite. Certification has to
    # be able to make a detector speak before it can check what it says.

    def speech_chunks(self, t0: float = 0.0) -> list[AudioChunk]:
        """Audio that MUST cause this detector to emit `speech_start`.

        The default is a loud tone, which suffices for amplitude detectors. A
        model trained on speech will not fire on a tone and MUST override this
        with real audio — `test_speech_fixture_triggers_detection` fails loudly
        rather than skipping, so a stimulus that does not fit the component is
        a certification failure and not an invisible pass.
        """
        rate = 16000
        frame = 512
        n_lead = 6 * frame
        n_tone = 25 * frame
        n_tail = 8 * frame
        t = np.arange(n_tone) / rate
        pcm = np.concatenate(
            [
                np.zeros(n_lead, dtype=np.float32),
                (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                np.zeros(n_tail, dtype=np.float32),
            ]
        )
        return [
            AudioChunk(
                pcm=pcm[i : i + frame],
                sample_rate=rate,
                t_start=t0 + i / rate,
            )
            for i in range(0, len(pcm), frame)
        ]

    def _detect(self, t0: float = 0.0) -> list[VadEvent]:
        component = self.make_component()
        events: list[VadEvent] = []
        for chunk in self.speech_chunks(t0):
            events.extend(component.push(chunk))
        events.extend(component.flush())
        return events

    def test_speech_fixture_triggers_detection(self):
        """The stimulus must actually make this detector fire.

        This test is what stops the rest of the timestamp suite from
        evaporating. If it fails, override `speech_chunks()` with audio the
        component can detect — do not relax the assertions below.
        """
        starts = [e for e in self._detect() if e.kind == "speech_start"]
        assert starts, (
            f"{type(self.make_component()).__name__} emitted no speech_start "
            "for this contract's stimulus, so none of the timestamp assertions "
            "can run. Override speech_chunks() with audio this detector "
            "recognises."
        )

    def test_timestamps_fall_within_the_pushed_audio(self):
        """`t_audio` must land on the timeline it was given.

        Catches a detector that hardcodes 0.0 while the stream starts later,
        and any fixed or accumulating offset large enough to leave the window.
        """
        t0 = 7.0
        chunks = self.speech_chunks(t0)
        lo = chunks[0].t_start
        hi = chunks[-1].t_start + chunks[-1].duration_s
        for event in self._detect(t0):
            assert lo - 1e-6 <= event.t_audio <= hi + 1e-6, (
                f"{event.kind} reported t_audio={event.t_audio:.3f}s, outside "
                f"the pushed audio [{lo:.3f}, {hi:.3f}]s. t_audio must be on "
                "the same timeline as AudioChunk.t_start."
            )

    def test_timestamps_shift_with_the_audio_timeline(self):
        """The same audio pushed later must report proportionally later.

        This is the invariant a hardcoded, drifting or otherwise fabricated
        timestamp cannot satisfy: identical PCM at t0 and t0+delta must yield
        identical event kinds with every t_audio moved by exactly delta.
        """
        delta = 7.0
        base = self._detect(0.0)
        shifted = self._detect(delta)

        assert [e.kind for e in base] == [e.kind for e in shifted], (
            "shifting the audio timeline changed which events were emitted; "
            "detection must depend on the audio, not on its t_start"
        )
        for first, second in zip(base, shifted, strict=True):
            assert abs((second.t_audio - first.t_audio) - delta) < 1e-6, (
                f"{first.kind}: t_audio moved by "
                f"{second.t_audio - first.t_audio:.3f}s when the audio moved "
                f"by {delta:.3f}s. The component is not reading "
                "AudioChunk.t_start."
            )

    def test_timestamps_are_monotonic_within_a_stream(self):
        """Boundaries must arrive in the order they occurred."""
        times = [e.t_audio for e in self._detect()]
        assert times == sorted(times), (
            f"t_audio went backwards within one stream: {times}"
        )

    def test_push_reflects_the_chunks_t_start(self):
        # _chunk() defaults t_start to 0.0, and every push above uses that
        # default. A VAD that ignores chunk.t_start and hardcodes
        # t_audio=0.0 would pass every test above. Push at two distinct
        # non-zero t_start values and require the results differ.
        #
        # The fixture chunk's PCM is silent, so a VAD correctly detecting no
        # speech may emit nothing at either t_start - that proves nothing
        # about t_start handling either way. Only assert the relationship
        # when the component actually emitted something to compare.
        first_events = self._push(self.make_component(), t_start=1.0)
        second_events = self._push(self.make_component(), t_start=2.0)
        if not first_events and not second_events:
            pytest.skip(
                "component emitted nothing at either t_start, which is "
                "legitimate for silent input"
            )
        assert self._describe(first_events) != self._describe(second_events)


class ASRContract(_StreamingContract):
    event_type = AsrEvent
    base_class = BaseASR

    def _push(self, component):
        return component.push(_chunk())


class MTContract(_StreamingContract):
    event_type = MtEvent
    base_class = BaseMT

    def _push(self, component):
        return component.push("xin chào")

    def _push_final(self, component):
        return component.push("xin chào", is_final=True)


class TTSContract(_StreamingContract):
    event_type = TtsEvent
    base_class = BaseTTS

    def _push(self, component):
        return component.push("hello")

    def _push_final(self, component):
        return component.push("hello", is_final=True)
