"""F5: every metric must measure what a reader believes it measures (D49).

Two mismatches were confirmed by audit and are pinned here.

1. `wer`/`cer` charge the model for casing and punctuation it was never asked
   to produce. Normalised variants are reported alongside — never instead of.
2. `scored` and `errored` count records, not results. An utterance that ran
   cleanly and produced nothing was invisible in both.
"""

from __future__ import annotations

from onevoice.eval.metrics.error_rate import (
    cer_normalized,
    normalize_transcript,
    wer,
    wer_normalized,
)

# --- normalisation does what it claims, and only that -----------------------


def test_casing_and_punctuation_are_removed():
    assert normalize_transcript("Xin chào, bạn!") == "xin chào bạn"


def test_vietnamese_diacritics_survive_normalisation():
    """The failure that would silently destroy the metric.

    A naive `isalnum()` filter strips combining marks in decomposed text, which
    would turn every toned vowel into a bare one and make WER look excellent.
    """
    text = "chủ nghĩa lãng mạn quyết định"
    assert normalize_transcript(text) == text


def test_normalisation_is_idempotent():
    once = normalize_transcript("Hello,  World!")
    assert normalize_transcript(once) == once


def test_normalisation_does_not_expand_numbers():
    """Documented limitation: the normalised figure still contains this.

    Whisper writes `2`, FLEURS writes `hai`. Expanding digits is
    language-specific enough to be a methodology change, so it is not done —
    and the docstring says so rather than letting a reader assume otherwise.
    """
    assert normalize_transcript("có 2 người") == "có 2 người"


def test_normalised_error_is_never_worse_than_raw_for_formatting_only_diffs():
    hyps = ["Xin chào, bạn!"]
    refs = ["xin chào bạn"]
    assert wer(hyps, refs).value > 0.0
    assert wer_normalized(hyps, refs).value == 0.0
    assert cer_normalized(hyps, refs).value == 0.0


def test_normalisation_does_not_hide_a_real_substitution():
    """The counterexample that matters: normalisation must not flatter.

    `ý bố` for `yếu tố` is a genuine ASR error. If normalising made it vanish,
    the normalised metric would be measuring nothing.
    """
    hyps = ["chủ nghĩa lặng mẹ có một ý bố lớn"]
    refs = ["chủ nghĩa lãng mạn có một yếu tố lớn"]
    # 4 substitutions over 9 words = 0.444. The threshold is set below the
    # measured value on purpose: the claim is that normalisation does not
    # flatter a real error, not that the error has a particular size.
    assert wer_normalized(hyps, refs).value > 0.4
    assert cer_normalized(hyps, refs).value > 0.1


def test_raw_metrics_are_untouched_by_the_addition():
    """D49 permits adding a measurement, not redefining one."""
    hyps = ["Xin chào, bạn!"]
    refs = ["xin chào bạn"]
    assert wer(hyps, refs).value == wer(hyps, refs).value
    assert wer(hyps, refs).name == "wer"
    assert wer_normalized(hyps, refs).name == "wer_normalized"


# --- empty outputs are visible ----------------------------------------------


def test_empty_outputs_are_counted_separately_from_errors(tmp_path):
    """A clean run that produced nothing is neither scored-success nor error.

    This is the count whose absence let a 24% dropout read as `errored: 0`.
    """
    import json

    from onevoice.eval.evaluator import evaluate

    refs = tmp_path / "refs.jsonl"
    runs = tmp_path / "run.jsonl"
    rows, records = [], []
    for i in range(4):
        rows.append(
            {
                "utt_id": f"u{i}",
                "audio_path": f"a{i}.wav",
                "src_lang": "vi",
                "tgt_lang": "en",
                "src_text": "xin chào",
                "ref_text": "hello",
                "speech_start_s": 0.0,
                "speech_end_s": 1.0,
            }
        )
        records.append(
            {
                "utt_id": f"u{i}",
                # Two produce nothing, cleanly.
                "hyp_text": "" if i < 2 else "hello",
                "asr_text": "" if i < 2 else "xin chào",
                "status": "ok",
                "audio_duration_s": 1.0,
                "timing": {},
            }
        )
    refs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    runs.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    table = evaluate(runs, refs, experiment="f5")
    assert table.counts["scored"] == 4
    assert table.counts["errored"] == 0
    assert table.counts["empty_outputs"] == 2


def test_a_fully_productive_run_reports_no_empty_outputs(tmp_path):
    """The counterweight: the count must not fire on a healthy run."""
    import json

    from onevoice.eval.evaluator import evaluate

    refs = tmp_path / "refs.jsonl"
    runs = tmp_path / "run.jsonl"
    row = {
        "utt_id": "u0",
        "audio_path": "a.wav",
        "src_lang": "vi",
        "tgt_lang": "en",
        "src_text": "xin chào",
        "ref_text": "hello",
        "speech_start_s": 0.0,
        "speech_end_s": 1.0,
    }
    rec = {
        "utt_id": "u0",
        "hyp_text": "hello",
        "asr_text": "xin chào",
        "status": "ok",
        "audio_duration_s": 1.0,
        "timing": {},
    }
    refs.write_text(json.dumps(row) + "\n")
    runs.write_text(json.dumps(rec) + "\n")
    assert evaluate(runs, refs, experiment="f5").counts["empty_outputs"] == 0
