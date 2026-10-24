"""End-to-end evaluation over the golden fixtures."""

import pytest

from onevoice.eval.evaluator import evaluate
from onevoice.eval.manifest import RunRecord, write_run

REFS = "fixtures/mini_vi_en/refs.jsonl"
RUN = "fixtures/mini_vi_en/run.jsonl"


def test_evaluates_the_golden_fixture():
    table = evaluate(RUN, REFS, experiment="golden")
    assert table.experiment == "golden"
    expected = {"bleu", "chrf", "wer", "cer", "end_to_speech_ms", "rtf"}
    assert expected <= set(table.metrics)


def test_errored_utterances_are_excluded_but_counted():
    table = evaluate(RUN, REFS)
    assert table.counts["total"] == 3
    assert table.counts["scored"] == 2
    assert table.counts["errored"] == 1


def test_error_count_is_reported_next_to_every_score():
    # The point of counts living on the table: no score can be read without it.
    table = evaluate(RUN, REFS)
    assert table.counts["errored"] == 1
    assert table.metrics["bleu"].value > 0


def test_perfect_run_scores_100_bleu(tmp_path):
    records = [
        RunRecord(
            utt_id="utt-001",
            hyp_text="Hello, how are you?",
            asr_text="Xin chào, anh khỏe không?",
            audio_duration_s=2.0,
            timing={"speech_end": 1.8, "tts_first_audio": 2.3, "tts_done": 3.0},
        ),
        RunRecord(
            utt_id="utt-002",
            hyp_text="The conveyor belt is broken.",
            asr_text="Cái băng chuyền bị hỏng rồi.",
            audio_duration_s=2.6,
            timing={"speech_end": 2.4, "tts_first_audio": 2.9, "tts_done": 3.6},
        ),
        RunRecord(
            utt_id="utt-003",
            hyp_text="Have you seen my safety gear?",
            asr_text="Anh có thấy đồ bảo hộ của tôi không?",
            audio_duration_s=3.1,
            timing={"speech_end": 2.9, "tts_first_audio": 3.4, "tts_done": 4.1},
        ),
    ]
    path = tmp_path / "perfect.jsonl"
    write_run(path, records)
    table = evaluate(path, REFS)
    assert table.metrics["bleu"].value == pytest.approx(100.0)
    assert table.metrics["wer"].value == pytest.approx(0.0)
    assert table.counts["errored"] == 0


def test_run_record_without_a_reference_is_a_hard_error(tmp_path):
    path = tmp_path / "orphan.jsonl"
    write_run(
        path,
        [RunRecord(utt_id="utt-999", hyp_text="x", audio_duration_s=1.0)],
    )
    with pytest.raises(KeyError, match="utt-999"):
        evaluate(path, REFS)


def test_all_utterances_errored_yields_no_scores_but_still_reports(tmp_path):
    # Must still cover the full reference corpus (finding 1): all 3 utt_ids,
    # all errored, not just one.
    path = tmp_path / "allbad.jsonl"
    write_run(
        path,
        [
            RunRecord(
                utt_id=utt_id,
                hyp_text="",
                status="error",
                error="boom",
                audio_duration_s=2.0,
            )
            for utt_id in ("utt-001", "utt-002", "utt-003")
        ],
    )
    table = evaluate(path, REFS)
    assert table.counts["scored"] == 0
    assert table.counts["errored"] == 3
    assert table.metrics["bleu"].value != table.metrics["bleu"].value  # NaN


def test_run_missing_reference_utterances_is_a_hard_error(tmp_path):
    # A run covering only utt-001 of the 3-utterance reference set must not
    # score as a clean, complete-looking table.
    path = tmp_path / "partial.jsonl"
    write_run(
        path,
        [
            RunRecord(
                utt_id="utt-001",
                hyp_text="Hello, how are you?",
                audio_duration_s=2.0,
            )
        ],
    )
    with pytest.raises(KeyError, match="utt-002"):
        evaluate(path, REFS)


def test_counts_report_the_full_reference_corpus_size(tmp_path):
    table = evaluate(RUN, REFS)
    assert table.counts["references"] == 3


def test_mixed_target_languages_are_rejected(tmp_path):
    # BLEU tokenizer is per target language (D14); a corpus spanning two
    # target languages has no single correct tokenizer.
    import json

    refs_path = tmp_path / "mixed_refs.jsonl"
    rows = [
        {
            "utt_id": "a",
            "audio_path": "a.wav",
            "src_lang": "vi",
            "tgt_lang": "en",
            "src_text": "x",
            "ref_text": "hello",
            "speech_start_s": 0.0,
            "speech_end_s": 1.0,
        },
        {
            "utt_id": "b",
            "audio_path": "b.wav",
            "src_lang": "vi",
            "tgt_lang": "zh",
            "src_text": "y",
            "ref_text": "你好",
            "speech_start_s": 0.0,
            "speech_end_s": 1.0,
        },
    ]
    refs_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    run_path = tmp_path / "mixed_run.jsonl"
    write_run(
        run_path,
        [
            RunRecord(utt_id="a", hyp_text="hello", audio_duration_s=1.0),
            RunRecord(utt_id="b", hyp_text="你好", audio_duration_s=1.0),
        ],
    )
    with pytest.raises(ValueError, match="target language"):
        evaluate(run_path, refs_path)
