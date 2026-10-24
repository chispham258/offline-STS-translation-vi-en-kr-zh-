"""Manifest schemas and JSONL round-tripping."""

import pytest
from pydantic import ValidationError

from onevoice.eval.manifest import RefRecord, RunRecord, read_refs, read_run, write_run

FIXTURES = "fixtures/mini_vi_en"


def test_reads_reference_fixture():
    refs = read_refs(f"{FIXTURES}/refs.jsonl")
    assert len(refs) == 3
    assert refs[0].utt_id == "utt-001"
    assert refs[0].src_lang == "vi"
    assert refs[0].tgt_lang == "en"
    assert refs[0].ref_text


def test_reads_run_fixture_including_an_errored_utterance():
    run = read_run(f"{FIXTURES}/run.jsonl")
    assert len(run) == 3
    statuses = {r.utt_id: r.status for r in run}
    assert statuses == {"utt-001": "ok", "utt-002": "ok", "utt-003": "error"}


def test_errored_record_carries_its_error_message():
    run = {r.utt_id: r for r in read_run(f"{FIXTURES}/run.jsonl")}
    assert run["utt-003"].error
    assert run["utt-003"].hyp_text == ""


def test_status_must_be_ok_or_error():
    with pytest.raises(ValidationError):
        RunRecord(utt_id="x", hyp_text="y", status="maybe", audio_duration_s=1.0)


def test_reference_requires_speech_boundaries():
    with pytest.raises(ValidationError):
        RefRecord(
            utt_id="x",
            audio_path="a.wav",
            src_lang="vi",
            tgt_lang="en",
            src_text="a",
            ref_text="b",
        )


def test_duplicate_utt_id_in_refs_is_rejected(tmp_path):
    # A retried/appended manifest with a repeated utt_id must not silently
    # keep-last; that turns a 1-utterance corpus into a fabricated 3.
    path = tmp_path / "dup_refs.jsonl"
    row = (
        '{"utt_id": "utt-001", "audio_path": "a.wav", "src_lang": "vi", '
        '"tgt_lang": "en", "src_text": "x", "ref_text": "y", '
        '"speech_start_s": 0.0, "speech_end_s": 1.0}'
    )
    path.write_text("\n".join([row, row]) + "\n")
    with pytest.raises(ValueError, match="utt-001"):
        read_refs(path)


def test_duplicate_utt_id_in_run_is_rejected(tmp_path):
    path = tmp_path / "dup_run.jsonl"
    records = [
        RunRecord(utt_id="utt-001", hyp_text="a", audio_duration_s=1.0),
        RunRecord(utt_id="utt-001", hyp_text="b", audio_duration_s=1.0),
    ]
    write_run(path, records)
    with pytest.raises(ValueError, match="utt-001"):
        read_run(path)


def test_run_records_round_trip_through_jsonl(tmp_path):
    records = [
        RunRecord(
            utt_id="utt-001",
            hyp_text="hello",
            status="ok",
            audio_duration_s=2.0,
            timing={"speech_end": 1.0, "tts_first_audio": 1.5},
        )
    ]
    path = tmp_path / "run.jsonl"
    write_run(path, records)
    assert read_run(path) == records
