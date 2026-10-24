"""Distribution tracker (D77): a person may take several sessions only in
different blocks. The key cannot know who is who, so the tracker checks it."""

from __future__ import annotations

import csv

from scripts.distribute_mos_panel import TRACKER_FIELDS, check_tracker


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACKER_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRACKER_FIELDS})
    return path


def test_one_person_in_two_different_blocks_is_fine(tmp_path):
    rows = [
        {"session": "1", "block": "0", "person": "An"},
        {"session": "2", "block": "1", "person": "An"},
        {"session": "3", "block": "2", "person": ""},
    ]
    assert check_tracker(_write(tmp_path / "t.csv", rows)) == []


def test_one_person_twice_in_the_same_block_is_flagged(tmp_path):
    rows = [
        {"session": "1", "block": "0", "person": "An"},
        {"session": "9", "block": "0", "person": " an "},  # same person, typed loosely
        {"session": "2", "block": "1", "person": "Binh"},
    ]
    problems = check_tracker(_write(tmp_path / "t.csv", rows))
    assert len(problems) == 1
    assert "block 0" in problems[0] and "1" in problems[0] and "9" in problems[0]
