"""MOS panel session plan (protocol §6-§8, D74).

Pins the frozen design: which clips a rater hears, that the blinding key hides
every system cue, the duplicate stride, the per-rater order rules, and the
Huggins-pitch headphone stimuli.
"""

from __future__ import annotations

import numpy as np
import pytest

from onevoice import panel

SYSTEMS = ("system:piper-vi-vais1000", "system:mms-tts-vie", "system:f5")
IDS = [str(1000 + 7 * i) for i in range(40)]
# Rater 1's headphone draw for these clips, recorded before D76.
PLAN_1_HEADPHONE = (2, 9, 1, 4, 10, 3)


def _records(ids=IDS, roles=("upper_anchor", "lower_anchor", *SYSTEMS)):
    return [
        {
            "role": role,
            "utt_id": u,
            "output_path": f"stimuli/{role}/{u}.wav",
            "output_sha256": f"{role}-{u}",
            "duration_s": 5.0,
        }
        for role in roles
        for u in ids
    ]


# --- clips and blinding -----------------------------------------------------


def test_every_stimulus_becomes_one_opaque_clip():
    clips = panel.build_clips(_records(), IDS)
    assert len(clips) == 200
    assert all(c.duplicate_of is None for c in clips)


def test_clip_ids_are_opaque_unique_and_reproducible():
    a, b = panel.build_clips(_records(), IDS), panel.build_clips(_records(), IDS)
    assert [c.clip_id for c in a] == [c.clip_id for c in b]
    ids = [c.clip_id for c in a]
    assert len(set(ids)) == len(ids)
    for c in a:
        assert c.clip_id.isalnum() and len(c.clip_id) == 16
        for cue in ("piper", "mms", "f5", "anchor", "system", c.utt_id):
            assert cue not in c.clip_id


# --- blocks and sessions (D77) ------------------------------------------------


def test_blocks_are_consecutive_fives_in_manifest_order():
    blocks = panel.blocks(IDS)
    assert len(blocks) == 8 and all(len(b) == 5 for b in blocks)
    assert [u for b in blocks for u in b] == IDS
    with pytest.raises(ValueError):
        panel.blocks(IDS[:12])  # not a multiple of the block size


def _session(clips, rater):
    plan = panel.rater_plan(clips, IDS, rater)
    by_id = {c.clip_id: c for c in [*clips, plan.duplicate]}
    return plan, [by_id[cid] for cid in plan.trial_order]


@pytest.mark.parametrize("rater", range(1, 25))
def test_a_session_is_one_block_all_systems_two_plus_two_anchors_one_repeat(rater):
    clips = panel.build_clips(_records(), IDS)
    plan, trials = _session(clips, rater)
    g, j = (rater - 1) % 8, (rater - 1) // 8
    blocks = panel.blocks(IDS)
    assert plan.block == g and len(trials) == 20
    system = [c for c in trials if c.role in SYSTEMS and c.duplicate_of is None]
    assert {(c.role, c.utt_id) for c in system} == {
        (r, u) for r in SYSTEMS for u in blocks[g]
    }
    positions = [(2 * j) % 5, (2 * j + 1) % 5]
    assert {c.utt_id for c in trials if c.role == "upper_anchor"} == {
        blocks[(g + 1) % 8][p] for p in positions
    }
    assert {c.utt_id for c in trials if c.role == "lower_anchor"} == {
        blocks[(g + 2) % 8][p] for p in positions
    }
    (dup,) = [c for c in trials if c.duplicate_of is not None]
    assert dup is plan.duplicate and dup.duplicate_of in {c.clip_id for c in system}
    src = next(c for c in system if c.clip_id == dup.duplicate_of)
    assert (dup.role, dup.utt_id, dup.stimulus_sha256) == (
        src.role,
        src.utt_id,
        src.stimulus_sha256,
    )


@pytest.mark.parametrize("rater", range(1, 41))
def test_no_utterance_is_heard_twice_in_a_row(rater):
    _, trials = _session(panel.build_clips(_records(), IDS), rater)
    assert all(a.utt_id != b.utt_id for a, b in zip(trials, trials[1:], strict=False))


def test_duplicate_ids_are_opaque_and_never_collide():
    clips = panel.build_clips(_records(), IDS)
    seen = {c.clip_id for c in clips}
    for rater in range(1, 81):
        dup = panel.rater_plan(clips, IDS, rater).duplicate
        assert dup.clip_id not in seen and len(dup.clip_id) == 16
        assert set(dup.clip_id) <= set(panel.CLIP_ID_ALPHABET)
        seen.add(dup.clip_id)


def test_eighty_sessions_give_ten_per_block():
    clips = panel.build_clips(_records(), IDS)
    blocks = [panel.rater_plan(clips, IDS, n).block for n in range(1, 81)]
    assert all(blocks.count(g) == 10 for g in range(8))


def test_plans_are_seeded_by_the_rater_and_differ_between_raters():
    clips = panel.build_clips(_records(), IDS)
    assert panel.rater_plan(clips, IDS, 9) == panel.rater_plan(clips, IDS, 9)
    # Raters 1 and 9 share block 0 but not their order or repeat.
    assert (
        panel.rater_plan(clips, IDS, 1).trial_order
        != panel.rater_plan(clips, IDS, 9).trial_order
    )


def test_each_rater_draws_six_distinct_headphone_trials_from_twelve():
    clips = panel.build_clips(_records(), IDS)
    for rater in range(1, 21):
        draw = panel.rater_plan(clips, IDS, rater).headphone_trials
        assert len(draw) == 6 == len(set(draw))
        assert all(0 <= t < 12 for t in draw)


def test_an_errored_cell_is_not_presented():
    records = [
        r
        for r in _records()
        if not (r["role"] == "system:f5" and r["utt_id"] == IDS[3])
    ]
    _, trials = _session(panel.build_clips(records, IDS), 1)
    assert ("system:f5", IDS[3]) not in {(c.role, c.utt_id) for c in trials}
    assert len(trials) == 19


def test_rater_numbers_start_at_one():
    with pytest.raises(ValueError):
        panel.rater_plan(panel.build_clips(_records(), IDS), IDS, 0)


# --- Huggins pitch ------------------------------------------------------------


def _band_mask(n):
    f = np.fft.rfftfreq(n, 1 / panel.HP_SR)
    return (f >= 600 * 0.94) & (f <= 600 * 1.06)


def test_the_pool_is_twelve_reproducible_stereo_trials():
    pool, again = panel.huggins_pool(), panel.huggins_pool()
    assert len(pool) == 12
    for (a, ta), (b, tb) in zip(pool, again, strict=True):
        assert ta == tb and np.array_equal(a, b)
    assert {t for _, t in pool} <= {0, 1, 2}
    audio, _ = pool[0]
    expected = 3 * panel.HP_SR + 2 * int(panel.HP_GAP_S * panel.HP_SR)
    assert audio.shape == (expected, 2)


@pytest.mark.parametrize("target", [True, False])
def test_only_a_target_interval_is_phase_inverted_and_only_in_the_band(target):
    # Checked before ramping: a time-domain ramp spreads energy across bins,
    # so the exact phase relation holds only for the raw dichotic pair.
    noise = np.random.default_rng(1).standard_normal(panel.HP_SR)
    left, right = panel.dichotic(noise, target=target).T
    cross = np.fft.rfft(left) * np.conj(np.fft.rfft(right))
    band = _band_mask(panel.HP_SR)
    inside = np.angle(cross[band])
    expected = np.pi if target else 0.0
    assert np.allclose(np.abs(inside), expected, atol=1e-9)
    assert np.allclose(np.angle(cross[~band]), 0.0, atol=1e-9)
    np.testing.assert_array_equal(right, noise)


def test_each_pool_trial_has_exactly_one_dichotic_interval():
    n, step = panel.HP_SR, panel.HP_SR + int(panel.HP_GAP_S * panel.HP_SR)
    for audio, target in panel.huggins_pool():
        differs = [
            not np.array_equal(
                audio[k * step : k * step + n, 0], audio[k * step : k * step + n, 1]
            )
            for k in range(3)
        ]
        assert differs == [k == target for k in range(3)]


def test_huggins_level_is_the_owner_set_rms():
    audio, _ = panel.huggins_pool()[0]
    n = panel.HP_SR
    ramp = int(panel.HP_RAMP_S * panel.HP_SR)
    steady = audio[ramp : n - ramp, 1]
    assert 20 * np.log10(np.sqrt(np.mean(steady**2))) == pytest.approx(
        panel.HP_RMS_DBFS, abs=0.3
    )


# --- rater package -------------------------------------------------------------

TEXT_KEYS = {
    "question",
    "scale",
    "instructions",
    "headphone_instructions",
    "play",
    "replay",
    "choose_interval",
    "failed_screen",
    "finished",
    "download",
    "progress",
    "stereo_instructions",
    "stereo_left_first",
    "stereo_right_first",
    "stereo_unsure",
    "stereo_fix",
}


def _text(approved=True):
    return {
        "approved": approved,
        "approved_by": "native speaker" if approved else None,
        "question": "Q?",
        "scale": {"5": "a", "4": "b", "3": "c", "2": "d", "1": "e"},
        **{k: k for k in TEXT_KEYS - {"question", "scale"}},
    }


def _stimuli(tmp_path):
    import soundfile as sf

    records = []
    for r in _records(ids=IDS[:15]):
        path = tmp_path / "stim" / r["role"].replace(":", "_") / f"{r['utt_id']}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(1600), 16_000, subtype="PCM_16")
        records.append({**r, "output_path": str(path)})
    return records


def test_a_package_holds_the_page_and_opaque_audio_only(tmp_path):
    pytest.importorskip("soundfile")
    clips = panel.build_clips(_stimuli(tmp_path), IDS[:15])
    plan = panel.rater_plan(clips, IDS[:15], 3)
    out = panel.build_package(clips, plan, _text(), tmp_path / "pkg")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert names == (
        {"index.html"}
        | {f"audio/{c}.wav" for c in plan.trial_order}
        | {f"audio/hp{k + 1}.wav" for k in range(panel.HP_TRIALS)}
        | {"audio/stereo_a.wav", "audio/stereo_b.wav"}
    )
    page = (out / "index.html").read_text(encoding="utf-8")
    for cue in ("piper", "mms", "f5", "anchor", "system", "vais", *IDS[:15]):
        assert cue not in page.lower()
    by_id = {c.clip_id: c for c in [*clips, plan.duplicate]}
    for cid in plan.trial_order:
        src = by_id[cid].stimulus_path
        assert (out / "audio" / f"{cid}.wav").read_bytes() == open(src, "rb").read()


def test_the_page_carries_the_plan_in_order(tmp_path):
    pytest.importorskip("soundfile")
    clips = panel.build_clips(_stimuli(tmp_path), IDS[:15])
    plan = panel.rater_plan(clips, IDS[:15], 3)
    out = panel.build_package(clips, plan, _text(), tmp_path / "pkg")
    session = panel.read_page_session(out / "index.html")
    assert [t["clip_id"] for t in session["trials"]] == list(plan.trial_order)
    assert session["rater"] == 3 and session["pilot"] is False
    assert len(session["headphone"]) == panel.HP_TRIALS
    assert session["max_replays"] == 2 and session["hp_pass"] == 6


def test_unapproved_text_builds_only_a_labelled_pilot(tmp_path):
    pytest.importorskip("soundfile")
    clips = panel.build_clips(_stimuli(tmp_path), IDS[:15])
    plan = panel.rater_plan(clips, IDS[:15], 1)
    with pytest.raises(ValueError, match="approved"):
        panel.build_package(clips, plan, _text(approved=False), tmp_path / "a")
    out = panel.build_package(
        clips, plan, _text(approved=False), tmp_path / "b", pilot=True
    )
    assert panel.read_page_session(out / "index.html")["pilot"] is True


def test_text_missing_a_field_is_refused(tmp_path):
    clips = panel.build_clips(_records(ids=IDS[:12]), IDS[:12])
    text = _text()
    del text["question"]
    with pytest.raises(ValueError, match="question"):
        panel.build_package(
            clips, panel.rater_plan(clips, IDS[:15], 1), text, tmp_path / "x"
        )


# --- stereo check (D76): an unscored setup aid before the screen -------------


@pytest.mark.parametrize("left_first", [True, False])
def test_the_stereo_check_plays_one_ear_then_the_other(left_first):
    audio = panel.stereo_check_audio(left_first)
    n = int(panel.STEREO_TONE_S * panel.STEREO_SR)
    first, second = audio[:n], audio[-n:]
    ear = 0 if left_first else 1
    assert np.any(first[:, ear]) and not np.any(first[:, 1 - ear])
    assert np.any(second[:, 1 - ear]) and not np.any(second[:, ear])


def test_the_session_carries_a_seeded_varying_stereo_order(tmp_path):
    pytest.importorskip("soundfile")
    import soundfile as sf

    clips = panel.build_clips(_stimuli(tmp_path), IDS[:15])
    plan = panel.rater_plan(clips, IDS[:15], 3)
    out = panel.build_package(clips, plan, _text(), tmp_path / "pkg")
    stereo = panel.read_page_session(out / "index.html")["stereo"]
    assert len(stereo) == panel.STEREO_ATTEMPTS
    assert {s["left_first"] for s in stereo} == {True, False}
    again = panel.build_package(clips, plan, _text(), tmp_path / "pkg2")
    assert panel.read_page_session(again / "index.html")["stereo"] == stereo
    for s in stereo:
        y, _ = sf.read(out / s["src"])
        n = int(panel.STEREO_TONE_S * panel.STEREO_SR)
        assert np.any(y[:n, 0 if s["left_first"] else 1])


def test_the_stereo_check_does_not_change_any_rater_plan():
    clips = panel.build_clips(_records(), IDS)
    # Pinned before D76: adding the check must not move the seeded draws.
    assert panel.rater_plan(clips, IDS[:15], 1).headphone_trials == PLAN_1_HEADPHONE
