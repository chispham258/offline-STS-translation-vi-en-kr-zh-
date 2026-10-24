"""MOS panel session plan (TTS pre-registration §6-§8, §9A; D74, D76, D77).

Builds what every rater hears: the clips behind opaque ids, each short
session's block, anchors, repeat, trial order and headphone-trial draw, and
the Huggins-pitch and stereo-check stimuli. Imports no model or pipeline code.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BLOCK_SIZE = 5  # D77: 8 blocks of 5 utterances
ANCHORS_PER_KIND = 2  # 2 upper + 2 lower per session
CLIP_ID_SEED = 0
CLIP_ID_LENGTH = 16
# No digits (utterance ids are digits) and none of the letters in "piper",
# "mms", "f5", "anchor" or "system": no clip id can spell a system cue.
CLIP_ID_ALPHABET = "bdgjklqvwxz"

# Huggins pitch after Milne et al. (2021); ramps, gap and level are owner-set.
HP_SR = 44_100
HP_INTERVAL_S = 1.0
HP_CENTRE_HZ = 600.0
HP_HALF_BAND = 0.06
HP_POOL = 12
HP_TRIALS = 6
HP_PASS = 6
HP_POOL_SEED = 0
HP_RAMP_S = 0.020
HP_GAP_S = 0.500
HP_RMS_DBFS = -26.0

# Stereo check (D76): an unscored setup aid shown before the screen. A beep in
# one ear, then the other, in a seeded varying order; the rater moves on only
# after naming the first ear. It is never an exclusion criterion.
STEREO_SR = 44_100
STEREO_TONE_HZ = 500.0
STEREO_TONE_S = 1.0
STEREO_GAP_S = 0.5
STEREO_AMPLITUDE = 0.2
STEREO_ATTEMPTS = 20


@dataclass(frozen=True)
class Clip:
    """One presentable item. `duplicate_of` names the original's clip id."""

    clip_id: str
    role: str
    utt_id: str
    stimulus_path: str
    stimulus_sha256: str
    duration_s: float
    duplicate_of: str | None = None


@dataclass(frozen=True)
class RaterPlan:
    """One short session (D77). `duplicate` is the session's own repeat clip."""

    rater: int
    seed: int
    block: int
    headphone_trials: tuple[int, ...]
    duplicate: Clip
    trial_order: tuple[str, ...]


def _opaque_id(rng: random.Random) -> str:
    return "".join(rng.choice(CLIP_ID_ALPHABET) for _ in range(CLIP_ID_LENGTH))


def build_clips(records: list[dict[str, Any]], manifest_ids: list[str]) -> list[Clip]:
    """Every prepared stimulus as one clip behind an opaque id. Errored cells
    have no record and so are never presented (§8)."""
    order = {u: i for i, u in enumerate(manifest_ids)}
    records = sorted(records, key=lambda r: (r["role"], order[r["utt_id"]]))
    rng = random.Random(CLIP_ID_SEED)
    used: set[str] = set()
    clips = []
    for r in records:
        cid = _opaque_id(rng)
        while cid in used:
            cid = _opaque_id(rng)
        used.add(cid)
        clips.append(
            Clip(
                clip_id=cid,
                role=r["role"],
                utt_id=r["utt_id"],
                stimulus_path=r["output_path"],
                stimulus_sha256=r["output_sha256"],
                duration_s=r["duration_s"],
            )
        )
    return clips


def blocks(manifest_ids: list[str]) -> list[list[str]]:
    """D77: consecutive blocks of `BLOCK_SIZE` in manifest order."""
    if len(manifest_ids) % BLOCK_SIZE:
        raise ValueError(f"{len(manifest_ids)} utterances do not split into blocks")
    return [
        list(manifest_ids[i : i + BLOCK_SIZE])
        for i in range(0, len(manifest_ids), BLOCK_SIZE)
    ]


def _no_consecutive(items: list[Clip]) -> list[Clip]:
    """Take items in the given (shuffled) order, skipping ahead whenever the
    next would repeat the previous utterance; a leftover that cannot be placed
    at the end goes into the first gap between two other utterances."""
    remaining, out = list(items), []
    while remaining:
        last = out[-1].utt_id if out else None
        pick = next((i for i, c in enumerate(remaining) if c.utt_id != last), None)
        if pick is None:
            c = remaining.pop(0)
            slot = next(
                i
                for i in range(len(out) + 1)
                if (i == 0 or out[i - 1].utt_id != c.utt_id)
                and (i == len(out) or out[i].utt_id != c.utt_id)
            )
            out.insert(slot, c)
        else:
            out.append(remaining.pop(pick))
    return out


def rater_plan(clips: list[Clip], manifest_ids: list[str], rater: int) -> RaterPlan:
    """Session `rater` (1-based) under D77.

    Block `g = (rater - 1) mod B`, index within it `j = (rater - 1) div B`. The
    15 system clips of block g, upper anchors of block g+1 and lower anchors of
    block g+2 at positions 2j and 2j+1 (mod 5), and one repeat. `Random(rater)`
    draws the headphone trials, then the repeated clip, then the shuffle.
    """
    if rater < 1:
        raise ValueError("raters are numbered from 1")
    bl = blocks(manifest_ids)
    if len(bl) < 3:
        raise ValueError("anchors come from two other blocks; need >= 3 blocks")
    g, j = (rater - 1) % len(bl), (rater - 1) // len(bl)
    positions = [(2 * j + k) % BLOCK_SIZE for k in range(ANCHORS_PER_KIND)]
    upper = {bl[(g + 1) % len(bl)][p] for p in positions}
    lower = {bl[(g + 2) % len(bl)][p] for p in positions}
    system = [c for c in clips if c.role.startswith("system:") and c.utt_id in bl[g]]
    anchors = [
        c
        for c in clips
        if (c.role == "upper_anchor" and c.utt_id in upper)
        or (c.role == "lower_anchor" and c.utt_id in lower)
    ]

    rng = random.Random(rater)
    headphone = tuple(rng.sample(range(HP_POOL), HP_TRIALS))
    src = rng.choice(system)
    taken = {c.clip_id for c in clips}
    id_rng = random.Random(f"duplicate-{rater}")
    dup_id = _opaque_id(id_rng)
    while dup_id in taken:
        dup_id = _opaque_id(id_rng)
    duplicate = Clip(
        clip_id=dup_id,
        role=src.role,
        utt_id=src.utt_id,
        stimulus_path=src.stimulus_path,
        stimulus_sha256=src.stimulus_sha256,
        duration_s=src.duration_s,
        duplicate_of=src.clip_id,
    )
    items = [*system, *anchors, duplicate]
    rng.shuffle(items)
    order = tuple(c.clip_id for c in _no_consecutive(items))
    return RaterPlan(rater, rater, g, headphone, duplicate, order)


def stereo_check_audio(left_first: bool) -> np.ndarray:
    """(n, 2): a beep in one ear only, silence, then a beep in the other."""
    n, ramp = int(STEREO_TONE_S * STEREO_SR), int(0.02 * STEREO_SR)
    t = np.arange(n) / STEREO_SR
    tone = STEREO_AMPLITUDE * np.sin(2 * np.pi * STEREO_TONE_HZ * t)
    rise = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    tone[:ramp] *= rise
    tone[-ramp:] *= rise[::-1]
    silent = np.zeros(n)
    first = np.column_stack([tone, silent] if left_first else [silent, tone])
    gap = np.zeros((int(STEREO_GAP_S * STEREO_SR), 2))
    return np.concatenate([first, gap, first[:, ::-1]])


def dichotic(noise: np.ndarray, target: bool) -> np.ndarray:
    """(n, 2) interval: right is `noise`; left is `noise` with a 180 degree
    shift over 600 Hz +/- 6% when `target`, else identical (diotic)."""
    left = noise
    if target:
        spectrum = np.fft.rfft(noise)
        f = np.fft.rfftfreq(len(noise), 1 / HP_SR)
        band = (f >= HP_CENTRE_HZ * (1 - HP_HALF_BAND)) & (
            f <= HP_CENTRE_HZ * (1 + HP_HALF_BAND)
        )
        spectrum[band] *= -1
        left = np.fft.irfft(spectrum, len(noise))
    return np.column_stack([left, noise])


def huggins_pool() -> list[tuple[np.ndarray, int]]:
    """The 12 pre-generated trials: (stereo audio, target interval 0-2)."""
    rng = np.random.default_rng(HP_POOL_SEED)
    n, ramp = int(HP_INTERVAL_S * HP_SR), int(HP_RAMP_S * HP_SR)
    envelope = np.ones(n)
    rise = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    envelope[:ramp], envelope[-ramp:] = rise, rise[::-1]
    gap = np.zeros((int(HP_GAP_S * HP_SR), 2))

    pool = []
    for _ in range(HP_POOL):
        target = int(rng.integers(3))
        parts = []
        for k in range(3):
            noise = rng.standard_normal(n)
            noise *= 10 ** (HP_RMS_DBFS / 20) / np.sqrt(np.mean(noise**2))
            parts.append(dichotic(noise, target=k == target) * envelope[:, None])
            if k < 2:
                parts.append(gap)
        pool.append((np.concatenate(parts), target))
    return pool


MAX_REPLAYS = 2
TEXT_FIELDS = (
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
)
_PAGE = Path(__file__).with_name("panel_page.html")
_SESSION_RE = re.compile(
    r'<script id="session" type="application/json">(.*?)</script>', re.S
)


def _check_text(text: dict[str, Any], pilot: bool) -> None:
    missing = [k for k in TEXT_FIELDS if not text.get(k)]
    if missing:
        raise ValueError(f"rater text missing: {', '.join(missing)}")
    if set(text["scale"]) != {"1", "2", "3", "4", "5"}:
        raise ValueError("rater text scale must label exactly 1-5")
    if not pilot and not (text.get("approved") and text.get("approved_by")):
        raise ValueError(
            "rater text is not approved by a native speaker; build with "
            "pilot=True for a labelled pilot"
        )


def build_package(
    clips: list[Clip],
    plan: RaterPlan,
    text: dict[str, Any],
    out: str | Path,
    *,
    pilot: bool = False,
) -> Path:
    """Write one rater's offline package: index.html plus audio under opaque
    names. Nothing in it names a system, role or utterance (§6 blinding)."""
    _check_text(text, pilot)  # before any import or file is written
    import soundfile

    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    by_id = {c.clip_id: c for c in [*clips, plan.duplicate]}
    for cid in plan.trial_order:
        shutil.copyfile(by_id[cid].stimulus_path, out / "audio" / f"{cid}.wav")
    pool = huggins_pool()
    headphone = []
    for k, trial in enumerate(plan.headphone_trials):
        audio, target = pool[trial]
        soundfile.write(out / "audio" / f"hp{k + 1}.wav", audio, HP_SR, "PCM_16")
        headphone.append(
            {"trial": trial, "src": f"audio/hp{k + 1}.wav", "target": target}
        )

    # Own seeded stream, so the rater plan's draws are untouched.
    orders = random.Random(f"stereo-{plan.rater}")
    stereo_orders = [orders.random() < 0.5 for _ in range(STEREO_ATTEMPTS)]
    stereo_orders[:2] = [True, False]  # both orders always occur early
    orders.shuffle(stereo_orders)
    for name, left_first in (("stereo_a", True), ("stereo_b", False)):
        soundfile.write(
            out / "audio" / f"{name}.wav",
            stereo_check_audio(left_first),
            STEREO_SR,
            "PCM_16",
        )
    stereo = [
        {"src": f"audio/stereo_{'a' if lf else 'b'}.wav", "left_first": lf}
        for lf in stereo_orders
    ]

    session: dict[str, Any] = {
        "rater": plan.rater,
        "pilot": pilot,
        "max_replays": MAX_REPLAYS,
        "hp_pass": HP_PASS,
        "headphone": headphone,
        "stereo": stereo,
        "trials": [
            {"clip_id": cid, "src": f"audio/{cid}.wav"} for cid in plan.trial_order
        ],
        "text": {k: text[k] for k in TEXT_FIELDS},
    }
    session["session_id"] = hashlib.sha256(
        json.dumps(session, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    payload = json.dumps(session, ensure_ascii=False).replace("</", "<\\/")
    page = _PAGE.read_text(encoding="utf-8").replace("__SESSION__", payload)
    (out / "index.html").write_text(page, encoding="utf-8")
    return out


def read_page_session(page: str | Path) -> dict[str, Any]:
    """The session a built page carries."""
    match = _SESSION_RE.search(Path(page).read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{page} carries no session")
    return json.loads(match.group(1).replace("<\\/", "</"))
