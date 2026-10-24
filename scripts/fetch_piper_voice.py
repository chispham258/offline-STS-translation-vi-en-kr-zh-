"""Download a Piper voice into the local voices directory.

Usage: uv run python scripts/fetch_piper_voice.py [voice-name]

Voices live in rhasspy/piper-voices under <lang>/<locale>/<name>/<quality>/.
The default is the Vietnamese baseline (D23).
"""

from __future__ import annotations

import sys
import urllib.request

from onevoice.components.piper_tts import DEFAULT_VOICE, voice_paths

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

#: Repository sub-path per voice. Add an entry when a new voice is adopted.
VOICE_PATHS = {
    # Vietnamese — the D23 baseline plus its two alternates.
    "vi_VN-vais1000-medium": "vi/vi_VN/vais1000/medium",
    "vi_VN-25hours_single-low": "vi/vi_VN/25hours_single/low",
    "vi_VN-vivos-x_low": "vi/vi_VN/vivos/x_low",
    # Korean and Chinese — Piper covers every competition language, which is
    # why MeloTTS is not needed (D36).
    "ko_KR-kss-medium": "ko/ko_KR/kss/medium",
    "zh_CN-huayan-medium": "zh/zh_CN/huayan/medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium",
}


def fetch(voice: str) -> int:
    try:
        repo_dir = VOICE_PATHS[voice]
    except KeyError:
        known = ", ".join(sorted(VOICE_PATHS))
        print(f"unknown voice {voice!r}; known: {known}", file=sys.stderr)
        return 2

    onnx_path, config_path = voice_paths(voice)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = (
        (onnx_path, f"{voice}.onnx"),
        (config_path, f"{voice}.onnx.json"),
    )
    for target, name in wanted:
        if target.is_file():
            print(f"have {target.name}")
            continue
        url = f"{BASE}/{repo_dir}/{name}"
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https host
        print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


def main(argv: list[str]) -> int:
    voice = argv[1] if len(argv) > 1 else DEFAULT_VOICE
    return fetch(voice)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
