#!/usr/bin/env python3
"""Lint OpenCode agent files. Exit 1 on any ERROR.

    uv run python scripts/agent_lint.py                  # ~/.config/opencode/agents
    uv run python scripts/agent_lint.py --dir .opencode/agents
    uv run python scripts/agent_lint.py --models         # also check `opencode models`

Rules come from bisecting OpenCode 2.0.32 (2026-10-22): an agent that breaks them
is silently 'not found', with no error message.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML missing: run with `uv run python scripts/agent_lint.py` "
        "or `pip install pyyaml`"
    )

BAD_MODELS = {"opus", "sonnet", "haiku", "inherit"}
VALID_MODES = ("primary", "subagent", "all")


def known_models() -> set[str]:
    """Return the provider/model ids that `opencode models` reports."""
    result = subprocess.run(
        ["opencode", "models"], capture_output=True, text=True, check=False
    )
    return {line.strip() for line in result.stdout.splitlines() if "/" in line}


def lint(path: Path, models: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one agent file."""
    errs: list[str] = []
    warns: list[str] = []
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return ["no frontmatter"], warns
    try:
        meta = yaml.safe_load(text.split("---", 2)[1]) or {}
    except yaml.YAMLError as exc:
        first = str(exc).splitlines()[0]
        return [f"YAML does not parse: {first} (quote the description)"], warns
    if not meta.get("description"):
        errs.append("missing description")
    if meta.get("name") and meta["name"] != path.stem:
        errs.append(f"name {meta['name']!r} != file name {path.stem!r}")
    mode = meta.get("mode")
    if mode is None:
        warns.append("no mode: set it explicitly (primary|subagent|all)")
    elif mode not in VALID_MODES:
        errs.append(f"mode must be primary|subagent|all, got {mode!r}")
    model = meta.get("model")
    if model is None:
        warns.append("no model: inherits the parent's model")
    elif model in BAD_MODELS or "/" not in str(model):
        errs.append(f"model {model!r} is not a provider/model id")
    elif models is not None and model not in models:
        errs.append(f"model {model} not in `opencode models`")
    color = meta.get("color")
    if color is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)):
        errs.append(
            f"color {color!r} must be \"#rrggbb\" "
            "(named colors make the agent unregistered)"
        )
    if isinstance(meta.get("tools"), str):
        errs.append(
            "tools: as a comma string makes the agent unregistered; use permission:"
        )
    if mode == "subagent" and not (
        "permission" in meta or "permissions" in meta
    ):
        warns.append(
            "no permission: block (state edit/bash explicitly for read-only agents)"
        )
    return errs, warns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="~/.config/opencode/agents")
    parser.add_argument("--models", action="store_true")
    args = parser.parse_args()
    root = Path(args.dir).expanduser()
    models = known_models() if args.models else None
    bad = 0
    for path in sorted(root.glob("*.md")):
        errs, warns = lint(path, models)
        tag = "ERROR" if errs else ("warn " if warns else "ok   ")
        print(f"{tag} {path.stem:32s} " + "; ".join(errs + warns))
        bad += bool(errs)
    print(f"\n{bad} file(s) with errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
