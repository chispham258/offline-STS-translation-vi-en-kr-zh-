#!/usr/bin/env python3
"""Pin `model:` (and optionally `mode: subagent`) in OpenCode agent frontmatter.

Default is a DRY RUN: it only prints what would change.
--apply writes; a timestamped backup of the agents directory is made first.

    uv run python scripts/agent_pin_models.py                    # dry run
    uv run python scripts/agent_pin_models.py --apply            # backup + write
    uv run python scripts/agent_pin_models.py --add-mode         # helpers -> subagents
    uv run python scripts/agent_pin_models.py --add-mode --apply
    uv run python scripts/agent_pin_models.py --dir PATH         # another agents dir

IDs are per profile: only the free profile this project can actually reach is
kept. `opencode-go/*` was dropped 2026-10-22 - that provider is not installed.
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

DS = "deepseek-v4.1-flash"
GLM = "glm-5.3-flash"
MM = "minimax-m3.1-flash"

# OpenCode agent name -> provider/model. Keep in step with AGENTS.md section 2.
MODEL_PINS = {
    "superpowers": f"opencode/{DS}-free",
    "superpowers-implementer": f"opencode/{DS}-free",
    "superpowers-spec-writer": f"opencode/{GLM}-free",
    "superpowers-plan-writer": f"opencode/{MM}-free",
    "superpowers-plan-writer-alt1": f"opencode/{DS}-free",
    # logged by NVIDIA: plans/code only, never real audio or transcripts
    "superpowers-plan-writer-alt2": "opencode/nemotron-3-ultra-free",
    "superpowers-code-reviewer": f"opencode/{GLM}-free",
    "code-reviewer": f"opencode/{GLM}-free",
    "atomic-reviewer": f"opencode/{GLM}-free",
    "code-simplifier": f"opencode/{GLM}-free",
    "silent-failure-hunter": f"opencode/{GLM}-free",
    "comment-analyzer": f"opencode/{GLM}-free",
    "pr-test-analyzer": f"opencode/{GLM}-free",
    "type-design-analyzer": f"opencode/{GLM}-free",
    "skill-reviewer": f"opencode/{GLM}-free",
    "plugin-validator": f"opencode/{GLM}-free",
    "code-architect": f"opencode/{GLM}-free",
    "code-explorer": f"opencode/{MM}-free",
    "atomic-explorer": f"opencode/{MM}-free",
    "agent-creator": f"opencode/{MM}-free",
    "math-proof-worker": f"opencode/{GLM}-free",
    "math-proof-judge": f"opencode/{DS}-free",
}

# Claude-Code-style helper agents carry no `mode:`, so OpenCode V2 registers them
# as PRIMARY and a subagent call returns "Unknown agent". --add-mode fixes that.
HELPERS = {
    "code-reviewer",
    "atomic-reviewer",
    "code-simplifier",
    "silent-failure-hunter",
    "comment-analyzer",
    "pr-test-analyzer",
    "type-design-analyzer",
    "skill-reviewer",
    "plugin-validator",
    "code-architect",
    "code-explorer",
    "atomic-explorer",
    "agent-creator",
    "math-proof-worker",
    "math-proof-judge",
}
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
MODEL_RE = re.compile(r"^model:\s*(.*?)\s*$")

COLORS = {
    "green": "#22c55e",
    "yellow": "#eab308",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "cyan": "#06b6d4",
    "magenta": "#d946ef",
    "purple": "#8b5cf6",
    "orange": "#f97316",
    "pink": "#ec4899",
}


def fence(lines: list[str]) -> tuple[int, int] | None:
    """Return (start, end) indexes of the frontmatter block, or None."""
    if not lines or lines[0].strip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return 0, index
    return None


def has_key(lines: list[str], start: int, end: int, key: str) -> bool:
    pattern = re.compile(rf"^{re.escape(key)}:")
    return any(pattern.match(lines[i]) for i in range(start + 1, end))


def parse_tools(lines: list[str], start: int, end: int) -> list[str] | None:
    """Legacy comma-separated `tools:` line -> list, or None if absent/empty."""
    for i in range(start + 1, end):
        match = re.match(r"^tools:\s*(.*)$", lines[i])
        if match:
            value = match.group(1).strip()
            if not value:
                return None
            return [item.strip() for item in value.split(",") if item.strip()]
    return None


def normalize(lines: list[str], notes: list[str]) -> list[str]:
    """Fixes that make OpenCode 2.0.x register Claude-Code-style agents.

    Verified by bisect, 2026-10-22: a named `color:` and a comma-string `tools:`
    make the agent 'not found'; an unquoted ': ' breaks YAML.
    """
    start, end = fence(lines)
    i = start + 1
    while i < end:
        line = lines[i]
        color = re.match(r"^color:\s*(.*?)\s*$", line)
        if color:
            value = color.group(1).strip("\"'")
            if not value.startswith("#"):
                if value.lower() in COLORS:
                    lines[i] = f'color: "{COLORS[value.lower()]}"'
                    notes.append(f"color {value} -> {COLORS[value.lower()]}")
                else:
                    del lines[i]
                    end -= 1
                    i -= 1
                    notes.append(f"drop color {value}")
        elif re.match(r"^tools:", line):
            del lines[i]
            end -= 1
            i -= 1
            notes.append("drop legacy tools:")
        else:
            description = re.match(r"^description:\s+(\S.*)$", line)
            if (
                description
                and description.group(1)[0] not in "|>\"'"
                and (": " in description.group(1) or " #" in description.group(1))
            ):
                quoted = json.dumps(description.group(1), ensure_ascii=False)
                lines[i] = "description: " + quoted
                notes.append("quote description")
        i += 1
    return lines


def edit_frontmatter(
    lines: list[str], new_model: str, add_mode: bool, normalize_it: bool = False
) -> tuple[list[str] | None, list[str]]:
    """Pin `model:` and optionally add `mode:`/`permission:`. Returns (lines, notes)."""
    bounds = fence(lines)
    if bounds is None:
        return None, ["no frontmatter"]
    start, end = bounds
    notes: list[str] = []

    pinned = False
    for i in range(start + 1, end):
        match = MODEL_RE.match(lines[i])
        if match:
            if match.group(1) != new_model:
                notes.append(f"{match.group(1)}  ->  {new_model}")
                lines[i] = f"model: {new_model}"
            pinned = True
            break
    if not pinned:
        lines.insert(end, f"model: {new_model}")
        end += 1
        notes.append(f"(none, inherits parent)  ->  {new_model}")

    has_mode = has_key(lines, start, end, "mode")
    legacy_tools = parse_tools(lines, start, end)
    wants_permission = legacy_tools is not None and not has_key(
        lines, start, end, "permission"
    )
    if add_mode and (not has_mode or wants_permission):
        insert = [] if has_mode else ["mode: subagent"]
        extra = ""
        if wants_permission and not has_key(lines, start, end, "permissions"):
            perms = []
            if not (WRITE_TOOLS & set(legacy_tools)):
                perms.append("  edit: deny")
            if "Bash" not in legacy_tools:
                perms.append("  bash: deny")
            if perms:
                insert += ["permission:"] + perms
                joined = ", ".join(item.strip() for item in perms)
                extra = f" + read-only permission ({joined})"
        for offset, line in enumerate(insert):
            lines.insert(end + offset, line)
        if insert:
            adds_mode = insert[0].startswith("mode")
            kind = "add mode: subagent" if adds_mode else "add permission"
            notes.append(kind + extra)
    if normalize_it:
        normalize(lines, notes)
    return lines, notes


def current_model(path: Path) -> str | None:
    lines = path.read_text(encoding="utf-8").split("\n")
    bounds = fence(lines)
    if bounds is None:
        return None
    for i in range(bounds[0] + 1, bounds[1]):
        match = MODEL_RE.match(lines[i])
        if match:
            return match.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="~/.config/opencode/agents")
    parser.add_argument(
        "--add-mode",
        action="store_true",
        help="add `mode: subagent` (and read-only permissions) to helper agents",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.dir).expanduser()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    print(f"add-mode={args.add_mode}  dir={root}\n")

    changes = []
    missing = []
    for stem, model in MODEL_PINS.items():
        path = root / f"{stem}.md"
        if not path.exists():
            missing.append(stem)
            continue
        lines = path.read_text(encoding="utf-8").split("\n")
        add_mode = args.add_mode and stem in HELPERS
        new_lines, notes = edit_frontmatter(lines, model, add_mode, stem in HELPERS)
        if new_lines is None:
            print(f"SKIP   {stem:32s} {notes[0]}")
        elif not notes:
            print(f"OK     {stem:32s} {model}")
        else:
            tag = "  [symlink -> vendor file]" if path.is_symlink() else ""
            print(f"CHANGE {stem:32s} {'; '.join(notes)}{tag}")
            changes.append((path, "\n".join(new_lines)))

    stray = [
        f"{path.stem} ({model})"
        for path in sorted(root.glob("*.md"))
        if path.stem not in MODEL_PINS
        and (model := current_model(path))
        and model.startswith("opencode-go/")
    ]
    if stray:
        print("\nWARN still pinned to opencode-go: " + ", ".join(stray))
    if missing:
        joined = ", ".join(missing)
        print(f"\nNot found in this directory (maybe provided elsewhere): {joined}")

    if not args.apply:
        print(f"\nDRY RUN: {len(changes)} file(s) would change. --apply to write.")
        return 0
    if not changes:
        print("\nNothing to change.")
        return 0

    backup = root.parent / f"agents.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copytree(root, backup)
    for path, new_text in changes:
        path.write_text(new_text, encoding="utf-8")
    print(f"\nApplied {len(changes)} change(s). Backup: {backup}")
    linked = [path.name for path, _ in changes if path.is_symlink()]
    if linked:
        print(
            "NOTE: these are symlinks, so the edit landed in the plugin's vendor "
            "files; a plugin update may overwrite it. Re-run after updating: "
            + ", ".join(linked)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
