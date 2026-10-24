"""OpenCode agent frontmatter: lint rules and model pinning.

Rules are the ones found by bisecting OpenCode 2.0.32 (2026-10-22): an agent that
breaks them is silently 'not found'. The linter is also the safety net for a
newly created agent - see `test_local_agent_files_have_no_errors`.
"""

import os
from pathlib import Path

import pytest

from scripts.agent_lint import lint
from scripts.agent_pin_models import MODEL_PINS, edit_frontmatter, fence, normalize

LOCAL_AGENTS = Path(
    os.environ.get("OPENCODE_AGENTS_DIR", "~/.config/opencode/agents")
).expanduser()


def write_agent(tmp_path: Path, *lines: str, name: str = "helper") -> Path:
    path = tmp_path / f"{name}.md"
    path.write_text("---\n" + "\n".join(lines) + "\n---\n\nBody.\n", encoding="utf-8")
    return path


def test_valid_frontmatter_has_no_errors(tmp_path):
    path = write_agent(
        tmp_path,
        "description: Read-only explorer",
        "mode: subagent",
        "model: opencode/x-free",
        "color: \"#22c55e\"",
        "permission:",
        "  edit: deny",
    )
    assert lint(path) == ([], [])


def test_missing_description_is_an_error(tmp_path):
    errs, _ = lint(write_agent(tmp_path, "mode: subagent"))
    assert "missing description" in errs


def test_name_must_match_the_file_stem(tmp_path):
    path = write_agent(tmp_path, "description: X", "name: other", name="helper")
    errs, _ = lint(path)
    assert any("!= file name" in err for err in errs)


def test_mode_must_be_one_of_three_values(tmp_path):
    errs, _ = lint(write_agent(tmp_path, "description: X", "mode: worker"))
    assert any("mode must be" in err for err in errs)


def test_claude_code_alias_is_not_a_model_id(tmp_path):
    errs, _ = lint(write_agent(tmp_path, "description: X", "model: sonnet"))
    assert any("provider/model id" in err for err in errs)


def test_model_must_exist_in_the_registry(tmp_path):
    path = write_agent(tmp_path, "description: X", "model: opencode/gone-free")
    errs, _ = lint(path, models={"opencode/known-free"})
    assert any("not in `opencode models`" in err for err in errs)
    assert lint(path, models={"opencode/gone-free"})[0] == []


def test_named_color_is_an_error_but_hex_passes(tmp_path):
    errs, _ = lint(write_agent(tmp_path, "description: X", "color: green"))
    assert any("must be" in err for err in errs)
    ok, _ = lint(write_agent(tmp_path, "description: X", "color: \"#aabbcc\""))
    assert ok == []


def test_comma_string_tools_is_an_error(tmp_path):
    errs, _ = lint(write_agent(tmp_path, "description: X", "tools: Read, Grep"))
    assert any("comma string" in err for err in errs)


def test_missing_frontmatter_is_an_error(tmp_path):
    path = tmp_path / "helper.md"
    path.write_text("no frontmatter here\n", encoding="utf-8")
    assert lint(path)[0] == ["no frontmatter"]


def test_unquoted_description_with_colon_is_reported(tmp_path):
    errs, _ = lint(write_agent(tmp_path, "description: Review this: carefully"))
    assert any("YAML does not parse" in err for err in errs)


def test_subagent_without_permission_only_warns(tmp_path):
    errs, warns = lint(write_agent(tmp_path, "description: X", "mode: subagent"))
    assert errs == []
    assert any("no permission" in warn for warn in warns)


@pytest.mark.skipif(not LOCAL_AGENTS.is_dir(), reason="no local agents directory")
def test_local_agent_files_have_no_errors():
    failures = {}
    for path in sorted(LOCAL_AGENTS.glob("*.md")):
        errs, _ = lint(path)
        if errs:
            failures[path.name] = errs
    assert not failures, failures


def test_every_pin_is_a_reachable_provider_model_id():
    for name, model in MODEL_PINS.items():
        assert "/" in model, name
        assert model.split("/")[0] == "opencode", name


def test_fence_requires_a_frontmatter_block():
    assert fence(["# Body"]) is None
    assert fence(["---", "a: b", "---", ""]) == (0, 2)


def test_pin_inserts_model_when_absent():
    lines = ["---", "description: Helper", "---", ""]
    out, notes = edit_frontmatter(lines, "opencode/x-free", add_mode=False)
    assert out is not None
    assert "model: opencode/x-free" in out
    assert notes


def test_pin_replaces_an_existing_model():
    lines = ["---", "model: old/model", "---", ""]
    out, _ = edit_frontmatter(lines, "opencode/x-free", add_mode=False)
    assert out is not None
    assert "model: opencode/x-free" in out
    assert "model: old/model" not in out


def test_pin_is_idempotent():
    lines = ["---", "description: Helper", "mode: subagent", "---", ""]
    first, _ = edit_frontmatter(lines, "opencode/x-free", add_mode=False)
    second, notes = edit_frontmatter(first, "opencode/x-free", add_mode=False)
    assert second == first
    assert notes == []


def test_add_mode_gives_a_read_only_helper_a_permission_block():
    lines = ["---", "description: Helper", "tools: Read, Grep", "---", ""]
    out, _ = edit_frontmatter(lines, "opencode/x-free", True, normalize_it=True)
    assert out is not None
    text = "\n".join(out)
    assert "mode: subagent" in text
    assert "permission:" in text
    assert "  edit: deny" in text
    assert "  bash: deny" in text
    assert "tools: Read, Grep" not in text


def test_write_tool_helper_keeps_edit_permission():
    lines = ["---", "description: Helper", "tools: Read, Write, Bash", "---", ""]
    out, _ = edit_frontmatter(lines, "opencode/x-free", True)
    assert out is not None
    text = "\n".join(out)
    assert "  edit: deny" not in text
    assert "  bash: deny" not in text


def test_normalize_converts_a_named_color():
    lines = ["---", "color: green", "---", ""]
    notes = []
    assert 'color: "#22c55e"' in normalize(lines, notes)
    assert notes == ["color green -> #22c55e"]


def test_normalize_quotes_a_description_with_a_colon():
    lines = ["---", "description: Review this: carefully", "---", ""]
    out = normalize(lines, [])
    assert out[1] == 'description: "Review this: carefully"'
