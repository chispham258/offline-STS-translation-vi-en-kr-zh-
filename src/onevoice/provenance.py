"""The provenance contract every production component satisfies (D46).

Spec: `docs/PROVENANCE.md`. This module is the executable half — one schema, one
validator, enumerated over the registry by `tests/test_provenance.py`, so
provenance evolves once rather than field-by-field as each wrapper is touched.

Deliberately dependency-free and importable without any model extra: the
validator has to run in CI, where no wrapper can be constructed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Component identity — *what implementation produced this result?*
IDENTITY_FIELDS = (
    "component",
    "implementation",
    "model_id",
    "revision",
    "params_hash",
)

#: Invocation identity — *what was that implementation asked to do?* Recorded
#: as a nested dict rather than flat keys so the two questions stay visibly
#: separate and a new request parameter cannot be mistaken for an identity one.
#: Required and non-empty: every production component in this framework is
#: asked for something (a language, a voice, a threshold), and a wrapper that
#: believes otherwise has usually forgotten a parameter rather than found an
#: exception.
INVOCATION_FIELD = "invocation"

#: Backwards-compatible alias. Identity fields were the whole schema before the
#: invocation split; the name is kept because tests parametrise over it.
REQUIRED_FIELDS = IDENTITY_FIELDS

#: At least one of these must be present and non-empty — the anchor rule. A Hub
#: model's commit IS its immutable identity, so re-hashing gigabytes to restate
#: it buys nothing; a file-based model has no commit, so its weight hash is the
#: only anchor available. The requirement is identity, not a fixed pair.
IMMUTABLE_ANCHORS = ("commit_hash", "weight_hash")

#: Never a valid value for a required field. "unknown" is the interesting one:
#: it is what a lazy implementation returns when a lookup fails, and it would
#: otherwise satisfy a mere presence check while recording nothing.
PLACEHOLDERS = {"", "unknown", "none", "null", "n/a", "todo"}


class ProvenanceError(ValueError):
    """A component's provenance does not satisfy the schema."""


def params_hash(params: dict[str, Any]) -> str:
    """Stable sha256 over a component's resolved constructor arguments.

    Sorted keys and `default=str` so the digest is reproducible across runs and
    never raises on a value that happens not to be JSON-serialisable — a
    provenance helper must not be able to break a benchmark.
    """
    encoded = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_provenance(record: dict[str, Any], *, label: str = "") -> None:
    """Raise `ProvenanceError` unless `record` satisfies the schema.

    Checks values, not just keys: a field present but set to "unknown" records
    nothing while passing any presence test, which is precisely how provenance
    rots without anyone noticing.
    """
    where = f"{label}: " if label else ""

    missing = [f for f in IDENTITY_FIELDS if f not in record]
    if missing:
        raise ProvenanceError(
            f"{where}missing required provenance fields {missing}. "
            "See docs/PROVENANCE.md."
        )

    empty = [
        f
        for f in IDENTITY_FIELDS
        if str(record[f]).strip().lower() in PLACEHOLDERS
    ]
    if empty:
        raise ProvenanceError(
            f"{where}required provenance fields {empty} are placeholders. "
            "A field recording 'unknown' satisfies a presence check and "
            "reproduces nothing."
        )

    invocation = record.get(INVOCATION_FIELD)
    if not isinstance(invocation, dict) or not invocation:
        raise ProvenanceError(
            f"{where}missing a non-empty '{INVOCATION_FIELD}' dict. Identity "
            "says what implementation ran; invocation says what it was asked "
            "to do - a source language, a voice, a decoding mode. Both are "
            "needed to reproduce a benchmark. See docs/PROVENANCE.md."
        )

    anchors = [
        a
        for a in IMMUTABLE_ANCHORS
        if a in record and str(record[a]).strip().lower() not in PLACEHOLDERS
    ]
    if not anchors:
        raise ProvenanceError(
            f"{where}no immutable anchor: exactly one of "
            f"{list(IMMUTABLE_ANCHORS)} must be present and real. A mutable "
            "revision alone cannot identify what produced a number."
        )

    if str(record["revision"]) == str(record.get("commit_hash")):
        raise ProvenanceError(
            f"{where}revision equals commit_hash. These answer different "
            "questions - what was requested versus what it resolved to - and "
            "collapsing them discards the first. Use 'default' when no "
            "revision was pinned."
        )


def hub_commit(model: Any) -> str:
    """The immutable commit a `from_pretrained` model resolved to.

    transformers records it on the config as `_commit_hash`. Returns the literal
    "unknown" when absent, which `validate_provenance` rejects as an anchor —
    failing the contract is the right outcome, because a Hub model whose commit
    cannot be determined cannot be reproduced.
    """
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    return str(commit) if commit else "unknown"


def describe_hub_model(
    *,
    component: str,
    implementation: type,
    model_id: str,
    model: Any,
    params: dict[str, Any],
    invocation: dict[str, Any],
    revision: str = "default",
    tokenizer_revision: str | None = None,
) -> dict[str, Any]:
    """Build a conforming record for a Hugging Face checkpoint.

    Shared so eight wrappers cannot drift into eight vocabularies, which is
    exactly what happened before this schema existed: `MMSTTS` collapsed
    `revision` into `commit_hash` while `PiperTTS` recorded neither.

    `revision` defaults to "default" — the literal string, meaning "nothing was
    pinned". It is never set to the resolved commit; that answers a different
    question and the validator rejects the collapse.
    """
    record: dict[str, Any] = {
        "component": component,
        "implementation": f"{implementation.__module__}.{implementation.__qualname__}",
        "model_id": model_id,
        "revision": revision,
        "commit_hash": hub_commit(model),
        "params_hash": params_hash(params),
        INVOCATION_FIELD: invocation,
    }
    if tokenizer_revision is not None:
        record["tokenizer_revision"] = tokenizer_revision
    return record


# --- run identity (F6) -------------------------------------------------------
#
# A different layer from component provenance and deliberately not merged with
# it. `params_hash` identifies how one component was constructed; `config_hash`
# identifies the whole experiment. A component must not duplicate run-level
# configuration, and the run must not stand in for component provenance.

RUN_IDENTITY_FIELDS = (
    "config_path",
    "config_hash",
    "refs_path",
    "refs_hash",
    "framework_commit",
    "framework_dirty",
)


def file_hash(path: str | Any) -> str:
    """sha256 of a file's bytes, or "missing" when it is not there."""
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.is_file():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return None


def framework_identity() -> dict[str, Any]:
    """Which code ran, and whether that answer can be trusted.

    `framework_dirty` exists because a commit hash recorded from a working tree
    with uncommitted changes is a lie: it names code that is not what ran. A run
    from a dirty tree is still worth recording — it just must not claim to be
    reproducible from that commit alone.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "framework_commit": commit or "unknown",
        # None (git failed) is not clean. Unknown provenance is never "fine".
        "framework_dirty": True if status is None else bool(status.strip()),
    }


def run_identity(config_path: str | Any, refs_path: str | Any) -> dict[str, Any]:
    """Everything needed to identify a benchmark run's inputs (F6).

    Records content hashes rather than only paths: a filename identifies a run
    only until someone edits the file, after which the record silently describes
    an experiment that never happened.
    """
    return {
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        "refs_path": str(refs_path),
        "refs_hash": file_hash(refs_path),
        **framework_identity(),
    }


def validate_run_identity(record: dict[str, Any]) -> None:
    """Raise `ProvenanceError` unless a run record identifies its inputs."""
    missing = [f for f in RUN_IDENTITY_FIELDS if f not in record]
    if missing:
        raise ProvenanceError(
            f"run metadata is missing {missing}. See docs/PROVENANCE.md."
        )
    for field in ("config_hash", "refs_hash"):
        if str(record[field]).strip().lower() in PLACEHOLDERS | {"missing"}:
            raise ProvenanceError(
                f"run metadata has no real {field}: a path alone identifies a "
                "run only until someone edits the file."
            )
    if not isinstance(record["framework_dirty"], bool):
        raise ProvenanceError(
            "framework_dirty must be a bool. A commit recorded from a dirty "
            "tree names code that is not what ran, and that must be explicit."
        )
