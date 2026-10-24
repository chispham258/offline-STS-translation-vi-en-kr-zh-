"""Phase 4 benchmark runner.

D24: the Q6 limitation statement is composed HERE, around the evaluator, not
inside it. The approved spec requires every benchmark report to carry the
statement; D21 forbids changing evaluator output. Both hold only if the
statement is prepended outside `onevoice.eval`, which also keeps a Phase 1
report and a Phase 4 report byte-comparable in their metric sections.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from onevoice.config import load_config
from onevoice.eval.evaluator import ResultsTable, evaluate
from onevoice.eval.manifest import RunRecord, read_refs, read_run
from onevoice.eval.metrics.timing import streaming_cost_summary
from onevoice.eval.report import to_markdown
from onevoice.pipeline import build_components, run_utterance
from onevoice.provenance import run_identity
from onevoice.types import SystemClock

Q6_LIMITATION = """\
> **Known limitation (Q6).** `rtf` and `end_to_speech_ms` measure only work
> performed after the final speech frame, so neither captures per-frame work
> inside `push()`. `pipeline_rtf` spans the whole utterance and does capture it.
>
> Quote **`streaming_cost`**: it is the per-utterance difference between the two,
> which is what `rtf` under-reports. The `rtf` and `pipeline_rtf` percentiles are
> descriptive only and must not be subtracted from one another - when a detector
> fails to fire, that utterance has a `pipeline_rtf` but no `rtf`, so the two
> percentiles describe different populations.
>
> These are host-machine figures, not device predictions.
"""


def q6_statement(records: list[RunRecord]) -> str:
    """The Q6 caveat, with this run's streaming cost measured (D45).

    The statement used to assert "all models in this run are batch-shaped, so
    their work falls inside the measured window and these figures are sound".
    That was true when written and false from Task 11 onward, and nothing
    noticed, because it was a constant asserting a property of the run rather
    than a value computed from it. It is now computed, and it leads with the
    paired statistic so a reader is guided to the comparison that is defined.
    """
    cost = streaming_cost_summary(records)
    n = cost.detail["paired_n"]
    if not n:
        return Q6_LIMITATION + (
            ">\n> **Measured.** Streaming cost could not be measured for this "
            "run: no utterance carried both stamps.\n"
        )

    median = cost.detail["p50"]
    worst = cost.detail["max"]
    if worst < 1e-4:
        verdict = (
            f"Every component in this run is batch-shaped (paired_n={n}): "
            "`rtf` and `pipeline_rtf` coincide on every utterance, so nothing "
            "is hidden."
        )
    else:
        verdict = (
            f"`rtf` hides a median of {median:.4f} of audio duration per "
            f"utterance (worst {worst:.4f}), which is "
            f"{cost.detail['p50_pct_of_pipeline']:.1f}% of real processing cost "
            f"(worst {cost.detail['max_pct_of_pipeline']:.1f}%). "
            f"Paired over paired_n={n} utterances; "
            f"{cost.detail['unpairable']} could not be paired."
        )
    return Q6_LIMITATION + f">\n> **Measured (paired).** {verdict}\n"


def classification_banner(config) -> str:
    """State what this report's numbers may be used to claim (F3, D48)."""
    kind = config.experiment.classification
    rule = {
        "smoke": (
            "**Smoke benchmark.** Detects obvious failure only. These figures "
            "are NOT suitable for comparing models."
        ),
        "development": (
            "**Development benchmark.** Tracks regressions and guides "
            "engineering. Not sufficient for final model selection: a "
            "difference here is not a ranking unless an interval says so."
        ),
        "selection": (
            "**Selection benchmark.** Intended to support choosing between "
            "models, and therefore required to carry interval evidence for "
            "every comparison it makes."
        ),
    }[kind]
    return f"> {rule} See `docs/INTERPRETATION.md`.\n"


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything one benchmark run produced."""

    table: ResultsTable
    manifest_path: Path
    report_path: Path
    metadata_path: Path = Path()
    utterances_fresh: int = 0
    utterances_resumed: int = 0
    status: str = "complete"


def collect_provenance(components: dict[str, object]) -> dict[str, object]:
    """Ask every component what exactly it is.

    A component opts in by defining `provenance()`. Voice and checkpoint
    selection are part of the experimental configuration just as much as the
    model family is, and a benchmark number without them cannot be reproduced
    or fairly compared later.
    """
    recorded: dict[str, object] = {}
    for kind, component in components.items():
        describe = getattr(component, "provenance", None)
        if callable(describe):
            recorded[kind] = describe()
        else:
            recorded[kind] = {"component": type(component).__name__}
    return recorded


def render_report(
    table: ResultsTable,
    records: list[RunRecord] | None = None,
    config=None,
) -> str:
    """The evaluator's report, with the Q6 statement prepended (D24).

    `records` lets the Q6 statement measure this run's streaming cost rather
    than assert it is zero (D45). Optional so existing callers keep working;
    without it the statement falls back to the unquantified text.
    """
    q6 = q6_statement(records) if records is not None else Q6_LIMITATION
    banner = classification_banner(config) if config is not None else ""
    return f"{banner}{q6}\n{to_markdown(table)}"


def _host_info() -> dict[str, str]:
    """Machine facts that explain latency differences later.

    Not correctness-relevant, but a latency number compared across two runs on
    different hardware is not a comparison, and by then nobody remembers which
    machine produced which figure.
    """
    import platform

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance must never break a run
            info["cpu"] = "unknown"
    return info


def _completed_utterance_ids(manifest_path: Path) -> set[str]:
    """utt_ids already present in a partial manifest."""
    if not manifest_path.is_file():
        return set()
    done = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["utt_id"])
    return done


def run_benchmark(
    config_path: str | Path,
    refs_path: str | Path,
    out_dir: str | Path,
    experiment: str = "",
    resume: bool = True,
) -> BenchmarkResult:
    """Run a config over a reference manifest and score it.

    **Checkpointed.** Each utterance is appended to the manifest the moment it
    completes, so an interrupted run resumes from the last finished utterance
    rather than starting over. A full end-to-end configuration over a real
    corpus takes tens of minutes; losing that to an interruption and silently
    re-running it is worse than the runtime itself.

    `resume=False` starts clean, which is what you want after changing a model
    or a config - resuming across a configuration change would silently mix two
    experiments into one manifest.
    """
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run.jsonl"

    config = load_config(config_path)
    # Archive the config beside the run. A hash detects that the file changed;
    # only a copy lets the experiment be re-run afterwards (F6).
    shutil.copyfile(config_path, out / "config.snapshot.yaml")
    components = build_components(config)
    refs = read_refs(refs_path)
    fixtures_root = Path(refs_path).parent

    if not resume and manifest_path.exists():
        manifest_path.unlink()
    already_done = _completed_utterance_ids(manifest_path)

    started_at = datetime.now(UTC)
    clock = SystemClock()
    fresh = 0

    with manifest_path.open("a", encoding="utf-8") as sink:
        for ref in refs:
            if ref.utt_id in already_done:
                continue
            record = run_utterance(ref, components, clock, out, fixtures_root)
            sink.write(record.model_dump_json() + "\n")
            sink.flush()  # survive an interruption mid-run
            fresh += 1

    finished_at = datetime.now(UTC)

    metadata_path = out / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "config": str(config_path),
                "refs": str(refs_path),
                "components": collect_provenance(components),
                "host": _host_info(),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": round(
                    (finished_at - started_at).total_seconds(), 3
                ),
                **run_identity(config_path, refs_path),
                "config_snapshot": "config.snapshot.yaml",
                "utterances_total": len(refs),
                "utterances_fresh": fresh,
                "utterances_resumed": len(already_done),
                "status": "resumed" if already_done else "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    table = evaluate(manifest_path, refs_path, experiment=experiment)
    report_path = out / "report.md"
    # Read back rather than accumulate: records are appended and flushed one at
    # a time so an interrupted run resumes (D40), so the manifest - not memory -
    # is the complete set, including anything a resumed run did not re-execute.
    report_path.write_text(
        render_report(table, read_run(manifest_path), config), encoding="utf-8"
    )

    return BenchmarkResult(
        table=table,
        manifest_path=manifest_path,
        report_path=report_path,
        metadata_path=metadata_path,
        utterances_fresh=fresh,
        utterances_resumed=len(already_done),
        status="resumed" if already_done else "complete",
    )
