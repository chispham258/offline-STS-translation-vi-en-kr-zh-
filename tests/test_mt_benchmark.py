"""MT component benchmark: gold (src_text, ref_text) pairs, no pipeline.

Task 3 established that pipeline BLEU cannot measure translation quality
while EchoASR is a dummy (D18). This harness bypasses ASR/VAD/TTS entirely:
push() each reference's src_text through the MT component alone, score the
hypothesis against ref_text with the evaluator's own bleu()/chrf() (D21).
"""

import json
import subprocess
import sys

import pytest

import onevoice.components.dummy  # noqa: F401  - registers IdentityMT etc.
from onevoice.config import load_config
from onevoice.eval.evaluator import ResultsTable
from onevoice.eval.metrics.base import MetricResult
from onevoice.mt_benchmark import MT_ONLY_CAVEAT, run_mt_benchmark
from onevoice.types import FakeClock

REFS = "fixtures/mini_vi_en/refs.jsonl"
IDENTITY_CONFIG = "configs/dummy_vi_en.yaml"


def _mt_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


requires_mt = pytest.mark.skipif(not _mt_available(), reason="mt extra not installed")


# --- the harness works end to end on IdentityMT, no extras needed -----------


def test_scores_identity_mt_end_to_end(tmp_path):
    result = run_mt_benchmark(IDENTITY_CONFIG, REFS, tmp_path, experiment="smoke")
    assert result.table.counts["utterances"] == 3
    assert result.hyps_path.is_file()
    assert result.report_path.is_file()
    assert len(result.hypotheses) == 3


def test_identity_mt_scores_low_but_deterministic(tmp_path):
    # IdentityMT returns vi src_text unchanged; scored against en ref_text
    # this must be low (it is not a translation) but stable across runs.
    first = run_mt_benchmark(IDENTITY_CONFIG, REFS, tmp_path / "a", experiment="s")
    second = run_mt_benchmark(IDENTITY_CONFIG, REFS, tmp_path / "b", experiment="s")
    assert first.table.metrics["bleu"].value == second.table.metrics["bleu"].value
    assert first.table.metrics["bleu"].value < 10.0


def test_no_asr_vad_or_tts_is_used():
    # This harness must not import the pipeline. Checking the live import
    # graph (rather than grepping source text) is what actually proves it -
    # a docstring mentioning "onevoice.pipeline" in prose would otherwise
    # trip a naive text search without meaning anything.
    check = (
        "import sys\n"
        "import onevoice.mt_benchmark\n"
        "print('onevoice.pipeline' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


# --- MT_ONLY_CAVEAT: scope must be unmistakable on every report ------------


def test_caveat_appears_in_every_report(tmp_path):
    result = run_mt_benchmark(IDENTITY_CONFIG, REFS, tmp_path, experiment="smoke")
    assert MT_ONLY_CAVEAT.strip() in result.report_path.read_text()


def test_caveat_names_ceiling_and_absence_of_asr():
    assert "ceiling" in MT_ONLY_CAVEAT.lower()
    assert "no asr" in MT_ONLY_CAVEAT.lower()


def test_caveat_warns_the_fixture_is_smoke_scale_only():
    # Three utterances is a smoke-level baseline, not a corpus result (Task 8).
    assert "three" in MT_ONLY_CAVEAT.lower()


# --- D28: raw hypotheses are archived, one line per utterance --------------


def test_archives_one_hypothesis_line_per_utterance_with_required_fields(tmp_path):
    result = run_mt_benchmark(IDENTITY_CONFIG, REFS, tmp_path, experiment="smoke")
    lines = result.hyps_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    for key in ("utt_id", "backend", "src_text", "ref_text", "hyp_text", "latency_ms"):
        assert key in row
    assert row["backend"] == "identity"


# --- latency: separate section, per utterance, p50/p95/max -----------------


def test_latency_is_exact_with_a_fake_clock_and_never_advances(tmp_path):
    # A clock that never ticks makes every latency exactly 0.0 - deterministic,
    # not merely "> 0" - proving the timing wiring rather than hoping it's fast.
    clock = FakeClock()
    result = run_mt_benchmark(
        IDENTITY_CONFIG, REFS, tmp_path, experiment="smoke", clock=clock
    )
    assert all(h.latency_ms == 0.0 for h in result.hypotheses)


def test_report_reports_latency_in_its_own_section_not_interleaved(tmp_path):
    result = run_mt_benchmark(IDENTITY_CONFIG, REFS, tmp_path, experiment="smoke")
    report = result.report_path.read_text()
    # Section headings, not a bare substring search: the caveat's own prose
    # mentions "latency" too, and that must not count as the section.
    quality_pos = report.lower().index("## translation quality")
    latency_pos = report.lower().index("## latency")
    assert latency_pos > quality_pos
    # p50/p95/max, not a bare mean (same discipline as the evaluator).
    latency_section = report[latency_pos:]
    assert "p50" in latency_section
    assert "p95" in latency_section
    assert "max" in latency_section


# --- a slow-marked real-model path exists ------------------------------------


@requires_mt
@pytest.mark.slow
def test_real_backend_opus_runs_end_to_end(tmp_path):
    import onevoice.components.opus_mt  # noqa: F401

    result = run_mt_benchmark(
        "configs/opus_vi_en.yaml", REFS, tmp_path, experiment="opus"
    )
    assert result.table.counts["utterances"] == 3
    assert all(h.hyp_text for h in result.hypotheses)


# --- D21: the evaluator and contracts are frozen for this task --------------


def test_evaluator_shows_zero_diff_against_main():
    """D21: the benchmark is frozen — the evaluator does not change in Phase 4.

    `src/onevoice/contracts.py` is deliberately NOT checked here. It changed
    under the D21 correctness exception granted 2026-08-03 (D32): the
    certification suite assumed determinism the interfaces never required and
    so rejected conforming VITS components. That exception is scoped to the
    suite; no evaluator behaviour, metric, latency figure or report changed,
    which is what this test still guards.

    A checkout without a `main` ref made this raise rather than fail, so CI
    reported a broken run instead of a broken freeze and the guard went
    unenforced for three pushes. It now resolves either ref name and fails
    loudly when neither exists — skipping would put it straight back to sleep.
    """
    base = None
    for candidate in ("main", "origin/main"):
        if (
            subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", candidate],
                capture_output=True,
            ).returncode
            == 0
        ):
            base = candidate
            break
    assert base is not None, (
        "neither 'main' nor 'origin/main' is resolvable, so D21's freeze cannot "
        "be checked. Fetch the base branch (actions/checkout fetch-depth: 0) "
        "rather than letting this check quietly disappear."
    )

    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            f"{base}...HEAD",
            "--",
            "src/onevoice/eval",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    changed = {line for line in result.stdout.split() if line}

    # D45 (2026-08-04) is a scoped correctness exception: `pipeline_rtf` and
    # `streaming_cost` were ADDED because `rtf` cannot see work done inside
    # push(), which stopped being harmless when Silero landed. The exception
    # permits adding a measurement, never redefining one -- that half is pinned
    # behaviourally by tests/eval/test_pipeline_rtf.py, which asserts rtf's
    # arithmetic and summary shape directly. This guard keeps enforcing zero
    # diff across the rest of the evaluator, so the exception cannot quietly
    # widen into the metrics D21 actually froze.
    allowed = {
        # D45: pipeline_rtf / streaming_cost added because rtf cannot see work
        # done inside push().
        "src/onevoice/eval/metrics/timing.py",
        "src/onevoice/eval/evaluator.py",
        # D49: wer_normalized / cer_normalized added alongside the raw metrics,
        # which are unchanged and verified byte-identical in every report.
        "src/onevoice/eval/metrics/error_rate.py",
    }
    unexpected = sorted(changed - allowed)
    assert not unexpected, (
        "D21: src/onevoice/eval is frozen except where a correctness exception "
        f"says otherwise (D45 covers {sorted(allowed)}). Changed outside that "
        f"scope:\n{unexpected}"
    )


def test_config_notes_render_before_any_number(tmp_path):
    """A qualification must not be separable from the figure it qualifies.

    The pivot config's numbers equal the direct backend's by construction, and
    reading them without that caveat would suggest pivot translation matches
    direct translation.
    """
    from onevoice.mt_benchmark import render_report

    table = ResultsTable(
        experiment="x",
        metrics={"bleu": MetricResult("bleu", 40.92, {})},
        counts={"utterances": 3},
    )
    report = render_report("x", table, [10.0], notes="Composition overhead only.")

    assert "Composition overhead only." in report
    assert report.index("Composition overhead only.") < report.index("40.92")


def test_a_config_without_notes_renders_no_empty_block(tmp_path):
    from onevoice.mt_benchmark import render_report

    table = ResultsTable(
        experiment="x",
        metrics={"bleu": MetricResult("bleu", 1.0, {})},
        counts={"utterances": 1},
    )
    assert "**This configuration.**" not in render_report("x", table, [1.0])


def test_the_shipped_pivot_config_carries_its_caveat():
    # The pivot backend is a real model composed with IdentityMT, so its BLEU
    # equals opus's by construction. That must travel with the config.
    notes = load_config("configs/pivot_vi_en.yaml").experiment.notes
    assert "NOT translation quality" in notes
    assert "IdentityMT" in notes
