# OneVoice — Project Knowledge

> **Archive note (2026-10-24).** This trimmed repo keeps the decision ledger and
> the consolidated results. The design spec, detailed investigations, handoff
> notes and old per-run benchmark reports are not copied. Measurements and
> decision IDs are preserved here and in `docs/RESULTS.md`; paths under
> `docs/superpowers/...` and `docs/benchmarks/...` below are historical
> references, not files in this checkout.

Living state for the OneVoice S2S benchmark framework. Updated at every phase
gate, not at the end of the project.

Spec of record:
`docs/superpowers/specs/2026-08-01-onevoice-benchmark-framework-design.md`

## Current status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffolding, tooling, CI, ledger, phase-gate skill | Gate passed |
| 1 | Evaluator: manifests, metrics, report | Gate passed |
| 2 | Interfaces, registry, config | Gate passed |
| 3 | Dummy components + sequential pipeline | Gate passed |
| 4 | Model integration: Whisper, Opus-MT/m2m100/NLLB, Piper/MMS, Silero | Gate passed |
| 4.1 | Framework hardening: certification, streaming metrics, provenance, reproducibility, interpretation | Closed - framework frozen as infrastructure (D50) |
| 5.0 | Whisper checkpoint quality reference | Closed - `turbo` is the quality reference, not the deployment model (D51) |
| 6 | Quality-first component selection, Track A per slot (D52, D54) | **In progress** - see below |
| 7–8 | Async scheduler, device port | Not designed |

**Phase 6 Track A**, as of 2026-10-23:

| Slot / direction | Status |
|---|---|
| ASR vi->en | Working deployment candidate: wav2vec2 + 4-gram LM (D80); `turbo` remains quality reference and fallback; formal D60 selection unresolved, n=400 escalation not run (D61) |
| MT vi->en | `envit5-translation` selected after the D66 correction (D65) |
| MT vi->zh, vi->ko | `madlad400-3b-mt` selected (D65) |
| MT en->vi | `envit5-translation` selected; madlad errored, not judged (D68) |
| MT zh->vi, ko->vi | Blocked at Gate D / no corpus (D67) |
| TTS vi | **In use: `vieneu` (VieNeu-TTS v3 Turbo, voice `Mai Anh`), owner decision without MOS (D79).** MOS track paused: **6A** protocol frozen (D69) · **6B** manifest built (D70) · stimulus preparation frozen (D71, D72) · **6C in progress** - 200 stimuli prepared; short-session panel (D77) and two-stage runner built; blocked on native-speaker approval of the rater text and on recruitment (>= 64 valid sessions, 80 built). Pilot dry run returned 2026-10-19 (item 5). **D78: owner accepts a single-rater panel as the official result**; analysis path for it not built, open sub-decisions Q9. No valid MOS exists yet |
| TTS en/zh/ko | Blocked - registry decision on Piper en/zh/ko incumbents deferred (D69) |

**Next session starts here (handoff 2026-10-19).** Vietnamese TTS is now
VieNeu-TTS v3 Turbo by owner decision (D79): `tts: vieneu`, default voice
`Mai Anh`, run in `.venv-tts-vieneu`. The Phase 6C MOS track is **paused**
("chưa cần quan tâm tới bài MOS"); its state is kept below for when it
resumes. If it resumes, read D77, D78, Q9 and 6C items 5-7 first.
The owner has decided (D78) that one rater's sessions are the official
result. Nothing implements that yet: `panel_analysis` still requires >= 8
raters per block and excludes 78/80 of the returned sessions on the anchor
rule. Q9 lists the decisions to get from the owner **before** writing code;
do not pick them silently. Working-tree state not mine to commit: `CLAUDE.md`
has an uncommitted owner edit; `DMH_OVC/` is untracked.
Pilot analysis output: `outputs/mos_vi_40/panel/pilot_analysis/`.

**6C status and open items** (verified 2026-10-19 against the tree):

1. **Stimuli prepared 2026-09-26** (pipeline check, approved by the owner):
   200/200, 0 errored - 40 upper anchors, 40 lower anchors, and 40 each for
   `piper-vi-vais1000`, `mms-tts-vie`, `f5`, all read back at -26.000 LUFS,
   peak <= -3.38 dBFS, hashes matching. In `outputs/mos_vi_40/` (untracked).
   33 F5 clips are `recovered` from a run that stalled under memory pressure
   on utt 1937 (73 words); their latency was not recorded and is null.
2. Manifest -> synthesis bridged by `mos_synthesis_refs` (40/40 text match);
   synthesis via `scripts/prepare_mos_stimuli.py synth`, resumable per clip.
3. **Panel tool built (D74-D77).** `onevoice.panel` (blocks, 20-trial
   sessions, anchors, repeats, Huggins screen, stereo check, offline page),
   `onevoice.panel_analysis` (exclusions, block-nested bootstrap, two-stage
   runner), `scripts/build_mos_panel.py --sessions 80`,
   `scripts/analyse_mos_panel.py resolution|compare`. Stage 2 runs only after
   the incumbent-only resolution file is committed. Rater text
   `configs/panel_text_vi.json` is **not approved**: it gained stereo-check,
   wired-headphone and clearer tone wording, and must be approved in full.
4. Panel recruitment is external work (Q4). Collecting any rating stops for
   owner approval first.
5. **Pilot dry run returned 2026-10-19 (`test_survey/`, deleted 2026-10-22) - not
   evidence.** 80/80 pilot sessions came back and every file matches
   `panel/key.json` (session id, trial order, headphone trials); all passed
   the stereo check and the 6/6 Huggins screen. All 80 carry one identical
   user agent, never overlap in time (74 on 2026-10-18, ~6 min each, ~90 s
   apart) and the tracker has no person or contact, so they are consistent
   with one person rating every session - not 80 independent raters.
   Ratings are compressed (96% are 2 or 3). Natural speech (upper anchor)
   averaged 2.59, below `f5` 2.98 and `mms-tts-vie` 2.68; lower anchor 2.18,
   separation 0.41 against the 1.5 gate. `analyse_mos_panel.py resolution
   --pilot` returns `insufficient_raters`: 78/80 excluded by the per-rater
   anchor rule (upper >= 3.5, lower <= 3.0), 2 valid; stage 2 refuses, as it
   should. What it shows: the package -> results -> two-stage runner path
   works end to end on real returned files. What it does not show: any
   system ranking. Open: why natural speech was rated ~2.6 (instructions,
   scale anchoring, or playback) - check before the real panel.
   **Noise hypothesis tested 2026-10-19, rejected as the explanation.**
   Frame-energy SNR (p95 - p10 of 25 ms frame power; heuristic, reads the
   designed 10 dB lower anchors as 15.5): upper anchors median 26.3 dB vs
   piper 30.8, f5 34.2, mms 55.5, so natural speech *is* the noisiest
   non-degraded group, and 4 upper anchors (1913, 1899, 1937, 1805) sit near
   12 dB, about lower-anchor level. But within upper anchors rating does not
   follow SNR (r = -0.08; the cleanest third scored lowest, 2.21), and r is
   ~0 in every group. Noisy anchors are a stimulus-selection weakness worth
   fixing, not the reason natural speech scored ~2.6.
6. **D78 (2026-10-19): single-rater panel is the official result.** Owner
   decision taken after seeing item 5. Consequences that must travel with
   any number it produces: intervals reflect utterance variance only (one
   rater cannot estimate rater variance), the MOS is one listener's
   judgement, and the rule was changed after data was seen. Open
   sub-decisions in Q9. `{return_method}` placeholder
   (`configs/panel_message_vi.md:26`) is filled at pack time with
   `distribute_mos_panel.py pack --return-method "..."`; less relevant now
   that no external raters are planned.
7. **VieNeu-TTS v3 Turbo smoke, 2026-10-19 - exploratory, outside D69.**
   `pnnbao-ump/VieNeu-TTS-v3-Turbo` @ `61b85e3d`, Apache-2.0, `pip install
   vieneu` 3.8.3, ONNX/CPU fp32, 48 kHz, 25 preset voices (default voice
   used). Env `.venv-tts-vieneu`; needs `onnxruntime<1.23` (1.31 rejects
   external-data files behind HF-cache symlinks) and graphs in
   `models/vieneu-v3-turbo/onnx_update` (passed as `onnx_dir`). 5 manifest
   utterances (1676, 1717, 1757, 1937, 1966): load 8.7 s, RTF 0.78-1.26 on
   this Mac, one clip peaks 1.06 (would clip without the -26 LUFS prep).
   Whisper-turbo round trip on the same 5: WER 0.011 / CER 0.005 vs piper
   0.063/0.042, mms 0.085/0.048, f5 0.063/0.055, natural 0.074/0.042.
   n = 5, intelligibility only, not naturalness and not evidence. Making it
   a 6C challenger is a D69 amendment (frozen challenger list) and needs
   its 40 stimuli prepared through §6-§7. Scripts and wavs:
   `outputs/vieneu_smoke/` (per-voice clips in `voices/`). Superseded in
   practice by D79: VieNeu adopted directly; per-voice CER table in D79.

**Phase 4 gate** (criteria from the Phase 4 design spec §9), run 2026-08-04:

| # | Criterion | Status |
|---|---|---|
| 1 | A complete configuration - real VAD, ASR, MT, TTS - produces a scored manifest on a real vi->en corpus | PASS - `configs/full_silero_vi_en.yaml` over FLEURS-50, 50/50 scored |
| 2 | Every wrapper passes the Phase 2 contract suite unmodified | PASS - Whisper, Opus-MT, PivotMT, m2m100, NLLB, Piper, MMS, Silero |
| 3 | `src/onevoice/eval/**` unchanged; `contracts.py` only by documented correctness fix | PASS - `eval/` zero diff vs `main`; `contracts.py` changed only under D32, which altered no metric, latency figure or report |
| 4 | `pytest` and `ruff check .` pass **without** the models extra | PASS - 454 passed, 146 skipped |
| 5 | CI green on push, models extra not installed | PASS - run 30872311456 |
| 6 | Both MT branches, Piper baseline and MMS challenger measured and recorded under Verified measurements | PASS |
| 7 | Every benchmark report carries the Q6 limitation statement | PASS - now enforced by test, not by eye |
| 8 | Piper runs by default in the shipped Vietnamese configuration (D23) | PASS - displaced by nothing; MMS measured 12x slower |
| 9 | Q7 resolved before Silero is integrated | PASS - D42 landed in Task 10, Silero in Task 11 |
| 10 | `scripts/phase_gate.py 4` passes | PASS |

Two criteria failed on the first run and were fixed rather than waived: CI had
been red for three pushes because the D21 freeze guard could not resolve `main`
in a shallow clone (so the freeze was unenforced in CI while passing locally),
and two hand-written reports referenced Q6 without carrying its statement.

**Phase 4.1 closing** (`docs/PHASE_4_1_CLOSING_REPORT.md`), 2026-08-05:

| Item | Problem | Decision | Status |
|---|---|---|---|
| F1 | VAD certification passed a detector hardcoding `t_audio=0.0` | see D44 | Closed |
| F2 | `rtf` blind to work inside `push()`; five reports asserted a false premise | see D45 | Closed (stage attribution -> Phase 6) |
| F3 | Point estimates published as conclusions | see D48 | Closed |
| F4 | Six of eight wrappers recorded no provenance | see D46 | Closed |
| F5 | Metrics diverged from their natural interpretation | see D49 | Closed |
| F6 | Runs identified inputs by filename | see D47 | Closed |

**No measurement changed anywhere.** Four interpretations were weakened; `rtf`
and raw `wer` are byte-identical in every regenerated report, and re-running the
Silero benchmark produced zero differing transcripts across 50 utterances.
Verdict: **READY FOR PHASE 5**, conditional on n>=200 for any MT selection.

**Phase 0 gate:**

| # | Criterion | Status |
|---|---|---|
| 1 | `python -c "import onevoice"` succeeds | PASS |
| 2 | `uv run pytest` passes | PASS |
| 3 | `uv run ruff check .` clean | PASS |
| 4 | CI green on push | PASS — run 30777808315 on `chispham258/OneVoiceDemo`, 340 passed / 3 skipped on the GitHub runner, 2026-08-03 |

**Phase 1 gate:**

| # | Criterion | Status |
|---|---|---|
| 1 | `python -c "import onevoice"` succeeds | PASS |
| 2 | `uv run ruff check .` clean | PASS |
| 3 | `uv run pytest` passes | PASS |

**Phase 2 gate:**

| # | Criterion | Status |
|---|---|---|
| 1 | `python -c "import onevoice"` succeeds | PASS |
| 2 | `uv run ruff check .` clean | PASS |
| 3 | `uv run pytest` passes | PASS |

**Phase 3 gate:**

| # | Criterion | Status |
|---|---|---|
| 1 | `python -c "import onevoice"` succeeds | PASS |
| 2 | `uv run ruff check .` clean | PASS |
| 3 | `uv run pytest` passes | PASS |

## Decision ledger

| # | Decision | Rationale |
|---|---|---|
| D1 | Mac-only development for now | No Snapdragon hardware on hand |
| D2 | Synthetic fixtures as the first dataset | Makes the harness testable before real corpora exist |
| D3 | Session scope is Phases 0–3 | Avoids multi-GB downloads before the harness is trustworthy |
| D4 | vi↔en both directions are first-class | zh/ko generalize from the same config surface later |
| D5 | Evaluator decoupled from pipeline via a run manifest | Re-score old runs; score device runs off-device; eval tests need no models |
| D6 | Interfaces streaming-shaped from day one | Retrofitting streaming into batch interfaces would rewrite every wrapper |
| D7 | COMET is an optional extra with a lazy import | Keeps CI free of torch and a 2.3 GB checkpoint |
| D8 | Sequential driver now, async scheduler at Phase 6 | Interfaces already permit parallelism; the driver does not need it yet |
| D9 | Internal latency target 1500 ms vs the 2000 ms hard limit | Margin for thermal throttling and device variance |
| D10 | Fixtures carry ground-truth speech boundaries, asserted against VAD | Leading silence would corrupt `speech_end` and make latency numbers fiction while tests stay green |
| D11 | The `models` optional-dependency extra is deferred to Phase 4 rather than declared empty in Phase 0 | An empty extra is noise; it is declared when the first model dependency exists |
| D12 | MT pivot is implemented as `PivotMT`, a `BaseMT` composing two `BaseMT` instances — not as branching in the pipeline driver | Keeps the pivot-vs-direct choice a config line rather than a code fork. The scheduler stays unaware that a hop happened, and the streaming `push`/`flush` contract (D6) is unchanged. |
| D13 | vi↔en always routes direct; vi→zh, vi→ko and ko→vi pivot through English by default, with `m2m100_418M` configured as the competing candidate | Direct models exist for the first-class direction (D4), so pivoting there would pay latency and error-compounding cost for nothing. For zh/ko no direct vi model exists, and pivot-vs-many-to-many is a measurement question, not an argument. Every leg is permissively licensed, so MT is insulated from Q1. |
| D14 | BLEU tokenizer is selected by target language: `13a` for vi/en, `zh` for Chinese, `ko-mecab` for Korean. Korean BLEU raises if `sacrebleu[ko]` is absent rather than falling back. chrF is computed for every language as a tokenizer-independent cross-check | sacreBLEU's default `13a` splits on whitespace, which is meaningless for unsegmented Chinese and wrong for Korean morphology. A wrong tokenizer does not fail — it returns a plausible, stable, fabricated number, which is exactly what the assumed/verified ledger exists to prevent |
| D15 | `push()` and `flush()` return a concrete `Sequence`, never a generator or iterator | A generator would make state mutation lazy. A driver that calls `push()` and does not immediately consume the result would leave the component un-advanced, and the bug would appear as missing output far downstream. Returning a list forces the work to happen at call time |
| D16 | `push()` after `flush()` raises `RuntimeError` until `reset()` is called | `flush()` means end of stream. Silently accepting more audio would produce a transcript spanning two utterances that looks perfectly well-formed. The pipeline calls `reset()` between utterances |
| D17 | Registering the same `(kind, name)` twice raises `ValueError` | Silent overwrite means whichever module imported last wins, so which model you benchmarked would depend on import order — unreproducible in exactly the way this framework exists to prevent |
| D18 | The dummy ASR derives text from audio rather than a script | A scripted ASR would need a cursor advancing per utterance that reset() must not rewind — the state-leak shape the Phase 2 contract suite exists to catch |
| D19 | Fixture audio uses stdlib `wave` plus numpy, no new dependency | Deterministic synthetic fixtures are generated without adding an audio dependency to the project |
| D20 | The dummy VAD is energy-based rather than passthrough | D10 requires detected boundaries to be asserted against the fixture's declared ground truth, and a passthrough reporting speech at t=0 would make that assertion vacuous |
| D21 | The benchmark interface is frozen for the duration of Phase 4. Evaluator APIs and metric semantics do not change unless a correctness bug is found; model integrations conform to the framework, not the reverse | Benchmark numbers are only comparable across experiments if the thing measuring them held still. A metric redefined mid-phase makes every earlier result incomparable without anyone noticing, and the framework exists precisely so model choices are settled by measurement. Q6's metric-semantics fix is deferred behind this freeze rather than applied during integration |
| D22 | Business Viability is judged on commercial feasibility and cost, not on deployment licensing. Model licences are not a gating criterion for model selection | Ruled by the project owner on 2026-08-03, resolving Q1. Re-opens the CC-BY-NC candidates (NLLB-200, MMS TTS) for selection on merit. Licences stay recorded in the Model registry as facts bearing on cost, but no longer eliminate a candidate before it has been measured |
| D23 | Piper `vi_VN-vais1000-medium` is the primary Vietnamese TTS baseline for Phase 4. Alternatives are integrated as challengers and must beat it on measured evidence to displace it | It is the only Vietnamese TTS candidate shipping ONNX, the format Qualcomm's runtime consumes, so it needs no conversion before Phase 6. MIT model with CC-BY-4.0 training data, the highest maturity of any Vietnamese TTS found (629 likes, 100+ dependent Spaces, versus 45 downloads for the best permissive alternative), and the highest quality tier of the three Vietnamese Piper voices. It appears in none of the four source documents, all of which framed Vietnamese TTS as an unsolved licensing problem. Standing risk: espeak-ng rule-based G2P on a six-tone language — MOS is the detector |
| D24 | The Q6 limitation statement is composed in `onevoice.benchmark`, around the evaluator's output, never inside it | The spec requires every report to carry the statement and D21 forbids changing evaluator output — both hold only if it is prepended outside `onevoice.eval`, which also keeps Phase 1 and Phase 4 reports byte-comparable in their metric sections |
| D25 | The Q6 statement names what it does NOT affect as well as what it does | Every model in this phase is batch-shaped, so their work falls inside the measured window and those figures are sound; a blanket "the metric is broken" warning would train readers to discount correct numbers |
| D26 | `tests/test_rollback.py` picks up every `configs/rollback_*.yaml` by glob and runs each end to end | Each later integration proves its own rollback path by adding a file rather than by asserting a capability in prose |
| D27 | `whisper-small` is the reference ASR configuration and the benchmark default (`configs/whisper_vi_en.yaml`). `whisper-tiny` remains available as a development configuration (`configs/whisper_tiny_dev.yaml`) whose numbers are not reported as benchmark results | Ruled by the project owner on 2026-08-03. Every reported ASR number must be measured against one fixed configuration or the comparisons are not like-for-like. Size selection remains a Phase 5 decision; this only fixes what 'the benchmark' means until then |
| D28 | Component benchmarks archive their raw hypotheses alongside the metrics — one JSONL line per utterance with `utt_id`, backend, source, reference, hypothesis and latency | D5's reasoning one level down: storing the artifact means a metric change, or a qualitative read of why a translation is wrong, needs no re-inference. Phase 5 will want to inspect actual translation errors, and re-running m2m100 or NLLB on CPU to see them again costs minutes per backend |
| D29 | `onevoice.mt_benchmark` scores MT directly on gold `(src_text, ref_text)` pairs from the reference manifest, bypassing `run_pipeline`/ASR/VAD/TTS entirely. `MT_ONLY_CAVEAT` is composed around the evaluator's own `bleu()`/`chrf()`, the same way D24 composes the Q6 statement around it, so `src/onevoice/eval/**` stays frozen (D21) | Pipeline BLEU has been 0 for every MT backend because EchoASR emits an audio-derived placeholder, not a transcript (D18), so it was measuring the dummy ASR, not translation. Feeding gold `src_text` measures MT quality given perfect ASR — a **ceiling** on end-to-end quality, not a prediction of it; the gap against a future end-to-end number is what the ASR stage costs. `configs/pivot_vi_en.yaml` composes the real `opus-mt-vi-en` leg with `IdentityMT` as a no-op second leg, because the fixture manifest is vi→en (D13 routes this direct already) and no vi→X model in the registry lands back in English (asymmetric Opus coverage — no `opus-mt-vi-zh`; the ko route's own first leg is `opus-mt-vi-en`, which would be circular). So this `pivot` config's BLEU/chrF equal `opus`'s by construction and its latency is composition overhead only, not the two-real-model compounding cost D13 describes — a genuine vi→zh/vi→ko pivot measurement waits for Task 7's real corpus |
| D30 | A config may carry an `experiment.notes` string, which every benchmark report renders immediately after the scope caveat and before any number | A qualification stored anywhere other than beside the figure it qualifies will eventually be separated from it. The pivot config's BLEU equals the direct backend's by construction, and read without that caveat it looks like evidence that pivot translation matches direct translation. Same reasoning as D24 and the MT-only caveat, generalised so any config can carry its own |
| D31 | Component benchmarks archive their synthesized audio alongside the metrics, as MT benchmarks archive hypotheses (D28) | Requested by the project owner on 2026-08-03. MOS evaluation in Phase 5 needs audio artifacts, and regenerating them means re-running TTS inference for every utterance. Archiving also makes qualitative regression analysis possible - comparing what a voice sounded like before and after a change - which is otherwise impossible after the fact |
| D32 | **D21 correctness exception, granted 2026-08-03.** The certification suite gains `deterministic: bool = True`; stochastic components set it False and their `test_reset_restores_fresh_behaviour` relaxes generated-array *length* equality to a bounded 1.6x ratio. Every other field is still compared exactly | Not a relaxation of the benchmark freeze: no evaluator behaviour, metric, latency figure or report changed, and `eval/**` shows zero diff. It fixes a defect in the suite, which assumed determinism the interfaces never required and so rejected a conforming implementation - Piper is VITS and samples its duration predictor (8704/9984/9472/9728 samples for one phrase). A contract that cannot certify a correct component is wrong. The first attempt at this exception was too loose and let a leaky stochastic component pass all 15 tests; the bounded ratio was added after that was demonstrated, and both a leaked buffer and a latched `is_final` are now caught |
| D33 | Components may implement `provenance()`; the benchmark runner records it to `metadata.json` beside the report, manifest and audio | Voice selection is part of the experimental configuration as much as the model family is - two Piper voices differ more than two checkpoints of one voice would. Phase 5's MOS comparison reads audio artifacts that outlive the run that produced them, and a figure whose exact voice and checkpoint hash are unknown cannot be reproduced or fairly compared |
| D34 | `provenance()` is a **recommended** interface in Phase 4 and an **expected** one from Phase 5: every production model wrapper should implement it. The target is that a benchmark artifact is reproducible from `metadata.json` alone — component, model id, revision/commit, weight hash, tokenizer revision, voice id, sample rate, config, manifest version, and the framework git commit | Direction of travel stated now so wrappers are written with it in mind, without forcing a refactor mid-phase. A benchmark number whose exact inputs cannot be reconstructed is not a measurement, it is an anecdote |
| D35 | TTS backends are benchmarked by one neutral instrument over the same manifest and protocol. Piper's baseline status (D23) governs which component ships by default, never how the comparison is run or reported | A baseline exists for comparison, not preference. If MMS or MeloTTS beat Piper on latency, footprint or Phase 5 MOS, the framework must make that obvious rather than bury it - the whole premise is that model choices are settled by measurement |
| D36 | MeloTTS is dropped. Piper covers all four competition languages (vi, en, zh, ko), so the `tts-melo` extra is removed rather than worked around | MeloTTS pins `transformers==4.27.4`; Whisper and NLLB need `>=4.44`. The two cannot share an environment, and `uv lock` fails outright with both declared. Verified before deciding: `rhasspy/piper-voices` ships `ko_KR-kss-medium` and four Chinese voices - the repo's own language tags omit `ko`, so this was found by listing the tree, not by trusting metadata. All three were synthesized successfully. Dropping MeloTTS removes a dependency conflict, a second TTS framework, and a large pinned G2P stack (mecab, unidic, gruut, librosa==0.9.1), and leaves one MIT ONNX engine for every language. Reversible in one line if a Phase 5 MOS result demands it |
| D37 | The TTS component benchmark (`onevoice.tts_benchmark`) names no backend in executable code, and a test enforces that by scanning the module's own source | A comparison whose harness knows which implementation it is running is not neutral, whatever its intent. The test was verified by injecting `if backend == "piper"` and confirming it fails, then reverting. It reports latency, audio duration and synthesis RTF, and deliberately no quality metric - nothing here evaluates naturalness, which is Phase 5's MOS panel (Q4) |
| D38 | Every reference manifest declares a `boundary_source` in a sidecar `<manifest>.meta.json`, and the D10 accuracy assertion applies only when that source is independent of the detector under test. `energy_derived` is explicitly forbidden for manifests scored against an energy VAD. See `docs/BOUNDARY_METHODOLOGY.md` | D10's assertion is only worth anything if the declared boundaries were obtained independently of the VAD being tested. Deriving them with an energy detector and then asserting against an energy detector passes by construction and measures nothing - the exact passes-for-the-wrong-reason failure this project keeps finding. The sidecar sits beside the manifest because `RefRecord` is in `eval/`, frozen by D21. Corpus quality and segmentation quality stay separable: a manifest may support one and not the other, and the report says which |
| D39 | FLEURS (`google/fleurs`, CC-BY-4.0) is the Phase 4 corpus; CoVoST2 is rejected | Settles the Phase 4 spec's open item 4 on evidence: **CoVoST2 has no Vietnamese** (its languages are ar/ca/cy/de/es/et/fa/fr/id/it/ja/lv/mn/nl/pt/ru/sl/sv/ta/tr/zh), so it fails the spec's first criterion outright. FLEURS ships `vi_vn` (857 test utterances, 691 MB) and `en_us`, is CC-BY-4.0, and is FLoRes-derived so its `id` links an utterance to the parallel sentence in every other language - giving vi audio with an en reference translation. It carries no timing annotation, hence D38's `file_extent` |
| D40 | Benchmark runs are checkpointed: each utterance is appended to the manifest the moment it completes, an interrupted run resumes from the last finished utterance, and metadata records `complete` or `resumed`. `resume=False` starts clean | A full end-to-end configuration over the corpus takes tens of minutes, and this session alone has seen several interruptions. A run that silently restarts wastes that time and looks identical to one that never crashed. resume=False exists because resuming across a configuration change would mix two experiments into one manifest, and the mixture would score as though it were a single run |
| D41 | Benchmark metadata records wall-clock start and finish, duration, and host facts (CPU model, OS, architecture, Python version) | Not correctness-relevant, but a latency figure compared across two runs on different hardware is not a comparison - and by the time anyone asks, nobody remembers which machine produced which number |

| D42 | ASR gating replays audio from the VAD event's own `t_audio` through a bounded `LookbackBuffer` (`src/onevoice/lookback.py`, 500 ms), rather than gating on whether the detector is open on the current frame. Asking for audio older than the window raises `LookbackUnderrun` rather than replaying a short prefix | Closes Q7. A detector that needs context reports `speech_start` only after that audio has passed, so the old gate dropped the leading frames of every utterance and ASR started mid-word - with nothing in the record saying so, only a worse transcript. Raising on underrun is deliberate: a truncated replay yields a clipped utterance and a plausible WER nobody can attribute, and an errored utterance is excluded and counted instead. The buffer is a no-op for a zero-lookahead detector, which is pinned by test so the Phase 4 baselines stay valid |

| D43 | `SileroVAD` is the detector of record for end-to-end runs on real speech; `EnergyVAD` remains the detector certified against D10 boundary accuracy | On the same corpus, changing only the VAD, Silero halves WER (0.6094 -> 0.2929), lifts BLEU 5.261 -> 9.459, eliminates all 12 silent utterances, and is faster (p50 3962 ms vs 4774 ms). But the synthetic fixtures are tone bursts and Silero is trained on speech, so it misses one fixture entirely and closes 1.5-2.5 s early on the others - a stimulus mismatch, not a defect. Widening D10's tolerance to fit would be a test that has stopped testing, so the two roles are split rather than merged. Silero's boundary accuracy on real speech is unmeasured, not good: FLEURS is file_extent per D38 |

| D44 | **F1, Phase 4.1.** `VADContract` supplies a non-silent stimulus via an overridable `speech_chunks()` hook, and asserts that the stimulus actually triggers detection before checking timestamps. Timestamps must fall inside the pushed audio, shift exactly with the audio timeline, and be monotonic. `tests/test_vad_certification.py` holds five deliberately broken detectors that must be rejected, and asserts each is rejected *by a timestamp assertion* rather than incidentally | The suite's only VAD fixture was silence. A correct VAD emits nothing on silence, so every timestamp assertion skipped itself and a detector hardcoding `t_audio=0.0` passed 14 of 16 tests - verified by probe. Certification strength is what a broken component fails, not what a correct one passes. The hardened contract immediately found a real defect: `SileroVAD` ignored `AudioChunk.t_start` and timed from `VADIterator`'s own counter, masked because the driver feeds every utterance from t=0. D42's lookback trusts `t_audio`, so this was latent rather than harmless |

| D45 | **D21 correctness exception, granted 2026-08-04 (F2), same process as D32.** `eval/` gains `pipeline_rtf` (`stream_start -> tts_done`) alongside the unchanged `rtf` (`speech_end -> tts_done`). The exception permits **adding** a measurement, never redefining one, and tests pin that scope. The Q6 statement is now computed from the run rather than hardcoded, and leads with `streaming_cost` - the **paired** per-utterance difference, reported with `paired_n` | `rtf` begins at `speech_end`, so it cannot see work a component does inside `push()`. That exclusion models the competition's end-of-speech metric and was harmless while every model was batch-shaped - the premise Q6 was deferred on. Silero broke it inside the same phase (144.7 ms/utterance, ~0.43% of audio duration), and nothing noticed because `Q6_LIMITATION` was a constant asserting a property of the run instead of a value computed from it. Both stamps already existed, so every archived run was re-scored rather than re-run, and `rtf` is byte-identical in all three regenerated reports. No conclusion changes: Silero remains ahead of EnergyVAD under either metric |

| D46 | **F4, Phase 4.1.** `docs/PROVENANCE.md` defines one canonical component provenance schema; `src/onevoice/provenance.py` validates it; `tests/test_provenance.py` enumerates the **registry** and constructs every non-dummy backend to check conformance. Required: `component`, `implementation`, `model_id`, `revision`, `params_hash`, plus **at least one immutable anchor** (`commit_hash` or `weight_hash`), plus a non-empty `invocation` dict. The schema is split in two: **component identity** (what implementation ran) and **invocation identity** (what it was asked to do - languages, voice, route, decoding mode) | Two implementations existed and had already diverged - `MMSTTS` set `revision` and `commit_hash` to the same value, losing what was requested versus what it resolved to, while `PiperTTS` recorded a weight hash and no commit. Six production wrappers had no provenance at all, so no benchmark recorded which Whisper or Opus checkpoint produced it. Filling fields wrapper-by-wrapper would have re-litigated the schema eight times; defining it once and auditing against it evolves provenance in one place. The anchor rule is a rule about *identity*, not a fixed pair of fields: a Hub commit already is immutable identity, and re-hashing gigabytes to restate it buys nothing, while a file-based Piper voice has no commit and its weight hash is the only anchor available. Enrolment is by registry membership, so a new wrapper cannot ship without provenance |

| D47 | **F6, Phase 4.1.** Every benchmark run records `config_hash`, `refs_hash`, `framework_commit` and `framework_dirty`, and archives the config as `config.snapshot.yaml` beside the run. `run_identity()` builds it, `validate_run_identity()` checks it, and a test enumerates all three runners | Metadata recorded config and refs by **path**, so editing a config silently made an old run's record describe an experiment that never happened. Hash and snapshot serve different purposes: the hash detects that an input changed, the snapshot lets the run be repeated afterwards - F6 needs both. `framework_dirty` exists because a commit recorded from a dirty tree names code that is not what ran; when git cannot be consulted the flag is True, since failure to determine cleanliness is not cleanliness. Found while implementing: `mt_benchmark` wrote **no metadata at all**, so the MT ceiling figures Task 9's correction rests on had no provenance whatsoever |

| D48 | **F3, Phase 4.1.** `docs/INTERPRETATION.md` is the single interpretation policy: observation, evidence and conclusion are kept separate; every comparison between configurations carries a paired bootstrap CI (`src/onevoice/stats.py`, outside `eval/`); every report declares a classification - **smoke**, **development** or **selection** - via `experiment.classification` | A point estimate had repeatedly become a conclusion. The MT 'reversal' claim was a coin flip (CI [-3.71, +5.12], P=0.58), and Task 9 bolded Branch B's whole column - a ranking asserted typographically - when BLEU's CI [-0.55, +4.47] does not support it and only chrF's [+0.15, +4.04] narrowly does. Classification exists so a development benchmark cannot read like a selection one. The wording rule (no comparative language without an interval) is enforced in `Comparison.sentence()` rather than left to whoever writes the report, because it only holds if the interval writes the language. Task 11 (Silero vs EnergyVAD) is the project's only Selection-grade conclusion |

| D49 | **F5, Phase 4.1.** `wer_normalized` / `cer_normalized` (casing- and punctuation-insensitive) are reported **alongside** raw `wer`/`cer`, never instead; a new `empty_outputs` count distinguishes utterances that ran cleanly and produced nothing from both successes and errors | Two definition/interpretation mismatches. (1) Whisper emits capitals, punctuation and digits while FLEURS transcripts are lowercase, unpunctuated and spelled out, so about a third of reported WER is formatting, not model error (0.2929 -> 0.1939 on the Silero run). Raw is kept because it is what every recorded Phase 4 figure means, and because normalisation embeds debatable choices - it does not expand numbers, so it is a lower bound on formatting effects rather than a clean model-error figure. (2) `scored` counts records the evaluator could score and `errored` counts records that raised; neither notices a clean run producing nothing, which is how Branch A reported `errored: 0` beside 12 empty utterances. An empty output is not an error - calling it one would misattribute a VAD threshold problem to run mechanics - but it is not a result either. Both changes additive; raw metrics verified byte-identical in all three regenerated reports. Probes that found nothing are recorded too: Unicode normalisation form is clean (all NFC both sides) and the BLEU tokenizer routes correctly |

| D50 | **Phase 4.1 closed 2026-08-06. The framework is stable infrastructure for Phase 5.** It is not revisited unless Phase 5 produces new evidence of a correctness defect, and any such defect goes through the Framework Correctness Exception process used for D32, D45 and D49 - state the problem, root cause, impact and scope, get approval, implement additively, verify no recorded measurement moved - rather than reopening completed hardening work | Phase 4.1 fixed six defects and every fix was verified not to move a measurement. Continuing to harden past that point trades Phase 5's actual question (which models to choose) for diminishing framework returns, and each further change to `eval/` widens the D21 exception surface that the freeze exists to keep narrow. The exception process is retained rather than dropped because Phase 5 will exercise paths Phase 4.1 could not - larger corpora, more checkpoints - and a defect found there still needs a route to being fixed, just not an open-ended one |

| D51 | **Phase 5.0 outcome, 2026-08-06.** `openai/whisper-large-v3-turbo` is the project's **quality reference checkpoint**, superseding D27. Selected by the pre-registered rule in the Phase 5.0 spec, applied mechanically by `scripts/phase5_select.py`. This establishes the quality reference, **not the deployment model** | `turbo` is **superior** on the primary endpoint: Delta `wer_normalized` = -0.1338, 98.33% CI [-0.1508, -0.1175] (Bonferroni over three comparisons, n=200), and the gain propagates - confirmatory BLEU +4.668, CI [+3.087, +6.385]. Both smaller checkpoints are clearly inferior, so **H4 is refuted**: there is no ASR headroom to be had by shrinking. `tiny` is unusable for Vietnamese at `wer_normalized` 0.98. **Measured cost:** on one Apple M1, CPU, unquantised and unoptimised, `turbo` records `rtf` 0.934 and 2.15x the incumbent's end-to-speech latency at 3.3x the parameters. **Not measured:** anything about Snapdragon-class hardware, or any quantised or accelerated form - so deployment feasibility is the primary engineering risk entering Phase 6, a hypothesis to test rather than an established finding. An earlier draft called it a 'blocker'; that inferred a Snapdragon outcome from an M1 measurement and was withdrawn, with the measurement unchanged |

| D52 | **Phase 6 answers one question: can the quality reference be deployed within the competition constraints?** No further model-selection experiment is opened unless deployment validation fails | Phase 5.0 settled which checkpoint is best on quality, decisively. Re-opening selection before knowing whether that checkpoint can be deployed would optimise against a constraint nobody has measured - and the levers that matter for deployment (quantisation, distillation, on-device runtimes) are not model-choice questions at all. Sequencing them the other way round would also let an M1 latency figure quietly veto a result established on 200 utterances with corrected intervals. If validation fails, the choice becomes optimised `turbo` versus a measurably worse quality level, and *that* is the point at which a new selection experiment is warranted |

| D53 | **Latency framing corrected, 2026-08-13. Supersedes D9's characterisation; D9's text is left intact.** The competition states *"Solutions with latency exceeding 2.0 seconds will receive significant point deductions"* - a **graded penalty, not a hard limit or disqualification threshold**. The 1500 ms figure survives as a legitimate **engineering target** but must stop being described as derived from a hard limit. RTF < 1.0 is the only hard numeric requirement now known, and both figures are **provisional** | D9 recorded a "2000 ms hard limit" sourced *"Adopted from the external GPT plan"*, and ~12 documents inherited the binary framing. Verified against both organiser sources: the Luma event page carries the numbers, the official Saigon AI Hub page carries none and states *"Specific benchmark values will be published prior to the prototype submission phase"*. Superseding rather than editing follows the D51-over-D27 pattern - the original stays readable and the correction is dated |
| D54 | **Quality-first model selection, 2026-08-13.** The order is quality selection -> system integration -> Qualcomm optimisation -> target-device validation. Qualcomm AI Hub availability is **recorded as Track C evidence and is explicitly non-gating** for model choice. Evaluation is split into three tracks that are never collapsed: A quality (BLEU, COMET, WER/CER, MOS), B system performance, C deployment feasibility | Reversing the order would let a model be chosen because Qualcomm happens to host it, eliminating better candidates before any quality evidence exists - and AI Hub's catalogue reflects Qualcomm's priorities, not Vietnamese speech-translation quality. A model winning Track A and failing Track C is an engineering result to be reported, not a contradiction to be avoided by merging the tracks. Round 2 requires a proposal, not a completed deployment, so only target-device validation is legitimately blocked by the absence of AI Hub credentials |
| D55 | **COMET is a prerequisite for any Track A selection (Gate 0).** No candidate selection is made until COMET runs reproducibly on the evaluation corpus | The competition scores accuracy as **BLEU + COMET**. COMET has sat behind an optional extra since D7 and has **never been run in this project**, so every quality conclusion to date is missing half the competition's accuracy criterion. Selecting models on BLEU alone would optimise against a metric the competition only half uses |

| D56 | **Gate D (diversity), 2026-08-13.** A slot's candidate matrix must contain **at least two architecturally distinct model families, and at least one candidate outside the incumbent's family**. A matrix failing Gate D may record its result as *"best within family X"* and **must not** be recorded as a slot selection. A fine-tune or size variant is the same family | Phase 5.0 compared four Whisper variants with a pre-registered rule, paired bootstrap and Bonferroni correction - and its conclusion was still *"the best Whisper variant"*, because no non-Whisper ASR was ever in the matrix. **Statistical rigour does not fix a predetermined matrix:** interval evidence says which candidate won, never that the right candidate was present. Applied retroactively this is a clarification, not a reversal - D51 says "quality reference checkpoint", which is exactly what a single-family result supports, and nothing is withdrawn. The gate catches the subtle case too: `PhoWhisper` is a Whisper fine-tune, so it feels like diversity and does not provide it |
| D57 | **Component interfaces must expose deployment metadata**: model id + revision, requested **and effective** precision, runtime/backend, requested **and effective** execution provider, device/platform, model size - alongside the existing Track A/B metrics. Implementation is an additive D50 exception | Requested-versus-effective pairs are separate fields by design: recording only the request records an intention rather than a measurement, and a run labelled NPU that silently fell back to CPU is the likeliest false conclusion in deployment work. This is the same provenance failure D46 fixed at component level and D47 at run level, arriving now at the deployment level before it can produce a wrong number rather than after |

| D58 | **Language-coverage evidence, 2026-08-14.** Every candidate records **how** its language claim was established, on a ranked scale: `unverified` < `model_name` < `model_card_prose` < `hub_metadata` < `config_inspected` < `output_tested`. Only the top three count as verified. `family` means a **checkable architecture**, not a brand | "Vietnamese supported" has to mean something checkable. Verifying the Step-1 matrix against Hub metadata found **six wrong entries**: PhoWhisper is BSD-3-Clause not Apache-2.0, envit5 is OpenRAIL not MIT, viXTTS is "other" not CPML, F5 is CC-BY-NC-SA-4.0 not CC-BY-NC-4.0 - and two family errors that mattered more. **NLLB's Hub architecture is `m2m_100` and MADLAD's is `t5`**, so Step 1's five MT "families" were really three; counting brands rather than architectures had inflated the diversity Gate D depends on. `facebook/mms-tts-vie` carries **no language tag at all** - its Vietnamese claim rests on the ISO-639-3 suffix `vie`, recorded as `model_name` rather than rounded up |

| D59 | **Coverage is a capability dimension, separate from quality, 2026-08-14.** Five dimensions are never collapsed: **coverage** (what a candidate claims), **evaluation_coverage** (what our corpus can test), **deployment_coverage** (what may run on target; empty means unevaluated, never "none"), **quality** and **performance** (both measured later, both living in run manifests). Core track is vi->en and decides slot selection; vi->zh and vi->ko are represented but must not distort it | Breadth is not merit and narrowness is not a defect. Without the separation, a Vietnamese-specialised ASR gets eliminated for zh/ko it was never asked for, a bilingual MT loses to a 2.9B multilingual on coverage rather than quality, and "only NLLB covers six directions" reads as a preference instead of the constraint it is. Building the substrate surfaced a real gap in **our** side: FLEURS gives Vietnamese *source* audio, so all three reverse directions (en->vi, zh->vi, ko->vi) are currently **unevaluable** - three of the competition's six. That is attributed to our corpus via `Coverage.unevaluable` rather than counted against any candidate |

| D60 | **Selection hierarchy and acceptance tolerance, 2026-08-14.** Order: Gate D -> eligibility/coverage -> quality -> non-inferiority -> performance/deployment tie-breakers. **delta is a ceiling on acceptable regression, not a budget to spend**, is **per slot and per metric** rather than global, and is `k x resolution` with **k=1.5 approved** - giving delta ~0.018 for ASR vi->en. Internal policy, **not a competition requirement** | Two concepts had been conflated. **Statistical resolution** (how small a difference the experiment can distinguish) is measured from the incumbent before any candidate is scored. **Engineering acceptance tolerance** (how much quality we will trade for speed or size) is a judgement. The noise floor is evidence *about* delta, never a derivation of it - an experiment becoming more sensitive does not make us more willing to lose quality. One global delta would also be meaningless across ASR WER, MT COMET and TTS MOS, which differ in scale and direction. The hierarchy is written so **"choose the fastest candidate within delta" is structurally impossible**: performance breaks ties only inside a *measured* quality-equivalence tier, and with no pairwise evidence no tie is declared, because declaring an unmeasured tie is precisely how a worse model wins on speed |

| D61 | **ASR vi->en selection status is UNRESOLVED, 2026-08-14, and the n=400 escalation is NOT run.** Turbo is retained as the **working reference**, not declared superior. Escalation is deferred because it **cannot change the outcome** | Projected from the existing paired data with delta held fixed: `phowhisper-large` would resolve NON-INFERIOR at n=400, `seamless-m4t-v2` would **not resolve at any feasible n** - its point estimate sits 0.0019 below delta, so it needs half-width <0.0019, i.e. n ~= 22,161, **26x the entire FLEURS Vietnamese test split (857)**. That is geometry, not under-powering. Decisive argument: simulating D60 with the projected intervals returns the incumbent in **every** reachable outcome, because neither challenger can be *superior* (both point estimates are worse, and intervals contract toward them) and both **lose the pre-registered size tie-break** - 1543M and 2309M against 809M. Twelve hours of compute would buy procedural closure, not knowledge. Size was **not** used to declare a winner; it enters only as D60's tie-break inside a quality-equivalence tier, and only in a simulation of what escalation *would* produce |

| D62 | **COMET verification gate, run before MT scoring: COMET is NOT the primary metric for vi->zh.** Primary is **COMET for vi->en and vi->ko**, **chrF for vi->zh**, with COMET still reported there but not decisive. delta is henceforth **per direction** as well as per slot and metric | Protocol section 5 required verifying COMET on zh/ko before use. Measured discriminative range (perfect minus content-shuffled references, which are fluent and on-domain but wrong): **vi->en 0.461, vi->ko 0.471, vi->zh 0.083**. On zh, content-randomised Chinese scores **0.907** and half-truncated Chinese **0.937** against a perfect 0.990 - the metric cannot separate good output from wrong output there, so it must not decide anything. BLEU and chrF were then checked as the protocol's named fallback and discriminate well everywhere (zh ranges 90.15 and 90.78). The alternative is defined **before** scoring and COMET is **not silently dropped** - it is reported as secondary. This also forces delta per direction: the same metric having a ~5.6x different effective scale across directions means a delta derived on vi->en COMET is meaningless on vi->zh COMET |
| D63 | **The D61 stopping rule is NOT adopted.** *"A challenger cannot win when its point estimate is worse and it loses the tie-break"* remains a proposal requiring separate methodological review and pre-registration; it does not govern the MT track or any other | It was formulated **after** observing candidate results. Adopting a selection rule derived from the data it would then govern is the precise failure D48 and the pre-registration discipline exist to prevent, however sound the reasoning looks. D61's *analysis* stands - the n=400 escalation has no decision value - but that is a one-off decision on one experiment, not a standing rule |
| D64 | **Long-running experiments use an isolated environment (`UV_PROJECT_ENVIRONMENT`); no `uv sync` against an active benchmark environment.** The underlying fragility is recorded as a D50 correctness-exception candidate and is **not** fixed inside a selection track | Two multi-hour ASR runs were corrupted at finalisation by environment mutation mid-experiment. D40 protects inference checkpoints but not the environment, so both runs survived only because every utterance was already flushed. `collect_provenance` aborts a completed run when a component's `provenance()` raises, while `provenance.py::_git` already carries the principle that provenance must never break a run - so the principle is real but not applied uniformly. Treating this as a process-control defect rather than operator carelessness is what makes it fixable |
| D65 | **MT track scored: `madlad400-3b-mt` selected for vi->zh and vi->ko; vi->en produced no selection.** delta approved and committed before any candidate ran (vi->en COMET 0.0176, vi->zh chrF 2.351, vi->ko COMET 0.0263) and unchanged after results. All five candidates executed, so Gate D holds on executed candidates in all three directions and the MADLAD contingency was not triggered | madlad is superior on quality in both expansion directions - vi->zh chrF +7.65 CI [+5.59,+9.86], vi->ko COMET +0.0840 CI [+0.0658,+0.1045]. vi->en initially returned escalate/no-selection because two candidates that are *worse* than the incumbent are unresolved against delta and the hierarchy short-circuited before reaching the superior branch (Q8); after the D66 correction was pre-registered and committed, re-scoring the archived hypotheses selects `envit5-translation` (COMET +0.0293 CI [+0.0183,+0.0408]). zh/ko evidence, verdicts and selections are byte-identical across the correction. Full result: `docs/benchmarks/2026-08-15-mt-track-result.md` |
| D66 | **Q8 selection-rule correction, formal definition: an unresolved candidate is `unresolved_decision_relevant` iff some admissible value inside its own uncertainty interval would make it the SELECTED candidate under the already-fixed hierarchy; otherwise `unresolved_decision_irrelevant`.** Only decision-relevant blocks. Computed as a counterfactual that re-applies `select` itself, so it cannot drift from the rule; inputs are the interval, orientation, fixed delta and selection rule only - never performance/size/compute. Neither status is ever converted to a resolved verdict, and decision-irrelevance is not a quality claim. Pre-registered and committed **before** any re-scoring; no model inference re-run | The hierarchy short-circuited on any `unresolved` before reaching the superior branch, so candidates that could not win blocked ones that already had (see Q8). Stated generically, not from the observed results: any track with one clearly-worse undecided challenger and one clearly-better challenger hits it. Weakest exclusion that fixes the defect - a candidate whose point estimate is *better* than the incumbent still blocks. Spec: `docs/superpowers/specs/2026-08-15-q8-selection-rule-correction.md` |
| D67 | **en->vi opened as a NEW directional track; COMET Vietnamese-target gate PASSED (discriminative range 0.5211, widest of any direction); primary COMET, delta = 0.0183 approved and fixed before any candidate ran.** zh->vi and ko->vi stay blocked at Gate D and coverage claims were NOT extended. TTS stays blocked on three independent blockers | Corpus `fleurs_en_vi_200` (refs sha256 `ec23245515…`) shares the same 200 FLEURS ids as `fleurs_vi_en_200` with source/reference roles reversed, so the two directions are **not independent observations** and must never be pooled. COMET had been gated for en/zh/ko targets only and had FAILED for zh, so Vietnamese-target validity was measured rather than assumed. Incumbent NLLB COMET 0.8419, half-width 0.0122 at 95%/1000/seed 0. Pre-registration: `docs/superpowers/specs/2026-08-15-en-vi-track-preregistration.md` |
| D68 | **en->vi result: `envit5-translation` selected (COMET +0.0255 CI [+0.0164,+0.0364] at 98.333%); `m2m100` inferior; `madlad400-3b-mt` ERRORED and excluded from all quality ranking.** MADLAD ran 1d10h49m without completing (~18x its own vi->X runtime, alive inside torch matmuls, no checkpoint, no decode cap) - an execution failure, never a quality verdict | Gate D held over EXECUTED families {m2m_100, t5} because envit5 also carries t5, unlike vi->zh/vi->ko where madlad was the sole carrier. The Bonferroni level was NOT loosened from the pre-registered 98.333% (3 planned challengers) to 97.5% after the failure: relaxing a multiplicity correction because a candidate did not execute would ease the test in response to an outcome. envit5's degenerate 480-word output on utterance 1667 was preserved and scored as produced (segment COMET 0.3267 vs corpus mean 0.8674) - a live failure mode of the WINNER, carried into Track B/C. A bounded-decoder MADLAD run would need a NEW pre-registered condition applied to all candidates, not a candidate-specific repair. Result: `docs/benchmarks/2026-08-17-en-vi-track-result.md` |
| D69 | **Phase 6A, 2026-09-03: TTS pre-registration FROZEN before any MOS exists, Vietnamese target only.** Incumbent `piper-vi-vais1000`; challengers exactly `mms-tts-vie` and `f5`. Gate D `{vits, f5}` carried solely by F5 (viXTTS cannot run Vietnamese); if F5 errors the direction is scoped/unresolved, never an incumbent win, and no `vits` substitute may restore the count. k = 1.5 frozen as **owner-set**; anchor gate 1.5 MOS owner-set; Bonferroni 97.5% for two challengers, not loosened if one errors. Paired candidate-vs-incumbent bootstrap resamples utterances and raters with shared indices. Duplicates `manifest_ids[0::10]`, exclusion statistic only. vi->en/zh/ko TTS stay blocked; registry untouched | Freezing while no rating, resolution or delta exists means no observed result can have shaped the rule. Claims that ~0.2 MOS is a noticeable-difference threshold were removed - nothing in this repository establishes one. The freeze surfaced a real D46 defect: both cloning wrappers lacked `revision`, `params_hash` and an immutable checkpoint anchor; fixed without relaxing their refusal guards. Spec: `docs/superpowers/specs/2026-08-17-tts-track-preregistration.md`; closure: `docs/superpowers/specs/2026-08-17-tts-protocol-closure.md` |
| D70 | **Phase 6B, 2026-09-03: Vietnamese MOS manifest built, text only.** `corpora/tts_mos_vi_40/manifest.jsonl`, n = 40, `select_ids(parent_ids, limit=40, seed=0)` over `corpora/fleurs_en_vi_200`, text = parent's Vietnamese `ref_text` (never a model output). Digest `df02975fb8b54f15496202fa1f77ef299c800a7c1464815c27709514cbd0b693`. Selection rule recorded in the frozen protocol §7 | 6A said what to build but not how to draw 40 from 200, and the duplicate stride depends on that order; the rule was escalated and approved before the build rather than chosen silently. The existing subsetter spreads the sample across the id range where "first 40" would draw one contiguous FLEURS block. The build refuses on a parent-hash mismatch; seven tests pin count, reproducibility, digest, parent, text source, duplicate stride and the absence of any rating/score/audio field |
| D71 | **Stimulus preparation frozen, 2026-09-05** (protocol §6-§7 amendment; no rule changed, no audio rendered). Order resample -> degrade -> level-normalise -> encode, in an isolated environment, float64. 16 kHz via `soxr==1.1.0` `quality='VHQ'`; -16 LUFS integrated via `pyloudnorm==0.2.0` (**superseded by D72: -26 LUFS**); -1.0 dBFS ceiling on the 4x-oversampled signal; 16-bit PCM mono WAV, no dither. Lower anchor: AWGN at exactly 10 dB SNR over the upper-anchor recording, `default_rng(0)` fresh per utterance, full-waveform RMS | The protocol named these parameters without giving them. The peak ceiling is **not** called dBTP: pyloudnorm measures sample peak and does not implement BS.1770-4 Annex 2, so that unit would claim conformance we cannot deliver. Accepted owner-set limitations: measuring RMS over silence raises the in-speech SNR by mean +0.97 dB (max +2.33 dB) across the 40 recordings, and additive noise degrades recording quality rather than machine-likeness, so the 1.5 MOS anchor gate could fail for a genuinely sensitive panel |
| D72 | **Loudness target amended to -26 LUFS, 2026-09-26, before any stimulus was rendered. Supersedes D71's -16 LUFS; D71's text is left intact.** The -1.0 dBFS ceiling is kept. A stimulus still over the ceiling after normalisation, or whose loudness cannot be measured, is `errored` (§8) - never limited, compressed or given its own gain | D71's two level rules contradicted each other on this corpus: at -16 LUFS the 4x oversampled peak exceeded -1.0 dBFS on 39/40 upper and 40/40 lower anchors (median +2.2, max +6.15 dBFS), measured in memory with no file written, and a limiter is excluded because normalisation may change absolute level only. Of the options, only lowering the common target keeps equal loudness across stimuli and honours the ceiling; -23 LUFS already fails one anchor, and peak-wins-per-clip would level-mismatch systems. At -26 LUFS anchors peak at most -3.85 dBFS. Owner-set, **not** P.56 -26 dBov. The only input was anchor peak level - no rating, synthesis output or candidate result existed |
| D73 | **D50 exception, 2026-09-26: the cloning-TTS checkpoint anchor is the Hub commit or nothing.** `_commit_from_snapshot` no longer resolves symlinks, and refuses a path outside a Hub snapshot instead of hashing it. Owner-approved. Audio, measurements and selection rules unchanged | Problem: Hub snapshot files are symlinks into `blobs/`; resolving them removed the `snapshots/<commit>` component, so F5's `commit_hash` silently fell back to sha256 of the *path string* - a value shaped like an anchor that anchors nothing. Introduced in 6A (`b5f52dd`), untested. Impact: F5 records only (viXTTS passes a snapshot directory, which was unaffected); the one MOS record was corrected in place with the wrong value kept beside it, since sha256 of `.../snapshots/50228ccc.../model_last.pt` equals what was recorded. Re-preparation after the fix produced byte-identical audio for all 200 stimuli. Verified on the real cache: F5 `50228ccc...`, viXTTS `c06f4378...` |
| D74 | **Panel delivery and headphone screen frozen, 2026-09-26, before any rater.** Offline HTML package per rater with opaque clip file names; results returned as JSON; blinding key never in a package. Rater *i* seeded `Random(i)`; no utterance in consecutive trials. Headphone screen: Huggins pitch after Milne et al. 2021 (600 Hz ± 6%, 180 degree, 1 s Gaussian noise intervals, 44.1 kHz, 12-trial pool seed 0, 6 drawn), **pass 6/6** - the authors' recommendation. Ramps, gap and level owner-set. Rater-facing Vietnamese text must be native-speaker approved before a page builds | The protocol required a headphone-screening item without naming one and did not say how clips reach raters. Huggins pitch was chosen over the antiphase test because it is harder to pass over loudspeakers. The threshold was first presented to the owner as 5/6 'published criterion' from memory; checking the paper showed 6/6 is the recommendation, the owner was re-asked with the corrected fact, and chose 6/6. Delivery as a static page needs no server, platform or account and works with directly recruited raters |
| D75 | **Exclusion readings fixed before any rater, 2026-09-26.** Rule 1 scored from the frozen pool, not the page's flag; rule 4 = median over trials of (trial time - clip duration) < 0; an incomplete session is excluded and reported | Rule 4's text ('median trial time below the clip duration') admits more than one reading, and the protocol never covered a partial session. Both are fixed now so no reading can be chosen after seeing ratings. Under this delivery rule 4 cannot fire, because rating is locked until playback ends; it stays as a guard against results the page did not produce |
| D76 | **Unscored stereo check before the headphone screen, 2026-09-26, before any rater.** Beep in one ear then the other, seeded varying order; the rater must name the first ear to continue, with setup advice and unlimited retries. Never an exclusion criterion; the 6/6 Huggins screen is unchanged; rater plans are untouched (pinned by test) | The owner's pilot failed the screen hearing only noise. The Huggins files were verified correct, so the likely cause was both ears receiving one mix, which is a correctable setup fault. Under D74 that fault costs a rater's whole session with no chance to fix it. A check that must be answered correctly catches loudspeakers, mono output and call-mode headsets before the scored screen, without making the screen easier. The browser run found and fixed a page bug: the retry button opened the next attempt without playing it. Adding rater-facing strings reset the text approval, since an approval cannot cover wording that did not exist |
| D77 | **Short sessions replace the 212-trial session, 2026-09-28, before any rater** (owner-approved). 8 blocks of 5 utterances; a session = 1 block x 3 systems (15) + 2 upper + 2 lower anchors from other blocks + 1 duplicate = 20 trials; >= 8 valid raters per block, 80 sessions built first; block-nested bootstrap (utterances, then raters within each block, same draws for every system). Wired headphones required; the Huggins 6/6 screen is kept | The owner's headphones could not pass the Huggins screen, and a 60-minute session is hard to recruit for. Short sessions make recruiting easier, and blocking keeps every system compared on the same sentences by the same rater, so pairing survives inside each rater. The cost is recorded: exclusion rules 2 and 3 now rest on 2+2 anchors and 1 duplicate, and raters are nested in blocks rather than crossed. Spec section 9A. Supersedes the D74 order, the manifest_ids[0::10] duplicate rule and both section 9 algorithms |
| D78 | **Single-rater panel accepted as the official 6C result, 2026-10-19** (owner decision). Supersedes D77's ">= 8 valid raters per block" for this cycle; blocks, trials, anchors, duplicates and the Huggins screen are unchanged. **Taken after the pilot data in 6C item 5 was seen**, so it is a post-hoc amendment, not a pre-registered rule, and must be reported as such | Owner judged recruitment not worth it this cycle. Cost recorded, not hidden: the block-nested bootstrap cannot resample raters with one rater, so intervals cover utterance variance only and are narrower than a panel's; the result says how one native listener hears the systems, not Vietnamese listeners in general; any report must say "single rater" next to every MOS. The code still treats each session as a rater (`rater` = session number), which with one person would silently fake 80 raters - the analysis path must key on the person, not the session. Open sub-decisions: Q9 |
| D79 | **VieNeu-TTS v3 Turbo adopted as the Vietnamese TTS, 2026-10-19, by owner decision, without MOS.** Component `tts: vieneu` (`src/onevoice/components/vieneu_tts.py`), config `configs/tts_vi_vieneu.yaml`, extra `tts-vieneu`, env `.venv-tts-vieneu` (`onnxruntime<1.23`). Any of the 25 presets via `voice`; default **`Mai Anh`** (female, Northern, news style). Phase 6C MOS is **paused, not closed**: D69/D77/D78 and the 200 stimuli stay as they are, and VieNeu was never a challenger in them | Owner: "chưa cần quan tâm tới bài MOS, sử dụng vieneu-tts luôn", default Northern because Hanoi is the capital. Not a selection result and must never be reported as one: no naturalness evidence exists for VieNeu or against Piper/MMS/F5. Only evidence is intelligibility on 5 manifest utterances (Whisper-turbo round trip, n=5): default voice WER 0.011 / CER 0.005 vs Piper 0.063/0.042; all 15 Northern presets CER 0.000-0.010, too close to rank on n=5. `Mai Anh` chosen as the standard Hanoi broadcast register, not by score (CER 0.004). Apache-2.0; output carries the library's audio watermark (recorded in provenance invocation). Sampled decoding, not sample-exact (D32) |
| D80 | **ASR vi->en working deployment candidate changed to `nguyenvulebinh/wav2vec2-base-vietnamese-250h` + its 4-gram KenLM, 2026-10-23. This is NOT a D60 selection.** Config `configs/full_vi_en_hybrid_async.yaml` now uses `asr: wav2vec2` with `lm_path`, `pyctcdecode` beam search (`beam_width=100`, alpha 0.5, beta 1.5, no unigrams). Full 200, development classification: WER norm 0.1373, CER norm 0.0838, BLEU 17.51, chrF 50.59, COMET 0.7545, pipeline RTF p50 0.1960 / p95 0.3267, 0 errored/empty. Turbo remains quality reference and fallback: WER 0.0981, CER 0.0517, COMET 0.7602, pipeline RTF p50 0.8795 | wav2vec2 + LM is the only measured arm meeting WER, COMET and pipeline RTF together; it is 4.5x faster than turbo and clears G2. It does not dominate turbo: turbo is better on WER, CER and COMET; wav2vec2 is better on BLEU, chrF and RTF. No paired bootstrap interval was computed, and the +0.0392 WER gap exceeds D60's ~0.018 delta, so wav2vec2 is not non-inferior on WER. The default config change is a deployment choice under G2, not a quality-slot selection; D61 remains UNRESOLVED. The 1.36 GB LM is CC-BY-NC-4.0 (non-commercial MVP); **owner decision 2026-10-23: 1.36 GB is acceptable for candidate status, keep it unpruned — size alone is not a rejection reason.** G3 host measurement 2026-10-23: KenLM open 0.45 s; full-200 peak RSS 1.79 GB, worst one-utterance 2.36 GB; ASR-only RTF p50 0.070. Quantized size is open: the distributed artifact is an unquantized probing binary, no ARPA, no `build_binary`; 0.842 GiB is only the compressed download. Host measurement, not Snapdragon. Full analysis: `docs/investigations/2026-10-22-asr-mt-bakeoff/report.md`; G3 detail: `docs/investigations/2026-10-22-asr-mt-bakeoff/g3-deployment.md` |
| D81 | **BLEU >20 error attribution, full 200, 2026-10-23.** Holding envit5 fixed: pipeline 17.51; MT on full-audio wav2vec2+LM 17.60; MT on VAD-gated pipeline ASR 17.51; MT on gold 23.86. BLEU gap 6.35 = ASR model 6.26 (98.6%) + VAD/streaming 0.09 (1.4%) + pipeline integration 0.00. COMET gap 0.0777 = ASR model 0.0761 (98.0%) + VAD 0.0016 (2.0%) + integration 0.0000. Pipeline hypothesis equals MT-on-pipeline-ASR 200/200 exactly. Oracle replacement: top 14 utterances reach BLEU 20.11; top 20 reach 20.81. Manual tags: 18/20 named entity or number, 2/20 general substitution; 11/20 number/date, 7/20 proper-noun-only. Replacing the 18 named-entity/number utterances with gold-MT reaches 20.53; the 11 number/date utterances reach 19.48; the 7 names-only reach 18.58 | The ASR model, not VAD and not async integration, is the BLEU blocker. WER already passes (0.1373 < 0.15), so aggregate WER reduction is not the right goal; the expensive errors are named entities and numbers. A number-only ITN cannot pass (19.48 oracle), a name-only fix cannot pass (18.58), and a generic beam/alpha-beta sweep is unlikely to fix OOV names. Next is a fixed-22 targeted ASR experiment: number ITN, then a general Vietnamese unigram/lexicon plus limited beam check; stronger ASR only if those fail. Gate: WER down or flat, BLEU up, COMET >= 0.75, pipeline RTF p50 < 0.30. No model/config change yet. Report: `docs/investigations/2026-10-23-bleu20-error-attribution/report.md` |
| D82 | **PhoWhisper CT2 INT8 fixed-22 screen, 2026-10-23.** Converted `vinai/PhoWhisper-large` (BSD-3-Clause, revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`, 1.5 GB CT2 INT8) and `vinai/PhoWhisper-medium` (BSD-3-Clause, revision `55a7e3eb6c906de891f8f06a107754427dd3be79`, 739 MB CT2 INT8); kept Silero VAD, envit5, Piper, references and async pipeline unchanged. Fixed-22 vs wav2vec2 + 4-gram LM baseline (`14.55` BLEU / `49.23` chrF / `0.1366` WER / `0.7781` COMET / `0.1747` RTF): large `17.40` / `53.08` / `0.1046` / `0.7796` / `1.7890`; medium `18.39` / `53.20` / `0.1720` / `0.7874` / `0.9204`. Both 22/22 ok, 0 errored/empty. Per-utterance ASR WER: large 3 worse / 12 better / 7 same; medium 6 worse / 10 better / 6 same. Large passes the 22 BLEU, COMET and WER thresholds but fails RTF, so verdict **speed_fallback**; medium fails WER ≤0.15 and RTF <0.30, so verdict **reject**. No full-200 run at fixed-22 decision time; the owner later approved a quality-only large full 200 (D83). | Large is a real quality signal: it recovers the `1976` year in utterance 1928 and `sachin tendulkar` in 1976 (though `rahul dravid` remains wrong), and improves 12/22 WER, but its pipeline RTF is 10.2× the baseline on this host. Medium is 1.94× faster than large and has the best 22 BLEU/COMET, but WER regresses above both baseline and the 0.15 gate, with repeated dot-tail/repetition hallucinations (1991 WER 0.000→0.467; 1709 0.139→0.556) and named-entity corruption; it is not a safe speed fallback. Neither is deployable under the quality-first Pareto gate, so keep D80 wav2vec2 + LM as working candidate, do not change the default config, and do not treat this as a D60 selection. Fixed-22 is a screen, not a ranking; host timings are not Snapdragon predictions. Next options: speed work for large (now the quality leader but RTF fail; see D83), a targeted number/date ITN on D80 (ceiling 19.48, fallback), or runtime work (C++/ONNX/NPU) before another ASR family. Report: `docs/investigations/2026-10-23-bleu20-step3b-phowhisper/report.md` |
| D83 | **PhoWhisper-large CT2 INT8 full-200 quality-only run, 2026-10-23.** Owner-approved after D82 fixed-22 `speed_fallback`. Same Silero VAD, envit5, Piper and async pipeline; only ASR is `faster_whisper` on `models/phowhisper-large-ct2-int8` (BSD-3-Clause, revision `b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb`). Full 200: 200/200 ok, 0 errored/empty; BLEU `20.26`, chrF `53.42`, WER norm `0.09124`, CER norm `0.06136`, COMET `0.7907`, pipeline RTF p50 `1.078`. Deltas vs D80 full 200 (`17.51` / `50.59` / `0.1373` / `0.0838` / `0.7545` / `0.1960`): `+2.74` BLEU, `+2.83` chrF, `-0.0461` WER, `-0.0224` CER, `+0.0362` COMET, `+0.882` RTF (5.5×). Per-utterance ASR WER: 50 better / 38 worse / 112 same. Full-200 quality point estimates **PASS** (BLEU 20.26 >20, COMET 0.7907 >0.75, WER 0.09124 <0.15, 0 errors/empty); speed gate **FAIL** (RTF 1.078 ≥ 0.60). BLEU absolute 95% CI [18.34, 22.30] includes 20, so the >20 target is not statistically resolved; paired vs D80: BLEU Δ+2.74 [1.45, 4.06], chrF +2.83 [1.78, 3.96], COMET +0.0362 [0.0256, 0.0476] (1000 resamples, 95%). | First arm with a full-200 BLEU point estimate >20; the absolute target is not resolved. It is not deployable on the host because RTF is 5.5× D80, so D80 remains the working deployment candidate, the default config is unchanged, and this is not a D60 selection. The binding constraint is now speed: D84 tested CT2 threading/worker/decode settings and found no gain, so next is a backend/device change (ONNX Runtime Whisper seq2seq, whisper.cpp/GGML, or QNN/NPU); do not promote on host quality alone. Medium was not full-run; D82's 22 WER and repetition failures stand. Report: `docs/investigations/2026-10-23-bleu20-step3b-phowhisper/report.md`; bootstrap: `runs/bakeoff_vi_en/asr_phowhisper_large_full200_pipeline/bootstrap.json` |
| D84 | **PhoWhisper-large CT2 decode-flag speed probe, 2026-10-24.** Fixed-5 ASR-only probe (full source audio, M1 CPU) on `models/phowhisper-large-ct2-int8`, `beam_size=1`, base `num_workers=4`, CT2 defaults. Baseline ASR-only RTF p50 `1.4471` (full-file; inflated vs pipeline VAD-trimmed). Variants: `cpu_threads=6` `1.03x` (noise), `cpu_threads=4` `0.99x`, `cpu_threads=8` `0.36x` (oversubscribe), `num_workers=1` `0.78x`, `without_timestamps=True` `0.75x` (changed token 1692), `temperature=0` `0.63x`, `condition_on_previous_text=False` `0.70x`. All non-`ts0` transcripts byte-identical to base; archived-vs-probe differences were the same 2/5 utterances before and after (full-file-vs-VAD proxy gap). No variant reached the 1.25x screen or 1.5x gate; not promoted to 22; no full-pipeline probe. | CT2 4.8.3 greedy defaults already near-optimal; do not wire decode flags; no source/config edit. Large remains quality-only (D83) and not deployable. Remaining speed path is backend/device work: ONNX Runtime Whisper encoder+decoder, whisper.cpp/GGML, or QNN/NPU. ONNX encoder + pyctcdecode is the D80 CTC path only; not applicable to PhoWhisper seq2seq. D80 remains deployment candidate. Plan: `docs/superpowers/plans/2026-10-24-bleu20-large-speed.md`; probe: `runs/bakeoff_vi_en/phowhisper_large_speed_probe/`. |
| D85 | **Argos Translate `translate-vi_en` 1.9 MT screen, 2026-10-24.** Official package (63 MB compressed; `model.bin` 76.9 MB; code MIT, model CC-BY-4.0; direct vi→en) run directly through CTranslate2 4.8.3 on M1 CPU, beam 4, length penalty 0.2, no Argos SBD. Fixed-22 D80 ASR-source, MT-only: int8 BLEU `14.60` / chrF `45.01` / COMET `0.6903` / MT wall p50 `330 ms`; float32 `14.68` / `44.47` / `0.6878` / `784 ms`; gold-source int8 `19.77` / `52.55` / `0.7837` / `307 ms`. Envit5 ASR-source baseline `14.55` / `49.23` / `0.7781` / `709 ms`; envit5 gold22 `23.98` / `58.11`. | Argos fails the quality-first cascade COMET gate (`0.6903 < 0.75`) despite matching BLEU, so it is a **screen reject as primary MT**. Its clean gold-source output is also lower than envit5; the candidate is less robust to ASR noise, and the output shows semantic substitutions. Keep only as a latency/fallback candidate if the owner relaxes quality; its size, permissive license and direct vi→en support are attractive. No full 200, no source/config change, not a D60 selection. Report, probe, metadata and per-utterance outputs: `runs/bakeoff_vi_en/mt_argos_screen22/`. Archive checklist: `docs/MODEL_TRIAL_ARCHIVE.md`. |

## Assumed claims

Nothing here has been measured by us. Promote a row to **Verified measurements**
only with a linked measurement taken on hardware we control.

| Claim | Value | Source |
|---|---|---|
| Streaming ASR stage latency | 300–400 ms | `deep-research-report.md` |
| MT stage latency | 200–300 ms | `deep-research-report.md` |
| TTS stage latency | 200–300 ms | `deep-research-report.md` |
| Total end-to-speech latency | ≈1.2 s | `deep-research-report.md` |
| INT8 size reduction | ~4× | `deep-research-report.md` |
| INT8 speedup on Hexagon NPU | ~3× | `GPT_suggestion.md` |
| TTS share of RTF | ~62% | `deep-research-report.md` |
| BLEU target, vi→en | ≥ 30 | `GPT_suggestion.md` — justified by citing an **en→vi** model score, so not evidence |
| COMET target | ≥ 0.70 | `GPT_suggestion.md` |
| WER target, vi ASR | ≤ 25% | `GPT_suggestion.md` |
| MOS target | ≥ 3.8 / 5 | `GPT_suggestion.md` |

**Note on INT8:** "4× smaller" and "3× faster" are unrelated claims about
different quantities. Do not merge them into a single figure.

**Source reliability.** `GPT_suggestion.md` attributes MMS to NVIDIA; it is
Meta's. Verify its model table rather than adopting it.

## Model registry (HF-verified)

Queried from the Hugging Face Hub API on 2026-08-01. These are repository facts
— parameter counts and license tags as published — **not** performance
measurements. Quality and latency remain unmeasured; see Assumed claims.

### ASR — all permissive

| Model | Params | License | Note |
|---|---|---|---|
| `openai/whisper-tiny` | 37.8M | apache-2.0 | vi in language tags |
| `openai/whisper-base` | 72.6M | apache-2.0 | |
| `openai/whisper-small` | 241.7M | apache-2.0 | |
| `openai/whisper-large-v3-turbo` | 808.9M | **mit** | Named by no source document — see below |

**Whisper parameter conflict — RESOLVED.** `deep-research-report.md` claimed
~70M for tiny; the Hub says 37.8M. `GPT_suggestion.md`'s 37.3M was
approximately right and the report conflated tiny with base (72.6M). Whisper-Small
is 241.7M, matching the report. All four sizes are Apache-2.0.

**`whisper-large-v3-turbo` is missing from every source document.** It is a
distilled-decoder Whisper built specifically for inference speed, MIT-licensed,
and 808.9M params. Four documents proposing a latency-constrained ASR pipeline
all omit the model designed for that constraint. It may still be too large for
Snapdragon — that is a Phase 5 measurement, not a reason to leave it off the
candidate list.

### MT — available models

| Model | Params | License | Direction |
|---|---|---|---|
| `Helsinki-NLP/opus-mt-vi-en` | — | apache-2.0 | vi→en |
| `Helsinki-NLP/opus-mt-en-vi` | — | apache-2.0 | en→vi |
| `Helsinki-NLP/opus-mt-zh-vi` | — | apache-2.0 | zh→vi |
| `Helsinki-NLP/opus-mt-en-zh` | — | apache-2.0 | en→zh |
| `Helsinki-NLP/opus-mt-zh-en` | — | **cc-by-4.0** | zh→en |
| `Helsinki-NLP/opus-mt-ko-en` | — | apache-2.0 | ko→en |
| `Helsinki-NLP/opus-mt-tc-big-en-ko` | 209.2M | **cc-by-4.0** | en→ko |
| `facebook/m2m100_418M` | 418M | **mit** | many-to-many |
| `facebook/m2m100_1.2B` | 1.2B | mit | many-to-many |
| `facebook/nllb-200-distilled-600M` | 600M | **cc-by-nc-4.0** | many-to-many — see Q1 |

**`Helsinki-NLP/opus-mt-en-ko` does not exist.** `deep-research-report.md` names
it explicitly as part of the Korean routing plan; the Hub returns 404. The real
model is `Helsinki-NLP/opus-mt-tc-big-en-ko` — different repo, different naming
convention, 209.2M params, cc-by-4.0. Fourth fabricated claim traced to that
document.

**Opus-MT licensing is not uniform.** Most are Apache-2.0, but `opus-mt-zh-en`
and `opus-mt-tc-big-en-ko` are CC-BY-4.0. CC-BY is attribution-only and
commercially usable, so this does not block anything — but "Opus-MT is Apache"
is not a safe blanket assumption. Check per repo.

**Asymmetric coverage, noted by no source document:** `opus-mt-zh-vi` exists but
there is **no `opus-mt-vi-zh`**.

### MT routing (D13)

Every leg below is permissively licensed. **This routes around Q1 entirely for
MT** — if the competition forbids CC-BY-NC, the MT stack is unaffected.

| Direction | Route | Models | Licenses |
|---|---|---|---|
| vi→en | direct | `opus-mt-vi-en` | apache-2.0 |
| en→vi | direct | `opus-mt-en-vi` | apache-2.0 |
| zh→vi | direct | `opus-mt-zh-vi` | apache-2.0 |
| vi→zh | pivot via en | `opus-mt-vi-en` + `opus-mt-en-zh` | apache + apache |
| vi→ko | pivot via en | `opus-mt-vi-en` + `opus-mt-tc-big-en-ko` | apache + cc-by-4.0 |
| ko→vi | pivot via en | `opus-mt-ko-en` + `opus-mt-en-vi` | apache + apache |

Six distinct models for full coverage. Costs of the pivot legs, to be measured
rather than assumed:

- **Latency doubles** on pivot directions — two sequential MT forward passes
  against a 200–300 ms budget. This consumes most of D9's 500 ms margin.
- **Errors compound.** A vi→en mistake becomes the input to en→zh. Pivot
  translation typically loses several BLEU against a direct model.
- **Q3 surface doubles.** A revised ASR partial invalidates the English
  intermediate *and* everything downstream of it, rather than one translation.

`facebook/m2m100_418M` (MIT, one model, one hop, all six directions, ~420 MB at
INT8) is the competing candidate for the pivot directions. Phase 5 measures the
two head to head; neither is committed to now.

### TTS — this is where the real gap is

| Model | Params | License | Note |
|---|---|---|---|
| `myshell-ai/MeloTTS-English` | — | mit | 25.0M downloads |
| `myshell-ai/MeloTTS-Chinese` | — | mit | 1.9M downloads |
| `myshell-ai/MeloTTS-Korean` | — | **mit** | 4.0M downloads — **exists**, contradicting the source documents |
| `facebook/mms-tts-vie` | 36.3M | cc-by-nc-4.0 | VITS; the quality Vietnamese option, but NC |
| `facebook/mms-tts-kor` | 36.3M | cc-by-nc-4.0 | Superseded by MeloTTS-Korean for licensing purposes |

**The source documents had the TTS gap backwards.** They state that MeloTTS
covers EN and ZH while Korean requires the CC-BY-NC MMS. In fact
`myshell-ai/MeloTTS-Korean` exists under MIT with 4.0M downloads. Korean is
solved. **Vietnamese is the unsolved language** — there is no MeloTTS-Vietnamese,
and a Hub search for Vietnamese TTS returns either NC licenses
(`facebook/mms-tts-vie` CC-BY-NC-4.0; the F5-TTS-Vietnamese-ViVoice family
CC-BY-NC-SA-4.0) or permissively-licensed models with negligible adoption
(`dinhthuan/index-tts-2-vietnamese`, apache-2.0, 45 downloads;
`thangquang09/parler-tts-vietnamese-v1-stage2`, apache-2.0, 37 downloads;
`multilingual-tts/VITS-OpenBible-Vietnamese`, cc-by-sa-4.0, 21 downloads and
Bible-domain training data).

This matters because D4 makes vi↔en first-class in **both** directions, and
en→vi requires Vietnamese TTS. The primary language direction is the one with no
proven permissive voice.

**HF metadata caveat.** Every `myshell-ai/MeloTTS-*` repo is tagged
`language: ko` regardless of its actual language — MeloTTS-English and
MeloTTS-Chinese included. Do not filter this family by the Hub language tag.

## Verified measurements

**MT component benchmark, vi→en, `fixtures/mini_vi_en/refs.jsonl` (n=3),
2026-08-03.** Measured by `onevoice.mt_benchmark.run_mt_benchmark`, which
pushes each reference's gold `src_text` straight through the MT component
(no ASR/VAD/TTS) and scores the hypothesis against `ref_text` with the
evaluator's own `bleu()`/`chrf()` (D21 — unchanged). **Scope: MT-only, given
perfect ASR — a ceiling on end-to-end quality, not a prediction of it.**
**Sample size: three utterances. This is a smoke-level baseline, not a
corpus result; do not quote it as a model comparison. The real corpus
arrives at Task 8.** No winner is declared — that is Phase 5.

> **Superseded, 2026-08-03 (Task 9).** Re-measured on the 50-utterance
> FLEURS subset, `opus` scores **17.29** BLEU and `m2m100` **17.82** — the
> 25-point gap below is refuted -- but **no ranking replaces it**: Delta=+0.53
> BLEU, 95% CI [-3.71, +5.12], P=0.58, statistically indistinguishable at n=50
> (F3 audit, 2026-08-04). The n=3 figures
> are retained as a record of what a three-sentence sample produces, and
> must not be cited for model comparison. See
> `docs/benchmarks/2026-08-03-task9-end-to-end.md`.

| Backend | Config | BLEU | chrF | Latency p50 / p95 / max (ms) |
|---|---|---|---|---|
| `opus` (direct) | `configs/opus_vi_en.yaml` | 40.92 | 58.83 | 204 / 223 / 223 |
| `pivot` (composite) | `configs/pivot_vi_en.yaml` | 40.92 | 58.83 | 220 / 232 / 232 |
| `m2m100` | `configs/m2m_vi_en.yaml` | 16.02 | 36.03 | 1338 / 5482 / 5482 |
| `nllb` | `configs/nllb_vi_en.yaml` | 37.60 | 42.19 | 795 / 5024 / 5024 |

`pivot` here is `opus-mt-vi-en` (real) composed with `IdentityMT` (no-op) —
see `configs/pivot_vi_en.yaml` for why: the fixture is vi→en, which D13
already routes direct, and no vi→X model in the registry lands back in
English (Opus coverage is asymmetric — no `opus-mt-vi-zh`; the ko route's
first leg is `opus-mt-vi-en` itself, which would be circular). So `pivot`'s
BLEU/chrF equal `opus`'s by construction, and its latency is composition
overhead on top of one real model call, not the two-real-model compounding
cost D13 describes — that needs Task 7's real vi→zh/vi→ko corpus. Raw
hypotheses for all four runs are archived per D28 (not committed — see the
Task 5 report for paths and samples).

**End-to-end benchmark, vi→en, FLEURS 50-utterance subset (seed 0),
2026-08-03.** Measured by `onevoice.benchmark.run_benchmark` on Apple M1 /
Python 3.11.14. Both branches share VAD (`EnergyVAD`), ASR (`whisper-small`)
and TTS (`piper`); only MT differs. Full analysis:
`docs/benchmarks/2026-08-03-task9-end-to-end.md`.

| | Branch A (`opus-mt-vi-en`) | Branch B (`m2m100`) |
|---|---|---|
| MT-only ceiling, gold transcript | 17.29 BLEU / 49.05 chrF | 17.82 / 51.53 |
| End to end, all 50 | 5.26 / 24.05 | 7.10 / 26.02 |
| End to end, 38 non-empty | 6.00 / 30.47 | 7.57 / 32.83 |
| WER / CER | 0.6094 / 0.4666 | 0.6094 / 0.4666 |
| `end_to_speech_ms` p50 | 4774 | 10127 |
| Wall clock | 3.1 min | 7.1 min |

**Scope: baseline. No tuning, no optimisation, no winner declared — that is
Phase 5.** WER/CER are identical across branches because the ASR is identical;
that is a check, not a coincidence. Against the matched ceiling the ASR stage
costs ~11 BLEU on both branches. Latency exceeds the 2000 ms competition
ceiling by 2.4× (A) and 5.1× (B) on unoptimised CPU. Q6 applies to both.

**Finding — silent VAD dropout.** 12 of 50 utterances produced no output while
both runs reported `errored: 0`: `EnergyVAD`'s fixed 0.05 threshold separates
the corpus exactly (fired 0.052–0.245, silent 0.013–0.045). Latency metrics
already exclude them as `unmeasurable: 12`; the quality metrics average them in
as empty hypotheses. Excluding them moves BLEU only 5.26→6.00 (A) and
7.10→7.57 (B), so this is not why the scores are low — but `errored: 0` reads
as a clean run when a quarter of the corpus was dropped. Not fixed under the
Task 9 baseline constraint; carried to the Phase 4 review and bears on Q7.

**Silero VAD vs EnergyVAD, vi->en, FLEURS 50-utterance subset, 2026-08-04.**
Configs differ in the VAD stage alone. Full analysis:
`docs/benchmarks/2026-08-04-task11-silero.md`.

| | `EnergyVAD` | `SileroVAD` |
|---|---|---|
| BLEU / chrF | 5.261 / 24.05 | **9.459** / **39.32** |
| WER / CER | 0.6094 / 0.4666 | **0.2929** / **0.1336** |
| Utterances producing no output | 12 | **0** |
| `end_to_speech_ms` p50 | 4774 | **3962** |
| Latency measured / unmeasurable | 38 / 12 | **50 / 0** |

**Scope: baseline, library defaults, no tuning.** Every metric improves,
latency included.

**Correction to Task 9.** Task 9 concluded the low scores were dominated by
genuine ASR error rather than VAD dropout, reasoning that excluding the 12
empty utterances moved BLEU only 5.26 -> 6.00. That reasoning was flawed: it
tests whether *dropping* mattered and cannot see audio *truncated* on the
utterances that survived. Restricted to the 38 `EnergyVAD` kept, Silero scores
BLEU 10.38 / chrF 40.07 / WER 0.2968 against EnergyVAD's 6.00 / 30.47 / 0.4840
- a 39% WER reduction on utterances reported as fine. Two of the four
"mis-transcriptions" Task 9 cited as evidence of Whisper's Vietnamese weakness
recover completely under Silero. The Task 9 numbers stand as measurements of
that configuration; their attribution does not. ASR now costs ~7 BLEU against
the matched ceiling, not ~11.

**D10 remains certified with `EnergyVAD`.** Measured against the tone fixtures
Silero misses utt-002 entirely and closes 1450-2550 ms early; the fixtures are
tone bursts and it is trained on speech. The margin is recorded rather than
accommodated by widening the tolerance (D43).

**Lookahead: 62 ms measured**, against D42's 500 ms window - 8x headroom, pinned
by test. On the pre-D42 driver this config would have silently truncated the
first ~62 ms of every utterance.

**TTS component benchmark, Vietnamese `src_text`, FLEURS 50-utterance subset,
2026-08-04.** Measured by `onevoice.tts_benchmark.run_tts_benchmark`, which is
backend-agnostic by construction (D37): both backends were given the same input
text, timed the same way and archived identically. Apple M1, CPU.

| Backend | Config | Latency p50 / p95 / max (ms) | Synthesis RTF p50 | Audio produced p50 (s) |
|---|---|---|---|---|
| `piper` (baseline, D23) | `configs/piper_vi.yaml` | **241 / 455 / 538** | **0.034** | 6.46 |
| `mms` (challenger) | `configs/mms_vi.yaml` | 2904 / 6867 / 8844 | 0.293 | 10.51 |

Piper is **12x faster on p50 latency and 7.6x better on RTF**. Both synthesized
all 50 utterances. MMS produces noticeably longer audio for the same text, i.e.
a slower speaking rate; RTF normalises for that and Piper still wins.

**No quality metric is reported and none is implied.** Nothing in this framework
scores synthesis naturalness - that is the Phase 5 MOS panel (Q4). These figures
justify D23's baseline choice on latency and footprint alone. A challenger
displaces Piper only on recorded measured evidence, and none here does.
Waveforms for both backends are archived (D31) so a MOS round needs no
re-inference.

**Phase 5.0 Whisper checkpoint selection, vi->en, FLEURS n=200 seed 0,
2026-08-06.** Classification **selection**. Silero VAD, Opus-MT and Piper held
constant; only the ASR checkpoint varies. Apple M1. Full analysis:
`docs/benchmarks/2026-08-06-phase5-0-whisper-checkpoint-selection.md`.

| Arm | Params | `wer_normalized` | BLEU | chrF | `end_to_speech_ms` p50 | `rtf` |
|---|---|---|---|---|---|---|
| `tiny` | 37.8M | 0.9811 | 0.96 | 21.43 | 2646 | 0.219 |
| `base` | 72.6M | 0.4243 | 3.56 | 30.21 | 3073 | 0.253 |
| `small` (incumbent) | 241.7M | 0.2245 | 8.87 | 39.05 | 5210 | 0.418 |
| **`turbo`** (selected) | 808.9M | **0.0908** | **13.54** | **45.42** | 11211 | **0.934** |

Paired bootstrap, 1000 resamples, Bonferroni 98.33%, against the fixed
incumbent: `tiny` +0.7566 [+0.5152, +1.0575] inferior; `base` +0.1998 [+0.1822,
+0.2189] inferior; `turbo` **-0.1338 [-0.1508, -0.1175] superior**. Confirmatory
BLEU `turbo` - `small` = +4.668 [+3.087, +6.385]. No comparison unresolved; the
n=400 escalation did not trigger.

**Not comparable to Phase 4 figures** - the n=200 draw is not nested in the n=50
draw, pre-registered in the spec. All four arms complete, `errored: 0`,
`empty_outputs: 0`.

**COMET on the Phase 5.0 arms, vi->en, FLEURS n=200, 2026-08-13 (Gate 0, D55).**
Scored by `scripts/score_comet.py` over the archived run manifests -- no model
was re-run (D5). Model `Unbabel/wmt22-comet-da`. 200/200 scored, 0 excluded, in
every arm.

| Arm | COMET | `wer_normalized` (Phase 5.0) |
|---|---|---|
| `tiny` | 0.3986 | 0.9811 |
| `base` | 0.4716 | 0.4243 |
| `small` | 0.5992 | 0.2245 |
| **`turbo`** | **0.7220** | **0.0908** |

**Reproducible:** two independent runs of the turbo arm produced
0.7220347634 both times, bit-identical.

**Confirmatory only -- this does not reopen Phase 5.** COMET is the
competition's other accuracy metric and had never been run in this project, so
D51 was decided without it. Scored now, COMET orders the arms exactly as
`wer_normalized` did, so the competition's own accuracy metric independently
corroborates the selection rather than contradicting it. No selection was
re-made and no measurement was altered.

**ASR vi->en component track, FLEURS n=200, 2026-08-14.** Pre-registered before
scoring; D60 hierarchy applied once to the complete four-challenger matrix.
Primary `wer_normalized`, paired bootstrap, Bonferroni 98.75%, delta=0.0175
(k=1.5 x measured resolution 0.0117). Full analysis:
`docs/benchmarks/2026-08-14-asr-vi-en-track-result.md`.

| Challenger | `wer_normalized` | Delta vs incumbent | 98.75% CI | Verdict |
|---|---|---|---|---|
| `whisper-small` | 0.2245 | +0.1338 | [+0.1170, +0.1512] | INFERIOR |
| `wav2vec2-vi-250h` | 0.1880 | +0.0973 | [+0.0755, +0.1218] | INFERIOR |
| `phowhisper-large` | 0.0946 | +0.0038 | [-0.0128, +0.0213] | **UNRESOLVED** |
| `seamless-m4t-v2` | 0.1063 | +0.0156 | [-0.0032, +0.0360] | **UNRESOLVED** |
| incumbent `whisper-large-v3-turbo` | 0.0908 | - | - | reference |

**STATUS: ASR vi->en UNRESOLVED.** No winner; incumbent retained as **working
reference**; `phowhisper-large` and `seamless-m4t-v2` unresolved; escalation
**pending review** (`docs/investigations/2026-08-14-asr-escalation-review.md`).
**Turbo is NOT claimed to be statistically superior.**

**No component winner. The matrix escalates** - two intervals straddle
delta, so the comparison cannot support a selection at n=200. The incumbent is
retained *pending escalation*, which is "cannot yet decide", **not** "the
incumbent won". Escalation per protocol section 8 is a full re-run of all arms
at n=400, never pooling.

All arms 200/200, errored 0, empty 0. **Scope: vi->en ASR component only.** Not
the final ASR, not the pipeline, not deployment, and nothing about the reverse
directions, which remain unevaluable (D59).

**Second confirmation that `wer_normalized` must stay primary.** On raw WER,
`phowhisper-large` (0.124) and `wav2vec2` (0.190) both appear to beat the
incumbent (0.218); on the normalised primary the ordering is entirely different.
Raw WER measures Vietnamese text convention, not transcription accuracy.

**ASR vi->en wav2vec2 + 4-gram LM pipeline, FLEURS n=200, 2026-10-23.**
Development classification; Silero VAD, envit5 MT and Piper English held
constant. Full analysis:
`docs/investigations/2026-10-22-asr-mt-bakeoff/report.md`. This is a G2 gate,
not a D60 selection.

| Metric | Turbo + envit5 | wav2vec2 + LM + envit5 |
|---|---:|---:|
| `wer_normalized` | **0.0981** | 0.1373 |
| `cer_normalized` | **0.0517** | 0.0838 |
| BLEU | 17.14 | **17.51** |
| chrF | 50.41 | **50.59** |
| COMET | **0.7602** | 0.7545 |
| `pipeline_rtf` p50 / p95 | 0.8795 / 3.5716 | **0.1960 / 0.3267** |
| ASR stage RTF p50 / p95 | 0.6988 / 2.7028 | **0.0585 / 0.1074** |
| Errored / empty | 0 / 0 | 0 / 0 |

wav2vec2 + LM meets WER < 0.15, COMET > 0.75 and pipeline RTF p50 < 0.6. It
does not meet BLEU > 20. Turbo remains the quality reference; wav2vec2 + LM is
the deployable working candidate. **No paired bootstrap interval was computed**,
so the point estimates are not a ranking. The LM is 1.36 GB, CC-BY-NC-4.0.
G3 host measurement (Apple M1, 16 GiB; not Snapdragon): KenLM open 0.45 s,
component init 6.35–7.03 s warm / 11.8 s cold, full-200 peak RSS 1.79 GB,
worst one-utterance 2.36 GB, ASR-only RTF p50 0.070. Quantized size is **open**:
the distributed LM is an unquantized probing binary with no ARPA, so
`trie -q 8` cannot be built or measured; 0.842 GiB is only the zip. Detail:
`docs/investigations/2026-10-22-asr-mt-bakeoff/g3-deployment.md`.

## Open questions

| # | Question | Blocks | Status |
|---|---|---|---|
| Q1 | Do the competition rules permit CC-BY-NC models? | Phase 4 | **RESOLVED 2026-08-03** — see below |
| Q2 | **Vietnamese** TTS under a permissive license — not Korean, as the source documents claimed. `myshell-ai/MeloTTS-Korean` is MIT with 4.0M downloads, so Korean is solved. Vietnamese has only NC options at quality (`facebook/mms-tts-vie`) or permissive options with negligible adoption (see Model registry). D4 makes en→vi first-class, so this blocks the primary direction. | Phase 4 | Open — no longer licence-blocked after Q1; now a quality selection question, pre-registered 2026-09-03 (D69): `piper-vi-vais1000` vs `mms-tts-vie` and `f5`, awaiting the 6C panel. **2026-10-19: answered in practice by owner decision D79 (VieNeu-TTS v3 Turbo, Apache-2.0), not by measurement; the quality question stays open while 6C is paused** |
| Q3 | Streaming MT revision policy: a partial transcript may change after its translation has been spoken. Commit and tolerate, or retract mid-utterance? Unnamed by every source document. | Phase 6 | Open |
| Q4 | MOS panel recruitment — native vi/en/zh/ko speakers. Scheduling work, not compute work; no document allocates time for it. | Phase 6C | Open — Vietnamese panel is the next step; rating procedure frozen in D69, delivery and headphone screen in D74, panel tool built. Under D77 that means >= 8 valid 20-trial sessions per block (>= 64; 80 built), wired headphones required. Recruitment not started. en/zh/ko panels not needed this cycle. **2026-10-19: superseded for vi by D78 - one rater (the owner's chosen listener) is the panel; no recruitment planned** |
| Q5 | Noise-robustness fixtures. Target environment is noisy worksites; every dataset discussed so far is clean speech. | Phase 5 | Open |
| Q6 | RTF and end-to-speech latency are blind to per-frame work done inside `push()`. A component doing real streaming compute reports near-zero cost. | Phase 4 | **Partially closed 2026-08-04 by D45.** `pipeline_rtf` measures the whole utterance and `streaming_cost` reports the **paired** per-utterance gap with its `paired_n`. The two RTF percentiles are descriptive only and must not be subtracted - an utterance whose detector never fired has a `pipeline_rtf` but no `rtf`, so they describe different populations. Remaining: the cost is not attributed to a *stage* - per-component in-`push()` instrumentation is deferred to Phase 6 with the device port, since it cannot be back-derived from archives. **Stage attribution added 2026-08-19** (`7642321`): `asr_first_partial`, `asr_final`, `mt_done` stamps and a `tts_synthesis_s` duration, observational only and consumed by no metric or selection rule; archived Phase 5.0 metrics recompute identically |
| Q7 | ASR gating uses the current frame's speaking flag rather than the VAD event's `t_audio`, so a lookahead VAD loses the leading frames of every utterance. | Phase 4 | **Closed 2026-08-04 by D42.** `LookbackBuffer` replays from `t_audio`; underrun raises. Verified a no-op for `EnergyVAD` at both unit and benchmark level, so Task 9 baselines stand. Unblocks Silero (Task 11) |
| Q8 | **RESOLVED 2026-08-15 by D66.** **The selection hierarchy short-circuited on any `unresolved` comparison, so a candidate that cannot win blocks one that already has.** On vi->en, `opus` and `m2m100` are both worse than the incumbent and unresolved only about whether their regression exceeds delta, while `envit5` and `madlad` are superior with intervals clear of the incumbent. Escalation cannot fix it: `opus` resolves at n~216 but `m2m100`'s regression is *smaller* than delta, so it can never be `inferior` and needs n~25,400 to be `non_inferior` against a 200-utterance corpus. Under the rule as written vi->en can never produce a selection at any feasible n | MT selection | **RESOLVED 2026-08-15 by D66** - amended generically and committed before re-scoring. vi->zh and vi->ko conclusions are unaffected; only vi->en is re-scored |
| Q9 | **How the single-rater panel (D78) is analysed and which data counts.** (a) **Answered 2026-10-22: the returned `test_survey/` was pilot data and has been deleted, so it is not the official dataset.** (b) Anchor rule: that rater scored natural speech 2.59, below `f5` 2.98; the per-session anchor rule excludes 78/80 sessions and the 1.5 anchor gate fails (separation 0.41). Keep the rule (then the result is "anchor gate failed, no verdict"), find the cause and re-rate, or amend - amending after seeing data is a second post-hoc change. (c) Interval method with one rater: utterance-only bootstrap within blocks. (d) Each of the 80 sessions rated a block 10 times over - pool repeats per block or keep first session per block only. | Phase 6C | Open - owner decides before any code is written |

### Q1 — RESOLVED: model licences are not a selection gate

**Ruling, 2026-08-03.** For this project the Business Viability criterion is
judged on **commercial feasibility and cost**, not on deployment licensing.
Model licences are therefore *not* a gating criterion for Phase 4 model
selection. Recorded as D22.

**Consequences.** The CC-BY-NC shortlist is back in play:
`facebook/nllb-200-distilled-600M` for MT and `facebook/mms-tts-vie` /
`facebook/mms-tts-kor` for TTS are all selectable on merit. Nothing is
*forced* by this — the permissive routing recorded in D13 remains a valid and
competing option, and D13's premise that MT is insulated from licensing is now
simply moot rather than load-bearing.

**What this does not change.** Selection is still decided by measurement
(BLEU / COMET / WER / latency / RTF / size), not by licence. Licences remain
recorded in the Model registry as facts, because they still bear on cost and on
whatever the competition ultimately asks for — they just no longer eliminate a
candidate before it is measured.

**What it changes about Q2.** Vietnamese TTS is no longer blocked on finding a
permissive option: `facebook/mms-tts-vie` (36.3M VITS) is now a first-class
candidate. Q2 survives as a quality and latency question rather than a
licensing one.

### Q6 — RTF and latency measurement semantics

**Problem.** Both timing metrics under-report, silently. Measured on the Phase 3
dummy pipeline by injecting artificial delay:

| Injected work | `end_to_speech_ms` | `rtf` |
|---|---|---|
| none (baseline) | 0.021 | 0.00001 |
| 150 ms in `asr.flush()` | 154.412 | 0.07721 |
| ~126 ms spread across `asr.push()` | **0.036** | **0.00002** |

126 ms of real compute reports as 0.036 ms. A streaming ASR running 5× slower
than real time would report `rtf ≈ 0`.

**Root cause.** `end_to_speech_ms` spans `speech_end → tts_first_audio` and
`rtf` spans `speech_end → tts_done`. Both windows open only *after* the last
speech frame, so all per-frame work done during the frame loop falls outside
them. Batch models are unaffected — D6 has them do their work in `flush()`,
which is inside the window — which is why the Phase 3 dummy run and any
Whisper-shaped wrapper look correct while a genuinely streaming component does
not. The driver already stamps `stream_start` (`pipeline.py`) and nothing reads
it.

**Second-order problem.** Because the driver consumes audio as fast as it can,
`end_to_speech_ms` measures *compute* latency after the falling edge. It
excludes VAD hangover, which a live system pays in wall-clock time. Against the
competition's 2000 ms ceiling this systematically under-reports.

**Design implications.** The obvious fix — redefine `rtf` as
`stream_start → tts_done` — cannot simply be applied. `fixtures/mini_vi_en/run.jsonl`
is a hand-written golden fixture with no `stream_start` stamp, and Phase 1's
evaluator tests assert exact scores against it. Redefining the span would make
the golden run unmeasurable and break those tests. Any change also alters the
metric contract, which D21 freezes for the duration of Phase 4.

**Proposed direction, not yet decided.** Prefer `stream_start` when present and
fall back to `speech_end` when it is absent, so old manifests remain scoreable;
report which span was used in the metric's `detail` so a number can never be
compared across two different definitions unknowingly. Separately, consider
reporting compute-RTF and wall-clock-latency as distinct metrics rather than
conflating them. Decide before Phase 5 benchmarking, since Phase 5 is where
these numbers start driving model selection.

### Q7 — Streaming ASR gating against VAD timestamps

**Problem.** The driver feeds a frame to ASR only if the VAD has already
declared speech open at the moment that frame is pushed. A VAD with lookahead —
Silero confirms an onset retroactively, after observing subsequent frames —
emits `speech_start` some frames *after* the audio it refers to. By then the
driver has already discarded those frames, so ASR loses the beginning of every
utterance.

**Root cause.** `run_utterance` tracks a boolean `speaking` flag updated as
events arrive, and gates `asr.push(chunk)` on that flag's value for the current
frame. The `VadEvent` carries a `t_audio` naming the audio time the boundary
actually occurred at, and the driver ignores it.

**Why Phase 3 did not surface it.** `EnergyVAD` is a zero-lookahead threshold
detector: it declares speech on the same frame the energy rises, so the flag and
the timestamp agree and the bug is invisible. Every realistic VAD has lookahead.

**Design implications.** Correcting this needs a lookback buffer, not a
one-line change: the driver must retain recent frames and, on receiving
`speech_start` at `t_audio`, feed ASR the frames from that audio time forward.
Buffer depth bounds the lookahead a VAD may have, which is a real constraint to
state rather than discover. It also interacts with the D10 boundary assertion —
detected boundaries currently land within 24 ms of declared, and a lookahead VAD
will shift that margin.

**Proposed direction, not yet decided.** Keep a ring buffer of frames sized to a
declared maximum lookahead; on `speech_start`, replay from `t_audio`. Assert the
buffer depth explicitly so a VAD exceeding it fails loudly rather than silently
truncating. Resolve before integrating a real VAD in Phase 4.
