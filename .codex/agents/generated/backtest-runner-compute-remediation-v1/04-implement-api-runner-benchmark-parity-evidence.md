---
prompt_name: backtest_runner_compute_remediation_v1_04_api_runner_benchmark_parity_memory_evidence
repo: roehub.com
branch: current
scope: "P0: add and run Mac Studio benchmark evidence proving the UI/API runner layer executes canonical compute, disposable child memory release, and bounded lazy cache-hit behavior."

language:
  implementation: python_benchmark_docs
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "canonical benchmark evidence policy"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_summary.md
      why: "accepted May 2 reference summary"
    - path: .codex/agents/generated/backtest-runner-compute-remediation-v1/03-implement-disposable-lazy-trades-and-cache-hit-memory.md
      why: "lazy child/cache-hit memory acceptance source"
  task_entrypoints:
    - path: scripts/backtest/run_iteration_8_execution_sizing_benchmark.py
      why: "reference benchmark parameters and summary/result writer"
      inspect_symbols:
        - BENCHMARK_TOP_K
        - REQUEST_TOP_N
        - main
        - _build_payload
    - path: scripts/backtest/validate_benchmark_accounting.py
      why: "local accounting validation for benchmark metadata"
      inspect_symbols:
        - main
    - path: apps/api/routes/backtests.py
      why: "public jobs API route used by UI-created jobs"
      inspect_symbols:
        - create_job
        - get_job
        - list_top_variants
    - path: apps/worker/backtest_job_runner/main/main.py
      why: "runner entrypoint under test"
      inspect_symbols:
        - main
  conditional_bundles:
    canonical_json_and_previous_results:
      read_when: "before implementing benchmark comparison and evidence schema"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/local_accounting_validation.json
    runner_smoke_helpers:
      read_when: "if existing production runner smoke can be reused"
      paths:
        - scripts/backtest/run_backtest_job_runner_prod_smoke.py
        - scripts/backtest/run_stage_8_5_create_path_load_smoke.py
        - scripts/macos/smoke_prod.sh
    lazy_cache_memory_surface:
      read_when: "when adding cache miss/hit memory evidence and bounded API cache-hit checks"
      paths:
        - src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
        - src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py
        - src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
    persistence_and_status:
      read_when: "if benchmark needs DB polling, top variants, or lazy detail verification"
      paths:
        - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
        - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py
        - src/trading/contexts/backtest/application/use_cases/backtest_jobs.py
    performance_hot_path:
      read_when: "if benchmark exposes a regression in ordinal/Numba/top-N implementation"
      paths:
        - src/trading/contexts/backtest/application/services/v2/combo_planning.py
        - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "runtime contract or fixture fields are unclear"
    - path: docs/architecture/backtest/backtest-job-runner-production-plan-v1.md
      read_when: "runner smoke or metrics acceptance is unclear"

style_references:
  - .codex/promt_template.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  macstudio_acceptance_required: true
  same_may_2_parameter_set_required: true
  heaviest_140s_job_excluded_from_required_benchmarks: true
  benchmark_must_use_all_other_reference_jobs: true
  new_benchmark_iteration_folder_required: true
  benchmark_results_json_required: true
  benchmark_summary_md_required: true
  local_accounting_validation_required: true
  api_runner_layer_must_be_measured: true
  canonical_compute_parity_required: true
  mixed_light_heavy_scheduling_evidence_required: true
  preflight_heavy_classification_evidence_required: true
  light_candidate_refinement_evidence_required: true
  heavy_fifo_evidence_required: true
  light_parallelism_evidence_required: true
  full_job_child_exit_memory_release_evidence_required: true
  lazy_cache_miss_child_exit_memory_release_evidence_required: true
  api_cache_hit_bounded_memory_evidence_required: true
  parent_retained_rss_delta_evidence_required: true
  vmmap_physical_footprint_evidence_required: true
  legacy_production_path_absence_evidence_required: true
  dead_code_audit_required: true
  docs_drift_audit_required: true
  active_docs_must_not_describe_removed_paths_as_current: true
  performance_claims_require_comparable_evidence: true

task_toggles:
  implementation_changes_allowed: true
  add_benchmark_script_if_missing: true
  write_docs_evidence: true
  run_macstudio_benchmark: true
  run_mixed_scheduler_smoke: true
  fix_introduced_benchmark_failures_allowed: true
  publish_after_success: false

skill_routing:
  - skill: backend-performance-evidence
    use_when: "designing, running, comparing, or reporting benchmark evidence"
    timing: before implementation and during verification
    reason: "acceptance depends on comparable Mac Studio benchmark evidence"
  - skill: root-cause-debugging
    use_when: "API-runner benchmark fails, hangs, diverges from canonical results, or underuses CPU"
    timing: if blocker
    reason: "benchmark failure must be localized instead of worked around"
  - skill: numba
    use_when: "benchmark reveals Numba thread, JIT warmup, or parallel diagnostics issues"
    timing: if blocker
    reason: "Mac Studio CPU evidence depends on compiled path behavior"
  - skill: backend-quality-gates
    use_when: "running docs, benchmark-accounting, tests, lint, or type gates"
    timing: during verification
    reason: "benchmark scripts and docs must stay reviewable"
  - skill: contract-impact-analysis
    use_when: "benchmark script requires API, DTO, telemetry, persistence, or config changes"
    timing: before final report
    reason: "evidence changes must not silently alter public runtime contracts"

target_envs:
  - local-dev
  - mac-studio
  - production-like

required_literals:
  - "BTCUSDT"
  - "15m"
  - "2026-05-02_iteration_8_execution_sizing_completion"
  - "benchmark_results.json"
  - "benchmark_summary.md"
  - "local_accounting_validation.json"
  - "REQUEST_TOP_N = 100"
  - "BENCHMARK_TOP_K = 5"
  - "exclude_heaviest_140s_job"
  - "queued -> running -> succeeded"
  - "queued -> running -> completed"
  - "scheduling_class"
  - "light_candidate"
  - "light"
  - "heavy"
  - "estimated_combinations_upper_bound"
  - "ROEHUB_BACKTEST_LIGHT_CONCURRENCY"
  - "ROEHUB_BACKTEST_HEAVY_CONCURRENCY"
  - "lazy_trades_compute"
  - "lazy_trades_cache_hit"
  - "retained_rss_delta"
  - "vmmap"
  - "physical footprint"
  - "legacy path absence"
  - "dead code audit"
  - "docs drift audit"
  - "historical_prefix_compatible"
  - "MacStudioDaniil"

non_goals:
  - "Do not invent new benchmark parameters as primary acceptance."
  - "Do not accept local benchmark evidence as production acceptance."
  - "Do not hide non-comparable service overhead inside canonical stage ratios."
  - "Do not claim success if API-runner result parity fails."
  - "Do not claim light parallelism is safe without CPU/RSS/API-latency evidence."
  - "Do not claim heavy FIFO if older heavy jobs can be bypassed by newer heavy jobs."
  - "Do not process old queued jobs as primary benchmark evidence."
  - "Do not include the single heaviest reference job that runs 140+ seconds in any required benchmark, smoke, or acceptance loop."
  - "Do not claim memory release from `gc.collect()` alone; child process exit evidence is required."
  - "Do not accept benchmark success if old in-process/full-detail production paths remain reachable."
  - "Do not accept benchmark success if active docs still describe removed paths as current production behavior."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Benchmark fixture"
    - "API-runner path"
    - "Mac Studio results"
    - "Parity"
    - "Performance"
    - "Memory release"
    - "Lazy cache-hit memory"
    - "Legacy path absence"
    - "Dead code audit"
    - "Docs drift audit"
    - "Artifacts"
    - "Contract impact"
    - "Risks"
    - "Next prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/services/v2"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run ruff check scripts/backtest apps/api apps/worker src/trading/contexts/backtest tests/unit"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<new_iteration_dir>/local_accounting_validation.json"
    expect: "passes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes"

expected_primary_touches:
  - "scripts/backtest/run_api_runner_benchmark_parity.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_<n>_api_runner_compute_memory_parity/benchmark_results.json"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_<n>_api_runner_compute_memory_parity/benchmark_summary.md"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_<n>_api_runner_compute_memory_parity/local_accounting_validation.json"

possible_secondary_touches:
  - "scripts/backtest/run_backtest_job_runner_prod_smoke.py"
  - "scripts/macos/smoke_prod.sh"
  - "tests/unit/apps/worker/backtest_job_runner/**"
  - "tests/unit/contexts/backtest/application/services/v2/**"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"

safety_notes:
  - "Benchmark evidence must record commit, host, hardware, artifact hashes, request hash, warmup policy, Numba threads, and cache state."
  - "Benchmark evidence must record the excluded heaviest 140+ second job identifier/name and reason."
  - "If Mac Studio cannot run, stop and report blocker; do not replace acceptance with local evidence."
  - "Keep service-only overhead separate from canonical notebook-compatible stage comparisons."
---

# Task

Create and run the benchmark evidence that proves the new API/runner layer executes the same canonical compute as the accepted May 2 benchmark while also proving disposable compute memory release and bounded lazy cache-hit behavior.

Done means:

- a new benchmark script or reusable harness exercises the public API job path and runner path, not only direct in-process service calls;
- it uses the same May 2 benchmark parameter set for primary acceptance: `BTCUSDT`, `15m`, `REQUEST_TOP_N = 100`, `BENCHMARK_TOP_K = 5`, and the same fixture semantics from Iteration 8;
- it runs all required reference jobs except the single heaviest job that takes 140+ seconds, and records that exclusion explicitly;
- it writes a new directory under `docs/architecture/backtest/benchmark_iterations/`;
- the new directory contains `benchmark_results.json`, `benchmark_summary.md`, and `local_accounting_validation.json`;
- Mac Studio evidence records parity, stage timings, CPU/thread behavior, RSS, service-only overhead, and pass/fail;
- Mac Studio evidence records full-job child peak RSS, parent retained RSS after child exit, `vmmap`/physical footprint where available, and pass/fail for memory release;
- Mac Studio evidence records lazy cache miss child memory release and API cache-hit retained memory without full detail loading;
- the job path reaches `queued -> running -> succeeded` and returns the expected result shape.
- mixed scheduling evidence proves bounded light parallelism and heavy FIFO behavior on Mac Studio.

## Context / Current State

The repository has accepted benchmark evidence through `2026-05-02_iteration_8_execution_sizing_completion`. That evidence is the reference for execution/sizing semantics and benchmark metadata. The new production concern is not just the compute kernel; it is whether a UI/API-created job uses the same compute semantics through the runner layer.

Acceptance must not use a new arbitrary workload as the primary proof. The primary parity gate must reuse the accepted May 2 benchmark parameters and record the same categories of metrics, with one explicit exception: remove the single heaviest 140+ second reference job from every required benchmark, smoke, and acceptance loop. All other May 2 reference jobs remain required.

Scheduling acceptance must be separate from canonical compute parity:

- canonical parity benchmark proves the API/runner path computes the same result;
- scheduler smoke proves multiple UI/API-created jobs behave correctly when classified as `light` or `heavy`;
- scheduler smoke proves obvious heavy jobs are classified as `heavy` at preflight, before prepare/exact compute;
- scheduler smoke proves possible light jobs start as `light_candidate` and are confirmed/promoted after prepare;
- light jobs may run concurrently only within configured cap;
- heavy jobs must be claimed FIFO and must not overlap with another heavy job;
- v1 must not overlap light jobs with an active heavy job unless an explicit benchmark flag and evidence prove safe host sharing.

Memory acceptance is separate from canonical compute parity:

- full-job compute must run in a disposable child process, and memory release is proven after child exit;
- lazy trades cache miss/materialization must run in a disposable child process, and memory release is proven after child exit;
- lazy trades cache hit must be proven through API read paths that do not load the full trades detail payload into API memory;
- RSS alone is not enough on macOS; record retained RSS delta plus `vmmap` or physical-footprint evidence when available.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Reuse the Iteration 8 parameter set and benchmark metadata semantics.
- Exclude the single heaviest 140+ second reference job from all required benchmark, smoke, and acceptance checks.
- Use all other May 2 reference jobs; do not replace them with arbitrary smaller fixtures for primary acceptance.
- Record the excluded job identifier/name, expected/observed runtime reason, and exclusion rationale in both JSON and Markdown.
- Create a new benchmark iteration folder using the repository naming policy.
- Write `benchmark_results.json` and `benchmark_summary.md` with the same core evidence categories as prior iterations.
- Run `validate_benchmark_accounting.py` and write `local_accounting_validation.json`.
- Record canonical notebook-compatible stages separately from service-only overhead.
- Exercise API create/status/top-result path plus runner execution.
- Record job state transition evidence: `queued -> running -> succeeded`.
- Record full-job child process lifecycle evidence: child pid, start time, exit time, exit code, peak RSS if available, parent RSS before/after, retained RSS delta, and `vmmap`/physical footprint if available.
- Record parent metrics responsiveness while full-job children are running and after they exit.
- Create a mixed scheduler smoke with at least two `light` jobs and two `heavy` jobs, using controlled fixtures small enough to finish.
- Include one obvious heavy request whose preflight upper bound exceeds light thresholds and prove it enters the heavy lane before prepare/exact scoring.
- Include one `light_candidate` request and prove post-prepare refinement confirms `light`.
- Include one controlled `light_candidate` promotion case if feasible; otherwise document why the promotion path is covered only by unit/integration tests.
- Prove no more than configured `ROEHUB_BACKTEST_LIGHT_CONCURRENCY` light jobs run concurrently.
- Prove heavy jobs run FIFO by `created_at ASC, job_id ASC`.
- Prove light jobs do not starve an older queued heavy job.
- Capture API responsiveness while light jobs are running concurrently.
- Trigger lazy detail cache miss for a benchmark top variant, prove it reaches `queued -> running -> completed`, and capture lazy child memory release after exit.
- Trigger lazy detail cache hit for the same variant and prove API retained memory is bounded and does not scale with full trades payload size.
- Prove cache-hit detail/stat/page/CSV paths use bounded/chunked/streaming readers rather than full payload read.
- Record legacy-path absence evidence: production runner parent must not construct the full compute graph, public API cache-hit endpoints must not use full-detail cache loads, and large-grid production routing must not use Python `itertools.product`.
- Add a dead-code audit section to benchmark artifacts: list old helpers retained as child-only/test-only/reference-only/migration-only, and list old production paths removed or replaced.
- Add a docs drift audit section to benchmark artifacts: list updated docs, remaining historical references, and any remaining active-doc blockers.
- Record artifact manifest hash, hit-times manifest hash if applicable, request hash, engine/config hash, host, commit, Python, Numba, `NUMBA_NUM_THREADS`, and warmup policy.
- Compare result parity against the accepted reference values from benchmark docs.
- Run on Mac Studio for acceptance. Local runs can only be developer evidence.

## Requirements (Should)

- Include secondary load profiles only if they are not the excluded 140+ second heaviest reference job and runtime budget allows.
- Capture parent metrics responsiveness during compute.
- Capture active child counts by `scheduling_class` if metrics are available.
- Capture CPU saturation evidence with process/thread details, not just wall time.
- Include lazy detail cache miss/hit memory checks as required acceptance after prompt 03 lands.

## Requirements (Nice-to-have)

- Make the benchmark script support `--smoke-only`, `--out-dir`, and `--canonical-json`.
- Emit a compact operator command block in `benchmark_summary.md` for reproduction.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. only the conditional bundle(s) required by touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- changed contracts are identified,
- touched files are bounded,
- acceptance criteria are implementable without ambiguity,
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for blockers, failing quality gates, unclear contracts, benchmark threshold conflicts, or architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules, benchmark policy, accepted May 2 summary;
- `task_entrypoints`: benchmark runner, accounting validator, API route, runner entrypoint;
- `conditional_bundles`: read only when the stated condition applies;
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation and during verification; owns benchmark comparability.
- `root-cause-debugging`: use if parity, progress, CPU, or service state fails.
- `numba`: use if thread/JIT/parallel diagnostics are needed.
- `backend-quality-gates`: use for test/lint/type/docs gates.
- `contract-impact-analysis`: use if script work requires runtime contract changes.

1. Reconstruct the Iteration 8 benchmark fixture and required metadata from existing docs/scripts.
2. Design the API-runner benchmark flow and identify how it waits for job state and reads top results.
3. Implement or update the benchmark script with deterministic output.
4. Add mixed scheduler smoke cases for preflight heavy classification, light-candidate refinement, bounded light parallelism, and heavy FIFO.
5. Add memory-release evidence capture for full-job child exit, lazy cache-miss child exit, and API cache-hit bounded reads.
6. Run local accounting/static checks.
7. Run the benchmark on Mac Studio and write the new evidence directory.
8. Compare parity and stage results against the May 2 accepted reference excluding only the recorded 140+ second heaviest job.
9. Report scheduler and memory-release evidence separately from canonical compute parity.
10. Stop on blocker if Mac Studio evidence is unavailable or non-comparable.

# Acceptance criteria (Definition of Done)

- New benchmark folder exists under `docs/architecture/backtest/benchmark_iterations/`.
- `benchmark_results.json` contains structured machine-readable evidence.
- `benchmark_summary.md` contains human-readable pass/fail, fixture, environment, warmup, runtime stages, service-only overhead, memory cleanup, contract coverage, correctness, and decision.
- The excluded 140+ second heaviest job is explicitly listed in `benchmark_results.json` and `benchmark_summary.md`; no required check runs it.
- `local_accounting_validation.json` exists and passes.
- Mac Studio benchmark records host/commit/config/artifact hashes and Numba thread settings.
- API-runner path is proven with `queued -> running -> succeeded`.
- Full-job child process exits after each benchmark job, and parent retained memory remains within the accepted threshold.
- Result parity against May 2 reference is explicitly pass/fail.
- Obvious heavy jobs are proven to enter the heavy lane from preflight classification.
- `light_candidate` jobs are proven to be refined after prepare before exact scoring.
- Mixed scheduler smoke proves `light` concurrency cap, `heavy` FIFO, and no starvation of queued heavy jobs.
- Lazy detail cache miss is proven with `queued -> running -> completed` and child memory release after exit.
- Lazy detail cache hit is proven without full trades-detail load in API memory.
- Memory release evidence includes retained RSS delta and `vmmap`/physical footprint where available.
- Legacy path absence evidence is recorded for full-job parent wiring, lazy cache-hit API paths, and large-grid Cartesian generation.
- Retained old helpers are explicitly classified as child-only, test-only, direct-benchmark-only, migration-only, or removed.
- Docs drift audit proves active docs no longer describe removed paths as current production behavior.
- The benchmark records whether light/heavy overlap is disabled or explicitly benchmark-enabled.
- Performance comparisons respect benchmark README comparability rules.

# Implementation constraints

## Determinism & ordering

- Keep output JSON stable and sorted where practical.
- Preserve benchmark stage names from the active benchmark policy.
- Do not compare non-equivalent stage boundaries.
- Record scheduling evidence as service/runtime behavior, not canonical notebook stage timing.

## API / contracts

- Benchmark scripts should not require public API breaking changes.
- Any new test-only or operator-only endpoint must be rejected unless explicitly approved.

## Evidence

- Mac Studio evidence is acceptance.
- Local evidence is developer evidence.
- Missing artifacts/config are blockers, not success.

# Files to indicate (expected touched areas)

Expected primary touches:

- `scripts/backtest/run_api_runner_benchmark_parity.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_<n>_api_runner_compute_memory_parity/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_<n>_api_runner_compute_memory_parity/benchmark_summary.md`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_<n>_api_runner_compute_memory_parity/local_accounting_validation.json`

Possible secondary touches:

- `scripts/backtest/run_backtest_job_runner_prod_smoke.py`
- `scripts/macos/smoke_prod.sh`
- `tests/unit/apps/worker/backtest_job_runner/**`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`

# Non-goals

- Do not invent a new primary acceptance workload.
- Do not accept old queued jobs as benchmark evidence.
- Do not hide service-only overhead inside notebook-compatible stage ratios.
- Do not turn scheduler smoke into the primary canonical compute benchmark.
- Do not run the single heaviest 140+ second reference job in required benchmark, smoke, or acceptance checks.
- Do not claim memory release from `gc.collect()` alone.
- Do not accept benchmark success if the old production in-process compute, old full-detail cache loader, or old large-grid Cartesian iterator remains reachable.
- Do not accept benchmark success if stale docs still describe those removed paths as active/current production behavior.
- Do not publish/deploy in this prompt unless explicitly instructed.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/worker/backtest_job_runner tests/unit/contexts/backtest/application/services/v2`
- `uv run ruff check scripts/backtest apps/api apps/worker src/trading/contexts/backtest tests/unit`
- `uv run pyright`
- `uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<new_iteration_dir>/local_accounting_validation.json`
- `uv run python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Write the final report in Russian with these sections:

- `Intent`
- `Benchmark fixture`
- `API-runner path`
- `Mac Studio results`
- `Parity`
- `Performance`
- `Memory release`
- `Lazy cache-hit memory`
- `Artifacts`
- `Contract impact`
- `Risks`
- `Next prompt`

Include exact benchmark folder path, command lines, commit SHA, Mac Studio host, and pass/fail decision.
