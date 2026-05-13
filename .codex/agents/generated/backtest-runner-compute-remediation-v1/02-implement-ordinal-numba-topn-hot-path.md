---
prompt_name: backtest_runner_compute_remediation_v1_02_ordinal_numba_topn_hot_path
repo: roehub.com
branch: current
scope: "P0: restore full-host compute behavior for large UI-created backtest jobs by replacing Python Cartesian chunking with ordinal/Numba streaming and fixing production `top_n=50`."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark comparability and evidence policy"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "latest compact state if present"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "current Python Cartesian chunker and combo planning telemetry"
      inspect_symbols:
        - iter_combo_chunks
        - BacktestComboPlanningService.execute
        - cartesian_combo_count
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "no-risk exact scoring, heap, top_n, and N-arity backend"
      inspect_symbols:
        - BacktestNoRiskExactScoringService
        - _iter_selected_candidate_batches
        - event_segments_n_no_risk
        - BacktestNoRiskExactConfig
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "risk-on exact scoring must preserve compatible behavior"
      inspect_symbols:
        - BacktestTpSlExactScoringService
        - tp_sl exact kernels
    - path: src/trading/contexts/backtest/application/services/v2/admission.py
      why: "ultra limits and request cost guardrails"
      inspect_symbols:
        - BacktestTierPolicy
        - ensure_full_job_request_allowed
  conditional_bundles:
    benchmark_reference:
      read_when: "before performance changes and when validating stage names or accepted baseline"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_summary.md
        - scripts/backtest/run_iteration_8_execution_sizing_benchmark.py
    dto_config_runtime:
      read_when: "when adding top_n, Numba thread, progress, or cost-estimate fields"
      paths:
        - src/trading/contexts/backtest/application/dto/no_risk_exact.py
        - src/trading/contexts/backtest/application/dto/runtime_preflight.py
        - src/trading/contexts/backtest/adapters/outbound/config/backtest_admission_runtime_config.py
    tests_hot_path:
      read_when: "when extending existing no-risk, combo planning, or admission tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_admission.py
    process_boundary:
      read_when: "if process-isolated runner changes are already present and progress/cancel hooks must integrate"
      paths:
        - apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
        - src/trading/contexts/backtest/application/use_cases/backtest_job_worker.py
  consult_if_needed:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      read_when: "runtime contract, arity, top_n, or stage semantics are ambiguous"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_3_heap_update_corrective/benchmark_summary.md
      read_when: "heap update regression boundary is unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  no_python_itertools_product_for_large_full_job: true
  no_double_full_cartesian_pass_when_proxy_filter_is_passthrough: true
  request_top_n_50_must_return_top_50: true
  benchmark_top_k_5_must_remain_benchmark_metadata_only: true
  explicit_backtest_numba_thread_config_required: true
  deterministic_light_heavy_cost_classification_required: true
  preflight_must_classify_heavy_from_conservative_upper_bound: true
  preflight_light_decision_must_be_light_candidate_only: true
  post_prepare_must_confirm_or_promote_light_candidate: true
  light_jobs_must_have_small_resource_budget: true
  heavy_jobs_must_keep_full_host_budget: true
  preserve_canonical_result_semantics: true
  macstudio_benchmark_required_before_claiming_performance: true
  optional_benchmarks_must_exclude_heaviest_140s_job: true
  remove_large_product_from_production_path_required: true
  old_cartesian_iterator_test_only_or_small_helper_only: true
  no_double_pass_for_passthrough_combo_planning_required: true
  active_docs_must_remove_large_cartesian_as_current_production_path: true
  docs_cleanup_required_for_hot_path_replacement: true

task_toggles:
  implementation_changes_allowed: true
  add_ordinal_chunking: true
  add_numba_decode_kernel: true
  fix_heap_capacity_for_request_top_n: true
  add_cost_estimate_for_scheduler: true
  add_per_class_numba_thread_budget_if_needed: true
  add_progress_hooks_if_low_risk: true
  change_public_api: false
  publish_after_success: false

skill_routing:
  - skill: backend-performance-evidence
    use_when: "selecting baseline, changing hot path, measuring CPU/wall/RSS, or reporting speed"
    timing: before implementation and during verification
    reason: "this prompt is performance-sensitive and benchmark-gated"
  - skill: numba
    use_when: "implementing ordinal decode, `@njit`, `prange`, threading config, or diagnostics"
    timing: during implementation
    reason: "the compute fix must move Python object churn out of the critical path"
  - skill: root-cause-debugging
    use_when: "parity, performance, threading, or progress behavior regresses"
    timing: if blocker
    reason: "localize semantic regressions before widening changes"
  - skill: contract-impact-analysis
    use_when: "changing `top_n`, admission limits, config keys, telemetry schema, or cache/request identity"
    timing: before final report
    reason: "top_n and runtime config are externally relied-on contracts"
  - skill: backend-quality-gates
    use_when: "running focused unit/lint/type gates"
    timing: during verification
    reason: "Roehub backend gates are uv-based"

target_envs:
  - local-dev
  - mac-studio

required_literals:
  - "BTCUSDT"
  - "15m"
  - "196^5"
  - "289254654976"
  - "top_n=50"
  - "benchmark_top_k=5"
  - "NUMBA_NUM_THREADS"
  - "ROEHUB_BACKTEST_NUMBA_NUM_THREADS"
  - "ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS"
  - "ROEHUB_BACKTEST_HEAVY_NUMBA_NUM_THREADS"
  - "scheduling_class"
  - "light_candidate"
  - "light"
  - "heavy"
  - "estimated_combinations_upper_bound"
  - "estimated_combinations"
  - "exclude_heaviest_140s_job"
  - "iter_combo_chunks"
  - "BacktestComboPlanningService.execute"
  - "BacktestNoRiskExactScoringService.execute"
  - "_iter_selected_candidate_batches"
  - "BacktestPreflightService.execute"
  - "docs cleanup"
  - "Cartesian chunk planning"
  - "ordinal chunking"
  - "exact_scoring"
  - "heap_update"

non_goals:
  - "Do not rewrite notebook semantics or scoring formulas."
  - "Do not change public API shape for job creation/results."
  - "Do not claim acceptance from local benchmarks."
  - "Do not broaden indicators catalog or UI work."
  - "Do not use random sampling to replace exact top-N computation."
  - "Do not run the single heaviest 140+ second benchmark job in any optional local or Mac Studio check from this prompt."

final_report_format:
  language: ru
  sections:
    - "Intent"
    - "Hot path changes"
    - "Top N / contracts"
    - "Numba / CPU"
    - "Removed/replaced paths"
    - "Docs cleanup"
    - "Local gates"
    - "Performance evidence"
    - "Contract impact"
    - "Next prompt"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_admission.py"
    expect: "passes; include new focused tests explicitly"
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes or unrelated pre-existing failures classified"
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<new_iteration_dir>/local_accounting_validation.json"
    expect: "passes after the benchmark prompt creates the new iteration dir; otherwise state not run"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/combo_planning.py"
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py if shared chunking affects risk-on"
  - "src/trading/contexts/backtest/application/services/v2/admission.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/dto/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/dto/runtime_preflight.py"
  - "src/trading/contexts/backtest/adapters/outbound/config/backtest_admission_runtime_config.py"
  - "apps/worker/backtest_job_runner/**"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.md"
  - "docs/architecture/backtest/benchmark_iterations/README.md"
  - "docs/architecture/backtest/backtest-job-runner-production-plan-v1.md"

safety_notes:
  - "Algorithmic parity is more important than speed. Stop on unexplained result drift."
  - "Use comparable benchmark evidence before claiming performance success."
  - "Do not weaken numerical policy, ranking, or deterministic ordering for speed."
---

# Task

Fix the backtest compute hot path so a UI-created production job with a large indicator grid can use the host effectively without being dominated by Python Cartesian object churn.

Done means:

- full-job compute no longer uses Python `itertools.product` for large Cartesian spaces;
- pass-through combo planning does not perform a complete Cartesian pre-pass before exact scoring;
- `top_n=50` returns/persists up to 50 top strategies for production jobs;
- `benchmark_top_k=5` remains benchmark metadata and does not cap production results;
- effective Numba thread configuration for the backtest engine is explicit and visible in telemetry/logs;
- preflight exposes deterministic conservative upper-bound estimates needed by the runner scheduler;
- obvious heavy jobs are classified as `heavy` during preflight, before prepare/exact compute;
- possible light jobs are initially classified as `light_candidate` and confirmed after prepare/basic stages;
- local parity tests pass and Mac Studio benchmark acceptance is deferred to prompt 04, where the single heaviest 140+ second benchmark job is excluded and all other reference jobs remain required.

## Context / Current State

The accepted May 2 benchmark proves canonical runtime semantics on Mac Studio, but the current UI/API production path can ask for much larger grids. A 5-indicator request with 196 rows each has `196^5 = 289254654976` combinations.

Current risk points:

- `iter_combo_chunks` uses `itertools.product`, Python lists, and `np.asarray` per chunk.
- combo planning may iterate the full Cartesian space even when proxy filtering is pass-through.
- exact scoring then iterates selected candidate batches again.
- production `request.top_n=50` can still be internally capped by benchmark heap capacity.
- Numba thread behavior is not controlled by a backtest-specific config surface.
- runner scheduling needs a preflight conservative cost model so obvious heavy jobs are routed to the exclusive host slot before any compute child spends time on prepare/exact scoring.

This prompt must preserve the canonical scoring algorithm. The goal is to change how combinations are generated, streamed, scored, and reported, not to change what a combination means.

## Method-level replacement / cleanup map

This prompt must replace the old large-grid production hot path, not add ordinal code beside an active Python Cartesian path.

Required current-method decisions:

- `src/trading/contexts/backtest/application/services/v2/combo_planning.py::iter_combo_chunks`
  - Remove from large/full production path or narrow to small-test/small-helper usage only.
  - If retained, guard it with explicit thresholds and documentation that it is not used for large full jobs.
  - Tests must prove large grids do not call the Python `itertools.product` implementation.
- `src/trading/contexts/backtest/application/services/v2/combo_planning.py::BacktestComboPlanningService.execute`
  - Replace full Cartesian pre-pass when proxy filtering is pass-through.
  - For pass-through, return a compact ordinal/range plan or equivalent stream descriptor instead of materialized combo batches.
  - Preserve telemetry fields, but distinguish estimated/streamed totals from materialized totals.
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py::BacktestNoRiskExactScoringService.execute`
  - Replace selected-candidate iteration that depends on materialized Cartesian batches with ordinal streaming.
  - Heap capacity must come from production `request.top_n` (`top_n=50`) and not from `benchmark_top_k=5`.
  - Keep deterministic ordering and tie-breaking equivalent to the old small-pool reference.
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py::_iter_selected_candidate_batches`
  - Replace for production scoring with ordinal decoded batches.
  - If the old helper remains, make it small-test/reference-only and verify it is not used for large production jobs.
- `src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py`
  - If risk-on exact scoring shares the same candidate batch contract, update it to consume the new ordinal/range plan rather than old materialized combo batches.
- `src/trading/contexts/backtest/application/services/v2/preflight.py::BacktestPreflightService.execute`
  - Add conservative upper-bound scheduling metadata before prepare/exact compute.
  - Do not rely on prepare/basic stages to classify obvious heavy requests such as `196^5`.
- DTO/config modules for preflight/admission/runtime config
  - Add only compatible fields/config keys needed for `estimated_combinations_upper_bound`, `scheduling_class`, and per-class thread budgets.
  - Remove or rename stale benchmark-only top-k config if it can still cap production result capacity.

Definition of “removed” for this prompt:

- The old Python Cartesian iterator can remain as a small reference path only if production routing cannot reach it for large jobs.
- Pass-through combo planning must not still traverse the full Cartesian space before exact scoring.
- A static or runtime test must fail if a large `196^5` request enters `itertools.product`-based generation.

## Documentation cleanup / drift map

Docs must be updated when the hot path changes. Do not leave active architecture docs saying the current production algorithm is deterministic Python Cartesian chunk planning for large jobs.

Required doc decisions:

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` and `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md`
  - Replace active descriptions of `combo_iteration` as Python Cartesian chunk generation for production large jobs.
  - Describe ordinal/range streaming, Numba decode, pass-through no-prepass behavior, and `top_n=50` production capacity.
  - Keep historical benchmark iteration descriptions only if they are clearly historical snapshots.
- `docs/architecture/backtest/benchmark_iterations/README.md`
  - Update benchmark template fields if new evidence includes ordinal streaming, excluded heaviest job, legacy-path absence, or memory release.
- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`
  - Update scheduler/cost-estimate text if new `scheduling_class`, `estimated_combinations_upper_bound`, or per-class Numba thread budgets become active.

Required docs verification:

- Run `rg -n "itertools.product|Cartesian chunk|Cartesian chunk planning|combo_iteration|benchmark_top_k|top_k=5" docs/architecture/backtest` and classify every remaining hit as historical reference, test/reference helper, or updated current-state text.
- Run `uv run python -m tools.docs.generate_docs_index --check` if Markdown docs change.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Preserve canonical result semantics, ranking, request hash, artifact identity, and deterministic ordering.
- Replace large Cartesian generation with ordinal chunking suitable for N-arity pools.
- Remove or isolate the old `itertools.product` implementation from large production jobs instead of leaving both paths active.
- Move ordinal decode into Numba or another compiled numeric path for hot loops.
- Avoid a full combo planning pass when proxy filtering is effectively pass-through.
- Ensure exact scoring streams chunks and updates heap without materializing the full combination set.
- Fix production top-N so `request.top_n=50` drives heap/result capacity for production jobs.
- Keep `benchmark_top_k=5` as benchmark/accounting metadata, not production result capacity.
- Add deterministic preflight upper-bound estimates: at minimum `estimated_combinations_upper_bound`, arity, row-count upper bounds by indicator, risk mode, requested range, and requested `top_n`.
- Add a conservative preflight `scheduling_class` decision: `heavy` when the upper bound exceeds light thresholds; `light_candidate` only when all configured upper-bound thresholds are met.
- Unknown, unmeasurable, or artifact-dependent preflight cost must classify as `heavy`.
- Add post-prepare refinement using actual row counts and `estimated_combinations`: confirm `light_candidate` as `light`, or promote/requeue it to `heavy` before exact scoring.
- Do not require prepare/basic stages to identify obvious heavy jobs such as `196^5`; these must be heavy at preflight.
- Define per-class thread budgets so parallel light jobs do not oversubscribe the M2 Max. Heavy jobs may use the full host budget.
- Add tests for ordinal chunk equivalence on small pools against legacy `itertools.product` order.
- Add tests proving `top_n=50` can produce 50 results when enough candidates exist.
- Add tests proving preflight heavy classification is deterministic and conservative.
- Add tests proving `light_candidate` is promoted to `heavy` after prepare if actual combinations exceed thresholds.
- Add or expose backtest-specific Numba thread config and log/telemetry evidence.
- Update active runtime/benchmark docs so they no longer describe the replaced Python Cartesian large-job path or benchmark-only `top_k=5` as production behavior.

## Requirements (Should)

- Preserve legacy chunk order for deterministic parity where practical.
- Keep the old iterator only for small tests or compatibility helpers if useful.
- Add progress hook points at chunk boundaries if this can be done without broad API changes.
- Add admission/preflight upper-bound cost estimate fields for combinations and arity if existing DTOs support it compatibly.
- Prefer config keys such as `ROEHUB_BACKTEST_LIGHT_MAX_COMBINATIONS`, `ROEHUB_BACKTEST_LIGHT_MAX_ARITY`, and `ROEHUB_BACKTEST_LIGHT_NUMBA_NUM_THREADS` over hard-coded thresholds.

## Requirements (Nice-to-have)

- Add a tiny local microbenchmark for ordinal decode versus legacy iterator as developer evidence only.
- Add Numba parallel diagnostics notes in the final report if available.

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

- `always_read`: repository rules, benchmark evidence policy, compact prior state;
- `task_entrypoints`: combo planning, exact scoring, risk-on scoring, and admission limits;
- `conditional_bundles`: read only when the stated condition applies;
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation to define comparable baseline and during verification to report measured or deferred evidence.
- `numba`: use while implementing ordinal decode, `@njit`, `prange`, and thread diagnostics.
- `root-cause-debugging`: use if parity or performance behavior is unexplained.
- `contract-impact-analysis`: use before final report for `top_n`, admission, config, or telemetry changes.
- `backend-quality-gates`: use during focused verification.

1. Establish the current hot path and parity-sensitive ordering on a small fixture.
2. Design ordinal chunking for N pools with deterministic product-order equivalence.
3. Implement compiled ordinal decode and stream it into exact scoring.
4. Remove or bypass the duplicate full Cartesian pass when proxy filtering is pass-through.
5. Add deterministic preflight upper-bound cost estimation and `heavy|light_candidate` classification for scheduler use.
6. Add post-prepare refinement from `light_candidate` to confirmed `light` or promoted `heavy`.
7. Fix production `top_n` capacity and preserve `benchmark_top_k=5` accounting.
8. Add focused tests for ordering, parity, top-N capacity, preflight classification, post-prepare promotion, and admission guardrails.
9. Run quality gates and report performance evidence as local developer evidence only unless run on Mac Studio.

# Acceptance criteria (Definition of Done)

- Small-pool ordinal chunks produce the same ordered combinations as the legacy iterator.
- Large-grid tests prove the legacy Python Cartesian iterator is not called in production routing.
- Pass-through combo planning tests prove there is no complete Cartesian pre-pass before exact scoring.
- Large full-job paths do not call Python `itertools.product`.
- Pass-through proxy filter does not force a full pre-scan of the Cartesian space.
- Docs cleanup evidence proves active architecture docs no longer describe large production jobs as Python Cartesian chunk planning.
- Exact scoring can stream selected chunks and update heap without full materialization.
- `top_n=50` is honored in production result capacity.
- Preflight cost estimate includes `estimated_combinations_upper_bound` and enough request metadata to classify obvious heavy jobs before compute.
- `scheduling_class=heavy` is assigned at preflight whenever the conservative upper bound exceeds light thresholds.
- `scheduling_class=light_candidate` is assigned at preflight only to bounded small jobs; unknown cost is `heavy`.
- `light_candidate` is confirmed as `light` after prepare, or promoted/requeued to `heavy` before exact scoring.
- Per-class Numba thread budget is explicit or intentionally deferred with a documented blocker.
- Numba thread count/config is explicit and visible in logs or telemetry.
- Tests prove semantic parity on bounded fixtures.
- Final report separates local developer evidence from Mac Studio acceptance evidence.

# Implementation constraints

## Determinism & ordering

- Preserve deterministic ranking and tie-breaking.
- Preserve canonical product order unless a documented compatibility reason requires a controlled change.
- Do not introduce random sampling or approximate top-N.

## API / contracts

- Public API shape target: no change.
- `top_n=50` fix is compatible if it returns the requested number of rows.
- If admission, preflight, job metadata, or telemetry adds scheduling/cost fields such as `estimated_combinations_upper_bound`, classify as `compatible-change`.

## Performance

- Do not claim a performance win without comparable measurement.
- Mac Studio benchmark acceptance belongs to the next prompt.
- Local microbenchmarks are developer evidence only.
- Do not enable light parallelism in production until the benchmark prompt proves mixed scheduling is safe.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/application/services/v2/combo_planning.py`
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py` if shared chunking affects risk-on
- `src/trading/contexts/backtest/application/services/v2/admission.py`
- `tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`

Possible secondary touches:

- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_admission_runtime_config.py`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`

# Non-goals

- Do not change scoring formulas.
- Do not implement final benchmark evidence writer here unless it is trivial and does not distract.
- Do not change Web UI layout.
- Do not publish or deploy from this prompt.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_admission.py`
- `uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest`
- `uv run pyright`

# Final output: report format (strict)

Write the final report in Russian with these sections:

- `Intent`
- `Hot path changes`
- `Top N / contracts`
- `Numba / CPU`
- `Local gates`
- `Performance evidence`
- `Contract impact`
- `Next prompt`

Include exact commands run, pass/fail status, and whether any performance numbers are acceptance evidence or only developer evidence.
