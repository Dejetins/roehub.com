---
prompt_name: backtest_service_iteration_6_tp_sl_exact_scoring_full_metrics
repo: roehub.com
branch: current
scope: "Iteration 6: implement TP/SL exact scoring, risk-on top-K heap, and full metrics while preserving notebook-compatible benchmark boundaries."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo engineering contract, performance discipline, skill routing, and safety invariants"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "canonical Iteration 6 stage contract, non-goals, and benchmark acceptance rules"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence manifest, stage naming rules, and Mac Studio acceptance contract"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical notebook target values and TP/SL top-result shape"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_5_tp_sl_hit_times_loading_validation/benchmark_summary.md
      why: "latest accepted risk-on artifact loading evidence and Mac Studio artifact policy"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
      why: "Iteration 5 accepted hit-times/15m loading, grid validation, manifest hash, and cleanup boundary"
      inspect_symbols:
        - BacktestTpSlHitTimesService
        - LOAD_HIT_TIMES_STAGE_NAME
        - TP_SL_GRID_VALIDATION_STAGE_NAME
        - HIT_TIMES_ARTIFACT_PATH_V2
    - path: src/trading/contexts/backtest/application/dto/tp_sl_hit_times.py
      why: "accepted TP/SL subset DTOs and array invariants for long/short hit-time tables"
      inspect_symbols:
        - BacktestTpSlHitTimesSubset
        - BacktestTpSlHitTimesResult
        - BacktestTpSlGridResolution
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "accepted no-risk exact scoring patterns, heap boundaries, self-check telemetry, and low-allocation top-K rules"
      inspect_symbols:
        - BacktestNoRiskExactScoringService
        - EXACT_SCORING_STAGE_NAME
        - HEAP_UPDATE_STAGE_NAME
    - path: src/trading/contexts/backtest/application/dto/no_risk_exact.py
      why: "accepted exact-scoring DTO style, top result shape, telemetry, and compatibility pattern"
      inspect_symbols:
        - BacktestNoRiskExactConfig
        - BacktestNoRiskTopResult
        - BacktestNoRiskExactTelemetry
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "risk-on backend selector and combo/exact context entrypoint"
      inspect_symbols:
        - EVENT_SEGMENTS_N_TP_SL_15M_GRID_BACKEND
        - BacktestComboPlanningService
        - BacktestBackendRegistry
    - path: src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
      why: "canonical vs service-only timing separation and total_without_warmup construction"
      inspect_symbols:
        - CANONICAL_STAGE_ORDER
        - TP_SL_TOTAL_COMPONENT_STAGES
        - SERVICE_ONLY_TELEMETRY_FIELDS
        - build_benchmark_accounting_record
  conditional_bundles:
    canonical_notebook_algorithm:
      read_when: "the service implementation details are ambiguous or parity with notebook top results fails"
      paths:
        - tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
    previous_benchmark_runner_patterns:
      read_when: "adding the Iteration 6 benchmark runner or summary writer"
      paths:
        - scripts/backtest/run_iteration_4_2_exact_scoring_benchmark.py
        - scripts/backtest/run_iteration_4_3_heap_update_benchmark.py
        - scripts/backtest/run_iteration_4_7_memory_cleanup_smoke.py
        - scripts/backtest/run_iteration_5_tp_sl_hit_times_benchmark.py
        - scripts/backtest/validate_benchmark_accounting.py
    unit_test_patterns:
      read_when: "adding service/DTO tests for TP/SL exact scoring, self-check, heap, or metric correctness"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    legacy_contract_reference:
      read_when: "a DTO invariant, dtype, tie rule, or historical metric definition is unclear after reading active v2 code and runtime doc"
      paths:
        - src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
      warning: "Use only as a typed invariant reference. Do not revive legacy StageB runtime modules, old execution-profile vocabulary, or old hit-times paths."
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_2_exact_scoring_self_check/benchmark_summary.md
      read_when: "copying accepted exact_scoring/self_check benchmark evidence style"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_3_heap_update_corrective/benchmark_summary.md
      read_when: "copying accepted heap_update evidence style and tiny-stage accounting rules"
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_7_memory_cleanup/benchmark_summary.md
      read_when: "copying accepted cleanup evidence style"

style_references:
  - .codex/promt_template.md
  - .codex/agents/generated/backtest-service-iteration-5/01-implement-tp-sl-hit-times-loading-validation.md
  - docs/architecture/backtest/benchmark_iterations/README.md

hard_requirements:
  hit_times_path_is_15m: true
  benchmark_top_k_is_5: true
  request_top_n_not_used_for_canonical_heap: true
  compare_only_notebook_compatible_stages: true
  keep_full_metrics_second_pass_service_only_if_used: true
  no_persistence_or_public_api_identity: true
  macstudio_acceptance_required: true
  preserve_iterations_1_to_5_acceptance: true
  max_implementation_attempts: 2

task_toggles:
  implement_tp_sl_exact_scoring: true
  implement_tp_sl_self_check: true
  implement_risk_on_heap_update: true
  implement_full_metrics_for_selected_tp_sl_cell: true
  implement_tp_sl_full_metrics_second_pass: true
  implement_persistence_or_public_api: false
  implement_lazy_trades: false
  update_benchmark_manifest_if_needed: true
  mark_main_doc_pass_only_after_macstudio_acceptance: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "defining, implementing, or reporting TP/SL benchmark comparisons against the canonical notebook"
    timing: before implementation and during verification
    reason: "Iteration 6 is accepted only through comparable Mac Studio benchmark evidence"
  - skill: numba
    use_when: "writing or changing TP/SL exact scoring kernels, prefix/diff buffers, self-check kernels, or allocation-sensitive hot paths"
    timing: during implementation and optimization
    reason: "the hot path must reproduce notebook semantics with Numba/NumPy performance discipline"
  - skill: contract-impact-analysis
    use_when: "adding DTOs, exports, runner JSON fields, metric names, or benchmark evidence schema"
    timing: before implementation and before final report
    reason: "DTO, metric, timing, and benchmark contracts must remain compatible with accepted stages"
  - skill: backend-quality-gates
    use_when: "running targeted lint, type, and pytest gates"
    timing: during verification
    reason: "Roehub backend gates are uv-based"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "hit_times/15m"
  - "event_segments_n_tp_sl_15m_grid"
  - "tp_sl_grid"
  - "tp_sl_exact_scoring"
  - "tp_sl_full_metrics_second_pass"
  - "load_hit_times"
  - "tp_sl_grid_validation"
  - "exact_scoring"
  - "heap_update"
  - "benchmark_top_k = 5"
  - "request.top_n = 100"
  - "historical_prefix_compatible"

non_goals:
  - "Do not implement persisted risk-on top-N rows; that belongs to Iteration 7."
  - "Do not implement public `variant_key`, `variant_hash`, storage identity mapping, or API read-model assembly; those belong to Iteration 7."
  - "Do not implement lazy trades, UI chart payloads, job orchestration, idempotency, authz, or cancel/list endpoints."
  - "Do not change accepted no-risk Iteration 4 behavior except for strictly shared, compatible helper reuse."
  - "Do not use legacy `hit_times/1m` runtime paths or old execution-profile vocabulary."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Stage contract"
    - "Benchmark / Mac Studio"
    - "Проверки"
    - "Contract impact"
    - "Ограничения / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest scripts/backtest"
    expect: "passes, or a narrower justified target passes if unrelated existing files fail"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/<new tp_sl exact tests>.py"
    expect: "passes after replacing the placeholder with the actual new test file"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "passes, or failing unrelated tests are isolated with evidence"
  - cmd: "uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<iteration_6_dir>/local_accounting_validation.json"
    expect: "passes after the Iteration 6 runner writes local accounting evidence"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs or benchmark summaries change"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/dto/<new tp_sl exact dto>.py"
  - "src/trading/contexts/backtest/application/services/v2/<new tp_sl exact service>.py"
  - "tests/unit/contexts/backtest/application/services/v2/<new tp_sl exact tests>.py"
  - "scripts/backtest/run_iteration_6_tp_sl_exact_scoring_benchmark.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_6_tp_sl_exact_scoring_full_metrics/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py"
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/README.md"

safety_notes:
  - "Iteration 6 acceptance is a Mac Studio benchmark gate, not local unit-test acceptance."
  - "If `tp_sl_full_metrics_second_pass` is implemented, record it as service-only CPU/RSS/latency evidence and do not include it in canonical `exact_scoring` or `tp_sl_exact_scoring` ratios."
  - "For TP/SL tie semantics, SL wins when TP and SL hit on the same bar: TP requires `t_tp < t_sl`, SL allows `t_sl <= t_tp`."
  - "The executor has only 2 implementation attempts. After the second failed corrective cycle, stop and report the blocker with exact evidence."
---

# Task

Implement Iteration 6: TP/SL exact scoring and full metrics for the artifact-backed backtest service.

Done means:

- `event_segments_n_tp_sl_15m_grid` can score risk-on TP/SL grid combinations for arity 1..10 using the Iteration 5 `hit_times/15m` subset;
- TP/SL self-check compares the fast path with a slow direct reference and records deterministic mismatch evidence;
- risk-on `heap_update` keeps notebook-compatible top rows with `benchmark_top_k = 5`;
- selected top rows include `total_return_pct`, `trade_count`, `best_tp_pct`, and `best_sl_pct` with notebook-compatible ranking and tie semantics;
- the service can produce the full risk-on metric set for selected best TP/SL cells;
- Mac Studio benchmark evidence is written under `docs/architecture/backtest/benchmark_iterations/<date>_iteration_6_tp_sl_exact_scoring_full_metrics/`;
- no persistence, public/storage identity assembly, API read models, or lazy trades are implemented in this iteration.

## Context / Current State

Context ledger from previous accepted iterations:

- completed:
  - Iteration 1: request normalization, preflight, artifact context.
  - Iteration 2: artifact arrays, slicing, `prepare_pools_core`.
  - Iteration 3: combo planning contexts.
  - Iteration 4: no-risk exact scoring/top-K path and cleanup benchmark evidence.
  - Iteration 5: `hit_times/15m` TP/SL grid validation and hit-times subset loading.
- open_items:
  - risk-on exact scoring is not implemented yet;
  - risk-on full metrics for selected best TP/SL cells are not implemented yet;
  - risk-on top-result persistence/API identity remains out of scope until Iteration 7.
- contract_changes:
  - Iteration 6 may add application DTO/service internals and benchmark-runner JSON fields;
  - Iteration 6 must not change public API, persistence schema, public `variant_key`, or storage identity semantics.
- touched_paths:
  - Iteration 5 added `tp_sl_hit_times` DTO/service and benchmark runner.
  - Iteration 4 added accepted no-risk exact scoring patterns that should be reused carefully.
- risks:
  - stage timing can become incomparable if full metrics or result assembly are mixed into `tp_sl_exact_scoring`;
  - tiny heap stages can fail from Python object materialization if low-allocation rules are ignored;
  - artifact hash may differ from canonical, so evidence must explicitly record `historical_prefix_compatible` when valid.
- next_focus:
  - implement TP/SL scoring kernels;
  - implement bounded risk-on top-K heap;
  - verify full metrics correctness for selected best cells;
  - produce Mac Studio evidence for arity 1..7 and correctness smoke for arity 8..10.

Additional context:

- Canonical notebook baseline is `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`.
- Canonical numeric target is `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.
- Runtime target path is `hit_times/15m`.
- Target TP/SL grid is `2.0..25.0` inclusive, step `0.5`.
- Canonical risk-on top rows currently contain `total_return_pct`, `trade_count`, `best_tp_pct`, `best_sl_pct`.
- Full metric set is a production service requirement. It may be computed in a bounded second pass for retained top rows.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the scoped change described in this prompt.
- Preserve all explicitly protected contracts and accepted benchmark stages.
- Add or update targeted tests for scoring, self-check, heap ordering, full metrics, and failure cases.
- Update related exports only when needed.
- Keep the implementation deterministic, bounded, and reviewable.

TP/SL exact scoring:

- Implement the `event_segments_n_tp_sl_15m_grid` path for arity 1..10.
- Consume the Iteration 5 `BacktestTpSlHitTimesSubset` arrays:
  - `long_tp`;
  - `long_sl`;
  - `short_tp`;
  - `short_sl`;
  - `tp_values`;
  - `sl_values`;
  - `sentinel_index`.
- Preserve notebook tie semantics:
  - TP wins only when `t_tp < t_sl`;
  - SL wins when `t_sl <= t_tp`;
  - same-bar TP/SL therefore resolves to SL.
- Reproduce the notebook prefix/diff algorithm:
  - `point` for exact TP/SL cells;
  - `row_diff` for TP-only ranges;
  - `col_diff` for SL-only ranges;
  - `rect_diff` for signal/final-close fallback rectangles;
  - prefix sums materialize the full TP/SL grid contribution for one combo.
- Select the best TP/SL cell by maximum log return and convert back with:
  - `total_return_pct = (exp(best_log) - 1) * 100`.
- Record `best_tp_idx`, `best_sl_idx`, `best_tp_pct`, `best_sl_pct`, `trade_count`, and `total_return_pct` for retained results.
- Support both accepted direction modes:
  - `long_only`;
  - `long_short_reversal`.

Self-check:

- Implement TP/SL self-check against a slow direct reference.
- Self-check must validate at least:
  - selected best TP/SL cell;
  - trade count;
  - total return within configured tolerance;
  - direction-mode semantics;
  - same-bar TP/SL tie behavior.
- Self-check telemetry must be summary-only and deterministic.

Risk-on heap:

- Implement notebook-compatible `heap_update` for risk-on top rows with `benchmark_top_k = 5`.
- Keep `request.top_n = 100` separate from canonical heap comparison.
- Rank by selected metric, default `total_return_pct desc`.
- Use deterministic tie-break keys derived from score, original row ids, and ordinal data.
- Apply the accepted low-allocation rules from Iteration 4.3:
  - first compute only rank score, original row ids, and heap admission key;
  - materialize full item only if the heap has capacity or candidate replaces the current worst row;
  - do not build public variant keys, hashes, persisted DTOs, or API rows inside `heap_update`.

Full metrics:

- Implement full summary metric calculation for selected best TP/SL cells:
  - `total_return_pct`;
  - `max_drawdown_pct`;
  - `return_over_max_drawdown`;
  - `profit_factor`;
  - `trade_count`;
  - `sharpe_trades`;
  - `win_rate_pct`;
  - `avg_trade_ret_pct`;
  - `avg_trade_exec_bars`;
  - `exposure_pct`;
  - `best_tp_pct`;
  - `best_sl_pct`.
- If these metrics are not free inside the hot path, implement `tp_sl_full_metrics_second_pass`.
- If `tp_sl_full_metrics_second_pass` exists, it must:
  - run only for retained top rows, not all evaluated combos;
  - use the same execution, fee, slippage, sizing, profit lock, direction, and close-on-end settings;
  - record CPU/RSS/latency as service-only evidence;
  - not be included in canonical `exact_scoring` or `tp_sl_exact_scoring` timing comparison.

Benchmark runner:

- Add `scripts/backtest/run_iteration_6_tp_sl_exact_scoring_benchmark.py`.
- Write evidence to:
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_6_tp_sl_exact_scoring_full_metrics/benchmark_results.json`;
  - `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_6_tp_sl_exact_scoring_full_metrics/benchmark_summary.md`.
- Compare only notebook-compatible stages against canonical notebook target:
  - `load_hit_times`;
  - `tp_sl_grid_validation`;
  - `prepare_pools_core`;
  - `build_exact_context`;
  - `build_proxy_context`;
  - `combo_iteration`;
  - `proxy_filter`;
  - `self_check`;
  - `exact_scoring` / `tp_sl_exact_scoring`;
  - `heap_update`.
- Do not double-count `exact_scoring` and `tp_sl_exact_scoring` in `total_without_warmup`. If both are recorded for compatibility, they must refer to the same notebook-compatible TP/SL scoring boundary and total construction must follow `benchmark_accounting.py`.
- Risk-on `total_without_warmup` must be the notebook-compatible sum documented in the runtime plan:
  - `load_hit_times + tp_sl_grid_validation + prepare_pools_core + build_exact_context + build_proxy_context + combo_iteration + proxy_filter + self_check + exact_scoring + heap_update`.
- Record service-only telemetry separately:
  - `tp_sl_full_metrics_second_pass`, if implemented;
  - `service_total_without_warmup`;
  - cleanup evidence.

Mac Studio acceptance:

- Benchmark acceptance requires Mac Studio evidence.
- For arity 1..7 and both direction modes, stage ratios must be `>= 0.9` for the benchmark-gated stages.
- For arity 8..10, run service-level correctness smoke on small row pools. These arities are not required for the canonical 90 percent target comparison.
- Evidence must record:
  - commit SHA;
  - artifact config and root;
  - artifact manifest hash;
  - hit-times manifest hash;
  - request hash;
  - artifact compatibility policy;
  - stage timings;
  - ratio definition;
  - top-result parity or best-cell correctness;
  - full metric correctness evidence;
  - cleanup evidence.

Memory cleanup:

- Add repeated-run cleanup evidence for:
  - hit-time subsets;
  - TP/SL diff buffers;
  - score arrays;
  - selected top rows.
- After a run finishes and results are available, heavy calculation objects must not remain strongly referenced by the service runner.
- Cleanup evidence is service hygiene evidence. Do not compare it against canonical notebook timing.

## Requirements (Should)

- Keep DTOs compact and immutable where possible, following the accepted no-risk and hit-times DTO style.
- Prefer Numba/NumPy array-first hot paths over Python object loops in scoring and heap admission.
- Reuse Iteration 4.3 heap accounting and Iteration 5 artifact compatibility reporting.
- Keep runner output shape close to prior benchmark summaries for easy comparison.
- Include enough local synthetic tests to isolate tie semantics, best-cell selection, and full metric calculations without relying on Mac Studio artifacts.
- If a performance miss occurs, profile stage boundaries before changing semantics or broadening scope.

## Requirements (Nice-to-have)

- Add an explicit service-only budget table for `tp_sl_full_metrics_second_pass` in the generated summary.
- Add a small diagnostic in the runner that prints the slowest failing `{arity, direction_mode, stage}` rows first.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. the Iteration 6 section and stage-contract sections of `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
3. benchmark manifest and canonical JSON target
4. latest accepted Iteration 5 evidence
5. task entrypoints
6. only the conditional bundle(s) required by touched contracts, failing checks, or parity ambiguity
7. consult-if-needed references only for blockers, ambiguity, or conflict resolution

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 10 files`
- `<= ~45k-60k tokens`

Stop reading once all of the following are true:

- scoring and heap stage contracts are identified;
- changed DTOs/services/scripts are bounded;
- benchmark evidence shape is implementable;
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for:

- benchmark threshold failures;
- incorrect top-result parity;
- unclear notebook scoring semantics;
- failing quality gates;
- contract ambiguity;
- Numba typing/performance blockers.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules;
  - current runtime target;
  - canonical benchmark target;
  - latest accepted Iteration 5 evidence.
- `task_entrypoints`:
  - existing accepted services and DTOs that anchor the implementation.
- `conditional_bundles`:
  - read only when the stated condition applies.
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation and during verification; owns benchmark comparability, ratio claims, CPU/RSS evidence, and failure diagnosis.
- `numba`: use during implementation and optimization when writing kernels or allocation-sensitive hot paths; owns JIT-specific design and typing/performance decisions.
- `contract-impact-analysis`: use before implementation and before final report when adding DTOs, exports, metric fields, or benchmark JSON fields; owns compatibility classification.
- `backend-quality-gates`: use during verification; owns uv-based lint, type, and test gates.

Implementation sequence:

1. Confirm current status and read the bounded context.
2. Inspect canonical risk-on runs in `benchmark_results.json` and record expected stage names, top-result fields, and target ratios.
3. Design compact TP/SL exact DTOs and service entrypoint.
4. Implement slow direct reference for correctness tests and self-check.
5. Implement fast TP/SL exact scoring for `event_segments_n_tp_sl_15m_grid`.
6. Implement risk-on heap update with `benchmark_top_k = 5` and low-allocation materialization.
7. Implement full metrics for selected best cells, using `tp_sl_full_metrics_second_pass` if needed.
8. Add local unit tests for:
   - TP/SL same-bar tie rule;
   - best-cell selection;
   - direction modes;
   - self-check mismatch reporting;
   - heap ordering/tie-breaks;
   - full metric correctness for a small synthetic fixture;
   - no persistence/API identity inside Iteration 6 outputs.
9. Add the Iteration 6 benchmark runner and evidence writer.
10. Run local quality gates.
11. Publish/sync only after local gates pass and the repository delivery path requires it.
12. Run Mac Studio benchmark on the deployed/runtime code and write evidence to the accepted manifest location.
13. If Mac Studio acceptance passes, update the main runtime document status for Iteration 6 and regenerate/check docs index if needed.
14. If acceptance fails, stop after at most 2 implementation attempts and report the blocker with exact stage rows, ratios, commit, and artifact hashes.

# Benchmark and Mac Studio pipeline

The acceptance benchmark must run on Mac Studio. The intended chain is:

1. Implement code locally in `/Users/daniildegtyarev/Projects/roehub.com`.
2. Run local gates.
3. Commit and push the implementation through the normal repository delivery path.
4. Ensure the Mac Studio runtime uses the new code:
   - if `/Users/daniildegtyarev/Projects/roehub.com` on Mac Studio is the target checkout, pull the new commit there;
   - if `/opt/roehub/app` is the deployed runtime bundle, make sure deployment or an explicitly recorded runtime sync has placed the new code there.
5. On Mac Studio, use an environment where `uv` is available. In non-login SSH shells, prefer an explicit PATH or absolute binary path:

```bash
export PATH="/opt/homebrew/bin:$PATH"
cd /Users/daniildegtyarev/Projects/roehub.com
git pull --ff-only
uv run python scripts/backtest/run_iteration_6_tp_sl_exact_scoring_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_6_tp_sl_exact_scoring_full_metrics
```

If the benchmark must run from `/opt/roehub/app`, record that explicitly in the summary and set the commit field through the runner or environment so evidence identifies the implementation commit.

6. Write or refresh:
   - `benchmark_results.json`;
   - `benchmark_summary.md`;
   - local accounting validation JSON if the runner produces one.
7. Confirm that JSON and summary agree:
   - same pass/fail decision;
   - same commit;
   - same artifact hashes;
   - same ratio definition;
   - same failed rows, if any.

# Acceptance criteria (Definition of Done)

- Local code implements TP/SL exact scoring for arity 1..10 and both direction modes.
- Local tests cover slow reference parity, tie semantics, best-cell selection, heap ordering, self-check, and full metrics.
- Mac Studio benchmark evidence exists under the Iteration 6 evidence directory.
- For arity 1..7 and both direction modes:
  - `self_check` passes correctness;
  - `exact_scoring` / `tp_sl_exact_scoring` ratio is `>= 0.9`;
  - `heap_update` ratio is `>= 0.9`;
  - `total_without_warmup` is computed from notebook-compatible components only;
  - top-result best cell and key summary fields are notebook-compatible.
- For arity 8..10:
  - service-level correctness smoke passes on small row pools.
- Full metric-set correctness evidence exists for selected best TP/SL cells.
- `tp_sl_full_metrics_second_pass`, if present, is recorded as service-only CPU/RSS/latency evidence and is not part of canonical notebook ratios.
- Repeated-run memory cleanup smoke passes or records a clear bounded residual risk.
- Iteration 1..5 acceptance remains unchanged.
- The main runtime document marks Iteration 6 as accepted only if the Mac Studio evidence is full pass.

# Implementation constraints

- Keep diffs scoped to Iteration 6.
- Do not silently change public request/response contracts.
- Do not add dependencies unless explicitly justified and unavoidable.
- Do not use Python object-heavy loops in the hot scoring path unless benchmark evidence proves they meet the target.
- Do not include public `variant_key`, storage `variant_hash`, persisted row assembly, DB writes, or API read-model building in `heap_update` or `tp_sl_exact_scoring`.
- Do not mix service-only full metric second-pass timing into canonical exact-scoring timing.
- Do not compare local machine benchmark timings as acceptance evidence.
- Do not mark Iteration 6 `pass` from local tests alone.

# Files to indicate (expected touched areas)

Expected primary files:

- `src/trading/contexts/backtest/application/dto/<new tp_sl exact dto>.py`
- `src/trading/contexts/backtest/application/services/v2/<new tp_sl exact service>.py`
- `tests/unit/contexts/backtest/application/services/v2/<new tp_sl exact tests>.py`
- `scripts/backtest/run_iteration_6_tp_sl_exact_scoring_benchmark.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_6_tp_sl_exact_scoring_full_metrics/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_6_tp_sl_exact_scoring_full_metrics/benchmark_summary.md`

Possible secondary files:

- `src/trading/contexts/backtest/application/dto/__init__.py`
- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/architecture/README.md`

# Non-goals

- No job orchestration.
- No persistence.
- No public/storage variant identity.
- No API endpoint implementation.
- No lazy trades.
- No UI chart payload.
- No roadmap rewrite.
- No legacy `hit_times/1m` runtime path.

# Quality gates (must run and pass)

Run local gates before claiming implementation completion:

```bash
uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest scripts/backtest
uv run pyright
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/<new tp_sl exact tests>.py
uv run pytest -q tests/unit/contexts/backtest/application/services/v2
```

After benchmark evidence is written:

```bash
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/<iteration_6_dir>/local_accounting_validation.json
uv run python -m tools.docs.generate_docs_index --check
```

Run the Mac Studio benchmark gate before claiming acceptance:

```bash
export PATH="/opt/homebrew/bin:$PATH"
uv run python scripts/backtest/run_iteration_6_tp_sl_exact_scoring_benchmark.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/$(date +%F)_iteration_6_tp_sl_exact_scoring_full_metrics
```

If a command cannot be run, state why, classify the risk, and do not claim that gate as passed.

# Contract impact report

Include this classification in the final report:

- Public API:
- Ports:
- DTO schema:
- Persisted schema:
- Config schema:
- Request/cache/persistence identity:
- Benchmark evidence schema:
- Runtime artifact contract:

Use one of:

- `none`;
- `compatible-change`;
- `breaking-change`;
- `unknown`.

# Failure/blocker behavior

You have only 2 implementation attempts.

An attempt is a full cycle of implementation, local gates, and Mac Studio benchmark or equivalent blocker evidence. If the second attempt still fails acceptance:

- stop;
- do not broaden scope into Iteration 7;
- do not hide failed benchmark rows;
- leave failed evidence in the benchmark folder unless explicitly asked to clean it;
- report:
  - implementation commit;
  - changed files;
  - exact failed `{arity, direction_mode, stage}` rows;
  - canonical time, service time, and ratio;
  - artifact hashes and compatibility policy;
  - whether the failure is correctness, stage-boundary accounting, JIT/warmup, memory, or pure hot-path performance;
  - the smallest next investigation step.

# Final output: report format (strict)

Use Russian.

## Что сделано

- Concise implementation summary.

## Stage contract

- Which stages are implemented.
- Which timings are canonical vs service-only.
- Confirm `benchmark_top_k = 5` and `request.top_n = 100` separation.

## Benchmark / Mac Studio

- Evidence directory.
- Commit.
- Artifact hashes.
- Pass/fail table summary.
- Failed rows first if any.

## Проверки

- Commands run and results.
- Commands not run and why.

## Contract impact

- Public API:
- Ports:
- DTO schema:
- Persisted schema:
- Config schema:
- Request/cache/persistence identity:
- Benchmark evidence schema:
- Runtime artifact contract:

## Ограничения / следующий шаг

- Remaining risks.
- If accepted, state that Iteration 7 is next.
- If not accepted, state the blocker and stop.
