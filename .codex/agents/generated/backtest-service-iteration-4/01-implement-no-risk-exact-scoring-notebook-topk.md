---
prompt_name: backtest-service-artifact-runtime-v1-iteration-4-no-risk-exact-scoring-notebook-topk
repo: roehub.com
branch: main
scope: "Повторно реализовать Iteration 4: no-risk exact scoring, self-check, full no-risk metrics и notebook-compatible top-K без persistence/identity work."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and safety rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "sole v1 implementation source"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/benchmark_summary.md
      why: "accepted prior stage evidence"

  task_entrypoints:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 4 stage contract"
      inspect_symbols:
        - "Измеряемая стадия бенчмарка: `exact_scoring`"
        - "Измеряемая стадия бенчмарка: `heap_update`"
        - "Измеряемая стадия бенчмарка: `top_result_proxy_fill`"
        - "Service-only стадия: `top_result_assembly`"
        - "Итерация 4: no-risk exact scoring и notebook-compatible top-K"
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      why: "canonical no-risk algorithm"
      inspect_symbols:
        - "score_trade_list_no_risk"
        - "evaluate_no_risk_streaming_two"
        - "evaluate_no_risk_event_segments_two"
        - "evaluate_no_risk_event_segments_n"
        - "evaluate_no_risk_exact_chunk"
        - "proxy_for_indicator_rows"
        - "run_fast_vs_reference_self_check_two"
        - "search_topk_indicator_no_risk"
        - "run_benchmark_matrix"
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "Iteration 3 contexts and chunk/filter helpers"
      inspect_symbols:
        - "BacktestComboPlanningService"
        - "BacktestBackendRegistry"
        - "BacktestComboChunk"
        - "build_segment_stack"
        - "build_local_row_pools"
        - "iter_combo_chunks"
        - "make_combo_idx_matrix"
    - path: src/trading/contexts/backtest/application/dto/combo_planning.py
      why: "backend, context, and proxy DTO shape"
      inspect_symbols:
        - "BacktestComboPlanningConfig"
        - "BacktestComboPlanningResult"
        - "BacktestExactContext"
        - "BacktestProxyContext"
        - "BacktestProxyFilterResult"

  conditional_bundles:
    prepare_pools_inputs:
      read_when: "prepared pool or execution mapping shapes are unclear"
      paths:
        - src/trading/contexts/backtest/application/dto/prepare_pools.py
        - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
        - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py

    test_style_and_fixtures:
      read_when: "creating targeted unit tests or fixtures"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py

    exports:
      read_when: "new no-risk DTO/service modules need package exports"
      paths:
        - src/trading/contexts/backtest/application/services/v2/__init__.py
        - src/trading/contexts/backtest/application/dto/__init__.py

    benchmark_targets:
      read_when: "building benchmark comparison or recording Iteration 4 evidence"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/benchmark_results.json

    failed_iteration_4_history:
      read_when: "performance or contract failures resemble the reverted attempt"
      paths:
        - docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md

  consult_if_needed:
    - path: .codex/agents/.context/promt_manager_state.yaml
      read_when: "only if current thread state is unavailable; ignore unrelated stale entries"
    - path: configs/prod/indicators.yaml
      read_when: "indicator catalog expansion affects a test fixture"
    - path: docs/architecture/backtest/deep-research-report_for_Backtest Service Artifact Runtime v1.md
      read_when: "architecture ambiguity remains after reading the runtime document"

style_references:
  - src/trading/contexts/backtest/application/services/v2/combo_planning.py
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
  - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py

hard_requirements:
  runtime_doc_is_sole_implementation_source: true
  iteration_1_2_3_are_accepted: true
  no_roadmap_docs_as_context: true
  no_risk_only: true
  benchmark_top_k_is_5: true
  public_top_n_100_is_not_canonical_topk: true
  no_persistence_or_public_identity: true
  no_top_result_assembly_in_measured_stages: true
  no_tp_sl_hit_times_or_risk_scoring: true
  no_lazy_trades_or_api_ui: true
  benchmark_acceptance_macstudio_only: true
  keep_existing_user_changes: true
  max_implementation_attempts: 2

task_toggles:
  implement_no_risk_dto: true
  implement_no_risk_exact_service: true
  implement_event_segments_2_no_risk: true
  implement_event_segments_n_no_risk: true
  implement_streaming_2_no_risk_fallback: true
  implement_no_risk_self_check: true
  implement_notebook_compatible_heap_update: true
  implement_notebook_compatible_top_result_proxy_fill: true
  add_targeted_tests: true
  prepare_macstudio_benchmark_evidence: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "implementing or reporting timers, top-K boundary, CPU/RSS, or benchmark status"
    timing: before implementation
    reason: "owns baseline selection, stage boundaries, and performance claims"
  - skill: numba
    use_when: "adding or changing @njit kernels for exact scoring or trade metrics"
    timing: during implementation
    reason: "owns Numba typing, allocation, threading, and kernel behavior"
  - skill: contract-impact-analysis
    use_when: "adding DTOs, exports, config fields, timing payloads, or result identity fields"
    timing: during investigation
    reason: "owns boundary compatibility and prevents accidental public/storage contract changes"
  - skill: backend-quality-gates
    use_when: "running pytest, ruff, pyright, docs checks, or triaging quality failures"
    timing: during verification
    reason: "owns backend verification gates"
  - skill: root-cause-debugging
    use_when: "performance misses, parity mismatches, or previous stages regress"
    timing: if blocker
    reason: "owns root-cause investigation before further optimization"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "event_segments_2_no_risk"
  - "event_segments_n_no_risk"
  - "streaming_2_no_risk"
  - "service_warmup"
  - "self_check"
  - "exact_scoring"
  - "heap_update"
  - "top_result_proxy_fill"
  - "benchmark_top_k = 5"
  - "top_k = 1"
  - "request.top_n = 100"
  - "top_results_count = 5"
  - "score_trade_list_no_risk"
  - "evaluate_no_risk_streaming_two"
  - "evaluate_no_risk_event_segments_two"
  - "evaluate_no_risk_event_segments_n"
  - "evaluate_no_risk_exact_chunk"
  - "build_trade_list_for_indicator_rows_slow"
  - "proxy_for_indicator_rows"
  - "run_fast_vs_reference_self_check_two"
  - "apply_direction_mode"
  - "long_only"
  - "long_short_reversal"

non_goals:
  - "Do not implement persisted top-N rows, Postgres writes, object storage writes, or DB migrations."
  - "Do not generate public `variant_key`, `variant_hash`, or `indicator_variant_hash` in Iteration 4."
  - "Do not implement `top_result_assembly`; Iteration 7 owns public/storage identity and DTO/read-model assembly."
  - "Do not benchmark canonical `heap_update` or `top_result_proxy_fill` with product `top_n = 100`."
  - "Do not implement TP/SL risk-on scoring, hit_times loading, TP/SL grid validation, or full risk metrics."
  - "Do not implement lazy trades, API routes, UI integration, job orchestration, or persistence."
  - "Do not edit notebooks or canonical benchmark target files."
  - "Do not copy the reverted Iteration 4 implementation as-is; use git history only as an anti-pattern reference."

final_report_format:
  language: ru
  sections:
    - "Изменения"
    - "Контрактное влияние"
    - "Проверки"
    - "Benchmark/Evidence"
    - "Остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py"
    expect: "passes after creating the no-risk exact tests"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes if Markdown docs were touched"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/dto/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_no_risk_exact_scoring_notebook_topk/"

safety_notes:
  - "The runtime document is the source of truth; old roadmap wording and legacy backtest code are compatibility context only."
  - "The failed reverted Iteration 4 attempt mixed product top_n=100 and production assembly into notebook timers; do not repeat that boundary."
  - "Keep `heap_update` and `top_result_proxy_fill` notebook-compatible: `benchmark_top_k=5`, no variant hashes, no DTO/read-model assembly, no persistence."
  - "The executor has exactly 2 implementation attempts. After the second failed attempt, stop changing code and report the blocker."
  - "If not running on Mac Studio, local timings are developer evidence only and cannot close benchmark acceptance."
  - "Generated `.pyc`/Numba cache files under `__pycache__` are not source files and must not drive implementation decisions."
---

# Task

Повторно реализовать `Iteration 4` из
`docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`:
no-risk exact scoring и notebook-compatible top-K.

Реализация должна перенести в production service только ту часть алгоритма,
которая в canonical notebook отвечает за:

- `event_segments_2_no_risk`;
- `event_segments_n_no_risk` для arity 1..10;
- `streaming_2_no_risk` как fallback/parity comparator;
- no-risk self-check против slow reference;
- full no-risk summary metrics;
- notebook-compatible `heap_update`;
- notebook-compatible `top_result_proxy_fill`.

Done means:

- service может принять `BacktestPreparePoolsResult` + Iteration 3 combo planning
  context/result и выполнить no-risk scoring для arity 1..10;
- exact scoring semantics совпадает с canonical notebook for no-risk;
- `heap_update` и `top_result_proxy_fill` сравнимы с notebook timers только при
  `benchmark_top_k = 5`;
- unit tests покрывают exact scoring, self-check, heap/top-K boundary,
  direction modes и proxy fill semantics;
- final report clearly states whether Mac Studio benchmark acceptance was run.

## Context / Current State

Context ledger:

- completed:
  - Iteration 1 accepted:
    `docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_1_request_normalization_artifact_context/`.
  - Iteration 2 accepted:
    `docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/`.
  - Iteration 3 accepted:
    `docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/`.
  - `combo_planning.py` already owns backend registry, exact/proxy contexts,
    deterministic chunk iteration, and proxy filter.
- open_items:
  - Iteration 4 implementation was attempted and reverted.
  - Current source tree does not contain `services/v2/no_risk_exact.py` or
    `dto/no_risk_exact.py`; stale `__pycache__` files may remain and must be ignored.
  - Iteration 4 must be reimplemented with the corrected stage contract.
- contract_changes:
  - public `top_n = 100` is a product/API contract.
  - canonical benchmark `benchmark_top_k = 5` is the measured top-K size for
    `heap_update`, `top_result_proxy_fill`, `total_without_warmup`, and result hash.
  - sample warmup uses `top_k = 1`.
  - `top_result_assembly`, public `variant_key`, `variant_hash`,
    `indicator_variant_hash`, DTO/read-model assembly, and persistence are
    service-only future work for Iteration 7, not Iteration 4.
- touched_paths:
  - likely new DTO/service/test files under
    `src/trading/contexts/backtest/application/.../no_risk_exact.py`;
  - exports may need updates in package `__init__.py` files.
- risks:
  - Using `request.top_n = 100` inside canonical measured `heap_update` or
    `top_result_proxy_fill` invalidates benchmark comparison.
  - Creating dataclasses, variant hashes, API DTOs, or persisted rows inside
    measured notebook-compatible timers will reproduce the failed attempt.
  - Optimizing proxy fill before matching notebook boundary can hide the real
    parity problem.
- next_focus:
  - First reproduce notebook semantics and stage boundaries literally.
  - Then run local correctness gates.
  - Mac Studio benchmark is the only acceptance benchmark.

Additional context:

- The previous failed attempt can be inspected with `git show 75386d3` only as
  an anti-pattern if needed. It used product-shaped work in measured stages and
  allowed `top_n=100` to contaminate canonical top-K comparison.
- Do not use roadmap docs as source of truth.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the scoped change described in this prompt.
- Preserve Iteration 1/2/3 behavior and tests.
- Use the Russian runtime document as the sole target contract.
- Implement no-risk exact scoring for `risk.mode = "none"` only.
- Dispatch default backend:
  - `event_segments_2_no_risk` for arity 2;
  - `event_segments_n_no_risk` for arity 1 and 3..10;
  - `streaming_2_no_risk` only as fallback/parity comparator.
- Support direction modes:
  - `long_only`;
  - `long_short_reversal`.
- Match notebook entry/exit semantics:
  - entries use open of the next 15m signal bar, mapped to 1m execution index;
  - exits use signal close/reversal, mapped to 1m open;
  - final close uses final 1m close when `close_on_end = true`;
  - no-risk hot path must not allocate full trade lists for every candidate.
- Compute full no-risk summary metrics:
  - `total_return_pct`;
  - `max_drawdown_pct`;
  - `return_over_max_drawdown`;
  - `profit_factor`;
  - `trade_count`;
  - `sharpe_trades`;
  - `win_rate_pct`;
  - `avg_trade_ret_pct`;
  - `avg_trade_exec_bars`;
  - `exposure_pct`.
- Implement self-check against slow reference:
  - validate `trade_count`;
  - validate `total_return_pct` drift within notebook tolerance;
  - fail fast on mismatch.
- Implement notebook-compatible `heap_update`:
  - measured runs use `benchmark_top_k = 5`;
  - sample warmup uses `top_k = 1`;
  - heap key is deterministic by score and original row ids;
  - heap stores only compact notebook-compatible items;
  - no public/storage identity or persisted rows inside this timer.
- Implement notebook-compatible `top_result_proxy_fill`:
  - input is final heap of size `top_k`;
  - sort heap descending by notebook key;
  - recompute proxy only for final top rows where `_proxy_pending = true`;
  - call notebook-equivalent `proxy_for_indicator_rows`;
  - remove `_local_indices` and `_proxy_pending`;
  - return `top_results_count = 5` for canonical measured runs.
- Record/return telemetry with canonical timer names:
  - `service_warmup` if the service path owns a warmup helper;
  - `self_check`;
  - `exact_scoring`;
  - `heap_update`;
  - `top_result_proxy_fill`.
- Unit tests must prove that `request.top_n = 100` does not change canonical
  `benchmark_top_k = 5` behavior.
- If benchmark evidence is produced, include:
  - `request.top_n = 100`;
  - `benchmark_top_k = 5`;
  - `top_results_count = 5`;
  - heap capacity;
  - exact backend display name;
  - implementation id;
  - CPU/RSS/timing evidence.
- The executor has only 2 implementation attempts:
  - Attempt 1 is the initial scoped implementation plus verification.
  - Attempt 2 is one corrective implementation cycle after root-cause analysis of
    a failed gate, parity mismatch, or benchmark miss.
  - If the task still cannot meet the acceptance criteria after Attempt 2, stop.
    Do not start a third code-change cycle, do not mask the failure, and report
    the exact blocker, failed evidence, and recommended next investigation.

## Requirements (Should)

- Prefer literal notebook stage boundaries before micro-optimizing.
- Keep hot-path arrays contiguous and low-allocation.
- Reuse `BacktestComboPlanningService`, `iter_combo_chunks`,
  `build_local_row_pools`, `make_combo_idx_matrix`, and existing prepared pools.
- Keep result rows internal and compact; add public/API DTOs only if required for
  tests, not for public contracts.
- Add direct tests for arity 1, arity 2, arity 3, and at least one arity 8..10
  small correctness smoke.
- Add tests for both direction modes.
- Add tests proving `top_result_proxy_fill` recomputes proxy metadata only for
  final top rows in pass-through/no active proxy mode.
- Add tests proving active proxy metadata is preserved when proxy filter provided
  `confirm`/`proxy`.
- If Mac Studio is unavailable, prepare the benchmark runner shape but report
  acceptance benchmark as not run.

## Requirements (Nice-to-have)

- Add small golden fixture helpers that make notebook parity failures easy to read.
- Add service-only diagnostic subsegments only if they do not alter canonical
  timer boundaries.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot,
   if available and relevant; ignore unrelated stale entries
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

Expand context only for:

- blockers,
- failing quality gates,
- unclear contracts,
- benchmark threshold conflicts,
- architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules,
  - runtime source of truth,
  - latest accepted prior stage evidence
- `task_entrypoints`:
  - canonical no-risk notebook methods,
  - current Iteration 3 service/DTO contracts,
  - exact Iteration 4 stage boundaries
- `conditional_bundles`:
  - read only when the stated condition applies
- `consult_if_needed`:
  - read only for blockers, ambiguity, or conflict resolution

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation and during reporting
  for stage boundary, baseline, and performance claims.
- `numba`: use during implementation for exact scoring kernels and Numba typing.
- `contract-impact-analysis`: use before changing DTOs/exports or any identity
  field to avoid accidental public/storage contract changes.
- `backend-quality-gates`: use during verification and failure triage.
- `root-cause-debugging`: use if parity, performance, or previous-stage tests fail.

Plan:

1. Confirm the exact Iteration 4 scope from the runtime doc and note non-goals.
2. Inspect notebook methods listed in the entrypoint only for no-risk semantics.
3. Inspect current Iteration 3 service/DTO contracts and decide the no-risk
   service input shape.
4. Add internal DTOs for no-risk scoring if needed. Keep them internal and free
   of public/storage identity.
5. Implement no-risk exact scoring kernels and dispatch:
   `event_segments_2_no_risk`, `event_segments_n_no_risk`, and
   `streaming_2_no_risk`.
6. Implement slow reference/self-check for bounded first candidates.
7. Implement notebook-compatible heap/top-K and proxy-fill stages with explicit
   `benchmark_top_k` separate from request `top_n`.
8. Add targeted unit tests for correctness, boundary protection, direction modes,
   and top-K cardinality.
9. Run quality gates.
10. If a gate fails, perform root-cause analysis and use the single allowed
    corrective cycle as Attempt 2.
11. If Attempt 2 still fails, stop implementation work and prepare the final
    blocker report. Do not continue with additional fixes.
12. If on Mac Studio, run/record Iteration 4 benchmark evidence. If not, report
    that benchmark acceptance was not run.

# Acceptance criteria (Definition of Done)

- Source includes no-risk exact scoring implementation under v2 application
  services and internal DTOs if needed.
- Package exports are updated only for internal service/DTO symbols that tests or
  future iterations need.
- `event_segments_2_no_risk` and `event_segments_n_no_risk` produce parity with
  slow reference for targeted fixtures.
- `streaming_2_no_risk` remains fallback/parity comparator, not default.
- Both `long_only` and `long_short_reversal` direction modes are tested.
- No-risk full summary metrics are present and tested at service level.
- `heap_update` tests prove top-K capacity is `benchmark_top_k = 5` for canonical
  measured runs even when request carries `top_n = 100`.
- `top_result_proxy_fill` tests prove proxy recompute is limited to final top rows.
- No code in Iteration 4 generates public `variant_key`, `variant_hash`,
  `indicator_variant_hash`, persisted row objects, DB writes, API DTOs, or lazy
  trades payloads.
- All quality gates listed below are run and pass, or unrelated existing failures
  are explicitly classified.
- Mac Studio benchmark evidence is created only from a real Mac Studio run; no
  synthetic benchmark acceptance files are committed.

# Implementation constraints

- Use `apply_patch` or normal editor tooling for manual edits; do not create
  files with shell redirection.
- Keep diffs scoped to Iteration 4.
- Do not modify notebooks.
- Do not modify canonical benchmark target evidence.
- Do not alter Iteration 1/2/3 accepted behavior.
- Do not read or depend on legacy roadmap docs.
- Do not use `__pycache__` or Numba cache artifacts as implementation source.
- Keep public API, DB schema, job orchestration, lazy trades, and UI untouched.
- If reverted commits are inspected, treat them as anti-pattern context only.
- Avoid object-heavy work inside measured `heap_update` and
  `top_result_proxy_fill`; preserve notebook-compatible timing boundaries.
- Separate these concepts in code and telemetry:
  - `request.top_n`;
  - `benchmark_top_k`;
  - actual `top_results_count`;
  - service-only future `top_result_assembly`.

# Files to indicate (expected touched areas)

Expected primary touches:

- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `src/trading/contexts/backtest/application/dto/__init__.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_no_risk_exact_scoring_notebook_topk/`

# Non-goals

- Persisted top-N rows.
- Public `variant_key`.
- `variant_hash`.
- `indicator_variant_hash`.
- `top_result_assembly`.
- API DTO/read-model assembly.
- Product `top_n = 100` performance acceptance.
- TP/SL risk-on exact scoring.
- `hit_times/15m` loading.
- TP/SL grid validation.
- Lazy trades.
- API routes.
- UI.
- Job orchestration.
- DB migrations.
- Roadmap doc cleanup.
- Notebook edits.

# Quality gates (must run and pass)

Run:

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py
```

Then run:

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2
uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest
uv run pyright
git diff --check
```

If Markdown docs or benchmark docs are touched, also run:

```bash
uv run python -m tools.docs.generate_docs_index --check
```

Mac Studio benchmark acceptance:

- Run only on Mac Studio.
- Use canonical benchmark target:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.
- Compare no-risk arity 1..7 x both direction modes.
- Record `request.top_n = 100`, `benchmark_top_k = 5`,
  `top_results_count = 5`, heap capacity, backend display name,
  implementation id, CPU/RSS, timings, and correctness/parity result.
- Do not mark Iteration 4 accepted without this evidence.

# Final output: report format (strict)

Report in Russian with exactly these sections:

## Изменения

- What files changed and what behavior was added.

## Контрактное влияние

- State explicitly: public API yes/no, persistence schema yes/no,
  public/storage identity yes/no, benchmark contract yes/no.

## Проверки

- List commands run and results.

## Benchmark/Evidence

- State whether Mac Studio benchmark ran.
- If it ran, give path to evidence folder and pass/fail summary.
- If it did not run, state that Iteration 4 is implemented but not accepted.

## Остаточные риски

- List blockers, missing evidence, or follow-up work.
- If two implementation attempts were exhausted, state that explicitly and include
  the failed gates, root cause found, and why a third attempt was not started.
