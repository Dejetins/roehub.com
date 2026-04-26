---
prompt_name: backtest-service-artifact-runtime-v1-iteration-2-prepare-pools
repo: roehub.com
branch: main
scope: "Реализовать Iteration 2: artifact array mmap loaders, [start,end) slicing, signal row extraction, row prefilter, compressed signal segments and `prepare_pools` benchmark gate."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and delivery rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "sole v1 implementation source"

  task_entrypoints:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 2 scope and prepare_pools contract"
      inspect_symbols:
        - "Измеряемая стадия бенчмарка: `prepare_pools`"
        - "Итерация 2: artifact arrays и `prepare_pools`"
        - "Политика бенчмарков"
        - "Матрица тестов"
    - path: src/trading/contexts/backtest/application/services/v2/preflight.py
      why: "Iteration 1 normalized request and runtime config"
      inspect_symbols:
        - "BacktestPreflightService"
        - "BacktestRuntimeConfig"
        - "SUPPORTED_BACKTEST_TIMEFRAMES_V1"
        - "BACKTEST_RISK_MODES_V1"
    - path: src/trading/contexts/backtest/application/dto/runtime_preflight.py
      why: "normalized request DTOs and artifact metadata shape"
      inspect_symbols:
        - "BacktestPreflightResult"
        - "BacktestArtifactMetadata"
        - "BacktestCoordinates"
        - "BacktestCostEstimate"
    - path: src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
      why: "trusted artifact filenames and manifest contracts"
      inspect_symbols:
        - "PRICES_DIRECTORY_LITERAL_V2"
        - "SIGNALS_DIRECTORY_LITERAL_V2"
        - "MAPPINGS_DIRECTORY_LITERAL_V2"
        - "OPEN_TIME_FILENAME_V2"
        - "OHLCV_FILENAME_V2"
        - "SIGNALS_FILENAME_V2"
        - "BAR_OPEN_MAPPING_FILENAME_V2"

  conditional_bundles:
    artifact_context_iteration_1:
      read_when: "reusing current pointer/root resolution from Iteration 1"
      paths:
        - src/trading/contexts/backtest/application/ports/artifact_context.py
        - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_context_resolver.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_validator.py

    artifact_path_patterns:
      read_when: "implementing filesystem mmap readers or path derivation"
      paths:
        - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/path_builder.py
        - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py

    existing_test_patterns:
      read_when: "creating prepare_pools and artifact array loader tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_backtest_preflight_service.py
        - tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_context_resolver.py
        - tests/unit/contexts/backtest/application/services/test_signals_from_indicators_v1.py
        - tests/unit/contexts/backtest/application/services/v2/test_yaml_backtest_artifact_loader_v2.py

    benchmark_targets:
      read_when: "writing Iteration 2 evidence or comparing prepare_pools targets"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md

  consult_if_needed:
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      read_when: "exact method semantics are ambiguous; inspect only named prepare_pools cells"
    - path: docs/architecture/backtest/deep-research-report_for_btcusdt_15m_research_engine.md
      read_when: "need prior explanation of extraction/prefilter performance tradeoffs"

style_references:
  - src/trading/contexts/backtest/application/services/v2/preflight.py
  - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_context_resolver.py
  - src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py

hard_requirements:
  runtime_doc_is_sole_implementation_source: true
  iteration_0_1_are_complete: true
  no_roadmap_docs_as_context: true
  no_scoring_or_combo_planning: true
  no_job_execution_or_persistence_changes: true
  no_internal_api: true
  no_user_supplied_artifact_paths: true
  preserve_public_api_contract: true
  benchmark_acceptance_macstudio_only: true
  keep_existing_user_changes: true

task_toggles:
  implement_mmap_array_loaders: true
  implement_time_range_slicing: true
  implement_signal_row_extraction: true
  implement_row_prefilter: true
  implement_compressed_signal_segments: true
  expose_prepare_pools_timing: true
  add_targeted_tests: true
  add_iteration_2_benchmark_record: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "implementing or reporting prepare_pools timing, memory, CPU, or benchmark status"
    timing: before implementation
    reason: "owns baseline selection, measurement method, and performance claims"
  - skill: numba
    use_when: "adding or changing @njit/prange kernels for row prefilter or segment building"
    timing: during implementation
    reason: "owns Numba typing, threading, and kernel performance choices"
  - skill: contract-impact-analysis
    use_when: "changing DTOs, ports, config, request hash inputs, artifact identity, or public API payloads"
    timing: during investigation
    reason: "owns boundary compatibility and rollout impact"
  - skill: backend-quality-gates
    use_when: "running pytest, ruff, pyright, docs checks, or triaging failures"
    timing: during verification
    reason: "owns backend verification gates"
  - skill: root-cause-debugging
    use_when: "an existing Iteration 1 test or artifact fixture fails unexpectedly"
    timing: if blocker
    reason: "owns root-cause investigation before fixes"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "prepare_pools"
  - "artifact_manifest_load"
  - "artifact_array_mmap_load"
  - "time_range_slice"
  - "signal_row_selection"
  - "row_prefilter"
  - "segment_build"
  - "extract_signal_rows"
  - "fused_row_prefilter_stats"
  - "build_signal_segments"
  - "fill_signal_segments_i8"
  - "prepare_indicator_pool"
  - "prepare_indicator_pools"
  - "np.load(..., mmap_mode=\"r\")"
  - "hit_times/15m"
  - "15m"
  - "1m"
  - "[start, end)"

non_goals:
  - "Implement build_exact_context, build_proxy_context, combo_iteration, proxy_filter, self_check, exact_scoring, heap_update, or top_result_proxy_fill."
  - "Implement TP/SL hit-time loading; Iteration 5 owns load_hit_times and tp_sl_grid_validation."
  - "Create jobs, persist top-N summaries, add job worker orchestration, or change DB schema."
  - "Change public API routes except optional internal wiring needed to call prepare_pools from tests."
  - "Read or patch roadmap docs."
  - "Edit notebooks."
  - "Add UI."

final_report_format:
  language: ru
  sections:
    - "Изменения"
    - "Контрактное влияние"
    - "Проверки"
    - "Benchmark/Evidence"
    - "Остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py"
    expect: "passes after creating targeted tests; if filenames differ, run the exact created test files"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2 tests/unit/contexts/backtest/adapters/outbound/artifacts_fs"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes, or unrelated existing failures are explicitly classified"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if Markdown docs were touched"
  - cmd: "git diff --check"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/prepare_pools.py"
  - "src/trading/contexts/backtest/application/dto/"
  - "src/trading/contexts/backtest/application/ports/"
  - "src/trading/contexts/backtest/adapters/outbound/artifacts_fs/"
  - "tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py"
  - "tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py"
  - "docs/architecture/backtest/benchmark_iterations/<date>_iteration_2_prepare_pools/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "src/trading/contexts/backtest/application/ports/__init__.py"
  - "src/trading/contexts/backtest/adapters/outbound/artifacts_fs/__init__.py"
  - "tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py"
  - "docs/architecture/backtest/benchmark_iterations/README.md"

safety_notes:
  - "Do not read roadmap docs as implementation context; use the runtime document as the sole target contract."
  - "Do not load full `.npy` arrays into memory when mmap is enough; copy only bounded selected signal rows."
  - "Do not accept artifact paths from request payloads; all paths come from trusted config/current pointer/manifest identity."
  - "If not running on Mac Studio, record local timing only as non-acceptance developer evidence."
  - "Preserve Iteration 1 preflight/runtime-defaults behavior unless this prompt explicitly requires a compatible extension."
---

# Task

Реализовать `Iteration 2` из `Backtest Service Artifact Runtime v1`: artifact
array loaders и measured stage `prepare_pools`.

Iteration 0/1 считаются завершенными. Эта итерация стартует после working
`POST /backtests/preflight` / `GET /backtests/runtime-defaults` и должна взять
normalized request + resolved artifact context как вход. Результатом является
service-level preparation layer для будущих compute stages: mmap array loading,
`[start, end)` slicing, signal row extraction, row prefilter, compressed signal
segments и timing `prepare_pools`.

Done means:

- service умеет по normalized request и artifact context подготовить indicator pools;
- loaders читают `prices`, `signals`, `mappings` через `np.load(..., mmap_mode="r")`;
- `[start, end)` slicing по 15m `open_time` детерминирован и протестирован;
- 15m return intervals и 15m-to-1m execution mapping для no-risk готовы;
- signal rows извлекаются через bounded contiguous copy, без full-array materialization;
- row prefilter и compressed signal segments реализованы по notebook semantics;
- timing `prepare_pools` и optional subsegments пишутся в evidence;
- Iteration 2 benchmark record создан перед переходом к Iteration 3.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Iteration 0/1 считаются выполненными пользователем;
  - canonical implementation source: `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`;
  - canonical notebook algorithm: `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
  - canonical target evidence: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`;
  - Iteration 1 produced normalized request, runtime defaults, preflight, request hash, cost estimate, and artifact metadata.
- open_items:
  - prepare_pools runtime path is not implemented;
  - artifact array mmap loaders are not implemented for the new service runtime;
  - row metadata/order hash needs parity evidence against notebook fixture;
  - Mac Studio benchmark record for Iteration 2 must be written before acceptance.
- contract_changes:
  - no public API change should be required for Iteration 2;
  - any new DTO/port is internal application/runtime contract only;
  - artifact path identity remains trusted config/current pointer/manifest based;
  - `hit_times/15m` remains documented but TP/SL table loading is not part of this iteration.
- touched_paths:
  - expected additions under `src/trading/contexts/backtest/application/services/v2/`;
  - expected filesystem artifact loader adapter under `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/`;
  - expected targeted unit tests and benchmark evidence folder.
- risks:
  - accidental advanced indexing can create large copies and miss the 90% benchmark target;
  - reading full `.npy` arrays instead of mmap can regress memory;
  - mixing Iteration 3 combo planning/scoring into this stage will blur benchmark evidence;
  - stale legacy runtime/scorer code may conflict with current notebook semantics.
- next_focus:
  - after Iteration 2 passes, Iteration 3 implements `build_exact_context`, `build_proxy_context`, `combo_iteration`, and `proxy_filter`.

Additional context:

- Canonical benchmark acceptance matrix covers arity `1..7`, risk modes
  `none` and `tp_sl_grid`, direction modes `long_only` and
  `long_short_reversal`.
- Iteration 2 benchmark gate compares `prepare_pools` against canonical notebook
  target for the same `{arity, risk_mode, direction_mode, backend}` tuple.
- Optional subsegments should include `artifact_manifest_load`,
  `artifact_array_mmap_load`, `time_range_slice`, `signal_row_selection`,
  `row_prefilter`, and `segment_build`.
- Acceptance benchmark evidence must come from `Mac Studio`; local timing is useful
  only as developer smoke.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only the scoped change described in this prompt.
- Preserve all explicitly protected contracts and invariants.
- Add or update targeted tests where needed.
- Update related exports / nearby docs when required.
- Keep the implementation deterministic and reviewable.
- Reuse Iteration 1 normalized request/runtime config/artifact metadata instead of rebuilding request validation.
- Implement an internal application service for `prepare_pools`.
- Implement or extend outbound artifact array loader ports/adapters for `prices`, `signals`, and `mappings`.
- Use trusted artifact root/current pointer/manifest-derived paths only.
- Load arrays with `np.load(..., mmap_mode="r")` by default.
- Slice 15m bars by `[start, end)` using 15m `open_time`.
- Load `prices/<tf>`, `prices/1m`, `mappings/<tf>`, and requested `signals/<tf>/<indicator_id>/signals.i8.npy`.
- Derive 15m return intervals from close prices.
- Derive no-risk 15m-to-1m execution mapping: signal on 15m bar `t` enters on open of next 15m bar, mapped to 1m.
- Implement source/window row mapping from normalized indicators to artifact signal row ids.
- Copy only requested signal rows into contiguous `int8` matrices.
- Implement row prefilter equivalent to notebook `fused_row_prefilter_stats` and `topk_fraction_idx`.
- Implement compressed signal segments equivalent to `build_signal_segments` and `fill_signal_segments_i8`.
- Return indicator pools containing `trade_T`, `eval_T`, segments, selected row ids, row scores, row metadata, `trade_T` length and evaluation length.
- Expose measured timing `prepare_pools` and optional subsegments.
- Produce deterministic row metadata/order hash for benchmark evidence.
- Do not implement combo planning, exact context, self-check, scoring, heap/top-N, persistence, or lazy trades.

## Requirements (Should)

- Keep tight-loop code data-oriented and allocation-aware.
- Prefer small internal DTOs for prepared pools over loose dictionaries if it improves testability without slowing the hot path.
- Keep Numba kernels isolated from filesystem and request DTOs.
- Use notebook method names in comments/test names only where they clarify parity.
- Add small fixture tests for contiguous row selection vs non-contiguous selection behavior.
- Add failure tests for missing price/signal/mapping artifacts and invalid time-range coverage.
- Store benchmark evidence under `docs/architecture/backtest/benchmark_iterations/<date>_iteration_2_prepare_pools/`.

## Requirements (Nice-to-have)

- Add row/order hash helper reusable by Iteration 3.
- Add local micro timing for `signal_row_selection`, `row_prefilter`, and `segment_build`.
- Add a small memory smoke that asserts selected signal copy shape stays bounded by selected rows x sliced time.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest executor final report for Iteration 1, if available in the current task context
3. task entrypoints
4. only the conditional bundle(s) required by touched contracts or failing checks
5. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources. Do not read roadmap docs for this task.
Do not treat `.codex/agents/.context/promt_manager_state.yaml` as authoritative
for this task if it discusses legacy exact-no-risk parity work instead of the
current artifact-runtime v1 plan.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- changed contracts are identified;
- touched files are bounded;
- acceptance criteria are implementable without ambiguity;
- no unresolved artifact path, DTO/port, benchmark, or public API ambiguity remains.

Expand context only for:

- blockers;
- failing quality gates;
- unclear contracts;
- benchmark threshold conflicts;
- artifact layout ambiguity;
- architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules and canonical runtime contract;
- `task_entrypoints`: Iteration 2 scope, Iteration 1 runtime request shape, artifact constants;
- `conditional_bundles`: artifact resolver/path patterns, tests, and benchmark targets only when needed;
- `consult_if_needed`: notebook cells or deep research only when prepare_pools semantics are ambiguous.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation for baseline/metric plan and during verification for benchmark reporting.
- `numba`: use during implementation only if adding/changing JIT kernels for row prefilter or segment building.
- `contract-impact-analysis`: use during investigation only if DTOs, ports, config, hashes, artifact identity, or public API behavior changes.
- `backend-quality-gates`: use during verification for pytest/ruff/pyright/docs checks.
- `root-cause-debugging`: use only if Iteration 1 tests or artifact fixtures fail unexpectedly.

1. Confirm Iteration 1 request/artifact context outputs and decide the smallest internal input DTO for `prepare_pools`.
2. Define artifact array loader port and filesystem adapter, reusing trusted artifact contracts and path conventions.
3. Implement mmap loading for required prices, mappings, and signal arrays.
4. Implement deterministic `[start, end)` slicing by 15m `open_time`.
5. Implement source/window to artifact signal row id resolution, or reuse Iteration 1 row materialization if already present.
6. Implement bounded contiguous signal row extraction.
7. Implement 15m returns and no-risk 15m-to-1m execution mapping.
8. Implement row prefilter and compressed signal segments with notebook-equivalent semantics.
9. Return prepared indicator pools plus timings/subsegment metrics.
10. Add targeted tests and failure tests.
11. Run quality gates.
12. If on Mac Studio, run Iteration 2 benchmark and write evidence record before calling the iteration accepted.

# Acceptance criteria (Definition of Done)

- `prepare_pools` can run from a normalized Iteration 1 request and resolved artifact context.
- The service reads artifact arrays through mmap and copies only bounded selected signal rows.
- `[start, end)` slicing uses 15m `open_time` and excludes `end`.
- No-risk execution mapping enters at next 15m open mapped to 1m, with final bounds handled deterministically.
- Per-indicator row metadata includes `{indicator_id, row_id, source, window}`.
- Row prefilter produces deterministic selected row ids, scores, nonzero counts, proxy values and change counts.
- Compressed segments include `starts`, `ends`, `values`, `counts`, and `change_count`.
- `prepare_pools` timing and optional subsegments are recorded.
- Row metadata/order hash matches notebook-derived fixture where a fixture is available.
- Unit tests cover success, missing artifacts, invalid time coverage, row selection, prefilter, segments, and mapping bounds.
- No combo planning, exact scoring, top-N persistence, job orchestration, TP/SL hit-time table loading, or UI code is added.
- Mac Studio benchmark evidence exists for acceptance; if not on Mac Studio, final report clearly says acceptance benchmark is pending.

# Implementation constraints

- Use `apply_patch` for manual edits.
- Preserve DDD / ports-and-adapters direction.
- Domain/application code must not import FastAPI.
- Filesystem adapters must not depend on public request payload paths.
- Avoid broad refactors of Iteration 1 code unless directly required.
- Do not read or patch roadmap docs.
- Do not edit notebooks.
- Do not add dependencies unless unavoidable and justified.
- Keep comments sparse and focused on non-obvious benchmark/parity choices.

# Files to indicate (expected touched areas)

Expected primary and possible secondary touched areas are listed in front matter.

If you choose different filenames or module boundaries, explain why and map them
back to the architecture document's target structure. If you touch public API,
persistence, or config schema, explicitly classify the contract impact.

# Non-goals

- No `build_exact_context`.
- No `build_proxy_context`.
- No `combo_iteration`.
- No `proxy_filter`.
- No `self_check`.
- No `exact_scoring` or `tp_sl_exact_scoring`.
- No `heap_update` or `top_result_proxy_fill`.
- No `load_hit_times` or `tp_sl_grid_validation`.
- No job create/status/top/list/cancel work.
- No lazy trades.
- No UI.
- No roadmap cleanup.

# Quality gates (must run and pass)

Run the commands listed in front-matter `quality_gates`.

If a listed test file does not exist until this iteration creates it, create the
targeted tests first and then run the command. If `uv run pyright` has unrelated
pre-existing failures, include enough output to prove they are unrelated.

For benchmark evidence:

- On non-Mac Studio hosts, record local smoke timings only as non-acceptance evidence.
- On Mac Studio, compare `prepare_pools` against canonical notebook target for arity
  `1..7` using the canonical fixture from `2026-04-26_engine_test_btcusdt_15m`.
- Record wall time, CPU time, process CPU percent, peak RSS/RSS delta, thread count,
  Numba thread count where relevant, request hash, artifact manifest hash, row
  metadata/order hash, and pass/fail against the 90% threshold.
- Write evidence under:
  `docs/architecture/backtest/benchmark_iterations/<date>_iteration_2_prepare_pools/`.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Изменения**
   - Files changed and behavior added.

2. **Контрактное влияние**
   - public API, DTO, port, persisted schema, config schema, request hash/cache identity, artifact identity, benchmark gate impact as yes/no.

3. **Проверки**
   - Commands run and result.

4. **Benchmark/Evidence**
   - Local timing, Mac Studio acceptance status, request/artifact/hash evidence, prepare_pools ratios.

5. **Остаточные риски**
   - Anything blocking Iteration 3 or production acceptance.
