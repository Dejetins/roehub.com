---
prompt_name: backtest-service-artifact-runtime-v1-iteration-3-combo-planning-contexts
repo: roehub.com
branch: main
scope: "Реализовать Iteration 3: backend registry, exact/proxy contexts, deterministic combo iteration и proxy_filter перед exact scoring."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and delivery rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "sole v1 implementation source"
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/benchmark_summary.md
      why: "accepted prior stage contract"

  task_entrypoints:
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 3 scope and stage semantics"
      inspect_symbols:
        - "Охват алгоритма и backends"
        - "Измеряемая стадия бенчмарка: `build_exact_context`"
        - "Измеряемая стадия бенчмарка: `build_proxy_context`"
        - "Измеряемая стадия бенчмарка: `combo_iteration`"
        - "Измеряемая стадия бенчмарка: `proxy_filter`"
        - "Итерация 3: combo planning contexts"
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "Iteration 2 output consumed by Iteration 3"
      inspect_symbols:
        - "BacktestPreparePoolsService"
        - "BacktestPreparePoolsResult"
        - "notebook_compatible_prepare_pools_core_s"
    - path: src/trading/contexts/backtest/application/dto/prepare_pools.py
      why: "prepared pool DTOs and segment/eval shapes"
      inspect_symbols:
        - "PreparedIndicatorPool"
        - "PreparedSignalSegments"
        - "BacktestPreparePoolsResult"
    - path: tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py
      why: "fixture style and Iteration 2 regression surface"
      inspect_symbols:
        - "prepare_pools"
        - "segments"
        - "timing"

  conditional_bundles:
    notebook_iteration_3_semantics:
      read_when: "exact packing, chunk order, or proxy filter behavior is ambiguous"
      paths:
        - tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb

    preflight_runtime_config:
      read_when: "adding combo planning config, backend selection, or guardrail defaults"
      paths:
        - src/trading/contexts/backtest/application/services/v2/preflight.py
        - src/trading/contexts/backtest/application/dto/runtime_preflight.py

    existing_exports:
      read_when: "new DTO/service modules need package exports"
      paths:
        - src/trading/contexts/backtest/application/services/v2/__init__.py
        - src/trading/contexts/backtest/application/dto/__init__.py

    benchmark_targets:
      read_when: "creating Iteration 3 benchmark evidence or comparing timer names"
      paths:
        - docs/architecture/backtest/benchmark_iterations/README.md
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_1_request_normalization_artifact_context/benchmark_summary.md
        - docs/architecture/backtest/benchmark_iterations/2026-04-26_iteration_2_prepare_pools/benchmark_summary.md

  consult_if_needed:
    - path: docs/architecture/backtest/deep-research-report_for_btcusdt_15m_research_engine.md
      read_when: "need prior explanation of notebook combo/proxy tradeoffs"
    - path: configs/prod/indicators.yaml
      read_when: "guardrail or indicator catalog behavior is unclear"

style_references:
  - src/trading/contexts/backtest/application/services/v2/preflight.py
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py

hard_requirements:
  runtime_doc_is_sole_implementation_source: true
  iteration_1_and_2_are_accepted: true
  no_roadmap_docs_as_context: true
  no_exact_scoring_or_heap_top_n: true
  no_hit_times_loading: true
  no_job_execution_or_persistence_changes: true
  no_public_api_shape_change: true
  deterministic_combo_ordering: true
  benchmark_acceptance_macstudio_only: true
  keep_existing_user_changes: true

task_toggles:
  implement_backend_registry: true
  implement_build_exact_context: true
  implement_build_proxy_context: true
  implement_combo_iteration: true
  implement_proxy_filter_pass_through: true
  implement_proxy_filter_active: true
  add_targeted_tests: true
  add_iteration_3_benchmark_record: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "implementing or reporting timer, memory, CPU, or benchmark status"
    timing: before implementation
    reason: "owns baseline selection, stage boundary, and performance claims"
  - skill: numba
    use_when: "adding or changing @njit/prange kernels for proxy_filter"
    timing: during implementation
    reason: "owns Numba typing, threading, and kernel performance choices"
  - skill: contract-impact-analysis
    use_when: "changing DTOs, runtime config, backend ids, or timing payloads"
    timing: during investigation
    reason: "owns boundary compatibility and rollout impact"
  - skill: backend-quality-gates
    use_when: "running pytest, ruff, pyright, docs checks, or triaging failures"
    timing: during verification
    reason: "owns backend verification gates"
  - skill: root-cause-debugging
    use_when: "Iteration 1/2 tests regress or benchmark evidence contradicts code"
    timing: if blocker
    reason: "owns root-cause investigation before fixes"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "event_segments_2_no_risk"
  - "event_segments_n_no_risk"
  - "streaming_2_no_risk"
  - "event_segments_n_tp_sl_15m_grid"
  - "build_exact_context"
  - "build_proxy_context"
  - "combo_iteration"
  - "proxy_filter"
  - "build_segment_stack"
  - "build_eval_stack"
  - "build_combo_proxy_cache_two"
  - "gather_combo_proxy_cache_two"
  - "proxy_prefilter_combos_chunk_two"
  - "proxy_prefilter_combos_chunk_n"
  - "topk_fraction_idx"
  - "iter_combo_chunks"
  - "combo_top_frac"
  - "combo_min_confirm"
  - "COMBO_CHUNK_SIZE"
  - "4096"
  - "prepare_pools_core"
  - "hit_times/15m"

non_goals:
  - "Implement self_check, exact_scoring, tp_sl_exact_scoring, heap_update, top_result_proxy_fill, or lazy trades."
  - "Load TP/SL hit-time arrays; Iteration 5 owns load_hit_times and tp_sl_grid_validation."
  - "Create jobs, persist top-N summaries, change DB schema, or add worker orchestration."
  - "Change public API routes or UI behavior."
  - "Read or patch roadmap docs."
  - "Edit notebooks."
  - "Treat prepare_pools_total as the notebook-compatible comparison metric."

final_report_format:
  language: ru
  sections:
    - "Изменения"
    - "Контрактное влияние"
    - "Проверки"
    - "Benchmark/Evidence"
    - "Остаточные риски"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py"
    expect: "passes after creating targeted tests; if filenames differ, run exact created test files"
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
  - "src/trading/contexts/backtest/application/services/v2/combo_planning.py"
  - "src/trading/contexts/backtest/application/dto/combo_planning.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_3_combo_planning_contexts/"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "src/trading/contexts/backtest/application/services/v2/preflight.py"
  - "src/trading/contexts/backtest/application/dto/runtime_preflight.py"
  - "docs/architecture/backtest/benchmark_iterations/README.md"

safety_notes:
  - "Use the Russian runtime document as the sole target contract; roadmap wording is not source of truth."
  - "Preserve Iteration 1/2 behavior and benchmark evidence; do not relax accepted gates."
  - "Keep pass-through proxy_filter near-zero in the canonical default: combo_top_frac = 1.0 and combo_min_confirm = 1."
  - "Build context arrays only when the selected backend needs them; arity 2 no-risk specialized path may read segments directly."
  - "If not running on Mac Studio, record local timing only as non-acceptance developer evidence."
---

# Task

Реализовать `Iteration 3` из `Backtest Service Artifact Runtime v1`: combo
planning contexts перед exact scoring.

Iteration 1 и Iteration 2 уже приняты. Вход этой итерации — результат
`prepare_pools_core` / `BacktestPreparePoolsResult`: filtered indicator pools,
compressed signal segments, eval arrays, 15m returns и execution mapping. Выход
этой итерации — runtime layer, который выбирает backend, готовит exact/proxy
contexts, детерминированно перечисляет candidate combinations чанками и
применяет optional proxy prefilter. Exact scoring, TP/SL scoring, heap/top-N и
job persistence не входят в задачу.

Done means:

- backend registry выбирает поддержанный backend для `risk.mode`, arity и direction;
- `build_exact_context` повторяет notebook semantics `build_segment_stack`;
- `build_proxy_context` повторяет semantics `build_eval_stack`,
  `build_combo_proxy_cache_two` и `gather_combo_proxy_cache_two`;
- `combo_iteration` повторяет deterministic Cartesian chunking `iter_combo_chunks`;
- `proxy_filter` поддерживает pass-through и active modes;
- stage timings записываются как `build_exact_context`, `build_proxy_context`,
  `combo_iteration`, `proxy_filter`;
- tests фиксируют deterministic combo ordering, candidate counts, inactive and
  active proxy-filter behavior;
- Iteration 3 benchmark record готов для Mac Studio acceptance до перехода к
  exact scoring.

## Context / Current State

Context ledger from prior iterations:

- completed:
  - canonical implementation source: `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`;
  - canonical notebook algorithm: `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`;
  - Iteration 1 accepted: request normalization, preflight, artifact context and failure evidence;
  - Iteration 2 accepted: `prepare_pools_core` is the notebook-compatible comparison stage;
  - Iteration 2 strict-total failure is preserved only as `stage_boundary_mismatch` evidence.
- open_items:
  - combo planning runtime path is not implemented;
  - backend registry for v1 compute modes is not implemented;
  - exact/proxy context packing needs notebook parity tests;
  - candidate-count and deterministic ordering evidence must be captured before scoring;
  - active proxy-filter fixture evidence is required even though canonical target default is pass-through.
- contract_changes:
  - new DTO/service contracts are internal runtime contracts unless explicitly exposed;
  - public API request/response shape must remain unchanged in this iteration;
  - no DB schema, job persistence, UI, lazy trades, or internal API changes are allowed;
  - backend ids must match the runtime document.
- touched_paths:
  - likely new service module under `src/trading/contexts/backtest/application/services/v2/`;
  - likely new DTO module under `src/trading/contexts/backtest/application/dto/`;
  - targeted unit tests under `tests/unit/contexts/backtest/application/services/v2/`;
  - benchmark evidence folder for Iteration 3.
- risks:
  - building heavy proxy contexts when pass-through is configured will fail the stage budget;
  - non-deterministic combo order will break variant identity and benchmark parity later;
  - implementing scoring early will blur benchmark boundaries and repeat the Iteration 2 mistake;
  - treating `prepare_pools_total` as target metric is explicitly wrong.
- next_focus:
  - create the minimal internal runtime contract for combo planning;
  - preserve notebook-compatible ordering and array shapes;
  - record stage-level evidence before exact scoring exists.

Additional context:

- Canonical target benchmark uses `combo_top_frac = 1.0` and
  `combo_min_confirm = 1`, so `build_proxy_context` and `proxy_filter` should
  be near-zero pass-through in default runs.
- Active proxy filtering is still part of v1 and needs tests: `combo_top_frac <
  1.0` or `combo_min_confirm > 1` must build/apply the proxy path.
- `COMBO_CHUNK_SIZE` is `4096` in the canonical benchmark.

## Requirements (Must)

- Read context using the protocol below and stop early once sufficient.
- Implement only Iteration 3 as described in the runtime document.
- Preserve all Iteration 1/2 tests and accepted stage-boundary semantics.
- Keep combo ordering deterministic and identical to notebook Cartesian order:
  indicator order from normalized request, local row pool order inside each
  indicator, bounded chunks of `4096` by default.
- Implement backend registry entries:
  - `event_segments_2_no_risk`;
  - `streaming_2_no_risk`;
  - `event_segments_n_no_risk`;
  - `event_segments_n_tp_sl_15m_grid`.
- `build_exact_context` must produce arity-first segment arrays when required:
  - `starts[arity, max_rows, max_segments]`;
  - `ends[arity, max_rows, max_segments]`;
  - `values[arity, max_rows, max_segments]`;
  - `counts[arity, max_rows]`.
- `build_proxy_context` must skip heavy work in pass-through mode and still
  record the stage timing.
- Active arity-2 proxy path must support matrix-backed confirm/proxy lookup
  semantics from `build_combo_proxy_cache_two` and `gather_combo_proxy_cache_two`.
- Active generic-N proxy path must support `eval_stack[arity, max_rows, n_intervals]`
  and `proxy_prefilter_combos_chunk_n` semantics.
- `proxy_filter` pass-through mode must select the whole chunk without changing
  order.
- `proxy_filter` active mode must apply `combo_min_confirm`, top fraction by
  proxy score, deterministic selected indexes, and candidate-count telemetry.
- Add targeted tests for shapes, dtypes where meaningful, deterministic order,
  candidate counts, inactive pass-through, and active proxy filtering.
- If benchmark evidence is written locally, mark it non-acceptance unless it ran
  on Mac Studio. Mac Studio acceptance must include host, commit, artifact
  manifest hash, request hash, stage timings, candidate counts, and pass/fail
  decision.

## Requirements (Should)

- Prefer a small internal service such as `BacktestComboPlanningService` plus
  explicit DTOs rather than expanding `prepare_pools.py`.
- Keep arrays contiguous where later kernels need contiguous memory.
- Avoid full materialization of all combinations; iterate chunks.
- Reuse existing timing/telemetry style from Iteration 2.
- Keep backend selection data-driven enough for later Iteration 4/5 dispatch,
  but do not implement scorer dispatch yet.
- Make active proxy-filter tests use small deterministic fixtures with expected
  selected indexes and counts.

## Requirements (Nice-to-have)

- Expose a clear stage result object that later exact scoring can consume
  without re-packing contexts.
- Include a lightweight local smoke benchmark script or helper if it fits the
  existing test style and does not expand public API.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` or latest state snapshot, if available and relevant
3. latest executor final report or accepted benchmark summary, if available
4. task entrypoints
5. only the conditional bundle(s) required by touched contracts or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- changed contracts are identified;
- touched files are bounded;
- acceptance criteria are implementable without ambiguity;
- backend registry semantics are clear;
- no unresolved public API or persistence-contract ambiguity remains.

Expand context only for:

- blockers;
- failing quality gates;
- unclear contracts;
- benchmark threshold conflicts;
- array shape or ordering ambiguity against the notebook.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`:
  - repository rules;
  - canonical v1 runtime document;
  - latest accepted prior-stage evidence.
- `task_entrypoints`:
  - Iteration 3 stage contract;
  - Iteration 2 result object and tests;
  - existing v2 service style.
- `conditional_bundles`:
  - notebook semantics only for named functions when ambiguity remains;
  - preflight/runtime config only if config/defaults are touched;
  - exports only if new modules require package wiring;
  - benchmark files only when writing evidence.
- `consult_if_needed`:
  - broader analysis docs or catalog only for blockers.

Do not convert this manifest into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `backend-performance-evidence`: use before implementation and again during
  benchmark reporting when defining stage boundaries and performance claims.
- `numba`: use only if adding or changing `@njit` / `prange` proxy kernels.
- `contract-impact-analysis`: use if DTOs, config, backend ids, timing payloads,
  or any public-facing default could change.
- `backend-quality-gates`: use during pytest/ruff/pyright/docs verification and
  failure classification.
- `root-cause-debugging`: use if existing Iteration 1/2 tests fail unexpectedly.

1. Inspect the Iteration 3 runtime-doc sections and current Iteration 2 result
   DTO/service shape.
2. Define internal DTOs for backend registry, exact context, proxy context, combo
   chunks, proxy filter result, and timing/candidate telemetry.
3. Implement backend registry selection with explicit unsupported-mode errors and
   no public API contract change.
4. Implement `build_exact_context` from prepared pools, matching
   `build_segment_stack` shapes and preserving indicator order.
5. Implement `build_proxy_context` with a cheap pass-through branch and active
   arity-2/generic-N context preparation.
6. Implement `combo_iteration` as chunked deterministic Cartesian product with
   counts and no full combo materialization.
7. Implement `proxy_filter` pass-through and active modes with deterministic
   selection output.
8. Add targeted unit tests for backend selection, context shapes/order, chunk
   boundaries/counts, pass-through filter, active filter, and Iteration 2
   regression.
9. Run quality gates. If Mac Studio is available in this execution context,
   create Iteration 3 benchmark evidence; otherwise prepare the benchmark record
   structure and state that acceptance was not run.

# Acceptance criteria (Definition of Done)

- Backend registry returns the expected backend for supported v1 combinations:
  `none` arity 2 specialized, `none` arity 1 and 3..10 generic,
  `tp_sl_grid` arity 1..10 generic, and optional `streaming_2_no_risk`
  fallback/parity comparator.
- Unsupported risk/arity/backend combinations fail explicitly with a typed
  internal error or validation result, not with an accidental `KeyError`.
- `build_exact_context` emits the documented arity-first arrays with correct
  order, shape, counts, and segment values for small fixtures.
- Specialized no-risk arity 2 path may avoid heavy stack allocation, but the
  service must still record `build_exact_context` timing and make the behavior
  explicit in tests.
- `build_proxy_context` pass-through mode avoids heavy proxy cache/stack work
  while recording timing.
- Active arity-2 proxy path computes/gathers confirm/proxy lookup values
  deterministically.
- Active generic-N proxy path applies `eval_stack` semantics and returns
  deterministic selected row indexes.
- `combo_iteration` produces the same Cartesian order as notebook
  `iter_combo_chunks`, preserves indicator order, respects chunk size `4096`,
  and records `cartesian_combinations`, `combo_chunks_processed`, and
  `exact_candidates_evaluated`.
- `proxy_filter` pass-through returns all combinations in original order and
  near-zero stage work.
- `proxy_filter` active mode applies `combo_min_confirm` and `combo_top_frac`,
  returns deterministic selected indexes, and records filtered candidate counts.
- No exact scoring, TP/SL hit-time loading, heap/top-N, job persistence, route,
  UI, or notebook change is introduced.
- All targeted tests and backend quality gates pass or unrelated existing
  failures are explicitly classified.
- Benchmark evidence for Iteration 3 is present or the final report clearly
  states that Mac Studio acceptance was not run.

# Implementation constraints

## Determinism & ordering

- Preserve normalized request indicator order everywhere.
- Preserve each prepared pool row order.
- Use Cartesian order equivalent to `itertools.product` over local row pools.
- Do not use unordered dict/set iteration to determine candidate order.
- Keep selected indexes stable across repeated runs with identical inputs.

## Stage boundaries

- Iteration 3 owns only:
  - `build_exact_context`;
  - `build_proxy_context`;
  - `combo_iteration`;
  - `proxy_filter`.
- Iteration 3 must not include:
  - `self_check`;
  - `exact_scoring`;
  - `tp_sl_exact_scoring`;
  - `heap_update`;
  - `top_result_proxy_fill`;
  - `load_hit_times`;
  - `tp_sl_grid_validation`.
- Do not compare aggregate service overhead to notebook target. Use the corrected
  Iteration 2 stage-boundary rule as precedent.

## API / contracts

- Public API shape must not change.
- Job lifecycle, persistence, variant identity, lazy trades, and UI chart payload
  are out of scope.
- New DTOs are internal application/runtime contracts unless explicitly wired
  into public routes later.
- If timing payload names are added, use exact runtime-doc stage names.

## Performance

- Do not materialize the full Cartesian product for large requests.
- Do not build proxy caches/stacks when pass-through config is active.
- Keep array copies bounded to context preparation needs.
- Make performance claims only with measured evidence and stage names.
- Mac Studio benchmark acceptance is the only acceptance benchmark authority.

## Notebook parity

- Use the notebook only for named Iteration 3 functions when semantics are
  ambiguous:
  - `build_segment_stack`;
  - `build_eval_stack`;
  - `build_combo_proxy_cache_two`;
  - `gather_combo_proxy_cache_two`;
  - `iter_combo_chunks`;
  - `proxy_prefilter_combos_chunk_two`;
  - `proxy_prefilter_combos_chunk_n`;
  - `topk_fraction_idx`.
- Do not edit notebooks.

# Files to indicate (expected touched areas)

Expected primary touched areas:

- `src/trading/contexts/backtest/application/services/v2/combo_planning.py`
- `src/trading/contexts/backtest/application/dto/combo_planning.py`
- `tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py`
- `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_3_combo_planning_contexts/`

Possible secondary touched areas:

- `src/trading/contexts/backtest/application/services/v2/__init__.py`
- `src/trading/contexts/backtest/application/dto/__init__.py`
- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/backtest/application/dto/runtime_preflight.py`
- `docs/architecture/backtest/benchmark_iterations/README.md`

# Non-goals

- Do not implement exact scoring or TP/SL scoring.
- Do not load `hit_times/15m`.
- Do not implement top-N, heap update, lazy trades, chart payloads, or UI.
- Do not create jobs, worker orchestration, persistence, migrations, or public
  API route changes.
- Do not edit notebooks.
- Do not read roadmap docs as implementation context.
- Do not rewrite Iteration 1/2 implementation beyond necessary regression fixes.

# Quality gates (must run and pass)

Run the narrowest meaningful gates first, then broader gates:

1. `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py`
2. `uv run pytest -q tests/unit/contexts/backtest/application/services/v2`
3. `uv run ruff check src/trading/contexts/backtest tests/unit/contexts/backtest`
4. `uv run pyright`
5. `uv run python -m tools.docs.generate_docs_index --check` if Markdown docs were touched
6. `git diff --check`

If a gate fails:

- classify whether it is introduced or pre-existing;
- use `root-cause-debugging` for unexpected regressions;
- fix introduced failures before final report;
- do not hide failures behind broad skips.

Benchmark evidence:

- For local development outside Mac Studio, report timing as non-acceptance
  evidence only.
- For Mac Studio acceptance, write results under
  `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_3_combo_planning_contexts/`
  with:
  - commit hash and git status;
  - artifact manifest hash and request hash;
  - warmup metrics if any JIT path is used;
  - runtime metrics without warmup;
  - `build_exact_context`, `build_proxy_context`, `combo_iteration`,
    `proxy_filter` timings;
  - candidate counts and chunk counts;
  - inactive and active proxy-filter fixture evidence;
  - pass/fail decision.

# Final output: report format (strict)

Ответь на русском и используй только эти секции:

## Изменения

- Что реализовано и где.

## Контрактное влияние

- Public API / DTO / backend id / timing changes.
- Если публичных изменений нет, так и напиши.

## Проверки

- Команды и результат.
- Если команда не запускалась, укажи причину.

## Benchmark/Evidence

- Где лежит evidence.
- Mac Studio acceptance status.
- Stage timings / candidate-count highlights.

## Остаточные риски

- Только реальные остаточные риски или `нет известных`.
