# Backtest v2 Benchmarks (R0 baseline)

Статус: source-of-truth для R0 benchmark/parity baseline  
Связанные документы:
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`

## Status

- Status: active benchmark/protocol reference for the delivered v2 runtime and summary-only UX.
- Umbrella authority note:
  - `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md` is the only
    remaining execution master-plan for parity/program closure;
  - this document remains benchmark/protocol/evidence authority only and does not reopen a second
    parallel roadmap.
- Compatibility note:
  - R10-02 synchronized docs around the shipped runtime and left R10-03 for perf/runbook
    closure only;
  - R10-03 uses this document as the canonical benchmark protocol for legacy R0 reference,
    artifact-backed v2 perf gates, and rollout evidence.
- D10 prerequisite note:
  - `wider TP/SL artifact grids` plus `grid-agnostic Stage B loaders` are a
    `completed prerequisite` for master-plan closure;
  - this document validates and measures that prerequisite together with
    `backtest-precompute-runner-v2.md`, `backtest-runtime-kernels-v2.md`, and the cited unit
    tests rather than reopening it as a new implementation milestone.
  - `RG-TTR contradiction reopens blocker`: if live `RG-TTR` closure evidence contradicts widened
    grid compatibility or grid-agnostic loader behavior, D10 prerequisite status must be reopened
    as a blocking defect before closure.
- Milestone A / EPIC A3 adds one additive rollout corpus on top of this baseline:
  - `docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md`
  - `tests/perf_smoke/contexts/backtest/fixtures/backtest_runtime_acceleration_benchmark_corpus_v1.json`
  - exact baseline slices in A3 reuse the approved R0 scenarios and R5 Stage B golden manifest
    from this document instead of introducing a second exact reference set.
- Milestone B / EPIC B1 exact-core acceleration reuses the same benchmark surface:
  - `exact_baseline` remains the canonical exact evidence anchor;
  - `small_grid_overhead` remains the lightweight small-run overhead check;
  - neither slice changes active default profile selection or implies rollout of `exact_parallel`
    launch policy by itself.
- Historical Milestones A-G naming is kept only for traceability; the active umbrella master-plan
  now reuses one frozen benchmark surface:
  - `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
  - `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
  - internal measurement helpers under
    `src/trading/contexts/backtest/application/services/v2/notebook_parity_benchmark_corpus_v2.py`
  - this surface remains the benchmark authority for `NR2`, `RG-TTR`, and `RG-ALT` after the
    accepted Stage A / Stage B cutover, and its gates stay blocking at closure time.

## Цель

R0 не внедряет runtime v2, а фиксирует:
- какие сценарии считаются representative baseline;
- какие метрики снимаются одинаково на каждом локальном прогоне;
- какие parity fixtures обязательны до cutover;
- где лежат reproducible inputs и как сохранять измерения.

R6-01 не меняет сам R0 baseline, R6-02 добавляет deterministic Stage A shortlist verification,
R6-03 добавляет Stage B artifact-backed risk kernel verification поверх этого baseline, а
R6-04 фиксирует полный runtime ranking contract и summary-only top-N materialization. Для
следующих milestone-сравнений обязательны:

- shared `slot-pinned context` bootstrap для sync/background;
- отсутствие runtime directory scanning;
- отсутствие hot-path hash recomputation;
- explicit mmap loading для `prices/<tf>`, `signals/<tf>/<indicator_id>/signals.i8.npy`,
  `mappings/<tf>/bar_open_1m_idx.u32.npy`,
  `mappings/<tf>/bar_close_1m_idx.u32.npy`, `hit_times/1m/manifest.yaml`.
- deterministic `final_signal` aggregation, compact trade construction и shortlist ordering на
  `artifacts-only inputs`.
- approved runtime ranking literals:
  `total_return_pct`, `max_drawdown_pct`, `return_over_max_drawdown`, `profit_factor`,
  `sharpe_trades`, `win_rate_pct`.
- deterministic Stage B `1m hit-times` risk exits, fast TP/SL search и exact best-cell replay без
  runtime recompute hit-times.
- exact-core orchestration cleanup may reduce loader/task/object overhead, but it must keep the
  same finalists, ranking order, and `variant_key` semantics as the approved exact baseline.
- summary-only runtime rows: ranking payload определяет inclusion в `top_n`, а report/trades тела
  не materialize'ятся в sync/jobs summary paths.
- Milestone C artifact dependency expanded the canonical `hit_times/1m` surface to:
  - `tp_values = [0.5, 1.0, ..., 50.0]` (`100` levels);
  - `sl_values = [0.5, 1.0, ..., 25.0]` (`50` levels);
  - raw table footprint is therefore `300` `uint32` cells per `1m` bar, which keeps the default
    `20_000`-bar incremental tail under `50_000_000` cells while moving bootstrap/full rebuilds
    into a documented multi-gibibyte memory range.

Для umbrella master-plan `v2` этот widened artifact contract читается как delivered prerequisite,
а не как ещё не начатая dependency branch.

## Артефакты R0

| EPIC | Артефакты |
|---|---|
| R0-01 | `docs/architecture/roadmap/base_refactor_plan.md`, `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`, status/superseded markers в v1 docs |
| R0-02 | `tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json`, `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json`, `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py` |
| R0-03 | `configs/<env>/backtest.yaml`, `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`, `apps/api/dto/backtest_runtime_defaults.py` |

## Scenario classes

### `sync small-run`

- Цель: baseline для текущего inline sync hot path с guard-safe grid.
- Локальный surrogate: template-mode `RunBacktestUseCase` + deterministic `1m` candle feed fake + current v1 close-fill scorer.
- Fixture source: `tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json` -> `sync-small-run`.

### `large-run`

- Цель: baseline для более плотного grid и более частого request TF без перехода к worker orchestration.
- Локальный surrogate: тот же use-case/hot path, но с увеличенными `target_bars`, `indicator_windows` и risk axes.
- Fixture source: `tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json` -> `large-run`.

### `background-run`

- Цель: baseline для background-sized запроса, измеряемого на shared compute core до storage/lease cutover.
- Локальный surrogate: тот же hot path, что и у sync, но с размером запроса, соответствующим будущему background workload.
- Важно: PG/lease/polling overhead в R0 не измеряется; R0 фиксирует именно compute/parity baseline.
- Fixture source: `tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json` -> `background-run`.

## Baseline metrics

- `wall_clock_seconds`: elapsed wall time одного локального сценария через `time.perf_counter()`.
- `cpu_time_seconds`: process CPU time сценария через `time.process_time()`.
- `peak_traced_memory_bytes`: peak Python allocation footprint через `tracemalloc`.
- `clickhouse_hot_path_calls`: локальный proxy для hot-path чтения свечей.
  В R0 deterministic baseline это `candle_feed.load_1m_dense(...)` call count.
- `indicator_compute_calls`: локальный proxy для текущего v1 signal/score compute path.

R0 intentionally не фиксирует machine-specific SLA. Проверяется наличие метрик, shape и стабильность протокола, а не одинаковые абсолютные миллисекунды между машинами.

## Notebook parity benchmark contract (A1 artifact, G1/G2 closure authority)

Этот contract не заменяет R0. Он сохраняет A1 fixture names, но на этапе closure фиксирует
active benchmark authority для umbrella master-plan из
`backtest-engine-vnext-parity-corrective-plan-v2.md`.

### Canonical classes

- `NR2`: no-risk two-indicator parity class anchored to
  `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`
- `RG-TTR`: risk-grid `total_return_pct` parity class anchored to
  `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`
- `RG-ALT`: risk-grid alternative-metric functional class for
  `max_drawdown_pct`, `return_over_max_drawdown`, `profit_factor`, `sharpe_trades`,
  `win_rate_pct`

### Measurement contract

Committed fixture:
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`

Executable perf-smoke harness:
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`

Required runtime-shape measurement fields:
- `wall_clock_seconds`
- `cpu_time_seconds`
- `peak_rss_bytes`
- `numba_threads_used`
- `max_python_processes_seen`
- `stage_b_execution_mode`
- `stage_b_process_fallback_threshold`
- `exact_replay_count`

The contract is intentionally internal-only. These fields are benchmark-only evidence and MUST NOT
be exposed through public API routes by this prompt.

### Authority layers

The committed corpus now distinguishes two explicit benchmark-authority layers:

- `synthetic_contract_validation`: local perf-smoke proves deterministic schema, serialization,
  equal-thread-budget normalization, and gate semantics for `NR2`, `RG-TTR`, and `RG-ALT`
- `live_host_measurement`: final corrective closure authority for canonical `NR2` and `RG-TTR`
  stays on explicit benchmark-host captures and cannot be inferred from synthetic perf-smoke alone

This split is intentional: passing local synthetic validation keeps the benchmark contract honest,
but it does not by itself close the umbrella execution master-plan.

### Blocking live host captures

The committed corpus also records explicit `live_host_captures` entries for canonical closure
scenarios:

- `nr2_live_host_canonical`
- `rg_ttr_live_host_canonical`

Each entry keeps one explicit `capture_status` plus an optional nested `captured_measurement`
payload that reuses the same internal runtime-shape contract:

- `wall_clock_seconds`
- `cpu_time_seconds`
- `peak_rss_bytes`
- `numba_threads_used`
- `max_python_processes_seen`
- `stage_b_execution_mode`
- `stage_b_process_fallback_threshold`
- `exact_replay_count`

While `capture_status = missing`, final closure for that scenario remains blocked even if the
synthetic harness passes locally.

Fresh post-D0-D5 `NR2` rerun authority on `2026-04-18` still keeps closure open:

- service persisted `run_id = 0deed418-27a1-4a6a-9633-d9960c1a1d0a` after a cold restart on clean
  `origin/main`; the isolated benchmark run required `ROEHUB_ENV = prod` so the service resolved
  the same published `slot_a` artifacts as the notebook anchor. Direct DB row confirms
  `execution_mode = sync_inline`, `effective_execution_profile_mode = exact_no_risk_parity`,
  `artifact_slot = slot_a`, `artifact_asof_date = 2026-04-14`
- service absolute metrics:
  `46.29263141600018s` wall, `59.251817656s` backend CPU,
  `19904364544` peak RSS bytes, `186.8%` peak CPU,
  cold baseline `201670656` RSS bytes, `1` Python process, top-1 `975.5496672803515%`
- notebook (`NUMBA_NUM_THREADS=4` on the same host / slot, clean `HEAD` anchor):
  `7.155866625000044s` wall, `18.814051s` CPU,
  `1372848128` peak RSS bytes, `397.1%` peak CPU,
  `1` Python process, top-1 `1621.7322019157828%`
- current `NR2` ratios remain far outside both primary and near-target closure gates:
  `wall_clock_ratio = 6.469185892072142x`,
  `cpu_time_ratio = 3.1493386329185564x`,
  `peak_rss_ratio = 14.498591751002483x`,
  `peak_cpu_ratio = 0.47041047595064217x`
- versus the prior post-D0-D5 authority rerun `a6628e23-f662-43b7-8e44-65e9f39c52cf`:
  wall ratio improved slightly (`6.485004443347636x -> 6.469185892072142x`),
  CPU ratio improved slightly (`3.3265449889872802x -> 3.1493386329185564x`),
  peak RSS ratio improved marginally (`14.539277419972706x -> 14.498591751002483x`),
  peak CPU ratio improved slightly (`0.4374299752645998x -> 0.47041047595064217x`),
  but winner parity did not improve because service top-1 remained `975.5496672803515%`
- closure blocker:
  this specific historical rerun still has no
  `backtest_job_stage_a_shortlist.parity_runtime_state_json`, so it remains `capture_status =
  missing`;
  D6 now requires canonical sync persistence to fail fast without
  **DB-backed runtime-shape literals** (`stage_b_execution_mode`,
  `stage_b_process_fallback_threshold`, `exact_replay_count`) before a new `NR2` live capture can
  move to `captured`.

### Equal-thread-budget normalization

All notebook-parity comparisons are valid only under `equal thread budget` rules:

- same host class
- same artifact slot
- identical `numba_threads_used`
- comparable runtime surfaces (`sync`, `worker`, `notebook`) interpreted together with
  `stage_b_execution_mode` and `stage_b_process_fallback_threshold`
- accepted benchmark thread budget currently freezes `max_numba_threads=4`,
  `stage_a_workers=4`, and `stage_b_workers=1` for the canonical `exact_parallel` backend shape

Invalid example:
- notebook on `12` threads vs backend on `4` threads

### Explicit baseline comparison points

The committed benchmark corpus stores explicit reviewable comparison points instead of relying on
memory or prose:

- `NR2` keeps current backend `181.3s` on `4` threads plus notebook references `7.54s` on
  `4` threads and `5.63s` on `12` threads, all on `macstudio-class`
- `RG-TTR` keeps the accepted single-process default comparison point:
  `max_python_processes_seen = 1`, `stage_b_execution_mode = in_process`,
  `stage_b_process_fallback_threshold = none`
- `RG-ALT` keeps the functional first-wave guardrail:
  `runtime_regression_ratio <= 1.10` vs the equal-thread-budget backend baseline

### Acceptance intent

- `NR2`: `wall_clock_ratio <= 1.18`, `peak_rss_ratio <= 1.35`, `single-process default`,
  `stage_b_execution_mode = bypassed_no_risk`,
  `stage_b_process_fallback_threshold = none`
- `RG-TTR`: `wall_clock_ratio <= 1.18`, `single-process default`,
  `stage_b_execution_mode = in_process`,
  `stage_b_process_fallback_threshold = none`, `finalist-only exact replay`,
  `exact_replay_count <= 64`
- `RG-ALT`: correctness first, runtime regression no worse than `10%`, no notebook parity claim in
  the first wave

### Rollout note

Maintainers should treat
`tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
as the blocking synthetic authority for the committed notebook parity contract. Final corrective
closure for canonical `NR2` and `RG-TTR` additionally requires explicit benchmark-host
`live_host_captures` evidence. Captured live benchmark runs remain incomplete if they fail any
`NR2`, `RG-TTR`, or `RG-ALT` gate, even when the docs still match.

## R10-03 closure protocol

R10-03 не меняет runtime/API contract. Closure фиксируется через один deterministic protocol
поверх тех же `r0_benchmark_scenarios.json` fixtures:

- legacy reference path:
  - direct `BacktestStagedRunnerV1` + `CloseFillBacktestStagedScorerV1`;
  - benchmark остаётся approved R0 baseline и больше не идёт через `RunBacktestUseCase`,
    потому что production launch после R10-01 уже artifact-backed.
- artifact-backed path:
  - real `RunBacktestUseCase`;
  - strict local artifact tree;
  - active engine semantics: `signal_tf + 1m_risk`.

### Acceptance thresholds

- `0 CH calls on hot path`
- `0 IndicatorCompute.compute(...) calls on hot path`
- measurable speedup reference:
  - `hot_path_external_calls_total =
    clickhouse_hot_path_calls + indicator_compute_calls`
  - artifact-backed v2 must reduce this counter relative to the approved R0 baseline for every
    representative scenario;
  - current fixture contract encodes `expected_hot_path_cost_reduction_min=2`, which means one
    legacy ClickHouse bootstrap proxy and one legacy indicator-compute hot-path call are both
    eliminated.
- `wall_clock_seconds` и `cpu_time_seconds` продолжают сниматься для operator diagnostics, но не
  становятся CI SLA, потому что на synthetic artifact store они чувствительны к локальному disk
  IO и mmap behavior.

### Closure matrix

| R10-03 scope item | Canonical verification |
|---|---|
| Legacy R0 reference remains executable | `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py::test_r0_baseline_perf_smoke_collects_metric_snapshots` |
| `0 CH calls on hot path` | `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py::test_r10_artifact_v2_perf_gates_reduce_hot_path_cost_vs_r0_baseline` |
| `0 IndicatorCompute.compute(...) calls on hot path` | `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py::test_r10_artifact_v2_perf_gates_reduce_hot_path_cost_vs_r0_baseline` |
| Stage B `signal_tf + 1m_risk` fixture baseline | `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py` |
| Background execution compatibility semantics | `tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py` |
| Summary-only persisted runs/history/detail compatibility | `tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py`, `tests/unit/apps/api/test_backtests_routes.py` |
| Rollout / rollback operations | `docs/runbooks/backtest-rollout-rollback.md` |

## Parity fixture scope

### Stage A no-risk

- Источник: `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json` -> `stage_a_no_risk`.
- Проверяемая область:
  - subset signal row loading;
  - deterministic `final_signal` c value set `{-1, 0, 1}`;
  - compact trades с `entry_exec_idx`, `sig_exit_exec_idx`, `sentinel_index`,
    local `bar_close_1m_idx`;
  - no-risk shortlist ordering без Stage B TP/SL replay;
  - `sharpe_trades` входит в no-risk ranking payload наравне с остальными approved literals;
  - deterministic tie-break по stable keys (`base_variant_key ASC` при равенстве ranking
    metrics);
  - `chunked variant processing` equivalence against non-chunked reference path.
- Reference docs: `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`.

### Stage B legacy close-fill

- Источник: `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json` -> `stage_b_legacy_close_fill`.
- Проверяемая область: текущий v1 exact scoring/reporting path как baseline до cutover.
- Reference docs: `docs/architecture/backtest/backtest-api-post-backtests-v1.md`.

### R5-03 v2 `signal_tf + 1m_risk` validation baseline

- Источник: `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json` -> `stage_b_signal_tf_1m_risk_reference`.
- Статус:
  - `reference-only` в R0 parity scope;
  - `validation-baseline` в
    `tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json`.
- Назначение: дать explicit deterministic baseline для R5/R6, не обещая parity с legacy
  close-fill.
- Canonical docs:
  - `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
  - `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- Canonical fixtures/tests:
  - `tests/unit/contexts/backtest/application/services/v2/fixtures/stage_b_golden_fixtures_v2.json`
  - `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`
  - `tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json`
- Scope для locked golden fixtures:
  - `signal timeline` / `execution timeline`
  - `compact trade list`
  - `1m hit-times`
  - `entry mapping request TF -> 1m`
  - `TP/SL earliest hit`
  - `earliest signal-exit mapping`
  - `signal exit wins on equal bar`
  - `SL wins TP tie`
  - `fast TP/SL grid search`
  - `exact replay of best TP/SL cell`
  - `metrics over compact trades`

## Deterministic inputs

- Benchmark scenarios:
  `tests/perf_smoke/contexts/backtest/fixtures/r0_benchmark_scenarios.json`
- Parity scope manifest:
  `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json`
- Runtime-acceleration benchmark corpus for later exact/hybrid/plugin rollout work:
  `tests/perf_smoke/contexts/backtest/fixtures/backtest_runtime_acceleration_benchmark_corpus_v1.json`
- Notebook-parity benchmark corpus for notebook parity closure authority:
  `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- Executable local baseline:
  `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py`
- Executable notebook-parity perf smoke:
  `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- Notebook references:
  `tests/notebook_tests/06_backtest_compute.ipynb`
  `tests/notebook_tests/05_hit_time_grid.ipynb`
  `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`
  `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`

## Reproducible execution protocol

1. Проверить contract baseline и локальные benchmark fixtures:

```bash
uv run pytest -q tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
```

Эта команда теперь покрывает и legacy R0 reference, и R10-03 artifact-backed zero-call perf
gates.

2. Проверить R5-03 executable Stage B baseline:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py
```

3. Проверить notebook parity benchmark authority:

```bash
uv run pytest -q tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py
```

Этот harness является blocking perf-smoke authority для committed notebook parity contract.
Он гарантирует, что benchmark surface, runtime-shape payload, equal-thread-budget rules и
committed comparison points остаются детерминированы и исполнимы. Это `synthetic_contract_validation`,
а не замена benchmark-host closure evidence: canonical `NR2` и `RG-TTR` всё равно должны получить
explicit `live_host_captures` payload с теми же runtime-shape fields и пройти те же gates на
benchmark host.

4. Сохранить measurement snapshot в файл:

```bash
ROEHUB_R0_BASELINE_PRINT=1 \
uv run pytest -q -s tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py \
> /tmp/roehub-backtest-r0-baseline.json
```

5. Проверить runtime/config freeze и additive runtime-defaults payload:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py \
  tests/unit/apps/api/test_backtests_routes.py
```

6. Проверить R6-01 loader/bootstrap guardrails:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_resolver_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
```

7. Проверить R6-02 Stage A kernels и additive shortlist bridge:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_signal_aggregator_kernel_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
```

7. Проверить R6-03 Stage B kernels и additive artifact-backed scorer bridge:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py \
  tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
```

8. Проверить R6-04 ranking contract и summary-only materialization:

```bash
uv run pytest -q \
  tests/unit/apps/api/test_backtests_dto.py \
  tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py \
  tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py \
  tests/unit/contexts/backtest/application/services/test_job_runner_streaming_v1.py \
  tests/unit/contexts/backtest/application/services/test_close_fill_scorer_v1.py
```

Где хранятся outputs:
- version-controlled fixtures: `tests/perf_smoke/contexts/backtest/fixtures/*.json`
- executable Stage B golden fixtures:
  `tests/unit/contexts/backtest/application/services/v2/fixtures/stage_b_golden_fixtures_v2.json`
- generated measurement snapshot: `/tmp/roehub-backtest-r0-baseline.json`
- docs index после markdown updates: `docs/INDEX.md`

## Migration note: `top_k` vs `top_n`

- Current v1 runtime and API остаются на `top_k_default` / `top_k_persisted_default`.
- Frozen R0 target contract публикуется additively как `top_n_default` / `top_n_max` через runtime defaults.
- До cutover нельзя silently переименовывать request/response поля; mapping должен быть явным в docs и tests.

## Consumers for next milestones

- R1 использует frozen request TF list, `signals.v1.params = default-only` и `top_n_*` contract.
- R5-02 использует `docs/architecture/backtest/backtest-runtime-kernels-v2.md` как canonical
  transfer contract для notebook-derived kernel semantics.
- R5-03 публикует `r5_stage_b_golden_cases.json` и unit fixture catalog как canonical
  validation baseline для `signal_tf + 1m_risk` golden fixtures.
- R6-01 фиксирует runtime bootstrap/loaders guardrails без kernel cutover.
- R6-02 добавляет Stage A kernels, deterministic no-risk shortlist metrics и additive
  artifact-backed shortlist path без Stage B risk execution cutover.
- R6-03 добавляет Stage B risk execution kernels, `metrics over compact trades`, fast TP/SL
  search и exact replay только по winning cell, сохраняя R6-04 ranking/top-N materialization вне
  scope.
- R6-R8 сравнивают `clickhouse_hot_path_calls` и `indicator_compute_calls` против R0 baseline,
  сохраняя запрет на runtime scanning и hot-path hash recomputation до cutover.
- R10 обновляет этот документ уже как post-cutover benchmark ledger, но не меняет задним числом R0 fixtures.
