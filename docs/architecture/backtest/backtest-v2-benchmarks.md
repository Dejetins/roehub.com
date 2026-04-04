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
- Compatibility note:
  - R10-02 synchronized docs around the shipped runtime and left R10-03 for perf/runbook
    closure only;
  - R10-03 uses this document as the canonical benchmark protocol for legacy R0 reference,
    artifact-backed v2 perf gates, and rollout evidence.
- Milestone A / EPIC A3 adds one additive rollout corpus on top of this baseline:
  - `docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md`
  - `tests/perf_smoke/contexts/backtest/fixtures/backtest_runtime_acceleration_benchmark_corpus_v1.json`
  - exact baseline slices in A3 reuse the approved R0 scenarios and R5 Stage B golden manifest
    from this document instead of introducing a second exact reference set.

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
- summary-only runtime rows: ranking payload определяет inclusion в `top_n`, а report/trades тела
  не materialize'ятся в sync/jobs summary paths.

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
- Executable local baseline:
  `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py`
- Notebook references:
  `tests/notebook_tests/06_backtest_compute.ipynb`
  `tests/notebook_tests/05_hit_time_grid.ipynb`

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

3. Сохранить measurement snapshot в файл:

```bash
ROEHUB_R0_BASELINE_PRINT=1 \
uv run pytest -q -s tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py \
> /tmp/roehub-backtest-r0-baseline.json
```

4. Проверить runtime/config freeze и additive runtime-defaults payload:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py \
  tests/unit/apps/api/test_backtests_routes.py
```

5. Проверить R6-01 loader/bootstrap guardrails:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_resolver_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py \
  tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_timeline_builder.py
```

6. Проверить R6-02 Stage A kernels и additive shortlist bridge:

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
