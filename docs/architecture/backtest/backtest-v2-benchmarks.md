# Backtest v2 Benchmarks (R0 baseline)

Статус: source-of-truth для R0 benchmark/parity baseline  
Связанные документы:
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`

## Цель

R0 не внедряет runtime v2, а фиксирует:
- какие сценарии считаются representative baseline;
- какие метрики снимаются одинаково на каждом локальном прогоне;
- какие parity fixtures обязательны до cutover;
- где лежат reproducible inputs и как сохранять измерения.

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

## Parity fixture scope

### Stage A no-risk

- Источник: `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json` -> `stage_a_no_risk`.
- Проверяемая область: signal selection, no-risk shortlist ordering, deterministic tie-break.
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
- R6-R8 сравнивают `clickhouse_hot_path_calls` и `indicator_compute_calls` против R0 baseline перед cutover.
- R10 обновляет этот документ уже как post-cutover benchmark ledger, но не меняет задним числом R0 fixtures.
