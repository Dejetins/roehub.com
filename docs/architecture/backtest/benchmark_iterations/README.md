# Backtest Benchmark Iterations

Рабочая папка для фиксации benchmark evidence по каждой итерации backtest service.
Активный контракт имен стадий берется из
`docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md` и
канонического JSON evidence
`docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.

## Назначение

Каждый документ в этой папке фиксирует одну benchmark-итерацию:

- что было реализовано;
- какой notebook baseline использовался;
- какой request/artifact fixture запускался;
- какие artifact manifest hash и request hash использовались;
- какие метрики получены на `Mac Studio`;
- прошел ли stage gate `>= 90%` по скорости, памяти и CPU-метрикам;
- прошли ли service-only absolute budgets.

Benchmark вне `Mac Studio` не считается acceptance evidence. Локальные docs/static
checks можно записывать как developer evidence, но не как acceptance benchmark.

## Канонический baseline

- Notebook baseline:
  `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`
- Числовые целевые значения:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Человекочитаемое summary:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md`
- Runtime target: `hit_times/15m`
- Словарь public API: `jobs`, `risk.mode`, readable `variant_key` plus
  stable `variant_hash`

Старый пятистадийный runtime словарь с `count_trades`,
`combo_proxy_prefilter` и `heap_top_k_python_work` не является active target для
v1 records. Такие имена допустимы только в historical notes, если они явно
помечены как superseded.

## Правило сопоставимости stage

Benchmark stage можно сравнивать с canonical notebook target только если
измеряется тот же участок алгоритма. Если service stage включает дополнительную
production-подготовку, ее нужно вынести в отдельные service-only overhead
segments и не смешивать с notebook-compatible timing.

Для Iteration 2 canonical notebook `prepare_pools` соответствует service
`prepare_pools_core`:

- signal row selection/extraction;
- row prefilter;
- compressed segment build;
- подготовка pool metadata, если она входит в notebook-equivalent pool build.

Следующие части не входят в canonical notebook `prepare_pools` target и должны
измеряться отдельно:

- `artifact_context_resolve`: current pointer, slot manifest identity, manifest
  hash validation, typed artifact context;
- `artifact_array_open`: opening `.npy` arrays через
  `np.load(..., mmap_mode="r")` и manifest-backed dtype/shape validation;
- `request_slice_prepare`: `[start, end)` 15m slicing, returns, execution
  mapping derivation;
- `prepare_pools_total`: aggregate service telemetry, not a direct notebook
  ratio target.

Правило `canonical_notebook_stage_s / service_stage_s >= 0.9` применяется к
notebook-compatible stage, например к `prepare_pools_core`. Service-only
overhead должен иметь отдельные absolute budgets или regression comparison
against previous accepted service baseline. Нельзя проваливать
notebook-compatible gate из-за overhead, которого нет в notebook timer.

Локальная validation-команда для проверки accounting rules перед Mac Studio
benchmark:

```bash
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/<iteration>/local_accounting_validation.json
```

Она проверяет, что `request.top_n = 100`, `benchmark_top_k = 5`,
`sample_warmup_top_k = 1`, `top_results_count = 5` и heap capacity записаны как
benchmark metadata, что notebook `prepare_pools` нормализован в
`prepare_pools_core`, а `service_total_without_warmup` и другие service-only
поля не попадают в canonical stage comparison.

## Имя файла

Формат для одиночной Markdown-записи:

```text
YYYY-MM-DD-iteration-<n>-<short-name>.md
```

Пример:

```text
2026-04-25-iteration-1-artifact-load.md
```

Если итерация требует рядом JSON/PNG/log evidence, допускается директория:

```text
YYYY-MM-DD_iteration_<n>_<short_name>/README.md
```

## Шаблон записи

```md
# Backtest Benchmark Iteration <n> — <name>

<Одна строка: что проверяли и зачем.>

## Scope

- Implemented:
- Not in scope:

## Version

- Branch:
- Commit:
- Service command:
- Benchmark command:
- Artifact config:
- Artifact root:
- Artifact slot:
- Artifact manifest hash:
- Notebook baseline:
- Notebook baseline output:
- Request hash:
- Engine/config hash:

## Fixture

- Coordinates:
- Timeframe:
- Time range:
- Indicators:
- Risk mode:
- Execution settings:
- Ranking:
- Top N:

## Environment

- Host: Mac Studio
- CPU:
- RAM:
- Python:
- Numba:
- NUMBA_NUM_THREADS:
- Warmup policy:

## Warmup Metrics

| Segment | Baseline wall s | Service wall s | Speed ratio | Baseline peak RSS | Service peak RSS | Memory ratio | CPU evidence | Pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| service_warmup | | | | | | | | |
| numba_warmup | | | | | | | | |
| sample_warmup | | | | | | | | |

## Runtime Metrics Without Warmup

Canonical notebook-compatible stages:

```text
total_without_warmup
load_hit_times              risk-on only
tp_sl_grid_validation       risk-on only
prepare_pools_core          notebook prepare_pools equivalent
build_exact_context
build_proxy_context
combo_iteration
proxy_filter
self_check                  benchmark/test evidence
exact_scoring
tp_sl_exact_scoring         risk-on only
heap_update
top_result_proxy_fill
```

Canonical JSON may include `total` for no-risk rows as a historical alias of
`total_without_warmup`. Do not treat it as a separate benchmark stage.

| Stage | Required | Notebook wall s | Service wall s | Speed ratio | Notebook peak RSS | Service peak RSS | Memory ratio | CPU evidence | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| total_without_warmup | yes | | | | | | | | |
| load_hit_times | `risk.mode=tp_sl_grid` | | | | | | | | |
| tp_sl_grid_validation | `risk.mode=tp_sl_grid` | | | | | | | | |
| prepare_pools_core | yes | | | | | | | | |
| build_exact_context | yes | | | | | | | | |
| build_proxy_context | yes | | | | | | | | |
| combo_iteration | yes | | | | | | | | |
| proxy_filter | yes | | | | | | | | |
| self_check | benchmark/test | | | | | | | | |
| exact_scoring | yes | | | | | | | | |
| tp_sl_exact_scoring | `risk.mode=tp_sl_grid` | | | | | | | | |
| heap_update | yes | | | | | | | | |
| top_result_proxy_fill | no-risk/top metadata | | | | | | | | |

## Service-Only Overhead

These segments do not exist in the notebook runtime total. They require an
absolute budget and, after the first accepted service baseline, regression
comparison against the previous service run.

| Segment | Service wall s | Absolute budget s | Peak RSS | CPU evidence | Pass |
|---|---:|---:|---:|---|---|
| artifact_context_resolve | | | | | |
| artifact_array_open | | | | | |
| request_slice_prepare | | | | | |
| prepare_pools_total | | | | | |
| service_total_without_warmup | | | | | |
| top_result_assembly | | | | | |
| tp_sl_full_metrics_second_pass | | | | | |
| persist_top_n_io | | | | | |
| lazy_trades_compute | | | | | |
| lazy_trades_cache_hit | | | | | |

## Memory Cleanup Evidence

Cleanup evidence is a service hygiene check, not a canonical notebook stage. It
does not change the ordered stage list above and must not be compared with
canonical notebook timers until a separate accepted service baseline exists.

| Check | Value |
|---|---|
| cleanup_duration_s | |
| rss_before_mb | |
| rss_peak_mb | |
| rss_after_cleanup_mb | |
| retained_rss_delta_mb | |
| repeated_run_count | |
| monotonic_retained_rss_growth | |
| worker_recycled | |
| pass | |

## Contract Coverage

- API create/status/list/top/variant/trades/cancel/defaults/preflight:
- Idempotency replay/conflict:
- Resource guardrails:
- Ownership/authz:
- Cache identity:
- Failure injection:

## Scenario Matrix

| Dimension | Covered values | Result |
|---|---|---|
| risk.mode | `none`, `tp_sl_grid` | |
| sizing.mode | `all_in`, `fixed_quote`, `fixed_equity_pct`, `fixed_equity_pct_min_quote`, `fixed_equity_pct_max_quote` | |
| profit_lock | disabled, enabled | |
| direction_mode | runtime-supported modes | |
| close_on_end | true, false | |

## Correctness Evidence

- Parity fixture:
- Max diff:
- Self-check:
- Result:

## Decision

- Status: pass/fail
- Reason:
- Next iteration:

## Notes

- Risk:
- Follow-up:
```
