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
prepare_pools
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

| Stage | Required | Notebook wall s | Service wall s | Speed ratio | Notebook peak RSS | Service peak RSS | Memory ratio | CPU evidence | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| total_without_warmup | yes | | | | | | | | |
| load_hit_times | `risk.mode=tp_sl_grid` | | | | | | | | |
| tp_sl_grid_validation | `risk.mode=tp_sl_grid` | | | | | | | | |
| prepare_pools | yes | | | | | | | | |
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
| persist_top_n_io | | | | | |
| lazy_trades_compute | | | | | |
| lazy_trades_cache_hit | | | | | |

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
