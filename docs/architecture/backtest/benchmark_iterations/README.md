# Backtest Benchmark Iterations

Рабочая папка для фиксации benchmark evidence по каждой итерации backtest service.

## Назначение

Каждый документ в этой папке фиксирует одну benchmark-итерацию:

- что было реализовано;
- какой notebook baseline использовался;
- какой request/artifact fixture запускался;
- какие artifact manifest hash и request hash использовались;
- какие метрики получены на `Mac Studio`;
- прошел ли stage gate `>= 90%` по скорости, памяти и CPU-метрикам;
- прошли ли service-only absolute budgets.

Benchmark вне `Mac Studio` не считается acceptance evidence.

## Имя файла

Формат:

```text
YYYY-MM-DD-iteration-<n>-<short-name>.md
```

Пример:

```text
2026-04-25-iteration-1-artifact-load.md
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

## Runtime Metrics Without Warmup

Canonical notebook-compatible stages:

```text
total                       3.012s
prepare indicator pools     0.964s
combo proxy prefilter       0.595s
count trades                0.637s
exact scoring               0.775s
heap/top-K Python work      0.026s
```

| Stage | Notebook wall s | Service wall s | Speed ratio | Notebook peak RSS | Service peak RSS | Memory ratio | CPU evidence | Pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| total_without_warmup | | | | | | | | |
| prepare_indicator_pools | | | | | | | | |
| combo_proxy_prefilter | | | | | | | | |
| count_trades | | | | | | | | |
| exact_scoring | | | | | | | | |
| heap_top_k_python_work | | | | | | | | |

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
