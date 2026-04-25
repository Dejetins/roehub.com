# Backtest Benchmark Iterations

Рабочая папка для фиксации benchmark evidence по каждой итерации backtest service.

## Назначение

Каждый документ в этой папке фиксирует одну benchmark-итерацию:

- что было реализовано;
- какой notebook baseline использовался;
- какой request/artifact fixture запускался;
- какие метрики получены на `Mac Studio`;
- прошел ли segment gate `>= 90%` по скорости, памяти и CPU-метрикам.

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
- Notebook baseline:

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

## Metrics

| Segment | Baseline wall s | Service wall s | Speed ratio | Baseline peak RSS | Service peak RSS | Memory ratio | CPU evidence | Pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| service_warmup | | | | | | | | |
| artifact_manifest_load | | | | | | | | |
| artifact_array_mmap_load | | | | | | | | |
| time_range_slice | | | | | | | | |
| signal_row_selection | | | | | | | | |
| stage_a_prefilter | | | | | | | | |
| combo_prefilter | | | | | | | | |
| no_risk_exact_scoring | | | | | | | | |
| tp_sl_exact_scoring | | | | | | | | |
| persist_top_n | | | | | | | | |
| lazy_trades_compute | | | | | | | | |
| lazy_trades_cache_hit | | | | | | | | |

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
