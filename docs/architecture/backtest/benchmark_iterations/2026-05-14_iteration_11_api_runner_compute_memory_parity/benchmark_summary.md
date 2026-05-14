# Iteration 11 API runner compute memory parity

Этот отчет собран из `benchmark_results.json` и оформлен в формате предыдущих benchmark summaries. Главная цель: понятно зафиксировать, какие проверки запускались, на каком периоде рыночных данных, где выполнялся тест, какие поверхности прошли acceptance, и что показал замер скорости относительно reference от 2 мая.

## Scope

- Проверено: создание jobs через API, переход через runner, disposable child process для full job, disposable child process для lazy trades cache miss, bounded cache-hit reads в API, light/heavy scheduler smoke, memory release, parity, отсутствие legacy hot paths.
- Не проверялось как hard gate: новая скорость относительно 2 мая. JSON содержит timing evidence, но `performance.pass = true` означает корректный сбор и разделение timings, а не прохождение speed parity.
- Важно: отчет не меняет artifact. В artifact зафиксирован `request.top_n = 100`; если нужна приемка именно с `top_n = 50`, нужен отдельный benchmark run или отдельная пометка, что этот JSON был снят до смены лимита.

## Version

- Host: `MacStudioDaniil`
- Git commit: `ffe6c50923b8ddafb04169098949612ea5368517`
- Python: `3.12.13`
- Benchmark started/generated at: `2026-05-14T08:32:17.146380+00:00`
- Benchmark finished at: `2026-05-14T09:08:17.562145+00:00`
- API base: `http://127.0.0.1:18081`
- Smoke user id: `6960d466-e9fb-4116-a9fb-8f1589c61914`
- Git status short: ``
- Schema: `backtest_api_runner_compute_memory_parity_v1`
- Overall pass: `yes`

## Карта Замеров И Периодов

| Проверка | Где запускалось | Период данных | Что именно проверялось | Результат |
|---|---|---|---|---|
| API-runner parity benchmark | `MacStudioDaniil`, `http://127.0.0.1:18081` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | 27 jobs через API create/status/top variants, runner claim, disposable full-job child process, persisted result read | `yes` |
| Reference comparison | accepted May 2 artifact `2026-05-02_iteration_8_execution_sizing_completion` | тот же canonical BTCUSDT 15m fixture; one heaviest job excluded | parity against accepted reference, top-result shape, telemetry/sample metrics | `yes` |
| Lazy trades cache miss | `MacStudioDaniil`, runner lazy child | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | POST trades materialization: `queued -> running -> completed`, disposable lazy child writes cache bundle | `yes` |
| Lazy trades cache hit/API reads | `MacStudioDaniil`, API process | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | page/series/monthly/symbol/csv reads from JSONL cache without loading full detail into API memory | `yes` |
| Scheduler light phase | `MacStudioDaniil`, runner scheduler smoke | `2026-01-01T00:00:00Z -> 2026-02-01T00:00:00Z` | two `light_candidate` jobs, configured light concurrency `2` | `yes` |
| Scheduler heavy phase | `MacStudioDaniil`, runner scheduler smoke | `2026-01-01T00:00:00Z -> 2026-02-01T00:00:00Z` | heavy FIFO, heavy concurrency `1`, no heavy overlap | `yes` |
| Scheduler promotion case | `MacStudioDaniil`, runner scheduler smoke | final terminal scheduling stores post-prepare class only; source fixture belongs to scheduler smoke | `light_candidate` requeued after post-prepare refinement to `heavy` | `yes` |
| Full-job memory release | `MacStudioDaniil`, child process evidence | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | child exit, parent RSS/physical footprint before-after, retained RSS delta, vmmap/physical footprint where available | `yes` |
| Legacy path absence | static audit from benchmark script | n/a | runner parent, API cache-hit path, large-grid Cartesian path | `yes` |
| Docs drift audit | static docs audit from benchmark script | n/a | active backtest docs checked for blockers | `yes` |

## Fixture

- Symbol/timeframe: `BTCUSDT` / `15m`
- Основной период API-runner jobs: `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z`
- Scheduler smoke period: light `2026-01-01T00:00:00Z -> 2026-02-01T00:00:00Z`; heavy `2026-01-01T00:00:00Z -> 2026-02-01T00:00:00Z`
- Request top N: `100`
- Benchmark top K: `5`
- Rows per indicator: `6`
- Warmup rows per indicator: `2`
- Exclude heaviest 140s job: `yes`
- Reference iteration: `2026-05-02_iteration_8_execution_sizing_completion`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference JSON: `docs/architecture/backtest/benchmark_iterations/2026-05-02_iteration_8_execution_sizing_completion/benchmark_results.json`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Hit-times manifest hash: `09143b055123190d6c919ca758817fa89915245d70719437f037340cdc4196fc`
- Artifact policy: `historical_prefix_compatible`
- Request hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`
- Excluded reference job: `tp_sl_grid/arity_7/long_only`; observed May 2 runtime `147.415s`; reason `exclude_heaviest_140s_job: single heaviest accepted May 2 reference job is omitted from required benchmark/smoke loops`.

## Pass Breakdown

| Surface | Result | Evidence |
|---|---|---|
| Overall benchmark artifact | `yes` | Top-level `pass`. |
| API-runner path | `yes` | 27/27 required jobs passed. |
| Parity | `yes` | Failed jobs: `[]`. |
| Performance evidence capture | `yes` | 27 jobs with stage timings; service-only overhead separated. |
| Full-job memory release | `yes` | 27 child processes checked; failed: `[]`. |
| Lazy cache miss/hit memory | `yes` | Disposable lazy child plus bounded API cache-hit reads. |
| Mixed scheduler smoke | `yes` | Light cap 2; heavy cap 1; overlap `disabled_v1`. |
| Legacy path absence | `yes` | Runner/API/cache/large-grid checks below. |
| Dead-code audit | `yes` | Replaced paths and retained helper classification recorded. |
| Docs drift audit | `yes` | Active blockers: `[]`. |

## API-runner Path

- Runner entrypoint: `BacktestJobWorkerUseCase.run_next`
- Child module: `apps.worker.backtest_job_runner.main.full_job_child`
- Required state path: `queued -> running -> succeeded`
- Required jobs: `27`
- Passed jobs: `27`
- Период данных для всех required jobs: `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z`
- Backlog before: full queued `0`, full running `0`, lazy queued `0`, lazy running `0`.
- Backlog after: full queued `0`, full running `0`, lazy queued `0`, lazy running `0`.

## Speed Summary

Benchmark действительно содержит замер скорости. Ниже сравнение current child-process run с accepted May 2 reference там, где есть сопоставимый reference timing. `exact_ratio = May2 exact_scoring / current exact_scoring`; значение ниже `1.00x` означает, что текущий API-runner child path медленнее reference от 2 мая на этой stage.

| Group | jobs | child elapsed sum s | child elapsed median s | child elapsed p95 s | exact sum s | May2 exact sum s | exact median ratio | worst exact ratio | max API latency ms | max peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 27 | 2006.661 | 12.412 | 317.553 | 1845.168 | 493.992 | 0.10x | 0.06x | 544.7 | 468.3 |
| no_risk | 14 | 1139.094 | 6.992 | 317.553 | 1087.424 | 311.165 | 0.08x | 0.06x | 544.7 | 267.9 |
| tp_sl_grid | 13 | 867.567 | 12.926 | 205.115 | 757.745 | 182.827 | 0.22x | 0.09x | 519.1 | 468.3 |
| heavy | 3 | 969.173 | 322.722 | 322.722 | 949.796 | 419.253 | 0.44x | 0.43x | 544.7 | 468.3 |
| light/light_candidate | 24 | 1037.488 | 9.076 | 196.488 | 895.372 | 74.739 | 0.09x | 0.06x | 162.5 | 460.6 |

### Speed Interpretation

- Timing evidence есть для всех 27 API-runner jobs: child elapsed time, service wall clock, stage timings, Numba thread count, API responsiveness samples.
- Artifact проходит parity/memory/scheduler surfaces, но derived comparison с 2 мая не выглядит здоровым: median exact-stage ratio `0.10x`, худший exact-stage ratio `0.06x`.
- `light_candidate` jobs использовали `2` Numba threads. Это ограничивает влияние легких jobs на host, но в этом fixture arity 5-6 jobs стали намного медленнее May 2 reference.
- `heavy` jobs использовали `12` Numba threads и запускались по одному, но их median exact-stage ratio против 2 мая только `0.44x`.
- Поэтому этот JSON подтверждает production path, memory release и scheduler behavior, но не должен считаться hard acceptance по May 2 speed parity.

## Runtime Metrics Without Warmup

| job | период данных | scheduling | combos upper bound | numba threads | API top count | child elapsed s | service wall s | exact s | May2 exact s | exact ratio | service total s | May2 service total s | total ratio | peak RSS MiB | retained RSS MiB | max API ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none/arity_1/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 6 | 2 | 6 | 3.077 | 0.186 | 0.006 | 0.001 | 0.19x | 0.491 | n/a | n/a | 267.9 | 1.5 | 51.5 |
| `none/arity_2/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 36 | 2 | 36 | 3.038 | 0.216 | 0.011 | 0.003 | 0.23x | 0.528 | n/a | n/a | 214.2 | 0.1 | 51.2 |
| `none/arity_3/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 216 | 2 | 100 | 3.827 | 1.566 | 0.667 | 0.048 | 0.07x | 1.900 | n/a | n/a | 207.4 | 0.4 | 52.7 |
| `none/arity_4/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 1296 | 2 | 100 | 7.291 | 4.891 | 3.944 | 0.340 | 0.09x | 5.227 | n/a | n/a | 221.4 | 0.1 | 79.5 |
| `none/arity_5/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 7776 | 2 | 100 | 28.750 | 25.899 | 24.925 | 2.020 | 0.08x | 26.240 | n/a | n/a | 219.8 | 0.1 | 52.3 |
| `none/arity_6/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 46656 | 2 | 100 | 208.881 | 206.148 | 203.307 | 15.694 | 0.08x | 206.508 | n/a | n/a | 238.0 | 0.1 | 55.3 |
| `none/arity_7/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `heavy` | 279936 | 12 | 100 | 328.899 | 326.375 | 323.758 | 138.755 | 0.43x | 326.710 | n/a | n/a | 247.7 | 0.1 | 505.9 |
| `none/arity_1/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 6 | 2 | 6 | 3.031 | 0.187 | 0.007 | 0.002 | 0.23x | 0.493 | n/a | n/a | 260.5 | 0.0 | 50.6 |
| `none/arity_2/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 36 | 2 | 36 | 3.075 | 0.213 | 0.007 | 0.001 | 0.15x | 0.527 | n/a | n/a | 178.8 | 0.0 | 52.9 |
| `none/arity_3/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 216 | 2 | 100 | 3.834 | 1.541 | 0.652 | 0.037 | 0.06x | 1.861 | n/a | n/a | 206.9 | 0.0 | 143.4 |
| `none/arity_4/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 1296 | 2 | 100 | 6.992 | 4.724 | 3.768 | 0.299 | 0.08x | 5.050 | n/a | n/a | 215.9 | 0.0 | 50.8 |
| `none/arity_5/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 7776 | 2 | 100 | 27.839 | 25.029 | 24.558 | 1.971 | 0.08x | 25.361 | n/a | n/a | 216.3 | 0.0 | 51.9 |
| `none/arity_6/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 46656 | 2 | 100 | 193.008 | 190.573 | 189.533 | 15.365 | 0.08x | 190.915 | n/a | n/a | 232.9 | 0.0 | 162.5 |
| `none/arity_7/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `heavy` | 279936 | 12 | 100 | 317.553 | 314.662 | 312.280 | 136.630 | 0.44x | 314.995 | n/a | n/a | 251.3 | 0.0 | 544.7 |
| `tp_sl_grid/arity_1/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 6 | 2 | 6 | 5.474 | 2.610 | 0.010 | 0.004 | 0.36x | 2.371 | 3.609 | 1.52x | 424.5 | 0.0 | 59.2 |
| `tp_sl_grid/arity_2/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 36 | 2 | 36 | 9.777 | 7.380 | 0.027 | 0.014 | 0.50x | 7.621 | 1.777 | 0.23x | 433.2 | 0.0 | 53.3 |
| `tp_sl_grid/arity_3/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 216 | 2 | 100 | 12.994 | 10.122 | 0.242 | 0.060 | 0.25x | 10.103 | 1.064 | 0.11x | 441.6 | 0.0 | 150.3 |
| `tp_sl_grid/arity_4/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 1296 | 2 | 100 | 12.926 | 10.618 | 4.269 | 0.434 | 0.10x | 14.648 | 1.446 | 0.10x | 446.7 | 0.0 | 53.5 |
| `tp_sl_grid/arity_5/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 7776 | 2 | 100 | 34.384 | 31.955 | 26.157 | 2.312 | 0.09x | 57.865 | 2.634 | 0.05x | 447.2 | 0.0 | 82.9 |
| `tp_sl_grid/arity_6/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 46656 | 2 | 100 | 205.115 | 202.831 | 196.468 | 17.446 | 0.09x | 399.073 | 17.803 | 0.04x | 460.6 | 0.0 | 132.1 |
| `tp_sl_grid/arity_1/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 6 | 2 | 6 | 7.114 | 4.296 | 0.018 | 0.009 | 0.48x | 4.532 | 7.115 | 1.57x | 431.3 | 0.0 | 52.7 |
| `tp_sl_grid/arity_2/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 36 | 2 | 36 | 5.473 | 2.654 | 0.016 | 0.007 | 0.44x | 2.886 | 0.269 | 0.09x | 425.3 | 0.0 | 59.8 |
| `tp_sl_grid/arity_3/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 216 | 2 | 100 | 9.076 | 6.646 | 0.169 | 0.038 | 0.22x | 6.566 | 0.317 | 0.05x | 438.8 | 0.0 | 54.0 |
| `tp_sl_grid/arity_4/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 1296 | 2 | 100 | 12.412 | 9.960 | 3.377 | 0.320 | 0.09x | 13.087 | 0.641 | 0.05x | 450.1 | 0.0 | 142.3 |
| `tp_sl_grid/arity_5/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 7776 | 2 | 100 | 33.612 | 30.718 | 24.828 | 2.112 | 0.09x | 55.297 | 2.428 | 0.04x | 448.2 | 0.0 | 53.9 |
| `tp_sl_grid/arity_6/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 46656 | 2 | 100 | 196.488 | 194.263 | 188.404 | 16.204 | 0.09x | 382.433 | 16.563 | 0.04x | 454.5 | 0.0 | 95.5 |
| `tp_sl_grid/arity_7/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `heavy` | 279936 | 12 | 100 | 322.722 | 319.766 | 313.758 | 143.868 | 0.46x | 633.261 | 144.401 | 0.23x | 468.3 | -27.3 | 519.1 |

## Slowest Jobs

| job | период данных | scheduling | numba threads | child elapsed s | exact s | May2 exact s | exact ratio | peak RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `none/arity_7/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `heavy` | 12 | 328.899 | 323.758 | 138.755 | 0.43x | 247.7 |
| `tp_sl_grid/arity_7/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `heavy` | 12 | 322.722 | 313.758 | 143.868 | 0.46x | 468.3 |
| `none/arity_7/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `heavy` | 12 | 317.553 | 312.280 | 136.630 | 0.44x | 251.3 |
| `none/arity_6/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 208.881 | 203.307 | 15.694 | 0.08x | 238.0 |
| `tp_sl_grid/arity_6/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 205.115 | 196.468 | 17.446 | 0.09x | 460.6 |
| `tp_sl_grid/arity_6/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 196.488 | 188.404 | 16.204 | 0.09x | 454.5 |
| `none/arity_6/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 193.008 | 189.533 | 15.365 | 0.08x | 232.9 |
| `tp_sl_grid/arity_5/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 34.384 | 26.157 | 2.312 | 0.09x | 447.2 |
| `tp_sl_grid/arity_5/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 33.612 | 24.828 | 2.112 | 0.09x | 448.2 |
| `none/arity_5/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 28.750 | 24.925 | 2.020 | 0.08x | 219.8 |

## Lowest Exact-stage Ratios

| job | период данных | scheduling | numba threads | exact s | May2 exact s | exact ratio |
|---|---|---|---:|---:|---:|---:|
| `none/arity_3/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 0.652 | 0.037 | 0.06x |
| `none/arity_3/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 0.667 | 0.048 | 0.07x |
| `none/arity_6/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 203.307 | 15.694 | 0.08x |
| `none/arity_4/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 3.768 | 0.299 | 0.08x |
| `none/arity_5/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 24.558 | 1.971 | 0.08x |
| `none/arity_5/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 24.925 | 2.020 | 0.08x |
| `none/arity_6/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 189.533 | 15.365 | 0.08x |
| `tp_sl_grid/arity_5/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 24.828 | 2.112 | 0.09x |
| `tp_sl_grid/arity_6/long_short_reversal` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 188.404 | 16.204 | 0.09x |
| `none/arity_4/long_only` | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` | `light_candidate` | 2 | 3.944 | 0.340 | 0.09x |

## Memory Release

| Metric | Value |
|---|---:|
| Full-job memory pass | `yes` |
| Checked child jobs | 27 |
| Period covered | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` |
| Failed checks | `[]` |
| Parent retained RSS delta evidence | `yes` |
| vmmap evidence | `yes` |
| Physical footprint evidence | `yes` |
| Max full-job peak RSS MiB | 468.3 |
| Max retained RSS delta MiB | 1.5 |
| Min retained RSS delta MiB | -27.3 |

Каждый full job запускался в одноразовом child process. Parent process retained-delta поля записаны по каждому job; child processes завершались с exit code `0`.

## Lazy Trades Cache Miss And Hit

| Metric | Value |
|---|---:|
| Lazy memory pass | `yes` |
| Period covered | `2020-01-11T20:08:00Z -> 2026-04-11T20:08:00Z` |
| Target job id | `add2c15b-0464-4dc6-a48b-044f1c8c5e5a` |
| Public variant key | `job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135` |
| Cache key | `3aee0dd2e9ab93b91234c7bc177564a91829e2c80cc2097fb6dfb42ff3854649` |
| Miss initial status | `202` / `queued` |
| Miss worker path | `queued -> running -> completed` |
| Lazy child elapsed s | 6.444 |
| Lazy child exit code | 0 |
| Lazy child peak RSS MiB | 422.2 |
| Lazy child peak physical footprint MiB | 272.0 |
| Parent retained RSS delta after lazy child MiB | 0.0 |
| API cache-hit retained RSS delta bytes | 49152 |
| API cache-hit retained RSS limit bytes | 67108864 |
| Trades JSONL rows | 30056 |
| Trades JSONL bytes | 18177058 |
| Metadata bytes | 3165 |

### Cache-hit Reads

| method | path | status | cache status | latency ms |
|---|---|---:|---|---:|
| `POST` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/trades` | 200 | `hit` | 58.5 |
| `GET` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/trades?page=1&page_size=5` | 200 | `miss` | 53.5 |
| `GET` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/equity?points=100` | 200 | `miss` | 401.7 |
| `GET` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/drawdown?points=100` | 200 | `miss` | 328.9 |
| `GET` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/monthly-stats` | 200 | `miss` | 184.9 |
| `GET` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/symbol-stats` | 200 | `miss` | 180.8 |
| `GET` | `/backtests/jobs/add2c15b-0464-4dc6-a48b-044f1c8c5e5a/variants/job_add2c15b5e5a__dema_close_w10__risk_none__vh_ac456135/trades.csv?max_rows=100` | 200 | `miss` | 28.3 |

Bounded reader static audit passed: `read_page`, `read_series`, `read_monthly_stats`, `read_symbol_stats`, and `read_csv` are present; forbidden monolithic full-detail cache-hit read не используется в API path.

## Scheduler Smoke

| Phase | период данных | Pass | Max active light | Max active heavy | Light cap pass | Heavy FIFO pass | Heavy no-overlap pass |
|---|---|---|---:|---:|---|---|---|
| light | `2026-01-01T00:00:00Z -> 2026-02-01T00:00:00Z` | `yes` | 2 | 1 | `yes` | `yes` | `yes` |
| heavy | `2026-01-01T00:00:00Z -> 2026-02-01T00:00:00Z` | `yes` | 2 | 1 | `yes` | `yes` | `yes` |

- Configured light concurrency: `2`
- Configured heavy concurrency: `1`
- Light/heavy overlap policy: `disabled_v1`
- Heavy preflight examples: arity 3 no-risk jobs with `estimated_combinations_upper_bound=7529536` were classified as `heavy`.
- Light examples: arity 1 no-risk jobs with `estimated_combinations_upper_bound=196` were classified as `light_candidate` and ran with max active light `2`.
- Promotion case covered: `yes`; path `queued -> running -> queued -> running -> succeeded`; post-prepare class `heavy`; reason `light_candidate_exceeded_actual_threshold`; actual combinations `38416`.
- Для promotion case final terminal scheduling после post-prepare refinement не содержит `requested_range`; период нужно читать как scheduler smoke fixture, а не как отдельный recorded field.

## Legacy Path Absence

| check | path | required symbol | required present | forbidden symbol | forbidden absent | pass |
|---|---|---|---|---|---|---|
| `runner_parent_does_not_construct_full_compute_graph` | `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py` | `BacktestChildProcessExecutor` | yes | `build_full_job_compute_executor` | yes | yes |
| `public_api_cache_hit_uses_bounded_cache_methods` | `src/trading/contexts/backtest/application/use_cases/backtest_jobs.py` | `cache.read_page(` | yes | `build_paginated_trades_read_model(` | yes | yes |
| `large_grid_production_no_itertools_product` | `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py` | `iter_ordinal_combo_chunks` | yes | `itertools.product` | yes | yes |

## Dead Code Audit

Removed or replaced paths:

- API create path replaced sync_inline compute with background_auto queued jobs
- lazy cache hit replaced monolithic full-detail JSON reads with metadata + JSONL readers
- large-grid production path uses ordinal streaming chunks instead of Cartesian product materialization

Retained helpers:

- `apps/worker/backtest_job_runner/wiring/modules/full_job_compute.py`: child-only
- `src/trading/contexts/backtest/application/services/v2/result_series.py`: reference-only for legacy in-memory builders; API path uses cache readers
- `tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb`: reference-only semantic baseline

## Docs Drift Audit

- Pass: `yes`
- Active blockers: `[]`
- Remaining historical references policy: historical benchmark references remain allowed when labeled historical

Checked active docs:

- `docs/architecture/backtest/backtest-job-runner-production-plan-v1.md`
- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md`
- `docs/architecture/backtest/benchmark_iterations/README.md`

## Local Accounting Validation

The companion `local_accounting_validation.json` records:

- `benchmark_top_k = 5`
- `request.top_n = 100`
- `heap_capacity = 5` for benchmark accounting validation
- `sample_warmup_top_k = 1`
- `top_results_count_values = [5]`
- `prepare_pools_alias_normalized = true`
- `service_total_compared_to_canonical = false`
- service-only fields: `artifact_context_resolve`, `artifact_array_open`, `request_slice_prepare`, `prepare_pools_total`, `service_total_without_warmup`, `top_result_assembly`, `tp_sl_full_metrics_second_pass`, `persist_top_n_io`, `lazy_trades_compute`, `lazy_trades_cache_hit`.

## Artifacts

- `docs/architecture/backtest/benchmark_iterations/2026-05-14_iteration_11_api_runner_compute_memory_parity/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/2026-05-14_iteration_11_api_runner_compute_memory_parity/benchmark_summary.md`
- `docs/architecture/backtest/benchmark_iterations/2026-05-14_iteration_11_api_runner_compute_memory_parity/local_accounting_validation.json`

## Decision

Accepted surfaces from this JSON:

- API-created jobs are processed through runner and disposable full-job child processes.
- Required API-runner jobs complete with `queued -> running -> succeeded`.
- Result parity and public result shape checks pass for all required jobs.
- Full-job child memory release evidence passes.
- Lazy trades cache miss runs in a disposable child process and cache-hit API reads stay bounded.
- Scheduler smoke confirms light concurrency `2`, heavy concurrency `1`, heavy FIFO, no heavy overlap, and light-candidate promotion to heavy.
- Legacy inline/full-detail/Cartesian paths are absent from the production surfaces checked by the benchmark.

Not accepted by this artifact as a hard speed gate:

- May 2 speed parity. Этот отчет выводит speed ratios из того же JSON и reference от 2 мая; ratios показывают существенное замедление API-runner child path. Если speed parity является release requirement, следующий шаг - root-cause анализ Numba thread allocation, scheduler classification thresholds, warm process vs disposable process effects, and exact scoring hot-path configuration.

## Operator Commands

Primary benchmark command recorded by the original summary:

```bash
PYTHONPATH=/tmp/roehub-api-runner-benchmark/src:/tmp/roehub-api-runner-benchmark \
ROEHUB_BENCHMARK_GIT_COMMIT=ffe6c50923b8ddafb04169098949612ea5368517 \
/Users/daniildegtyarev/Projects/roehub.com/.venv/bin/python \
  scripts/backtest/run_api_runner_benchmark_parity.py \
  --api-base http://127.0.0.1:18081 \
  --out-dir /tmp/roehub-api-runner-benchmark-output \
  --timeout-seconds 21600 \
  --poll-interval-seconds 0.2
```

Accounting validation:

```bash
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/2026-05-14_iteration_11_api_runner_compute_memory_parity/local_accounting_validation.json
```
