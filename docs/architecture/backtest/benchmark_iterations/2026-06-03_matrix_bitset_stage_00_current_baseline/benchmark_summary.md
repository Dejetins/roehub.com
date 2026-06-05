# Stage 00 Current Baseline — API runner clean arity-6 CPU/memory benchmark

Stage 00 baseline for `Backtest Compute Acceleration Plan v1`, generated through
the existing API-runner benchmark harness.

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / arity 6, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `0.999..1.054`.
- CPU: `pass`; mean child CPU = `482.568..1163.418%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить приемочный путь API-created job -> runner -> одноразовый heavy child process с 12 Numba threads и сравнить arity 6 с May 2 reference.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `d9bfa5811e3f5bccab9fb2635166f97e43f100bb`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / arity 6 only / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
- Full jobs policy: `heavy_only`, `ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`, `ROEHUB_BACKTEST_LIGHT_CONCURRENCY=0`, `NUMBA_NUM_THREADS=12`.
- CPU sampler: sustained `ps` samples by child `--job-id`; `vmmap`/physical-footprint observation is disabled for clean timing.
- Excluded: `tp_sl_grid/arity_7/long_only` because `exclude_heaviest_140s_job`.

## API-runner path

- Required jobs: `4`
- Pass: `yes`
- State requirement: `queued -> running -> succeeded`.

## Mac Studio results

- Overall pass: `yes`
- Scheduler pass: `yes`
- Lazy cache memory pass: `yes`

## Parity

- Passed jobs: `4/4`
- Failed jobs: `[]`

## Performance

- Stage timing jobs: `4`
- CPU sampling jobs: `4`
- CPU sampling failed jobs: `[]`
- Policy: API/runner wall and service-only overhead are recorded separately from May 2 notebook-compatible stage timings.

| Job | Threads | Exact current s | Exact May2 s | Ratio | CPU mean % | CPU p50 % | CPU max % | System memory gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 12 | 15.704 | 15.694 | 0.999 | 784.059 | 1147.900 | 1184.200 | pass |
| `none/arity_6/long_short_reversal` | 12 | 15.111 | 15.365 | 1.017 | 1163.418 | 1176.100 | 1186.200 | pass |
| `tp_sl_grid/arity_6/long_only` | 12 | 17.206 | 17.446 | 1.014 | 482.568 | 100.000 | 1189.500 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 12 | 15.367 | 16.204 | 1.054 | 1130.175 | 1171.650 | 1185.900 | pass |

## Memory release

- Checked jobs: `4`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `8b805247-cae1-4914-bcd4-37399de0525b`
- Cache hit retained RSS delta: `16384`
- Pass: `yes`

## Legacy path absence

- Pass: `yes`

## Dead code audit

- Pass: `yes`

## Docs drift audit

- Pass: `yes`

## Artifacts

- `benchmark_results.json`
- `benchmark_summary.md`
- `local_accounting_validation.json`

## Operator Commands

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py
uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/<iteration>/local_accounting_validation.json
```
