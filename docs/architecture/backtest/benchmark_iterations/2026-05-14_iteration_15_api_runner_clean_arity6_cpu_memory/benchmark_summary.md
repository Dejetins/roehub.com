# Iteration 15 API runner clean arity-6 CPU/memory benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / arity 6, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `0.972..1.053`.
- CPU: `pass`; mean child CPU = `995.031..1135.136%`.
- Acceptance: `fail`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `['none/arity_6/long_only', 'none/arity_6/long_short_reversal']`, lazy status path = `queued -> running -> failed`.

## Intent

Проверить приемочный путь API-created job -> runner -> одноразовый heavy child process с 12 Numba threads и сравнить arity 6 с May 2 reference.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `unavailable`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / arity 6 only / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
- Full jobs policy: `heavy_only`, `ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`, `ROEHUB_BACKTEST_LIGHT_CONCURRENCY=0`, `NUMBA_NUM_THREADS=12`.
- CPU sampler: sustained `ps` samples by child `--job-id`; `vmmap`/physical-footprint observation is disabled for clean timing.
- Excluded: `tp_sl_grid/arity_7/long_only` because `exclude_heaviest_140s_job`.

## API-runner path

- Required jobs: `4`
- Pass: `no`
- State requirement: `queued -> running -> succeeded`.

## Mac Studio results

- Overall pass: `no`
- Scheduler pass: `yes`
- Lazy cache memory pass: `no`

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
| `none/arity_6/long_only` | 12 | 15.968 | 15.694 | 0.983 | 1135.136 | 1142.100 | 1163.700 | fail |
| `none/arity_6/long_short_reversal` | 12 | 15.810 | 15.365 | 0.972 | 1131.073 | 1135.600 | 1160.500 | fail |
| `tp_sl_grid/arity_6/long_only` | 12 | 16.566 | 17.446 | 1.053 | 995.031 | 1132.400 | 1152.700 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 12 | 15.504 | 16.204 | 1.045 | 1047.192 | 1128.100 | 1148.100 | pass |

## Memory release

- Checked jobs: `4`
- Failed jobs: `['none/arity_6/long_only', 'none/arity_6/long_short_reversal']`
- System memory failed jobs: `['none/arity_6/long_only', 'none/arity_6/long_short_reversal']`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `a27f5f3e-3c1e-4f71-8ab4-241d9c8dc3f6`
- Cache hit retained RSS delta: `32768`
- Pass: `no`

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
