# Iteration 15 API runner clean arity-6 CPU/memory benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / arity 6, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `1.013..1.085`.
- CPU: `pass`; mean child CPU = `1103.708..1175.700%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить приемочный путь API-created job -> runner -> одноразовый heavy child process с 12 Numba threads и сравнить arity 6 с May 2 reference.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `16ab06dc1506edaa7e292c2497595fcc3f008664`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / arity 6 only / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
- Full jobs policy: `heavy_only`, `ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`, `ROEHUB_BACKTEST_LIGHT_CONCURRENCY=0`, `NUMBA_NUM_THREADS=12`.
- CPU sampler: sustained `ps` samples by child `--job-id`; `vmmap`/physical-footprint observation is disabled for clean timing.
- Excluded: `tp_sl_grid/arity_7/long_only` because `exclude_heaviest_140s_job`.

## Runtime env

- Env file: `/Users/daniildegtyarev/.config/roehub/roehub.env`
- Env file loaded: `yes`
- Postgres DSN keys present: `['STRATEGY_PG_DSN', 'POSTGRES_DSN', 'IDENTITY_PG_DSN']`
- Postgres component keys present: `['POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']`
- Filled DSN keys: `[]`
- Artifact config path: `configs/prod/backtest_artifacts.yaml`
- Filled runtime keys: `['ROEHUB_ENV', 'ROEHUB_BACKTEST_ARTIFACTS_CONFIG']`
- Secret values: not recorded.

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
| `none/arity_6/long_only` | 12 | 15.416 | 15.694 | 1.018 | 1165.255 | 1184.300 | 1191.800 | pass |
| `none/arity_6/long_short_reversal` | 12 | 15.168 | 15.365 | 1.013 | 1175.700 | 1176.300 | 1195.400 | pass |
| `tp_sl_grid/arity_6/long_only` | 12 | 16.077 | 17.446 | 1.085 | 1103.708 | 1176.950 | 1189.700 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 12 | 15.175 | 16.204 | 1.068 | 1155.800 | 1173.300 | 1186.000 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `34`

| Job | artifact load ms | signal pack ms | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | null fields |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 78.115 | 23.617 | 1970496 | 3421 | True | True | 46656 | 46656 | 3026.389 | n/a | 36 | 36 | 46656 | 10.327 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 81.556 | 24.067 | 1970496 | 3421 | True | True | 46656 | 46656 | 3075.996 | n/a | 36 | 36 | 46656 | 10.378 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `tp_sl_grid/arity_6/long_only` | 78.806 | 24.195 | 1970496 | 3421 | True | True | 46656 | 46656 | 2901.988 | 6410492.391 | 36 | 36 | 46656 | 10.636 | `['rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |
| `tp_sl_grid/arity_6/long_short_reversal` | 80.451 | 24.063 | 1970496 | 3421 | True | True | 46656 | 46656 | 3074.583 | 6791754.397 | 36 | 36 | 46656 | 10.391 | `['rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |

## Memory release

- Checked jobs: `4`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `e737abca-b012-4729-96a3-e69344e907cd`
- Cache hit retained RSS delta: `0`
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
