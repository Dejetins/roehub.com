# Stage 04 matrix bitset no-risk MVP API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / arity 2-3 / long_only, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `fail`; speed ratio May2/current = `0.691..2.590`.
- CPU: `pass`; mean child CPU = `43.300..75.700%`.
- Acceptance: `fail`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить приемочный путь API-created job -> runner -> одноразовый heavy child process с 12 Numba threads для Stage 04 `matrix_bitset_no_risk_v1` MVP rows и сравнить exact scoring с May 2 reference.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `3dc4726f30081968687299e38c01a196c8d7e443`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / none/arity_2..3/long_only / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
- Full jobs policy: `heavy_only`, `ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`, `ROEHUB_BACKTEST_LIGHT_CONCURRENCY=0`, `NUMBA_NUM_THREADS=12`.
- CPU sampler: sustained `ps` samples by child `--job-id`; `vmmap`/physical-footprint observation is disabled for clean timing.
- Excluded: `None` because `exclude_heaviest_140s_job`.

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

- Required jobs: `2`
- Pass: `yes`
- State requirement: `queued -> running -> succeeded`.

## Mac Studio results

- Overall pass: `no`
- Scheduler pass: `yes`
- Lazy cache memory pass: `yes`

## Parity

- Passed jobs: `2/2`
- Failed jobs: `[]`

## Performance

- Stage timing jobs: `2`
- CPU sampling jobs: `2`
- CPU sampling failed jobs: `[]`
- Policy: API/runner wall and service-only overhead are recorded separately from May 2 notebook-compatible stage timings.

| Job | Threads | Exact current s | Exact May2 s | Ratio | CPU mean % | CPU p50 % | CPU max % | System memory gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_2/long_only` | 12 | 0.004 | 0.003 | 0.691 | 75.700 | 75.700 | 75.700 | pass |
| `none/arity_3/long_only` | 12 | 0.018 | 0.048 | 2.590 | 43.300 | 43.300 | 86.600 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `34`

| Job | artifact load ms | signal pack ms | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | null fields |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_2/long_only` | 77.792 | 28.448 | 656832 | 3421 | True | True | 36 | 36 | 9772.098 | n/a | 12 | 12 | 36 | 42.478 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_3/long_only` | 89.995 | 27.501 | 985248 | 3421 | True | True | 216 | 216 | 11748.442 | n/a | 18 | 18 | 216 | 249.562 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |

## Memory release

- Checked jobs: `2`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `8533e27d-b259-4052-a79e-0b98af41e406`
- Cache hit retained RSS delta: `65536`
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
- `child_process_evidence/*.json`

## Operator Commands

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py
```

Accounting validator note: `validate_benchmark_accounting.py` expects canonical notebook benchmark JSON, not the API-runner `benchmark_results.json` schema emitted by this harness.
