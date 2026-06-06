# Stage 05 matrix bitset no-risk heavy API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / arity 6 / long_only + long_short_reversal, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `5.139..15.274`.
- CPU: `pass`; mean child CPU = `481.200..859.100%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить приемочный путь API-created job -> runner -> одноразовый heavy child process с 12 Numba threads для Stage 05 `matrix_bitset_no_risk_v1` no-risk arity 6 rows и сравнить exact scoring с May 2 reference.

## Benchmark fixture

- Host: `Mac`
- Git commit: `e985b30123ca9070ef5b1fc3227ffef6dd3fdf35`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / none/arity_6/long_only + none/arity_6/long_short_reversal / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
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

## Matrix sidecar

- Enabled: `yes`
- Artifact dir: `docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets/sidecar_artifacts`
- sidecar_generate_ms: `7882.282`
- Fairness: `accepted_for_learning_sidecar_precomputed`
- Policy: Sidecar speedup is benchmark/test-only unless generation cost is included or a publisher plan is approved.

## API-runner path

- Required jobs: `2`
- Pass: `yes`
- State requirement: `queued -> running -> succeeded`.

## Mac Studio results

- Overall pass: `yes`
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
| `none/arity_6/long_only` | 12 | 1.028 | 15.694 | 15.274 | 481.200 | 481.200 | 481.200 | pass |
| `none/arity_6/long_short_reversal` | 12 | 2.990 | 15.365 | 5.139 | 859.100 | 887.200 | 904.200 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `40`

| Job | artifact load ms | signal pack source | signal pack ms | sidecar load ms | sidecar used | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | sidecar fallback | null fields |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `none/arity_6/long_only` | 80.734 | sidecar | 102.640 | 81.530 | True | 1970496 | 3421 | True | True | 46656 | 46656 | 45406.620 | n/a | 36 | 36 | 46656 | 10.605 | n/a | `['sidecar_fallback_reason', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 78.997 | sidecar | 97.899 | 75.238 | True | 1970496 | 3421 | True | True | 46656 | 46656 | 15604.110 | n/a | 36 | 36 | 46656 | 10.543 | n/a | `['sidecar_fallback_reason', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |

## Memory release

- Checked jobs: `2`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `3e0c1253-0522-492f-b004-de0438cff9af`
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
- `child_process_evidence/*.json`

## Operator Commands

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py
```

Accounting validator note: `validate_benchmark_accounting.py` expects canonical notebook benchmark JSON, not the API-runner `benchmark_results.json` schema emitted by this harness.
