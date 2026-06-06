# Stage 02 Row/Signature Telemetry — API runner clean arity-6 CPU/memory benchmark

Stage 02 shadow row/signature telemetry benchmark на Mac Studio для API-runner
path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / arity 6, 12 Numba
threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `1.007..1.088`.
- CPU: `pass`; mean child CPU = `1075.975..1170.991%`.
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
| `none/arity_6/long_only` | 12 | 15.446 | 15.694 | 1.016 | 1170.991 | 1170.200 | 1189.100 | pass |
| `none/arity_6/long_short_reversal` | 12 | 15.258 | 15.365 | 1.007 | 1170.245 | 1168.300 | 1185.800 | pass |
| `tp_sl_grid/arity_6/long_only` | 12 | 16.042 | 17.446 | 1.088 | 1075.975 | 1162.000 | 1186.600 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 12 | 15.417 | 16.204 | 1.051 | 1152.618 | 1165.600 | 1180.700 | pass |

## Stage 01 instrumentation counters

- Pass: `yes`
- Required fields: `26`

| Job | artifact load ms | signal pack ms | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | null fields |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 81.124 | n/a | 46656 | 46656 | 3020.587 | n/a | 36 | 36 | 46656 | 10.388 | `['signals_pack_ms', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 79.260 | n/a | 46656 | 46656 | 3057.798 | n/a | 36 | 36 | 46656 | 10.400 | `['signals_pack_ms', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `tp_sl_grid/arity_6/long_only` | 81.113 | n/a | 46656 | 46656 | 2908.366 | 6424580.597 | 36 | 36 | 46656 | 10.653 | `['signals_pack_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |
| `tp_sl_grid/arity_6/long_short_reversal` | 83.599 | n/a | 46656 | 46656 | 3026.310 | 6685119.370 | 36 | 36 | 46656 | 10.800 | `['signals_pack_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |

## Memory release

- Checked jobs: `4`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `1fbb9939-1d49-474e-ad30-145f2d0cb49d`
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
