# Iteration 15 API runner clean arity-6 CPU/memory benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / arity 6, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `1.014..1.079`.
- CPU: `pass`; mean child CPU = `1077.015..1169.518%`.
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
| `none/arity_6/long_only` | 12 | 15.484 | 15.694 | 1.014 | 1169.518 | 1174.200 | 1185.400 | pass |
| `none/arity_6/long_short_reversal` | 12 | 15.078 | 15.365 | 1.019 | 1161.925 | 1178.300 | 1196.000 | pass |
| `tp_sl_grid/arity_6/long_only` | 12 | 16.175 | 17.446 | 1.079 | 1077.015 | 1175.600 | 1183.600 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 12 | 15.254 | 16.204 | 1.062 | 1161.755 | 1168.100 | 1181.100 | pass |

## Stage 01 instrumentation counters

- Pass: `yes`
- Required fields: `19`

| Job | artifact load ms | signal pack ms | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | null fields |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 86.278 | n/a | 46656 | 46656 | 3013.182 | n/a | 36 | `['signals_pack_ms', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 80.319 | n/a | 46656 | 46656 | 3094.228 | n/a | 36 | `['signals_pack_ms', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `tp_sl_grid/arity_6/long_only` | 79.925 | n/a | 46656 | 46656 | 2884.457 | 6371765.328 | 36 | `['signals_pack_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |
| `tp_sl_grid/arity_6/long_short_reversal` | 79.095 | n/a | 46656 | 46656 | 3058.664 | 6756589.105 | 36 | `['signals_pack_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |

## Memory release

- Checked jobs: `4`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `8812690a-d860-4225-a860-069bb607a1fe`
- Cache hit retained RSS delta: `49152`
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
