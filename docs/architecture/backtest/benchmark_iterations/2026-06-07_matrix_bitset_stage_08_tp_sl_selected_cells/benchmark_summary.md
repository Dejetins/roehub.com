# Stage 08 TP/SL selected-cell API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / tp_sl_grid / selected 8x8 cells, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `0.857..1.269`.
- CPU: `pass`; mean child CPU = `98.000..109.000%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить Stage 08 TP/SL selected-cell shadow: parity для `tp_count <= 8` и `sl_count <= 8`, правило `SL wins`, by-entry hit-times layout counters и отсутствие production top-N feed из shadow path.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `e985b30123ca9070ef5b1fc3227ffef6dd3fdf35`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / tp_sl_grid selected `tp_count <= 8`, `sl_count <= 8` / long_only + long_short_reversal / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
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

- Enabled: `no`
- Artifact dir: `None`
- sidecar_generate_ms: `n/a`
- Fairness: `no_sidecar`
- Policy: None

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
| `tp_sl_grid/arity_1/long_only` | 12 | 0.003 | 0.004 | 1.269 | 98.000 | 98.000 | 98.000 | pass |
| `tp_sl_grid/arity_2/long_short_reversal` | 12 | 0.008 | 0.007 | 0.857 | 109.000 | 109.000 | 109.000 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `45`

| Job | artifact load ms | signal pack source | signal pack ms | sidecar load ms | sidecar used | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | sidecar fallback | null fields |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `tp_sl_grid/arity_1/long_only` | 73.778 | runtime_pack | 31.324 | n/a | False | 328416 | 3421 | True | True | 6 | 6 | 2083.032 | 133314.031 | 6 | 6 | 6 | 12.040 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |
| `tp_sl_grid/arity_2/long_short_reversal` | 76.256 | runtime_pack | 25.970 | n/a | False | 656832 | 3421 | True | True | 36 | 36 | 4429.883 | 283512.528 | 12 | 12 | 36 | 43.056 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate']` |

## Memory release

- Checked jobs: `2`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `d7b7a61d-ff59-4044-b0a0-8d5a58fea5c6`
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
