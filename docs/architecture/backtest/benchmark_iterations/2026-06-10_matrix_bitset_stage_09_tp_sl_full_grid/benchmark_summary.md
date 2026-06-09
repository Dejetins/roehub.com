# Stage 09 TP/SL full-grid cell backend API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / tp_sl_grid / arity 6 / full request grid, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `fail`; speed ratio May2/current = `0.697..0.937`.
- CPU: `pass`; mean child CPU = `1023.511..1181.664%`.
- Acceptance: `fail`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить Stage 09 `matrix_cell_tp_sl_v1` full-grid TP/SL path: exact parity, cell-block counters, `trade_cell_evals_per_sec`, memory cleanup и service wall против May 2 reference.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `c3d38bcaca98e1837e0a55ea652fa4c28b0ac09e+dirty-f7acc0d6`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / tp_sl_grid/arity_6/long_only + tp_sl_grid/arity_6/long_short_reversal / full grid / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
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
| `tp_sl_grid/arity_6/long_only` | 12 | 25.032 | 17.446 | 0.697 | 1023.511 | 1163.250 | 1188.200 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 12 | 17.289 | 16.204 | 0.937 | 1181.664 | 1186.400 | 1190.200 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `50`

| Job | artifact load ms | signal pack source | signal pack ms | sidecar load ms | sidecar used | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | sidecar fallback | null fields |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `tp_sl_grid/arity_6/long_only` | 85.768 | runtime_pack | 28.019 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 1863.854 | 4117253.170 | 36 | 36 | 46656 | 10.657 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes']` |
| `tp_sl_grid/arity_6/long_short_reversal` | 78.942 | runtime_pack | 24.333 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 2698.520 | 5961029.814 | 36 | 36 | 46656 | 10.405 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes']` |

## TP/SL cell backend

| Job | backend | block | blocks/candidate | block bytes | trade-cell evals |
| --- | --- | --- | ---: | ---: | ---: |
| `tp_sl_grid/arity_6/long_only` | matrix_cell_tp_sl_v1 | 16 x 16 | 9 | 6664 | 103063104 |
| `tp_sl_grid/arity_6/long_short_reversal` | matrix_cell_tp_sl_v1 | 16 x 16 | 9 | 6664 | 103063104 |

## Memory release

- Checked jobs: `2`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `3021d77f-be1b-464c-bf36-2e543c7156e3`
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
- `child_process_evidence/*.json`

## Operator Commands

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py
```

Accounting validator note: `validate_benchmark_accounting.py` expects canonical notebook benchmark JSON, not the API-runner `benchmark_results.json` schema emitted by this harness.
