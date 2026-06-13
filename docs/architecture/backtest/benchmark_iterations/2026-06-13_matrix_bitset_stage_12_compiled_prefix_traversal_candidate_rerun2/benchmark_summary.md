# Stage 12 compiled prefix traversal API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / arity 6-7 / long_only + long_short_reversal, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `5.259..26.472`.
- CPU: `pass`; mean child CPU = `772.400..985.115%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить Stage 12 `compiled_prefix_product_traversal_v1`: compiled prefix counters, arity-7 service wall, arity-6 no-regression, top-N parity and canonical output identity.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `d05d26f80509816f3251063cd1e3c99f3b361050+dirty-62caaca6`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / none/arity_6..7/long_only + none/arity_6..7/long_short_reversal / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
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
| `none/arity_6/long_only` | 12 | 0.871 | 15.694 | 18.016 | 772.400 | 772.400 | 772.400 | pass |
| `none/arity_6/long_short_reversal` | 12 | 2.922 | 15.365 | 5.259 | 920.350 | 920.350 | 945.800 | pass |
| `none/arity_7/long_only` | 12 | 5.242 | 138.755 | 26.472 | 851.400 | 950.100 | 1015.400 | pass |
| `none/arity_7/long_short_reversal` | 12 | 17.549 | 136.630 | 7.786 | 985.115 | 1078.700 | 1136.000 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `60`

| Job | artifact load ms | signal pack source | signal pack ms | sidecar load ms | sidecar used | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | sidecar fallback | null fields |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `none/arity_6/long_only` | 79.726 | runtime_pack | 24.695 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 22316.507 | n/a | 36 | 36 | 46656 | 10.434 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 80.342 | runtime_pack | 24.370 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 15080.937 | n/a | 36 | 36 | 46656 | 10.414 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |
| `none/arity_7/long_only` | 89.825 | runtime_pack | 25.234 | n/a | False | 2298912 | 3421 | True | True | 279936 | 279936 | 22252.969 | n/a | 42 | 42 | 279936 | 12.097 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |
| `none/arity_7/long_short_reversal` | 81.619 | runtime_pack | 25.183 | n/a | False | 2298912 | 3421 | True | True | 279936 | 279936 | 15065.700 | n/a | 42 | 42 | 279936 | 12.368 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |

## TP/SL cell backend

| Job | backend | block | blocks/candidate | block bytes | trade-cell evals |
| --- | --- | --- | ---: | ---: | ---: |
| `none/arity_6/long_only` | n/a | n/a | n/a | n/a | n/a |
| `none/arity_6/long_short_reversal` | n/a | n/a | n/a | n/a | n/a |
| `none/arity_7/long_only` | n/a | n/a | n/a | n/a | n/a |
| `none/arity_7/long_short_reversal` | n/a | n/a | n/a | n/a | n/a |

## Memory release

- Checked jobs: `4`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `25d9ad3a-c4f7-4195-913d-c5329c3fe609`
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
