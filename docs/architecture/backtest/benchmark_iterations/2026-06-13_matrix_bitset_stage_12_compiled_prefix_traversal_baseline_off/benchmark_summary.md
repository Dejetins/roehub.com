# Stage 12 compiled prefix traversal API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / arity 6-7 / long_only + long_short_reversal, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `1.008..1.021`.
- CPU: `pass`; mean child CPU = `1171.346..1181.600%`.
- Acceptance: `fail`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `['none/arity_7/long_short_reversal']`, lazy status path = `queued -> running -> completed`.

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
- Pass: `no`
- State requirement: `queued -> running -> succeeded`.

## Mac Studio results

- Overall pass: `no`
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
| `none/arity_6/long_only` | 12 | 15.441 | 15.694 | 1.016 | 1181.600 | 1180.600 | 1187.900 | pass |
| `none/arity_6/long_short_reversal` | 12 | 15.053 | 15.365 | 1.021 | 1175.164 | 1182.600 | 1193.400 | pass |
| `none/arity_7/long_only` | 12 | 137.120 | 138.755 | 1.012 | 1171.737 | 1180.950 | 1196.200 | pass |
| `none/arity_7/long_short_reversal` | 12 | 135.552 | 136.630 | 1.008 | 1171.346 | 1175.500 | 1196.000 | fail |

## Instrumentation counters

- Pass: `yes`
- Required fields: `60`

| Job | artifact load ms | signal pack source | signal pack ms | sidecar load ms | sidecar used | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | sidecar fallback | null fields |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `none/arity_6/long_only` | 87.023 | runtime_pack | 28.664 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 3021.623 | n/a | 36 | 36 | 46656 | 10.707 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'prefix_nodes_visited', 'prefix_nodes_reused', 'prefix_pruned_subtrees', 'prefix_pruned_candidate_upper_bound', 'prefix_candidates_selected', 'prefix_candidates_pruned', 'selectivity_order', 'combo_iteration_candidates_per_sec', 'prefix_total_elapsed_s', 'prefix_compiled_loop_elapsed_s', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 78.685 | runtime_pack | 24.753 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 3099.349 | n/a | 36 | 36 | 46656 | 10.397 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'prefix_nodes_visited', 'prefix_nodes_reused', 'prefix_pruned_subtrees', 'prefix_pruned_candidate_upper_bound', 'prefix_candidates_selected', 'prefix_candidates_pruned', 'selectivity_order', 'combo_iteration_candidates_per_sec', 'prefix_total_elapsed_s', 'prefix_compiled_loop_elapsed_s', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |
| `none/arity_7/long_only` | 89.897 | runtime_pack | 25.021 | n/a | False | 2298912 | 3421 | True | True | 279936 | 279936 | 2041.541 | n/a | 42 | 42 | 279936 | 12.199 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'prefix_nodes_visited', 'prefix_nodes_reused', 'prefix_pruned_subtrees', 'prefix_pruned_candidate_upper_bound', 'prefix_candidates_selected', 'prefix_candidates_pruned', 'selectivity_order', 'combo_iteration_candidates_per_sec', 'prefix_total_elapsed_s', 'prefix_compiled_loop_elapsed_s', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |
| `none/arity_7/long_short_reversal` | 88.548 | runtime_pack | 25.126 | n/a | False | 2298912 | 3421 | True | True | 279936 | 279936 | 2065.159 | n/a | 42 | 42 | 279936 | 12.297 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'prefix_nodes_visited', 'prefix_nodes_reused', 'prefix_pruned_subtrees', 'prefix_pruned_candidate_upper_bound', 'prefix_candidates_selected', 'prefix_candidates_pruned', 'selectivity_order', 'combo_iteration_candidates_per_sec', 'prefix_total_elapsed_s', 'prefix_compiled_loop_elapsed_s', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'trade_cell_evals_per_sec']` |

## TP/SL cell backend

| Job | backend | block | blocks/candidate | block bytes | trade-cell evals |
| --- | --- | --- | ---: | ---: | ---: |
| `none/arity_6/long_only` | n/a | n/a | n/a | n/a | n/a |
| `none/arity_6/long_short_reversal` | n/a | n/a | n/a | n/a | n/a |
| `none/arity_7/long_only` | n/a | n/a | n/a | n/a | n/a |
| `none/arity_7/long_short_reversal` | n/a | n/a | n/a | n/a | n/a |

## Memory release

- Checked jobs: `4`
- Failed jobs: `['none/arity_7/long_short_reversal']`
- System memory failed jobs: `['none/arity_7/long_short_reversal']`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `df0753cc-d596-42a8-aced-c31d7d73d835`
- Cache hit retained RSS delta: `32768`
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
