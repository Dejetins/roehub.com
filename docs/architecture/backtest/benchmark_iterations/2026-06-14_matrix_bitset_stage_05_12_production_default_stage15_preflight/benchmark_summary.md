# Stage 05+12 no-risk production default API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / production default arity 6-7 / long_only + long_short_reversal, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `4.974..26.653`.
- CPU: `pass`; mean child CPU = `127.875..988.231%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить production composite default `stage_05_and_12_no_risk`: Stage 05 `matrix_bitset_no_risk_v1` для no-risk arity 6, Stage 12 `compiled_prefix_product_traversal_v1` для no-risk arity 7, parity, service wall and exact-scoring speed against accepted evidence.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `80a4391f+dirty-4b7ae7aa`
- Canonical JSON: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Reference: `2026-05-02_iteration_8_execution_sizing_completion`
- BTCUSDT / 15m / none/arity_6 Stage 05 + none/arity_7 Stage 12 production default / long_only + long_short_reversal / REQUEST_TOP_N = 50 / BENCHMARK_TOP_K = 5
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
| `none/arity_6/long_only` | 12 | 1.038 | 15.694 | 15.122 | 127.875 | 99.450 | 346.000 | pass |
| `none/arity_6/long_short_reversal` | 12 | 3.089 | 15.365 | 4.974 | 882.600 | 900.800 | 931.100 | pass |
| `none/arity_7/long_only` | 12 | 5.206 | 138.755 | 26.653 | 762.640 | 930.300 | 960.400 | pass |
| `none/arity_7/long_short_reversal` | 12 | 17.769 | 136.630 | 7.689 | 988.231 | 1075.400 | 1123.800 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `63`

| Job | artifact load ms | signal pack source | signal pack ms | sidecar load ms | sidecar used | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | sidecar fallback | null fields |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `none/arity_6/long_only` | 90.290 | runtime_pack | 28.142 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 44954.166 | n/a | 36 | 36 | 46656 | 10.614 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'prefix_nodes_visited', 'prefix_nodes_reused', 'prefix_pruned_subtrees', 'prefix_pruned_candidate_upper_bound', 'prefix_candidates_selected', 'prefix_candidates_pruned', 'selectivity_order', 'combo_iteration_candidates_per_sec', 'prefix_total_elapsed_s', 'prefix_compiled_loop_elapsed_s', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'early_abandon_candidates', 'early_abandon_cells', 'early_abandon_bound_ms', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 79.904 | runtime_pack | 24.295 | n/a | False | 1970496 | 3421 | True | True | 46656 | 46656 | 15104.008 | n/a | 36 | 36 | 46656 | 10.422 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'prefix_nodes_visited', 'prefix_nodes_reused', 'prefix_pruned_subtrees', 'prefix_pruned_candidate_upper_bound', 'prefix_candidates_selected', 'prefix_candidates_pruned', 'selectivity_order', 'combo_iteration_candidates_per_sec', 'prefix_total_elapsed_s', 'prefix_compiled_loop_elapsed_s', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'early_abandon_candidates', 'early_abandon_cells', 'early_abandon_bound_ms', 'trade_cell_evals_per_sec']` |
| `none/arity_7/long_only` | 80.212 | runtime_pack | 25.767 | n/a | False | 2298912 | 3421 | True | True | 279936 | 279936 | 22405.006 | n/a | 42 | 42 | 279936 | 12.408 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'early_abandon_candidates', 'early_abandon_cells', 'early_abandon_bound_ms', 'trade_cell_evals_per_sec']` |
| `none/arity_7/long_short_reversal` | 80.067 | runtime_pack | 25.389 | n/a | False | 2298912 | 3421 | True | True | 279936 | 279936 | 14878.636 | n/a | 42 | 42 | 279936 | 12.082 | n/a | `['sidecar_load_ms', 'sidecar_fallback_reason', 'sidecar_dir', 'tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'tp_sl_cell_backend_id', 'tp_sl_cell_block_shape', 'tp_sl_cell_blocks_per_candidate', 'tp_sl_cell_block_estimated_peak_bytes', 'tp_sl_cell_trade_cell_evals', 'tp_sl_selected_cell_shadow_status', 'tp_sl_selected_cell_parity_pass', 'tp_sl_selected_cell_scores', 'tp_sl_selected_cell_elapsed_ms', 'tp_sl_by_entry_selected_arrays_bytes', 'early_abandon_candidates', 'early_abandon_cells', 'early_abandon_bound_ms', 'trade_cell_evals_per_sec']` |

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

- Target job: `e1a348ff-5e23-46f7-9f1c-7775ab3f17dc`
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
