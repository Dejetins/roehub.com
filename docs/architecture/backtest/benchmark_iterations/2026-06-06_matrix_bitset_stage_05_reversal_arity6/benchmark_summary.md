# Stage 05 matrix bitset no-risk heavy API-runner benchmark

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / arity 6 / long_only + long_short_reversal, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `5.323..15.543`.
- CPU: `pass`; mean child CPU = `367.400..745.400%`.
- Acceptance: `pass`; старые vmmap-contaminated результаты не учитываются.
- Что не прошло: memory failed jobs = `[]`, lazy status path = `queued -> running -> completed`.

## Intent

Проверить приемочный путь API-created job -> runner -> одноразовый heavy child process с 12 Numba threads для Stage 05 `matrix_bitset_no_risk_v1` no-risk arity 6 rows и сравнить exact scoring с May 2 reference.

## Benchmark fixture

- Host: `MacStudioDaniil`
- Git commit: `9ecdb97591d32f1691291ac7c3335cfc3ef530c7`
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

## Old vs Matrix Top-50 A/B

Отдельный Mac Studio A/B прогон для `none/arity_6/long_only` сравнил старый
`event_segments_n_no_risk` и новый `matrix_bitset_no_risk_v1` на одном request
hash. Raw API payload hash не используется как identity-сигнал, потому что
`variant_key` и `links` содержат job id. Для вопроса "те же стратегии?"
сравнивается canonical strategy identity: rank, `variant_hash`,
`indicator_variant_hash`, canonical params, TP/SL поля и ordered top-50
variant sequence; summary metrics проверяются отдельно с числовой дельтой.

| Job | Request hash same | Old backend | Matrix backend | Top count old/matrix | Raw payload hash old | Raw payload hash matrix | Same canonical payload hash | Strategy identity hash old | Strategy identity hash matrix | Same strategy identity | Ordered variant hash same | Max metric abs diff | Exact old s | Exact matrix s |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `none/arity_6/long_only` | true | `event_segments_n_no_risk` | `matrix_bitset_no_risk_v1` | `50 / 50` | `e90723790642810a88fd5054259a6b3f3bcca6d0b28d65170e9c60baeb5e33cd` | `c703fd09e07547682275557c59850a2f667284cc51b543b84ec47e33801f26c7` | false | `30996c11f947d4dafaee2ceff4d51e509acaa468436d29773594d032b75684ab` | `30996c11f947d4dafaee2ceff4d51e509acaa468436d29773594d032b75684ab` | true | true | `1.066e-13` | 15.420 | 1.076 |

Canonical payload hash old/matrix:
`d924fffbab7daccc4a61b1a088ec9309d4e45131162c407edad4ddaa753143df` /
`c59ea128b8e0290323a563754d5acf693b6a3898788bf1f23b420fb6ff7ed1d6`.

Evidence: `ab_top50_parity.json`.

## Performance

- Stage timing jobs: `2`
- CPU sampling jobs: `2`
- CPU sampling failed jobs: `[]`
- Policy: API/runner wall and service-only overhead are recorded separately from May 2 notebook-compatible stage timings.

| Job | Threads | Exact current s | Exact May2 s | Ratio | CPU mean % | CPU p50 % | CPU max % | System memory gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 12 | 1.010 | 15.694 | 15.543 | 367.400 | 397.600 | 632.900 | pass |
| `none/arity_6/long_short_reversal` | 12 | 2.887 | 15.365 | 5.323 | 745.400 | 902.700 | 958.900 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `34`

| Job | artifact load ms | signal pack ms | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | null fields |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 87.398 | 24.982 | 1970496 | 3421 | True | True | 46656 | 46656 | 46207.826 | n/a | 36 | 36 | 46656 | 10.378 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 80.506 | 26.060 | 1970496 | 3421 | True | True | 46656 | 46656 | 16162.103 | n/a | 36 | 36 | 46656 | 10.591 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |

## Memory release

- Checked jobs: `2`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `0b605f32-49c1-49e4-8cda-1941058d250a`
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
- `ab_top50_parity.json`
- `ab_top50_child_process_evidence/*.json`
- `child_process_evidence/*.json`

## Operator Commands

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py
```

Accounting validator note: `validate_benchmark_accounting.py` expects canonical notebook benchmark JSON, not the API-runner `benchmark_results.json` schema emitted by this harness.
