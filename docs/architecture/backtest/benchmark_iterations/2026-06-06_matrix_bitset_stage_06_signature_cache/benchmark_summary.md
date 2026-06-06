# Stage 06 consensus signature cache API-runner benchmark

Stage 06 measured an opt-in consensus signature cache candidate on top of the
accepted Stage 05 `matrix_bitset_no_risk_v1` heavy rows. The benchmark passed
parity, memory, legacy-path and May 2 comparison gates, but it failed the Stage
06 acceptance comparison against the immediately accepted Stage 05 baseline.

Decision: Stage 06 is `rejected`. Cache hit-rate was measurable, but the
candidate regressed the hot exact-scoring and service-wall timings.

| Job | Stage 05 exact s | Stage 06 exact s | Stage 05 service wall s | Stage 06 service wall s | Cache hit-rate | Cache hits | Unique consensus | Collision count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none/arity_6/long_only` | 1.010 | 4.932 | 1.590 | 5.366 | 0.202396 | 9443 | 37213 | 0 |
| `none/arity_6/long_short_reversal` | 2.887 | 6.166 | 3.135 | 6.414 | 0.202396 | 9443 | 37213 | 0 |

The measured candidate used a dirty Mac Studio checkout based on
`9ecdb97591d32f1691291ac7c3335cfc3ef530c7` with the local Stage 05 source
changes and the Stage 06 cache candidate copied into the checkout. The remote
checkout was restored after evidence was copied back.

Чистый Mac Studio benchmark для API-runner path без `vmmap`-наблюдения в hot path: BTCUSDT / 15m / none / arity 6 / long_only + long_short_reversal, 12 Numba threads, `top_n=50`, `benchmark_top_k=5`, sustained CPU sampler и memory gate.

## Короткий вывод

- Compute/performance: `pass`; speed ratio May2/current = `2.492..3.182`.
- CPU: `pass`; mean child CPU = `220.140..344.317%`.
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
- Filled runtime keys: `[]`
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

## Performance

- Stage timing jobs: `2`
- CPU sampling jobs: `2`
- CPU sampling failed jobs: `[]`
- Policy: API/runner wall and service-only overhead are recorded separately from May 2 notebook-compatible stage timings.

| Job | Threads | Exact current s | Exact May2 s | Ratio | CPU mean % | CPU p50 % | CPU max % | System memory gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 12 | 4.932 | 15.694 | 3.182 | 220.140 | 202.700 | 280.200 | pass |
| `none/arity_6/long_short_reversal` | 12 | 6.166 | 15.365 | 2.492 | 344.317 | 287.600 | 484.500 | pass |

## Instrumentation counters

- Pass: `yes`
- Required fields: `40`

| Job | artifact load ms | signal pack ms | signal pack bytes | W | padding valid | consensus sample parity | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | unique rows | consensus signatures | row signature ms | null fields |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none/arity_6/long_only` | 79.847 | 24.534 | 1970496 | 3421 | True | True | 46656 | 46656 | 9459.577 | n/a | 36 | 36 | 46656 | 10.395 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |
| `none/arity_6/long_short_reversal` | 78.118 | 24.458 | 1970496 | 3421 | True | True | 46656 | 46656 | 7566.751 | n/a | 36 | 36 | 46656 | 10.586 | `['tp_sl_exact_scoring_ms', 'rows_before_prefilter', 'avg_segments_per_candidate', 'avg_trades_per_candidate', 'tp_count', 'sl_count', 'trade_cell_evals_per_sec']` |

## Memory release

- Checked jobs: `2`
- Failed jobs: `[]`
- System memory failed jobs: `[]`
- System cleanup limit bytes: `536870912`
- vmmap / physical footprint: `no`

## Lazy cache-hit memory

- Target job: `4a1faa93-3bd8-4d15-bf72-9c04c40ef5e7`
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
