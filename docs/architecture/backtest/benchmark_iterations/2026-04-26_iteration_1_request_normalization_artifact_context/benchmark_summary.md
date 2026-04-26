# Backtest Benchmark Iteration 1 — Request Normalization And Artifact Context

Mac Studio benchmark gate for Iteration 1 API shell: request normalization, preflight smoke, artifact current/root resolution, and deterministic failure evidence.

## Scope

- Implemented: benchmark evidence for `GET /backtests/runtime-defaults` / `POST /backtests/preflight` shell, request normalization, cost estimate, request hash, and artifact context resolver.
- Not in scope: job creation, persistence, `prepare_pools`, scoring, top-N heap, lazy trades, UI, or internal API.

## Version

- Branch: `main`
- Commit: `13128b29bbfef916b26ab50d66a654e8efd2f115`
- Git status on Mac Studio: `clean`
- Benchmark command: remote Python smoke via `ssh macstudio`, saved to `benchmark_results.json`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`
- Artifact slot: `slot_a` generation `3`
- Artifact manifest hash: `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`
- Hit-times manifest hash: `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`
- Notebook baseline: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`
- Canonical baseline request hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`
- Preflight request hash: `21592bfd156191344a2e7500196a5e41a1b2a7aee28847d588a6ac2581258418`
- Result config hash: `d4ae1799e9498d81d132eb12aa80c9476358a60e3da5652651db50b553fdb9a9`

## Fixture

- Coordinates: `{'exchange': 'binance', 'market_type': 'spot', 'symbol': 'BTCUSDT'}`
- Timeframe: `15m`
- Time range: `{'end': '2026-04-11T20:08:00Z', 'start': '2020-01-11T20:08:00Z'}`
- Indicators: `[{'indicator_id': 'ma.dema', 'sources': ['close'], 'window': {'start': 5, 'step': 1, 'stop': 10}}]`
- Risk mode: `none`
- Execution settings: `{'close_on_end': True, 'direction_mode': 'long_short_reversal', 'fee_rate': 0.00075, 'initial_cash_quote': 10000.0, 'profit_lock': {'enabled': False}, 'sizing': {'equity_pct': 10.0, 'mode': 'fixed_equity_pct'}, 'slippage_rate': 0.0001}`
- Ranking: `{'direction': 'desc', 'primary_metric': 'total_return_pct'}`
- Top N: `100`

## Environment

- Host: `MacStudioDaniil` (`Mac Studio`, `Mac14,13`, Apple M2 Max)
- Platform: `macOS-15.7.5-arm64-arm-64bit`
- Python: `3.12.13`
- uv: `uv 0.10.11 (Homebrew 2026-03-16)`
- NUMBA_NUM_THREADS: `12`
- Thread count: `13`
- Load average at start: `[2.248046875, 2.11572265625, 2.0341796875]`

## Warmup Metrics

Iteration 1 has no Numba/scoring warmup. The measured code paths were warmed once before runtime timing. Full production warmup segments start in later iterations.

## Runtime Metrics Without Warmup

| Segment | Iterations | Mean ms | Median ms | P95 ms | Min ms | Max ms | CPU evidence | RSS delta bytes | Pass |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| request_normalization_preflight_without_artifact_io | 200 | 0.139 | 0.137 | 0.146 | 0.136 | 0.173 | cpu 100.0% | 0 | pass |
| artifact_context_resolve_current_root_hit_times_manifest | 200 | 68.018 | 66.673 | 73.963 | 66.179 | 77.091 | cpu 100.0% | 10469376 | pass |
| preflight_total_with_artifact_context | 200 | 68.190 | 66.828 | 74.137 | 66.216 | 77.610 | cpu 100.0% | 49152 | pass |

## Failure Evidence

| Case | Result | Error code | First issue |
|---|---|---|---|
| invalid_indicator | passed | `backtest.invalid_request` | `indicators.0.indicator_id` / `unknown_indicator` |
| invalid_source | passed | `backtest.invalid_request` | `indicators.0.sources.0` / `invalid_source` |
| invalid_window | passed | `backtest.invalid_request` | `indicators.0.window` / `invalid_window` |
| request_too_expensive | passed | `backtest.request_too_expensive` | `indicators` / `max_indicator_rows` |

## Quality Gates On Mac Studio

| Gate | Result |
|---|---|
| targeted Iteration 1 tests | 22 passed in 1.30s |
| `uv run pytest -q tests/unit/contexts/backtest` | 178 passed in 25.84s |
| `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py` | 4 passed in 0.55s |
| ruff | All checks passed! |
| pyright | 0 errors, 0 warnings, 0 informations |
| docs index check | OK: docs/architecture/README.md is up-to-date |
| git diff check | passed |

## Request Hash Parity

- Status: `not_applicable`
- Expected canonical notebook hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`
- Observed Iteration 1 preflight hash: `21592bfd156191344a2e7500196a5e41a1b2a7aee28847d588a6ac2581258418`
- Reason: canonical notebook fixture uses notebook request schema and full seven-indicator window_range workload; Iteration 1 preflight smoke uses a guardrail-valid public API fixture before Iteration 2 row prefilter semantics.

## Decision

- Status: pass
- Reason: Iteration 1 benchmark gate ran on Mac Studio with real artifact current/root manifests, deterministic preflight request hash, artifact hashes matching canonical current artifact, and required failure evidence.
- Next iteration: Iteration 2 may use this record as the accepted shell baseline before artifact arrays and `prepare_pools` work.

## Notes

- The 90% canonical scoring/runtime matrix is not applicable to Iteration 1 because no scoring or kernels are in scope.
- `artifact_context_resolve_current_root_hit_times_manifest` is service-only overhead and should be watched for regression once later stages add array loading.
