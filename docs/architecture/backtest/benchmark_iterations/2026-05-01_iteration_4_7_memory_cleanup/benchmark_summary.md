# Iteration 4.7 memory cleanup evidence

Repeated no-risk service smoke for bounded per-job reference lifecycle.

## Scope

- Implemented check: `memory cleanup evidence` for no-risk scoring result compactness.
- Not in scope: canonical benchmark stage addition, notebook timer comparison, scoring changes.
- Cleanup is service hygiene, not a canonical benchmark stage.

## Version

- Branch/commit: `54f97cffbc2b4bf948670eb0d33f070f2928e8a0`
- Git status: `runtime-copy-no-git`
- Service command: `uv run python scripts/backtest/run_iteration_4_7_memory_cleanup_smoke.py`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Canonical JSON for request shape: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`

## Environment

- Host: `MacStudioDaniil`
- Python: `3.12.13`
- Worker lifecycle: one Python process; production worker recycle boundary is not implemented here.

## Fixture

- Risk mode: `none`
- Arity: `7`
- Direction mode: `long_short_reversal`
- Rows per indicator: `6`
- Warmup runs: `1`
- Repeated run count: `3`

## Memory Cleanup Evidence

Cleanup evidence is a service hygiene check, not a canonical notebook stage. It does not change the ordered stage list and is not compared with `>= 90%` notebook targets.

| Check | Value |
|---|---:|
| cleanup_duration_s | 0.000015417 |
| rss_before | 1376518144 |
| rss_peak | 2325315584 |
| rss_after_cleanup | 1913389056 |
| retained_rss_delta | 536870912 |
| rss_before_mb | 1312.750 |
| rss_peak_mb | 2217.594 |
| rss_after_cleanup_mb | 1824.750 |
| retained_rss_delta_mb | 512.000 |
| retained_rss_delta_series_mb | `493.781, 9.547, 8.672` |
| rss_after_cleanup_series_mb | `1806.531, 1816.078, 1824.750` |
| repeated_run_count | 3 |
| monotonic_retained_rss_growth | `False` |
| monotonic_rss_after_cleanup_growth | `True` |
| worker_recycled | `False` |
| pass | `True` |

Per-run retained RSS:

| run | rss_before_mb | rss_peak_mb | rss_after_cleanup_mb | retained_rss_delta_mb | cleanup_duration_s | compact |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1312.750 | 2177.516 | 1806.531 | 493.781 | 0.000015417 | `True` |
| 2 | 1806.531 | 2209.000 | 1816.078 | 9.547 | 0.000005500 | `True` |
| 3 | 1816.078 | 2217.594 | 1824.750 | 8.672 | 0.000005833 | `True` |

## Decision

- Overall pass: `yes`
- macOS RSS note: allocator caches may keep RSS above the starting value; this evidence checks the per-run `retained_rss_delta` trend and compact DTOs rather than expecting immediate OS return.
