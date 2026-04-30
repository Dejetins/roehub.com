# Backtest Benchmark Iteration 4.6 — benchmark runner accounting

Validation record for benchmark runner accounting boundaries. This record checks
canonical stage normalization and service-only telemetry separation; it does not
implement no-risk scoring algorithms.

## Scope

- Implemented: accounting helper and CLI validation for canonical stage order,
  alias normalization, notebook-compatible `total_without_warmup`, separate
  `service_total_without_warmup`, and runner metadata fields.
- Not in scope: no-risk scoring algorithms, heap/proxy optimization, public API,
  persistence, lazy trades implementation.

## Version

- Branch: pending commit
- Commit: pending commit
- Service command: `uv run python scripts/backtest/validate_benchmark_accounting.py`
- Benchmark command:
  `uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/local_accounting_validation.json`
- Notebook baseline:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`

## Accounting Rules

- Canonical stage order is fixed and includes `prepare_pools_core` as the service
  alias for notebook `prepare_pools`.
- Canonical JSON `total` is treated as a historical alias of
  `total_without_warmup`, not as a separate comparison stage.
- `total_without_warmup` is computed only from notebook-compatible stages.
- `service_total_without_warmup` is recorded under service-only telemetry and is
  not compared against canonical `total_without_warmup`.
- Memory cleanup evidence remains service hygiene and is not a canonical stage.

## Local Validation Evidence

- Evidence JSON: `local_accounting_validation.json`
- Result: pass
- `request.top_n = 100`
- `benchmark_top_k = 5`
- `sample_warmup_top_k = 1`
- `top_results_count = 5`
- heap capacity: `5`
- canonical runs checked: `28`
- `prepare_pools_core` normalized presence: `28 / 28`
- `service_total_without_warmup` in canonical stage list: no

## Mac Studio Pipeline

Pending until the local change is committed, pushed, and validated in
`/opt/roehub/app`.

Planned command on Mac Studio:

```bash
cd /opt/roehub/app
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/macstudio_accounting_validation.json
```

## Decision

- Status: local pass, Mac Studio validation pending.
- Reason: local validation confirms accounting separation and canonical stage
  normalization against current canonical JSON.
- Next iteration: run the same validation after the pushed revision is present
  in the Mac Studio runtime.
