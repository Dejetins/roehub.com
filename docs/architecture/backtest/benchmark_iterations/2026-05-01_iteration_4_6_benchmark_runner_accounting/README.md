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

- Branch: `main`
- Commit: `de93e76068c8047c1f8fb83c5ad03b1ed250d656`
- Service command: `uv run python scripts/backtest/validate_benchmark_accounting.py`
- Benchmark command:
  `uv run python scripts/backtest/validate_benchmark_accounting.py --out docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/local_accounting_validation.json`
- Notebook baseline:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`

## Environment

- Host: `MacStudioDaniil`
- Runtime path: `/opt/roehub/app`
- Runtime layout: deployed runtime copy, not a git checkout.
- Commit verification: Mac Studio repo checkout
  `/Users/daniildegtyarev/Projects/roehub.com` was fast-forwarded to
  `de93e76068c8047c1f8fb83c5ad03b1ed250d656`; the `/opt/roehub/app` copies of
  `scripts/backtest/validate_benchmark_accounting.py`,
  `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
  and the canonical `benchmark_results.json` matched that commit by SHA-256.

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

Mac Studio validation passed in `/opt/roehub/app`.

Executed command:

```bash
cd /opt/roehub/app
export PATH="/opt/homebrew/bin:$PATH"
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_4_6_benchmark_runner_accounting/macstudio_accounting_validation.json
```

Evidence JSON: `macstudio_accounting_validation.json`

| Field | Local | Mac Studio |
|---|---:|---:|
| `request.top_n` | `100` | `100` |
| `benchmark_top_k` | `5` | `5` |
| `sample_warmup_top_k` | `1` | `1` |
| `top_results_count_values` | `[5]` | `[5]` |
| `heap_capacity` | `5` | `5` |
| `prepare_pools_alias_normalized` | `true` | `true` |
| `service_total_compared_to_canonical` | `false` | `false` |

Alias checks:

- `prepare_pools` is normalized to `prepare_pools_core`;
- `total` is normalized to `total_without_warmup`;
- `service_total_without_warmup` is service-only telemetry and is not part of
  canonical stage comparison.

## Decision

- Status: pass.
- Reason: local and Mac Studio validation agree on accounting metadata, canonical
  stage order, alias normalization and service-only telemetry separation.
- Next iteration: use this accounting record as the runner boundary before
  measuring scorer/heap/proxy-fill stages.
