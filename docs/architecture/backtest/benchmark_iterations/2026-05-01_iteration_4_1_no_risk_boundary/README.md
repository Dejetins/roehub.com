# Backtest Benchmark Iteration 4.1 — no-risk boundary smoke

Service hygiene smoke for the Iteration 4.1 no-risk exact DTO/service boundary.
This is not canonical benchmark acceptance and does not add a measured benchmark
stage.

## Scope

- Implemented: internal no-risk exact config, compact result DTOs, service shell,
  deterministic `risk.mode='none'` boundary validation.
- Not in scope: exact scoring kernels, self-check implementation, heap update,
  top result proxy fill, persistence/API identity.

## Version

- Branch: `main`
- Commit: `ac773f309ae82e101930889a82a46168076dbbda`
- Service command: inline Python smoke in `/opt/roehub/app`
- Benchmark command: none
- Notebook baseline: not used
- Request hash: not generated
- Engine/config hash: not generated

## Environment

- Host: Mac Studio
- Runtime path: `/opt/roehub/app`
- Python: `/opt/roehub/app/.venv/bin/python`
- Delivery note: `/opt/roehub/app` is a deployed runtime copy, not a git
  checkout. File presence and sha256 hashes were verified against the local
  commit after GitHub deploy workflows completed.

## Runtime Metrics Without Warmup

No canonical notebook-compatible stage was measured for Iteration 4.1.

| Stage | Required | Notebook wall s | Service wall s | Pass |
|---|---|---:|---:|---|
| exact_scoring | no | | | not run |
| heap_update | no | | | not run |
| top_result_proxy_fill | no | | | not run |

## Service Hygiene Smoke

| Check | Value |
|---|---|
| smoke_kind | service hygiene smoke, not canonical benchmark acceptance |
| risk.mode | `none` |
| request_top_n | `100` |
| benchmark_top_k | `5` |
| heap_capacity | `5` |
| top_results_count | `0` |
| result_is_compact | `true` |
| result_contains_heavy_references | `false` |
| retained_heavy_reference_names | `[]` |
| heavy_references_released | `trade_T`, `signal_returns_15m`, `exact_context_starts`, `proxy_context_eval_stack` |

## Memory Cleanup Evidence

Cleanup evidence is a service hygiene check, not a canonical notebook stage.

| Check | Value |
|---|---|
| cleanup_duration_s | not measured |
| rss_before_mb | not measured |
| rss_peak_mb | not measured |
| rss_after_cleanup_mb | not measured |
| retained_rss_delta_mb | not measured |
| repeated_run_count | 1 |
| monotonic_retained_rss_growth | not applicable |
| worker_recycled | not applicable |
| pass | true |

Raw smoke output:

```json
{"benchmark_top_k": 5, "heap_capacity": 5, "heavy_references_released": {"exact_context_starts": true, "proxy_context_eval_stack": true, "signal_returns_15m": true, "trade_T": true}, "memory cleanup evidence": {"checked_reference_names": ["prepared_result", "combo_planning_result", "prepared_pools", "exact_context", "proxy_context"], "result_contains_heavy_references": false, "result_is_compact": true, "retained_heavy_reference_names": []}, "request_top_n": 100, "risk.mode": "none", "smoke_kind": "service hygiene smoke, not canonical benchmark acceptance", "top_results_count": 0}
```

## CI / Deploy Evidence

- CI: `25190327875`, success
- Deploy Backend: `25190327851`, success
- Publish App Image: `25190327857`, success
- Deploy Web: `25190363673`, success

## Decision

- Status: pass for Iteration 4.1 service boundary hygiene.
- Reason: boundary imports cleanly on Mac Studio, preserves
  `request_top_n` separately from `benchmark_top_k`, returns compact telemetry
  and result DTOs, and does not retain heavy per-job arrays through the result.
- Next iteration: Iteration 4.2 may add exact scoring kernels and self-check,
  with measured benchmark stages kept separate from this boundary smoke.
