# Backtest Benchmark Iteration 2 - prepare_pools

Iteration 2 implements artifact array mmap loading, `[start, end)` 15m slicing,
signal row extraction, row prefilter, compressed signal segments, and
`prepare_pools` timing.

## Status

- Acceptance: pending Mac Studio.
- Local evidence: synthetic developer smoke only, not acceptance evidence.
- Canonical target:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.

## Local Smoke

| Metric | Value |
|---|---:|
| Host | `Mac` |
| Stage | `prepare_pools` |
| Wall time | `0.7342308330116794` |
| Process CPU time | `0.683835064` |
| Process CPU percent equivalent | `93.13624997128422` |
| Peak RSS MB | `157.328125` |
| RSS delta MB | `58.578125` |
| Thread count | `12` |
| Numba threads | `11` |

## Subsegments

| Segment | Wall time s |
|---|---:|
| `artifact_manifest_load` | `0.0033002079580910504` |
| `artifact_array_mmap_load` | `0.006055750011000782` |
| `time_range_slice` | `0.0003587499959394336` |
| `signal_row_selection` | `0.00008920795517042279` |
| `row_prefilter` | `0.48249816696625203` |
| `segment_build` | `0.241826250043232` |

## Identity

- Request hash:
  `99577549d04783838e3c325657b601a0830c36280417533d413420b952aecce2`
- Artifact manifest hash:
  `b7a2970e8bb8f240c92ef55b3926467f642ad8d25f625f731496d13c092d51eb`
- Row metadata/order hash:
  `45876855a2852ec2e3a2a023aaa1890e8b9dc25ef7aae0fde06414195dc0127b`

## Decision

- Status: pending.
- Reason: benchmark policy accepts only Mac Studio runs against canonical arity
  `1..7` fixture. This record intentionally stores local timing as
  non-acceptance developer evidence.
- Next iteration gate: run canonical `prepare_pools` benchmark on Mac Studio and
  compare against the 90% threshold before starting Iteration 3 acceptance work.
