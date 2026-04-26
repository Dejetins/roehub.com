# Backtest Benchmark Iteration 2 - prepare_pools

Iteration 2 implements artifact array mmap loading, `[start, end)` 15m slicing,
signal row extraction, row prefilter, compressed signal segments, and
`prepare_pools` timing.

## Status

- Historical strict-total acceptance: fail.
- Corrected `prepare_pools_core` acceptance: pending Mac Studio rerun.
- Host: `MacStudioDaniil` (`Mac Studio` acceptance host).
- Recorded at UTC: `2026-04-26T19:20:08.800461+00:00`.
- Canonical target:
  `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.

## Decision

- Status: fail.
- Reason: strict service-level `prepare_pools` folded service overhead into the
  notebook-compatible stage and compared non-equivalent scopes.
- Historical pass rule was:
  `canonical_notebook_prepare_pools_s / service_prepare_pools_s >= 0.9`
  for every `{arity, risk_mode, direction_mode, backend}` tuple.
- In seconds, this means:
  `service_prepare_pools_s <= canonical_notebook_prepare_pools_s / 0.9`.
- Historical strict-total result: `0 / 28` tuples pass.
- Corrected pass rule:
  `canonical_notebook_prepare_pools_s / prepare_pools_core_s >= 0.9`.

## Target

The target is relative to the canonical notebook `prepare_pools` timer for each
tuple, not one fixed wall-clock number. The service may be at most `1.111111x`
slower than the notebook timer. After the stage-contract fix this target applies
only to `prepare_pools_core`, not to `prepare_pools_total`.

| Metric | Min s | Median s | Max s |
|---|---:|---:|---:|
| Canonical notebook `prepare_pools` | `0.0020979999999326537` | `0.006444124999688938` | `0.013242167000498739` |
| Accepted service target | `0.002331111111036282` | `0.007160138888543265` | `0.014713518889443044` |
| Actual service `prepare_pools` | `0.06990979200054426` | `0.0817651039978955` | `0.1018253750007716` |

Historical strict service-level `prepare_pools` was `6.019x` to `32.050x`
slower than the maximum accepted service time, with median `11.365x` over target.

## Corrected Boundary

| Stage | Classification | Compared to canonical notebook `prepare_pools` |
|---|---|---:|
| `artifact_context_resolve` | service overhead | no |
| `artifact_array_open` | service overhead | no |
| `request_slice_prepare` | service overhead | no |
| `prepare_pools_core` | notebook-compatible | yes |
| `prepare_pools_total` | aggregate service telemetry | no |

Compatibility aliases remain in evidence:
`artifact_manifest_load -> artifact_context_resolve`,
`artifact_array_mmap_load -> artifact_array_open`, and
`time_range_slice -> request_slice_prepare`.

## Summary

| Metric | Value |
|---|---:|
| Min speed ratio | `0.028081118022878822` |
| Median speed ratio | `0.07919640686294169` |
| Max speed ratio | `0.14952214378520207` |
| Max service `prepare_pools` s | `0.1018253750007716` |
| Max outer wall s | `0.10319708400493255` |
| Max RSS delta MB | `18.734375` |
| Max peak RSS delta MB | `20.09375` |

## Ratios By Arity

| Arity | Min ratio | Max ratio |
|---:|---:|---:|
| 1 | `0.028081118022878822` | `0.048351645757614435` |
| 2 | `0.037368742453466906` | `0.07423767927072274` |
| 3 | `0.05248315193573622` | `0.06309003942541376` |
| 4 | `0.07648300672370996` | `0.09527915318289835` |
| 5 | `0.08772647861970659` | `0.10806400511678675` |
| 6 | `0.09294153311658732` | `0.1052746847450555` |
| 7 | `0.12412487689152887` | `0.14952214378520207` |

## Identity

- Git branch: `main`.
- Git commit: `853395183ab80eb43019364a76c3d4174119565b`.
- Request fixture hash from canonical baseline:
  `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`.
- Artifact manifest hash:
  `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`.
- Hit-times manifest hash:
  `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`.
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`.
- Artifact slot: `slot_a`.

## Bottleneck Notes

- Median `artifact_manifest_load` was about `0.0668s`, which dominated the
  historical strict service-level timer.
- Median `artifact_array_mmap_load` was about `0.0067s`.
- Diagnostic notebook-core subsegments only
  (`signal_row_selection + row_prefilter + segment_build`) passed `18 / 28`
  tuples, with ratios from `0.8294583811184056` to `1.6686255986909326`.

## Next Gate

- Do not treat Iteration 2 as benchmark-accepted yet.
- Run a corrected Mac Studio benchmark that reports `prepare_pools_core`,
  `artifact_context_resolve`, `artifact_array_open`, `request_slice_prepare`,
  and `prepare_pools_total`.
- Acceptance is based on `prepare_pools_core` only.
