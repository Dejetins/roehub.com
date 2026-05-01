# Iteration 4.3 heap_update benchmark

## Scope

- Compared stage: `heap_update`.
- Not compared: `top_result_proxy_fill`, persistence, public API identity.
- Acceptance ratio threshold: `0.9`.
- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.
- `request.top_n = 100`
- `benchmark_top_k = 5`
- `sample_warmup_top_k = 1`
- `top_results_count_values = [5]`
- `heap_capacity_values = [5]`

## Environment

- Host: `MacStudioDaniil`
- Git commit: `b37ed90c7b90190208ad23d00ef01d8f144d9739`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Artifact hash matches canonical: `False`

## Results

| arity | direction_mode | backend | heap_s | canonical_heap_s | heap_ratio | top_results | identity | pass |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | `long_only` | `event_segments_1_no_risk` | 0.000037 | 0.000023 | 0.638 | 5 | `yes` | `no` |
| 2 | `long_only` | `event_segments_2_no_risk` | 0.000066 | 0.000083 | 1.256 | 5 | `yes` | `yes` |
| 3 | `long_only` | `event_segments_3_no_risk` | 0.000272 | 0.000435 | 1.600 | 5 | `yes` | `yes` |
| 4 | `long_only` | `event_segments_4_no_risk` | 0.001611 | 0.002742 | 1.702 | 5 | `yes` | `yes` |
| 5 | `long_only` | `event_segments_5_no_risk` | 0.013709 | 0.019045 | 1.389 | 5 | `yes` | `yes` |
| 6 | `long_only` | `event_segments_6_no_risk` | 0.089802 | 0.123256 | 1.373 | 5 | `yes` | `yes` |
| 7 | `long_only` | `event_segments_7_no_risk` | 0.584861 | 0.804793 | 1.376 | 5 | `yes` | `yes` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | 0.000034 | 0.000036 | 1.084 | 5 | `yes` | `yes` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | 0.000056 | 0.000079 | 1.402 | 5 | `yes` | `yes` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | 0.000222 | 0.000476 | 2.145 | 5 | `yes` | `yes` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | 0.001308 | 0.002959 | 2.263 | 5 | `yes` | `yes` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | 0.008203 | 0.018599 | 2.267 | 5 | `yes` | `yes` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | 0.055392 | 0.122326 | 2.208 | 5 | `yes` | `yes` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | 0.364800 | 0.784161 | 2.150 | 5 | `yes` | `yes` |

## Decision

- Stage pass: `no`
- Artifact hash matches canonical: `False`
- Overall pass: `no`
