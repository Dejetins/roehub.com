# Iteration 4.3 heap_update benchmark

## Scope

- Compared stage: `heap_update`.
- Service-only timer recorded outside comparison: `top_result_assembly`.
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
- Git commit: `8a3cd7cad0e1856de64b9ae1b58a6e54953135bd`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Artifact hash matches canonical: `False`
- Artifact policy: `historical_prefix_compatible`
- Artifact historical-prefix compatible: `True`
- Artifact compatibility evidence: `canonical request-slice top identity matched for all 14 heap runs`

## Results

| arity | direction_mode | backend | heap_s | top_result_assembly_s | canonical_heap_s | heap_ratio | top_results | identity | pass |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | `long_only` | `event_segments_1_no_risk` | 0.000021 | 0.000047 | 0.000023 | 1.115 | 5 | `yes` | `yes` |
| 2 | `long_only` | `event_segments_2_no_risk` | 0.000055 | 0.000040 | 0.000083 | 1.502 | 5 | `yes` | `yes` |
| 3 | `long_only` | `event_segments_3_no_risk` | 0.000239 | 0.000047 | 0.000435 | 1.817 | 5 | `yes` | `yes` |
| 4 | `long_only` | `event_segments_4_no_risk` | 0.001462 | 0.000061 | 0.002742 | 1.876 | 5 | `yes` | `yes` |
| 5 | `long_only` | `event_segments_5_no_risk` | 0.011546 | 0.000069 | 0.019045 | 1.649 | 5 | `yes` | `yes` |
| 6 | `long_only` | `event_segments_6_no_risk` | 0.075655 | 0.000077 | 0.123256 | 1.629 | 5 | `yes` | `yes` |
| 7 | `long_only` | `event_segments_7_no_risk` | 0.490238 | 0.000089 | 0.804793 | 1.642 | 5 | `yes` | `yes` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | 0.000020 | 0.000048 | 0.000036 | 1.862 | 5 | `yes` | `yes` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | 0.000059 | 0.000045 | 0.000079 | 1.323 | 5 | `yes` | `yes` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | 0.000200 | 0.000052 | 0.000476 | 2.382 | 5 | `yes` | `yes` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | 0.001217 | 0.000058 | 0.002959 | 2.432 | 5 | `yes` | `yes` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | 0.007886 | 0.000082 | 0.018599 | 2.359 | 5 | `yes` | `yes` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | 0.053633 | 0.000079 | 0.122326 | 2.281 | 5 | `yes` | `yes` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | 0.359076 | 0.000123 | 0.784161 | 2.184 | 5 | `yes` | `yes` |

## Decision

- Stage pass: `yes`
- Top identity pass: `yes`
- Artifact hash matches canonical: `False`
- Artifact historical-prefix compatible: `True`
- Overall pass: `yes`
