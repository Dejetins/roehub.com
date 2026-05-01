# Iteration 4.5 result shape/hash parity

## Scope

- Compared: top result shape, row ordering, semantic metric parity, proxy metadata parity, strict result hash.
- Not compared: exact scoring, heap update, lazy trades, persistence, public API identity.
- `request.top_n = 100`
- `benchmark_top_k = 5`
- `sample_warmup_top_k = 1`
- `top_results_count_values = [5]`
- `heap_capacity_values = [5]`

## Environment

- Host: `MacStudioDaniil`
- Git commit: `2d42a2f70f8f62ff7137491446fe12fafd3b8ee8`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Artifact hash matches canonical: `False`
- Artifact policy: `historical_prefix_compatible`
- Artifact historical-prefix compatible: `True`

## Results

| arity | direction_mode | backend | top_results | shape | semantic | proxy | strict_hash | waiver | timer_clean | pass |
|---:|---|---|---:|---|---|---|---|---|---|---|
| 1 | `long_only` | `event_segments_1_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 2 | `long_only` | `event_segments_2_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 3 | `long_only` | `event_segments_3_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 4 | `long_only` | `event_segments_4_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 5 | `long_only` | `event_segments_5_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 6 | `long_only` | `event_segments_6_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 7 | `long_only` | `event_segments_7_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | 5 | `yes` | `yes` | `yes` | `yes` | `no` | `yes` | `yes` |

## Decision

- Top identity pass: `14 / 14`
- Shape pass: `14 / 14`
- semantic metric parity: `14 / 14`
- proxy metadata parity: `14 / 14`
- strict result hash: `14 / 14`
- Strict hash or waiver: `14 / 14`
- Measured stage timers unchanged by serialization: `14 / 14`
- Artifact hash matches canonical: `False`
- Artifact historical-prefix compatible: `True`
- Overall pass: `yes`
