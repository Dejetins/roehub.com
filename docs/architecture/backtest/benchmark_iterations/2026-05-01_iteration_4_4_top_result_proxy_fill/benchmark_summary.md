# Iteration 4.4 top_result_proxy_fill benchmark

## Scope

- Compared stage: `top_result_proxy_fill`.
- Not compared: exact scoring, heap update, lazy trades, persistence, public API identity.
- Acceptance ratio threshold: `0.9`.
- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.
- `request.top_n = 100`
- `benchmark_top_k = 5`
- `sample_warmup_top_k = 1`
- `top_results_count_values = [5]`
- `heap_capacity_values = [5]`

## Environment

- Host: `MacStudioDaniil`
- Git commit: `4f390bd59fc4121b2c5edebd1accfc7f62c83491`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Artifact hash matches canonical: `False`
- Artifact policy: `historical_prefix_compatible`
- Artifact historical-prefix compatible: `True`
- Artifact compatibility evidence: `canonical request-slice top identity and proxy metadata matched for all 14 proxy-fill runs`

## Results

| arity | direction_mode | backend | proxy_fill_s | canonical_proxy_fill_s | proxy_fill_ratio | top_results | identity | proxy_metadata | pass |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| 1 | `long_only` | `event_segments_1_no_risk` | 0.007797 | 0.007521 | 0.965 | 5 | `yes` | `yes` | `yes` |
| 2 | `long_only` | `event_segments_2_no_risk` | 0.002573 | 0.002586 | 1.005 | 5 | `yes` | `yes` | `yes` |
| 3 | `long_only` | `event_segments_3_no_risk` | 0.010339 | 0.010923 | 1.056 | 5 | `yes` | `yes` | `yes` |
| 4 | `long_only` | `event_segments_4_no_risk` | 0.012179 | 0.011894 | 0.977 | 5 | `yes` | `yes` | `yes` |
| 5 | `long_only` | `event_segments_5_no_risk` | 0.012945 | 0.012689 | 0.980 | 5 | `yes` | `yes` | `yes` |
| 6 | `long_only` | `event_segments_6_no_risk` | 0.014625 | 0.014286 | 0.977 | 5 | `yes` | `yes` | `yes` |
| 7 | `long_only` | `event_segments_7_no_risk` | 0.016551 | 0.016414 | 0.992 | 5 | `yes` | `yes` | `yes` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | 0.007623 | 0.007748 | 1.016 | 5 | `yes` | `yes` | `yes` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | 0.002947 | 0.002827 | 0.960 | 5 | `yes` | `yes` | `yes` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | 0.011834 | 0.010718 | 0.906 | 5 | `yes` | `yes` | `yes` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | 0.014189 | 0.013927 | 0.982 | 5 | `yes` | `yes` | `yes` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | 0.012695 | 0.012212 | 0.962 | 5 | `yes` | `yes` | `yes` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | 0.013213 | 0.013266 | 1.004 | 5 | `yes` | `yes` | `yes` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | 0.014388 | 0.014292 | 0.993 | 5 | `yes` | `yes` | `yes` |

## Decision

- Stage pass: `yes`
- Top identity pass: `yes`
- Proxy metadata pass: `yes`
- Artifact hash matches canonical: `False`
- Artifact historical-prefix compatible: `True`
- Overall pass: `yes`
