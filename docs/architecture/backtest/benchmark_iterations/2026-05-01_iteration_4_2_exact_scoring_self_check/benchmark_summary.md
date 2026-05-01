# Iteration 4.2 exact scoring self-check benchmark

## Scope

- Compared stages: `self_check`, `exact_scoring`.
- Not compared: `heap_update`, `top_result_proxy_fill`, persistence, public API identity.
- Acceptance ratio threshold: `0.9`.
- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.
- `benchmark_top_k = 5` is telemetry only.

## Environment

- Host: `MacStudioDaniil`
- Git commit: `8f3ad2c93e7177bfc9fc4adaadac5b9eb284e1e7`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Artifact hash matches canonical: `False`
- Artifact policy: `historical_prefix_compatible`
- Artifact historical-prefix compatible: `True`
- Artifact compatibility evidence: `canonical request-slice row identity and 15m length matched for all 14 exact/self-check runs`
- Observed trade_T length values: `[218913]`
- Observed eval_T length values: `[218912]`

## Results

| arity | direction_mode | backend | exact_s | canonical_exact_s | exact_ratio | self_check_s | canonical_self_check_s | self_ratio | self_check | pass |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `long_only` | `event_segments_1_no_risk` | 0.003398 | 0.003169 | 0.933 | 0.094760 | 0.119143 | 1.257 | `passed` | `yes` |
| 2 | `long_only` | `event_segments_2_no_risk` | 0.002845 | 0.003229 | 1.135 | 0.052374 | 0.141234 | 2.697 | `passed` | `yes` |
| 3 | `long_only` | `event_segments_3_no_risk` | 0.042255 | 0.048646 | 1.151 | 0.028018 | 0.142451 | 5.084 | `passed` | `yes` |
| 4 | `long_only` | `event_segments_4_no_risk` | 0.347277 | 0.352216 | 1.014 | 0.030720 | 0.148307 | 4.828 | `passed` | `yes` |
| 5 | `long_only` | `event_segments_5_no_risk` | 2.066141 | 2.090653 | 1.012 | 0.004953 | 0.152132 | 30.717 | `passed` | `yes` |
| 6 | `long_only` | `event_segments_6_no_risk` | 15.605805 | 15.736088 | 1.008 | 0.005897 | 0.151057 | 25.615 | `passed` | `yes` |
| 7 | `long_only` | `event_segments_7_no_risk` | 138.269624 | 139.585680 | 1.010 | 0.008948 | 0.156421 | 17.480 | `passed` | `yes` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | 0.001939 | 0.002300 | 1.186 | 0.152519 | 0.153056 | 1.004 | `passed` | `yes` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | 0.000998 | 0.001013 | 1.014 | 0.035099 | 0.141189 | 4.023 | `passed` | `yes` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | 0.038153 | 0.035367 | 0.927 | 0.033214 | 0.153257 | 4.614 | `passed` | `yes` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | 0.291821 | 0.290841 | 0.997 | 0.033385 | 0.163679 | 4.903 | `passed` | `yes` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | 1.990453 | 1.961637 | 0.986 | 0.012907 | 0.163118 | 12.638 | `passed` | `yes` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | 15.304793 | 15.186989 | 0.992 | 0.009214 | 0.164118 | 17.812 | `passed` | `yes` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | 135.821834 | 136.112667 | 1.002 | 0.013052 | 0.167379 | 12.824 | `passed` | `yes` |

## Decision

- Stage pass: `yes`
- Request-slice identity pass: `yes`
- Artifact hash matches canonical: `False`
- Artifact historical-prefix compatible: `True`
- Overall pass: `yes`
