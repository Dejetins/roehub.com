# Iteration 4.2 exact scoring self-check benchmark

## Scope

- Compared stages: `self_check`, `exact_scoring`.
- Not compared: `heap_update`, `top_result_proxy_fill`, persistence, public API identity.
- Acceptance ratio threshold: `0.9`.
- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.
- `benchmark_top_k = 5` is telemetry only.

## Environment

- Host: `MacStudioDaniil`
- Git commit: `00db858992846aaa1a8e7eb435a7457369cf4032`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Artifact hash matches canonical: `False`

## Results

| arity | direction_mode | backend | exact_s | canonical_exact_s | exact_ratio | self_check_s | canonical_self_check_s | self_ratio | self_check | pass |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | `long_only` | `event_segments_1_no_risk` | 0.001282 | 0.003169 | 2.472 | 0.091560 | 0.119143 | 1.301 | `passed` | `yes` |
| 2 | `long_only` | `event_segments_2_no_risk` | 0.002480 | 0.003229 | 1.302 | 0.050615 | 0.141234 | 2.790 | `passed` | `yes` |
| 3 | `long_only` | `event_segments_3_no_risk` | 0.040793 | 0.048646 | 1.193 | 0.027269 | 0.142451 | 5.224 | `passed` | `yes` |
| 4 | `long_only` | `event_segments_4_no_risk` | 0.336550 | 0.352216 | 1.047 | 0.029245 | 0.148307 | 5.071 | `passed` | `yes` |
| 5 | `long_only` | `event_segments_5_no_risk` | 2.017287 | 2.090653 | 1.036 | 0.004960 | 0.152132 | 30.675 | `passed` | `yes` |
| 6 | `long_only` | `event_segments_6_no_risk` | 15.514992 | 15.736088 | 1.014 | 0.005469 | 0.151057 | 27.621 | `passed` | `yes` |
| 7 | `long_only` | `event_segments_7_no_risk` | 136.679779 | 139.585680 | 1.021 | 0.008617 | 0.156421 | 18.152 | `passed` | `yes` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | 0.001937 | 0.002300 | 1.188 | 0.152524 | 0.153056 | 1.003 | `passed` | `yes` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | 0.001105 | 0.001013 | 0.916 | 0.034879 | 0.141189 | 4.048 | `passed` | `yes` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | 0.034084 | 0.035367 | 1.038 | 0.033012 | 0.153257 | 4.643 | `passed` | `yes` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | 0.287563 | 0.290841 | 1.011 | 0.032440 | 0.163679 | 5.046 | `passed` | `yes` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | 1.984661 | 1.961637 | 0.988 | 0.012815 | 0.163118 | 12.729 | `passed` | `yes` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | 15.118295 | 15.186989 | 1.005 | 0.011282 | 0.164118 | 14.547 | `passed` | `yes` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | 133.921820 | 136.112667 | 1.016 | 0.012655 | 0.167379 | 13.226 | `passed` | `yes` |

## Decision

- Stage pass: `yes`
- Artifact hash matches canonical: `False`
- Overall pass: `no`
