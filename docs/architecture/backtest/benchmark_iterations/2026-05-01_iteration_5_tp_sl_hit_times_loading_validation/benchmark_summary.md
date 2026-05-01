# Iteration 5 TP/SL hit-times loading and grid validation

## Scope

- Compared stages: `load_hit_times`, `tp_sl_grid_validation`.
- Not implemented: `tp_sl_exact_scoring`, heap/top-K, persistence, public API.
- Runtime target path: `hit_times/15m`.
- Target grid: `2.0..25.0` inclusive, `step 0.5`.
- Acceptance ratio threshold: `0.9`.
- Ratio definition: `canonical_stage_seconds / service_stage_seconds`.

## Environment

- Host: `MacStudioDaniil`
- Git commit: `ae8366b3558e812ccbda8b7cef8752710078cc6f`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Hit-times manifest hash: `09143b055123190d6c919ca758817fa89915245d70719437f037340cdc4196fc`
- Artifact hash matches canonical: `False`
- Hit-times hash matches canonical: `False`
- Artifact policy: `historical_prefix_compatible`
- Artifact historical-prefix compatible: `True`

## Results

| arity | direction_mode | load_hit_times_s | canonical_load_s | load_ratio | tp_sl_grid_validation_s | canonical_validation_s | validation_ratio | pass |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `long_only` | 0.032736 | 0.030224 | 0.923 | 0.001587 | 0.002698 | 1.700 | `yes` |
| 2 | `long_only` | 0.032909 | 0.030224 | 0.918 | 0.001506 | 0.002698 | 1.792 | `yes` |
| 3 | `long_only` | 0.032880 | 0.030224 | 0.919 | 0.001491 | 0.002698 | 1.809 | `yes` |
| 4 | `long_only` | 0.032950 | 0.030224 | 0.917 | 0.001574 | 0.002698 | 1.713 | `yes` |
| 5 | `long_only` | 0.032955 | 0.030224 | 0.917 | 0.001483 | 0.002698 | 1.819 | `yes` |
| 6 | `long_only` | 0.032722 | 0.030224 | 0.924 | 0.001499 | 0.002698 | 1.800 | `yes` |
| 7 | `long_only` | 0.032368 | 0.030224 | 0.934 | 0.001492 | 0.002698 | 1.808 | `yes` |
| 1 | `long_short_reversal` | 0.032926 | 0.030224 | 0.918 | 0.001494 | 0.002698 | 1.805 | `yes` |
| 2 | `long_short_reversal` | 0.032417 | 0.030224 | 0.932 | 0.001488 | 0.002698 | 1.812 | `yes` |
| 3 | `long_short_reversal` | 0.032611 | 0.030224 | 0.927 | 0.001558 | 0.002698 | 1.732 | `yes` |
| 4 | `long_short_reversal` | 0.032771 | 0.030224 | 0.922 | 0.001477 | 0.002698 | 1.827 | `yes` |
| 5 | `long_short_reversal` | 0.032965 | 0.030224 | 0.917 | 0.001491 | 0.002698 | 1.809 | `yes` |
| 6 | `long_short_reversal` | 0.032854 | 0.030224 | 0.920 | 0.001497 | 0.002698 | 1.802 | `yes` |
| 7 | `long_short_reversal` | 0.033262 | 0.030224 | 0.909 | 0.001521 | 0.002698 | 1.773 | `yes` |

## Grid And Failure Evidence

- Target grid covered by artifact: `yes`
- Missing-level failure code: `backtest.tp_sl_grid_not_covered`
- Failed-load cleanup code: `backtest.artifacts_unavailable`
- Failure evidence pass: `yes`

## Decision

- `load_hit_times` pass: `yes`
- `tp_sl_grid_validation` pass: `yes`
- Stage pass: `yes`
- Artifact historical-prefix compatible: `True`
- Overall pass: `yes`
