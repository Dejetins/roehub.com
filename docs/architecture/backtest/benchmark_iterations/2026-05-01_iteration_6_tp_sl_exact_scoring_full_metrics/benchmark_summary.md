# Iteration 6 TP/SL exact scoring and full metrics

## Scope

- Implemented: `event_segments_n_tp_sl_15m_grid`, TP/SL self-check, `heap_update`, and `tp_sl_full_metrics_second_pass`.
- Not implemented: persistence, public/storage identity, API read models, lazy trades.
- Runtime target path: `hit_times/15m`.
- `benchmark_top_k = 5`; `request.top_n = 100` is recorded separately.
- `tp_sl_full_metrics_second_pass` is service-only and excluded from `total_without_warmup`.

## Environment

- Host: `MacStudioDaniil`
- Git commit: `1672d95ef98dc43bf7bd09504bcf785fd92d22dd`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Hit-times manifest hash: `09143b055123190d6c919ca758817fa89915245d70719437f037340cdc4196fc`
- Artifact hash matches canonical: `False`
- Hit-times hash matches canonical: `False`
- Artifact policy: `historical_prefix_compatible`
- Artifact historical-prefix compatible: `True`

## Failed Rows

| arity | direction_mode | stage | canonical_s | service_s | ratio | reason |
|---:|---|---|---:|---:|---:|---|
| 1 | `long_only` | `build_proxy_context` | 0.000000 | 0.000008 | 0.000 | stage_ratio |
| 1 | `long_only` | `proxy_filter` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 1 | `long_short_reversal` | `build_proxy_context` | 0.000000 | 0.000007 | 0.000 | stage_ratio |
| 1 | `long_short_reversal` | `proxy_filter` | 0.000000 | 0.000006 | 0.000 | stage_ratio |
| 2 | `long_only` | `build_proxy_context` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 2 | `long_only` | `proxy_filter` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 2 | `long_short_reversal` | `build_proxy_context` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 2 | `long_short_reversal` | `proxy_filter` | 0.000000 | 0.000004 | 0.000 | stage_ratio |
| 3 | `long_only` | `build_proxy_context` | 0.000000 | 0.000006 | 0.000 | stage_ratio |
| 3 | `long_only` | `proxy_filter` | 0.000000 | 0.000007 | 0.000 | stage_ratio |
| 3 | `long_short_reversal` | `build_proxy_context` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 3 | `long_short_reversal` | `proxy_filter` | 0.000000 | 0.000004 | 0.000 | stage_ratio |
| 4 | `long_only` | `build_proxy_context` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 4 | `long_only` | `proxy_filter` | 0.000000 | 0.000006 | 0.000 | stage_ratio |
| 4 | `long_short_reversal` | `build_proxy_context` | 0.000000 | 0.000004 | 0.000 | stage_ratio |
| 4 | `long_short_reversal` | `proxy_filter` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 5 | `long_only` | `build_proxy_context` | 0.000000 | 0.000005 | 0.000 | stage_ratio |
| 5 | `long_only` | `proxy_filter` | 0.000000 | 0.000010 | 0.000 | stage_ratio |
| 5 | `long_short_reversal` | `build_proxy_context` | 0.000000 | 0.000006 | 0.000 | stage_ratio |
| 5 | `long_short_reversal` | `proxy_filter` | 0.000000 | 0.000011 | 0.000 | stage_ratio |

## Results

| arity | direction_mode | exact_s | canonical_exact_s | exact_ratio | heap_s | canonical_heap_s | heap_ratio | full_metrics_s | pass |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `long_only` | 0.003510 | 0.003561 | 1.015 | 0.000041 | 0.000019 | 0.477 | 0.628887 | `no` |
| 2 | `long_only` | 0.015603 | 0.020015 | 1.283 | 0.000044 | 0.000034 | 0.777 | 0.087631 | `no` |
| 3 | `long_only` | 0.054965 | 0.056349 | 1.025 | 0.000038 | 0.000144 | 3.833 | 0.088397 | `no` |
| 4 | `long_only` | 0.425468 | 0.428111 | 1.006 | 0.000047 | 0.000753 | 16.155 | 0.090403 | `no` |
| 5 | `long_only` | 2.298283 | 2.299528 | 1.001 | 0.000128 | 0.004514 | 35.141 | 0.092244 | `no` |
| 6 | `long_only` | 17.191681 | 16.822684 | 0.979 | 0.000826 | 0.027209 | 32.926 | 0.093465 | `no` |
| 7 | `long_only` | 148.093383 | 146.899213 | 0.992 | 0.004469 | 0.165521 | 37.034 | 0.094398 | `no` |
| 1 | `long_short_reversal` | 0.008512 | 0.008307 | 0.976 | 0.000036 | 0.000018 | 0.487 | 1.159840 | `no` |
| 2 | `long_short_reversal` | 0.009499 | 0.007455 | 0.785 | 0.000025 | 0.000032 | 1.263 | 0.111054 | `no` |
| 3 | `long_short_reversal` | 0.037265 | 0.038231 | 1.026 | 0.000032 | 0.000143 | 4.526 | 0.107678 | `no` |
| 4 | `long_short_reversal` | 0.312823 | 0.308558 | 0.986 | 0.000039 | 0.000770 | 19.741 | 0.109801 | `no` |
| 5 | `long_short_reversal` | 2.092360 | 2.087266 | 0.998 | 0.000135 | 0.004668 | 34.486 | 0.097981 | `no` |
| 6 | `long_short_reversal` | 16.154290 | 15.795686 | 0.978 | 0.000873 | 0.027475 | 31.456 | 0.142896 | `no` |
| 7 | `long_short_reversal` | 143.468516 | 140.994417 | 0.983 | 0.005166 | 0.165107 | 31.962 | 0.128072 | `no` |

## Smoke 8..10

| arity | direction_mode | candidates | self_check | full_metrics | pass |
|---:|---|---:|---|---|---|
| 8 | `long_only` | 128 | `passed` | `yes` | `yes` |
| 9 | `long_only` | 128 | `passed` | `yes` | `yes` |
| 10 | `long_only` | 128 | `passed` | `yes` | `yes` |
| 8 | `long_short_reversal` | 128 | `passed` | `yes` | `yes` |
| 9 | `long_short_reversal` | 128 | `passed` | `yes` | `yes` |
| 10 | `long_short_reversal` | 128 | `passed` | `yes` | `yes` |

## Decision

- Stage pass: `no`
- Top-result parity pass: `yes`
- Full metrics pass: `yes`
- Cleanup pass: `yes`
- Smoke pass: `yes`
- Overall pass: `no`
