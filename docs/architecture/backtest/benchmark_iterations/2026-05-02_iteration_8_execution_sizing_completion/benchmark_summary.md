# Iteration 8 execution sizing completion

## Version

- Host: `MacStudioDaniil`
- Git commit: `92258b304819d5b8f2c4aebf2a1be4bdab517da3`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Hit-times manifest hash: `09143b055123190d6c919ca758817fa89915245d70719437f037340cdc4196fc`
- Artifact policy: `historical_prefix_compatible`
- Overall pass: `yes`

## Pass Breakdown

- sizing_smoke: `yes`
- tp_sl_sizing_smoke: `yes`
- close_on_end: `yes`
- no_risk_regression: `yes`
- tp_sl_regression: `yes`

## Sizing Smoke

| risk | sizing | profit_lock | service_return_pct | safe_quote | trades | compiled_parity | first compiled parity point | pass |
|---|---|---:|---:|---:|---:|---|---|---|
| `none` | `all_in` | false | 136.295115 | 0.000000 | 228 | pass | no | yes |
| `none` | `all_in` | true | 325.154992 | 28177.432302 | 228 | pass | no | yes |
| `none` | `fixed_quote` | false | 6.168308 | 0.000000 | 228 | pass | no | yes |
| `none` | `fixed_quote` | true | 6.168308 | 233.027874 | 228 | pass | no | yes |
| `none` | `fixed_equity_pct` | false | 55.630123 | 0.000000 | 228 | pass | yes | yes |
| `none` | `fixed_equity_pct` | true | 55.630123 | 2460.418333 | 228 | pass | yes | yes |
| `none` | `fixed_equity_pct_min_quote` | false | 55.630123 | 0.000000 | 228 | pass | yes | yes |
| `none` | `fixed_equity_pct_min_quote` | true | 55.630123 | 2460.418333 | 228 | pass | yes | yes |
| `none` | `fixed_equity_pct_max_quote` | false | 30.841540 | 0.000000 | 228 | pass | yes | yes |
| `none` | `fixed_equity_pct_max_quote` | true | 30.841540 | 1165.139369 | 228 | pass | yes | yes |

## TP/SL Sizing Smoke

| sizing | profit_lock | self_check | top_return_pct | trades | pass |
|---|---:|---|---:|---:|---|
| `all_in` | false | `passed` | 15.053810 | 228 | yes |
| `all_in` | true | `passed` | 10.599158 | 228 | yes |
| `fixed_quote` | false | `passed` | 0.235892 | 228 | yes |
| `fixed_quote` | true | `passed` | 0.235892 | 228 | yes |
| `fixed_equity_pct` | false | `passed` | 2.277603 | 228 | yes |
| `fixed_equity_pct` | true | `passed` | 2.277603 | 228 | yes |
| `fixed_equity_pct_min_quote` | false | `passed` | 2.277603 | 228 | yes |
| `fixed_equity_pct_min_quote` | true | `passed` | 2.277603 | 228 | yes |
| `fixed_equity_pct_max_quote` | false | `passed` | 1.179459 | 228 | yes |
| `fixed_equity_pct_max_quote` | true | `passed` | 1.179459 | 228 | yes |

## Close On End

| risk | close_on_end | return_pct | trades | self_check | pass |
|---|---:|---:|---:|---|---|
| `none` | true | 55.630123 | 228 | `n/a` | yes |
| `none` | false | 65.186325 | 227 | `n/a` | yes |
| `tp_sl_grid` | true | 2.277603 | 228 | `passed` | yes |
| `tp_sl_grid` | false | 2.277603 | 228 | `passed` | yes |

## Regression Envelope

- No-risk runs: `14`
- TP/SL runs: `14`
- No-risk historical stage threshold pass: `no`
- TP/SL historical stage threshold pass: `no`
- Historical threshold policy: `recorded_regression_envelope_not_iteration_8_acceptance_gate_when_non_comparable_zero_or_service_overhead_stages_fail`
