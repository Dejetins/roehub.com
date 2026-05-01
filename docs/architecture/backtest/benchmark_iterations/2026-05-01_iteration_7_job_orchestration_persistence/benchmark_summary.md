# Iteration 7 job orchestration and persistence

## Scope

- Implemented: service-only `top_result_assembly`, summary-only top-N IO measurement, public/storage identity mapping evidence.
- Not implemented: lazy trades detail payloads and Iteration 8 sizing expansion.

## Environment

- Host: `MacStudioDaniil`
- Git commit: `3a3fa3535e9ad0121294b5692ab19368f8f74d03`
- Artifact config: `configs/prod/backtest_artifacts.yaml`
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`
- Artifact manifest hash: `595be8b41c7d50d9d7456f4cc49ec625b7b9f0de557ace7cbdf3c066f091d0af`
- Hit-times manifest hash: `09143b055123190d6c919ca758817fa89915245d70719437f037340cdc4196fc`
- Artifact policy: `historical_prefix_compatible`

## Service-Only Stages

| risk_mode | top_result_assembly_s | persist_top_n_io_s | service_total_without_warmup_s | rows | summary_only | pass |
|---|---:|---:|---:|---:|---|---|
| none | 0.000939 | 0.000099 | 0.093441 | 5 | `True` | `yes` |
| tp_sl_grid | 0.000920 | 0.000109 | 2.131335 | 5 | `True` | `yes` |

## Decision

- Overall pass: `yes`
