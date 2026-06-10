# Iteration 9 lazy trades detail

## Version

- Host: `MacStudioDaniil`
- Git commit: `34fac40074bcf082e86a4396daae3ab6dbdde1a3`
- Artifact manifest hash: `bc9bf0934f903511c9aa259effa27f74ac1e3ccb3a0e5a8525f652729a1cc226`
- Hit-times manifest hash: `56f140eb4ff3b028b7d872d06010a9871eaaf357373947c1230f791966f947b6`
- Artifact policy: `historical_prefix_compatible`
- Overall pass: `yes`

## Service-only Stages

| risk | variant_key | variant_hash | trades | lazy_trades_compute | lazy_trades_cache_hit | parity | pass |
|---|---|---|---:|---:|---:|---|---|
| `none` | `job_34e1e7ca1581__dema_close_w10__risk_none__vh_8f2a6419` | `8f2a641908745dd7241336e3dd08ee8b5d4779e2fdb8c4128973164873347bb8` | 60108 | 2.869005 | 0.000305 | `yes` | `yes` |
| `tp_sl_grid` | `job_7762e8665a88__dema_close_w5__risk_tp_sl_grid__tp_7_5__sl_11__vh_53826f61` | `53826f615006949eca675ae2f69f4b59d7c57bed7d769c0939959e7c9955c30e` | 82783 | 4.334214 | 0.000301 | `yes` | `yes` |

## Cache

- Cache root: `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline/sidecar_artifacts/lazy_trades_cache`
- Cache TTL seconds: `1209600`
- `lazy_trades_compute` and `lazy_trades_cache_hit` are service-only and are not included in `total_without_warmup`.
