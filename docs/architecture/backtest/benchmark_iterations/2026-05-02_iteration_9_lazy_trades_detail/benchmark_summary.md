# Iteration 9 lazy trades detail

## Version

- Host: `MacStudioDaniil`
- Git commit: `bb50e10fc56c0e7a7a2c5c46400162ab5ae229dd`
- Artifact manifest hash: `6c41e8eb57d8865c2243d15d7d390054bcbbcd601d56c490290168185ef1e554`
- Hit-times manifest hash: `ee5dc838b491a563783becd85cc5107d07b8f2d4c90c3c1903dc31383752da26`
- Artifact policy: `historical_prefix_compatible`
- Overall pass: `yes`

## Service-only Stages

| risk | variant_key | variant_hash | trades | lazy_trades_compute | lazy_trades_cache_hit | parity | pass |
|---|---|---|---:|---:|---:|---|---|
| `none` | `job_fffcb9d534a8__dema_close_w10__risk_none__vh_8f2a6419` | `8f2a641908745dd7241336e3dd08ee8b5d4779e2fdb8c4128973164873347bb8` | 60108 | 4.034278 | 0.563095 | `yes` | `yes` |
| `tp_sl_grid` | `job_4a029574c013__dema_close_w5__risk_tp_sl_grid__tp_7_5__sl_11__vh_53826f61` | `53826f615006949eca675ae2f69f4b59d7c57bed7d769c0939959e7c9955c30e` | 82783 | 5.862834 | 0.751331 | `yes` | `yes` |

## Cache

- Cache root: `/opt/roehub/state/backtest/trades_cache`
- Cache TTL seconds: `172800`
- `lazy_trades_compute` and `lazy_trades_cache_hit` are service-only and are not included in `total_without_warmup`.
