# BTCUSDT 15m Backtest Engine Benchmark

- request_hash: `58c80c1996b71dc585b931b1570bd43f6b32bd4d6890897a0c9b8f62591fe27f`
- artifact_manifest_hash: `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`
- hit_times_manifest_hash: `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`
- rows_per_indicator: `2`
- TP/SL grid: `2.0..25.0` inclusive, step `0.5`

| arity | risk_mode | backend | combos | exact_s | total_s | peak_rss_mb | cpu_time_s | top_return_pct | self_check |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | none | `event_segments_1_no_risk` | 2 | 0.002355 | 0.130446 | 470.9 | 0.137 | -100.000000 | pass |
| 1 | tp_sl_grid | `event_segments_1_tp_sl_15m_grid` | 2 | 0.005417 | 1.871460 | 695.4 | 1.884 | -100.000000 | pass |
| 2 | none | `event_segments_2_no_risk` | 4 | 0.000372 | 0.118036 | 722.8 | 0.122 | 447.927825 | pass |
| 2 | tp_sl_grid | `event_segments_2_tp_sl_15m_grid` | 4 | 0.001666 | 0.116107 | 729.1 | 0.124 | 15.053810 | pass |
| 3 | none | `event_segments_3_no_risk` | 8 | 0.008145 | 0.145739 | 730.9 | 0.205 | 443.667940 | pass |
| 3 | tp_sl_grid | `event_segments_3_tp_sl_15m_grid` | 8 | 0.008573 | 0.135229 | 735.1 | 0.198 | 10.118418 | pass |
| 4 | none | `event_segments_4_no_risk` | 16 | 0.018442 | 0.170294 | 738.1 | 0.354 | 457.475265 | pass |
| 4 | tp_sl_grid | `event_segments_4_tp_sl_15m_grid` | 16 | 0.015638 | 0.155236 | 743.2 | 0.297 | 19.057530 | pass |
| 5 | none | `event_segments_5_no_risk` | 32 | 0.022723 | 0.177121 | 744.7 | 0.380 | 544.703020 | pass |
| 5 | tp_sl_grid | `event_segments_5_tp_sl_15m_grid` | 32 | 0.020751 | 0.158345 | 762.4 | 0.326 | 58.862078 | pass |
| 6 | none | `event_segments_6_no_risk` | 64 | 0.027722 | 0.184210 | 762.8 | 0.441 | 528.029774 | pass |
| 6 | tp_sl_grid | `event_segments_6_tp_sl_15m_grid` | 64 | 0.027161 | 0.168265 | 780.0 | 0.433 | 54.441547 | pass |
| 7 | none | `event_segments_7_no_risk` | 128 | 0.068946 | 0.230286 | 780.5 | 0.959 | 528.029774 | pass |
| 7 | tp_sl_grid | `event_segments_7_tp_sl_15m_grid` | 128 | 0.070963 | 0.220442 | 793.5 | 0.972 | 54.441547 | pass |

## Sizing Smoke

| sizing | profit_lock | trade_count | total_return_pct | compiled_parity |
|---|---:|---:|---:|---|
| `all_in` | false | 228 | 136.295115 | pass |
| `all_in` | true | 228 | 325.154992 | pass |
| `fixed_quote` | false | 228 | 6.168308 | pass |
| `fixed_quote` | true | 228 | 6.168308 | pass |
| `fixed_equity_pct` | false | 228 | 55.630123 | reference-only |
| `fixed_equity_pct` | true | 228 | 55.630123 | reference-only |
| `fixed_equity_pct_min_quote` | false | 228 | 55.630123 | reference-only |
| `fixed_equity_pct_min_quote` | true | 228 | 55.630123 | reference-only |
| `fixed_equity_pct_max_quote` | false | 228 | 30.841540 | reference-only |
| `fixed_equity_pct_max_quote` | true | 228 | 30.841540 | reference-only |
