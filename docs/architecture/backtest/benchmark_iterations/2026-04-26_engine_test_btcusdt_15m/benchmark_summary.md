# BTCUSDT 15m Backtest Engine Benchmark

## Methodology

- host: `macstudio`
- period_start: `2020-01-11 20:08:00+00:00`
- period_end_exclusive: `2026-04-11 20:08:00+00:00`
- period_semantics: `[start, end)` by 15m `open_time`
- rows_per_indicator: `6`
- warmup_rows_per_indicator: `2`
- arities: `1..7`
- risk modes: `none`, `tp_sl_grid`
- direction modes: `long_only`, `long_short_reversal`
- indicator set: `ma.dema`, `ma.hma`, `ma.ema`, `ma.sma`, `ma.wma`, `ma.rma`, `ma.zlema`
- TP/SL grid: `2.0..25.0` inclusive, step `0.5` (`47 x 47 = 2209` cells)
- warmup policy: sample warmup only, no full dry-run warmup; measured totals exclude warmup
- long_only: `+1` opens/holds long; `0` or `-1` closes long; short trades are not opened
- long_short_reversal: `+1` opens/holds long; `-1` opens/holds short; opposite signal closes and reverses

## Evidence Hashes

- request_hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`
- artifact_manifest_hash: `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`
- hit_times_manifest_hash: `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`

## Run Matrix Summary

| arity | risk_mode | direction_mode | backend | combos | warmup_combos | sample_warmup_s | exact_s | total_s | peak_rss_mb | cpu_time_s | cpu_pct | top_return_pct | self_check |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | none | long_only | `event_segments_1_no_risk` | 6 | 2 | 4.346890 | 0.003169 | 0.132407 | 504.2 | 0.153 | 115.3 | -100.000000 | pass |
| 1 | tp_sl_grid | long_only | `event_segments_1_tp_sl_15m_grid` | 6 | 2 | 20.194462 | 0.003561 | 0.937853 | 1278.8 | 0.961 | 102.4 | -100.000000 | pass |
| 2 | none | long_only | `event_segments_2_no_risk` | 36 | 4 | 3.732236 | 0.003229 | 0.150559 | 1278.8 | 0.182 | 118.7 | 0.000000 | pass |
| 2 | tp_sl_grid | long_only | `event_segments_2_tp_sl_15m_grid` | 36 | 4 | 0.197160 | 0.020015 | 0.605699 | 1278.8 | 0.784 | 129.4 | 0.000000 | pass |
| 3 | none | long_only | `event_segments_3_no_risk` | 216 | 8 | 0.093849 | 0.048646 | 0.208065 | 1278.8 | 0.626 | 298.2 | 0.000000 | pass |
| 3 | tp_sl_grid | long_only | `event_segments_3_tp_sl_15m_grid` | 216 | 8 | 0.196730 | 0.056349 | 0.419376 | 1287.2 | 0.915 | 218.2 | 0.000000 | pass |
| 4 | none | long_only | `event_segments_4_no_risk` | 1296 | 16 | 0.111187 | 0.352216 | 0.524729 | 1288.5 | 3.838 | 729.0 | 0.000000 | pass |
| 4 | tp_sl_grid | long_only | `event_segments_4_tp_sl_15m_grid` | 1296 | 16 | 0.214594 | 0.428111 | 0.801609 | 1303.7 | 4.611 | 575.2 | 0.000000 | pass |
| 5 | none | long_only | `event_segments_5_no_risk` | 7776 | 32 | 0.110745 | 2.090653 | 2.288478 | 1315.5 | 23.120 | 1009.0 | 0.000000 | pass |
| 5 | tp_sl_grid | long_only | `event_segments_5_tp_sl_15m_grid` | 7776 | 32 | 0.103217 | 2.299528 | 2.474264 | 1334.4 | 24.998 | 1010.3 | 4.224572 | pass |
| 6 | none | long_only | `event_segments_6_no_risk` | 46656 | 64 | 0.133640 | 15.736088 | 16.059058 | 1348.5 | 176.859 | 1101.1 | 0.000000 | pass |
| 6 | tp_sl_grid | long_only | `event_segments_6_tp_sl_15m_grid` | 46656 | 64 | 0.119158 | 16.822684 | 17.041356 | 1371.3 | 188.252 | 1104.7 | 8.873285 | pass |
| 7 | none | long_only | `event_segments_7_no_risk` | 279936 | 128 | 0.174498 | 139.585680 | 140.746091 | 1391.2 | 1583.026 | 1124.7 | 0.000000 | pass |
| 7 | tp_sl_grid | long_only | `event_segments_7_tp_sl_15m_grid` | 279936 | 128 | 0.174262 | 146.899213 | 147.415075 | 1430.4 | 1658.546 | 1125.1 | 8.873285 | pass |
| 1 | none | long_short_reversal | `event_segments_1_no_risk` | 6 | 2 | 0.087412 | 0.002300 | 0.166162 | 1430.4 | 0.183 | 110.2 | -100.000000 | pass |
| 1 | tp_sl_grid | long_short_reversal | `event_segments_1_tp_sl_15m_grid` | 6 | 2 | 1.019470 | 0.008307 | 1.956840 | 1430.4 | 2.005 | 102.4 | -100.000000 | pass |
| 2 | none | long_short_reversal | `event_segments_2_no_risk` | 36 | 4 | 0.078483 | 0.001013 | 0.148671 | 1430.4 | 0.165 | 109.3 | 799.522940 | pass |
| 2 | tp_sl_grid | long_short_reversal | `event_segments_2_tp_sl_15m_grid` | 36 | 4 | 0.078452 | 0.007455 | 0.153499 | 1430.4 | 0.218 | 142.1 | 24.812570 | pass |
| 3 | none | long_short_reversal | `event_segments_3_no_risk` | 216 | 8 | 0.091704 | 0.035367 | 0.205869 | 1430.4 | 0.544 | 261.8 | 799.522940 | pass |
| 3 | tp_sl_grid | long_short_reversal | `event_segments_3_tp_sl_15m_grid` | 216 | 8 | 0.091672 | 0.038231 | 0.202960 | 1430.4 | 0.569 | 280.2 | 24.971136 | pass |
| 4 | none | long_short_reversal | `event_segments_4_no_risk` | 1296 | 16 | 0.103341 | 0.290841 | 0.479330 | 1430.4 | 3.308 | 685.6 | 799.522940 | pass |
| 4 | tp_sl_grid | long_short_reversal | `event_segments_4_tp_sl_15m_grid` | 1296 | 16 | 0.106586 | 0.308558 | 0.484671 | 1430.4 | 3.534 | 729.2 | 28.954521 | pass |
| 5 | none | long_short_reversal | `event_segments_5_no_risk` | 7776 | 32 | 0.110009 | 1.961637 | 2.167670 | 1434.7 | 22.340 | 1030.4 | 799.522940 | pass |
| 5 | tp_sl_grid | long_short_reversal | `event_segments_5_tp_sl_15m_grid` | 7776 | 32 | 0.110549 | 2.087266 | 2.269673 | 1443.3 | 23.357 | 1029.1 | 76.623619 | pass |
| 6 | none | long_short_reversal | `event_segments_6_no_risk` | 46656 | 64 | 0.126789 | 15.186989 | 15.523999 | 1443.3 | 172.890 | 1113.5 | 799.522940 | pass |
| 6 | tp_sl_grid | long_short_reversal | `event_segments_6_tp_sl_15m_grid` | 46656 | 64 | 0.124807 | 15.795686 | 16.026725 | 1453.3 | 178.263 | 1112.3 | 73.421431 | pass |
| 7 | none | long_short_reversal | `event_segments_7_no_risk` | 279936 | 128 | 0.174333 | 136.112667 | 137.263877 | 1453.3 | 1549.530 | 1128.9 | 799.522940 | pass |
| 7 | tp_sl_grid | long_short_reversal | `event_segments_7_tp_sl_15m_grid` | 279936 | 128 | 0.174711 | 140.994417 | 141.519415 | 1473.9 | 1601.716 | 1131.8 | 73.421431 | pass |

## Stage Metrics

| arity | risk_mode | direction_mode | prepare_s | build_exact_s | build_proxy_s | combo_iter_s | proxy_filter_s | self_check_s | exact_s | heap_s | proxy_fill_s | load_hit_times_s | tp_sl_validation_s | total_s |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | none | long_only | 0.002102 | 0.000417 | 0.000001 | 0.000012 | 0.000001 | 0.119143 | 0.003169 | 0.000023 | 0.007521 | 0.000000 | 0.000000 | 0.132407 |
| 1 | tp_sl_grid | long_only | 0.002098 | 0.000378 | 0.000000 | 0.000013 | 0.000000 | 0.931589 | 0.003561 | 0.000019 | 0.000000 | 0.030224 | 0.002698 | 0.937853 |
| 2 | none | long_only | 0.003391 | 0.000000 | 0.000001 | 0.000019 | 0.000001 | 0.141234 | 0.003229 | 0.000083 | 0.002586 | 0.000000 | 0.000000 | 0.150559 |
| 2 | tp_sl_grid | long_only | 0.005607 | 0.000460 | 0.000000 | 0.000023 | 0.000000 | 0.579394 | 0.020015 | 0.000034 | 0.000000 | 0.030224 | 0.002698 | 0.605699 |
| 3 | none | long_only | 0.004668 | 0.000833 | 0.000003 | 0.000092 | 0.000001 | 0.142451 | 0.048646 | 0.000435 | 0.010923 | 0.000000 | 0.000000 | 0.208065 |
| 3 | tp_sl_grid | long_only | 0.004797 | 0.000860 | 0.000000 | 0.000096 | 0.000000 | 0.356971 | 0.056349 | 0.000144 | 0.000000 | 0.030224 | 0.002698 | 0.419376 |
| 4 | none | long_only | 0.007772 | 0.001258 | 0.000001 | 0.000524 | 0.000001 | 0.148307 | 0.352216 | 0.002742 | 0.011894 | 0.000000 | 0.000000 | 0.524729 |
| 4 | tp_sl_grid | long_only | 0.006355 | 0.001141 | 0.000000 | 0.000518 | 0.000000 | 0.364601 | 0.428111 | 0.000753 | 0.000000 | 0.030224 | 0.002698 | 0.801609 |
| 5 | none | long_only | 0.009206 | 0.001104 | 0.000001 | 0.003620 | 0.000006 | 0.152132 | 2.090653 | 0.019045 | 0.012689 | 0.000000 | 0.000000 | 2.288478 |
| 5 | tp_sl_grid | long_only | 0.008846 | 0.001168 | 0.000000 | 0.004655 | 0.000000 | 0.155400 | 2.299528 | 0.004514 | 0.000000 | 0.030224 | 0.002698 | 2.474264 |
| 6 | none | long_only | 0.008893 | 0.001375 | 0.000001 | 0.023943 | 0.000053 | 0.151057 | 15.736088 | 0.123256 | 0.014286 | 0.000000 | 0.000000 | 16.059058 |
| 6 | tp_sl_grid | long_only | 0.009741 | 0.001453 | 0.000000 | 0.025738 | 0.000000 | 0.153014 | 16.822684 | 0.027209 | 0.000000 | 0.030224 | 0.002698 | 17.041356 |
| 7 | none | long_only | 0.013242 | 0.001972 | 0.000001 | 0.166747 | 0.000342 | 0.156421 | 139.585680 | 0.804793 | 0.016414 | 0.000000 | 0.000000 | 140.746091 |
| 7 | tp_sl_grid | long_only | 0.013039 | 0.003101 | 0.000000 | 0.167374 | 0.000000 | 0.164225 | 146.899213 | 0.165521 | 0.000000 | 0.030224 | 0.002698 | 147.415075 |
| 1 | none | long_short_reversal | 0.002621 | 0.000373 | 0.000001 | 0.000011 | 0.000001 | 0.153056 | 0.002300 | 0.000036 | 0.007748 | 0.000000 | 0.000000 | 0.166162 |
| 1 | tp_sl_grid | long_short_reversal | 0.003396 | 0.000417 | 0.000000 | 0.000012 | 0.000000 | 1.944529 | 0.008307 | 0.000018 | 0.000000 | 0.030224 | 0.002698 | 1.956840 |
| 2 | none | long_short_reversal | 0.003519 | 0.000000 | 0.000001 | 0.000029 | 0.000001 | 0.141189 | 0.001013 | 0.000079 | 0.002827 | 0.000000 | 0.000000 | 0.148671 |
| 2 | tp_sl_grid | long_short_reversal | 0.003187 | 0.000651 | 0.000000 | 0.000017 | 0.000000 | 0.142071 | 0.007455 | 0.000032 | 0.000000 | 0.030224 | 0.002698 | 0.153499 |
| 3 | none | long_short_reversal | 0.005041 | 0.000915 | 0.000001 | 0.000083 | 0.000001 | 0.153257 | 0.035367 | 0.000476 | 0.010718 | 0.000000 | 0.000000 | 0.205869 |
| 3 | tp_sl_grid | long_short_reversal | 0.004790 | 0.000898 | 0.000000 | 0.000082 | 0.000000 | 0.158685 | 0.038231 | 0.000143 | 0.000000 | 0.030224 | 0.002698 | 0.202960 |
| 4 | none | long_short_reversal | 0.006283 | 0.001115 | 0.000000 | 0.000511 | 0.000001 | 0.163679 | 0.290841 | 0.002959 | 0.013927 | 0.000000 | 0.000000 | 0.479330 |
| 4 | tp_sl_grid | long_short_reversal | 0.006533 | 0.001101 | 0.000000 | 0.000530 | 0.000000 | 0.167087 | 0.308558 | 0.000770 | 0.000000 | 0.030224 | 0.002698 | 0.484671 |
| 5 | none | long_short_reversal | 0.007721 | 0.000831 | 0.000001 | 0.003518 | 0.000007 | 0.163118 | 1.961637 | 0.018599 | 0.012212 | 0.000000 | 0.000000 | 2.167670 |
| 5 | tp_sl_grid | long_short_reversal | 0.007489 | 0.000586 | 0.000000 | 0.004620 | 0.000000 | 0.164901 | 2.087266 | 0.004668 | 0.000000 | 0.030224 | 0.002698 | 2.269673 |
| 6 | none | long_short_reversal | 0.009092 | 0.001630 | 0.000000 | 0.026427 | 0.000059 | 0.164118 | 15.186989 | 0.122326 | 0.013266 | 0.000000 | 0.000000 | 15.523999 |
| 6 | tp_sl_grid | long_short_reversal | 0.008740 | 0.001758 | 0.000000 | 0.027337 | 0.000000 | 0.165314 | 15.795686 | 0.027475 | 0.000000 | 0.030224 | 0.002698 | 16.026725 |
| 7 | none | long_short_reversal | 0.012561 | 0.002770 | 0.000001 | 0.169205 | 0.000362 | 0.167379 | 136.112667 | 0.784161 | 0.014292 | 0.000000 | 0.000000 | 137.263877 |
| 7 | tp_sl_grid | long_short_reversal | 0.011836 | 0.003029 | 0.000000 | 0.170321 | 0.000000 | 0.170980 | 140.994417 | 0.165107 | 0.000000 | 0.030224 | 0.002698 | 141.519415 |

## Correctness And Hashes

| arity | risk_mode | direction_mode | self_check | trade_count_equal | max_return_diff | best_cell_equal | result_hash |
|---:|---|---|---|---|---:|---|---|
| 1 | none | long_only | pass | True | 0 | n/a | `e0434a0b895f7ae354aa0ed7df633379ec69efa9e053b69c4bea1aaaec50dcb3` |
| 1 | tp_sl_grid | long_only | pass | True | 0 | True | `1d76657d31957ba7d38cb5d38daff1538849dfe588f4ad4e3a0c91ba2ed2bb81` |
| 2 | none | long_only | pass | True | 0 | n/a | `2fad97724afb919ec9477e195f58a5f31182d44cd36b0f3d9e1590782b81d485` |
| 2 | tp_sl_grid | long_only | pass | True | 1.15066015605e-08 | True | `c395a85c22f353df469201e1e674b19065f7efd42a989b7f803d33bd640420cf` |
| 3 | none | long_only | pass | True | 1.42108547152e-14 | n/a | `9d8ea2ca3b811abeda33c25c99ba70c57ad0d551d6c8a84799368fb752ecec8d` |
| 3 | tp_sl_grid | long_only | pass | True | 2.73283453645e-08 | True | `c395a85c22f353df469201e1e674b19065f7efd42a989b7f803d33bd640420cf` |
| 4 | none | long_only | pass | True | 0 | n/a | `b8a28bb93737fb3b50ab5c16225f0977b01f5d09155848835400347eeaff785f` |
| 4 | tp_sl_grid | long_only | pass | True | 7.01259417202e-09 | True | `c395a85c22f353df469201e1e674b19065f7efd42a989b7f803d33bd640420cf` |
| 5 | none | long_only | pass | True | 9.23705556488e-14 | n/a | `48f398d98236b4d1b42326e9cae04b792f0918597a7ddc83201a7d7318958e8e` |
| 5 | tp_sl_grid | long_only | pass | True | 3.83113167857e-08 | True | `179fdfda12611156ad755d7fc44db0ac7a79ff984039a925817394b1babeb479` |
| 6 | none | long_only | pass | True | 9.23705556488e-14 | n/a | `a03ed0afb60ceb608e76b2598009606a46aa57dae0afd500d8b8059ed42d91b9` |
| 6 | tp_sl_grid | long_only | pass | True | 3.83113167857e-08 | True | `bcfa46d25a6dcb7deebff566a3d0eeee9744c00a31a9b0f8e62f652a75a33b71` |
| 7 | none | long_only | pass | True | 9.23705556488e-14 | n/a | `489322dd5b8a2c71c656beebc1f46ca45f68942f9c605ece6b43af49e9bb65b4` |
| 7 | tp_sl_grid | long_only | pass | True | 3.83113167857e-08 | True | `bcfa46d25a6dcb7deebff566a3d0eeee9744c00a31a9b0f8e62f652a75a33b71` |
| 1 | none | long_short_reversal | pass | True | 0 | n/a | `268ad29fa2e1a3d8da0b91442d538fe95ed7d150f151306f71013c6e8bdc0249` |
| 1 | tp_sl_grid | long_short_reversal | pass | True | 0 | True | `159441300671251c2181401092aae4b806e742d6608fb571fe9458f7406a8712` |
| 2 | none | long_short_reversal | pass | True | 3.41060513165e-13 | n/a | `856f7aa6a5a42ac32edce2c9cf126b8871a43ec6416c3d5e2496f5cff828e0ad` |
| 2 | tp_sl_grid | long_short_reversal | pass | True | 4.75977933956e-07 | True | `d05c224b892a0aa910387e80a4d4bd2a5d85b19db27131e8f06c6c8b2ef8b5dd` |
| 3 | none | long_short_reversal | pass | True | 8.52651282912e-13 | n/a | `973ae53048345c7a86a5539b63c661379c86278732ffff4df1d91bb4e0ca613e` |
| 3 | tp_sl_grid | long_short_reversal | pass | True | 5.87255492013e-07 | True | `7fd5d4155f3bfffd270f0299710c2e64afa124f76badefea26417855338c4b5c` |
| 4 | none | long_short_reversal | pass | True | 1.02318153949e-12 | n/a | `b9d61cb0bae9ac87f15fad6b06cb84c2ad445b7ae3f3ad40e0f26c97028e1d92` |
| 4 | tp_sl_grid | long_short_reversal | pass | True | 4.35901890139e-07 | True | `f3ca32229a267da26660f78db3bc51f4ff6d846b19aaa5bc85b7b09b5dbc5736` |
| 5 | none | long_short_reversal | pass | True | 2.13162820728e-14 | n/a | `1640b5685d457229d5feedd45f09c5fca7cf905066a45625bf76c9f934a4be03` |
| 5 | tp_sl_grid | long_short_reversal | pass | True | 4.57533126941e-08 | True | `77b6e0eca3d885b2113dca803b0a04d0c3a186e7a0e45c84adea0547bd2cc5b4` |
| 6 | none | long_short_reversal | pass | True | 1.56319401867e-13 | n/a | `6c64c6945a1ab995d77bb2e1dd6a0c66c52d407d5eec439b5bc5cda2a5fd7e4e` |
| 6 | tp_sl_grid | long_short_reversal | pass | True | 1.1338608874e-07 | True | `82684222fc06530d10cdbe904e9c58d92eb5ce9f6b54d54630363f4d6997a824` |
| 7 | none | long_short_reversal | pass | True | 0 | n/a | `5fbb8fb598e4c34e8d73de296e26e641210152de9518055775228dce0303c6c8` |
| 7 | tp_sl_grid | long_short_reversal | pass | True | 1.1338608874e-07 | True | `82684222fc06530d10cdbe904e9c58d92eb5ce9f6b54d54630363f4d6997a824` |

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
