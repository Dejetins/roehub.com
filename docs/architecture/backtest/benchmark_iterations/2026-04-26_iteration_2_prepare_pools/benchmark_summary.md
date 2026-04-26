# Backtest Benchmark Iteration 2 - prepare_pools

Iteration 2 implements artifact array mmap loading, `[start, end)` 15m slicing,
signal row extraction, row prefilter, compressed signal segments, and
prepare-pools timing. This record preserves the historical strict-total
failure and the corrected notebook-compatible Mac Studio benchmark.

## Decision

- Current acceptance status: `pass`.
- Accepted comparison stage: `prepare_pools_core`.
- Historical strict-total status: `fail`; it compared non-equivalent scopes.
- Corrected pass count: `28 / 28`.
- Corrected benchmark recorded at UTC: `2026-04-26T20:15:56.159765+00:00`.
- Rule: `canonical_notebook_prepare_pools_s / prepare_pools_core_s >= 0.9`.
- `prepare_pools_total` is aggregate service telemetry and is not compared to the notebook target.

## Stage Contract

| Stage | Classification | Compared to canonical notebook `prepare_pools` |
|---|---|---:|
| `artifact_context_resolve` | service overhead | no |
| `artifact_array_open` | service overhead | no |
| `request_slice_prepare` | service overhead | no |
| `prepare_pools_core` | notebook-compatible compute | yes |
| `prepare_pools_total` | aggregate service telemetry | no |

Compatibility aliases remain in the JSON for historical continuity:
`artifact_manifest_load -> artifact_context_resolve`,
`artifact_array_mmap_load -> artifact_array_open`, and
`time_range_slice -> request_slice_prepare`.

## Identity

- Acceptance host: `Mac Studio`.
- Hostname: `MacStudioDaniil`.
- Platform: `macOS-15.7.5-arm64-arm-64bit`.
- Git branch: `codex/backtest-stage-contract-remediation`.
- Git commit used for corrected benchmark: `4b6ad0d01f5f6b04dc8360cc7b60da1e8f5e5c7b`.
- Baseline path: `docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json`.
- Baseline request hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`.
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`.
- Artifact slot: `slot_a` generation `3`.
- Artifact manifest hash: `a76ccba27c8fabb3d5a6ad14c7d8f121839a5e22c107d038223261159367b259`.
- Hit-times manifest hash: `2366cc2f5a44ccc7faf716ed65a4f37bcbb91150471eec177d7f633a615dbaba`.
- Artifact published at UTC: `2026-04-26T01:40:23Z`.

## Target

The target is relative per tuple. The accepted service time is
`canonical_notebook_prepare_pools_s / 0.9`, so the service may be at most
`1.111111x` slower than the canonical notebook timer for `prepare_pools_core`.

| Metric | Min s | Median s | Max s |
|---|---:|---:|---:|
| Canonical notebook `prepare_pools` | `0.00209799999993` | `0.00644412499969` | `0.0132421670005` |
| Accepted `prepare_pools_core` target | `0.00233111111104` | `0.00716013888854` | `0.0147135188894` |
| Actual `prepare_pools_core` | `0.00176162499702` | `0.0059315000035` | `0.0103970830023` |

## Corrected Summary

| Metric | Value |
|---|---:|
| Status | `pass` |
| Pass count | `28 / 28` |
| Min speed ratio baseline/core | `1.01146943624` |
| Median speed ratio baseline/core | `1.16900637902` |
| Max speed ratio baseline/core | `1.92752884759` |
| Median core over accepted target | `0.770042932872` |
| Max core over accepted target | `0.889794558054` |
| Max `prepare_pools_core` s | `0.0103970830023` |
| Max `prepare_pools_total` s | `0.0996565829992` |
| Max `artifact_context_resolve` s | `0.0810297499993` |
| Max `artifact_array_open` s | `0.0135704999993` |
| Max `request_slice_prepare` s | `0.00115112500498` |
| Max RSS delta MB | `19.078125` |
| Max peak RSS delta MB | `20.4375` |

## Corrected Stage Metrics

| Stage/Metric | Min | Median | Max |
|---|---:|---:|---:|
| `prepare_pools_core` s | `0.00176162499702` | `0.0059315000035` | `0.0103970830023` |
| `prepare_pools_total` s | `0.0697366250024` | `0.079406541503` | `0.0996565829992` |
| `artifact_context_resolve` s | `0.0648297920052` | `0.0654224169957` | `0.0810297499993` |
| `artifact_array_open` s | `0.00200012500136` | `0.00596283350023` | `0.0135704999993` |
| `request_slice_prepare` s | `0.000624167005299` | `0.000733395994757` | `0.00115112500498` |
| outer wall s | `0.0712386669984` | `0.0803389795001` | `0.100671209002` |
| CPU time delta s | `0.075427696` | `0.094871236` | `0.124681552` |
| process CPU percent sample | `101.8` | `117.25` | `127.4` |
| RSS delta MB | `0` | `4.984375` | `19.078125` |
| Peak RSS delta MB | `0` | `5.1796875` | `20.4375` |
| speed ratio baseline/core | `1.01146943624` | `1.16900637902` | `1.92752884759` |
| core over accepted target | `0.466919081977` | `0.770042932872` | `0.889794558054` |

## Sample Warmup

| Metric | Min s | Median s | Max s |
|---|---:|---:|---:|
| sample warmup wall | `0.0696575830007` | `0.0789072085026` | `0.144370124995` |
| sample warmup core | `0.00120037499437` | `0.00451035450169` | `0.0765573329991` |

## Corrected By Arity

| Arity | Pass | Min ratio | Median ratio | Max ratio | Max core s | Max total s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 / 4 | `1.01146943624` | `1.26043527593` | `1.92752884759` | `0.00207795799361` | `0.0860639169987` |
| 2 | 4 / 4 | `1.18576950246` | `1.23350474737` | `1.89734258875` | `0.00295520800137` | `0.077163957998` |
| 3 | 4 / 4 | `1.0223882761` | `1.0787487309` | `1.19620761968` | `0.00456529100484` | `0.0879466249971` |
| 4 | 4 / 4 | `1.02422820797` | `1.12217020073` | `1.26338667993` | `0.00620491600421` | `0.0868589169986` |
| 5 | 4 / 4 | `1.10222308836` | `1.22865773768` | `1.32953854369` | `0.00692458300182` | `0.079522292006` |
| 6 | 4 / 4 | `1.04353845667` | `1.1057895172` | `1.15224325559` | `0.0085221670015` | `0.0847466249979` |
| 7 | 4 / 4 | `1.18886560474` | `1.26897890555` | `1.27364252046` | `0.0103970830023` | `0.0996565829992` |

## Corrected Tuple Matrix

| Arity | Risk | Direction | Backend | Baseline s | Target core s | Core s | Ratio | Core/target | Total s | Ctx s | Open s | Slice s | Pass |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `none` | `long_only` | `event_segments_1_no_risk` | `0.00210179100031` | `0.00233532333368` | `0.00207795799361` | `1.01146943624` | `0.889794558054` | `0.0741832080021` | `0.0685776669998` | `0.00232266599778` | `0.00115112500498` | `yes` |
| 1 | `tp_sl_grid` | `long_only` | `event_segments_1_tp_sl_15m_grid` | `0.00209799999993` | `0.00233111111104` | `0.00202400000126` | `1.03656126415` | `0.868255481978` | `0.0839519159999` | `0.078607375006` | `0.00222575000225` | `0.00105199999962` | `yes` |
| 2 | `none` | `long_only` | `event_segments_2_no_risk` | `0.00339095799973` | `0.00376773111081` | `0.00284541699511` | `1.19172620588` | `0.755207022856` | `0.077163957998` | `0.0653357499978` | `0.00825804199849` | `0.000668458000291` | `yes` |
| 2 | `tp_sl_grid` | `long_only` | `event_segments_2_tp_sl_15m_grid` | `0.00560704199961` | `0.00623004666623` | `0.00295520800137` | `1.89734258875` | `0.474347650939` | `0.0717777909958` | `0.0650524580051` | `0.00310425000498` | `0.000624167005299` | `yes` |
| 3 | `none` | `long_only` | `event_segments_3_no_risk` | `0.00466750000032` | `0.00518611111147` | `0.00456529100484` | `1.0223882761` | `0.880291784482` | `0.0879466249971` | `0.0779467910033` | `0.00443999999698` | `0.000869707997481` | `yes` |
| 3 | `tp_sl_grid` | `long_only` | `event_segments_3_tp_sl_15m_grid` | `0.00479687500047` | `0.00532986111163` | `0.0044732919996` | `1.07233665964` | `0.839288661733` | `0.0747118750005` | `0.0651106250007` | `0.00430720800068` | `0.000708040999598` | `yes` |
| 4 | `none` | `long_only` | `event_segments_4_no_risk` | `0.00777225000002` | `0.00863583333335` | `0.0061519170049` | `1.26338667993` | `0.712370974222` | `0.0868589169986` | `0.0692638750043` | `0.0103115419988` | `0.00103362500522` | `yes` |
| 4 | `tp_sl_grid` | `long_only` | `event_segments_4_tp_sl_15m_grid` | `0.00635524999961` | `0.00706138888846` | `0.00620491600421` | `1.02422820797` | `0.878710421168` | `0.0799980000011` | `0.067124125002` | `0.00554062500305` | `0.00099791699904` | `yes` |
| 5 | `none` | `long_only` | `event_segments_5_no_risk` | `0.00920649999989` | `0.0102294444443` | `0.00692458300182` | `1.32953854369` | `0.676926595526` | `0.0791908749961` | `0.0650144169995` | `0.00644879200263` | `0.000691917004588` | `yes` |
| 5 | `tp_sl_grid` | `long_only` | `event_segments_5_tp_sl_15m_grid` | `0.00884624999981` | `0.00982916666645` | `0.00676070799818` | `1.30847982226` | `0.687821076558` | `0.0787310000014` | `0.0648297920052` | `0.00638504199742` | `0.000635874996078` | `yes` |
| 6 | `none` | `long_only` | `event_segments_6_no_risk` | `0.00889320900023` | `0.00988134333359` | `0.0085221670015` | `1.04353845667` | `0.862450247278` | `0.0847466249979` | `0.0672032079965` | `0.00785995800106` | `0.00103220900201` | `yes` |
| 6 | `tp_sl_grid` | `long_only` | `event_segments_6_tp_sl_15m_grid` | `0.00974058400061` | `0.0108228711118` | `0.00845358300285` | `1.15224325559` | `0.781085066572` | `0.0831544579996` | `0.0657812920035` | `0.00811108399648` | `0.000676833005855` | `yes` |
| 7 | `none` | `long_only` | `event_segments_7_no_risk` | `0.0132421670005` | `0.0147135188894` | `0.0103970830023` | `1.27364252046` | `0.706634699723` | `0.0996565829992` | `0.0790337920043` | `0.00913116699667` | `0.000923250001506` | `yes` |
| 7 | `tp_sl_grid` | `long_only` | `event_segments_7_tp_sl_15m_grid` | `0.013039083` | `0.01448787` | `0.0102769580044` | `1.26876873433` | `0.709349131682` | `0.0896501669995` | `0.0650460409961` | `0.0135704999993` | `0.000640792000922` | `yes` |
| 1 | `none` | `long_short_reversal` | `event_segments_1_no_risk` | `0.00262116700014` | `0.00291240777794` | `0.0017659169971` | `1.48430928772` | `0.606342631852` | `0.0697366250024` | `0.0652312079983` | `0.00200012500136` | `0.000700582997524` | `yes` |
| 1 | `tp_sl_grid` | `long_short_reversal` | `event_segments_1_tp_sl_15m_grid` | `0.00339558300038` | `0.00377287000043` | `0.00176162499702` | `1.92752884759` | `0.466919081977` | `0.0860639169987` | `0.0810297499993` | `0.00218937500176` | `0.00103825000406` | `yes` |
| 2 | `none` | `long_short_reversal` | `event_segments_2_no_risk` | `0.00351866599976` | `0.00390962888863` | `0.00275912499637` | `1.27528328886` | `0.705725549655` | `0.0741865000018` | `0.0673528750049` | `0.00319570799911` | `0.000832624995383` | `yes` |
| 2 | `tp_sl_grid` | `long_short_reversal` | `event_segments_2_tp_sl_15m_grid` | `0.00318724999943` | `0.00354138888825` | `0.00268791699636` | `1.18576950246` | `0.759000799173` | `0.0721860420017` | `0.0653711249979` | `0.00308791700081` | `0.000663791994157` | `yes` |
| 3 | `none` | `long_short_reversal` | `event_segments_3_no_risk` | `0.00504116699994` | `0.00560129666661` | `0.00421429099515` | `1.19620761968` | `0.752377752151` | `0.0751827499989` | `0.0656814999966` | `0.00436666699534` | `0.00076004100265` | `yes` |
| 3 | `tp_sl_grid` | `long_short_reversal` | `event_segments_3_tp_sl_15m_grid` | `0.00478958399981` | `0.00532175999979` | `0.0044137090008` | `1.08516080216` | `0.829370170954` | `0.0751731249984` | `0.065372457997` | `0.0043618339987` | `0.000683874997776` | `yes` |
| 4 | `none` | `long_short_reversal` | `event_segments_4_no_risk` | `0.00628270800007` | `0.00698078666675` | `0.0057110830021` | `1.10009047282` | `0.818114529886` | `0.0770962089955` | `0.0652581660033` | `0.00539166700037` | `0.000644207997539` | `yes` |
| 4 | `tp_sl_grid` | `long_short_reversal` | `event_segments_4_tp_sl_15m_grid` | `0.00653299999976` | `0.00725888888863` | `0.00570941700425` | `1.14424992865` | `0.786541451709` | `0.0773354999983` | `0.0651829999988` | `0.00532445799763` | `0.00067691699951` | `yes` |
| 5 | `none` | `long_short_reversal` | `event_segments_5_no_risk` | `0.00772074999986` | `0.00857861111096` | `0.00672049999412` | `1.14883565309` | `0.783401870908` | `0.0792907909999` | `0.064986790996` | `0.00653979100025` | `0.000746666999476` | `yes` |
| 5 | `tp_sl_grid` | `long_short_reversal` | `event_segments_5_tp_sl_15m_grid` | `0.00748891700005` | `0.00832101888894` | `0.00679437500366` | `1.10222308836` | `0.816531616422` | `0.079522292006` | `0.0654009169957` | `0.0063927500014` | `0.000665209001454` | `yes` |
| 6 | `none` | `long_short_reversal` | `event_segments_6_no_risk` | `0.00909241600039` | `0.0101026844449` | `0.00800066599913` | `1.13645739009` | `0.791934662791` | `0.0820969170018` | `0.0654439169957` | `0.00754549999692` | `0.000729791994672` | `yes` |
| 6 | `tp_sl_grid` | `long_short_reversal` | `event_segments_6_tp_sl_15m_grid` | `0.00873970900011` | `0.0097107877779` | `0.00812904199847` | `1.07512164432` | `0.83711457653` | `0.081958540999` | `0.065113167002` | `0.00777095900412` | `0.000742958996852` | `yes` |
| 7 | `none` | `long_short_reversal` | `event_segments_7_no_risk` | `0.0125605829999` | `0.0139562033332` | `0.00989654199657` | `1.26918907677` | `0.709114202502` | `0.0992615840005` | `0.0790757919967` | `0.00890845899994` | `0.000966917003097` | `yes` |
| 7 | `tp_sl_grid` | `long_short_reversal` | `event_segments_7_tp_sl_15m_grid` | `0.0118360000006` | `0.0131511111118` | `0.00995570899977` | `1.18886560474` | `0.757024171963` | `0.0851303750023` | `0.0655492499936` | `0.00873295799829` | `0.000736999994842` | `yes` |

## Historical Strict-Total Failure

The previous run is preserved because it is valid evidence of a bad stage
boundary. It compared service `prepare_pools_total` with canonical notebook
`prepare_pools` and therefore failed `0 / 28`.

| Metric | Value |
|---|---:|
| Historical status | `fail` |
| Failure classification | `stage_boundary_mismatch` |
| Pass count | `0 / 28` |
| Min strict-total ratio | `0.0280811180229` |
| Median strict-total ratio | `0.0791964068629` |
| Max strict-total ratio | `0.149522143785` |
| Max strict-total service s | `0.101825375001` |
| Max strict-total RSS delta MB | `18.734375` |

## Historical By Arity

| Arity | Pass | Min strict ratio | Max strict ratio | Max strict service s |
|---:|---:|---:|---:|---:|
| 1 | 0 / 4 | `0.0280811180229` | `0.0483516457576` | `0.0747121250024` |
| 2 | 0 / 4 | `0.0373687424535` | `0.0742376792707` | `0.0907431660016` |
| 3 | 0 / 4 | `0.0524831519357` | `0.0630900394254` | `0.0912594580004` |
| 4 | 0 / 4 | `0.0764830067237` | `0.0952791531829` | `0.0854176670036` |
| 5 | 0 / 4 | `0.0877264786197` | `0.108064005117` | `0.0948251659938` |
| 6 | 0 / 4 | `0.0929415331166` | `0.105274684745` | `0.0978294170054` |
| 7 | 0 / 4 | `0.124124876892` | `0.149522143785` | `0.101825375001` |

## Next Gate

- Iteration 2 corrected stage-boundary benchmark is accepted for `prepare_pools_core`.
- Keep `prepare_pools_total` as service telemetry and do not compare it directly to canonical notebook `prepare_pools`.
- Iteration 3 may proceed only against the corrected stage contract.
