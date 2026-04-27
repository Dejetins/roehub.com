# Backtest Benchmark Iteration 4 - no-risk exact scoring notebook top-K

Iteration 4 implements no-risk exact scoring, no-risk self-check, full no-risk summary metrics, and notebook-compatible `heap_update` / `top_result_proxy_fill` boundaries. This record was produced on the Mac Studio after the implementation was deployed to `/opt/roehub/app`.

## Decision

- Current acceptance status: `fail`.
- Accepted comparison scope: no-risk arity 1..7 x `long_only` / `long_short_reversal` against the canonical notebook benchmark target.
- Semantic metric parity pass count: `14 / 14`.
- Proxy metadata parity pass count: `14 / 14`.
- Strict result hash pass count: `10 / 14`.
- `exact_scoring` latency pass count: `14 / 14`.
- `heap_update` latency pass count: `1 / 14`.
- `top_result_proxy_fill` latency pass count: `12 / 14`.
- `total_without_warmup` latency pass count: `6 / 14`.
- Top-K boundary pass count: `14 / 14`.
- Self-check pass count: `14 / 14`.
- Benchmark recorded at UTC: `2026-04-27T23:20:03.721991+00:00`.
- Rule: `request.top_n = 100` remains a product input, while measured `heap_update`, `top_result_proxy_fill`, `total_without_warmup`, and result hashes use `benchmark_top_k = 5`.
- Failure classification: canonical timer threshold miss for `heap_update` and strict result hash drift for arity 1/2. `top_result_proxy_fill` also misses threshold for arity 2 in both directions.

## Stage Contract

| Stage | Classification | Iteration 4 gate |
|---|---|---:|
| `service_warmup` | sample warmup, `top_k = 1` | yes |
| `self_check` | bounded exact-vs-slow no-risk parity | yes |
| `exact_scoring` | notebook-compatible no-risk exact scoring | yes |
| `heap_update` | compact notebook top-K heap, `benchmark_top_k = 5` | yes |
| `top_result_proxy_fill` | final heap proxy metadata fill only | yes |
| `top_result_assembly` | service-only future public/read-model assembly | no |

`prepare_pools_core`, combo planning, and pass-through proxy filtering are setup stages inherited from Iterations 2 and 3. TP/SL risk scoring, hit-times loading, persisted top-N rows, public identity hashes, API DTOs, lazy trades, and job orchestration are out of scope.

## Identity

- Acceptance host: `Mac Studio`.
- Hostname: `MacStudioDaniil`.
- Platform: `macOS-15.7.5-arm64-arm-64bit`.
- Python: `3.12.13`.
- Git branch: `main`.
- Git commit used for benchmark: `2669ead65485287956f1743ce3423de58fdd9005`.
- Deploy workflow: [`25024400386`](https://github.com/Dejetins/roehub.com/actions/runs/25024400386) with conclusion `success`.
- Deployed runtime path: `/opt/roehub/app`.
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`.
- Artifact slot: `slot_b` generation `4`.
- Artifact manifest hash: `bd81f5d19b1b13ddd843143236b90780802ad9baf395bb047bf549d24f71d40e`.
- Hit-times manifest hash: `1f5d3bf464f4beba3e73105d7c561cdd523255db5e27329ae5b22ddd63f170a9`.
- Request hash: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`.
- Baseline request hash reference: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`.
- Raw remote evidence path: `/tmp/roehub_iteration4_no_risk_benchmark_raw.json`.
- Raw evidence SHA256: `d914630be26ba9c22bbdfb7ef7b63cef702b9a38598ff1e9f8354e3e712f049c`.
- Raw evidence bytes: `119274`.

## Target

The target is the canonical notebook benchmark target from `2026-04-26_engine_test_btcusdt_15m`. The service must keep the notebook-compatible measured top-K boundary independent from product `top_n`.

| Gate | Expected | Actual | Pass |
|---|---|---|---:|
| Request top-N | `request.top_n = 100` | `100` | yes |
| Measured top-K | `benchmark_top_k = 5` | `5` | yes |
| Sample warmup top-K | `top_k = 1` | `1` | yes |
| Top results count | `top_results_count = 5` | `14 / 14` | yes |
| Semantic metric parity | all top-row scoring metrics within tolerance | `14 / 14` | yes |
| Proxy metadata parity | proxy fields within tolerance | `14 / 14` | yes |
| Strict result hash | exact hash match | `10 / 14` | no |
| `exact_scoring` latency | target/service >= `0.9` | `14 / 14` | yes |
| `heap_update` latency | target/service >= `0.9` | `1 / 14` | no |
| `top_result_proxy_fill` latency | target/service >= `0.9` | `12 / 14` | no |
| `total_without_warmup` latency | target/service >= `0.9` | `6 / 14` | no |
| Self-check | exact vs slow reference | `14 / 14` | yes |

## Summary

| Metric | Value |
|---|---:|
| Status | `fail` |
| Semantic metric parity pass count | `14 / 14` |
| Proxy metadata parity pass count | `14 / 14` |
| Strict result hash pass count | `10 / 14` |
| `exact_scoring` latency pass count | `14 / 14` |
| `heap_update` latency pass count | `1 / 14` |
| `top_result_proxy_fill` latency pass count | `12 / 14` |
| `total_without_warmup` latency pass count | `6 / 14` |
| Runtime wall pass count | `6 / 14` |
| Runtime CPU-time pass count | `9 / 14` |
| Peak RSS pass count | `14 / 14` |
| RSS delta pass count | `6 / 14` |
| Min exact speed ratio target/service | `0.960822920027` |
| Median exact speed ratio target/service | `1.00347729307` |
| Max exact speed ratio target/service | `2.71169284946` |
| Min heap speed ratio target/service | `0.702594336949` |
| Median heap speed ratio target/service | `0.831496096315` |
| Max heap speed ratio target/service | `1.55694577878` |
| Min proxy-fill speed ratio target/service | `0.34745947566` |
| Median proxy-fill speed ratio target/service | `0.992337639804` |
| Max proxy-fill speed ratio target/service | `1.13318404714` |
| Max `exact_scoring` s | `138.419522372` |
| Max `heap_update` s | `0.963062373892` |
| Max `top_result_proxy_fill` s | `0.0168697920162` |
| Max RSS delta MB | `52.75` |
| Max peak RSS MB | `539.953125` |
| Max process CPU percent equivalent | `1121.11725778` |

## Stage Metrics

| Stage/Metric | Min | Median | Max |
|---|---:|---:|---:|
| `exact_scoring` s | `0.00105395799619` | `0.316229708493` | `138.419522372` |
| `exact_scoring` speed ratio target/service | `0.960822920027` | `1.00347729307` | `2.71169284946` |
| `heap_update` s | `2.34170001931e-05` | `0.00336735449673` | `0.963062373892` |
| `heap_update` speed ratio target/service | `0.702594336949` | `0.831496096315` | `1.55694577878` |
| `top_result_proxy_fill` s | `0.0073402080161` | `0.0121563124994` | `0.0168697920162` |
| `top_result_proxy_fill` speed ratio target/service | `0.34745947566` | `0.992337639804` | `1.13318404714` |
| `service_total_without_warmup` s | `0.204376999987` | `0.581794416998` | `139.985434875` |
| `service_total_without_warmup` speed ratio target/service | `0.627856822322` | `0.862516380305` | `1.00543382067` |
| RSS delta MB | `0` | `6.8125` | `52.75` |
| peak RSS MB | `280.109375` | `513.375` | `539.953125` |

## By Arity

| Arity | Metric parity | Strict hash | Exact latency | Heap latency | Proxy latency | Total latency | Min exact ratio | Max exact s | Backend display names |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `2 / 2` | `0 / 2` | `2 / 2` | `1 / 2` | `2 / 2` | `0 / 2` | `1.33392922108` | `0.00172404199839` | `event_segments_1_no_risk` |
| 2 | `2 / 2` | `0 / 2` | `2 / 2` | `0 / 2` | `0 / 2` | `0 / 2` | `0.960822920027` | `0.00244012501207` | `event_segments_2_no_risk` |
| 3 | `2 / 2` | `2 / 2` | `2 / 2` | `0 / 2` | `2 / 2` | `0 / 2` | `0.968464757399` | `0.0438977080048` | `event_segments_3_no_risk` |
| 4 | `2 / 2` | `2 / 2` | `2 / 2` | `0 / 2` | `2 / 2` | `0 / 2` | `0.983157616307` | `0.336635791988` | `event_segments_4_no_risk` |
| 5 | `2 / 2` | `2 / 2` | `2 / 2` | `0 / 2` | `2 / 2` | `2 / 2` | `0.979026269155` | `2.050857917` | `event_segments_5_no_risk` |
| 6 | `2 / 2` | `2 / 2` | `2 / 2` | `0 / 2` | `2 / 2` | `2 / 2` | `0.980381718499` | `15.796470541` | `event_segments_6_no_risk` |
| 7 | `2 / 2` | `2 / 2` | `2 / 2` | `0 / 2` | `2 / 2` | `2 / 2` | `0.998529781289` | `138.419522372` | `event_segments_7_no_risk` |

## Tuple Matrix

| Arity | Direction | Backend | Combos | Exact s | Heap s | Proxy fill s | Exact ratio | Heap ratio | Proxy ratio | Total ratio | Hash | Metric parity | Top results | Pass |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `long_only` | `event_segments_1_no_risk` | `6` | `0.00116874999367` | `3.19170067087e-05` | `0.00764687499031` | `2.71169284946` | `0.732368167892` | `0.983517320344` | `0.647856657101` | `no` | `yes` | `5` | `no` |
| 2 | `long_only` | `event_segments_2_no_risk` | `36` | `0.00244012501207` | `0.000102874997538` | `0.0073402080161` | `1.32322441856` | `0.807620917754` | `0.352300370014` | `0.648241925052` | `no` | `yes` | `5` | `no` |
| 3 | `long_only` | `event_segments_3_no_risk` | `216` | `0.0438977080048` | `0.000547625008039` | `0.00963891702122` | `1.10817195729` | `0.794034227698` | `1.13318404714` | `0.7376826045` | `yes` | `yes` | `5` | `no` |
| 4 | `long_only` | `event_segments_4_no_risk` | `1296` | `0.336635791988` | `0.00333729200065` | `0.0121330829861` | `1.04628208997` | `0.821724020465` | `0.980332699738` | `0.889154745461` | `yes` | `yes` | `5` | `no` |
| 5 | `long_only` | `event_segments_5_no_risk` | `7776` | `2.050857917` | `0.0224730000191` | `0.0127748750092` | `1.01940399316` | `0.847479908511` | `0.993316959286` | `0.978948816654` | `yes` | `yes` | `5` | `no` |
| 6 | `long_only` | `event_segments_6_no_risk` | `46656` | `15.796470541` | `0.143867583014` | `0.0143203329935` | `0.996177461172` | `0.856729371673` | `0.997582109749` | `0.988455709377` | `yes` | `yes` | `5` | `no` |
| 7 | `long_only` | `event_segments_7_no_risk` | `279936` | `138.419522372` | `0.963062373892` | `0.0168697920162` | `1.00842480485` | `0.835660437803` | `0.973001503783` | `1.00543382067` | `yes` | `yes` | `5` | `no` |
| 1 | `long_short_reversal` | `event_segments_1_no_risk` | `6` | `0.00172404199839` | `2.34170001931e-05` | `0.00774099997943` | `1.33392922108` | `1.55694577878` | `1.00095814756` | `0.636082591909` | `no` | `yes` | `5` | `no` |
| 2 | `long_short_reversal` | `event_segments_2_no_risk` | `36` | `0.00105395799619` | `0.000110249995487` | `0.00813704100437` | `0.960822920027` | `0.713151958826` | `0.34745947566` | `0.627856822322` | `no` | `yes` | `5` | `no` |
| 3 | `long_short_reversal` | `event_segments_3_no_risk` | `216` | `0.036518665991` | `0.00067766700522` | `0.010797749972` | `0.968464757399` | `0.702594336949` | `0.992567898692` | `0.71430188506` | `yes` | `yes` | `5` | `no` |
| 4 | `long_short_reversal` | `event_segments_4_no_risk` | `1296` | `0.295823624998` | `0.00339741699281` | `0.014037416986` | `0.983157616307` | `0.871041443173` | `0.992107380915` | `0.835878015148` | `yes` | `yes` | `5` | `no` |
| 5 | `long_short_reversal` | `event_segments_5_no_risk` | `7776` | `2.00366116702` | `0.022354042012` | `0.0121795420127` | `0.979026269155` | `0.832004654435` | `1.00264788177` | `0.94072318589` | `yes` | `yes` | `5` | `no` |
| 6 | `long_short_reversal` | `event_segments_6_no_risk` | `46656` | `15.490893584` | `0.144301042019` | `0.0131106659828` | `0.980381718499` | `0.847712152931` | `1.01182243657` | `0.972674967462` | `yes` | `yes` | `5` | `no` |
| 7 | `long_short_reversal` | `event_segments_7_no_risk` | `279936` | `136.313076824` | `0.943649374938` | `0.0146523329895` | `0.998529781289` | `0.830987538196` | `0.975396410284` | `0.995550144004` | `yes` | `yes` | `5` | `no` |

## Failure Evidence

Strict result hash mismatches are limited to arity 1/2. Top-row identities, scoring metrics, and proxy metadata match within tolerance, so this is classified separately from semantic parity.

| Arity | Direction | Service hash | Target hash | Classification |
|---:|---|---|---|---|
| 1 | `long_only` | `56dc16e91c0484b3dd1c362940d2f4e76754d62600533fcaa4b1c554addbb1a8` | `e0434a0b895f7ae354aa0ed7df633379ec69efa9e053b69c4bea1aaaec50dcb3` | `float_representation_drift` |
| 2 | `long_only` | `ea6ac9e77f11fc42f4bc47fb5a799e5323c1baaf09ae6915db826a315c690d1a` | `2fad97724afb919ec9477e195f58a5f31182d44cd36b0f3d9e1590782b81d485` | `float_representation_drift` |
| 1 | `long_short_reversal` | `0d0b688a3cc08ce8983bc7573a95f33e47fe3be1177ad4b7ffca9cf74af7600a` | `268ad29fa2e1a3d8da0b91442d538fe95ed7d150f151306f71013c6e8bdc0249` | `float_representation_drift` |
| 2 | `long_short_reversal` | `e6ee7ab97983f8eec2335524b5a0319ecbca28d4e7948c9a2f16067f4333c446` | `856f7aa6a5a42ac32edce2c9cf126b8871a43ec6416c3d5e2496f5cff828e0ad` | `float_representation_drift` |

Canonical timer misses are shown below. The most consistent miss is `heap_update`, which fails the `target/service >= 0.9` threshold for 13 of 14 tuples.

| Arity | Direction | Exact pass | Heap pass | Proxy pass | Total pass | Exact ratio | Heap ratio | Proxy ratio | Total ratio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `long_only` | `yes` | `no` | `yes` | `no` | `2.71169284946` | `0.732368167892` | `0.983517320344` | `0.647856657101` |
| 2 | `long_only` | `yes` | `no` | `no` | `no` | `1.32322441856` | `0.807620917754` | `0.352300370014` | `0.648241925052` |
| 3 | `long_only` | `yes` | `no` | `yes` | `no` | `1.10817195729` | `0.794034227698` | `1.13318404714` | `0.7376826045` |
| 4 | `long_only` | `yes` | `no` | `yes` | `no` | `1.04628208997` | `0.821724020465` | `0.980332699738` | `0.889154745461` |
| 5 | `long_only` | `yes` | `no` | `yes` | `yes` | `1.01940399316` | `0.847479908511` | `0.993316959286` | `0.978948816654` |
| 6 | `long_only` | `yes` | `no` | `yes` | `yes` | `0.996177461172` | `0.856729371673` | `0.997582109749` | `0.988455709377` |
| 7 | `long_only` | `yes` | `no` | `yes` | `yes` | `1.00842480485` | `0.835660437803` | `0.973001503783` | `1.00543382067` |
| 1 | `long_short_reversal` | `yes` | `yes` | `yes` | `no` | `1.33392922108` | `1.55694577878` | `1.00095814756` | `0.636082591909` |
| 2 | `long_short_reversal` | `yes` | `no` | `no` | `no` | `0.960822920027` | `0.713151958826` | `0.34745947566` | `0.627856822322` |
| 3 | `long_short_reversal` | `yes` | `no` | `yes` | `no` | `0.968464757399` | `0.702594336949` | `0.992567898692` | `0.71430188506` |
| 4 | `long_short_reversal` | `yes` | `no` | `yes` | `no` | `0.983157616307` | `0.871041443173` | `0.992107380915` | `0.835878015148` |
| 5 | `long_short_reversal` | `yes` | `no` | `yes` | `yes` | `0.979026269155` | `0.832004654435` | `1.00264788177` | `0.94072318589` |
| 6 | `long_short_reversal` | `yes` | `no` | `yes` | `yes` | `0.980381718499` | `0.847712152931` | `1.01182243657` | `0.972674967462` |
| 7 | `long_short_reversal` | `yes` | `no` | `yes` | `yes` | `0.998529781289` | `0.830987538196` | `0.975396410284` | `0.995550144004` |

## Next Gate

- Iteration 4 service implementation is deployed and benchmarked on Mac Studio, but the benchmark acceptance status is `fail`.
- `request.top_n = 100`, `benchmark_top_k = 5`, heap capacity `5`, sample warmup `top_k = 1`, and `top_results_count = 5` were preserved in every measured tuple.
- Do not treat this record as accepted until `heap_update` and `total_without_warmup` meet the canonical timer threshold and strict result-hash parity is resolved or explicitly waived as non-semantic.
