# Backtest Benchmark Iteration 3 - combo planning contexts

Iteration 3 implements the backend registry, exact/proxy planning contexts,
deterministic Cartesian combo chunking, and optional proxy prefilter boundary.
This record is the Mac Studio acceptance benchmark for combo planning before
exact scoring.

## Decision

- Current acceptance status: `pass`.
- Accepted comparison scope: Iteration 3 combo planning stages only.
- Accepted pass count: `28 / 28`.
- Active proxy fixture pass count: `2 / 2`.
- Benchmark recorded at UTC: `2026-04-27T19:51:19.052485+00:00`.
- Rule: all supported v1 backend selections, deterministic combo counts/chunks,
  pass-through candidate counts, exact-context requirements, and active proxy
  fixtures must match expected values.
- No notebook speed-ratio target is applied in Iteration 3 because exact scoring,
  heap/top-N, self-check, and hit-time loading are not part of this stage.

## Stage Contract

| Stage | Classification | Iteration 3 gate |
|---|---|---:|
| `build_exact_context` | arity-first segment context packing | yes |
| `build_proxy_context` | pass-through or active proxy context setup | yes |
| `combo_iteration` | deterministic Cartesian chunk planning | yes |
| `proxy_filter` | pass-through or active candidate pruning | yes |

`prepare_pools_core` is setup input construction for this record and is not an
Iteration 3 comparison target. `exact_scoring`, `tp_sl_exact_scoring`,
`heap_update`, `top_result_proxy_fill`, `load_hit_times`, job persistence, and
notebook edits are out of scope.

## Identity

- Acceptance host: `Mac Studio`.
- Hostname: `MacStudioDaniil`.
- Platform: `macOS-15.7.5-arm64-arm-64bit`.
- Python: `3.12.13`.
- Git branch: `main`.
- Git commit used for benchmark: `c4dbfc79eb7fa7be481a3fab8a04093da9d23d20`.
- Evidence commit: `7f406f1b8f151c4ea04727a9675b8ded7345e7d9`.
- Git status at benchmark: `## main...origin/main`.
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`.
- Artifact slot: `slot_b` generation `4`.
- Artifact manifest hash: `bd81f5d19b1b13ddd843143236b90780802ad9baf395bb047bf549d24f71d40e`.
- Hit-times manifest hash: `1f5d3bf464f4beba3e73105d7c561cdd523255db5e27329ae5b22ddd63f170a9`.
- Artifact published at UTC: `2026-04-27T01:34:12Z`.
- Request hash: `ceaaec911055082f9c1ecbe8c9e806f1372d0b084ba146e3dc02b70b953e3754`.
- Baseline request hash reference: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`.
- Raw evidence path:
  `docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/benchmark_results.json`.

## Target

The target is a deterministic stage-boundary acceptance gate, not a
notebook-runtime ratio. The canonical Iteration 3 full matrix uses
`combo_top_frac = 1.0`, `combo_min_confirm = 1`, and `COMBO_CHUNK_SIZE = 4096`,
so `build_proxy_context` and `proxy_filter` are expected to be pass-through and
near-zero in the full acceptance matrix.

| Gate | Expected | Actual | Pass |
|---|---|---|---:|
| Supported tuple matrix | `28 / 28` | `28 / 28` | yes |
| Active proxy fixtures | `2 / 2` | `2 / 2` | yes |
| Deterministic ordering | `itertools.product` order | matched first chunk evidence | yes |
| Full-matrix proxy mode | pass-through | pass-through, all candidates selected | yes |
| Hit-times loading | not loaded | not loaded | yes |
| Exact scoring / heap | not run | not run | yes |

## Accepted Summary

| Metric | Value |
|---|---:|
| Status | `pass` |
| Pass count | `28 / 28` |
| Active proxy fixture pass count | `2 / 2` |
| Total Cartesian combinations | `1343688` |
| Total combo chunks processed | `348` |
| Max combinations in one run | `279936` |
| Max chunks in one run | `69` |
| Max combo planning outer wall s | `0.159469791994` |
| Max CPU time delta s | `0.159977` |
| Max `build_exact_context` s | `0.00413016600942` |
| Max `build_proxy_context` s | `1.15000002552e-05` |
| Max `combo_iteration` s | `0.157087292871` |
| Max `proxy_filter` s | `0.000306039000861` |

## Accepted Stage Metrics

| Stage/Metric | Min | Median | Max |
|---|---:|---:|---:|
| `build_exact_context` s | `5.45800139662e-06` | `0.00104335449578` | `0.00413016600942` |
| `build_proxy_context` s | `1.91699655261e-06` | `2.39549990511e-06` | `1.15000002552e-05` |
| `combo_iteration` s | `7.70699989516e-06` | `0.000489146004838` | `0.157087292871` |
| `proxy_filter` s | `2.70901364274e-06` | `4.58350405097e-06` | `0.000306039000861` |

## Accepted By Arity

| Arity | Pass | Max combos | Max chunks | Max exact ctx s | Max proxy ctx s | Max combo iter s | Max proxy filter s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 / 4 | `6` | `1` | `0.000455417000921` | `1.15000002552e-05` | `2.29169963859e-05` | `8.1250036601e-06` |
| 2 | 4 / 4 | `36` | `1` | `0.000607542009675` | `2.54200131167e-06` | `1.86250108527e-05` | `3.16699151881e-06` |
| 3 | 4 / 4 | `216` | `1` | `0.00103016699723` | `2.4159962777e-06` | `8.84580076672e-05` | `3.41699342243e-06` |
| 4 | 4 / 4 | `1296` | `1` | `0.00151437499153` | `2.25000258069e-06` | `0.000500875990838` | `3.70800262317e-06` |
| 5 | 4 / 4 | `7776` | `2` | `0.00107404201117` | `3.58300167136e-06` | `0.0034978340118` | `0.000108624008135` |
| 6 | 4 / 4 | `46656` | `12` | `0.00191116701171` | `6.49999128655e-06` | `0.0229020010011` | `5.23329799762e-05` |
| 7 | 4 / 4 | `279936` | `69` | `0.00413016600942` | `4.58299473394e-06` | `0.157087292871` | `0.000306039000861` |

## Accepted Tuple Matrix

| Arity | Risk | Direction | Backend | Combos | Chunks | Exact ctx | Exact ctx s | Proxy ctx s | Combo iter s | Proxy filter s | Candidates | Pass |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `none` | `long_only` | `event_segments_n_no_risk` | `6` | `1` | `yes` | `0.000326083987602` | `6.50000583846e-06` | `1.37919996632e-05` | `8.1250036601e-06` | `6` | `yes` |
| 1 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `6` | `1` | `yes` | `0.000381500009098` | `3.58400575351e-06` | `9.50001413003e-06` | `4.08400956076e-06` | `6` | `yes` |
| 2 | `none` | `long_only` | `event_segments_2_no_risk` | `36` | `1` | `no` | `5.45800139662e-06` | `2.37500353251e-06` | `1.68760016095e-05` | `3.16699151881e-06` | `36` | `yes` |
| 2 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `36` | `1` | `yes` | `0.000463582997327` | `2.25000258069e-06` | `1.50839914568e-05` | `2.79200321529e-06` | `36` | `yes` |
| 3 | `none` | `long_only` | `event_segments_n_no_risk` | `216` | `1` | `yes` | `0.00102762501047` | `2.37500353251e-06` | `8.19160050014e-05` | `3.41699342243e-06` | `216` | `yes` |
| 3 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `216` | `1` | `yes` | `0.000803040995379` | `2.29099532589e-06` | `7.93750077719e-05` | `2.91600008495e-06` | `216` | `yes` |
| 4 | `none` | `long_only` | `event_segments_n_no_risk` | `1296` | `1` | `yes` | `0.0011359170021` | `2.25000258069e-06` | `0.000490959006129` | `3.37499659508e-06` | `1296` | `yes` |
| 4 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `1296` | `1` | `yes` | `0.00113074999535` | `2.04199750442e-06` | `0.000482665986056` | `3.16600198857e-06` | `1296` | `yes` |
| 5 | `none` | `long_only` | `event_segments_n_no_risk` | `7776` | `2` | `yes` | `0.000979834003374` | `2.12500162888e-06` | `0.00343970897666` | `7.99899862614e-06` | `7776` | `yes` |
| 5 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `7776` | `2` | `yes` | `0.000968209002167` | `2.08400888368e-06` | `0.0034978340118` | `7.6669966802e-06` | `7776` | `yes` |
| 6 | `none` | `long_only` | `event_segments_n_no_risk` | `46656` | `12` | `yes` | `0.00124179199338` | `1.99998612516e-06` | `0.022595041999` | `4.49980143458e-05` | `46656` | `yes` |
| 6 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `46656` | `12` | `yes` | `0.00117849999515` | `2.50000448432e-06` | `0.0226766260166` | `4.38710121671e-05` | `46656` | `yes` |
| 7 | `none` | `long_only` | `event_segments_n_no_risk` | `279936` | `69` | `yes` | `0.00191408400133` | `2.41700035986e-06` | `0.154808044084` | `0.000269829004537` | `279936` | `yes` |
| 7 | `tp_sl_grid` | `long_only` | `event_segments_n_tp_sl_15m_grid` | `279936` | `69` | `yes` | `0.00201179200667` | `2.41700035986e-06` | `0.157087292871` | `0.000306039000861` | `279936` | `yes` |
| 1 | `none` | `long_short_reversal` | `event_segments_n_no_risk` | `6` | `1` | `yes` | `0.000455417000921` | `1.15000002552e-05` | `2.29169963859e-05` | `5.08299854118e-06` | `6` | `yes` |
| 1 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `6` | `1` | `yes` | `0.000307958005578` | `3.4999975469e-06` | `7.70699989516e-06` | `3.12499469146e-06` | `6` | `yes` |
| 2 | `none` | `long_short_reversal` | `event_segments_2_no_risk` | `36` | `1` | `no` | `5.49999822397e-06` | `2.20899528358e-06` | `1.86250108527e-05` | `2.79200321529e-06` | `36` | `yes` |
| 2 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `36` | `1` | `yes` | `0.000607542009675` | `2.54200131167e-06` | `1.47920072777e-05` | `2.70901364274e-06` | `36` | `yes` |
| 3 | `none` | `long_short_reversal` | `event_segments_n_no_risk` | `216` | `1` | `yes` | `0.00103016699723` | `2.29100987781e-06` | `8.84580076672e-05` | `3.08299786411e-06` | `216` | `yes` |
| 3 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `216` | `1` | `yes` | `0.00101587499375` | `2.4159962777e-06` | `7.53749918658e-05` | `2.91600008495e-06` | `216` | `yes` |
| 4 | `none` | `long_short_reversal` | `event_segments_n_no_risk` | `1296` | `1` | `yes` | `0.00151437499153` | `2.12500162888e-06` | `0.000500875990838` | `3.70800262317e-06` | `1296` | `yes` |
| 4 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `1296` | `1` | `yes` | `0.0012548329978` | `1.91699655261e-06` | `0.000487333003548` | `3.24999564327e-06` | `1296` | `yes` |
| 5 | `none` | `long_short_reversal` | `event_segments_n_no_risk` | `7776` | `2` | `yes` | `0.00105654199433` | `2.16699845623e-06` | `0.00345433299663` | `0.000108624008135` | `7776` | `yes` |
| 5 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `7776` | `2` | `yes` | `0.00107404201117` | `3.58300167136e-06` | `0.00345216599817` | `1.1916999938e-05` | `7776` | `yes` |
| 6 | `none` | `long_short_reversal` | `event_segments_n_no_risk` | `46656` | `12` | `yes` | `0.00191116701171` | `2.74999183603e-06` | `0.0228322100011` | `5.23329799762e-05` | `46656` | `yes` |
| 6 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `46656` | `12` | `yes` | `0.00132049999957` | `6.49999128655e-06` | `0.0229020010011` | `4.4622036512e-05` | `46656` | `yes` |
| 7 | `none` | `long_short_reversal` | `event_segments_n_no_risk` | `279936` | `69` | `yes` | `0.00312049999775` | `3.41700797435e-06` | `0.154693746052` | `0.000270369026111` | `279936` | `yes` |
| 7 | `tp_sl_grid` | `long_short_reversal` | `event_segments_n_tp_sl_15m_grid` | `279936` | `69` | `yes` | `0.00413016600942` | `4.58299473394e-06` | `0.155035047006` | `0.000268036004854` | `279936` | `yes` |

## Active Proxy Fixture Evidence

The full benchmark matrix intentionally uses the canonical pass-through proxy
configuration. Active proxy behavior is preserved with deterministic small
fixtures because active full-matrix generic-N proxy filtering is not the
canonical Iteration 3 target.

| Fixture | Context | Config | Expected selected indexes | Actual selected indexes | Input | Valid | Selected | Build ctx s | Filter s | Pass |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `arity_2_matrix_cache` | `matrix_two` | `combo_top_frac=0.5`, `combo_min_confirm=1` | `[1, 4]` | `[1, 4]` | `6` | `4` | `2` | `6.20000064373e-05` | `5.74159930693e-05` | `yes` |
| `generic_n_eval_stack` | `generic_n` | `combo_top_frac=1.0`, `combo_min_confirm=2` | `[2, 11]` | `[2, 11]` | `12` | `2` | `2` | `1.95839966182e-05` | `6.88750005793e-05` | `yes` |

## Deterministic Ordering Evidence

| Evidence | Expected | Actual | Pass |
|---|---|---|---:|
| First 10 triples for arity 3, rows `0..5`, chunk size `10` | `itertools.product` order | `[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]]` | yes |

## Setup Prepare-Pools Evidence

Prepare-pools was used only to construct Iteration 3 inputs. It is setup
evidence, not an Iteration 3 comparison stage.

| Arity | Rows per indicator | `prepare_pools_core` s | `prepare_pools_total` s | Pass |
|---:|---|---:|---:|---:|
| 1 | `ma.dema:6` | `0.0778961670003` | `0.152532499997` | `yes` |
| 2 | `ma.dema:6, ma.hma:6` | `0.00317387499672` | `0.0740937079972` | `yes` |
| 3 | `ma.dema:6, ma.ema:6, ma.hma:6` | `0.00488400000904` | `0.0893392080034` | `yes` |
| 4 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.sma:6` | `0.00630016700597` | `0.0772944580094` | `yes` |
| 5 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.sma:6, ma.wma:6` | `0.0072641250008` | `0.079453665996` | `yes` |
| 6 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.rma:6, ma.sma:6, ma.wma:6` | `0.00922858300328` | `0.0890325000073` | `yes` |
| 7 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.rma:6, ma.sma:6, ma.wma:6, ma.zlema:6` | `0.0112367080001` | `0.0868143749976` | `yes` |

## Next Gate

- Iteration 3 combo-planning stage boundary is accepted on Mac Studio.
- Preserve pass-through proxy behavior for canonical default
  `combo_top_frac = 1.0`, `combo_min_confirm = 1`.
- Keep active proxy fixture coverage for `matrix_two` and `generic_n` paths.
- Iteration 4 may proceed to no-risk exact scoring and top-N without adding
  exact scoring, heap, or persistence work retroactively into this benchmark.
