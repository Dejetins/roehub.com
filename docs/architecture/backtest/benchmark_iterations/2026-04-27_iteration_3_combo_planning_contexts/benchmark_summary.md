# Backtest Benchmark Iteration 3 - combo planning contexts

Iteration 3 records the backend registry, exact/proxy context packing, deterministic Cartesian chunk planning, and optional proxy prefilter boundary before exact scoring.

## Decision

- Current acceptance status: `pass`.
- Mac Studio acceptance: `yes`.
- Pass count: `28 / 28`.
- Measured stages: `build_exact_context`, `build_proxy_context`, `combo_iteration`, `proxy_filter`.
- Canonical combo prefilter config: `combo_top_frac = 1.0`, `combo_min_confirm = 1`, `COMBO_CHUNK_SIZE = 4096`.
- Active proxy filtering is covered by deterministic fixture evidence, not by the full canonical pass-through matrix.

## Stage Contract

| Stage | Classification | Iteration 3 gate |
|---|---|---:|
| `build_exact_context` | combo planning compute | yes |
| `build_proxy_context` | pass-through or active proxy context setup | yes |
| `combo_iteration` | deterministic Cartesian chunk planning | yes |
| `proxy_filter` | pass-through or active candidate pruning | yes |

Exact scoring, TP/SL hit-time loading, heap/top-N, job persistence, and notebook changes are not part of this record.

## Identity

- Acceptance host: `Mac Studio`.
- Hostname: `MacStudioDaniil`.
- Platform: `macOS-15.7.5-arm64-arm-64bit`.
- Python: `3.12.13`.
- Git branch: `main`.
- Git commit: `c4dbfc79eb7fa7be481a3fab8a04093da9d23d20`.
- Git status: `## main...origin/main`.
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`.
- Artifact slot: `slot_b` generation `4`.
- Artifact manifest hash: `bd81f5d19b1b13ddd843143236b90780802ad9baf395bb047bf549d24f71d40e`.
- Hit-times manifest hash: `1f5d3bf464f4beba3e73105d7c561cdd523255db5e27329ae5b22ddd63f170a9` (identity only; `hit_times/15m` arrays were not loaded).
- Request hash: `ceaaec911055082f9c1ecbe8c9e806f1372d0b084ba146e3dc02b70b953e3754`.
- Baseline request hash reference: `22d1a64757a3461507481fabea6d1434de1997f3fd063a180b289a524692c9f1`.
- Artifact published at UTC: `2026-04-27T01:34:12Z`.

## Fixture

- Coordinates: `binance / spot / BTCUSDT`.
- Timeframe: `15m`.
- Time range: `2020-01-11T20:08:00+00:00` to `2026-04-11T20:08:00+00:00`, `[start, end)`.
- Indicator arities: `1..7`; rows per indicator after setup prepare-pools: `6`.
- Risk modes: `none`, `tp_sl_grid`.
- Direction modes: `long_only`, `long_short_reversal`.
- Full matrix proxy mode: `pass_through`.

## Runtime Metrics Without Warmup

| Stage | Min s | Median s | Max s |
|---|---:|---:|---:|
| `build_exact_context` | `5.45800139662e-06` | `0.00104335449578` | `0.00413016600942` |
| `build_proxy_context` | `1.91699655261e-06` | `2.39549990511e-06` | `1.15000002552e-05` |
| `combo_iteration` | `7.70699989516e-06` | `0.000489146004838` | `0.157087292871` |
| `proxy_filter` | `2.70901364274e-06` | `4.58350405097e-06` | `0.000306039000861` |

## Candidate Counts

- Total Cartesian combinations across measured runs: `1343688`.
- Total combo chunks processed: `348`.
- Max combinations in one run: `279936`.
- Max chunks in one run: `69`.
- Max combo planning outer wall time: `0.159469791994` s.

## By Arity

| Arity | Pass | Max combos | Max chunks | Max build_exact s | Max build_proxy s | Max combo_iteration s | Max proxy_filter s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4 / 4` | `6` | `1` | `0.000455417000921` | `1.15000002552e-05` | `2.29169963859e-05` | `8.1250036601e-06` |
| 2 | `4 / 4` | `36` | `1` | `0.000607542009675` | `2.54200131167e-06` | `1.86250108527e-05` | `3.16699151881e-06` |
| 3 | `4 / 4` | `216` | `1` | `0.00103016699723` | `2.4159962777e-06` | `8.84580076672e-05` | `3.41699342243e-06` |
| 4 | `4 / 4` | `1296` | `1` | `0.00151437499153` | `2.25000258069e-06` | `0.000500875990838` | `3.70800262317e-06` |
| 5 | `4 / 4` | `7776` | `2` | `0.00107404201117` | `3.58300167136e-06` | `0.0034978340118` | `0.000108624008135` |
| 6 | `4 / 4` | `46656` | `12` | `0.00191116701171` | `6.49999128655e-06` | `0.0229020010011` | `5.23329799762e-05` |
| 7 | `4 / 4` | `279936` | `69` | `0.00413016600942` | `4.58299473394e-06` | `0.157087292871` | `0.000306039000861` |

## Determinism Evidence

- Cartesian first-chunk order matches `itertools.product`: `pass`.
- Observed first 10 triples: `[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]]`.

## Proxy Filter Evidence

- Inactive/pass-through full matrix: all `28 / 28` runs selected every candidate without changing candidate counts.
- Active fixture `arity_2_matrix_cache`: `pass`, selected indexes `[1, 4]`, input `6`, valid `4`, selected `2`.
- Active fixture `generic_n_eval_stack`: `pass`, selected indexes `[2, 11]`, input `12`, valid `2`, selected `2`.

## Setup Prepare-Pools Evidence

Prepare-pools was used only to construct Iteration 3 inputs. Its aggregate time is recorded as setup evidence and is not compared as an Iteration 3 benchmark stage.

| Arity | Rows per indicator | prepare_pools_core s | prepare_pools_total s | Pass |
|---:|---|---:|---:|---|
| 1 | `ma.dema:6` | `0.0778961670003` | `0.152532499997` | `yes` |
| 2 | `ma.dema:6, ma.hma:6` | `0.00317387499672` | `0.0740937079972` | `yes` |
| 3 | `ma.dema:6, ma.ema:6, ma.hma:6` | `0.00488400000904` | `0.0893392080034` | `yes` |
| 4 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.sma:6` | `0.00630016700597` | `0.0772944580094` | `yes` |
| 5 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.sma:6, ma.wma:6` | `0.0072641250008` | `0.079453665996` | `yes` |
| 6 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.rma:6, ma.sma:6, ma.wma:6` | `0.00922858300328` | `0.0890325000073` | `yes` |
| 7 | `ma.dema:6, ma.ema:6, ma.hma:6, ma.rma:6, ma.sma:6, ma.wma:6, ma.zlema:6` | `0.0112367080001` | `0.0868143749976` | `yes` |

## Decision Detail

- Status: `pass`.
- Reason: stage names, backend ids, exact context requirement, deterministic combo counts/chunks, pass-through candidate counts, and active proxy fixtures all matched expected values.
- Next iteration: proceed to no-risk exact scoring/top-N only after this stage boundary remains unchanged.

