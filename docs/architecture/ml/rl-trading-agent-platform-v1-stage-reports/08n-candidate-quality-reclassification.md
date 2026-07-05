---
doc: rl-trading-agent-platform-v1-stage-08n-candidate-quality-reclassification
stage: 08N
status: accepted
quality_status: accepted_for_research_only
updated_at: 2026-07-05
---

# Stage 08N: candidate quality reclassification

Stage `08N` reclassifies accepted Stage `08M` candidate
`stage08m_a3823cbd01143878_fd7c614b` before any Stage `17+`
runtime-load, soak, mainnet-readiness, canary or product rollout progression.

Decision: `quality_status=accepted_for_research_only`.

Historical Stage `08M stage09_allowed=true` remains valid only as the
registry/plumbing handoff that opened Stage `09`. For future stages, `08M` is
downgraded to a narrower current classification: it is accepted research
evidence, not promotion-grade trading quality, not product suitability, and not
mainnet readiness.

Proof boundary label: `target_host_readiness_pre_main`.
Stage-specific evidence tag: `target_host_non_production_quality_reclassification_pre_main`.
The evidence is artifact-only analysis of Mac Studio files under
`/opt/roehub/state/rl_trading/` plus local docs/ledger updates. This is not
`read_only_existing_runtime_smoke`, not `post_main_production_runtime_proof`,
and not a production claim for changed code. Any future
`post_main_production_runtime_proof` for changed runtime behavior requires the
target revision to be on `main`, green CI/GitHub Actions for that revision,
deploy or verified sync into `/opt/roehub/app`, and then the relevant production
runtime smoke from `/opt/roehub/app`.

## Prompt And Prerequisites

| Item | Result |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/08n-candidate-quality-reclassification.md` |
| Prompt sha256 | `61c8dda9468431e0d2b7505410b7ca6aedfbf2a52f1782eccab39f5e3fe76d2e` |
| `current_stage` before execution | `08N` |
| `08M` prerequisite | accepted |
| `09`, `09B`, `10`, `10A`, `11`, `12`, `13`, `14`, `15`, `16` prerequisites | accepted |
| `17` status before execution | pending; no accepted Stage `17` report |
| Credential requirement | none; no browser auth, provider credential or secret read |
| Runtime activation | none; no training, tuning, registry mutation, promotion, paper/testnet/live submit or mainnet submit |

## Verified Artifacts

| Artifact | Verified result |
|---|---|
| `08M` summary | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json` |
| `08M` summary sha256 | `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7` |
| `08M` candidate manifest | `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` |
| `08M` candidate manifest sha256 | `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |
| Stage `10` calibration summary | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_summary.json`, sha256 `d4bfe8aaeb337e5941ba976de5d0fe043cb16469f298820737e03758af401ad6` |
| Stage `10` calibration pack | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_pack.json`, sha256 `7ee51c9f58d8054be97ba2c444a585a99aabbf50ba3ca2e47a78f0d7dbae4219` |
| Stage `10` registry record | `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_calibration_registry_record.json`, sha256 `c0cb139c4a585fcce2a16d6a17098ddd379655de1bd8bb9f42ffb7b7c5eaa5fd` |

Local `/opt/roehub/state/rl_trading/` did not contain these artifacts. The
hashes and metrics above were read from `macstudio:/opt/roehub/state/rl_trading/`.

## Methodology

| Field | Value |
|---|---|
| Уровень глубины | `standard_candidate_quality_reclassification` |
| Тип задачи | governance-quality decision from existing trading artifacts |
| Единица анализа | final-holdout candidate, ticker row, month bucket, volatility bucket and Stage `10` calibration row |
| Основные метрики | net PnL after costs, artifact return field, closed trades, average/median trade PnL, win rate, month/ticker/volatility stability, cost sensitivity, calibration actionability |
| Проверка качества данных | required artifact hashes matched; local `/opt` missing, Mac Studio artifact root used; no raw provider payloads or secrets read |
| Статус вывода | `partially_confirmed_for_research_only`; positive PnL is real in artifacts, promotion-grade product quality is not confirmed |

## Business Impact

`08N` prevents a technically accepted RL plumbing chain from being mistaken for
a tradable product. For business and operations this means:

- users must not see this candidate as a paid live-trading edge;
- support/legal/product readiness work for mainnet remains closed;
- infrastructure engineers may still measure scheduler/feed/load behavior using
  the candidate as a non-product technical payload;
- the next useful investment is either infrastructure proof under Stage `17` or
  a separate research-quality improvement, not wider rollout.

The practical reading of the `08M` result is: the candidate made money in the
stored final-holdout artifact, but the edge is too thin and uneven to support a
real-money or product promise. Average PnL per trade is only `5.530614796964642`
quote units after recorded costs, `258/323` ticker rows fail closed in Stage
`10`, April and low-volatility buckets are negative, and doubled recorded
fee/slippage makes total PnL negative.

Anti-bullshit check:

| Claim | Evidence | Limitation | Status |
|---|---|---|---|
| `08M` is positive after recorded costs | verified summary hash plus read-only recomputation from candidate manifest: `23018.4187849668` net PnL, `4162` trades | scorecard is non-production artifact evidence, not live trading | confirmed |
| `08M` is promotion-grade | Stage `10` leaves only `65/323` ticker rows actionable, April and low-volatility buckets are negative, doubled fee/slippage turns net PnL negative | no funding stress, no drawdown in `08M` scorecard | not confirmed |
| Stage `17+` can continue as product/mainnet readiness | no evidence resolves weak aggregate/product concern | only Stage `17` infrastructure-only load evidence is useful now, and that future evidence still must not be labeled `post_main_production_runtime_proof` unless it meets the `main` plus green CI/GitHub Actions plus deploy/sync boundary | not confirmed |

## Quality Reclassification Matrix

| Surface | Evidence | Decision |
|---|---|---|
| Aggregate return | final PnL `23018.4187849668`; artifact `return_pct_after_costs=2.3018418785`; source `initial_balance=10000.0`; `position_fraction=0.5`; `4162` closed trades | positive, but not promotion-grade by itself |
| Per-trade expectancy | average PnL per trade `5.530614796964642`; median PnL per trade `21.164443899400037`; p05 `-262.0266961315691`; p95 `206.172945932252`; min `-2263.246796840615`; max `932.5301688401104` | weak average edge with heavy downside tail |
| Baselines | `hold_no_trade=0.0`; `deterministic_random_contextual_bandit=-9100.9261598118`; `simple_recent_return_threshold_contextual_bandit=-53438.2414711871`; candidate `23018.4187849668`; oracle diagnostic `425716.5220601573` | beats weak technical baselines, far below oracle diagnostic |
| Stage `08K` comparison | blocked native DQN article candidate had PnL `12502.65333026`, `316` trades, blockers `single_group_dominates_final_result` and `ticker_stability_obviously_broken` | `08M` is better and more stable than blocked `08K`, but this does not prove promotion-grade quality |
| Monthly stability | `2025-03`: `11521.3437407485`, positive ratio `0.6001780944`; `2025-04`: `-5982.7499163305`, positive ratio `0.5276450512`; `2025-05`: `17479.8249605489`, positive ratio `0.6010165184`; monthly dominance `0.4996531449664722` | one month is negative; not a clean product-quality series |
| Volatility buckets | high: `21716.3145223347`, positive ratio `0.6157173756`; medium: `3592.6975685012`, positive ratio `0.553314121`; low: `-2290.593305869`, positive ratio `0.5558759913`; strict-gate volatility dominance `0.786834239482547` vs limit `0.8` | barely below dominance limit; low-volatility evidence is negative |
| Ticker stability | `323` ticker rows; `211` positive-PnL rows; `112` non-positive-PnL rows; median ticker PnL `110.497332241`; ticker positive group ratio `0.653250773993808`; dominant ticker `ACTUSDT` share `0.023532616397542595` | broad enough for research, but many tickers fail product actionability |
| Drawdown | not available in Stage `08M` scorecard; Stage `10` records `drawdown=not_available_in_stage08m_scorecard_fail_closed_for_stage10` and weight `0.0` | fail-closed for promotion-grade decision |
| Fee/slippage stress | recorded cost ratio `0.0013`, cost per trade `6.5`, total cost proxy `27053.0`, net-to-cost ratio `0.8508638149176372`; +1bp round trip net `20937.418784966845`; +5bp net `12613.418784966836`; doubled cost net `-4034.5812150331512` | does not survive doubled recorded fee/slippage |
| Funding stress | funding is not available in Stage `08M` artifacts | fail-closed for mainnet/product readiness |
| Turnover and cost efficiency | candidate trades every backtest session: `4162/4162`; no hold predictions; gross before recorded fee/slippage proxy `50071.41878496684`, net `23018.418784966838` | high turnover consumes more in recorded cost than it keeps in net edge |
| Action distribution and one-sided bias | `open_long=1664`, `open_short=2498`, open-side dominance `0.6001922152811149`; `hold=0`; `close=4162` | not one-sided enough to block research, but always-trade behavior is product risk |
| Stage `10` calibration impact | `323` rows, `65` accepted/actionable, `258` fail-closed/blocked; blocked reasons: `198` insufficient ticker sessions, `112` non-positive ticker PnL, `73` positive-ratio below minimum, `258` `ticker_calibration_not_accepted` | no product-eligible tickers under `08N`; `65` are research-only/actionable for technical infrastructure tests; `258` fail closed |

## Per-Ticker And Calibration Samples

Full per-ticker evidence remains in the verified `08M` summary
`stability_by_ticker` rows and the Stage `10` `ticker_calibrations` pack. This
report summarizes representative rows and the decision logic instead of copying
all `323` rows into docs.

Top positive `08M` ticker rows:

| Symbol | PnL | Trades/sessions | Win-rate proxy | Stage `10` actionability |
|---|---:|---:|---:|---|
| `BANANAS31USDT` | `2587.9930047754` | `39` | `0.7435897436` | research-only accepted/actionable |
| `BABYUSDT` | `1727.1519607005` | `27` | `0.6666666667` | research-only accepted/actionable |
| `SCRTUSDT` | `1602.5066088503` | `12` | `0.9166666667` | research-only accepted/actionable |
| `INITUSDT` | `1517.6164372805` | `29` | `0.6896551724` | research-only accepted/actionable |
| `BROCCOLI714USDT` | `1392.644593047` | `55` | `0.6545454545` | research-only accepted/actionable |

Worst negative `08M` ticker rows:

| Symbol | PnL | Trades/sessions | Win-rate proxy | Stage `10` fail-closed reason |
|---|---:|---:|---:|---|
| `ACTUSDT` | `-2914.7404740582` | `36` | `0.4722222222` | `non_positive_ticker_pnl_after_costs`, `ticker_positive_ratio_below_minimum` |
| `TUTUSDT` | `-1773.8188370412` | `51` | `0.5098039216` | `non_positive_ticker_pnl_after_costs` |
| `SWARMSUSDT` | `-1631.7517755629` | `39` | `0.4871794872` | `non_positive_ticker_pnl_after_costs`, `ticker_positive_ratio_below_minimum` |
| `JELLYJELLYUSDT` | `-1609.5770562804` | `72` | `0.5277777778` | `non_positive_ticker_pnl_after_costs` |
| `API3USDT` | `-1585.2296194106` | `19` | `0.4210526316` | `non_positive_ticker_pnl_after_costs`, `ticker_positive_ratio_below_minimum` |

Stage `10` reclassification:

| Class | Count | Meaning after `08N` |
|---|---:|---|
| product-eligible | `0` | no ticker is product/mainnet eligible because candidate is not promotion-grade |
| research-only/actionable | `65` | may be used only inside infrastructure-only or monitor-only technical proof, with no trading-quality claim |
| fail-closed | `258` | must skip signal/sizing through `ticker_calibration_not_accepted` and concrete blockers |

## Downstream Gates

| Gate | Boolean | Reason |
|---|---|---|
| `stage17_infrastructure_only_allowed` | `true` | Stage `17` may benchmark runtime/load plumbing only and must label evidence `infrastructure_only` |
| `stage17_full_runtime_allowed` | `false` | candidate is not promotion-grade and cannot claim trading/product quality |
| `stage18_monitor_only_technical_soak_allowed` | `true` | after accepted infrastructure-only Stage `17`, a monitor-only technical soak may prove runtime stability only |
| `stage18_soak_allowed` | `false` | no full trade-readiness soak for this candidate |
| `stage19_mainnet_readiness_allowed` | `false` | weak aggregate/product concern unresolved |
| `stage20_mainnet_canary_allowed` | `false` | no real-money canary |
| `stage21_product_rollout_allowed` | `false` | no product rollout |

Next executable prompt: `.codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md`
only with `infrastructure_only`. Stage `18` may not run until Stage `17` is
accepted, and only `stage18_monitor_only_technical_soak_allowed=true` can be
used from this `08N` decision.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | no API route, payload or browser behavior changed |
| Port contract | `none` | no port/interface changed |
| DTO schema | `none` | no DTO changed |
| Persisted schema | `none` | no migration or storage schema changed |
| Config schema/defaults | `none` | no runtime config or feature flag changed |
| Request hash/cache key/persistence identity | `none` | no hash/key identity changed |
| Service-call auth/timeout/retry/error semantics | `none` | no service calls changed |
| External side-effect/idempotency/unknown-state semantics | `none` | no provider, exchange, paper/testnet/live or DB side effect |
| Logs/metrics/traces/audit/redaction | `compatible-change` | adds sanitized architecture report and ledger decision; no secrets or raw provider payloads |
| Benchmark/rollout gates | `compatible-change` | fail-closes Stage `17+` product/mainnet progression and opens only infrastructure-only Stage `17` |
| Browser-visible behavior | `none` | browser/auth work explicitly out of scope |
| Performance hot path | `none` | no runtime code touched |

## File Manifest

Repository files:

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md` | created | Stage `08N` quality reclassification report and evidence handoff | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark `08N` accepted, set downstream booleans, open Stage `17` as `infrastructure_only` only | `compatible-change` docs/ledger |
| `docs/architecture/README.md` | modified | docs index regeneration after adding Stage `08N` report | `compatible-change` docs index |

Runtime artifact paths used read-only:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json` | read from `macstudio`; sha256 matched `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_manifest.json` | read from `macstudio`; sha256 matched `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c` |
| `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_summary.json` | read from `macstudio`; sha256 `d4bfe8aaeb337e5941ba976de5d0fe043cb16469f298820737e03758af401ad6` |
| `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_per_ticker_calibration_pack.json` | read from `macstudio`; sha256 `7ee51c9f58d8054be97ba2c444a585a99aabbf50ba3ca2e47a78f0d7dbae4219` |
| `/opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1/stage10_macstudio_20260702t213000z/stage10_calibration_registry_record.json` | read from `macstudio`; sha256 `c0cb139c4a585fcce2a16d6a17098ddd379655de1bd8bb9f42ffb7b7c5eaa5fd` |

Foreign changes intentionally excluded: pre-existing dirty prompt files
`.codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md`,
`18-rl-soak-incident-drills.md`, `19-mainnet-readiness-review.md`,
`20-bounded-mainnet-canary.md`, `21-product-rollout.md`, the dirty plan doc
`docs/architecture/ml/rl-trading-agent-platform-v1.md`, and the pre-existing
untracked prompt file
`.codex/agents/generated/rl-trading-agent-platform-v1/08n-candidate-quality-reclassification.md`.

## Validation

| Check | Result |
|---|---|
| `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08n-candidate-quality-reclassification.md` | `61c8dda9468431e0d2b7505410b7ca6aedfbf2a52f1782eccab39f5e3fe76d2e` |
| Stage gate from ledger | passed: `current_stage=08N`, prerequisites accepted, `17` pending |
| `08M` summary and candidate manifest hash check on `macstudio` | passed |
| Stage `10` summary/pack/registry hash check on `macstudio` | passed |
| Read-only candidate-median and cost-stress recomputation | passed; recomputed net PnL `23018.418784966838` matched summary |
| Runtime activation side effects | none |

## Residual Risks

- Stage `08M` artifacts do not include drawdown or funding stress, so product
  and mainnet quality stay fail-closed.
- The candidate's average trade edge is small relative to recorded fee/slippage
  cost and turns negative if recorded fee/slippage is doubled.
- Stage `10` leaves `258/323` tickers fail-closed, so later runtime stages must
  preserve ticker skip reasons and must not fall back to a global-only threshold.
- Stage `17` can produce infrastructure evidence only. It cannot be used as
  model-quality, trade-readiness, product-readiness or mainnet-readiness proof.

## Next-Stage Handoff

Run `.codex/agents/generated/rl-trading-agent-platform-v1/17-multi-ticker-runtime-load.md`
next only as `infrastructure_only`.

Stage `17` must carry this `08N` decision forward:

- `quality_status=accepted_for_research_only`;
- `stage17_infrastructure_only_allowed=true`;
- `stage17_full_runtime_allowed=false`;
- `stage18_monitor_only_technical_soak_allowed=true`;
- `stage18_soak_allowed=false`;
- `stage19_mainnet_readiness_allowed=false`;
- `stage20_mainnet_canary_allowed=false`;
- `stage21_product_rollout_allowed=false`.
