---
doc: rl-trading-agent-platform-v1-stage-08o-stage08k-dqn-forensic-decomposition
stage: 08O
status: accepted
updated_at: 2026-07-06
proof_boundary: target_host_non_production_forensic_pre_main
---

# Stage 08O: Stage 08K DQN forensic decomposition

Статус: `accepted`.

User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat

Stage `08O` выполнил forensic decomposition заблокированного Stage `08K` Roehub-native DQN candidate. Обучение, fine-tuning, `Optuna`, registry mutation, runtime activation, `/opt/roehub/app` sync/restart, browser/auth, paper/testnet/live/mainnet execution и exchange/provider side effects не выполнялись.

Доказательная граница этого stage: `target_host_non_production_forensic_pre_main`.
Использованы только local checkout docs/code и read/write sanitized ML artifacts
under `macstudio:/opt/roehub/state/rl_trading/`. Это offline ML artifact
forensic evidence only: оно не читает, не синхронизирует, не перезапускает и не
проверяет `/opt/roehub/app`, не доказывает production runtime behavior и не
является delivery/runtime acceptance claim.

| Canonical proof boundary | Stage `08O` status |
|---|---|
| `target_host_readiness_pre_main` | not collected; Stage `08O` does not perform host readiness checks |
| `read_only_existing_runtime_smoke` | not collected; Stage `08O` does not read or smoke `/opt/roehub/app` |
| `post_main_production_runtime_proof` | not collected; requires target revision on `main`, green CI/GitHub Actions, deploy or verified sync into `/opt/roehub/app`, and production runtime smoke from the synced runtime tree |

## Gate

| Check | Result |
|---|---|
| Stage-start ledger | `current_stage=08O` |
| Completion ledger | `current_stage=none`; no executable prompt is open until an `08P` prompt is generated and inserted |
| `08J` prerequisite | `accepted`; article dataset manifest sha256 `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| `08K` prerequisite | `blocked`; strict blockers `single_group_dominates_final_result`, `ticker_stability_obviously_broken` |
| `08L` prerequisite | `accepted`; research proxy only, `stage09_allowed=false` |
| `08M` prerequisite | `accepted`; fallback/context only for this stage, not the main object |
| `08N` prerequisite | `accepted`; `quality_status=accepted_for_research_only`; `stage19_mainnet_readiness_allowed=false` |
| Stage `17` prerequisite | `accepted` only as `infrastructure_only` |
| Stage `18` prerequisite | `accepted` only as `monitor_only_technical_soak` |
| Stage `19` state | `pending`, cannot run; `stage19_mainnet_readiness_allowed=false` |
| Prompt | `.codex/agents/generated/rl-trading-agent-platform-v1/08o-stage08k-dqn-forensic-decomposition.md` |
| Prompt sha256 | `86918e6ffba475256c97432a6322dd4fed4a8c96582334d635b1e6de62cea178` |

## Methodology

| Field | Value |
|---|---|
| Уровень глубины | `candidate_forensic_decomposition` |
| Тип задачи | PnL/stability/source-comparison forensic for blocked DQN candidate |
| Единица анализа | final holdout scorecard, selected session, closed trade, day, month, ticker group, volatility bucket and long/short side |
| Основные метрики | PnL after costs, return percent, closed trades, max drawdown, Sharpe/Sortino-like daily metrics, dominance shares, positive group ratios, action balance |
| Проверка качества данных | Prompt hash recorded; required `08K`/`08J` artifact hashes matched on `macstudio`; PnL recomputed independently from balance changes |
| Статус вывода | `accepted_for_forensic_research_only`; product/mainnet/registry path remains fail-closed |

## Verified Artifacts

| Artifact | sha256 |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/stage08k_dual_branch_cpu_76f51186c00ecb54255e/stage08k_dual_branch_cpu_run_summary.json` | `70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/final_holdout_b2adb7da3abc/stage08f_evaluation_manifest.json` | `c16a08dffa6f49fd8f24d9a1af6737114d54363b4cca3630a4e83c2661880b07` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/stage08k_optuna_summary.json` | `8585d4342dab24850cd077e5287de5faab251e848f18eb044f70cc410ebf6cec` |
| `/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json` | `fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/final_holdout_b2adb7da3abc/scorecards.json` | `df439376befd0bf7b4678584f75f9d2f1958298b0e365f4ee5db3a99d344c3a6` |
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08k_article_demo_profile_training_evaluation_v1/stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/final_holdout_b2adb7da3abc/filtered_backtest_balance_curve.json` | `e4902f21fdb15a4f93acda721779400e27d3665f9c2ae747cb19d980a2eb2eeb` |

Sanitized Stage `08O` summary:

| Field | Value |
|---|---|
| Path | `/opt/roehub/state/rl_trading/evaluation_runs/stage08o_stage08k_dqn_forensic_decomposition_v1/stage08o_stage08k_forensic_summary.json` |
| sha256 | `19a8d1f90f5d7630fb09eeef4c801eec11cdd98bffa172ce410fb79c071355b0` |
| Status | `accepted` |

## Math Reconciliation

| Metric | Value |
|---|---:|
| Starting equity | `10000.0` |
| Final shared balance | `22502.65333026` |
| Recomputed PnL from balance changes | `12502.65333026` |
| Scorecard PnL after costs | `12502.65333026` |
| Recomputed return after costs | `125.0265333026%` |
| Scorecard return after costs | `125.0265333%` |
| Closed trades from balance changes | `316` |
| Scorecard closed trades | `316` |
| Profitable trades | `186` |
| Win rate | `0.58860759` |
| Max drawdown from balance curve | `-19.44498897306477%`, trough `TSTUSDT` at `2025-05-30T17:22:00Z` |
| Scorecard max drawdown magnitude | `19.44498897%` |

Final holdout scope:

| Scope | Value |
|---|---:|
| Final split total sessions | `4162` |
| Final split selected sessions in `Optuna` summary | `4162` |
| Grouped filtered scorecard sessions | `3139` |
| Decision rows | `31390` |
| Period | `2025-03-01T02:39:00Z` to `2025-05-31T22:47:00Z` |

Interpretation: aggregate return is real in the artifact. The blocker is not a PnL arithmetic failure.

## Decomposition

Monthly official scorecard:

| Period | PnL | Closed trades | Profitable trades | Win rate |
|---|---:|---:|---:|---:|
| `2025-03` | `900.86596212` | `61` | `31` | `0.50819672` |
| `2025-04` | `5122.91251839` | `105` | `64` | `0.60952381` |
| `2025-05` | `6478.87484974` | `150` | `91` | `0.60666667` |

Top and bottom days from reconstructed trades:

| Rank | Best day | PnL | Trades | Worst day | PnL | Trades |
|---:|---|---:|---:|---|---:|---:|
| 1 | `2025-05-12` | `1684.10104196` | `8` | `2025-05-18` | `-1436.25724104` | `6` |
| 2 | `2025-05-14` | `1537.52750056` | `7` | `2025-05-16` | `-1396.71440804` | `9` |
| 3 | `2025-05-08` | `1419.24034634` | `14` | `2025-05-19` | `-1109.15061938` | `2` |
| 4 | `2025-05-10` | `1303.45287156` | `6` | `2025-04-18` | `-991.38480596` | `4` |
| 5 | `2025-05-07` | `986.59704709` | `4` | `2025-05-02` | `-962.55963126` | `5` |

Volatility buckets:

| Bucket | PnL | Closed trades | Sessions | Win rate |
|---|---:|---:|---:|---:|
| `high` | `12482.1319445` | `243` | `1046` | `0.6090535` |
| `low` | `307.01029846` | `24` | `1047` | `0.58333333` |
| `medium` | `-286.48891271` | `49` | `1046` | `0.48979592` |

Side decomposition from reconstructed closed trades:

| Side | PnL | Closed trades | Profitable trades | Win rate |
|---|---:|---:|---:|---:|
| `long` | `14932.7523078` | `253` | `156` | `0.6166007905` |
| `short` | `-2430.09897754` | `63` | `27` | `0.4285714286` |

Ticker decomposition:

| Top positive ticker | PnL | Closed trades | Sessions | Worst ticker | PnL | Closed trades | Sessions |
|---|---:|---:|---:|---|---:|---:|---:|
| `BROCCOLIF3BUSDT` | `1998.7718669` | `17` | `97` | `JELLYJELLYUSDT` | `-1463.8926806` | `9` | `61` |
| `BANANAS31USDT` | `1766.60438004` | `7` | `36` | `SIRENUSDT` | `-1160.64481486` | `11` | `50` |
| `SXTUSDT` | `1319.32064847` | `8` | `21` | `SFPUSDT` | `-1109.15061938` | `2` | `1` |
| `KERNELUSDT` | `1087.954987` | `8` | `15` | `BMTUSDT` | `-992.7925091` | `2` | `26` |
| `PORTALUSDT` | `994.50364901` | `4` | `5` | `DOLOUSDT` | `-905.53779394` | `4` | `11` |

Trade sequences:

| Sequence | Value |
|---|---|
| Best single trade | `BROCCOLIF3BUSDT` long, `2025-05-05T12:26:00Z`, `+684.72290904` |
| Worst single trade | `TSTUSDT` long, `2025-05-30T17:22:00Z`, `-1541.45964907` |
| Best contiguous sequence | trades `75`-`113`, `39` trades, `+7275.19798318`, `2025-04-01T03:00:00Z` to `2025-04-13T18:01:00Z` |
| Worst contiguous sequence | trades `257`-`283`, `27` trades, `-7212.76973382`, `2025-05-16T13:39:00Z` to `2025-05-24T04:56:00Z` |

## Root Cause

### `single_group_dominates_final_result`

Confirmed as real regime concentration, not a scorecard sign bug.

Formula reconstructed from official scorecard:

```text
abs(high bucket PnL) / (abs(high) + abs(low) + abs(medium))
= 12482.1319445 / 13075.63115567
= 0.954610281973835
```

Limit is `0.8`, so blocker is valid. The `high` bucket contributes almost all absolute bucket-level PnL. `medium` is negative and `low` is small positive. This supports only high-volatility-regime research; it does not support a stable global candidate.

### `ticker_stability_obviously_broken`

Confirmed as a full-universe coverage blocker, not an arithmetic bug.

| Ratio | Value |
|---|---:|
| Official all selected ticker groups | `70 / 291 = 0.24054982817869416` |
| Minimum | `0.25` |
| Flat/no-trade ticker groups | `179` |
| Active traded ticker groups only | `70 / 112 = 0.625` |

Interpretation: among tickers where the filter actually traded, the result is not obviously broken. The strict gate correctly fails closed because product/tariff-style user ticker coverage cannot ignore the many selected ticker groups where the policy produced no positive PnL or no trade.

## Scorecard Gate Bug Audit

| Check | Result |
|---|---|
| PnL/return reconciliation | passed; balance changes exactly reproduce `12502.65333026` and `125.0265333%` |
| Closed trade count | passed; `316` balance changes |
| Baseline beating | passed; best sanity baseline `0.0` |
| Monthly dominance | passed; `0.5181999915221556` |
| Volatility dominance | blocked; `0.954610281973835` |
| Ticker dominance | passed; `0.05630844156819175` for `BROCCOLIF3BUSDT` |
| Ticker positive group ratio | blocked; `0.24054982817869416` |
| Open-side dominance | passed; `0.8006329113924051`, long-heavy but below `0.95` |
| Side reconstruction | passed; `0` unknown closes |
| Verdict | `no_scorecard_gate_bug_found` |

## Article Comparison

Source references:

- Habr article: `https://habr.com/ru/articles/934258/`
- GitHub repo: `https://github.com/YuriyKolesnikov/rl-trading-binance`
- Source metric implementation: `https://raw.githubusercontent.com/YuriyKolesnikov/rl-trading-binance/main/backtest_engine.py`

Published article metrics available in the source text: return `+144.23%`, Sharpe `1.85`, Sortino `2.05`, signal accuracy `69.6%`, max drawdown `-22.49%`, `56` trading days, `44` profit days, `112` trades, average trade size `11324.29 USDT`, `~2.00` trades/day.

| Metric | Article/source | Roehub Stage `08K` artifact | Status |
|---|---:|---:|---|
| Return after costs | `144.23%` | `125.0265333%` | comparable aggregate only |
| Max drawdown | `-22.49%` | `-19.44498897%` | comparable from artifact |
| Sharpe | `1.85` | `2.511839549` source-formula-like from trade days | computed comparable proxy |
| Sortino | `2.05` | `3.091705983` source-formula-like from trade days | computed comparable proxy |
| Signal accuracy | `69.6%` | unavailable; artifact has win rate `58.860759%`, not source-style accuracy | unavailable |
| Profit days | `44 / 56 = 78.57%` | `56 / 81 = 69.135802%` | comparable from trade days |
| Trades/day | `~2.00` | `3.901234568` | materially different |
| Average trade size | `11324.29 USDT` | unavailable; artifact stores realized PnL/shared balance, not exact `trade_amount` | unavailable |
| Trade count | `112` | `316` | materially different |
| Equity-curve shape | article claims balanced risk-return | Roehub curve has real positive return but high-volatility concentration and a late `TSTUSDT` drawdown trough | not source-faithful |

Article similarity decision: `likely_aggregate_return_coincidence`. The aggregate return is close by one coarse metric, but trade count/cadence, unavailable source-style accuracy, no exact average trade size, high-volatility dominance and full-universe ticker instability prevent a source-faithful methodology claim.

## Allowlist And Regime Feasibility

| Feasibility surface | Decision |
|---|---|
| Restricted high-volatility research | possible |
| Per-ticker allowlist research | possible |
| Product allowlist | not allowed |
| Stage `09` for `08K` | not allowed |
| Stage `19+` | not allowed |

Seed tickers for future research-only allowlist/calibration investigation: `BROCCOLIF3BUSDT`, `BANANAS31USDT`, `SXTUSDT`, `KERNELUSDT`, `PORTALUSDT`, `AIOTUSDT`, `DOODUSDT`, `CTKUSDT`, `BANKUSDT`, `HYPERUSDT`.

This is a research seed only. It is not a runtime config, registry entry, entitlement, UI behavior or trading permission.

## Decision

| Field | Value |
|---|---|
| Stage `08O` status | `accepted` |
| `08k_forensic_status` | `per_ticker_per_regime_calibration_candidate` |
| `article_similarity_status` | `likely_aggregate_return_coincidence` |
| `stage09_for_08k_allowed` | `false` |
| `stage19_mainnet_readiness_allowed` | `false` |
| `stage20_mainnet_canary_allowed` | `false` |
| `stage21_product_rollout_allowed` | `false` |
| `08p_allowed` | `true` for research-only prompt-pack insertion, not as an already runnable prompt |
| `next_prompt` | `08p-stage08k-per-ticker-regime-calibration.md` is not generated by Stage `08O`; it requires explicit prompt-pack insertion before execution |

Business interpretation: `08K` is no longer just "positive PnL but blocked with no explanation". It has a real high-volatility/per-ticker research signal. It still fails the strict full-universe stability gate and cannot move to registry, runtime, product or mainnet. The next useful work is a separate research-only calibration prompt, not Stage `09` or Stage `19`.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No route, DTO or browser behavior changed. |
| Port contract | `none` | No application port/interface changed. |
| DTO schema | `none` | No wire DTO changed. |
| Persisted schema | `none` | No DB migration/table/storage schema changed. |
| Config schema/defaults | `none` | No runtime config/default changed. |
| Request hash / cache key / persistence identity | `none` | No production request/cache identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No service calls changed. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange/provider, paper/testnet/live/mainnet, Redis dispatch or DB side effect. |
| Logs/metrics/traces/audit/ledger/report/redaction | `compatible-change` | Adds sanitized `08O` summary/report and ledger facts; no secrets/raw provider payloads. |
| Benchmark / rollout gates | `compatible-change` | `08K` research classification changes, while `stage09_for_08k_allowed=false` and Stage `19+` remain closed. |
| Browser-visible behavior | `none` | Browser/auth is out of scope. |
| Performance hot path | `none` | Offline artifact parser only; no runtime hot path touched. |

## File Manifest

Repository files:

| Path | State | Reason | Contract impact |
|---|---|---|---|
| `scripts/rl_trading/stage08o_stage08k_dqn_forensic_decomposition.py` | created | Deterministic Stage `08O` artifact parser and sanitized summary writer. | `compatible-change` additive offline research CLI |
| `tests/unit/scripts/rl_trading/test_stage08o_stage08k_dqn_forensic_decomposition.py` | created | Focused coverage for trade reconstruction, volatility dominance and positive-ratio denominator semantics. | `none` test-only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08o-stage08k-dqn-forensic-decomposition.md` | created | Stage `08O` accepted report and handoff. | `compatible-change` docs/report |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | modified | Mark Stage `08O` accepted, close current executable prompt and keep Stage `19+` blocked. | `compatible-change` docs/ledger |
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | modified | Sync plan narrative/stage table with accepted `08O` and research-only `08P` handoff. | `compatible-change` docs/plan |
| `docs/architecture/README.md` | modified | Docs index sync after adding Stage `08O` report. | `compatible-change` docs index |

Runtime artifacts outside git:

| Path | State |
|---|---|
| `/opt/roehub/state/rl_trading/evaluation_runs/stage08o_stage08k_dqn_forensic_decomposition_v1/stage08o_stage08k_forensic_summary.json` | created on `macstudio`; sanitized summary sha256 `19a8d1f90f5d7630fb09eeef4c801eec11cdd98bffa172ce410fb79c071355b0` |

Foreign/pre-existing changes intentionally preserved and excluded from ownership until this stage edited them: the pre-existing `08O` prompt insertion changes in `docs/architecture/ml/rl-trading-agent-platform-v1.md`, the pre-existing `08O` insertion changes in this ledger, and untracked `.codex/agents/generated/rl-trading-agent-platform-v1/08o-stage08k-dqn-forensic-decomposition.md`.

## Quality Gates

| Gate | Result |
|---|---|
| Focused pytest | passed: `uv run pytest -q tests/unit/scripts/rl_trading/test_stage08o_stage08k_dqn_forensic_decomposition.py` -> `3 passed` |
| Focused ruff | passed: `uv run ruff check scripts/rl_trading/stage08o_stage08k_dqn_forensic_decomposition.py tests/unit/scripts/rl_trading/test_stage08o_stage08k_dqn_forensic_decomposition.py` |
| Focused pyright | passed: `uv run pyright scripts/rl_trading/stage08o_stage08k_dqn_forensic_decomposition.py tests/unit/scripts/rl_trading/test_stage08o_stage08k_dqn_forensic_decomposition.py` -> `0 errors` |
| Mac Studio summary generation | passed; summary sha256 `19a8d1f90f5d7630fb09eeef4c801eec11cdd98bffa172ce410fb79c071355b0` |
| Prompt-level docs index | passed: `uv run python -m tools.docs.generate_docs_index --check` |
| `git diff --check` | passed |
| Prompt-level ruff | passed: `uv run ruff check scripts/rl_trading tests/unit/scripts/rl_trading` |
| Prompt-level pyright | passed: `uv run pyright scripts/rl_trading tests/unit/scripts/rl_trading` -> `0 errors` |
| Prompt-level pytest | passed: `uv run pytest -q tests/unit/scripts/rl_trading` -> `47 passed` |

## Residual Risks

- Stage `08O` does not create the future `08P` prompt. `08p_allowed=true` means "research-only prompt-pack insertion is justified", not that a runnable stage artifact already exists.
- Source-style signal accuracy and exact average trade size are unavailable in Stage `08K` artifacts; this blocks a fully source-faithful article-reproduction claim.
- High-volatility regime concentration remains severe. A future calibration stage must prove whether this is a usable restricted regime or overfit.
- Stage `19+` remains closed by Stage `08N` and by this report. No product/mainnet claim is made.

## Cold-Head Review

Cold-head review: completed.
Mode: cold self-review fallback. Independent subagent review was not used because current multi-agent tool policy requires explicit user request for delegation.
Review scope: Stage `08O` helper, sanitized summary, report, ledger/plan handoff, proof-boundary wording, source comparison and downstream gates.
Verdict: accepted after fixing proof-boundary wording and replacing stale pending verification rows with actual gate results.
