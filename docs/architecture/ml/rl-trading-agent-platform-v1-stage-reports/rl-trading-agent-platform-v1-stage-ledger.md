# RL Trading Agent Platform v1 — журнал выполнения stages

Единый handoff-документ для плана `docs/architecture/ml/rl-trading-agent-platform-v1.md`.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/ml/rl-trading-agent-platform-v1.md` |
| `ledger_status` | `active` |
| `current_stage` | `02A` |
| `updated_at` | `2026-06-17` |
| `owner` | `Roehub agents / implementation executors` |

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage обновляет этот ledger после validation и до финального отчета. |
| Источник фактов | Писать только проверенные факты: tests, runtime calls, DB evidence, browser QA, CI, benchmark, deploy/smoke или явно помеченные blockers. |
| Статусы | Использовать только `pending`, `in_progress`, `accepted`, `blocked`, `skipped`, `superseded`. |
| Tests не acceptance | Non-trivial stage accepted только после real-boundary/e2e evidence по затронутой поверхности. |
| Секреты | Не писать secrets, tokens, cookies, passphrases, ciphertext, raw provider payloads, HMAC, API keys или credentials. |
| Large artifacts | Не коммитить datasets/checkpoints/log dumps; хранить runtime artifacts под `/opt/roehub/state/rl_trading/` и писать в docs только sanitized summary/hash. |
| Classic producer dependency | RL paper/testnet/live stages не стартуют, пока нужные stages `strategy-producer-paper-testnet-trading-v1` не accepted. |
| Mac Studio | Git на `macstudio` только в `/Users/daniildegtyarev/Projects/roehub.com`; runtime checks в `/opt/roehub/app`; ML artifacts в `/opt/roehub/state/rl_trading/`. |
| Mainnet | Mainnet submit blocked до Stage `19` readiness review и отдельного explicit approval. |
| File manifest | Каждый stage report обязан фиксировать `Created / Modified / Deleted / Reason / Contract impact`. |
| GitHub delivery | По умолчанию не делать direct `git push origin main`. Если stage публикуется, использовать `github:yeet`: scoped staging, одна `codex/*` branch только при старте с default branch, draft PR, `delivered-to-main` только после merge/evidence, временную `codex/*` branch удалить после успешного PR/test/deploy path. |

## Stage Status

| Stage | Статус | Stage report | Validation depth | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|
| `01` Baseline and plan freeze | accepted | `01-baseline-plan-freeze.md` | docs + DB snapshot + archival prompt repair | Plan/ledger created; ClickHouse feature snapshot observed; classic producer dependency recorded; docs index passed; archival prompt path/hash recorded without implementation/runtime changes. | none | yes |
| `02A` Data source inventory | pending | `02a-data-source-inventory.md` | ClickHouse + HF + artifact manifest + lifecycle/gap/classic blocker inventory | Prompt pack generated under `.codex/agents/generated/rl-trading-agent-platform-v1/`; cold-head High finding fixed; implementation not started. | none; implementation pending | no until Stage `02A` accepted |
| `02B` Feature and live-feed contract | pending | `02b-feature-live-feed-contract.md` | feature contract + live-feed parity decision + futures metadata gate | TBD | Stage `02A` required | no |
| `02C` Action/state/reward contract | pending | `02c-action-state-reward-contract.md` | action/reward/state fixtures + strategy ownership semantics | TBD | Stage `02B` required | no |
| `03` Mac Studio ML environment | pending | `03-mac-studio-ml-environment.md` | target runtime + benchmark + resource isolation | TBD | Stage `02C` required | no |
| `04` External repo/HF reproducibility | pending | `04-hf-reproducibility.md` | dataset + training smoke | TBD | TBD | no |
| `05` Roehub dataset builder v1 | pending | `05-roehub-dataset-builder-v1.md` | raw feature slabs + manifests + feature parity fixtures | TBD | TBD | no |
| `06` Dataset QA and session extractor | pending | `06-dataset-qa-session-extractor.md` | accepted sessionized datasets + leak/gap/split/overlap checks | TBD | TBD | no |
| `07` D3QN/PER training runner | pending | `07-d3qn-per-training-runner.md` | training + perf evidence | TBD | TBD | no |
| `08` Roehub backtest/evaluation harness | pending | `08-roehub-backtest-evaluation.md` | scorecard + sanity baselines | TBD | TBD | no |
| `09` Model registry and activation gates | pending | `09-model-registry-activation.md` | persistence + state machine invariants + hash validation + artifact ops + checkpoint security | TBD | TBD | no |
| `09B` Local artifact backup and restore drill | pending | `09b-local-artifact-backup-restore.md` | accepted artifact backup + registry dump + restore drill | TBD | Stage `09` required | no |
| `10` Per-ticker calibration | pending | `10-per-ticker-calibration.md` | calibration report | TBD | Stage `09B` required | no |
| `10A` Retraining and promotion lifecycle | pending | `10a-retraining-promotion-lifecycle.md` | candidate/champion + numeric promotion profile + schedule/manual triggers + host-local rollback | TBD | Stage `09B` and `10` required | no |
| `11` RL tab UI skeleton | pending | `11-rl-tab-ui-skeleton.md` | browser + API + operator rollback UI + reusable signal/outcome read model | TBD | Stage `10A` required for action controls | no |
| `12` Backend entitlements | pending | `12-backend-entitlements.md` | API/DB/browser | TBD | TBD | no |
| `13` Monitor-only inference producer | pending | `13-monitor-only-inference-producer.md` | runtime + source events + Redis/live feature parity | TBD | Stage `12` required | no |
| `14` User risk/sizing policy | pending | `14-user-risk-sizing-policy.md` | API/UI/domain + synthetic exits | TBD | Stage `13` required | no |
| `15` Paper RL integration | pending | `15-paper-rl-integration.md` | paper execution ledger + simulator/paper accounting parity | TBD | classic producer Stage `07` required | no |
| `16` Testnet RL integration | pending | `16-testnet-rl-integration.md` | real testnet exchange | TBD | classic producer Stage `09` required | no |
| `17` Multi-ticker runtime/load gate | pending | `17-multi-ticker-runtime-load.md` | load + metrics | TBD | TBD | no |
| `18` 24h/7d RL soak and incident drills | pending | `18-rl-soak-incident-drills.md` | 24h/7d runtime + kill-switch/rollback/stale-feed drills | TBD | TBD | no |
| `19` Mainnet readiness architecture review | pending | `19-mainnet-readiness-review.md` | architecture + ops + product/legal/support review | TBD | explicit approval required after review | no |
| `20` Bounded mainnet canary | pending | `20-bounded-mainnet-canary.md` | real-money canary | TBD | Stage `19` approval required | no |
| `21` Product rollout | pending | `21-product-rollout.md` | rollout metrics + support/legal/backup gates | TBD | Stage `20` accepted and backup/support/legal/product gates closed | no |
| `22` Final docs/prompt closure | pending | `22-final-docs-prompt-closure.md` | docs + delivery | TBD | TBD | no |

## Что Обязательно Знать Дальше

| Fact / decision | Why it matters | Evidence |
|---|---|---|
| RL execution must reuse existing `live_execution`/`exchange-execution`; ML worker never calls exchange SDK directly. | Prevents duplicate money boundary and secret leakage. | `docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md`; `docs/architecture/ml/rl-trading-agent-platform-v1.md` |
| Classic strategy producer is currently `current_stage=05`, and Stage `05` is blocked on Binance futures testnet credential custody (`legacy/non-Transit ciphertexts`, fail-closed exchange-control); RL execution still cannot assume paper/testnet foundation until classic Stage `07`/`09` are accepted after Stage `05` repair. | Avoids hidden dependency on a blocked execution foundation. | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` |
| Article-compatible feature set is 7 columns; current backtest artifact `ohlcv` is 5 columns. | Dataset builder must augment/enrich before full model training. | `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`; ClickHouse schema snapshot |
| Binance canonical rows currently have `trades_count`; Bybit canonical rows currently do not. | Bybit activation needs enrich or feature-mask branch. | Mac Studio ClickHouse schema/coverage queries, 2026-06-17 |
| Stage `02` is split into `02A` data inventory, `02B` feature/live-feed contract and `02C` action/state/reward contract. | Prevents one oversized acceptance gate from hiding data, feature and RL-environment decisions. | `rl-trading-agent-platform-v1.md` |
| Stage `02B` must output a Binance/Bybit × spot/futures activation matrix and a live-feed `trades_count|feature-mask|blocked` decision. | Prevents accidental activation of markets with incomplete feature/funding/contract/live-feed coverage. | `rl-trading-agent-platform-v1.md`; Redis live candle payload currently publishes OHLCV and `volume_quote`, not `trades_count` |
| Roehub action contract is `0 hold`, `1 open_long`, `2 open_short`, `3 close`; close is scoped to the owning RL strategy run only. | Prevents cross-strategy position interference and preserves multiple strategies on the same ticker. | User decision, 2026-06-17; external repo `trading_environment.py`; `rl-trading-agent-platform-v1.md` |
| Training reward v1 mirrors the external repo: realized PnL delta over initial balance minus flat-hold inaction penalty; no mark-to-market reward rewrite. | Prevents accidental behavior change during first port. | external repo `trading_environment.py`; user decision, 2026-06-17 |
| Train/live feature parity requires shared feature builder and golden fixtures. | Prevents model training on one representation and inference on another. | `rl-trading-agent-platform-v1.md` |
| Stage `13` monitor-only cannot be accepted unless Redis/live feature window and offline dataset fixture produce identical feature vectors for the same candle window. | Prevents heavy ClickHouse repair or schema drift from entering the live inference hot path. | `rl-trading-agent-platform-v1.md`; `redis_streams_live_candle_publisher.py` |
| Overlap is allowed inside a split only; split boundaries require time embargo and lifecycle-aware leakage proof. | Keeps session extraction sample count while blocking leakage. | User decision, 2026-06-17; `rl-trading-agent-platform-v1.md` |
| Stage `08` research evaluation requires scorecard structure and sanity baseline artifacts before any candidate is saved. | Prevents accepting simulator/reward bugs as profitable candidates. | User decision, 2026-06-17; `rl-trading-agent-platform-v1.md` |
| Stage `08` may accept only a research candidate; Stage `10A` must create a promotion-grade numeric threshold profile before paper/testnet/live progression. | Keeps research-positive PnL from becoming production approval. | `rl-trading-agent-platform-v1.md` |
| Stage `09` owns registry state machine invariant tests, and Stage `09B` owns local backup/restore drill. | Prevents runtime activation from depending on missing/corrupt artifacts without restore/rollback evidence. | `rl-trading-agent-platform-v1.md` |
| Checkpoints are trusted local artifacts only; no user upload. | Keeps PyTorch checkpoint loading inside Roehub-controlled trust boundary. | User decision, 2026-06-17; `rl-trading-agent-platform-v1.md` |
| Retraining supports manual/scheduled host-local backend triggers immediately; RL/ML UI controls are delivered in Stage `11`; drift may create/run candidate training but never auto-promotes. | Supports production lifecycle without in-place live mutation and prevents Stage `10A` from depending on UI that does not exist yet. | User decision, 2026-06-17; `rl-trading-agent-platform-v1.md` |
| Stage `05` emits raw feature slabs/manifests/golden fixtures only; Stage `06` emits accepted sessionized train/val/test/backtest datasets. | Prevents training on final-looking artifacts before session extraction, leakage controls and split policy are accepted. | `rl-trading-agent-platform-v1.md` |
| Model retrain/promote/rollback controls start as host-local operator commands; web UI action controls require server-side operator/admin authorization in Stage `11`. | Prevents exposing platform model lifecycle controls through ordinary owner-scoped user routes. | `rl-trading-agent-platform-v1.md` |
| RL signals must extend the reusable strategy signal/outcome read model for `source_type=ml_agent_decision`. | Keeps future Telegram/notification delivery reusable instead of building an RL-only signal journal. | `rl-trading-agent-platform-v1.md`; existing `/strategies` outcome/read-model pattern |
| RL v1 TP/SL/trailing are synthetic platform-side exits, not native exchange OCO/TP/SL/trailing order fields. | Current live-execution order model rejects advanced fields; RL must create separate close intents through existing risk/execution path. | `rl-trading-agent-platform-v1.md`; `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/10-execution-source-intent-order-model.md` |
| User-specific paper/testnet/live outcomes are monitoring/evaluation data only in v1, not platform-wide retraining input. | Prevents user outcome data from crossing privacy/product governance boundaries without consent/redaction/lineage policy. | `rl-trading-agent-platform-v1.md` |
| Stage `18` must prove safe-mode/incident drills before Stage `19`; Stage `19` must include product/legal/support go/no-go. | Mainnet readiness requires operational and product controls, not only model metrics. | `rl-trading-agent-platform-v1.md` |
| Active ticker quota counts only `live` RL tickers, not monitor/paper/testnet. | Defines backend entitlement contract and UI slot counts. | User requirement, 2026-06-17 |
| Model is platform-wide, but per-ticker calibration is expected. | Avoids training one full model per user while preserving ticker-specific behavior. | User requirement, 2026-06-17 |
| Safety boundary must not add material live-execution latency. | Stage `13`/`16`/`17` must measure segment-level p95 latency and optimize hot-path blocking work instead of removing audit/risk gates. | User requirement, 2026-06-17; `rl-trading-agent-platform-v1.md` |
| Training/retraining is platform-owned and offline; users only activate eligible tickers by tariff. | Prevents user-controlled model mutation and makes model promotion auditable. | User requirement, 2026-06-17; `rl-trading-agent-platform-v1.md` |
| Futures activation requires funding/fee/slippage/contract metadata coverage or an explicit accepted approximation. | Prevents false-positive futures backtests. | `rl-trading-agent-platform-v1.md` |
| Runtime/code stage acceptance must record prompt path/hash and delivery state. | Prevents local-only work from being mistaken for delivered Mac Studio production state. | `.codex/PLANS.md`; `.codex/AGENTS.md`; `rl-trading-agent-platform-v1.md` |
| Executor prompt pack is generated and cold-reviewed for Stage `01` repair plus Stages `02A`-`22`. Cold-head initially blocked on ambiguous worker topology; prompts now use planned `apps/worker/rl_trading_trainer/` and `apps/worker/rl_trading_inference/` paths. | Stage `02A` can start from its prompt; later stages still depend on their ledger prerequisites and evidence gates. | `.codex/agents/generated/rl-trading-agent-platform-v1/`; cold-head review `019ed68b-d9ad-7ba1-9886-7eab1ae5337c`; local follow-up check |
| Current identity plan codes are `base|free|pro|ultra`, while product labels requested for RL are `Free|Pro|Premium|Enterprise`. | Stage `12` must map codes to RL entitlements explicitly and keep ambiguous `base` fail-closed until product evidence says otherwise. | `src/trading/shared_kernel/primitives/paid_level.py`; `src/trading/contexts/identity/application/use_cases/account_settings.py`; `rl-trading-agent-platform-v1.md` |

## Contract Impact Summary

| Surface | Expected classification | Notes |
|---|---|---|
| Public API | `compatible-change` | New RL/ML read models, active ticker config, model summaries. |
| Ports | `compatible-change` | New `rl_trading` ports and `live_execution` ACL for ML decisions. |
| DTO schema | `compatible-change` | Additive RL strategy/ticker/model DTOs. |
| Persisted schema | `compatible-change` | Additive metadata tables for models, datasets, calibrations, active ticker slots. |
| Config schema/defaults | `compatible-change` | New optional ML dependency group/env and service configs; no torch in API runtime. |
| External side effects | `compatible-change` until mainnet | Source events/paper/testnet via existing execution; mainnet requires Stage `19` approval. |
| Browser-visible behavior | `compatible-change` | New RL/ML tab on `/strategies`; no existing tab removal. |
| Performance | `unknown` until Stage `03`/`07` | Mac Studio MPS/CPU benchmarks required. |

## Checks And Evidence

| Stage | Local gates | Real-boundary / e2e evidence | Result | Evidence path |
|---|---|---|---|---|
| `01` | `python -m tools.docs.generate_docs_index --check` passed | ClickHouse schema/coverage queried on Mac Studio; plan/ledger created | accepted | `01-baseline-plan-freeze.md` |
| `02A` | TBD | ClickHouse + HF inventory + artifact manifests + lifecycle/gap report | TBD | Stage report |
| `02B` | TBD | Feature contract + Redis/live-feed parity decision + futures metadata gate | TBD | Stage report |
| `02C` | TBD | Action/state/reward fixtures and ownership contract | TBD | Stage report |
| `03` | TBD | Mac Studio PyTorch CPU/MPS smoke | TBD | Stage report |
| `04`-`22` | TBD | See plan stage acceptance | TBD | Stage reports |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence | Next stage allowed |
|---|---|---|---|---|---|
| `15`,`16` | Classic producer Stage `05` is blocked on Binance futures testnet credential custody (`legacy/non-Transit ciphertexts`, fail-closed exchange-control); RL execution remains blocked even before the later Stage `07`/`09` gates. | High | Complete/unblock strategy producer Stage `05`, then continue classic Stage `07`/`09`. | TBD | no |
| `15` | Classic producer Stage `07` must be accepted before RL paper integration acceptance. | High | Complete/unblock strategy producer plan. | TBD | no |
| `16` | Classic producer Stage `09` must be accepted before RL testnet integration acceptance. | High | Complete/unblock strategy producer plan. | TBD | no |
| `20` | Explicit mainnet approval required after Stage `19`. | Blocker | User/operator approval through separate readiness review. | TBD | no |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| 2026-06-17 | plan | Created separate RL Trading Agent Platform v1 plan and ledger. | `docs/architecture/ml/rl-trading-agent-platform-v1.md` |
| 2026-06-17 | `01` | Accepted Stage `01` after docs index regeneration/check and cold self-review fallback. | `01-baseline-plan-freeze.md`; `docs/architecture/README.md` |
| 2026-06-17 | review | Tightened plan/ledger after explicit architecture review request: retraining lifecycle, futures data contract, prompt/delivery handoff, entitlement mapping and current classic producer status. | `docs/architecture/ml/rl-trading-agent-platform-v1.md`; this ledger |
| 2026-06-17 | plan | Added full lifecycle contracts: action/reward/state, train/live feature parity, session extraction, promotion scorecard, sanity baselines, artifact operations, checkpoint security, retraining triggers/cadence and staged rollback controls. | `docs/architecture/ml/rl-trading-agent-platform-v1.md`; this ledger |
| 2026-06-17 | review | Closed cold-head review findings: classic producer status refreshed to Stage `04`, next data/feature prompt gate made blocking, Stage `05`/`06` dataset ownership split, operator authority grounded, reusable signal/outcome read model required. | `docs/architecture/ml/rl-trading-agent-platform-v1.md`; this ledger |
| 2026-06-17 | plan hardening | Integrated external completeness review gaps: split Stage `02` into `02A/02B/02C`, recorded classic Stage `05` blocker, added live-feed feature gate, futures metadata gate, registry state machine, Stage `09B` backup/restore, promotion-grade thresholds, synthetic exits, simulator/paper parity, resource isolation, incident drills, live-outcome governance and product/legal mainnet gate. | `docs/architecture/ml/rl-trading-agent-platform-v1.md`; this ledger |
| 2026-06-17 | prompt pack | Generated Stage `01` repair prompt and Stage `02A`-`22` implementation prompts, ran one independent cold-head review, fixed the High worker-topology mismatch, and completed local follow-up checks. | `.codex/agents/generated/rl-trading-agent-platform-v1/`; this ledger |
| 2026-06-17 | prompt pack hardening | Added explicit GitHub delivery and branch-hygiene contract to prompts: use `github:yeet`, avoid direct main push, avoid unnecessary branches, and clean temporary `codex/*` branches after successful PR path. | `.codex/agents/generated/rl-trading-agent-platform-v1/`; this ledger |
| 2026-06-17 | prompt pack hardening | Ran second cold-head review for GitHub delivery semantics; no blockers found, and low wording gaps were closed for `publish-ci-deploy` scope and `published-to-branch/draft-pr` delivery state. | `.codex/agents/generated/rl-trading-agent-platform-v1/`; cold-head review `019ed6eb-0f3d-7073-bf62-576ec53566cb`; this ledger |
| 2026-06-17 | `01` archival repair | Confirmed Stage `01` was already accepted; recorded repair prompt path/hash and `local-only` delivery state without changing code, schemas, API, UI, runtime services, exchange paths, or ML artifacts. | `01-baseline-plan-freeze.md`; `.codex/agents/generated/rl-trading-agent-platform-v1/01-baseline-plan-freeze.md`; this ledger |
