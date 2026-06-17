---
doc: rl-trading-agent-platform-v1-stage-01-baseline-plan-freeze
stage: "01"
status: accepted
plan: docs/architecture/ml/rl-trading-agent-platform-v1.md
ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
collected_at: "2026-06-17"
---

# Stage 01: Baseline And Plan Freeze

Статус: `accepted`.

Stage `01` создает source-of-truth архитектурный план и ledger для RL Trading Agent Platform v1. Реализация кода, схем, API, UI, runtime services и ML artifacts в этом stage не выполняется.

User required before start: nothing.

Archival repair prompt precondition: User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat.

## Scope

Входит:

- зафиксировать пользовательские решения по Binance/Bybit, spot/futures, тарифным live ticker slots, Mac Studio-only runtime, platform-wide model и per-ticker calibration;
- учесть dependency на `strategy-producer-paper-testnet-trading-v1.md`;
- записать текущий feature/data snapshot из ClickHouse и artifact loader;
- создать plan и ledger.

Не входит:

- установка PyTorch;
- скачивание HF dataset;
- обучение модели;
- изменение API/UI;
- изменение execution/risk gate;
- запуск paper/testnet/mainnet.

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/ml/rl-trading-agent-platform-v1.md` | - | - | Source-of-truth architecture and implementation plan for RL trading agent. | `compatible-change` docs only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` | - | - | Stage ledger for RL trading rollout. | `compatible-change` docs/ledger only |
| `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md` | - | - | Stage `01` report. | `compatible-change` docs/report only |
| - | `docs/architecture/README.md` | - | Docs index update after adding architecture docs. | `compatible-change` docs index only |

## Archival Repair Prompt Evidence

Stage `01` was already `accepted` in the ledger when prompt `01-baseline-plan-freeze` was executed. This prompt is archival and repair-only; accepted plan/report content was not rewritten, and no implementation/runtime change was made.

| Field | Value |
|---|---|
| Prompt path | `.codex/agents/generated/rl-trading-agent-platform-v1/01-baseline-plan-freeze.md` |
| Prompt sha256 | `af0af5022fce926daed78d2a1c5390e6963931272d92c7e08c9519063d9aedc9` |
| Ledger state observed before repair | Stage `01` accepted; `current_stage=02A`; Stage `02A` pending |
| Planned concrete file list before edit | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md`; `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` |
| Created | none |
| Modified | `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md`; `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` |
| Deleted | none |
| Outside expected paths | none |
| Delivery state | `local-only`; no branch, PR, main delivery, deploy, runtime, schema, API, UI, service, exchange, or ML artifact change |
| Validation after repair | `python -m tools.docs.generate_docs_index --check` passed; `docs/architecture/README.md` is up-to-date |
| Cold self-review after repair | `Release`; local fallback used because subagent spawning requires an explicit user request; no Blocker/High findings |

## Observed State

| Area | Evidence summary |
|---|---|
| Classic strategy producer | `strategy-producer-paper-testnet-trading-v1` ledger is accepted through Stage `04`, but Stage `05` is now blocked on Binance futures testnet credential custody (`legacy/non-Transit ciphertexts`, fail-closed exchange-control). RL execution stages still must not assume paper/testnet foundation before classic Stage `07`/`09` after Stage `05` repair. |
| Backtest artifacts | `FilesystemBacktestArtifactArrayLoader.load_price_arrays` validates `ohlcv` as `float32` and exactly five columns. |
| ClickHouse canonical schema | `market_data.canonical_candles_1m` has `open/high/low/close`, `volume_base`, `volume_quote`, `trades_count`, taker volumes, source and ingestion metadata. |
| Binance data | Current coverage query showed Binance spot/futures rows have `volume_quote` and `trades_count`. |
| Bybit data | Current coverage query showed Bybit spot/futures rows have `volume_quote`, but `trades_count_rows=0`. |
| Entitlement baseline | Existing account limits are hardcoded in `AccountSettingsUseCase.get_limits`; full billing is not implemented and not required for v1. |

## Contract Impact

| Surface | Classification | Reason |
|---|---|---|
| Public API contract | `none` | No API changed in this stage. |
| Port contract | `none` | No interfaces changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or storage change. |
| Config schema/defaults | `none` | No env/YAML/default change. |
| Request hash / cache key / persistence identity | `none` | No runtime identity changed. |
| Service-call auth/timeout/retry/error semantics | `none` | Read-only evidence only. |
| External side-effect/idempotency/unknown-state semantics | `none` | No exchange/provider side effect. |
| Logs/metrics/traces/audit/ledger/report semantics | `compatible-change` | Adds documentation/ledger semantics only. |
| Alert/runbook semantics | `none` | No monitoring config changed. |
| Browser-visible behavior | `none` | No UI changed. |

## Quality Gates

| Gate | Result |
|---|---|
| `python -m tools.docs.generate_docs_index --check` | passed after `python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md` |

## Cold Self-Review

Mode: `cold self-review fallback`.

Final result: `Release`.

Findings resolved before Stage `01` acceptance:

- update `docs/architecture/README.md` via docs index generator;
- record final docs index check result;
- update ledger Stage `01` status to `accepted` after docs gate passes.

## Post-review Hardening

After the explicit architecture-review request on 2026-06-17, an independent read-only subagent returned `Release after fixes`. The plan and ledger were tightened without code changes:

- added platform-owned retraining/fine-tuning, candidate/champion promotion, drift trigger and rollback requirements;
- made futures funding/fee/slippage/contract metadata an explicit dataset/backtest prerequisite;
- changed Stage `05` wording from Binance-only initial dataset to Binance/Bybit spot/futures branches with incomplete branches explicitly blocked;
- added executor prompt path/hash and delivery-state requirements for future implementation stages.
- reconciled RL tariff labels with current identity `paid_level` codes `base|free|pro|ultra` and backend override semantics;
- refreshed current classic strategy producer status from accepted-through-Stage-`04` ledger evidence.
- added full lifecycle contracts from follow-up decisions: action/reward/state, strategy-scoped close ownership, feature parity, session extraction, promotion scorecard, sanity baselines, artifact operations, checkpoint security, retraining cadence and staged rollback controls.
- closed cold-head review findings after a read-only subagent pass: next data/feature prompt gate is blocking, Stage `05` owns raw feature slabs only, Stage `06` owns accepted sessionized datasets, model lifecycle controls require host-local command first and server-side operator/admin guard before UI actions, and RL signals extend the reusable strategy signal/outcome read model.
- integrated external completeness review gaps: split Stage `02` into `02A/02B/02C`, recorded the classic Stage `05` blocker, made live-feed feature parity and futures metadata explicit gates, added registry state machine, Stage `09B` backup/restore, promotion-grade thresholds, synthetic exits, simulator/paper parity, Mac Studio resource isolation, incident drills, live-outcome governance and product/legal mainnet gate.

## Next-Stage Handoff

Stage `02A` starts from these facts:

- article-compatible feature set requires 7 features;
- current Roehub artifacts provide 5 OHLCV fields;
- ClickHouse can provide `vwap` from `volume_quote / volume_base`;
- Bybit `trades_count` is currently missing and must be resolved by enrich or feature-mask decision;
- Stage `02A` inventories data sources, market coverage, lifecycle and gaps;
- Stage `02B` must produce Binance/Bybit × spot/futures activation matrix: `trainable`, `blocked`, `feature-mask`, or `research_only_approximation`, plus live-feed `trades_count|feature-mask|blocked` decision and futures metadata gate;
- Stage `02C` must freeze Roehub action/reward/state contract before implementation; `close` is scoped to the owning RL strategy run, and multiple strategies may trade the same ticker independently;
- Stage `02A` cannot start implementation until its executor prompt exists and passes cold-head review;
- Stage `05` emits raw feature slabs/manifests/golden fixtures only; Stage `06` emits accepted sessionized train/val/test/backtest datasets;
- Stage `09B` must prove local backup/restore drill before runtime activation;
- `paper`/`testnet`/`live` execution stages depend on the separate classic strategy producer plan repairing Stage `05` and reaching its required Stage `07`/`09` gates;
- Stage `18` must prove safe-mode/incident drills before Stage `19`; Stage `19` must include product/legal/support go/no-go;
- Stage `12` must map current identity codes `base|free|pro|ultra` to RL entitlements explicitly; `Enterprise` is an RL override, not a current identity enum.
