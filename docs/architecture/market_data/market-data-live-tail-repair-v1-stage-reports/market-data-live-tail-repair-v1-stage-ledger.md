# Market Data Live Tail Repair v1 — журнал выполнения stages

Единый handoff-документ для плана `docs/architecture/market_data/market-data-live-tail-repair-v1.md`.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/market_data/market-data-live-tail-repair-v1.md` |
| `prompt_pack` | `.codex/agents/generated/market-data-live-tail-repair-v1/` |
| `ledger_status` | `active` |
| `current_stage` | `02` |
| `created_at` | `2026-06-29` |
| `owner` | `Roehub agents / implementation executors` |

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage обновляет этот документ после validation и до финального отчета. |
| Источник фактов | Писать только проверенные факты: tests, Redis/DB/API/runtime calls, metrics, CI/deploy, Mac Studio proof или явно помеченные blockers. |
| Статусы | Использовать только `pending`, `in_progress`, `accepted`, `blocked`, `skipped`, `superseded`. |
| Tests не acceptance | Unit/integration tests, lint, type checks и docs check обязательны как gates, но non-trivial stage accepted только после real-boundary evidence. |
| Секреты | Не писать secrets, tokens, cookies, passphrases, DSNs, raw provider payloads, API keys или credentials. |
| Следующий stage | Заполнять handoff так, чтобы следующий executor не перечитывал весь чат. |
| Branch lifecycle | По умолчанию ветки не создаются. Branch/PR/worktree/stash только по прямой просьбе пользователя. |
| Main delivery | Accepted stage требует delivery evidence в `origin/main` или явный blocked/deferred publish record. |
| Mac Studio | Git-команды на `macstudio` только в `/Users/daniildegtyarev/Projects/roehub.com`; pre-main checks may only be `target_host_readiness_pre_main` or `read_only_existing_runtime_smoke`. `post_main_production_runtime_proof` requires the changed revision on `main`, green GitHub Actions/CI, deploy/sync into `/opt/roehub/app`, then runtime smoke/proof. |
| File manifest | Каждый stage report обязан фиксировать `Created / Modified / Deleted / Reason / Contract impact`; любой файл вне prompt expected paths должен иметь отдельное объяснение. |

## Stage Status

| Stage | Статус | Prompt / task | Stage report | Validation depth | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|
| `01` Contract foundation and audit schema | accepted | `.codex/agents/generated/market-data-live-tail-repair-v1/01-contract-foundation-audit-schema.md` | `01-contract-foundation-audit-schema.md` | DB migration + port contract tests + repository-boundary audit proof + full publish gates | Contract foundation/audit schema implemented, locally validated, scoped publish contents approved, and ready for direct-main delivery evidence in the accepted Stage `01` commit. | none | yes |
| `02` Redis hot cache | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/02-redis-hot-cache.md` | `02-redis-hot-cache.md` | Redis integration + metrics | TBD | none until Stage `02` execution starts | yes |
| `03` Tail provider source chain | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/03-tail-provider-source-chain.md` | `03-tail-provider-source-chain.md` | provider integration + audit DB + REST adapter boundary | TBD | blocked until `02 accepted` | no |
| `04` Strategy runner integration and ACK policy | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/04-strategy-runner-integration-ack-policy.md` | `04-strategy-runner-integration-ack-policy.md` | runner integration + Redis pending/backlog proof | TBD | blocked until `03 accepted` | no |
| `05` Metrics alerts runbook | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/05-metrics-alerts-runbook.md` | `05-metrics-alerts-runbook.md` | metrics endpoint + alert rule/runbook proof | TBD | blocked until `04 accepted` | no |
| `06` Mac Studio repair proof | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/06-macstudio-repair-proof.md` | `06-macstudio-repair-proof.md` | post_main_production_runtime_proof | TBD | blocked until `05 accepted` | no |
| `07` Stage 12.4 rerun handoff | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/07-stage-12-4-rerun-handoff.md` | `07-stage-12-4-rerun-handoff.md` | sustained soak rerun or explicit handoff | TBD | blocked until `06 accepted` | no |

## Что Обязательно Знать Дальше

| Stage | Факт / решение / ограничение | Почему важно следующему stage | Evidence |
|---|---|---|---|
| plan | `Stage 12.4` strategy-producer blocked because a short Redis candle gap plus ClickHouse HTTP failure stopped signal production. | Repair work must focus on Market Data live-tail resilience, not on strategy evaluator logic. | `12-4-sustained-6h-soak.md`; strategy-producer ledger |
| plan | `Market Data` owns REST tail recovery; `StrategyLiveRunner` must not call Binance/Bybit REST directly. | Preserves bounded context, provider secrets boundary, and adapter reuse. | `market-data-live-tail-repair-v1.md` |
| plan | ClickHouse remains historical truth but cannot be the only live repair source. | Provider chain must include Redis hot cache and REST tail fallback with circuit breaker. | `market-data-live-tail-repair-v1.md` |
| plan | Redis stream remains transport; Redis hot cache is separate range-store. | Stage `02` must not misuse stream consumer group state as range repair storage. | `market-data-live-tail-repair-v1.md` |
| plan | ACK policy must be proven, not assumed. | Stage `04` must demonstrate failed repair followed by retry without candle loss and without duplicate signals. | `market-data-live-tail-repair-v1.md` |
| plan | `12.5` remains blocked until `12.4 accepted`. | Stage `07` may rerun/open `12.4`, but cannot open `12.5` without accepted 6h evidence. | strategy-producer ledger |
| `01` | `ClosedCandleTailProvider`, repair DTOs, `CandleRepairAuditRepository`, and additive migration `20260629_0038` are implemented and pass focused/full gates. | Stage `02` can build Redis hot cache against these contracts after scoped direct-main delivery remains green. | `01-contract-foundation-audit-schema.md`; local gates listed there |
| `01` | No runtime repair behavior is wired in Stage `01`; `_repair_gap`, Redis hot cache, REST fallback, ACK policy, metrics, and production proof are later stages. | Prevents treating contract/audit foundation as proof that live-tail repair works in runtime. | `01-contract-foundation-audit-schema.md` |

## Contract Impact Matrix

| Stage | Public API | Persistence | Config / Ops | Runtime / UI | Rollback |
|---|---|---|---|---|---|
| `01` | none | additive Postgres audit table | none or minimal config defaults | none | rollback migration only before production writes; after writes keep table unused |
| `02` | none | none | Redis hot cache config/metrics | market-data worker writes extra Redis keys | disable hot cache feature flag |
| `03` | none | audit rows written | ClickHouse circuit/REST tail config | provider chain available to Strategy wiring | disable REST tail fallback, keep Redis/CH |
| `04` | none | strategy run metadata/audit may include repair data | strategy runner config | checkpoint/ACK behavior changes | disable new provider and restore ClickHouse-only path only if no accepted live repair dependency |
| `05` | none | none | Prometheus alerts/runbook | ops visibility | disable alert rules |
| `06` | none | runtime audit/evidence rows | deploy/sync | production runtime behavior proof | rollback deployed SHA through normal deploy |
| `07` | none | stage report/ledger rows | none | 6h soak state | stop collector and keep repair accepted |

## Business / Service-Call Coverage Notes

| Stage | Business impact | Service-call / ops coverage |
|---|---|---|
| `01` | Creates auditable repair contracts so later runtime fixes can be reviewed before reopening the blocked paper/testnet strategy-producer soak. It does not restore production runtime behavior by itself. | Runtime service calls, auth/secrets, provider payloads, timeout/retry behavior, ACK/checkpoint behavior, alerts, and runbooks are `N/A` for Stage `01`; no runtime wiring changes. |
| `02`-`06` | Restores live-tail continuity and then proves changed-code behavior on Mac Studio before rerunning `12.4`. | Covered in the relevant stage prompts; `post_main_production_runtime_proof` is only valid after `main`, green CI, deploy/sync, and runtime smoke/proof. |
| `07` | Converts accepted repair evidence into the strategy-producer rerun/handoff decision. | No new service-call surface unless the rerun finds a new blocker; document any new blocker separately. |

## Validation Ledger

| Stage | Required gates | Real-boundary acceptance | Result | Evidence path | Residual risk | Next action |
|---|---|---|---|---|---|---|
| `01` | focused tests, migration check, docs index, repo-wide publish gates | Repository-boundary audit insert/read proof through `MarketDataPostgresGateway` fake; fake provider contract call for continuous/missing results; Alembic head check. Live Postgres/runtime proof is `N/A` for dormant Stage `01`. | accepted | `01-contract-foundation-audit-schema.md` | Runtime repair behavior is still not implemented; Redis hot cache, provider chain, ACK policy, metrics, and production proof remain later stages. | Run Stage `02` Redis hot cache after direct-main delivery/CI stays green. |
| `02` | focused tests, Redis integration call, docs index | real Redis duplicate write/range read/retention/metrics | TBD | Stage report | TBD | TBD |
| `03` | focused tests, provider integration call, docs index | provider returns continuous range with ClickHouse failure and REST tail fallback; audit row exists | TBD | Stage report | TBD | TBD |
| `04` | focused runner tests, Redis pending/backlog integration, docs index | failed repair does not lose future candle; later retry advances checkpoint and creates no duplicate signals | TBD | Stage report | TBD | TBD |
| `05` | metrics tests, alert rule parse, docs index | metrics endpoint exposes repair/cache/circuit/checkpoint-stall signals after synthetic repair call | TBD | Stage report | TBD | TBD |
| `06` | local gates, green GitHub Actions/CI, deploy/sync, Mac Studio runtime smoke | `post_main_production_runtime_proof` only: changed revision is on `main`, CI is green, deploy/sync into `/opt/roehub/app` is complete, then controlled missing minute + ClickHouse unavailable + REST tail recovery is proven in deployed runtime. | TBD | Stage report | TBD | TBD |
| `07` | docs index, runtime collector | `12.4` accepted rerun or explicit unrelated blocker | TBD | Stage report + strategy-producer ledger | TBD | TBD |

## Publish / Deploy Handoff

| Stage | Branch policy | Delivery requirement | CI/deploy | Host sync | Notes |
|---|---|---|---|---|---|
| all | `main`, no speculative branch | successful stages publish via `publish-ci-deploy` direct-main discipline | required when code/config changes | required for runtime/code stages | docs-only blocked stages may record publish `N/A` only with reason |
| `01` | `main`, no speculative branch | scoped direct-main publish of Stage `01` code/tests/report plus approved prompt-pack/plan/ledger artifacts | required for direct-main delivery | runtime host sync `N/A` for dormant contract/schema stage | User explicitly approved including pre-existing untracked `.codex/agents/generated/market-data-live-tail-repair-v1/`, `market-data-live-tail-repair-v1.md`, and market-data stage ledger/report docs in the scoped publish set. |

## Blockers / Handoff

| From | Blocker | Severity | Handoff |
|---|---|---|---|
| strategy-producer `12.4` | Live tail gap could not be repaired when ClickHouse HTTP failed. | high | Complete this repair-cycle through Stage `06`, then rerun/open strategy-producer `12.4`. |

## Next Prompt

Next prompt to run:

```text
.codex/agents/generated/market-data-live-tail-repair-v1/02-redis-hot-cache.md
```

Reason: Stage `01` contract foundation/audit schema is accepted after scoped direct-main delivery approval and required local validation. Stage `02` is the next pending stage and is explicitly allowed by this ledger once publish/CI remains green.

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| 2026-06-29 | plan | Created repair-cycle plan and ledger for Market Data live tail repair after strategy-producer Stage `12.4` blocker. | `market-data-live-tail-repair-v1.md`; this ledger |
| 2026-06-29 | `01` | Implemented and locally validated contract foundation/audit schema, included user-approved pre-existing market-data prompt-pack/plan/ledger artifacts in scoped direct-main publish scope, and opened Stage `02`. | `01-contract-foundation-audit-schema.md`; focused/full gates listed there |
