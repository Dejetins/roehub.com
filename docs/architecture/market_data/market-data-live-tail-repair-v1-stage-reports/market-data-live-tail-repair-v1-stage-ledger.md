# Market Data Live Tail Repair v1 — журнал выполнения stages

Единый handoff-документ для плана `docs/architecture/market_data/market-data-live-tail-repair-v1.md`.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/market_data/market-data-live-tail-repair-v1.md` |
| `prompt_pack` | `.codex/agents/generated/market-data-live-tail-repair-v1/` |
| `ledger_status` | `active` |
| `current_stage` | `04` |
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
| `02` Redis hot cache | accepted | `.codex/agents/generated/market-data-live-tail-repair-v1/02-redis-hot-cache.md` | `02-redis-hot-cache.md` | Redis integration + metrics + real Redis proof | Redis hot cache writer/reader, retention config, worker fan-out wiring, metrics hooks, tests, and isolated `macstudio` Redis duplicate/range proof are complete and ready for direct-main delivery evidence in the accepted Stage `02` commit. | none | yes |
| `03` Tail provider source chain | accepted | `.codex/agents/generated/market-data-live-tail-repair-v1/03-tail-provider-source-chain.md` | `03-tail-provider-source-chain.md` | provider integration + audit DB + REST adapter boundary | `MarketDataClosedCandleTailProvider` implements Redis -> ClickHouse -> REST -> audit chain; focused integration tests prove ClickHouse failure fallback, REST hot-cache write, Redis second-call hit, audit write, and miss guards. | none | yes |
| `04` Strategy runner integration and ACK policy | pending | `.codex/agents/generated/market-data-live-tail-repair-v1/04-strategy-runner-integration-ack-policy.md` | `04-strategy-runner-integration-ack-policy.md` | runner integration + Redis pending/backlog proof | TBD | none until Stage `04` execution starts | yes |
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
| `02` | `RedisCandleHotCache` writes normalized closed 1m candles to `md:hot:1m:<instrument_key>:h` and `md:hot:1m:<instrument_key>:z`, both keyed by `ts_open_epoch_ms`; duplicate writes overwrite the same hash field and zset member. | Stage `03` can use this adapter as the first source in the provider chain and can rely on deterministic no-ambiguity cache rows. | `02-redis-hot-cache.md`; focused tests |
| `02` | Real Redis proof used approved isolated synthetic keys on `macstudio` via SSH tunnel: `zcard=3`, `hlen=3`, range timestamps `12:00`, `12:01`, `12:02`, duplicate write count `4`, cleanup `cleanup_remaining_keys=0`. | Stage `03` can treat Redis hot cache range read behavior as accepted adapter evidence, but not as production changed-code runtime proof. | `02-redis-hot-cache.md` |
| `02` | Prompt pytest gate was adapted because `tests/integration` is absent in this repo snapshot. Equivalent evidence is `tests/unit/contexts/market_data` plus the real Redis proof. | Future stages should not claim `tests/integration` ran unless the directory exists or the prompt is updated. | `02-redis-hot-cache.md` |
| `03` | `MarketDataClosedCandleTailProvider` now provides the concrete source chain using injected hot cache, `CanonicalCandleReader`, `CandleIngestSource`, `CandleRepairAuditRepository`, and `Clock`. | Stage `04` can inject this provider into Strategy runner without giving Strategy direct REST/ClickHouse responsibilities. | `03-tail-provider-source-chain.md`; focused tests |
| `03` | Provider proof forced ClickHouse failure, restored the missing candle from fake REST, wrote it to hot cache before success, and proved the second call was a Redis hit without REST/ClickHouse calls. | Stage `04` can focus on runner gap repair and ACK/checkpoint policy instead of rebuilding source-chain behavior. | `03-tail-provider-source-chain.md` |
| `03` | Provider rejects current-open ranges and REST fallback ranges older than `ClosedCandleTailRepairPolicy.rest_tail_limit_minutes`, returning `continuous=false` and writing redacted audit miss. | Stage `04` must preserve these fail-closed semantics when runner repair asks for a range. | `03-tail-provider-source-chain.md` |

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
| `02` | focused tests, Redis integration call, docs index | real Redis duplicate write/range read/retention/metrics | accepted | `02-redis-hot-cache.md` | Strategy provider chain is not implemented yet; Redis hot cache is not proof of end-to-end live-tail repair. `post_main_production_runtime_proof` remains Stage `06`. | Run Stage `03` Tail provider source chain after direct-main delivery/CI stays green. |
| `03` | focused tests, provider integration call, docs index | provider returns continuous range with ClickHouse failure and REST tail fallback; audit row exists | accepted | `03-tail-provider-source-chain.md` | Strategy runner is not wired yet; ACK/checkpoint behavior remains unchanged until Stage `04`. | Run Stage `04` Strategy runner integration and ACK policy after direct-main delivery/CI stays green. |
| `04` | focused runner tests, Redis pending/backlog integration, docs index | failed repair does not lose future candle; later retry advances checkpoint and creates no duplicate signals | TBD | Stage report | TBD | TBD |
| `05` | metrics tests, alert rule parse, docs index | metrics endpoint exposes repair/cache/circuit/checkpoint-stall signals after synthetic repair call | TBD | Stage report | TBD | TBD |
| `06` | local gates, green GitHub Actions/CI, deploy/sync, Mac Studio runtime smoke | `post_main_production_runtime_proof` only: changed revision is on `main`, CI is green, deploy/sync into `/opt/roehub/app` is complete, then controlled missing minute + ClickHouse unavailable + REST tail recovery is proven in deployed runtime. | TBD | Stage report | TBD | TBD |
| `07` | docs index, runtime collector | `12.4` accepted rerun or explicit unrelated blocker | TBD | Stage report + strategy-producer ledger | TBD | TBD |

## Publish / Deploy Handoff

| Stage | Branch policy | Delivery requirement | CI/deploy | Host sync | Notes |
|---|---|---|---|---|---|
| all | `main`, no speculative branch | successful stages publish via `publish-ci-deploy` direct-main discipline | required when code/config changes | required for runtime/code stages | docs-only blocked stages may record publish `N/A` only with reason |
| `01` | `main`, no speculative branch | scoped direct-main publish of Stage `01` code/tests/report plus approved prompt-pack/plan/ledger artifacts | required for direct-main delivery | runtime host sync `N/A` for dormant contract/schema stage | User explicitly approved including pre-existing untracked `.codex/agents/generated/market-data-live-tail-repair-v1/`, `market-data-live-tail-repair-v1.md`, and market-data stage ledger/report docs in the scoped publish set. |
| `02` | `main`, no speculative branch | scoped direct-main publish of Stage `02` code/config/tests/report/docs | required for direct-main delivery | host sync/prod runtime proof deferred to Stage `06`; Stage `02` collected only isolated pre-main Redis proof | User explicitly approved isolated synthetic-key Redis proof against `macstudio`; proof keys were cleaned up. |
| `03` | `main`, no speculative branch | scoped direct-main publish of Stage `03` provider/tests/report/docs | required for direct-main delivery | host sync/prod runtime proof deferred to Stage `06`; Stage `03` uses fake/safe REST and fake Postgres gateway proof | Strategy runner is intentionally not wired until Stage `04`. |

## Blockers / Handoff

| From | Blocker | Severity | Handoff |
|---|---|---|---|
| strategy-producer `12.4` | Live tail gap could not be repaired when ClickHouse HTTP failed. | high | Complete this repair-cycle through Stage `06`, then rerun/open strategy-producer `12.4`. |

## Next Prompt

Next prompt to run:

```text
.codex/agents/generated/market-data-live-tail-repair-v1/04-strategy-runner-integration-ack-policy.md
```

Reason: Stage `03` provider source chain is accepted after focused provider-chain, Redis-hit, REST-fallback, and audit repository boundary tests. Stage `04` is the next pending stage and is explicitly allowed once Stage `03` scoped delivery/CI remains green.

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| 2026-06-29 | plan | Created repair-cycle plan and ledger for Market Data live tail repair after strategy-producer Stage `12.4` blocker. | `market-data-live-tail-repair-v1.md`; this ledger |
| 2026-06-29 | `01` | Implemented and locally validated contract foundation/audit schema, included user-approved pre-existing market-data prompt-pack/plan/ledger artifacts in scoped direct-main publish scope, and opened Stage `02`. | `01-contract-foundation-audit-schema.md`; focused/full gates listed there |
| 2026-06-30 | `02` | Implemented Redis hot cache writer/reader, production config, worker fan-out wiring, metrics hooks, tests, docs sync, and approved isolated `macstudio` Redis duplicate/range proof with cleanup. | `02-redis-hot-cache.md` |
| 2026-06-30 | `03` | Implemented provider source chain, ClickHouse failure circuit fallback to REST, REST hot-cache write before success, redacted repair audit writes, closed-tail guards, tests, and docs sync. | `03-tail-provider-source-chain.md` |
