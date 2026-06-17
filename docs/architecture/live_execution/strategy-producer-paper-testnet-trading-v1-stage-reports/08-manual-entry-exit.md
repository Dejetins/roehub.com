# Stage 08: Manual entry and manual exit

Статус: `in_progress`

## Pre-Start

User required before start: nothing

Stage `07` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` до implementation edits: статус `accepted`, `Next stage allowed = yes`, активных blockers нет.

## Scope

Stage `08` добавляет отдельные manual entry и manual stop/exit controls на `/strategies`. Команды должны создавать `manual_request` source events и идти через тот же `ExecutionIngressService` risk/intent/dispatch/outbox path, что и strategy signals. Для `paper` режима ручная команда должна оставаться no-exchange-submit и не публиковать Redis execution dispatch; для `testnet` команда должна fail-closed, если profile/account/config/readiness не доказаны.

## Concrete Planned File List Before Editing

Ожидаемый broad path `src/trading/contexts/live_execution` сужен до конкретных файлов до implementation edits:

| File | Planned action | Reason |
|---|---:|---|
| `apps/api/routes/strategies.py` | modify | Add strategy-scoped manual entry/exit DTOs and endpoints that map authenticated user/profile/run state into live-execution commands. |
| `apps/api/routes/ui_execution.py` | no code change planned | Existing lower-level source/intent API remains the shared execution boundary; manual strategy commands reuse the same services. |
| `apps/api/wiring/modules/live_execution.py` | modify | Expose reusable live-execution service builder so strategy routes and UI execution routes share the same repository/dispatch wiring. |
| `apps/api/wiring/modules/strategy.py` | modify | Inject shared live-execution services into strategy manual command endpoints. |
| `apps/api/main/app.py` | modify | Build shared live-execution services once and pass them to both routers. |
| `src/trading/contexts/live_execution/domain/execution_source.py` | no code change planned | `manual_request` source type already exists; keep source contract stable. |
| `src/trading/contexts/live_execution/domain/risk_gate.py` | modify | Add paper manual no-exchange-submit branch while preserving account/testnet fail-closed checks. |
| `src/trading/contexts/live_execution/domain/paper_accounting.py` | modify | Allow paper order rows to carry a manual `source_event_id` without changing existing strategy signal rows. |
| `src/trading/contexts/live_execution/application/use_cases/paper_accounting.py` | modify | Add idempotent manual paper execution recording for paper manual entry/exit. |
| `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/paper_accounting_repository.py` | modify | Preserve idempotent manual paper rows in unit tests. |
| `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/paper_accounting_repository.py` | modify | Persist optional `source_event_id` on paper orders. |
| `alembic/versions/20260618_0036_manual_paper_orders_source_event_v1.py` | create | Add nullable paper order source-event linkage for manual paper proof. |
| `apps/web/templates/pages/strategies.html` | modify | Add separate manual entry and manual exit buttons and endpoint templates. |
| `apps/web/dist/js/pages/strategies.js` | modify | Wire manual button clicks with stable per-selected-strategy idempotency keys and visible outcomes. |
| `apps/web/locales/en.json` | modify | Add manual action labels/statuses. |
| `apps/web/locales/ru.json` | modify | Add manual action labels/statuses. |
| `tests/unit/apps/api/test_strategies_routes.py` | modify | Cover manual endpoint idempotency, paper no-dispatch outcome, and blocked modes. |
| `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | modify | Cover manual paper no-exchange-submit risk branch. |
| `tests/unit/contexts/live_execution/test_paper_accounting_service.py` | modify | Cover idempotent manual paper order/fill/accounting rows. |
| `tests/unit/apps/migrations/test_manual_paper_orders_source_event_sql.py` | create | Assert additive paper order source-event migration. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/08-manual-entry-exit.md` | create/modify | Stage report, manifest, evidence, blockers. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage status/evidence/handoff. |
| `docs/architecture/README.md` | check/update if generated index requires it | Required docs index check after adding this stage report; existing unrelated RL docs changes must not be reverted. |

## Initial Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `compatible-change` | Adds strategy-scoped manual action endpoints; existing endpoints/payloads remain valid. |
| Port contract | `compatible-change` | Reuses existing live-execution ingress/dispatch services; paper accounting gains additive manual source-event support. |
| DTO schema | `compatible-change` | New request/response DTOs for manual commands only. |
| Persisted schema | `compatible-change` | Additive nullable `paper_orders.source_event_id`; no existing table rewrite or required field. |
| Config schema | `none` | No new required env/config planned. |
| Request hash / cache / identity | `compatible-change` | Manual idempotency keys are action/profile/run scoped; existing launch/request hashes unchanged. |
| Service-call semantics | `compatible-change` | Manual commands use existing risk/dispatch path; paper remains no-dispatch. |
| External side effects | `compatible-change` | Duplicate manual clicks return existing outcome; mainnet remains unavailable. |
| Logs / metrics / audit / report | `compatible-change` | Existing live-execution source/intent/risk/outbox metrics include `manual_request`. |
| Browser-visible behavior | `compatible-change` | Adds manual entry/exit buttons and visible pending/accepted/rejected/unknown result text on `/strategies`. |

## Evidence

Local implementation and gates are complete. Stage remains `in_progress` until main delivery, CI/deploy, Mac Studio sync/smoke, browser clicks, DB/Redis/metrics proof, and testnet-safe proof/blocker are recorded.

### Implementation Summary

- Added strategy-scoped `POST /strategies/{strategy_id}/manual-entry` and `POST /strategies/{strategy_id}/manual-exit` commands.
- Manual commands create `manual_request` source events, create execution intents through `ExecutionIngressService`, run source-aware risk, reuse existing notification outbox behavior for rejects, and dispatch only accepted non-paper intents.
- Paper manual commands use the same Stage `07` no-exchange-submit pattern: risk outcome is explicit `paper_no_exchange_submit`, Redis execution dispatch is skipped, and idempotent paper order/fill/accounting rows are recorded.
- Added a shared live-execution service builder so `/ui/execution/*` and strategy manual endpoints use the same repository/dispatch wiring in the API process.
- Added `/strategies` manual entry and manual exit buttons with generated idempotency keys and visible `pending`/`accepted`/`rejected`/`unknown` response text.
- Added additive `paper_orders.source_event_id` linkage for manual paper proof without changing existing strategy-signal paper rows.

### Local Gates

| Gate | Result |
|---|---|
| `uv run ruff check apps src/trading/contexts/live_execution tests` | passed |
| `uv run pyright apps src/trading/contexts/live_execution tests` | passed, `0 errors` |
| `uv run pytest -q tests/unit/apps tests/unit/contexts/live_execution` | passed, `374 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index update |
| `uv run ruff check .` | passed |
| `uv run pyright` | passed, `0 errors` |
| `uv run pytest -q -ra` | passed, `1193 passed, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed |

### Focused Local Proof

| Check | Evidence |
|---|---|
| Manual paper API idempotency | `test_manual_entry_paper_creates_idempotent_source_intent_and_paper_order`: same `Idempotency-Key` returns the same `source_event_id` and `intent_id`; one source event, one intent, one paper order, one fill, one accounting row. |
| Manual risk branch | `test_manual_request_paper_no_exchange_submit_uses_no_dispatch_risk_branch`: `manual_request` with `paper_no_exchange_submit=True` records a rejected no-dispatch risk outcome with reason `paper_no_exchange_submit`. |
| Paper accounting idempotency | `test_manual_paper_execution_records_idempotent_order_fill_and_accounting`: same manual `source_event_id` does not duplicate paper money rows. |
| Additive migration | `test_manual_paper_orders_source_event_migration_is_additive`: migration adds nullable `source_event_id` and does not drop `paper_orders`. |

### Pending Runtime Evidence

| Surface | Status |
|---|---|
| Main delivery / CI / deploy | pending |
| Mac Studio checkout sync and `/opt/roehub/app` smoke | pending |
| Playwright manual entry click in paper mode | pending |
| Playwright manual exit click in paper mode | pending |
| DB proof for source event, intent, risk audit, paper order/fill/accounting, outbox | pending |
| Redis/metrics proof of expected dispatch or no-dispatch behavior | pending |
| Testnet-safe representative manual action or exact Stage `05` blocker | pending |

## File Manifest

| Action | File | Reason | Contract impact |
|---|---|---|---|
| Modified | `apps/api/main/app.py` | Build shared live-execution services once for API routers. | `compatible-change` service wiring |
| Modified | `apps/api/routes/strategies.py` | Add manual entry/exit DTOs, endpoints, idempotency, risk/dispatch/paper response mapping. | `compatible-change` API/DTO |
| Modified | `apps/api/wiring/modules/__init__.py` | Export shared live-execution builder/service container. | `none` runtime behavior |
| Modified | `apps/api/wiring/modules/live_execution.py` | Add reusable `LiveExecutionServices` builder and allow UI execution router to consume shared services. | `compatible-change` service wiring |
| Modified | `apps/api/wiring/modules/strategy.py` | Inject shared live-execution services and paper accounting into strategy routes. | `compatible-change` service wiring |
| Modified | `apps/web/dist/js/pages/strategies.js` | Wire manual entry/exit clicks, idempotency keys, pending/result status, and refresh. | `compatible-change` browser-visible behavior |
| Modified | `apps/web/locales/en.json` | Add manual action labels/status text. | `compatible-change` browser-visible copy |
| Modified | `apps/web/locales/ru.json` | Add manual action labels/status text. | `compatible-change` browser-visible copy |
| Modified | `apps/web/templates/pages/strategies.html` | Add manual entry/exit buttons and endpoint templates. | `compatible-change` browser-visible behavior |
| Created | `alembic/versions/20260618_0036_manual_paper_orders_source_event_v1.py` | Add nullable `paper_orders.source_event_id` and partial unique index for manual paper idempotency evidence. | `compatible-change` additive migration |
| Modified | `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/paper_accounting_repository.py` | Keep in-memory paper execution idempotent for manual source-event rows. | `none` production runtime |
| Modified | `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/paper_accounting_repository.py` | Persist optional manual `source_event_id` on paper orders. | `compatible-change` persistence |
| Modified | `src/trading/contexts/live_execution/application/use_cases/paper_accounting.py` | Add idempotent manual paper order/fill/accounting recording. | `compatible-change` paper execution behavior |
| Modified | `src/trading/contexts/live_execution/domain/paper_accounting.py` | Add optional `source_event_id` to paper order domain record. | `compatible-change` additive domain field |
| Modified | `src/trading/contexts/live_execution/domain/risk_gate.py` | Add manual paper no-exchange-submit risk branch. | `compatible-change` risk semantics |
| Modified | `tests/unit/apps/api/test_app_strategy_router_toggle.py` | Update app-router test double for shared live-execution argument. | `none` production runtime |
| Modified | `tests/unit/apps/api/test_strategies_routes.py` | Cover manual paper API idempotency and DB-row creation via in-memory adapters. | `none` production runtime |
| Created | `tests/unit/apps/migrations/test_manual_paper_orders_source_event_sql.py` | Guard additive migration contract. | `none` production runtime |
| Modified | `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | Cover manual paper no-dispatch risk branch. | `none` production runtime |
| Modified | `tests/unit/contexts/live_execution/test_paper_accounting_service.py` | Cover manual paper accounting idempotency. | `none` production runtime |
| Created | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/08-manual-entry-exit.md` | Stage report and local evidence. | `none` runtime |
| Modified | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | Stage status/evidence/handoff. | `none` runtime |
| Modified | `docs/architecture/README.md` | Docs index entry for Stage `08`; unrelated RL index line remains a pre-existing user change and is not part of Stage `08` staging. | `none` runtime |
| Deleted | none | No files deleted. | `none` |

## Blockers

Pending main delivery, CI/deploy, Mac Studio sync/smoke, browser/API/DB/Redis/metrics runtime proof, and testnet-safe representative proof or exact blocker. Stage is not accepted yet.

## Handoff

Continue with scoped main delivery. Do not stage unrelated RL prompt/docs work currently present in the worktree. When staging `docs/architecture/README.md`, include only the Stage `08` index hunk and leave the pre-existing RL Stage `02A` hunk untouched unless the owner explicitly includes that scope.
