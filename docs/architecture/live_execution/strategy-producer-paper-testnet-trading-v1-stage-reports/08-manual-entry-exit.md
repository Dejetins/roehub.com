# Stage 08: Manual entry and manual exit

Статус: `accepted`

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

Stage accepted on `2026-06-18` after direct `main` delivery, CI/deploy success, Mac Studio sync/smoke, browser manual entry/exit proof, DB/Redis/metrics proof, testnet-safe fail-closed proof, and cleanup.

### Implementation Summary

- Added strategy-scoped `POST /strategies/{strategy_id}/manual-entry` and `POST /strategies/{strategy_id}/manual-exit` commands.
- Manual commands create `manual_request` source events, create execution intents through `ExecutionIngressService`, run source-aware risk, reuse existing notification outbox behavior for rejects, and dispatch only accepted non-paper intents.
- Paper manual commands use the same Stage `07` no-exchange-submit pattern: risk outcome is explicit `paper_no_exchange_submit`, Redis execution dispatch is skipped, and idempotent paper order/fill/accounting rows are recorded.
- Added a shared live-execution service builder so `/ui/execution/*` and strategy manual endpoints use the same repository/dispatch wiring in the API process.
- Added `/strategies` manual entry and manual exit buttons with generated idempotency keys and visible `pending`/`accepted`/`rejected`/`unknown` response text.
- Added additive `paper_orders.source_event_id` linkage for manual paper proof without changing existing strategy-signal paper rows.
- Repaired runtime-found HTTP error mapping so `strategy_manual_execution.*` errors return stable client statuses instead of leaking as `500`.

### Local Gates

| Gate | Result |
|---|---|
| `uv run ruff check apps src/trading/contexts/live_execution tests` | passed |
| `uv run pyright apps src/trading/contexts/live_execution tests` | passed, `0 errors` |
| `uv run pytest -q tests/unit/apps tests/unit/contexts/live_execution` | passed, `374 passed, 3 warnings` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index update |
| `uv run ruff check .` | passed |
| `uv run pyright` | passed, `0 errors` |
| `uv run pytest -q -ra` | passed before runtime bugfix, `1193 passed, 3 warnings`; passed after bugfix, `1195 passed, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed |
| `uv run pytest -q tests/unit/apps/api/test_api_error_handlers.py::test_roehub_error_handler_maps_manual_execution_errors tests/unit/apps/api/test_strategies_routes.py::test_manual_entry_without_active_run_returns_conflict tests/unit/apps/api/test_strategies_routes.py::test_manual_entry_paper_creates_idempotent_source_intent_and_paper_order` | passed after runtime bugfix, `3 passed` |

### Focused Local Proof

| Check | Evidence |
|---|---|
| Manual paper API idempotency | `test_manual_entry_paper_creates_idempotent_source_intent_and_paper_order`: same `Idempotency-Key` returns the same `source_event_id` and `intent_id`; one source event, one intent, one paper order, one fill, one accounting row. |
| Manual risk branch | `test_manual_request_paper_no_exchange_submit_uses_no_dispatch_risk_branch`: `manual_request` with `paper_no_exchange_submit=True` records a rejected no-dispatch risk outcome with reason `paper_no_exchange_submit`. |
| Paper accounting idempotency | `test_manual_paper_execution_records_idempotent_order_fill_and_accounting`: same manual `source_event_id` does not duplicate paper money rows. |
| Additive migration | `test_manual_paper_orders_source_event_migration_is_additive`: migration adds nullable `source_event_id` and does not drop `paper_orders`. |

### Main Delivery, CI, Deploy

| Surface | Evidence |
|---|---|
| Main implementation delivery | `ea239c9f9e428359c8ca791b4d2b694fc75e59b3` (`Add manual strategy execution controls`) delivered to `origin/main`. |
| Runtime-found bugfix delivery | `9ae17321b5952ba0ff18ac6fe88f517e07a31715` (`Map manual execution errors`) delivered to `origin/main`. |
| Current main during acceptance proof | `1a6be0ae77c8ea1755cbdc35e4ff48937b762761`; Stage `08` commits are ancestors. This commit also contains unrelated RL docs changes and is not Stage `08` implementation scope. |
| CI for implementation commit | CI `27722721031` succeeded; Deploy Backend `27722798318` succeeded; Publish App Image `27722798352` succeeded; Deploy Web `27722798348` and follow-up `27722865764` succeeded. |
| CI for bugfix commit | CI `27723350993` succeeded; Deploy Backend `27723436097` succeeded; Publish App Image `27723436035` succeeded; Deploy Web `27723436079` and follow-up `27723444241` succeeded. |
| CI for current main proof SHA | CI `27723455674` succeeded; Deploy Backend `27723481220` succeeded; Publish App Image `27723481247` succeeded; Deploy Web `27723481275` and follow-up `27723487907` succeeded. |

### Mac Studio Runtime Proof

| Surface | Evidence |
|---|---|
| Checkout sync | Mac Studio checkout fast-forwarded through Stage `08` implementation and then to current main `1a6be0ae77c8ea1755cbdc35e4ff48937b762761`. |
| Runtime file proof | `/opt/roehub/app` contains migration `20260618_0036`, strategy `manual-entry`/`manual-exit` routes, `/strategies` JS manual controls, and `strategy_manual_execution.*` HTTP error mapping. |
| Smoke | `bash scripts/macos/smoke_prod.sh` exited `0`: core services loaded, expected API `401`, Redis `PONG`, Tailscale `Running`. |

### Browser, API, DB, Redis, Metrics

Runtime proof used synthetic subject `codex:stage08-manual-entry-exit-final:20260617T222334-4b31756b`, paper strategy `52cab273-7c88-4549-865a-b853b1bffa28`, and run `7b0af690-335b-4e44-9c24-e3f3fad94e37`.

| Surface | Evidence |
|---|---|
| Browser manual entry/exit | Playwright opened `https://roehub.com/strategies?strategy_id=52cab273-7c88-4549-865a-b853b1bffa28`, clicked manual entry and manual exit, and observed both POSTs return `200`. Dashboard refreshes returned `200`; console had `0` errors/warnings. |
| Browser visible result | DOM contained `manual:entry`, `manual:exit`, no-dispatch text, and action status `accepted: filled`. Screenshot: `output/playwright/stage08-manual-entry-exit-final-strategies.png`. |
| DB counts | `source_events=2`, `intents=2`, `risk_audits=2`, `paper_orders=2`, `paper_fills=2`, `accounting_rows=2`, `notifications=2`, `dispatched_rows=0`. |
| Entry ledger row | `source_event_ref=manual:entry:7b0af690-335b-4e44-9c24-e3f3fad94e37`; intent/risk rejected with `paper_no_exchange_submit`; dispatch fields `NULL`; paper order filled with `paper_market_fill_from_manual_request`; accounting completeness `paper_fee_fixed_bps_funding_not_applicable`; notification `producer_rejected/paper_no_exchange_submit`. |
| Exit ledger row | `source_event_ref=manual:exit:7b0af690-335b-4e44-9c24-e3f3fad94e37`; same no-dispatch risk/outbox shape; accounting completeness `paper_spot_short_borrow_not_modeled`. |
| Redis dispatch proof | Redis execution streams were unchanged before/after manual actions: `execution.requests.v1=15`, `execution.requests.retry.v1=1`, `execution.requests.dlq.v1=2`. |
| Metrics proof | Metrics exposed manual request rows: `execution_source_event_total{result="recorded",source_type="manual_request"} 2.0`; rejected intent/risk totals with `paper_no_exchange_submit` each `2.0`; warning notification total `2.0`; paper accounting totals for the entry/exit completeness reasons each `1.0`. |

### Testnet-Safe Proof

| Surface | Evidence |
|---|---|
| Testnet profile/readiness | Testnet strategy `2fc641c6-da50-465f-b9b6-2319d5962429` profile update returned `200`; readiness was `blocked/exchange_connection_not_found`. |
| Inactive run block | Testnet run attempt returned `409` with code `strategy_run.capital_reservation_blocked`, reason `capital_projection_missing`. |
| Manual fail-closed | Manual entry returned `409` with code `strategy_manual_execution.blocked`, reason `strategy_run_inactive`; no source/intent/dispatch rows were written and Redis stream lengths stayed unchanged. |

### Cleanup

| Surface | Evidence |
|---|---|
| Synthetic state cleanup | Final cleanup showed `active_runs=0`, `active_sessions=0`; Playwright smoke session `stage08-manual-final` was closed. |

## File Manifest

Formal manifest shape normalized during Stage `14` audit; historical evidence and acceptance are unchanged.

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `alembic/versions/20260618_0036_manual_paper_orders_source_event_v1.py`, `tests/unit/apps/migrations/test_manual_paper_orders_source_event_sql.py`, `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/08-manual-entry-exit.md` | none | none | Add manual paper idempotency schema coverage, migration test, and Stage `08` report. | `compatible-change`: additive manual paper persistence and docs/handoff. |
| none | API error/router/wiring files, strategy UI assets/locales/template, live-execution paper/risk/accounting paths, focused tests, stage ledger, docs index | none | Add manual entry/exit controls and prove idempotent paper no-dispatch accounting plus fail-closed testnet manual behavior. | `compatible-change`: additive API/DTO, browser-visible controls, and paper execution behavior. |

Historical detailed manifest:

| Action | File | Reason | Contract impact |
|---|---|---|---|
| Modified | `apps/api/common/errors.py` | Map `strategy_manual_execution.*` domain errors to stable HTTP statuses after runtime proof found a `500` leak. | `compatible-change` API error status correction |
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
| Modified | `tests/unit/apps/api/test_api_error_handlers.py` | Cover manual execution domain error HTTP status mapping. | `none` production runtime |
| Modified | `tests/unit/apps/api/test_strategies_routes.py` | Cover manual paper API idempotency, DB-row creation via in-memory adapters, and inactive-run conflict mapping. | `none` production runtime |
| Created | `tests/unit/apps/migrations/test_manual_paper_orders_source_event_sql.py` | Guard additive migration contract. | `none` production runtime |
| Modified | `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | Cover manual paper no-dispatch risk branch. | `none` production runtime |
| Modified | `tests/unit/contexts/live_execution/test_paper_accounting_service.py` | Cover manual paper accounting idempotency. | `none` production runtime |
| Created | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/08-manual-entry-exit.md` | Stage report and local evidence. | `none` runtime |
| Modified | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | Stage status/evidence/handoff. | `none` runtime |
| Modified | `docs/architecture/README.md` | Docs index entry for Stage `08`; unrelated RL index line remains a pre-existing user change and is not part of Stage `08` staging. | `none` runtime |
| Deleted | none | No files deleted. | `none` |

## Blockers

None. Stage `08` is accepted.

## Handoff

Stage `09` may start. Stage `08` proves manual entry/exit controls, idempotent paper no-dispatch accounting, and fail-closed testnet manual behavior when the run/readiness boundary is not active. It does not authorize real testnet submit; Stage `09` still owns representative real testnet orders, exchange submit/fill/reconciliation proof, and must preserve the no-mainnet/no-auto-config/no-chat-secrets boundary.

The current `main` proof SHA `1a6be0ae` includes unrelated RL docs work in addition to Stage `08` commits; do not conflate those RL changes with Stage `08` implementation scope.
