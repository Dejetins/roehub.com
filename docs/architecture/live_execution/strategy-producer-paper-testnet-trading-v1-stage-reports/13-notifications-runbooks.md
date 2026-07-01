# Stage 13: Notifications And Operator Runbooks

Статус: `accepted`.

Дата: `2026-07-02`.

## Pre-Start

User required before start: nothing.

No password, cookie, token, DSN, exchange key, raw credential, raw provider
payload, raw session value, or secret-bearing browser state was printed or
written to this report.

Previous stage ledger gate: `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`
was read before implementation. The Stage Status table records:

| Gate | Ledger status | Evidence |
|---|---|---|
| `12.1` Readiness gate | `accepted` | Scoped Testnet subject, producer enablement, API/DB/Redis/Monit/Prometheus/RSS readiness, no mainnet order growth. |
| `12.2` Functional canary | `accepted` | `32m03s` rerun with `+32` signals and `+32` execution source events, Redis pending/lag `0`, browser/API proof, no intents/orders/mainnet rows. |
| `12.3` Burst/resource gate | `accepted` | `180` controlled `testnet` strategies, no retry/DLQ growth, no production intent/order/mainnet deltas, resource recovery passed. |
| `12.4` Sustained 6h soak | `accepted` | Fixed collector artifact completed `21600s` / `7` snapshots with `360` candles/signals/source events and browser/API proof. |
| `12.5` Closure | `accepted` | Current run `c665f9e7-b4a6-4ede-83ee-b33a311f0ef4` reached `running`; final proof had fresh signals/source events, Redis pending/lag `0/0`, execution pending `0`, no order/mainnet/unknown growth, and authenticated dashboard API `200`. |

Stage `13` starts from the accepted Stage `12` chain and keeps the same
`paper,testnet` only / no-mainnet boundary.

## Операторский Итог

Stage `13` принят как delivery-neutral слой уведомлений и runbook-реакций для
strategy producer. Теперь оператор видит не только общий reject, а отдельные
факты для rejected signal, rejected order, manual exit, reconciliation pending,
strategy lifecycle, soak result и resource threshold breach. Эти факты пишутся
в `execution_notification_outbox`, попадают в Prometheus alert groups и имеют
описанные действия в runbook. Реальная отправка в Telegram/email по-прежнему не
включалась и не доказывалась; это отдельный canary с отдельным approval.

## Concrete File Plan Recorded Before Implementation

Broad prompt paths were narrowed to this concrete Stage `13` plan:

| File | Planned reason |
|---|---|
| `src/trading/contexts/live_execution/domain/notification.py` | Widen delivery-neutral outbox event type contract. |
| `apps/api/dto/ui_execution.py` | Accept new event types through `/api/ui/execution/notifications`. |
| `src/trading/contexts/live_execution/application/use_cases/execution_ingress.py` | Classify rejected signal/order/manual-exit notifications without changing source-event shape. |
| `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py` | Emit specific order-rejected notification for order guard failures. |
| `alembic/versions/20260702_0039_execution_notification_stage13_event_types.py` | Widen Postgres `execution_notification_event_type_chk`. |
| `src/trading/contexts/notifications/application/source_router.py` | Keep future delivery-router synthetic compatibility for new outbox types. |
| `infra/macos/prometheus/rules/strategy-producer.rules.yml` | Add Stage `13` alert rules with severity/owner/escalation/runbook/action. |
| `infra/macos/prometheus/rules/live-execution-stage17.rules.yml` | Keep unknown-state alert compatible with `producer_reconciliation_pending`. |
| `docs/runbooks/strategy-live-worker.md` | Add operator actions and non-destructive runbook dry-run. |
| `docs/architecture/operations/native-service-control-monitoring-admin-target-v1.md` | Link operations target to Stage `13` outbox alert coverage. |
| Focused tests under `tests/unit/contexts/live_execution`, `tests/unit/apps`, `tests/unit/contexts/notifications`, `tests/unit/infra` | Lock API, persistence constraint, router compatibility, and alert-rule validation. |
| This report and the stage ledger | Record evidence, manifest, contract impact, and next-stage handoff. |

## Implementation Summary

The outbox contract remains delivery-neutral. Telegram/email delivery channels
are still out of scope.

Implemented event-type coverage:

| Required state | Outbox event type | Default severity | Emission/proof path |
|---|---|---:|---|
| Rejected signal | `producer_signal_rejected` | `warning` | Automatic for `strategy_signal` risk rejection. |
| Rejected order | `producer_order_rejected` | `warning` | Automatic for order-model rejection and exchange guard rejection. |
| Fill | `producer_fill` | `info` | Existing exchange execution fill path retained. |
| Manual exit | `producer_manual_exit` | `info` | Automatic when a `manual_request` source event has `action=exit` or `manual:exit:*` source ref. |
| Kill switch | `producer_kill_switch` | `critical` | Existing risk gate kill-switch path retained. |
| Unknown/reconciliation pending | `producer_unknown`, `producer_reconciliation_pending` | `critical` | Existing unknown-state path retained; new explicit event type added for dry-run/future delivery compatibility. |
| Strategy stopped/restarted | `producer_strategy_stopped`, `producer_strategy_restarted` | `info`/operator-selected | Added to API/domain/DB contract for future strategy lifecycle emitters. |
| 6h soak failure/success | `producer_soak_failed`, `producer_soak_succeeded` | `critical`/`info` | Added to API/domain/DB contract for soak collectors and runbook dry-run. |
| Resource threshold breach | `producer_resource_threshold_breached` | `critical` | Added to API/domain/DB contract for load/soak collectors and runbook dry-run. |

Payload/redaction behavior remains bounded by `sanitize_notification_labels`:
secret-like label keys are rejected, labels are size-limited, and the report
records only sanitized ids/reasons.

## Business Impact

| Layer | Impact |
|---|---|
| Operator confidence | Producer incidents are no longer collapsed into one generic rejection bucket: signal rejection, order rejection, manual exit, reconciliation pending, soak result, lifecycle state, and resource threshold breach have explicit outbox facts and alert/runbook routes. |
| Release risk | The change is additive and delivery-neutral. It widens accepted notification event values and alert classification without enabling Telegram/email sends or changing exchange submit behavior. |
| Incident response | Critical cases now route through `StrategyProducerCriticalIncidentNotification`; warning/rejection and lifecycle cases have separate Prometheus alerts with owner, escalation, runbook, and action annotations. |
| Customer-visible behavior | Browser UI did not change. The API notification list can now contain the new event-type strings, so operator-facing consumers get more precise status facts without new side effects. |
| Money safety | No mainnet, provider submit, credential, or exchange state mutation path was added. Runtime proof used only `ops_test` pending outbox rows. |

## Alert And Runbook Matrix

| Alert | Severity | Owner | Event surface | Runbook action |
|---|---|---|---|---|
| `StrategyProducerExecutionRejected` | `warning` | `strategy-producer` | `producer_rejected`, `producer_signal_rejected`, `producer_order_rejected` | Inspect outbox/source/intent/risk/order guard rows before replay. |
| `StrategyProducerCriticalIncidentNotification` | `critical` | `strategy-producer` | `producer_kill_switch`, `producer_unknown`, `producer_reconciliation_pending`, `producer_soak_failed`, `producer_resource_threshold_breached` | Stop broader fan-out, keep mainnet disabled, preserve evidence, reconcile before retry. |
| `StrategyProducerRunStateNotification` | `warning` | `strategy-producer` | `producer_manual_exit`, `producer_strategy_stopped`, `producer_strategy_restarted`, `producer_soak_succeeded`, `producer_terminal`, `producer_fill` | Verify expected operator action/test window; investigate duplicates or unexpected loops. |
| `LiveExecutionUnknownState` | `critical` | `live-execution` | `producer_unknown`, `producer_reconciliation_pending`, `unknown_needs_reconciliation` | Existing exchange-execution runbook path widened for the new event type. |

Runbook dry-run is non-destructive: emit a dry-run `ops_test` notification
through the existing UI execution notification API or local TestClient, verify
`status=pending` and redacted labels, run Prometheus rule validation, and do
not send Telegram/email or mutate exchange state.

## Local API / Outbox Evidence

| Evidence | Result |
|---|---|
| `test_ui_execution_notifications_accept_stage13_event_types` | API route accepts and lists all new Stage `13` event types through `/ui/execution/notifications` using `ops_test` dry-run payloads. |
| `test_stage13_notification_event_types_are_supported` | Domain/use case accepts all new event types and stores sanitized labels in the outbox repository. |
| `test_manual_exit_paper_no_exchange_submit_emits_manual_exit_notification` | Manual exit source event with `action=exit` emits `producer_manual_exit` with `severity=info`. |
| `test_rejects_unsupported_order_model_and_links_source_event_outcome` | Unsupported order model emits `producer_order_rejected` while keeping source-event outcome `order_model_rejected`. |
| `test_testnet_adapter_rejects_mainnet_connection` | Mainnet hard-block order guard emits `producer_order_rejected` and preserves money-safety fail-closed behavior. |

Production API/SQL and SQL-visible outbox proof passed after direct-main
delivery, migration, deploy, Mac Studio checkout sync, and runtime smoke.

## Post-Main Runtime Evidence

| Evidence | Result |
|---|---|
| Implementation commit | `5f471819f43fd8a1d18cf42d5a15f1b35ac8860c` (`Add strategy producer notification runbooks`) delivered Stage `13` code, migration, alert rules, runbook, tests, and report to `origin/main`. |
| Current main after parallel work | `2f5b3833bf706a287f5e70caea934942d595c090` includes Stage `13` as an ancestor plus unrelated RL Stage `08I3` work. Stage `13` scope did not modify RL files. |
| GitHub CI/deploy for Stage `13` commit | CI `28552271738`, Deploy Backend `28552366865`, Publish App Image `28552366900`, Deploy Web `28552366886`/`28552374322` succeeded for `5f471819`. |
| GitHub CI/deploy for current main | CI `28552442114`, Deploy Backend `28552646604`, Publish App Image `28552646569`, Deploy Web `28552646584`/`28552655717` succeeded for `2f5b3833`. |
| Mac Studio checkout sync | `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded from `34db3fcc` to `2f5b3833`; `git status --short --branch` reported `## main...origin/main`. |
| Runtime files | `/opt/roehub/app/alembic/versions/20260702_0039_execution_notification_stage13_event_types.py`, `/opt/roehub/config/prometheus.rules/strategy-producer.rules.yml`, and `/opt/roehub/config/prometheus.rules/live-execution-stage17.rules.yml` were present. |
| Runtime migration | Production Alembic version was `20260702_0039`; `execution_notification_event_type_chk` contained all Stage `13` event values. |
| Runtime outbox dry-run | Marker `stage13-runtime-20260701T224420Z` created `11` `ops_test` `pending` outbox rows covering `producer_signal_rejected`, `producer_order_rejected`, `producer_fill`, `producer_manual_exit`, `producer_kill_switch`, `producer_reconciliation_pending`, `producer_strategy_stopped`, `producer_strategy_restarted`, `producer_soak_failed`, `producer_soak_succeeded`, and `producer_resource_threshold_breached`; labels were sanitized dry-run labels only. |
| Prometheus loaded rules | Prometheus API returned `StrategyProducerExecutionRejected`, `StrategyProducerCriticalIncidentNotification`, `StrategyProducerRunStateNotification`, and `LiveExecutionUnknownState`. |
| Mac Studio smoke | `bash scripts/macos/smoke_prod.sh` exited `0` after current-main deploy/sync. |

No Telegram/email delivery channel was enabled or exercised. No exchange submit
or mainnet path was touched by the dry-run.

## Conditional Service-Call Coverage

| Caller / callee | Purpose | Evidence | Failure behavior |
|---|---|---|---|
| API/UI execution route -> `ExecutionIngressService` -> outbox repository | Accept delivery-neutral dry-run notification event types and list recent notifications. | `test_ui_execution_notifications_accept_stage13_event_types` covered all new Stage `13` values through the FastAPI route. | Unsupported event types still fail validation; labels with secret-like keys are rejected. |
| `ExecutionIngressService` -> Postgres outbox | Persist new event types under the widened CHECK constraint. | Runtime marker `stage13-runtime-20260701T224420Z` wrote `11` `pending` rows under Alembic `20260702_0039`; constraint definition included all Stage `13` values. | Duplicate identity remains the existing outbox dedupe; failed inserts surface as DB errors and do not silently drop proof. |
| Prometheus -> rule files | Classify outbox events into warning/critical/lifecycle alert paths. | Prometheus API loaded `StrategyProducerExecutionRejected`, `StrategyProducerCriticalIncidentNotification`, `StrategyProducerRunStateNotification`, and `LiveExecutionUnknownState`. | Missing rule load would block acceptance; no alert delivery channel was assumed. |
| Telegram/email/provider delivery | Out of scope for Stage `13`. | No provider was enabled; no delivery provider call was made. | Real delivery still requires a separately approved canary/readiness path. |

## Prometheus Rule Validation

Local `promtool` is not installed in this environment. Repository validation
used the existing YAML/unit-test rule path:

| Check | Result |
|---|---|
| YAML parse for all `infra/macos/prometheus/rules/*.yml` | passed; `strategy-producer.rules.yml` has `6` alerts and `live-execution-stage17.rules.yml` has `9` alerts. |
| `uv run pytest -q tests/unit/infra/test_monitoring_assets.py` | passed: `9 passed`; asserts alert names, severity/owner labels, runbook anchors, escalation, and action annotations. |

## Quality Gates

| Gate | Result |
|---|---:|
| `uv run pytest -q tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_notifications_producers_sql.py tests/unit/contexts/notifications/test_source_router.py` | passed: `61 passed` |
| `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/notifications apps/api tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_notifications_producers_sql.py tests/unit/contexts/notifications/test_source_router.py tests/unit/infra/test_monitoring_assets.py alembic/versions/20260702_0039_execution_notification_stage13_event_types.py` | passed |
| `uv run pyright src/trading/contexts/live_execution/application/use_cases/execution_ingress.py src/trading/contexts/live_execution/domain/notification.py apps/api/dto/ui_execution.py src/trading/contexts/notifications/application/source_router.py tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py` | passed: `0 errors` |
| `uv run ruff check src/trading/contexts/live_execution apps tests` | passed |
| `uv run pyright src/trading/contexts/live_execution apps tests` | passed: `0 errors` |
| `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps` | passed: `438 passed, 3 warnings` |
| `uv run pytest -q tests/unit/contexts/notifications/test_source_router.py tests/unit/infra/test_monitoring_assets.py` | passed: `13 passed` |
| `python -m tools.docs.generate_docs_index --check` | passed |
| `uv run ruff check .` | passed |
| `uv run pyright` | passed: `0 errors` |
| `uv run pytest -q -ra` | passed: `1488 passed, 3 warnings` |
| `uv run python -m tools.docs.generate_docs_index --check` | passed |
| Direct-main delivery / CI / Mac Studio sync / runtime smoke | passed: Stage `13` implementation commit `5f471819`; current main `2f5b3833`; CI/deploy green; Mac Studio checkout synced; production smoke exited `0`; runtime SQL/outbox proof marker `stage13-runtime-20260701T224420Z` created `11` pending rows. |

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `compatible-change` | `/ui/execution/notifications` accepts additional `event_type` values. Existing values and response shape remain unchanged. |
| Port contract | `none` | `ExecutionIntentRepository` and use-case command shapes are unchanged. |
| DTO schema | `compatible-change` | Request validation widens the notification event-type enum. |
| Persisted schema | `compatible-change` | New migration widens only `execution_notification_event_type_chk`; no table/column rewrite or data deletion. |
| Config schema/defaults | `none` | No config default changed. |
| Request hash / cache key / persistence identity | `none` | Source-event, intent, and outbox dedupe identities are unchanged. |
| Service-call auth/timeout/retry/error semantics | `none` | No service-call boundary changed. |
| External side-effect semantics | `none` | Outbox rows remain `pending`; no Telegram/email/exchange side effect is added. |
| Logs, metrics, traces, audit, ledger, report, redaction semantics | `compatible-change` | Metrics/alerts can now classify explicit Stage `13` event types; label redaction stays enforced. |
| Alert/runbook semantics | `compatible-change` | Adds strategy producer notification alert rules and runbook actions; widens unknown-state alert compatibility. |
| Browser-visible behavior | `none` | No browser UI file changed; API notification lists can contain new event-type strings. |
| Performance risk on verified hot path | `none` | No compute hot path or tight loop was changed; notification classification is small control-flow logic. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `alembic/versions/20260702_0039_execution_notification_stage13_event_types.py` | none | none | Widen persisted outbox event-type CHECK constraint. | `compatible-change`: additive allowed values. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/13-notifications-runbooks.md` | none | none | Stage `13` report and evidence ledger. | `none`: documentation/handoff. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark Stage `13` accepted and open Stage `14`. | `none`: documentation/handoff. |
| none | `docs/architecture/README.md` | none | Regenerated architecture docs index for accepted Stage `13` status. | `none`: generated docs index. |
| none | `src/trading/contexts/live_execution/domain/notification.py` | none | Add Stage `13` event types to domain contract. | `compatible-change`. |
| none | `apps/api/dto/ui_execution.py` | none | Allow API dry-run/outbox emission for new event types. | `compatible-change`: widened enum. |
| none | `src/trading/contexts/live_execution/application/use_cases/execution_ingress.py` | none | Emit specific rejected signal/order/manual-exit notifications. | `compatible-change`: more specific outbox facts. |
| none | `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py` | none | Emit `producer_order_rejected` for order guard failures. | `compatible-change`: more specific outbox facts. |
| none | `src/trading/contexts/notifications/application/source_router.py` | none | Keep future delivery-router synthetic coverage for new live-execution event types. | `compatible-change`; outside primary path but required by delivery compatibility. |
| none | `infra/macos/prometheus/rules/strategy-producer.rules.yml` | none | Add Stage `13` notification alert rules. | `compatible-change`: additive alerts. |
| none | `infra/macos/prometheus/rules/live-execution-stage17.rules.yml` | none | Widen unknown-state alert for `producer_reconciliation_pending`. | `compatible-change`: additive alert match. |
| none | `docs/runbooks/strategy-live-worker.md` | none | Add Stage `13` operator actions and dry-run procedure. | `compatible-change`: operational runbook. |
| none | `docs/architecture/operations/native-service-control-monitoring-admin-target-v1.md` | none | Link operations target to Stage `13` alert coverage. | `none`: documentation alignment. |
| none | `tests/unit/contexts/live_execution/test_execution_ingress_service.py` | none | Cover event classification and Stage `13` event support. | `none`: tests. |
| none | `tests/unit/contexts/live_execution/test_exchange_execution_process.py` | none | Cover order-guard outbox event. | `none`: tests. |
| none | `tests/unit/apps/api/test_ui_execution_routes.py` | none | Cover API acceptance/listing of Stage `13` event types. | `none`: tests. |
| none | `tests/unit/apps/migrations/test_execution_notifications_producers_sql.py` | none | Cover migration constraint widening. | `none`: tests. |
| none | `tests/unit/contexts/notifications/test_source_router.py` | none | Cover delivery-router synthetic compatibility. | `none`: tests. |
| none | `tests/unit/infra/test_monitoring_assets.py` | none | Cover strategy-producer alert rule names and annotations. | `none`: tests. |

Foreign changes intentionally excluded from Stage `13`: existing RL/ML diffs
under `docs/architecture/ml`, `src/trading/contexts/rl_trading`,
`scripts/rl_trading`, and related tests, plus any pre-existing
`docs/architecture/README.md` hunk not produced by this stage.

## Cold Review

| Field | Result |
|---|---|
| Review mode | Cold self-review fallback. A separate subagent review was not used because current multi-agent policy allows spawning only after an explicit user request for delegation. |
| Verdict | Release after fixes. Stage `13` report, ledger, docs index, runtime evidence, contract impact, and next prompt are coherent. |
| Fixed blockers | Added Russian operator summary, business impact, and conditional service-call coverage; updated stale `12.5` ledger helper rows that still described the old blocked rerun; regenerated and checked `docs/architecture/README.md`. |
| Follow-up check | `uv run python -m tools.docs.generate_docs_index --check` passed; `git diff --check` passed; focused consistency search found only historical blocked rows in earlier Stage `12`/`12.4`/`12.5` reports and the historical 2026-07-01 change-log row. |
| Residual risks | No Telegram/email delivery proof was attempted or claimed. Final acceptance docs commit SHA is recorded by the executor handoff rather than inside this self-referential report. |

## Next Action

Stage `13` is accepted. Next prompt to run:
`.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/14-final-readiness-docs-closure.md`.

Stage `14` must start from current main `2f5b3833` or newer, preserve the
Stage `13` delivery-neutral boundary, and treat the runtime dry-run rows under
marker `stage13-runtime-20260701T224420Z` as operator-runbook proof only, not
as notification delivery proof.
