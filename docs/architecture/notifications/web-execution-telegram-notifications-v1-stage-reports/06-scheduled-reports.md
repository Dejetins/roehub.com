# Stage 06: Scheduled Reports

Дата: `2026-06-29`

Статус: `accepted`

Acceptance boundary: Stage `06` добавляет provider-neutral weekly/monthly portfolio report scheduler. Stage accepted after implementation commit `e3d28f7811ecafd6b329d1b1f2d17cd77da4d55a` was published to `main`, GitHub CI/deploy passed, `macstudio` checkout synchronized to the same commit and production smoke passed.

## User Required Before Start

Nothing.

No Telegram token, chat id, admin route, password, cookie or provider payload was required or printed.

## Scope

Implemented:

- `NotificationReportScheduler` for active user `portfolio_report` routes;
- previous completed weekly/monthly period windows using route timezone, with explicit default timezone fallback;
- stable report dedupe key from user, route, report type, period start/end and scope digest;
- report runs with `complete`, `partial` or `unavailable` quality from the Stage `05` stats service;
- rendered fake/log delivery candidates for scheduled report runs;
- missed schedule metrics hook;
- worker wiring shell under `apps/worker/notification_report_scheduler`;
- seeded scheduler smoke proving report-run dedupe, delivery rows and dispatcher attempts through `LogOnlyNotificationProvider`.

Not implemented in this stage:

- production SQL report-run repository;
- real Telegram provider schedule delivery;
- web settings for schedule management;
- admin reports.

## Scheduler Behavior

| Behavior | Result |
|---|---|
| Active route discovery | Uses active user routes with `mode in {"reports", "all"}` and `portfolio_report` category. |
| Weekly period | Previous completed local week in route timezone, stored as UTC `period_start`/`period_end`. |
| Monthly period | Previous completed local month in route timezone, stored as UTC `period_start`/`period_end`. |
| Timezone | Uses route `schedule_json.timezone` or per-period timezone; otherwise records `timezone_source=default`. |
| Dedupe | Re-running the same scheduler period returns existing report runs and creates no duplicate deliveries. |
| Delivery | Creates `pending` `NotificationDelivery` rows with `report_run_id` and provider from the route. |
| Provider boundary | Stage proof uses `log_only`/`fake`; no real Telegram provider is required. |

## Real Boundary Evidence

Seeded scheduler smoke executed with application wiring, seeded stats rows and `LogOnlyNotificationProvider`:

| Evidence | Result |
|---|---|
| Weekly report run | created with `quality_status=complete` |
| Monthly report run | created with `quality_status=complete` |
| Duplicate scheduler run | `deduped_runs=2`, no duplicate report runs |
| Delivery rows | `deliveries=2` |
| Dispatcher attempts | `sent=2`, `attempts=2` |
| Smoke line | `stage06_scheduler_smoke=ok created_runs=2 deduped_runs=2 deliveries=2 sent=2 attempts=2 report_types=portfolio_monthly,portfolio_weekly qualities=complete,complete` |

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications/test_report_scheduler.py tests/unit/apps/test_notification_report_scheduler_wiring.py` | passed: `5 passed` |
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps` | passed: `400 passed, 3 warnings` |
| `uv run ruff check src/trading/contexts/notifications apps/worker/notification_report_scheduler tests/unit/contexts/notifications tests/unit/apps` | passed |
| `uv run pyright src/trading/contexts/notifications apps/worker/notification_report_scheduler tests/unit/contexts/notifications tests/unit/apps` | passed |
| Seeded scheduler smoke | passed: `stage06_scheduler_smoke=ok ... attempts=2` |
| `uv run python -m tools.docs.generate_docs_index --check` | local check expected to fail until unrelated untracked `market-data-live-tail-repair-v1` docs are either indexed or removed from the dirty checkout; this Stage `06` report entry was added to `docs/architecture/README.md` manually |
| GitHub CI for implementation commit `e3d28f7811ecafd6b329d1b1f2d17cd77da4d55a` | passed: run `28396303302` |
| GitHub deploy/image for implementation commit `e3d28f7811ecafd6b329d1b1f2d17cd77da4d55a` | passed: Backend `28396409107`, Web `28396409103`, App Image `28396409084` |
| `macstudio` checkout sync | passed: `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `e3d28f7811ecafd6b329d1b1f2d17cd77da4d55a` |
| `macstudio` production smoke | passed: `cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh` exited `0` |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No HTTP route changed. |
| DTO schema | `none` | No API DTO changed. |
| Ports | `compatible-change` | Adds report-route listing and report-run lookup methods to `NotificationRepository`; existing methods unchanged. |
| Application exports | `compatible-change` | Adds scheduler service/config/result and report renderer exports. |
| Persisted schema | `none` | No migration changed. |
| Config/defaults | `none` | Scheduler config is code-level only in this stage. |
| External service calls | `none` | No network/provider call added; fake/log provider only in tests and smoke. |
| External side effects | `compatible-change` | When invoked with active fake/log routes, scheduler creates report runs and delivery rows. |
| Browser-visible behavior | `none` | No UI changed. |
| Performance | `unknown` | SQL-backed route/report-run repository is future work; current proof uses seeded in-memory fixtures. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/report_scheduler.py` | created | Add idempotent scheduled weekly/monthly report scheduler and renderer. | `compatible-change` application surface |
| `src/trading/contexts/notifications/application/stats_query.py` | modified | Add explicit portfolio stats by window for timezone-aware report periods. | `compatible-change` application method |
| `src/trading/contexts/notifications/application/ports/notification_repository.py` | modified | Add report route listing and report-run dedupe lookup. | `compatible-change` port |
| `src/trading/contexts/notifications/adapters/outbound/persistence/in_memory_notification_repository.py` | modified | Implement report route listing and report-run dedupe lookup. | `none` production |
| `src/trading/contexts/notifications/application/__init__.py` | modified | Export scheduler surface. | `compatible-change` application export |
| `src/trading/contexts/notifications/__init__.py` | modified | Export scheduler surface at context root. | `compatible-change` context export |
| `apps/worker/notification_report_scheduler/__init__.py` | created | Add worker package shell. | `none` runtime until wired |
| `apps/worker/notification_report_scheduler/wiring/__init__.py` | created | Add scheduler composition helper for fixture/runtime wiring. | `compatible-change` app wiring |
| `tests/unit/contexts/notifications/test_report_scheduler.py` | created | Cover dedupe, period rendering, timezone/default handling and dispatcher attempt proof. | `none` |
| `tests/unit/apps/test_notification_report_scheduler_wiring.py` | created | Cover worker wiring composition. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/06-scheduled-reports.md` | created | Stage `06` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `06` local implementation and coverage evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `06` report to docs index. | `none` |

## Residual Risks

- Production SQL-backed route/report-run persistence remains future work before enabling scheduled reports outside seeded fixtures.
- Stage `06` does not enable real Telegram delivery; real provider proof remains Stage `09`.
- Schedule settings UI/API remains Stage `08`.
