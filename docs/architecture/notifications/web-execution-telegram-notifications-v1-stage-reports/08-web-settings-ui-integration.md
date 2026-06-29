# Stage 08: Web Settings UI Integration

Дата: `2026-06-29`

Статус: `completed-local`

Acceptance boundary: Stage `08` adds additive settings API/UI for Telegram binding status, scoped notification mode and weekly/monthly report schedule. Stage remains `completed-local` until implementation commit is published to `main`, GitHub CI/deploy passes, `macstudio` checkout is synchronized, production smoke passes and authenticated browser QA proves the changed settings flow.

## User Required Before Start

Required only if smoke auth env is missing or the user wants manual visual confirmation.

`ROEHUB_SMOKE_E2E_PASSWORD` presence was checked on `macstudio`; the value was not printed. No Telegram token, chat id, cookie, password or provider payload was printed.

## Scope

Implemented:

- additive `/ui/account/notifications/scoped` read/update API;
- existing `/ui/account/notifications` response kept compatible;
- settings projection for Telegram route `mode` values: `off`, `critical_only`, `signals`, `trades`, `reports`, `all`;
- weekly/monthly portfolio report schedule stored in `NotificationRoute.schedule_json`;
- Settings UI Telegram binding status without displaying raw chat id;
- Settings UI scoped mode dropdown and weekly/monthly report toggles;
- regression tests for API compatibility, redaction and web hooks.

Not implemented in this stage:

- real Telegram canary message;
- durable SQL-backed `NotificationRepository` adapter;
- production browser acceptance before the implementation commit is deployed.

## API And UI Behavior

| Behavior | Result |
|---|---|
| Legacy compatibility | `/ui/account/notifications` keeps the existing DTO shape and seven legacy items. |
| Scoped settings | `/ui/account/notifications/scoped` returns binding state, route mode, route status, masked recipient ref and report schedule. |
| Binding display | UI shows connected/unbound status without raw chat id. |
| Scoped modes | UI/API cover `critical_only`, `signals`, `trades`, `reports`, `all` and `off`. |
| Report schedule | UI/API update weekly/monthly toggles and timezone through `schedule_json`. |

## Validation

| Check | Result |
|---|---|
| Focused pytest: `tests/unit/apps/api/test_ui_account_routes.py::test_ui_account_scoped_notification_settings_are_additive_and_secret_safe` plus related binding/legacy/web-hook tests | passed: `4 passed` |
| `uv run pytest -q tests/unit/apps/api tests/unit/apps/web` | passed: `254 passed, 3 warnings` |
| `uv run ruff check apps/api apps/web src/trading/contexts/notifications tests/unit/apps` | passed |
| `uv run pyright apps/api apps/web src/trading/contexts/notifications tests/unit/apps` | passed |
| `node --check apps/web/dist/js/pages/settings.js` | passed |
| Browser QA | pending post-main: changed-code production browser proof requires implementation commit deployed to `macstudio` runtime |
| `uv run python -m tools.docs.generate_docs_index --check` | local check failed because the dirty checkout still contains unrelated untracked market-data docs; Stage `08` README entry was added and clean CI must re-check |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `compatible-change` | Adds `/ui/account/notifications/scoped`; existing account notifications endpoint remains unchanged. |
| DTO schema | `compatible-change` | Adds scoped notification DTOs; no fields removed or retyped from existing DTOs. |
| Ports | `compatible-change` | Adds `get_route` to `NotificationRepository` protocol and in-memory adapter. |
| Persisted schema | `none` | No migration changed. |
| Config/defaults | `none` | No runtime config key added. |
| Request/cache identity | `none` | No cache key or request hash changed. |
| Service-call semantics | `none` | No external provider call added. |
| External side effects | `compatible-change` | Scoped API upserts one user `NotificationRoute` in the configured notification repository. |
| Redaction/logging/reporting | `compatible-change` | UI/API expose only masked Telegram references and do not display raw chat id. |
| Browser-visible behavior | `compatible-change` | Settings integrations tab gains Telegram binding, scoped mode and report schedule controls. |
| Alerts/runbooks | `none` | No alert rule changed. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/user_settings.py` | created | Add user notification settings projection/update service. | `compatible-change` application surface |
| `src/trading/contexts/notifications/application/ports/notification_repository.py` | modified | Add route lookup needed for settings projection. | `compatible-change` port |
| `src/trading/contexts/notifications/adapters/outbound/persistence/in_memory_notification_repository.py` | modified | Implement route lookup. | `compatible-change` adapter |
| `src/trading/contexts/notifications/application/__init__.py` | modified | Export settings service types. | `compatible-change` application export |
| `apps/api/dto/ui_account.py` | modified | Add scoped notification DTOs. | `compatible-change` DTO |
| `apps/api/routes/ui_account.py` | modified | Add scoped notification read/update routes. | `compatible-change` API |
| `apps/web/templates/pages/settings.html` | modified | Add scoped endpoint hook. | `compatible-change` browser-visible |
| `apps/web/templates/fragments/account/notifications.html` | modified | Add binding/scoped/schedule controls. | `compatible-change` browser-visible |
| `apps/web/dist/js/pages/settings.js` | modified | Load/render/update scoped settings. | `compatible-change` browser-visible |
| `apps/web/dist/css/pages/settings.css` | modified | Layout for scoped notification controls. | `none` |
| `apps/web/locales/en.json` | modified | English UI copy. | `none` |
| `apps/web/locales/ru.json` | modified | Russian UI copy. | `none` |
| `tests/unit/apps/api/test_ui_account_routes.py` | modified | Cover scoped API, legacy compatibility and redaction. | `none` |
| `tests/unit/apps/web/test_security.py` | modified | Cover template/asset hooks and no chat id copy. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/08-web-settings-ui-integration.md` | created | Stage `08` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `08` local implementation and evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `08` report to docs index. | `none` |

## Residual Risks

- Production browser QA is still pending until the implementation commit is delivered to `main` and deployed.
- The scoped settings API uses the configured notification repository; current production wiring still defaults to in-memory notifications repository for this surface.
- Real Telegram binding/canary remains Stage `09`.
