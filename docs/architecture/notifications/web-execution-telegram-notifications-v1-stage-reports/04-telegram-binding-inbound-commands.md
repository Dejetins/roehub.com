# Stage 04: Telegram Binding And Inbound Commands

Дата: `2026-06-29`

Статус: `completed-local`

Acceptance boundary: Stage `04` добавляет безопасный web-generated Telegram binding code flow, redacted inbound Telegram update mapper, idempotent command handler and worker shell. Stage remains `completed-local` until implementation commit is published to `main`, GitHub CI/deploy passes, `macstudio` checkout is synchronized and production smoke passes.

## User Required Before Start

Synthetic command tests require nothing.

Real Telegram `/start <code>` smoke was skipped in this local pass because no user-driven Telegram canary was required before synthetic Stage `04` checks. No Telegram token, raw chat id, raw update payload, cookies or passwords were printed.

## Checkout And Branch

| Field | Value |
|---|---|
| Checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Branch | `main` |
| Branch/worktree/stash workflow | not created |
| Unrelated dirty work observed | yes: existing `.codex/*`, RL files and untracked market-data docs remain outside Stage `04` scope; Stage `04` staging must include only notifications/API/worker/config/tests/docs scoped files |

## Scope

Implemented:

- one-time Telegram binding code service with hashed code storage, TTL, owner binding and one-time confirmation;
- additive account API endpoints for Telegram binding status and binding-code generation;
- inbound Telegram update mapper that produces redacted `chat_id_ref` and does not keep raw provider payload;
- idempotent command handler for duplicate `telegram_update_id`;
- command coverage for `/start`, `/stats today|week|month`, `/strategy <id>`, `/exchange <connection>`, `/settings`, `/critical_only`, `/signals_on`, `/signals_off`, `/reports weekly on|off` and `/reports monthly on|off`;
- fail-closed strategy/exchange scope checks through an authorizer port;
- worker composition shell for Telegram bot config and credential-presence reporting as booleans only;
- disabled-by-default `telegram_bot` config in dev/test/prod.

Not implemented in this stage:

- real Telegram `/start` canary;
- Telegram long-poll network loop;
- SQL-backed command handler composition for the production worker;
- day/week/month stats query service; command responses return explicit unavailable placeholders until Stage `05`;
- web settings UI rendering; API-only binding surface is added here and UI is Stage `08`.

## Binding Security

| Requirement | Evidence |
|---|---|
| Binding code is one-time and owner-scoped | `NotificationTelegramBindingService.confirm_binding_code` consumes the code and stores confirmed owner by redacted `chat_id_ref`; reuse fails closed |
| Binding code is stored hashed | `test_start_binding_uses_hashed_one_time_code_and_idempotent_updates`; API test asserts returned code is not stored in the in-memory binding-code record |
| TTL enforced | `test_expired_binding_code_fails_closed` |
| User id text is not proof | `/start` only accepts generated binding code; owner comes from the code record, not from Telegram text |
| No raw chat id in application mapper | `TelegramUpdateMapper` emits `telegram_ref:<hash-prefix>:<last4>` |
| No raw provider payload evidence | tests and smoke use synthetic dicts and sanitized output only |

## Command Coverage

| Command | Local evidence | Result |
|---|---|---|
| `/start <code>` | `test_start_binding_uses_hashed_one_time_code_and_idempotent_updates` | binding confirmed; duplicate update ignored; code reuse failed |
| `/stats today` | `test_bound_command_coverage_creates_command_response_delivery` | handled with unavailable placeholder |
| `/stats week` | `test_bound_command_coverage_creates_command_response_delivery`; synthetic smoke | handled with unavailable placeholder |
| `/stats month` | `test_bound_command_coverage_creates_command_response_delivery` | handled with unavailable placeholder |
| `/strategy <id>` | `test_strategy_and_exchange_scopes_fail_closed_when_unauthorized` | unauthorized scope failed closed; authorized synthetic scope handled |
| `/exchange <connection>` | `test_strategy_and_exchange_scopes_fail_closed_when_unauthorized` | unauthorized scope failed closed; authorized synthetic scope handled |
| `/settings` | `test_bound_command_coverage_creates_command_response_delivery` | handled |
| `/critical_only` | `test_bound_command_coverage_creates_command_response_delivery` | handled |
| `/signals_on`, `/signals_off` | `test_bound_command_coverage_creates_command_response_delivery` | handled |
| `/reports weekly on`, `/reports monthly off` | `test_bound_command_coverage_creates_command_response_delivery` | handled |

## Real Boundary Evidence

Local synthetic smoke executed with no Telegram network:

| Evidence | Result |
|---|---|
| Binding code flow | generated code, stored as hash, confirmed through `/start` |
| Duplicate update | same `telegram_update_id` returned idempotent replay and did not create a second delivery |
| Bound command | `/stats week` created a pending command-response delivery |
| Output redaction | smoke output printed only statuses/counts and a boolean hash-storage check |
| Smoke line | `stage04_telegram_smoke=ok bind_status=handled duplicate_replay=True stats_status=handled updates=2 deliveries=2 code_stored_as_hash=True` |

Real Telegram binding smoke: skipped.

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/api/test_ui_account_routes.py::test_ui_account_telegram_binding_code_and_status_are_secret_safe tests/unit/apps/worker/test_telegram_bot_worker_wiring.py` | passed: `35 passed` |
| `uv run ruff check src/trading/contexts/notifications apps/api apps/worker/telegram_bot_worker tests/unit/contexts/notifications tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/worker/test_telegram_bot_worker_wiring.py` | passed |
| `uv run pyright src/trading/contexts/notifications apps/api apps/worker/telegram_bot_worker tests/unit/contexts/notifications tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/worker/test_telegram_bot_worker_wiring.py` | passed |
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/api tests/unit/apps` | passed: `244 passed` |
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/api tests/unit/apps/worker` | passed: `274 passed` |
| `uv run ruff check src/trading/contexts/notifications apps/api apps/worker tests/unit/contexts/notifications tests/unit/apps` | passed |
| `uv run pyright src/trading/contexts/notifications apps/api apps/worker tests/unit/contexts/notifications tests/unit/apps` | passed |
| Synthetic command/binding smoke | passed: `stage04_telegram_smoke=ok ... code_stored_as_hash=True` |
| `uv run python -m tools.docs.generate_docs_index --check` | local check failed because the dirty checkout contains unrelated untracked `market-data-live-tail-repair-v1` docs; generated diff inspection showed the Stage `04` README entry matches the generator and only those unrelated market-data entries remain missing |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `compatible-change` | Adds Telegram binding status and binding-code endpoints under existing UI account router. Existing account endpoints remain stable. |
| DTO schema | `compatible-change` | Adds response DTOs only. |
| Ports | `compatible-change` | Adds Telegram binding service/store protocol, command authorizer protocol and repository read method for Telegram updates. |
| Persisted schema | `none` | No migration changed in Stage `04`; Stage `01` already added `notification_telegram_updates`. |
| Config/defaults | `compatible-change` | Adds disabled-by-default `telegram_bot` config. |
| External service calls | `none` | No real Telegram network call is made by this stage. |
| External side effects | `compatible-change` | Command responses create internal `NotificationDelivery` rows in the repository abstraction; real send is still dispatcher/provider-gated. |
| Logs/metrics/audit/redaction | `compatible-change` | Credential presence is boolean-only; mapper redacts chat references and tests avoid raw payload evidence. |
| Browser-visible behavior | `none` | API endpoints are added, but no UI is rendered in this stage. |
| Performance | `unknown` | No production polling or DB benchmark in Stage `04`. |

## Business Impact

| Layer | Impact | Notes |
|---|---|---|
| User notifications | additive foundation | Users can receive a one-time code from API and bind Telegram through `/start` once UI/polling are connected. |
| User self-service stats | partial foundation | Commands exist but stats content remains unavailable until Stage `05`. |
| Admin notifications | no direct change | Admin command/report route work remains Stage `07`. |
| Trading boundary | no order or exchange side effect | Telegram commands are read/settings/report controls only; no order submission path is added. |
| Secret handling | compatible improvement | Binding code and Telegram chat evidence are redacted or hashed in local tests and docs. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/telegram_binding.py` | created | Add one-time binding code service and in-memory store. | `compatible-change` application surface |
| `src/trading/contexts/notifications/application/telegram_commands.py` | created | Add inbound command handler, idempotency and scope checks. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/inbound/telegram/update_mapper.py` | created | Add redacted Telegram update mapper. | `compatible-change` inbound adapter |
| `src/trading/contexts/notifications/application/ports/notification_repository.py` | modified | Add Telegram update read method for idempotency. | `compatible-change` port |
| `src/trading/contexts/notifications/adapters/outbound/persistence/in_memory_notification_repository.py` | modified | Add idempotent Telegram update recording/read support. | `compatible-change` test/support adapter |
| `apps/api/dto/ui_account.py` | modified | Add Telegram binding response DTOs. | `compatible-change` DTO |
| `apps/api/routes/ui_account.py` | modified | Add Telegram binding status and code endpoints. | `compatible-change` API |
| `apps/api/wiring/modules/ui_account.py` | modified | Wire binding service into the UI account router. | `compatible-change` API composition |
| `apps/worker/telegram_bot_worker/` | created | Add worker shell config and command-handler composition. | `compatible-change` runtime surface, disabled by config |
| `configs/dev/notifications.yaml` | modified | Add disabled Telegram bot worker config. | `compatible-change` config |
| `configs/test/notifications.yaml` | modified | Add disabled Telegram bot worker config. | `compatible-change` config |
| `configs/prod/notifications.yaml` | modified | Add disabled Telegram bot worker config. | `compatible-change` config |
| `tests/unit/contexts/notifications/test_telegram_commands.py` | created | Cover binding, idempotency, command matrix and fail-closed scopes. | `none` |
| `tests/unit/contexts/notifications/test_telegram_update_mapper.py` | created | Cover update mapping and chat ref redaction. | `none` |
| `tests/unit/apps/api/test_ui_account_routes.py` | modified | Cover Telegram binding API and secret-safe code storage. | `none` |
| `tests/unit/apps/worker/test_telegram_bot_worker_wiring.py` | created | Cover config loading, credential presence and command handler composition. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/04-telegram-binding-inbound-commands.md` | created | Stage `04` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `04` local implementation and evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `04` report to docs index. | `none` |

## Residual Risks

- Real Telegram `/start` canary was skipped; Stage `09` remains the production canary boundary.
- SQL-backed Telegram polling/command worker wiring is not enabled in this local Stage `04` implementation; current production defaults remain disabled.
- Binding-code store is in-memory in this stage's composition. Production persistence or identity-table write-through must be decided before enabling real user binding.
- Stats command responses intentionally do not invent metrics; Stage `05` must replace unavailable placeholders with quality-aware stats snapshots.
