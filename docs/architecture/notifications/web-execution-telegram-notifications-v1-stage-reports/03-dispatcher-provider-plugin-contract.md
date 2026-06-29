# Stage 03: Dispatcher And Provider Plug-in Contract

Дата: `2026-06-29`

Статус: `completed-local`

Acceptance boundary: Stage `03` добавляет provider-neutral dispatcher для `NotificationDelivery`: claim/lease, попытки отправки, retry/backoff, `unknown`, `dead_letter`, `suppressed`, fake/log provider и Telegram Bot API provider за безопасной конфигурацией. Stage остается `completed-local` до публикации в `main`, green CI/deploy и синхронизации `macstudio`.

## User Required Before Start

Nothing.

Реальный Telegram token, chat id, admin route или binding не требовались и не запрашивались. Telegram provider реализован за `enabled=false` defaults; real Telegram send не выполнялся.

## Checkout And Branch

| Field | Value |
|---|---|
| Checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Branch | `main` |
| Branch/worktree/stash workflow | not created |
| Unrelated dirty work observed | yes: existing `.codex/*`, RL files and untracked market-data docs remain outside Stage `03` scope; Stage `03` staging must include only notifications/config/tests/docs scoped files |

## Scope

Implemented:

- `NotificationProvider` port and `NotificationDispatcher` claim/lease loop;
- deterministic delivery attempt recording for `sent`, `retry`, `unknown`, `dead_letter`, `suppressed` and provider-missing outcomes;
- retry scheduling with max-attempt dead-letter behavior;
- expired lease reclaim while active claims stay untouched;
- fake/log provider adapters for non-network delivery proof;
- Telegram Bot API provider behind explicit config, with disabled-by-default behavior, timeout and 5xx mapped to `unknown`, rate limit mapped to `retry`, and redacted request/response hashes;
- worker composition root and Prometheus metrics for claimed deliveries, results, latency, pending age and unknown count;
- dev/test/prod notification config defaults with dispatcher disabled, log-only mode and Telegram disabled.

Not implemented in this stage:

- real Telegram canary or production Telegram send;
- inbound Telegram updates, binding codes or `/stats` commands;
- stats query service and scheduled reports;
- persistent SQL repository implementation for dispatcher workers.

## Delivery Lifecycle Evidence

| Scenario | Evidence |
|---|---|
| `pending -> claimed -> sent` | `test_dispatcher_claims_pending_delivery_and_marks_sent`; composition root smoke drains log-only backlog to `sent` |
| `pending -> retry -> dead_letter` | `test_dispatcher_schedules_retry_until_attempt_budget_is_exhausted` |
| expired claim reclaim | `test_dispatcher_reclaims_expired_claim_without_double_sending_active_claim` |
| active claim is not double-sent | `test_dispatcher_reclaims_expired_claim_without_double_sending_active_claim` |
| provider timeout / ambiguous state | `test_dispatcher_marks_unknown_without_blind_retry`; `test_telegram_timeout_and_5xx_are_unknown_states` |
| missing provider | `test_dispatcher_dead_letters_missing_provider` |
| disabled Telegram provider | `test_telegram_provider_disabled_suppresses_without_network_call` |
| Telegram rate limit | `test_telegram_rate_limit_retries_with_retry_after` |
| metrics surface | `test_composition_root_drains_backlog_with_log_only_provider`; real-boundary smoke checked all Stage `03` metric names |

## Real Boundary Evidence

Local composition-root smoke executed against the in-memory repository integration fixture with no Telegram network:

| Evidence | Result |
|---|---|
| Composition root config | `configs/test/notifications.yaml` loaded through `load_notification_dispatcher_runtime_config` |
| Repository | `InMemoryNotificationRepository` with three pending deliveries |
| Providers | synthetic `log_only=sent`, `fake=unknown`, `telegram_bot_api=dead_letter` test providers |
| Transitions | `claimed=3`, `sent=1`, `unknown=1`, `dead_letter=1`, `attempts=3` |
| Metrics | `notification_dispatcher_deliveries_claimed_total`, `notification_dispatcher_delivery_results_total`, `notification_dispatcher_delivery_latency_seconds`, `notification_dispatcher_pending_age_seconds`, `notification_dispatcher_unknown_deliveries` present |
| External effects | no real Telegram call; no token/chat id printed |

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/worker/test_notification_dispatcher_wiring.py` | passed: `22 passed` |
| `uv run ruff check src/trading/contexts/notifications apps/worker/notification_dispatcher tests/unit/contexts/notifications tests/unit/apps` | passed |
| `uv run pyright src/trading/contexts/notifications apps/worker/notification_dispatcher tests/unit/contexts/notifications tests/unit/apps` | passed |
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps` | passed: `374 passed, 3 warnings` |
| Real-boundary composition-root smoke | passed: `stage03_dispatcher_smoke=ok claimed=3 sent=1 unknown=1 dead_letter=1 attempts=3` |
| `uv run python -m tools.docs.generate_docs_index --check` | local check failed because the dirty checkout contains unrelated untracked `market-data-live-tail-repair-v1` docs; generated diff inspection showed the Stage `03` README entry matches the generator and only those unrelated market-data entries remain missing |
| GitHub CI/deploy/host sync | pending |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No HTTP routes or browser-visible DTOs changed. |
| DTO schema | `none` | No DTOs changed. |
| Ports | `compatible-change` | Adds provider port and additive repository methods for due delivery listing, status counting and updates. |
| Persisted schema | `none` | No migration or table shape changed in this stage. |
| Config/defaults | `compatible-change` | Adds `configs/*/notifications.yaml`; dispatcher and Telegram are disabled by default. |
| External service calls | `compatible-change` | Telegram adapter exists but performs no call unless explicitly enabled and configured. |
| External side effects | `compatible-change` | Default fake/log path is local-only; Telegram send remains gated. |
| Logs/metrics/audit/redaction | `compatible-change` | Adds redacted provider hashes and dispatcher metrics. Env credential presence is reported only as booleans. |
| Browser-visible behavior | `none` | No UI/API surface changed. |
| Performance | `unknown` | No production backlog benchmark yet; Stage `03` proves bounded batch/lease semantics and metrics only. |

## Business Impact

| Layer | Impact | Notes |
|---|---|---|
| User notifications | no direct user-visible change | Dispatcher can process delivery rows, but all runtime defaults are disabled/log-only and no Telegram canary is claimed. |
| Admin notifications | no direct admin-visible change | Admin delivery rows will use the same provider contract when later stages create routes/events. |
| Trading boundary | no order or exchange side effect | Stage `03` does not touch order submission, exchange credentials or strategy execution. |
| Support/debugging | compatible additive improvement | Attempts, terminal state and `unknown` state are now explicit in dispatcher behavior and tests. |
| Secret handling | compatible additive improvement | Telegram credential presence is boolean-only in config helpers; provider results carry redacted hashes, not raw provider payloads. |

## Alerts Monitoring Runbook Coverage

| Surface | Coverage |
|---|---|
| Metrics | Implemented Prometheus metrics for claimed deliveries, delivery results, delivery latency, pending age and unknown delivery count. |
| Alerts | N/A for Stage `03`; alert thresholds and admin runbooks are Stage `07`. |
| Runbooks | N/A for Stage `03`; operational runbook publication is deferred to Stage `07`/`09`. |
| Real provider monitoring | N/A for Stage `03`; Telegram canary and production provider health proof are Stage `09`. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/ports/notification_provider.py` | created | Add provider adapter interface and result contract. | `compatible-change` port |
| `src/trading/contexts/notifications/application/ports/notification_repository.py` | modified | Add due-delivery listing, update and status-count methods required by dispatcher. | `compatible-change` port |
| `src/trading/contexts/notifications/application/dispatcher.py` | created | Add claim/lease/retry/unknown/dead-letter dispatch loop. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/outbound/persistence/in_memory_notification_repository.py` | modified | Support dispatcher lifecycle in repository integration fixture. | `compatible-change` test/support adapter |
| `src/trading/contexts/notifications/adapters/outbound/providers/log_only_notification_provider.py` | created | Add deterministic fake/log send provider. | `compatible-change` support adapter |
| `src/trading/contexts/notifications/adapters/outbound/providers/telegram_bot_api_notification_provider.py` | created | Add gated Telegram provider adapter with redaction and unknown-state semantics. | `compatible-change` provider adapter |
| `apps/worker/notification_dispatcher/` | created | Add worker package and composition root wiring. | `compatible-change` runtime surface, disabled by config |
| `configs/dev/notifications.yaml` | created | Add disabled safe defaults. | `compatible-change` config |
| `configs/test/notifications.yaml` | created | Add disabled safe defaults for tests/smoke. | `compatible-change` config |
| `configs/prod/notifications.yaml` | created | Add disabled safe production defaults. | `compatible-change` config |
| `tests/unit/contexts/notifications/test_dispatcher.py` | created | Cover dispatcher lifecycle. | `none` |
| `tests/unit/contexts/notifications/test_notification_providers.py` | created | Cover fake/log and Telegram adapter result mapping/redaction. | `none` |
| `tests/unit/apps/worker/test_notification_dispatcher_wiring.py` | created | Cover config loading, credential-presence booleans, metrics and composition root backlog drain. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/03-dispatcher-provider-plugin-contract.md` | created | Stage `03` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `03` local implementation and evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `03` report to docs index. | `none` |

## Residual Risks

- No production Telegram canary is claimed in Stage `03`; real Telegram send remains later-stage gated work.
- SQL-backed dispatcher repository and worker supervision are not implemented in this stage.
- `unknown` delivery replay policy is deliberately conservative; later operational tooling must provide explicit replay/reconciliation rather than blind retry.
