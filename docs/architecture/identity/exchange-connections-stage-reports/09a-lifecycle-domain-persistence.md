# Stage 09A: Lifecycle Domain Persistence

Дата проверки: 2026-05-26.

Статус: accepted locally; direct-main delivery evidence is recorded after push,
CI/deploy and target-runtime checks.

Scope ограничен persistence/domain foundation для lifecycle state `archived`.
Public `/settings` UI, default account list semantics, public archive endpoint,
permission semantics и cleanup старых `stage08_*` rows остаются в следующих
Stage 09B-09D.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage 08 prerequisite | Stage 08 must be accepted before Stage 09A starts. | Iteration ledger marks Stage 08 as `accepted; direct-main delivered; production Playwright evidence complete`; Stage 8 report status is accepted. | Accepted. | None. |
| Persistence lifecycle | `exchange_connections` can represent `active`, `disabled`, `archived` with lifecycle timestamp constraints. | `0008_exchange_connections_v1.sql` additively adds `archived_at`, allows status `archived`, replaces old disabled-only check with `exchange_connections_lifecycle_timestamps_chk`, and preserves existing rows/backfill. | Accepted. | Existing production rows must satisfy old active/disabled invariants before the new constraint is applied. |
| Domain archive command | Archive only owned disabled connections; active archive is deterministic rejection; archived archive is idempotent; rotate/validate archived reject. | Exchange-control runtime tests cover create -> rotate -> disable -> archive -> archive again, active archive rejection, and rotate/validate archived returning not found. | Accepted. | Public facade does not expose archive until Stage 09B. |
| Audit and metrics | `exchange_connection_archived` event path and archive/cleanup metrics exist without secret-bearing labels. | `0007` audit enum and identity port type include `exchange_connection_archived`; account use-case has a redacted archive audit writer; exchange-control exports `exchange_connection_archive_total{exchange,result,reason}` and `exchange_connection_cleanup_total{source,result}`. | Accepted. | Stage 09B must wire public/API audit emission when archive becomes user-facing. |
| No physical delete | Archive is soft lifecycle state, not physical deletion. | Migration tests assert no `DELETE FROM exchange_connections` / `DELETE FROM exchange_credential_versions`; repository implements `UPDATE ... SET status='archived'`. physical hard delete запрещен. | Accepted. | Operator cleanup in Stage 09D must continue using repository/use-case or internal API, not ad hoc SQL deletes. |

## Lifecycle Transition Table

| Command | From | To | Stage 09A behavior |
|---|---|---|---|
| `create` | N/A | `active` | Unchanged. |
| `disable` | `active` | `disabled` | Unchanged user-disabled transition; `disabled_at` is required. |
| `archive` | `active` | N/A | Rejected with `exchange_connection_not_disabled`. |
| `archive` | `disabled` | `archived` | Soft update; preserves `connection_id`, active credential version reference and disabled credential version. |
| `archive` | `archived` | `archived` | Idempotent success. |
| `rotate` | `archived` | N/A | Rejected as not found. |
| `validate` | `archived` | N/A | Rejected as not found. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `none` | No public archive route, UI action, status filter, or default-list behavior is introduced in Stage 09A. |
| Internal API contract | `compatible-change` | Adds local-only, service-auth protected `exchange_connections.archive` capability and `POST /internal/v1/exchange-connections/{connection_id}/archive`; existing internal commands are unchanged. |
| Port contract | `compatible-change` | `ExchangeConnectionRepository` gains `archive`; connection view/record gain optional `archived_at`. |
| Persisted schema | `compatible-change` | `0008` additively adds nullable `archived_at`, extends status enum check, and replaces lifecycle timestamp check. |
| Audit schema | `compatible-change` | `0007` additively accepts `exchange_connection_archived`; metadata contract is redacted state identifiers only. |
| Config / env | `none` | No new env variables or runtime config keys. |
| Metrics / ops | `compatible-change` | Adds bounded archive/cleanup counters and archived status metric seed; labels exclude user, connection, credential and secret values. |
| Browser-visible behavior | `none` | `/settings` and default account list behavior intentionally remain unchanged until Stage 09B. |
| Trading execution | `none` | No order placement, signal-to-execution, exchange-execution or order ledger path is added. |

## Implementation Evidence

| Surface | Change | Evidence |
|---|---|---|
| Migration `0008` | Added `archived_at`, `active|disabled|archived`, lifecycle timestamp check, idempotent existing-table `ALTER`, and backfill `archived_at=NULL`. | `tests/unit/apps/migrations/test_exchange_connections_sql.py`. |
| Migration `0007` | Added `exchange_connection_archived` to audit event check. | `tests/unit/apps/migrations/test_identity_exchange_audit_events_sql.py`. |
| Domain / repository | Added archive command on service/repository, active-only rotate/validate/disable guard, idempotent archived handling, and Postgres soft update. | `tests/unit/contexts/exchange_control/test_exchange_control_runtime.py`. |
| Internal API | Added local-only archive route and capability; response includes `archived_at`; app client has a matching internal method for later stages. | Runtime tests and pyright. |
| Metrics | Added `exchange_connection_archive_total` and `exchange_connection_cleanup_total` with bounded labels. | Runtime metrics test. |
| Audit path | Added typed archive audit event and redacted account use-case writer for future public archive wiring. | Static type support and migration test. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/migrations` | Passed: `35 passed`. | Local run on 2026-05-26. |
| `uv run ruff check src/trading/contexts/exchange_control tests/unit/contexts/exchange_control tests/unit/apps/migrations` | Passed. | Local run on 2026-05-26. |
| `uv run pyright src/trading/contexts/exchange_control tests/unit/contexts/exchange_control` | Passed: `0 errors`. | Local run on 2026-05-26. |
| `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py` | Passed: `21 passed`. | Extra compatibility smoke for apps/api fake client and public account routes. |
| `python -m tools.docs.generate_docs_index --check` | Passed. | Docs index refreshed and checked on 2026-05-26. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current` | `main`. |
| Fast-forward | `git pull --ff-only origin main` | Already up to date before implementation. |
| Commit / push | Pending. | To be updated after direct-main delivery. |
| CI / deploy | Pending. | To be updated after GitHub Actions/deploy observation. |
| Mac Studio runtime | Pending. | To be updated after shipped revision is deployed and smoke checked. |

## Residual Risk And Stage 09B Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Public account list still uses current Stage 08 semantics and does not expose archive. | 09B | Add explicit status filter/history, default active-only list, and public archive action only for disabled rows. |
| Audit archive writer is prepared but not yet called by public archive facade. | 09B | Emit `exchange_connection_archived` with redacted metadata when user-facing archive is added. |
| Cleanup metrics exist but no cleanup command runs in 09A. | 09D | Implement controlled dry-run/execution cleanup through supported command/API, not direct deletes. |
| Permission semantics are unchanged. | 09C | Introduce requested/exchange/effective permissions without reusing lifecycle status for capability decisions. |
