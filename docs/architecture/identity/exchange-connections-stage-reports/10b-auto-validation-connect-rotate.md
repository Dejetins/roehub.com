# Stage 10B: Auto-Validation On Connect And Rotate

Дата проверки: 2026-05-26.

Статус: implementation validated locally; runtime acceptance and direct-main
delivery pending.

Scope: backend/domain/API command semantics for auto-validation on exchange
connection create and credential rotate. Stage 10B does not change `/settings`
browser CJM, does not remove the legacy permissions selector, does not
reclassify existing rows, does not place or simulate orders, and does not add
exchange-execution behavior.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 10A must be accepted before 10B starts. | Iteration ledger marks 10A accepted with direct-main CI/deploy and runtime evidence complete; 10A report is accepted. | Accepted. | None. |
| Create/connect | New connection create must validate before a row can be active/trading-ready. | `create_connection_with_validation` validates plaintext inside exchange-control before persistence and creates either `active/ready_for_trading` or `disabled/status_reason=auto_validation_failed`. | Accepted locally. | Runtime readonly proof pending. |
| Rotate | New credential version must validate before replacing the active version. | `rotate_connection_with_validation` validates the new plaintext first; non-ready validation raises deterministic code and does not call `replace_active_credential`. | Accepted locally. | Runtime rotate proof is not required by 10B acceptance commands but remains covered by unit tests. |
| Non-ready outcomes | Readonly, unsafe, invalid, missing IP restriction and validation unavailable must not be active/limit-consuming. | Domain truth-table tests cover all outcomes; public facade tests prove readonly create is `disabled`, absent from Active, and not counted in limits. | Accepted locally. | Existing pre-10B active rows are not repaired until 10D. |
| Secret boundary | Secret-bearing validation stays in exchange-control, not apps/api. | apps/api still forwards plaintext only to local internal exchange-control command API after CSRF/recent-auth; validation and encryption remain in exchange-control. | Accepted locally. | Runtime secret-safe curl evidence pending. |
| Metrics/audit | Auto-validation outcomes need bounded, secret-free observability. | Added `exchange_connection_auto_validation_total{exchange,result,reason}` and existing `exchange_connection_validated` account audit entries with `validation_mode=auto_validation` plus bounded operation/result/reason metadata. | Accepted locally. | Runtime metric scrape pending. |

## Create/Rotate Decision Table

| Operation | Validation outcome | Persisted / returned state | Capability/readiness | Deterministic reason | Limit impact |
|---|---|---|---|---|---|
| create | `valid_trade_enabled`, exchange `trade`, IP policy OK | `active`, no `status_reason` | `effective_capability=trading`, `connection_readiness=ready_for_trading` | `trading_policy_ok` | Counts as active/API key. |
| create | readonly or requested-trade mismatch | `disabled`, `status_reason=auto_validation_failed` | `effective_capability=none`, `connection_readiness=rejected` | `read_only_not_supported` | Not counted. |
| create | withdrawal/transfer permission | `disabled`, `status_reason=auto_validation_failed` | `effective_capability=none`, `connection_readiness=rejected` | `unsafe_permissions` | Not counted. |
| create | invalid credentials | `disabled`, `status_reason=auto_validation_failed` | `effective_capability=none`, `connection_readiness=rejected` | `invalid_credentials` | Not counted. |
| create | missing mainnet IP restriction | `disabled`, `status_reason=auto_validation_failed` | `effective_capability=none`, `connection_readiness=needs_action` | `ip_restriction_required` | Not counted. |
| create | validation skipped/unavailable | `disabled`, `status_reason=auto_validation_failed` | `effective_capability=none`, `connection_readiness=needs_action` | `validation_unavailable` | Not counted. |
| rotate | `valid_trade_enabled`, exchange `trade`, IP policy OK | Active credential version is replaced and validation metadata is recorded. | `effective_capability=trading`, `connection_readiness=ready_for_trading` | `trading_policy_ok` | Remains counted. |
| rotate | any non-ready outcome | Request fails before credential replacement. | Existing active row remains unchanged. | Readiness reason, for example `read_only_not_supported`. | Existing state preserved. |

## Implementation Evidence

| Surface | Change | Evidence |
|---|---|---|
| Domain service | Added auto-validation create/rotate methods; create validates before persistence; rotate validates before active credential replacement. | `src/trading/contexts/exchange_control/application/connections.py`; `tests/unit/contexts/exchange_control/test_exchange_connection_readiness.py`. |
| Readiness semantics | Stage 10A readiness mapping is reused; auto-validation maps skipped validation to `validation_unavailable` and never active-ready. | Domain truth-table tests for ready, readonly, unsafe, invalid, missing IP and unavailable. |
| Persistence | Failed create attempts are durable `disabled` rows with `status_reason=auto_validation_failed`; no new columns or migration. | `permission_summary_json` stores auto-validation readiness/capability metadata. |
| Internal command API | Internal create/rotate routes call auto-validation methods and emit validation/readiness/auto-validation metrics. | `src/trading/contexts/exchange_control/adapters/inbound/http/app.py`; runtime unit tests. |
| Public account facade | Public create/rotate still enforce same-origin and recent-auth before mutation; public DTO shape remains Stage 10A compatible. | `apps/api/routes/ui_account.py`; `apps/api/exchange_control_client.py`; API route tests. |
| Audit | Account audit records auto-validation outcomes on the existing `exchange_connection_validated` event type with `validation_mode=auto_validation`, without secrets. | `src/trading/contexts/identity/application/use_cases/account_settings.py`; API route tests. |

## Runtime Evidence

| Surface | Command | Sanitized result | Verdict |
|---|---|---|---|
| Required env | Presence check for `ROEHUB_E2E_BYBIT_MAINNET_READONLY_API_KEY`, `ROEHUB_E2E_BYBIT_MAINNET_READONLY_API_SECRET`, public session/CSRF and PG DSN env. | Local execution shell did not contain the required runtime env at implementation time. | Pending. |
| Readonly create | `curl -i -X POST "$ROEHUB_BASE_URL/api/ui/account/exchange-connections" ... label=stage10b_readonly_reject ...` | Pending direct runtime execution with env-backed readonly Bybit credentials. | Pending. |
| Active exclusion | `curl -fsS "$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active" ... | jq -e 'all(.items[]; .label != "stage10b_readonly_reject")'` | Pending. | Pending. |
| Invalid create | `curl -i -X POST "$ROEHUB_BASE_URL/api/ui/account/exchange-connections" ... label=stage10b_invalid_reject ... INVALID ...` | Pending. | Pending. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_auto_validation_total|exchange_connection_trading_readiness_total'` | Pending. | Pending. |
| Postgres read model | `psql "$ROEHUB_PG_DSN" -c "SELECT label, status, status_reason, permission_summary_json FROM exchange_connections WHERE label IN ('stage10b_readonly_reject','stage10b_invalid_reject') ORDER BY created_at DESC;"` | Pending. | Pending. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `breaking-change` by behavior, structurally compatible | Same endpoint/DTO shape, but create/rotate no longer treat non-trading credentials as successful active connections. This is intentional Stage 10 product semantics. |
| Public errors | `compatible-change` | Adds deterministic Roehub error codes for rotate failure: `read_only_not_supported`, `unsafe_permissions`, `ip_restriction_required`, `invalid_credentials`, `validation_unavailable`, `invalid_permissions`, `unsupported_account_mode`. |
| Internal API | `breaking-change` by behavior, structurally compatible | Internal create/rotate now auto-validate and can return disabled create rows or reject rotate before replacement. Response shape remains additive/compatible. |
| Domain / port | `compatible-change` | Adds service methods and an optional disable `status_reason`; existing manual create/rotate methods remain available for tests/backward compatibility but internal commands no longer use them. |
| Persistence schema | `none` | No migration, table or column change. |
| Persistence semantics | `compatible-change` | Failed create attempts use accepted lifecycle state `disabled` and existing `permission_summary_json`; failed rotate leaves current active credential untouched. |
| Config / env | `none` | No new runtime env key; Stage 10B acceptance requires existing env-backed readonly credentials. |
| Metrics / ops | `compatible-change` | Adds bounded `exchange_connection_auto_validation_total{exchange,result,reason}`. |
| Audit | `compatible-change` | Reuses existing `exchange_connection_validated` audit event type with additive bounded metadata `validation_mode=auto_validation`; no secrets or raw exchange bodies. |
| Trading execution | `none` | No order placement, order simulation, exchange execution or order ledger code added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Focused domain pytest | Passed: `42 passed`. | `uv run pytest -q tests/unit/contexts/exchange_control`. |
| Focused public API pytest | Passed: `24 passed`. | `uv run pytest -q tests/unit/apps/api/test_ui_account_routes.py`. |
| Focused ruff | Passed. | `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`. |
| Focused pyright | Passed: `0 errors`. | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`. |
| Required gates | Pending final full run after report/ledger/docs-index update. | Pending. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | Pending. | Pending local gates, runtime acceptance and direct-main push. |
| CI / deploy | Pending. | Pending. |
| Post-deploy runtime | Pending. | Pending. |

## Residual Risk And Stage 10C Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| UI still shows legacy permission selector and old wording. | 10C | Remove selector and align `/settings` CJM with trading-only auto-validation. |
| Existing active non-trading rows are not reclassified. | 10D | Controlled dry-run then execution must move non-ready active rows out of Active through supported lifecycle paths. |
| Runtime acceptance depends on env-backed readonly Bybit credentials and authenticated session/CSRF. | 10B | If unavailable, mark 10B blocked/partial; unit tests alone are not acceptance. |
| Manual validation remains available only as re-check for active rows. | 10C/10E | UI should expose this as Re-check, not as a required post-create success step. |
