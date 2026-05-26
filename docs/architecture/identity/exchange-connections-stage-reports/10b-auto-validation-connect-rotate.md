# Stage 10B: Auto-Validation On Connect And Rotate

Дата проверки: 2026-05-26.

Статус: accepted; implementation commits `c1c8e234` and `b3f0c1cc`
direct-main delivered; CI/deploy and Mac Studio runtime evidence complete.

Scope: backend/domain/API command semantics for auto-validation on exchange
connection create and credential rotate. Stage 10B does not change `/settings`
browser CJM, does not remove the legacy permissions selector, does not
reclassify existing rows, does not place or simulate orders, and does not add
exchange-execution behavior.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 10A must be accepted before 10B starts. | Iteration ledger marks 10A accepted with direct-main CI/deploy and runtime evidence complete; 10A report is accepted. | Accepted. | None. |
| Create/connect | New connection create must validate before a row can be active/trading-ready. | `create_connection_with_validation` validates plaintext inside exchange-control before persistence and creates either `active/ready_for_trading` or `disabled/status_reason=auto_validation_failed`; runtime readonly create returned disabled. | Accepted. | None for new writes. |
| Rotate | New credential version must validate before replacing the active version. | `rotate_connection_with_validation` validates the new plaintext first; non-ready validation raises deterministic code and does not call `replace_active_credential`; unit test proves credential version preservation. | Accepted. | Runtime rotate proof is not required by 10B acceptance commands. |
| Non-ready outcomes | Readonly, unsafe, invalid, missing IP restriction and validation unavailable must not be active/limit-consuming. | Domain truth-table tests cover all outcomes; public facade tests prove readonly create is `disabled`, absent from Active, and not counted in limits; runtime readonly and invalid creates were absent from Active. | Accepted. | Existing pre-10B active rows are not repaired until 10D. |
| Secret boundary | Secret-bearing validation stays in exchange-control, not apps/api. | apps/api still forwards plaintext only to local internal exchange-control command API after CSRF/recent-auth; validation and encryption remain in exchange-control; runtime evidence output selected only bounded fields. | Accepted. | None. |
| Metrics/audit | Auto-validation outcomes need bounded, secret-free observability. | Runtime metrics exposed `exchange_connection_auto_validation_total{exchange="bybit",result="rejected",reason="read_only_not_supported"}` and `reason="invalid_credentials"`; audit rows use existing `exchange_connection_validated` with `validation_mode=auto_validation`. | Accepted. | None. |

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
| Required env | Host-local presence check for readonly Bybit credential env. | Mac Studio did not expose the exact `ROEHUB_E2E_BYBIT_MAINNET_READONLY_*` aliases, but did expose existing readonly Bybit secret env `ROEHUB_TEST_BYBIT_READONLY_API_KEY/SECRET`; the runtime command shell exported those values into the required E2E variable names without printing values. | Pass; env-backed readonly credentials were used. |
| Readonly create | Equivalent Mac Studio app-local command: `curl -X POST "$ROEHUB_BASE_URL/ui/account/exchange-connections" ... label=stage10b_readonly_reject ...` with `ROEHUB_BASE_URL=http://127.0.0.1:8000`. Public `https://roehub.com/api/...` hairpin from Mac Studio timed out, so the same deployed FastAPI route was exercised directly. | `readonly_http=201`; sanitized body: `status=disabled`, `status_reason=auto_validation_failed`, `validation_status=valid_readonly`, `connection_readiness=rejected`, `connection_readiness_reason=read_only_not_supported`, `effective_capability=none`, `permissions_deprecated=true`. | Pass. |
| Active exclusion | `curl -fsS "$ROEHUB_BASE_URL/ui/account/exchange-connections?status=active" ... | jq -e 'all(.items[]; .label != "stage10b_readonly_reject")'` | `active_exclusion=true`. | Pass. |
| Invalid create | `curl -X POST "$ROEHUB_BASE_URL/ui/account/exchange-connections" ... label=stage10b_invalid_reject ... INVALID ...` | `invalid_http=201`; sanitized body: `status=disabled`, `status_reason=auto_validation_failed`, `validation_status=invalid_credentials`, `connection_readiness=rejected`, `connection_readiness_reason=invalid_credentials`, `effective_capability=none`, `permissions_deprecated=true`. | Pass. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics | grep -E 'exchange_connection_auto_validation_total|exchange_connection_trading_readiness_total'` | Metrics included `exchange_connection_trading_readiness_total{exchange="bybit",reason="read_only_not_supported",result="rejected"} 1.0`, `reason="invalid_credentials" 1.0`, and matching `exchange_connection_auto_validation_total` series. | Pass. |
| Postgres read model | `psql "$PG_DSN" -c "SELECT label, status, status_reason, permission_summary_json ..."` without secret-bearing columns. | `stage10b_readonly_reject` and `stage10b_invalid_reject` rows are `disabled`, `status_reason=auto_validation_failed`; JSON stores `requested_capability=trading`, `effective_capability=none`, `connection_readiness=rejected`, reasons `read_only_not_supported` and `invalid_credentials`, and `permissions_deprecated=true`. | Pass. |
| Audit | `psql "$PG_DSN" -c "SELECT event_type, metadata_json ... WHERE metadata_json ->> 'validation_mode' = 'auto_validation'"`. | Latest audit rows use `event_type=exchange_connection_validated`, `validation_mode=auto_validation`, `operation=create`, `result=rejected`, reasons `read_only_not_supported` and `invalid_credentials`; no secret-bearing metadata. | Pass. |
| Cleanup | Smoke session revocation query. | `smoke_session_revoked=true`; failed connection attempts remain durable disabled records, not physically deleted. | Pass. |

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
| Required pytest | Passed: `81 passed`. | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/migrations`. |
| Required ruff | Passed. | `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/migrations`. |
| Required pyright | Passed: `0 errors`. | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`; additionally `uv run pyright` passed after CI exposed the identity audit type surface. |
| Docs index | Passed. | `python -m tools.docs.generate_docs_index --check`; docs index regenerated with `python -m tools.docs.generate_docs_index`. |
| No-order grep | Passed. | `rg -n "place_order|create_order|order placement|exchange-execution|exchange_execution|submit_order|cancel_order" ...`; matches were only Stage 10B non-goal text. |
| Runtime acceptance | Passed. | Mac Studio readonly, invalid, Active-list, metrics, DB and audit evidence above. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | `c1c8e234 Add exchange connection auto validation`; `b3f0c1cc Fix auto validation audit contract`; pushed to `origin/main`. | Pass; no stage branch or PR. |
| CI / deploy | CI `26475706403` success; Deploy Backend `26475748728` success; Publish App Image `26475748616` success; Deploy Web `26475748610` success. Earlier CI `26475554720` failed static type check on a new audit enum and was fixed by `b3f0c1cc`. | Pass. |
| Post-deploy runtime | Mac Studio backend deploy smoke passed; `/health/ready` returned `ready`; deployed bundle contains `create_connection_with_validation`; runtime acceptance table above passed. | Pass. |

## Residual Risk And Stage 10C Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| UI still shows legacy permission selector and old wording. | 10C | Remove selector and align `/settings` CJM with trading-only auto-validation. |
| Existing active non-trading rows are not reclassified. | 10D | Controlled dry-run then execution must move non-ready active rows out of Active through supported lifecycle paths. |
| Runtime acceptance depends on env-backed readonly Bybit credentials and authenticated session/CSRF. | 10B | If unavailable, mark 10B blocked/partial; unit tests alone are not acceptance. |
| Manual validation remains available only as re-check for active rows. | 10C/10E | UI should expose this as Re-check, not as a required post-create success step. |
