# Stage 1: Security baseline для exchange-key mutations

Дата проверки: 2026-05-24.

Статус: accepted locally. GitHub draft PR handoff обязателен перед началом Stage 2.

## Scope

Stage 1 hardened текущий compatibility surface `POST/DELETE /api/exchange-keys`
без добавления `exchange_connections`, external validation, Transit/OpenBao или
order execution.

Изменено:

- exchange credential mutations fail closed без trusted `Origin`/`Referer`;
- cross-origin exchange mutations отклоняются детерминированно;
- add/delete legacy exchange keys требуют recent Keycloak-backed Roehub session;
- `identity_audit_events_type_check` принимает явные `exchange_*` event types;
- create/delete пишут audit events без secret, ciphertext, HMAC или raw exchange
  error body;
- `ROEHUB_ENV=prod` отклоняет static dev-only exchange-key KEK.

## Scenario Matrix

| Mutation scenario | Expected result | Observed result | DB/audit evidence | Blocker |
|---|---|---|---|---|
| No `Origin`/`Referer`, no CSRF signal | `403` with `csrf_required` | `test_exchange_keys_mutations_fail_closed_without_origin_or_referer` passed; payload is deterministic and redacted. | No audit row is written before CSRF gate. | None |
| Cross-origin `Origin: https://evil.example` | `403` with `csrf_required`, reason `csrf_origin_mismatch` | `test_exchange_keys_mutations_reject_cross_origin_requests` passed. | No audit row is written before CSRF gate. | None |
| Same-origin mutation after recent-auth window | `403` with `recent_auth_required` | `test_exchange_keys_mutations_require_recent_auth_after_same_origin_check` passed using session age `11m` and active session TTL. | No audit row is written before recent-auth gate. | None |
| Same-origin mutation with recent session | `201` create / `204` delete; response remains secret-safe | `test_exchange_keys_crud_routes_hide_secrets_and_apply_soft_delete` passed with same-origin headers. | `exchange_key_created` and `exchange_key_deleted` audit events are appended with only `surface`, `key_id`, `exchange_name`, `market_type`, `permissions` where applicable. | None |
| Audit schema | CHECK constraint accepts account events plus required exchange events only | `test_identity_exchange_audit_events_migration_extends_check_constraint` and bootstrap migration tests passed. | `migrations/postgres/0007_identity_exchange_audit_events_v1.sql` replaces `identity_audit_events_type_check`. | None |
| Product/live-ready dev-only KEK | Product config fails closed | `test_identity_wiring_prod_rejects_dev_only_exchange_keys_kek` passed. | Runtime settings reject the documented static dev-only KEK when `ROEHUB_ENV=prod`. | None |

## Acceptance Calls

Exact curl shape for runtime smoke after a local authenticated session is
available:

```bash
# no Origin/Referer -> csrf_required
curl -i -X POST "$ROEHUB_BASE_URL/api/exchange-keys" \
  -H "Cookie: $ROEHUB_SESSION_COOKIE" \
  -H "Content-Type: application/json" \
  --data '{"exchange_name":"binance","market_type":"spot","label":"blocked","permissions":"read","api_key":"TEST","api_secret":"<redacted-test-secret>"}'

# cross-origin -> csrf_required / csrf_origin_mismatch
curl -i -X POST "$ROEHUB_BASE_URL/api/exchange-keys" \
  -H "Origin: https://evil.example" \
  -H "Cookie: $ROEHUB_SESSION_COOKIE" \
  -H "Content-Type: application/json" \
  --data '{"exchange_name":"binance","market_type":"spot","label":"blocked","permissions":"read","api_key":"TEST","api_secret":"<redacted-test-secret>"}'

# same-origin, stale session -> recent_auth_required
curl -i -X POST "$ROEHUB_BASE_URL/api/exchange-keys" \
  -H "Origin: $ROEHUB_BASE_URL" \
  -H "Cookie: $ROEHUB_SESSION_COOKIE" \
  -H "X-CSRF-Token: $ROEHUB_CSRF_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"exchange_name":"binance","market_type":"spot","label":"needs-recent-auth","permissions":"read","api_key":"TEST","api_secret":"<redacted-test-secret>"}'

# same-origin, recent session -> mutation allowed
curl -i -X POST "$ROEHUB_BASE_URL/api/exchange-keys" \
  -H "Origin: $ROEHUB_BASE_URL" \
  -H "Cookie: $ROEHUB_RECENT_AUTH_SESSION_COOKIE" \
  -H "X-CSRF-Token: $ROEHUB_CSRF_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"exchange_name":"binance","market_type":"spot","label":"allowed","permissions":"read","api_key":"TEST","api_secret":"<redacted-test-secret>"}'
```

Observed local equivalent: focused FastAPI `TestClient` tests above exercise the
same route functions, cookie session dependency, same-origin headers,
recent-auth age and audit writer.

## SQL Evidence

```sql
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conname = 'identity_audit_events_type_check';

SELECT event_type, owner_user_id, created_at, metadata_json
FROM identity_audit_events
WHERE event_type LIKE 'exchange_%'
ORDER BY created_at DESC
LIMIT 10;
```

Expected CHECK event set:

- `exchange_key_created`
- `exchange_key_deleted`
- `exchange_connection_created`
- `exchange_connection_validated`
- `exchange_connection_validation_failed`
- `exchange_credential_rotated`
- `exchange_connection_disabled`
- `exchange_connection_deleted`

## Verification Commands

| Command | Outcome |
|---|---|
| `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/api/test_identity_wiring_module.py` | Passed: `26 passed in 1.36s`. |
| `uv run pytest -q tests/unit/apps/migrations` | Passed: `13 passed in 0.29s`. |
| `uv run ruff check apps/api src/trading/contexts/identity tests/unit/apps/api tests/unit/apps/migrations` | Passed. |
| `uv run pyright apps/api src/trading/contexts/identity tests/unit/apps/api` | Passed: `0 errors, 0 warnings, 0 informations`; pyright reported an available version update only. |
| `python -m tools.docs.generate_docs_index --check` | Initial check failed because `docs/architecture/README.md` needed the new Stage 1 report entry; `python -m tools.docs.generate_docs_index` updated it; rerun passed. |
| `rg -n "TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE" logs output .playwright-cli \|\| true` | No matches in checked paths; local `logs` directory is absent. |
| `gh --version && gh auth status` | Passed: `gh version 2.85.0`; authenticated to `github.com` as `Dejetins`. |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `compatible-change` | Existing `POST/DELETE /api/exchange-keys` remain, but unauthenticated-safe mutation preconditions now reject missing/cross-origin CSRF context and stale sessions. |
| DTO schema | `none` | Request and response DTO fields are unchanged. |
| Persisted schema | `compatible-change` | Audit event enum CHECK is extended additively for exchange events; no existing rows need data rewrite. |
| Config schema | `compatible-change` | Existing `ROEHUB_ENV=prod` / `IDENTITY_EXCHANGE_KEYS_KEK_B64` config now rejects the documented dev-only KEK. |
| Cache/request identity | `none` | No cache key, request hash or exchange-key duplicate identity changed. |
| Browser-visible behavior | `compatible-change` | Browser mutations must now carry same-origin context for exchange credentials. `/settings` UI already sends same-origin browser requests. |

## Stage 2 Handoff Facts

- Stage 2 can rely on deterministic error literals `csrf_required` and
  `recent_auth_required` for credential mutations.
- Current recent-auth source is local Roehub session `created_at` from the
  Keycloak-backed login callback. The accepted window is 10 minutes.
- The shared same-origin validator supports fail-open legacy account mutations
  and fail-closed exchange credential mutations.
- The audit event writer is wired to the same identity persistence family as
  exchange-key storage in production.
- Stage 2 must not add external exchange calls or order execution through this
  legacy route surface.
