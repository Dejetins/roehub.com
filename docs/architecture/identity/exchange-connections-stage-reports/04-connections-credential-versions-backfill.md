# Stage 4: Connections Credential Versions Backfill

Дата проверки: 2026-05-24.

Статус: accepted for implementation validation; direct-main delivery pending.
Stage 3A, Stage 3B and Stage 3C are accepted, so Stage 4 added the additive
connection schema, compatibility backfill path, account API facade and
`exchange-control` internal command handlers. The command handlers use
Postgres persistence when `IDENTITY_PG_DSN` is configured and keep the
in-memory repository as dev/test fallback only. Production delivery evidence is
recorded in the shared ledger after push/deploy.

## Scope

Stage 4 adds `exchange_connections`, `exchange_credential_versions`, stable
`connection_id`, replaceable `credential_version_id`, local-dev create/list/
rotate/disable evidence and legacy read compatibility. It does not remove
`identity_exchange_keys`, does not support `linear`/`inverse`, does not call
Binance/Bybit, does not add UI completion and does not implement order
execution.

## Prerequisite Evidence

| Migration phase | Source of truth | Command / SQL | Expected result | Observed result | Rollback |
|---|---|---|---|---|---|
| Stage 3A | `03a-openbao-vault-runtime-provisioning.md` | Report/ledger review | OpenBao/Vault Transit runtime accepted. | Accepted; no blocker. | N/A |
| Stage 3B | `03b-transit-application-integration.md` | Report/ledger review | Transit-backed `ExchangeSecretCipher` accepted. | Accepted; no blocker. | N/A |
| Stage 3C | `03c-exchange-control-internal-command-api.md` | Report/ledger review | `apps/api -> exchange-control` internal boundary accepted. | Accepted; Stage 4 may start. | N/A |

## Schema And Backfill Evidence

| Migration phase | Source of truth | Command / SQL | Expected result | Observed result | Rollback |
|---|---|---|---|---|---|
| Phase A: schema | `migrations/postgres/0008_exchange_connections_v1.sql` | `CREATE TABLE IF NOT EXISTS exchange_connections` | Additive connection table exists. | Implemented with `connection_id`, owner, exchange, `market_type`, environment, status and `active_credential_version_id`. | Drop new tables only before Phase C writes; `identity_exchange_keys` remains untouched. |
| Phase A: credential versions | `migrations/postgres/0008_exchange_connections_v1.sql` | `CREATE TABLE IF NOT EXISTS exchange_credential_versions` | Replaceable credential versions exist. | Implemented with `credential_version_id`, ciphertext columns, masked suffix, fingerprint HMAC, cipher metadata and status. | Drop new tables only before Phase C writes. |
| Phase B: backfill | `INSERT INTO exchange_connections ... FROM identity_exchange_keys` | Legacy rows can be copied deterministically. | Backfill uses legacy `key_id` as stable `connection_id`; active version is set to legacy `key_id`; deleted rows become disabled. | Reverse-backfill required after Phase C writes if rolling back to legacy-only code. |
| Phase B: compatibility projection | `PostgresIdentityExchangeKeysRepository.list_active_for_user` | New tables are read first, then legacy fallback. | Unit evidence proves projection from `exchange_connections` returns legacy-safe `ExchangeKeyView` with masked key. | Keep dual-read version during rollback; do not deploy legacy-only code after Phase C without reverse-backfill. |
| Phase C: command writes | `PostgresExchangeConnectionRepository` via `IDENTITY_PG_DSN` | `exchange-control` create/rotate/disable writes target `exchange_connections` and `exchange_credential_versions`. | Implemented; prod `ExchangeControlRuntimeConfig` now fails closed without `IDENTITY_PG_DSN`; dev/test may use deterministic in-memory fallback. | Roll back to dual-read build or reverse-backfill new-only writes before legacy-only rollback. |
| Market type v1 | SQL and DTO literals | `CHECK (market_type IN ('spot', 'futures'))`; Pydantic literals | `linear` and `inverse` rejected in v1. | SQL and API tests reject unsupported market types. | Future support requires separate migration and compatibility report. |

## API Evidence

| Migration phase | Source of truth | Command / SQL | Expected result | Observed result | Rollback |
|---|---|---|---|---|---|
| Create | Local-dev curl `POST /api/ui/account/exchange-connections` | `--data @fixtures/nonreal-binance-connection.json` | Response returns `connection_id` and no secrets. | Passed: returned `connection_id=00000000-0000-0000-0000-000000000001`, masked `api_key=****1234`, no secret/ciphertext/HMAC fields. | Disable public route flag or roll back facade/client. |
| List | Local-dev curl `GET /api/ui/account/exchange-connections` | Cookie-authenticated request | List returns masked connection rows. | Passed: returned one masked item with stable `connection_id`. | Disable public route flag or roll back facade/client. |
| Rotate | Local-dev curl `POST /api/ui/account/exchange-connections/{connection_id}/rotate` | Non-real rotated credential payload | `connection_id` stays stable and `credential_version_id` changes. | Passed: same `connection_id`; `credential_version_id` changed from `...03e9` to `...03ea`; masked key changed to `****9876`. | Keep dual-read code; reverse-backfill if legacy-only rollback is required. |
| Disable | Local-dev curl `POST /api/ui/account/exchange-connections/{connection_id}/disable` | Same-origin authenticated call | Connection status becomes disabled. | Passed: `status=disabled`, `status_reason=user_disabled`. | Re-enable by future explicit operation; no delete performed. |
| Legacy compatibility | Local-dev curl `GET /api/exchange-keys` | Cookie-authenticated request | Legacy endpoint still responds without secrets. | Passed on seeded local-dev runtime: `[]`; unit projection covers new-table compatibility read. | Legacy table remains present. |
| Stage 3C preflight | `curl http://127.0.0.1:9205/internal/v1/capabilities ... stage-4-preflight` | Internal boundary reachable. | Passed locally: returned `exchange_connections.create/list/rotate/disable` capabilities and Stage 5 validation pending. | Stop Stage 4 if internal boundary becomes unavailable. |

## Secret-Safety Evidence

| Surface | Command / SQL | Expected result | Observed result | Rollback |
|---|---|---|---|---|
| API responses | Focused route tests and local curl | No `api_secret`, passphrase, ciphertext, fingerprint or HMAC in responses. | Passed; response DTOs expose only masked `api_key`. | Remove public facade if leakage appears. |
| `apps/api` grep | `rg -n "ExchangeSecretCipher\|decrypt\|openbao\|vault\|binance\|bybit\|pybit\|api_secret\|passphrase" apps/api \|\| true` | No direct secret/decrypt/native exchange adapter imports. | No `ExchangeSecretCipher`, `decrypt`, `openbao`, `vault`, `pybit` imports. Stage 4 intentionally has `api_secret`/`passphrase` request DTO and client-forwarding fields; existing `binance` literals remain UI/backtest labels. | Keep all encryption/decrypt work inside `exchange-control`. |
| Artifact grep | `rg -n "TEST_SECRET\|TEST_API_SECRET\|TEST_PASSPHRASE" logs output .playwright-cli \|\| true` | No secret markers in generated artifacts. | Passed; `logs` directory absent, no artifact matches. | Delete generated artifacts before commit if markers appear. |

## Quality Gates

| Gate | Expected result | Observed result | Rollback |
|---|---|---|---|
| `uv run pytest -q tests/unit/apps/migrations tests/unit/contexts/exchange_control tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py` | Focused migration, command API and API facade tests pass. | Passed: `49 passed`. | Do not push if failing. |
| `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | Lint passes. | Passed. | Do not push if failing. |
| `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api` | Type check passes. | Passed: `0 errors`. | Do not push if failing. |
| `python -m tools.docs.generate_docs_index --check` | Docs index current after report creation. | Passed. | Regenerate docs index before push. |

## Contract Impact Classification

| Dimension | Classification | Reason |
|---|---|---|
| Public API contract | `compatible-change` | New account endpoints are additive; legacy `/api/exchange-keys` remains. |
| Internal API/client contract | `compatible-change` | Stage 3C namespace gains create/list/rotate/disable handlers and client methods. |
| DTO schema | `compatible-change` | New secret-bearing request DTOs and secret-safe response DTOs are additive. |
| Persisted schema | `compatible-change` | `0008` adds new tables and backfill without deleting `identity_exchange_keys`. |
| Config schema | `compatible-change` | Stage 4 reuses Stage 3C internal base URL/token config and requires `IDENTITY_PG_DSN` for product `exchange-control` persistence. |
| Request hash/cache identity | `none` | No cache or request-hash semantics changed. |
| Persistence identity | `compatible-change` | Stable `connection_id` is introduced; rotation changes only `credential_version_id`. |

## Rollback Notes

| Phase | Rollback path | Data risk | Evidence required before rollback |
|---|---|---|---|
| Before Phase C writes | Revert app code and drop `exchange_connections` / `exchange_credential_versions`. | Low; legacy table remains source of truth. | SQL count of new writes is zero. |
| After Phase C writes | Roll back only to a version with dual-read support, or run reverse-backfill into `identity_exchange_keys` first. | Medium; legacy-only code cannot see new-only writes. | Row counts and sampled `connection_id` / `credential_version_id` mapping. |
| Emergency disable | Keep legacy table, disable public account routes and stop using new write endpoints. | Low for reads; writes paused. | Route smoke confirms create/rotate/disable unavailable while legacy GET still works. |

## Stage 5 Handoff Facts

- Stage 5 must add exchange validation behind `exchange-control`; `apps/api`
  remains a facade/client and must not import native exchange SDKs.
- `market_type` v1 remains `spot|futures`; `linear`/`inverse` are unsupported
  until a separate compatibility migration exists.
- `connection_id` is stable across rotation; `credential_version_id` changes on
  rotation.
- `identity_exchange_keys` is still present and must not be deleted in Stage 5.
- Stage 5 may use `exchange_connections.status`, `status_reason`,
  `permission_summary_json` and `last_validated_at` for validation results.
- Product `exchange-control` now requires `IDENTITY_PG_DSN`; local-dev tests may
  still use in-memory storage, but production acceptance must use Postgres-backed
  command writes.
