# Stage 0: Текущий baseline `/api/exchange-keys` и `/settings`

Дата проверки: 2026-05-24.

Статус: текущий compatibility baseline зафиксирован. Runtime-поведение не менялось.

## Scope

Stage 0 фиксирует только существующий контракт перед Exchange Control v1:

- текущий API surface `/api/exchange-keys`;
- текущий browser-visible surface `/settings`;
- persisted schema `identity_exchange_keys`;
- отсутствие `api_secret`, `passphrase`, ciphertext, fingerprint и `hmac` в API/UI/test artifacts;
- текущий `market_type` v1: `spot|futures`.

Trading execution, `exchange_connections`, validation, rotation, OpenBao/Vault и risk/order ledger не входят в этот stage.

## Route Matrix

| Surface | Текущий route / action | Request fields | Response / behavior | Evidence | Gap |
| --- | --- | --- | --- | --- | --- |
| API | `POST /api/exchange-keys` через mounted router `/exchange-keys` | `exchange_name: binance|bybit`, `market_type: spot|futures`, `label: string|null`, `permissions: read|trade`, `api_key`, `api_secret`, `passphrase: string|null` | `201` и `ExchangeKeyResponse`: `key_id`, `exchange_name`, `market_type`, `label`, `permissions`, masked `api_key`, `created_at`, `updated_at` | `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`; `tests/unit/apps/api/test_identity_exchange_keys_routes.py` | Нет. |
| API | `GET /api/exchange-keys` через mounted router `/exchange-keys` | Authenticated current user only | `200` array of the same safe response DTO, sorted by `created_at ASC, key_id ASC`; soft-deleted rows excluded | `ListExchangeKeysUseCase.list_for_user`; `test_exchange_keys_crud_routes_hide_secrets_and_apply_soft_delete` | Нет. |
| API | `DELETE /api/exchange-keys/{key_id}` через mounted router `/exchange-keys/{key_id}` | UUID path param `key_id` | `204` empty response on success; soft-delete only | `DeleteExchangeKeyUseCase.delete`; route test asserts row `is_deleted=True` and `deleted_at` set | Нет. |
| Duplicate | Active duplicate create | Same user + normalized `exchange_name`, `market_type`, API key hash | `409` with `{"detail":{"error":"exchange_key_already_exists","message":"Exchange API key already exists."}}` | `ExchangeKeyAlreadyExistsError`; route duplicate test | Нет. |
| Delete error | Missing, foreign, or already-deleted key | UUID path param `key_id` | `404` with `{"detail":{"error":"exchange_key_not_found","message":"Exchange API key was not found."}}` | `ExchangeKeyNotFoundError`; route missing-delete test | Нет. |
| Auth | Unauthenticated create/list/delete | No valid current user | Deterministic `401` from current-user dependency; tested for POST/GET/DELETE | `test_exchange_keys_routes_require_authenticated_user` | Нет. |

## Storage Evidence

| Artifact | Текущий факт | Evidence | Gap |
| --- | --- | --- | --- |
| Table | `identity_exchange_keys` exists with `key_id`, `user_id`, `exchange_name`, `market_type`, `label`, `permissions`, `api_key`, `api_secret_enc`, `passphrase_enc`, timestamps and soft-delete columns | `migrations/postgres/0003_identity_exchange_keys_v1.sql` | Нет. |
| Enum constraints | SQL checks enforce `exchange_name IN ('binance', 'bybit')`, `market_type IN ('spot', 'futures')`, `permissions IN ('read', 'trade')` | `identity_exchange_keys_*_chk` constraints in migration 0003 | Нет. |
| Duplicate identity | Active unique index is `(user_id, exchange_name, market_type, api_key)` where `is_deleted = FALSE`; application repositories also use normalized API key hash for duplicate semantics | `idx_identity_exchange_keys_active_unique_key`; `ExchangeKeysRepository` ports/tests | Нет. |
| Secret storage | Use case encrypts `api_key`, `api_secret`, and optional `passphrase` before repository create; tests assert stored row has no plaintext `api_key` attr and encrypted/hash fields differ from test input | `CreateExchangeKeyUseCase.create`; route CRUD test | Current v1 migration names blobs `api_secret_enc` / `passphrase_enc`, not `ciphertext` or HMAC fingerprint columns. |
| Compatibility docs | Existing storage policy v2 says API must not return `api_secret`, `passphrase`, `api_key_enc`, `api_secret_enc`, `passphrase_enc`, `api_key_hash` | `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md` | No stale current-behavior contradiction found. |

## UI Evidence

| Surface | Текущий факт | Evidence | Gap |
| --- | --- | --- | --- |
| `/settings` endpoint wiring | Settings page root includes `data-exchange-keys-endpoint="/api/exchange-keys"` | `apps/web/templates/pages/settings.html` | Нет. |
| Form fields | UI form collects `label`, `api_key`, `api_secret`, `passphrase`; exchange dropdown is `binance|bybit`; market dropdown is only `spot|futures` | `apps/web/templates/fragments/account/exchange_keys.html` | Нет. |
| Defaults | JS state defaults to `exchangeName: "binance"` and `marketType: "futures"` | `apps/web/dist/js/pages/settings.js` | Нет. |
| Submit payload | JS sends `exchange_name`, `market_type`, `label`, `permissions: "trade"`, `api_key`, `api_secret`, `passphrase` to `/api/exchange-keys` | `apps/web/dist/js/pages/settings.js` | UI has no validation/rotation status yet; current table renders synthetic active/latency state. This is expected Stage 0 baseline, not a Stage 0 blocker. |
| Rendered table | UI renders response fields `exchange_name`, `label`, masked `api_key`, `permissions`, `market_type`, `updated_at`; no secret/ciphertext/HMAC fields are read | `renderExchangeKeys` in settings JS | Нет. |

## Secret / Artifact Evidence

| Check | Result | Evidence |
| --- | --- | --- |
| API DTO fields | Response DTO contains only `key_id`, `exchange_name`, `market_type`, `label`, `permissions`, masked `api_key`, `created_at`, `updated_at` | `ExchangeKeyResponse` in `exchange_keys.py`; route CRUD test asserts exact key order. |
| Forbidden response fields | Tests assert `api_secret`, `passphrase`, `api_key_enc`, `api_secret_enc`, `passphrase_enc`, `api_key_hash` are absent from create/list responses | `test_exchange_keys_crud_routes_hide_secrets_and_apply_soft_delete`. |
| Ciphertext/fingerprint/HMAC | Current v1 API response does not expose any ciphertext, fingerprint, or `hmac` field because the response DTO has no such fields | `ExchangeKeyResponse`; `rg` evidence below. |
| Required artifact grep | `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE" logs output .playwright-cli || true` returned no matches; stderr noted missing `logs` directory | Gate run on 2026-05-23. |
| Runtime curl | Authenticated Keycloak e2e through public edge returned `/api/auth/current-user=200`, `/settings=200`, `/api/exchange-keys=200`, response shape `list_len=0`, and logout cleared the session (`/api/auth/current-user=401` after logout) | Smoke rerun on 2026-05-24 with dedicated e2e user. No credentials, cookie values, or exchange secrets were written to this report. |

## `market_type` Contract

| Layer | Current contract | Evidence |
| --- | --- | --- |
| SQL schema | `CHECK (market_type IN ('spot', 'futures'))` | `migrations/postgres/0003_identity_exchange_keys_v1.sql`. |
| API request DTO | `market_type: Literal["spot", "futures"]` | `CreateExchangeKeyRequest`. |
| API response DTO | `market_type: Literal["spot", "futures"]` | `ExchangeKeyResponse` and `_to_market_type_literal`. |
| Application validation | `_ALLOWED_MARKET_TYPES = {"spot", "futures"}` and invalid message `market_type must be one of: spot, futures.` | `CreateExchangeKeyUseCase`. |
| Domain entity | `_ALLOWED_MARKET_TYPES = {"spot", "futures"}` | `src/trading/contexts/identity/domain/entities/exchange_key.py`. |
| UI | Dropdown contains only `Spot` / `Futures`; JS defaults and fallback to `futures` | `exchange_keys.html`; `settings.js`. |
| Drift check | `rg -n "spot|futures|linear|inverse" ...` found `spot|futures` in schema/API/domain/UI and no active `linear|inverse` in the checked v1 surfaces | Gate run on 2026-05-23. |

## Verification Commands

| Command | Outcome |
| --- | --- |
| `uv run pytest -q tests/unit/apps/api/test_identity_exchange_keys_routes.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py` | Passed: `42 passed, 3 warnings in 1.64s`. Warnings are httpx per-request cookie deprecations in existing web route tests. |
| `rg -n "spot|futures|linear|inverse" migrations/postgres/0003_identity_exchange_keys_v1.sql src/trading/contexts/identity apps/api/routes apps/web/templates/fragments/account/exchange_keys.html apps/web/dist/js/pages/settings.js` | Confirmed v1 `spot|futures` across schema/API/application/domain/UI; no checked v1 `linear|inverse` occurrence. |
| `rg -n "TEST_SECRET|TEST_API_SECRET|TEST_PASSPHRASE" logs output .playwright-cli || true` | No matches; `logs` directory is absent locally. |
| `python -m tools.docs.generate_docs_index --check` | Initial check failed because `docs/architecture/README.md` was out of date after adding this report; `python -m tools.docs.generate_docs_index` updated the generated index; rerun passed: `OK: ... docs/architecture/README.md is up-to-date.` |
| Authenticated runtime smoke | Keycloak e2e login redirected with `307`; final URL was `/settings`; `/api/auth/current-user` returned `200`; `/settings` returned `200` and contained one `data-exchange-keys-endpoint="/api/exchange-keys"` marker; `/api/exchange-keys` returned `200` with `list_len=0`; forbidden response grep for `api_secret|passphrase|ciphertext|fingerprint|hmac|api_secret_enc|passphrase_enc|api_key_hash` returned no matches; logout returned `204`; after logout `/api/auth/current-user` returned `401`. |

## Contract Impact Classification

| Dimension | Classification | Reason |
| --- | --- | --- |
| Public API contract | `none` | Stage 0 changed no API behavior; it only records current `POST/GET/DELETE /api/exchange-keys` behavior. |
| DTO schema | `none` | Request/response DTOs unchanged. |
| Persisted schema | `none` | `identity_exchange_keys` and `market_type spot|futures` unchanged. |
| Browser-visible behavior | `none` | `/settings` templates/JS unchanged. |
| Config/defaults | `none` | No config edits. |
| Cache/request identity | `none` | No identity/hash/cache behavior changed. |
| Documentation | `compatible-change` | New baseline report documents current behavior for Stage 1. |

## Blockers To Stage 1

No implementation blocker found for starting Stage 1 from this baseline.

Residual gaps to carry into Stage 1:

- Current `/settings` exchange table shows synthetic status/latency and has no real validation status; Stage 1+ owns validation/read-model work.
- Current v1 storage uses encrypted blobs and API-key hash/last4 compatibility behavior; new `exchange_connections`, HMAC fingerprint, validation, rotation, audit, metrics, and operational controls remain future implementation work.
