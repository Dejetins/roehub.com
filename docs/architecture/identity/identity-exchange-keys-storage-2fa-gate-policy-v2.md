# Identity exchange keys storage policy v2

Документ фиксирует актуальный storage/API-контракт `exchange keys` для локальной аутентификации.

## Статус

- active
- supersedes: `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md`

## Цель

- безопасное хранение API-ключей биржи (storage-only);
- отсутствие утечек секретов через API/логи;
- auth через локальную Roehub session model, без local 2FA dependency.

## Scope

- Endpoints:
  - `POST /exchange-keys`
  - `GET /exchange-keys`
  - `DELETE /exchange-keys/{key_id}`
- Storage invariants:
  - `api_key` не хранится plaintext;
  - `api_key_enc`, `api_secret_enc`, `passphrase_enc` — encrypted blobs;
  - `api_key_hash` + `api_key_last4` используются для дедупа и masked UI;
  - soft-delete (`is_deleted`, `deleted_at`).

## Auth и policy

- API-auth: только через `RequireCurrentUserDependency` (cookie `roehub_session_id`).
- Local `/2fa/*` и local 2FA gate в API отсутствуют.
- Для чувствительных изменений требуется свежая локальная аутентификация (`recent-auth`).
- Keycloak и внешний OIDC удалены; внешнего step-up провайдера нет.

## Контракты

- duplicate активного ключа: `409` + `exchange_key_already_exists`;
- delete чужого/отсутствующего/уже удалённого ключа: `404`;
- list сортировка: `created_at ASC, key_id ASC`;
- API не возвращает `api_secret`, `passphrase`, `api_key_enc`, `api_secret_enc`, `passphrase_enc`, `api_key_hash`.

## DDL

- baseline: `migrations/postgres/0003_identity_exchange_keys_v1.sql`
- v2 update: `migrations/postgres/0004_identity_exchange_keys_v2.sql`

## Связанные документы

- `docs/architecture/identity/local-auth-sessions-recovery-v1.md`
- `docs/runbooks/offline-release-installation.md`

## Связанные файлы

- `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/adapters/outbound/security/exchange_keys/aes_gcm_envelope_secret_cipher.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/exchange_keys_repository.py`
