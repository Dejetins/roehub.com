# Identity exchange keys storage + 2FA gate policy v2

Документ фиксирует контракт ID-EPIC-03 для хранения exchange API keys (storage-only) c обязательным 2FA gate и шифрованием всех чувствительных полей.

## Цель

Обновить storage/API-контракт exchange keys v1 до v2 так, чтобы:

- `api_key`, `api_secret`, `passphrase` хранились только в зашифрованном виде (envelope AES-GCM),
- уникальность активных ключей обеспечивалась через детерминированный `api_key_hash` (без plaintext `api_key` в БД),
- API оставался без утечек секретов (только masked `api_key`),
- 2FA gate и payload отказа оставались строго неизменными,
- delete не раскрывал существование/принадлежность ключа (`404` для missing/foreign/already-deleted).

## Контекст

Уже есть:

- Telegram login + JWT cookie auth (`identity-telegram-login-user-model-v1.md`),
- TOTP 2FA + reusable gate/dependency (`identity-2fa-totp-policy-v1.md`),
- exchange keys CRUD v1 (`identity-exchange-keys-storage-2fa-gate-policy-v1.md`).

v1 хранил `api_key` в plaintext. v2 заменяет это на `api_key_enc + api_key_hash + api_key_last4`.

## Scope

- Postgres storage `identity_exchange_keys`:
  - `api_key_enc BYTEA NOT NULL`
  - `api_key_hash BYTEA NOT NULL`
  - `api_key_last4 TEXT NOT NULL`
  - `api_secret_enc BYTEA NOT NULL`
  - `passphrase_enc BYTEA NULL`
  - soft-delete: `is_deleted` + `deleted_at`
- API endpoints:
  - `POST /exchange-keys`
  - `GET /exchange-keys`
  - `DELETE /exchange-keys/{key_id}`
- Шифрование:
  - envelope AES-GCM, KEK: `IDENTITY_EXCHANGE_KEYS_KEK_B64`
  - AAD binding: `roehub.identity.exchange_keys.v2|user_id|key_id|field_name`
- Дубликаты:
  - `409 Conflict`, `{"error":"exchange_key_already_exists","message":"Exchange API key already exists."}`

## Non-goals

- Execution/trading gateway и любые order flows.
- Онлайн-проверка ключей через Binance/Bybit API.
- Возврат plaintext/encrypted secret fields в API.
- Изменение поведения/payload 2FA gate.

## Ключевые решения

### 1) Хранение `api_key`: шифрование + hash

- `api_key` не хранится plaintext.
- В БД сохраняются:
  - `api_key_enc` (ciphertext),
  - `api_key_hash = SHA-256(normalized_api_key_utf8)`,
  - `api_key_last4` для UI-маскировки без дешифрования.

### 2) Уникальность только по hash

Partial unique index для активных ключей:

- `(user_id, exchange_name, market_type, api_key_hash) WHERE is_deleted = FALSE`

Это оставляет дедуп детерминированным и не требует plaintext `api_key`.

### 3) 2FA gate без изменений

Для всех exchange keys endpoints без включённой 2FA возвращается точно:

`{"error":"two_factor_required","message":"Two-factor authentication must be enabled."}`

### 4) Delete не раскрывает существование

`DELETE /exchange-keys/{key_id}`:

- `204` при успешном soft-delete,
- `404` для missing/foreign/already-deleted.

### 5) Детерминированный conflict

`POST /exchange-keys` при активном дубле возвращает:

- HTTP `409`,
- `error = exchange_key_already_exists`,
- `message = Exchange API key already exists.`

## Контракты и инварианты

- `exchange_name IN ('binance','bybit')`
- `market_type IN ('spot','futures')`
- `permissions IN ('read','trade')`
- `octet_length(api_key_hash) = 32`
- `char_length(api_key_last4) BETWEEN 1 AND 4`
- soft-delete invariant:
  - `(is_deleted = TRUE AND deleted_at IS NOT NULL) OR (is_deleted = FALSE AND deleted_at IS NULL)`
- list order: `created_at ASC, key_id ASC`
- API не возвращает `api_secret`, `passphrase`, `api_key_enc`, `api_secret_enc`, `passphrase_enc`, `api_key_hash`.

## DDL (v2)

- Базовая таблица создаётся миграцией v1: `migrations/postgres/0003_identity_exchange_keys_v1.sql`.
- Переход на v2 выполняется миграцией: `migrations/postgres/0004_identity_exchange_keys_v2.sql`.
- В v2 миграции добавляются `api_key_enc/api_key_hash/api_key_last4`, удаляется plaintext `api_key`, пересоздаётся unique index по hash.

## Связанные файлы

- `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md`
- `docs/architecture/identity/identity-2fa-totp-policy-v1.md`
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md`
- `src/trading/contexts/identity/application/use_cases/create_exchange_key.py`
- `src/trading/contexts/identity/application/use_cases/delete_exchange_key.py`
- `src/trading/contexts/identity/application/use_cases/list_exchange_keys.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`
- `src/trading/contexts/identity/adapters/outbound/security/exchange_keys/aes_gcm_envelope_secret_cipher.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/exchange_keys_repository.py`
- `migrations/postgres/0004_identity_exchange_keys_v2.sql`

## Как проверить

```bash
python -m tools.docs.generate_docs_index
ruff check .
uv run pyright
uv run pytest -q
```

## Риски и открытые вопросы

- Для окружений с уже существующими v1 данными нужен отдельный runbook/процедура re-encrypt.
- Ротация `IDENTITY_EXCHANGE_KEYS_KEK_B64` не входит в scope и должна быть оформлена отдельным runbook/ADR.
