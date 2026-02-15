# Identity exchange keys storage + 2FA gate policy v1

Документ фиксирует контракт ID-EPIC-03: безопасное хранение exchange API keys (только storage) с обязательным 2FA gate для всех операций CRUD v1.

## Цель

Добавить в identity v1 управление биржевыми ключами (`create/list/delete`) так, чтобы:
- `api_secret` и `passphrase` всегда хранились только в зашифрованном виде,
- операции exchange keys были недоступны без включённой 2FA,
- API никогда не возвращал секреты и не раскрывал их в ошибках.

## Контекст

Уже реализованы:
- Telegram login + JWT cookie auth (`docs/architecture/identity/identity-telegram-login-user-model-v1.md`),
- TOTP 2FA + reusable policy gate/dependency (`docs/architecture/identity/identity-2fa-totp-policy-v1.md`) с точным 403 payload.

ID-EPIC-03 расширяет identity-контур новым storage API для exchange keys без выполнения трейдинга и без сетевой валидации ключей на биржах.

## Scope

- Postgres таблица `identity_exchange_keys` (Variant A):
  - `exchange_name ∈ {'binance','bybit'}`,
  - `market_type ∈ {'spot','futures'}`,
  - `permissions ∈ {'read','trade'}`,
  - soft-delete через `is_deleted` + `deleted_at`.
- API endpoints:
  - `POST /exchange-keys`
  - `GET /exchange-keys`
  - `DELETE /exchange-keys/{key_id}`
- Шифрование секретов:
  - envelope AES-GCM (`aesgcm_envelope_v1`),
  - отдельный KEK env: `IDENTITY_EXCHANGE_KEYS_KEK_B64`,
  - отдельный AAD namespace для exchange keys.
- Политика доступа:
  - все endpoints exchange keys требуют auth + enabled 2FA через существующий reusable gate.

## Non-goals

- Execution/trading операции.
- Проверка валидности ключей через вызовы `binance`/`bybit` API.
- Возврат `api_secret`/`passphrase`/encrypted blob из API.
- Новые auth-механизмы за пределами текущего JWT cookie flow.

## Ключевые решения

### 1) Envelope AES-GCM с отдельным KEK для exchange keys

Секретные поля (`api_secret`, `passphrase`) шифруются в versioned opaque blob через AES-GCM envelope (DEK + KEK).

- KEK: `IDENTITY_EXCHANGE_KEYS_KEK_B64`
- AAD: `roehub.identity.exchange_keys.v1`
- В `prod` по умолчанию (или при `IDENTITY_FAIL_FAST=true`) отсутствие KEK блокирует старт.

Последствия:
- уменьшается blast-radius при утечке БД,
- KEK для 2FA и KEK для exchange keys разделены.

### 2) 2FA gate обязателен для всех exchange keys endpoints

`POST/GET/DELETE /exchange-keys` подключают существующий `RequireTwoFactorEnabledDependency`.

Если 2FA выключена, возвращается ровно:
`{"error":"two_factor_required","message":"Two-factor authentication must be enabled."}`.

Последствия:
- единая политика безопасности без дублирования логики,
- детерминированный UX/контракт для всех клиентов.

### 3) Секреты никогда не возвращаются API

`GET /exchange-keys` и `POST /exchange-keys` возвращают только non-secret поля.
`api_key` в ответе возвращается в masked виде (например `****ABCD`) для снижения риска утечки.

Последствия:
- отсутствуют каналы утечки `api_secret`/`passphrase` через API,
- UI получает достаточно данных для управления ключами.

### 4) Soft-delete вместо hard delete

Удаление ключа выполняется как:
- `is_deleted = true`
- `deleted_at = now()`
- `updated_at = now()`

`GET /exchange-keys` возвращает только `is_deleted = false`.

Последствия:
- сохраняется audit-friendly история,
- возможна будущая retention/purge политика.

## Контракты и инварианты

- `exchange_name IN ('binance','bybit')`
- `market_type IN ('spot','futures')`
- `permissions IN ('read','trade')`
- `api_secret_enc` всегда `NOT NULL`.
- `deleted_at IS NOT NULL` только если `is_deleted=true`.
- `GET /exchange-keys` сортируется детерминированно: `created_at ASC, key_id ASC`.
- 2FA gate обязателен для всех операций exchange keys.
- В логах/ошибках не должно быть `api_secret`, `passphrase`, encrypted blob.

## DDL (v1)

`identity_exchange_keys`:
- `key_id UUID PRIMARY KEY`
- `user_id UUID NOT NULL REFERENCES identity_users(user_id)`
- `exchange_name TEXT NOT NULL CHECK (exchange_name IN ('binance','bybit'))`
- `market_type TEXT NOT NULL CHECK (market_type IN ('spot','futures'))`
- `label TEXT NULL`
- `permissions TEXT NOT NULL CHECK (permissions IN ('read','trade'))`
- `api_key TEXT NOT NULL`
- `api_secret_enc BYTEA NOT NULL`
- `passphrase_enc BYTEA NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `is_deleted BOOLEAN NOT NULL DEFAULT FALSE`
- `deleted_at TIMESTAMPTZ NULL`

Индексы:
- `(user_id, is_deleted, created_at, key_id)`
- partial index для активных ключей (`WHERE is_deleted = FALSE`)
- partial unique index на активные дубли `(user_id, exchange_name, market_type, api_key)`

## Связанные файлы

- `docs/architecture/identity/identity-2fa-totp-policy-v1.md` — reusable 2FA gate policy и deterministic 403.
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — auth/current-user модель.
- `migrations/postgres/0003_identity_exchange_keys_v1.sql` — DDL exchange keys v1.
- `apps/api/routes/identity.py` — подключение exchange-keys роутов.
- `apps/api/wiring/modules/identity.py` — fail-fast и wiring зависимостей.
- `src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py` — reusable 2FA dependency.
- `src/trading/contexts/identity/adapters/outbound/security/exchange_keys/aes_gcm_envelope_secret_cipher.py` — envelope cipher для exchange keys.

## Как проверить

```bash
python -m tools.docs.generate_docs_index
ruff check .
uv run pyright
pytest -q
```

## Риски и открытые вопросы

- Риск: пока нет процедуры ротации `IDENTITY_EXCHANGE_KEYS_KEK_B64` (нужен отдельный runbook/ADR).
- Риск: при неаккуратном логировании ошибок можно раскрыть чувствительные поля; нужны тесты на отсутствие секретов в payload.
- Вопрос: срок хранения soft-deleted ключей и политика purge (вне scope v1).
