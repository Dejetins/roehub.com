````md
# Identity exchange keys storage + 2FA gate policy v2

Документ фиксирует контракт ID-EPIC-03: безопасное хранение exchange API keys (storage-only) с шифрованием **всех** чувствительных полей и обязательным 2FA gate для CRUD v1.

## Цель

Добавить в identity контур управление биржевыми ключами (`create/list/delete`) так, чтобы:

- **api_key / api_secret / passphrase** хранились **только в зашифрованном виде** (envelope AES-GCM),
- для детерминированного дедупа/индексов использовался **хэш api_key** (без хранения plaintext),
- все операции были **недоступны без включённой 2FA** (существующий reusable gate),
- API **никогда** не возвращал секреты и не раскрывал их в ошибках/логах,
- list/delete не раскрывали существование чужих ключей (delete всегда “not found”).

## Контекст

Уже реализованы:

- Telegram login + JWT cookie auth (`docs/architecture/identity/identity-telegram-login-user-model-v1.md`),
- TOTP 2FA + reusable policy gate/dependency (`docs/architecture/identity/identity-2fa-totp-policy-v1.md`) с точным 403 payload,
- Exchange keys CRUD v1 (v1 вариант) — но **api_key хранился plaintext**, а AAD был только namespace.

Текущая версия (v2) уточняет контракт: **шифруем api_key**, добавляем **api_key_hash** для уникальности, и усиливаем AAD (binding к user_id/key_id/field).

## Scope

- Postgres storage для exchange keys (v2 schema):
  - `exchange_name ∈ {'binance','bybit'}`
  - `market_type ∈ {'spot','futures'}`
  - `permissions ∈ {'read','trade'}`
  - soft-delete: `is_deleted + deleted_at`
  - хранение:
    - `api_key_enc BYTEA NOT NULL`
    - `api_key_hash BYTEA NOT NULL` (для уникальности/поиска)
    - `api_key_last4 TEXT NOT NULL` (для UI masked без дешифрования)
    - `api_secret_enc BYTEA NOT NULL`
    - `passphrase_enc BYTEA NULL`
- API endpoints:
  - `POST /exchange-keys`
  - `GET /exchange-keys`
  - `DELETE /exchange-keys/{key_id}`
- Шифрование секретов:
  - envelope AES-GCM (`aesgcm_envelope_v1`),
  - отдельный KEK env: `IDENTITY_EXCHANGE_KEYS_KEK_B64`,
  - AAD включает namespace + `user_id` + `key_id` + `field_name`.
- Политика доступа:
  - все endpoints exchange keys требуют auth + enabled 2FA через существующий reusable gate,
  - без 2FA — строго фиксированный 403 payload.

## Non-goals

- Execution/trading операции.
- Проверка валидности ключей через реальные вызовы `binance`/`bybit` API (может появиться позже как best-effort).
- Возврат `api_key`/`api_secret`/`passphrase`/encrypted blob из API.
- Любые новые auth-механизмы за пределами текущего JWT cookie flow.
- Процедура ротации KEK и re-encrypt существующих записей (выносится в отдельный runbook/ADR).

## Ключевые решения

### 1) Шифрование api_key + хэш для уникальности и индексов

В v2 **api_key не хранится plaintext**. Вместо этого:

- `api_key_enc` — ciphertext blob (envelope AES-GCM),
- `api_key_hash` — детерминированный хэш API ключа (для unique/index),
- `api_key_last4` — последние 4 символа для UI masked (`****last4`) без дешифрования.

Рекомендуемый алгоритм хэша:

- `api_key_hash = SHA-256(normalized_api_key_utf8)` (32 bytes, хранить как `BYTEA`).
- Normalization: `strip()` (как сейчас в use-case).

Последствия:

- ✅ исключаем утечку api_key из БД даже при read-only компрометации,
- ✅ сохраняем детерминированный дедуп и быстрые индексы без хранения plaintext,
- ✅ list может быть без дешифрования api_key (используем `api_key_last4`).

### 2) Envelope AES-GCM с отдельным KEK и усиленным AAD (binding)

Шифруем `api_key/api_secret/passphrase` через DEK+KEK envelope AES-GCM.

- KEK: `IDENTITY_EXCHANGE_KEYS_KEK_B64` (отдельный от 2FA)
- AAD (v2): `namespace + "|" + user_id + "|" + key_id + "|" + field_name`

Пример namespace:
- `roehub.identity.exchange_keys.v2`

Пример field_name:
- `api_key`, `api_secret`, `passphrase`

Последствия:

- ✅ ciphertext нельзя “перенести” между пользователями/записями/полями без детекта (GCM tag fail),
- ✅ blast-radius меньше (KEK разделён с 2FA),
- ✅ формат остаётся versioned/opaque.

### 3) 2FA gate обязателен для всех exchange keys endpoints (без дублирования)

Все `POST/GET/DELETE /exchange-keys` подключают существующий `RequireTwoFactorEnabledDependency`.

Если 2FA выключена, возвращается ровно:

`{"error":"two_factor_required","message":"Two-factor authentication must be enabled."}`

Последствия:

- ✅ единая политика безопасности во всех клиентах,
- ✅ детерминированный UX/контракт.

### 4) Конфликт активного дубля — 409 с детерминированным кодом ошибки

При попытке создать активный дубликат по ключу уникальности возвращаем:

- HTTP `409 Conflict`
- payload:
  - `error`: `exchange_key_already_exists`
  - `message`: `Exchange API key already exists.`

(Код уже соответствует смыслу и стабилен; менять не требуется.)

Последствия:

- ✅ детерминированный контракт для UI,
- ✅ можно обрабатывать как “уже добавлено”.

### 5) Delete семантика “не найдено” для missing/not-owned/already-deleted

`DELETE /exchange-keys/{key_id}`:

- всегда возвращает `204` только когда запись реально soft-deleted,
- иначе возвращает `404` с:
  - `error`: `exchange_key_not_found`
  - `message`: `Exchange API key was not found.`

Последствия:

- ✅ не раскрываем существование чужих key_id,
- ✅ повторное удаление неотличимо от “не существует”.

## Контракты и инварианты

- `exchange_name IN ('binance','bybit')`
- `market_type IN ('spot','futures')`
- `permissions IN ('read','trade')`
- `api_key_enc IS NOT NULL`
- `api_key_hash IS NOT NULL` (ровно 32 байта для SHA-256)
- `api_key_last4 IS NOT NULL` (len 1..4, после normalize)
- `api_secret_enc IS NOT NULL`
- `passphrase_enc IS NULL` допустим (зависит от биржи)
- soft-delete инвариант:
  - `(is_deleted = TRUE AND deleted_at IS NOT NULL) OR (is_deleted = FALSE AND deleted_at IS NULL)`
- `GET /exchange-keys` возвращает только `is_deleted = false`
- детерминированная сортировка list: `created_at ASC, key_id ASC`
- без 2FA любой endpoint exchange keys возвращает фиксированный 403 payload (см. выше)
- в логах/ошибках **не должно быть** `api_key/api_secret/passphrase` и их encrypted blobs

## DDL (v2)

Целевой shape таблицы `identity_exchange_keys` (поля v2):

- `key_id UUID PRIMARY KEY`
- `user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE`
- `exchange_name TEXT NOT NULL CHECK (...)`
- `market_type TEXT NOT NULL CHECK (...)`
- `label TEXT NULL`
- `permissions TEXT NOT NULL CHECK (...)`
- `api_key_enc BYTEA NOT NULL`
- `api_key_hash BYTEA NOT NULL`
- `api_key_last4 TEXT NOT NULL`
- `api_secret_enc BYTEA NOT NULL`
- `passphrase_enc BYTEA NULL`
- `created_at TIMESTAMPTZ NOT NULL`
- `updated_at TIMESTAMPTZ NOT NULL`
- `is_deleted BOOLEAN NOT NULL DEFAULT FALSE`
- `deleted_at TIMESTAMPTZ NULL`

Индексы:

- `INDEX (user_id, is_deleted, created_at, key_id)`
- `INDEX (user_id, created_at, key_id) WHERE is_deleted = FALSE`
- `UNIQUE (user_id, exchange_name, market_type, api_key_hash) WHERE is_deleted = FALSE`

## Связанные файлы

- `docs/architecture/identity/identity-2fa-totp-policy-v1.md` — reusable 2FA gate policy и deterministic 403
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — auth/current-user модель
- `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md` — этот документ
- `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py` — HTTP endpoints (create/list/delete)
- `src/trading/contexts/identity/application/use_cases/create_exchange_key.py` — нормализация, шифрование, хэширование, создание
- `src/trading/contexts/identity/application/use_cases/list_exchange_keys.py` — детерминированная выдача
- `src/trading/contexts/identity/application/use_cases/delete_exchange_key.py` — soft-delete (404 for non-owned/missing)
- `src/trading/contexts/identity/application/use_cases/exchange_keys_errors.py` — детерминированные ошибки (409/404/422)
- `src/trading/contexts/identity/adapters/outbound/security/exchange_keys/aes_gcm_envelope_secret_cipher.py` — envelope AES-GCM (AAD binding v2)
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/exchange_keys_repository.py` — postgres storage adapter
- `apps/api/wiring/modules/identity.py` — wiring + fail-fast по `IDENTITY_EXCHANGE_KEYS_KEK_B64`
- `migrations/postgres/0003_identity_exchange_keys_v1.sql` — текущая миграция v1 (будет расширена новой миграцией v2)

## Как проверить

```bash
# запускать из корня репозитория

python -m tools.docs.generate_docs_index

ruff check .
uv run pyright
uv run pytest -q
````

## Риски и открытые вопросы

* Риск: требуется миграция данных v1 → v2 (api_key plaintext → api_key_enc/hash/last4). Нужно выбрать стратегию:

  * либо принудительный purge v1 ключей в dev/test,
  * либо фоновой “re-encrypt” миграцией (в prod).
* Риск: отсутствие runbook по ротации `IDENTITY_EXCHANGE_KEYS_KEK_B64` (нужен отдельный документ + процедура re-encrypt).
* Риск: при неаккуратном логировании исключений возможно раскрытие чувствительных полей; нужны тесты/ревью на “no secrets in logs”.
* Вопрос: политика retention/purge для soft-deleted ключей (сколько храним и когда физически удаляем).

```
```
