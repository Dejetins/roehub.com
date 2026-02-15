````md
# Identity 2FA TOTP policy v1

Документ фиксирует минимальный TOTP-флоу (setup/verify) и политику “exchange keys require 2FA” для Roehub Identity (milestone 3, ID-EPIC-02).

## Цель

Включить 2FA (TOTP) для пользователя и гарантировать, что любые операции управления биржевыми ключами доступны только при включённой 2FA.

## Контекст

Уже реализован `ID-EPIC-01` (Telegram-only login + `user_id` + JWT cookie) и создан bounded context `src/trading/contexts/identity/*`. Теперь нужно добавить 2FA v1 поверх существующего контекста и подготовить переиспользуемый gate/dependency для будущего контекста “exchange keys”.

Ограничения и предпосылки:

- В v1 используется TOTP (RFC 6238) и стандартный `otpauth://` URI.
- QR генерируется на стороне UI: API возвращает только `otpauth_uri`.
- Секрет TOTP хранится только в зашифрованном виде (envelope encryption).
- Сброс/выключение 2FA не входит в этот этап.

## Scope

- Новый PG DDL для `identity_2fa`:
  - хранение `totp_secret_enc` (bytea),
  - флаги `enabled`, `enabled_at`, `updated_at`,
  - FK/PK `user_id` (на `identity_users`).
- API endpoints (inbound):
  - `POST /2fa/setup` — возвращает `otpauth_uri` для UI (QR рисуется на UI),
  - `POST /2fa/verify` — проверяет код и включает 2FA.
- Envelope encryption:
  - KEK берётся из env (`IDENTITY_2FA_KEK_B64`),
  - секрет шифруется на бекенде перед записью в Postgres.
- Политика “keys require 2FA”:
  - любые операции exchange keys (create/update/delete/list/read) должны быть запрещены, пока 2FA **не включена**,
  - готовим переиспользуемый identity gate/dependency/port для будущего подключения в “exchange keys”.
- Fail-fast:
  - в prod при включённом `IDENTITY_FAIL_FAST=true` сервис не стартует без `IDENTITY_2FA_KEK_B64`,
  - dev/test могут иметь ослабление по флагу (аналогично ID-EPIC-01).

## Non-goals

- Recovery codes.
- Device management / trusted devices.
- Flow выключения/сброса 2FA (включая подтверждение через Telegram).
- Реализация exchange keys (только gate/policy для будущего).

## Ключевые решения

### 1) TOTP setup возвращает только `otpauth_uri` (QR на UI)

`POST /2fa/setup` возвращает строку `otpauth_uri`, UI строит QR самостоятельно.

Параметры `otpauth_uri`:

- `issuer = "Roehub"`
- `account_label = <user_id>` (без PII, стабильно)

Причины:

- меньше поверхностей утечки (не возвращаем base64/png/svg),
- проще бекенд (не тащим QR-генераторы),
- стандартный и переносимый формат для большинства UI/клиентов.

Последствия:

- UI обязан уметь строить QR по `otpauth_uri`,
- нужно строго запретить логирование `otpauth_uri`/секрета (policy на уровне логов/трассировки).

### 2) Option 1: после включения 2FA нельзя переинициализировать setup/verify

Поведение:

- если `enabled=true`: `POST /2fa/setup` и `POST /2fa/verify` не выдают новый секрет и не меняют состояние; возвращают детерминированный отказ (409/422 по контракту API).
- если `enabled=false`:
  - `POST /2fa/setup` генерирует новый секрет и **перезаписывает** незавершённый setup (секрет в БД обновляется),
  - `POST /2fa/verify` проверяет код и включает 2FA.

Причины:

- защищает от сценария “злоумышленник получил доступ к сессии и переинициализировал 2FA на своё устройство”,
- уменьшает риск случайного “перевыпуска” 2FA самим пользователем.

Последствия:

- для безопасного сброса/выключения 2FA понадобится отдельный flow (вне scope), вероятно с подтверждением через Telegram.

### 3) Хранение секрета через envelope encryption (AES-GCM), KEK из env

TOTP секрет хранится в `identity_2fa.totp_secret_enc` только в зашифрованном виде:

- KEK: `IDENTITY_2FA_KEK_B64` (base64),
- алгоритм: AES-GCM,
- формат `totp_secret_enc`: opaque blob (версионируемый формат внутри).

Причины:

- секрет не должен лежать в БД в открытом виде,
- KEK отделён от базы (можно менять/ротировать отдельно).

Последствия:

- нужен fail-fast на наличие KEK в prod,
- операции verify требуют дешифрования секрета на бекенде.

### 4) Политика “exchange keys require 2FA” реализуется как reusable gate/dependency

Так как контекст “exchange keys” ещё не реализован, делаем в `identity` переиспользуемый компонент:

- application port: проверка “2FA включена для user_id”
- inbound dependency (FastAPI): `RequireTwoFactorEnabledDependency` (используется как `Depends(...)`)

Единый отказ для любых операций “keys” при выключенной 2FA:

- HTTP status: 403
- payload: `{"error":"two_factor_required","message":"Two-factor authentication must be enabled."}` (детерминированно)

Последствия:

- при появлении ключей в API роуты просто подключат dependency без копипасты логики.

## Контракты и инварианты

- `identity_2fa.user_id` — PK/FK на `identity_users(user_id)`; одна запись 2FA на пользователя.
- `enabled=true` ⇒ `enabled_at IS NOT NULL`.
- `totp_secret_enc` всегда `NOT NULL` и содержит только зашифрованный секрет.
- `POST /2fa/setup` доступен только для аутентифицированного пользователя (JWT cookie).
- `POST /2fa/verify` доступен только для аутентифицированного пользователя (JWT cookie).
- Если `identity_2fa.enabled=true`, повторный setup/verify запрещён (Option 1).
- Политика keys:
  - любые операции exchange keys (create/update/delete/list/read) запрещены, пока 2FA **не включена**,
  - запрет оформлен через единый gate/dependency/port (deterministic behaviour/payload).
- Fail-fast:
  - в `prod` при `IDENTITY_FAIL_FAST=true` отсутствие `IDENTITY_2FA_KEK_B64` запрещает старт сервиса.

## Связанные файлы

- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — базовый identity v1 (Telegram-only login, JWT cookie).
- `migrations/postgres/0001_identity_v1.sql` — текущие таблицы identity v1.
- `migrations/postgres/0002_identity_2fa_totp_v1.sql` — новая миграция (добавить в рамках EPIC-02).
- `apps/api/routes/__init__.py` — экспорт роутеров (`build_identity_router`).
- `apps/api/wiring/modules/identity.py` — wiring/settings/fail-fast identity (расширить под KEK/2FA).
- `apps/api/routes/identity.py` / `apps/api/routes/identity.py` (фактически wiring identity router) — точка сборки identity API.
- `src/trading/contexts/identity/*` — bounded context identity:
  - `adapters/inbound/api/*` — endpoints и dependencies,
  - `application/ports/*` — ports для 2FA status/gate и crypto,
  - `adapters/outbound/*` — Postgres repo + crypto/encryption,
  - `domain/*` — value objects/инварианты (если нужно).
- `pyproject.toml` — добавить зависимости `pyotp`, `cryptography`.

## Как проверить

```bash
# запускать из корня репозитория

# обновить индекс docs
python -m tools.docs.generate_docs_index

# линтер
ruff check .

# типы
uv run pyright

# тесты
pytest -q
````

## Риски и открытые вопросы

* Риск: утечка `otpauth_uri` или секретов через логи/трассировку (особенно в debug). Влияние: компрометация 2FA.
* Риск: управление жизненным циклом KEK (ротация) не входит в v1. Влияние: операционный долг; требуется ADR/процедура позже.
* Вопрос (на следующий milestone): безопасный “disable/reset 2FA” flow (вероятно через Telegram-подтверждение + возможно delay) и recovery codes.

```
```
