## Куда класть `UserId` как новый примитив (по твоему дереву)

Лучший вариант для этого репозитория: **в shared-kernel**, потому что `user_id` будет нужен *сразу* в нескольких bounded context’ах (strategy/backtest/optimize/risk) и не должен тащить зависимость на `identity`-домен.

**Решение:**

* `src/trading/shared_kernel/primitives/user_id.py` — `UserId`
* `src/trading/shared_kernel/primitives/paid_level.py` — `PaidLevel` (free/base/pro/ultra) **тоже shared**, потому что его удобно читать в разных контекстах и прокидывать в JWT.
* Экспорт через:

  * `src/trading/shared_kernel/primitives/__init__.py`
  * при желании (если так уже принято) — `src/trading/shared_kernel/__init__.py`

**Почему не внутри identity:**

* Иначе другие контексты начнут импортировать `identity.domain...UserId`, что создаёт нежелательную связность bounded context’ов.
* Shared-kernel у тебя уже есть и используется как “общий язык” (Timeframe/InstrumentId/etc). `UserId` — точно такой же общий язык, просто не про рынок.

**Что остаётся в identity, а не в shared-kernel:**

* `TelegramUserId`, `TelegramChatId` (узкоспецифичные идентификаторы)
* агрегат/сущность `User` и правила upsert/reactivate

---

````md
# Identity v1: Telegram-only login + user model + CurrentUser

Документ фиксирует архитектуру identity-контекста: вход только через Telegram Login Widget (вариант A), выпуск JWT cookie и наличие `user_id` в контексте всех API запросов.

## Цель

1) Пользователь появляется в системе только через Telegram login (вариант A).  
2) После логина любые strategy/backtest/etc endpoint’ы получают стабильный `user_id` из request context.  
3) Заложить минимальную модель пользователя + `paid_level` (free/base/pro/ultra) и основу под Telegram notifications (chat_id table).

## Контекст

- Проект уже имеет DDD-структуру `src/trading/contexts/*` и shared primitives в `src/trading/shared_kernel/primitives/*`.
- В Milestone 3 нужно ввести понятие пользователя (`user_id`) как “сквозной идентификатор” для всех будущих доменов (strategy/backtest/optimize/…).
- Авторизация/аутентификация должна быть простой в v1: только Telegram (без паролей, без OAuth провайдеров).

## Scope

- Новый bounded context: `src/trading/contexts/identity/*` (domain + application ports + adapters).
- Telegram login flow: **Telegram Login Widget (вариант A)**:
  - валидация payload (hash/auth_date) по алгоритму Telegram,
  - создание `user_id: UUID` и привязка `telegram_user_id`,
  - обновление `last_login_at`,
  - выпуск JWT и установка его в **HttpOnly cookie** (TTL 7 дней).
- `paid_level` для каждого пользователя: `free|base|pro|ultra` (default: `free`).
- Минимальный порт `CurrentUser` для API: даёт `user_id` (и при необходимости `paid_level`) в контексте запроса.
- DDL + минимальные postgres-адаптеры репозиториев identity.

## Non-goals

- полноценные роли/права/админка (RBAC/ABAC)
- OAuth провайдеры кроме Telegram
- refresh-token схема, device/session management, revoke lists (в v1 — 1 JWT cookie на 7 дней)
- полноценный “удалить аккаунт” use-case и оркестрация purge по всем контекстам (в v1 закладываем модель и политику, но не обязаны реализовывать полный каскад)

## Ключевые решения

### 1) `UserId` и `PaidLevel` — shared-kernel primitives

**Решение:** `UserId` и `PaidLevel` размещаем в `src/trading/shared_kernel/primitives/*`, а не внутри identity-домена.

Причины:
- `user_id` потребуется сразу в нескольких bounded context’ах.
- shared-kernel — “общий язык”, который не создаёт связность контекстов.
- identity-контекст остаётся владельцем “логики и истины” по пользователю, но не единственным местом определения типа `UserId`.

Последствия:
- Другие контексты импортируют `UserId` из shared-kernel, не зная про identity.
- Telegram-специфичные идентификаторы остаются внутри identity-контекста.

### 2) Telegram auth: вариант A — Telegram Login Widget

**Решение:** используем Telegram Login Widget payload validation.

Минимальные правила валидации:
- payload проверяется по Telegram алгоритму (data_check_string + hash через HMAC-SHA256).
- `auth_date` проверяется на “свежесть” (например, не старше 24 часов) — чтобы исключить replay.

Последствия:
- UI обязан использовать Telegram Login Widget для получения подписанного payload.
- `telegram_chat_id` из Widget не получаем: таблицу каналов создаём заранее, подтверждение chat_id — отдельный будущий этап (через bot handshake).

### 3) JWT в HttpOnly cookie, TTL = 7 дней (упрощённый v1)

**Решение:** один JWT cookie, срок жизни 7 дней.

Cookie policy (v1):
- `HttpOnly=true`
- `Secure=true` (в prod)
- `SameSite=Lax` (или `Strict`, если UI и API на одном site и это не ломает UX)
- `Path=/`
- `Max-Age=604800` (7 дней)

Последствия:
- Пользователь логинится примерно раз в неделю.
- Нет server-side revoke в v1; компромисс — сравнительно короткий TTL.

### 4) Политика удаления: “можно восстановить при логине” (reactivate)

**Решение:** если пользователь помечен `is_deleted=true`, то при валидном Telegram логине:
- пользователь **реактивируется**: `is_deleted=false`, `last_login_at=now()`,
- **user_id сохраняется** (не меняется),
- downstream данные (стратегии/сделки/ключи) считаются удалёнными (purge делается отдельным use-case/оркестрацией, вне EPIC-01).

Последствия:
- Реактивация не создаёт новый `user_id` → у системы стабильная идентичность по telegram_user_id.
- Для “полного удаления всего” потребуется отдельная задача: очистка данных в других контекстах + возможный сброс `paid_level` в `free`.

### 5) Fail-fast конфигурация Telegram/JWT

**Решение:** в prod приложение не стартует, если не заданы секреты для:
- проверки Telegram payload,
- подписи JWT.

Исключение:
- dev/test могут запускаться по флагу (например `IDENTITY_FAIL_FAST=false`), чтобы не требовать реальные секреты локально.

Последствия:
- Ошибки конфигурации ловятся сразу при старте (а не в рантайме).
- Снижается риск “случайно отключили проверку подписи”.

## Контракты и инварианты

- **I-001:** Пользователь создаётся/находится только по `telegram_user_id`; в системе не существует “password login”.
- **I-002:** `telegram_user_id` уникален → 1 Telegram user == 1 `user_id`.
- **I-003:** `user_id` — UUID и является сквозным идентификатором во всех API контекстах.
- **I-004:** JWT обязателен для всех protected endpoints; отсутствие/невалидность → 401.
- **I-005:** Если `identity_users.is_deleted=true`, то доступ запрещён **до логина**, а при валидном логине → re-activate (см. решение 4).
- **I-006:** `paid_level ∈ {free, base, pro, ultra}`; default `free`.

## PG DDL (минимум)

Таблица пользователей:

- `identity_users(
    user_id uuid primary key,
    telegram_user_id bigint unique not null,
    paid_level text not null default 'free',
    created_at timestamptz not null,
    last_login_at timestamptz null,
    is_deleted boolean not null default false
  )`

Таблица Telegram-каналов (под будущие уведомления):

- `identity_telegram_channels(
    user_id uuid not null references identity_users(user_id),
    chat_id bigint not null,
    is_confirmed boolean not null default false,
    confirmed_at timestamptz null
  )`

Индексы/ограничения:
- unique index на `identity_users.telegram_user_id`
- unique index на `identity_telegram_channels.chat_id` (один chat_id не может принадлежать разным user)
- index на `identity_users.is_deleted` (опционально)

## Размещение файлов (план)

### Shared kernel primitives (новое)

- `src/trading/shared_kernel/primitives/user_id.py` — `UserId`
- `src/trading/shared_kernel/primitives/paid_level.py` — `PaidLevel`
- `src/trading/shared_kernel/primitives/__init__.py` — стабильные экспорты

### Identity context (новое)

- `src/trading/contexts/identity/domain/entities/user.py` — агрегат/сущность `User`
- `src/trading/contexts/identity/domain/value_objects/telegram_user_id.py` — `TelegramUserId`
- `src/trading/contexts/identity/domain/value_objects/telegram_chat_id.py` — `TelegramChatId` (если нужен тип)
- `src/trading/contexts/identity/domain/errors/*` — ошибки валидации/доступа

- `src/trading/contexts/identity/application/ports/repositories/user_repository.py` — `UserRepository`
- `src/trading/contexts/identity/application/ports/current_user.py` — `CurrentUser`
- `src/trading/contexts/identity/application/use_cases/telegram_login.py` — use-case логина

- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py` — endpoint логина/логаута
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py` — FastAPI dependency (читает cookie, валидирует JWT)

- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/*` — репозитории и gateway
- `src/trading/contexts/identity/adapters/outbound/security/jwt/*` — encode/decode JWT (если не выносим в platform)

### Apps wiring (обновить)

- `apps/api/routes/__init__.py` — подключить auth routes
- `apps/api/wiring/modules/identity.py` — wiring identity
- `apps/api/main/app.py` — подключить роутер + dependency overrides (если нужно)

### Migrations

- `migrations/postgres/0001_identity_v1.sql`

## Связанные файлы

- `src/trading/shared_kernel/primitives/` — shared primitives (пополняется `UserId`, `PaidLevel`)
- `apps/api/main/app.py` — composition root для API
- `docs/architecture/shared-kernel-primitives.md` — описание shared-kernel примитивов (добавить ссылку/секцию про `UserId`)
- `docs/architecture/roadmap/milestone-3-epics-v1.md` — эпики Milestone 3 (identity входит как базовый)

## Как проверить

```bash
# запускать из корня репозитория
ruff check .
pyright
pytest -q

# (дополнительно, после внедрения)
# 1) Логин: POST /auth/telegram/login (валидный payload) -> Set-Cookie
# 2) Protected endpoint (например /strategies) -> 200 и корректный user_id в контексте
````

## Риски и открытые вопросы

* Риск: cookie-based auth без CSRF-стратегии может быть уязвим на state-changing endpoints.

  * Митигация v1: `SameSite=Lax` минимум; v2: CSRF token/Double Submit для mutating запросов.
* Риск: replay атаки при отсутствии строгой проверки `auth_date`.

  * Митигация: reject если `now - auth_date > 24h` (или меньше).
* Вопрос: домен/сайт размещения UI и API (для точной настройки `SameSite` и CORS).
* Вопрос: политика `paid_level` при удалении аккаунта (сбрасывать в `free` или сохранять при re-activate).

````

После добавления — обновить индекс:
```bash
python -m tools.docs.generate_docs_index
````
