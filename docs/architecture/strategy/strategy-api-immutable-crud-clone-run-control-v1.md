````md
# Strategy API v1: immutable CRUD + clone + run control + identity integration (v1)

Фиксирует архитектурные решения для STR-EPIC-02: доменно-очевидные правила владения стратегией, неизменяемые стратегии, клонирование, управление запусками и интеграция с identity через порт `current_user`.

## Цель

Сделать так, чтобы пользователь через API мог:
- создать стратегию (immutable),
- клонировать стратегию из шаблона/существующей,
- просматривать только свои стратегии,
- запускать/останавливать исполнение (runs) с корректной state machine,
при этом правила были доменно-очевидными (в use-case), а identity использовалась через зависимость/порт (без копирования JWT-логики в strategy).

## Контекст

- В repo уже есть устоявшиеся практики: DDD слои (ports/adapters), детерминизм, fail-fast, единый стиль ошибок (422) в indicators/market_data.
- Нельзя допустить “случайного” поведения из SQL (например, фильтры ownership/deleted должны быть явными и тестируемыми как бизнес-правило).
- Strategy должна “логично использовать то что есть в identity”: текущий пользователь доступен через dependency/порт `current_user`.
- Runner обязан считать warmup детерминированно по индикаторам и позволять “второй run” после остановки (следующий запуск после stop).

## Scope

1) Endpoints (API v1)
- `POST /strategies` — create (immutable)
- `POST /strategies/clone` — clone from template/existing → new strategy
- `GET /strategies` — list by current user (owner)
- `GET /strategies/{id}` — get by id (owner)
- `POST /strategies/{id}/run` — create run + transition по правилам
- `POST /strategies/{id}/stop` — stop active run (stopping → stopped)
- `DELETE /strategies/{id}` — soft delete (deleted)

2) Application/use-cases (явные правила)
- Ownership/visibility rule оформляется use-case, а не “вшивается” в SQL.
- Все переходы состояния run оформлены доменно (и тестируются).

3) Identity integration
- `current_user` как порт/зависимость (из identity), используемый в use-cases Strategy.

4) Runner semantics
- Warmup считается в runner детерминированно (по индикаторам стратегии).
- “Второй run” после stop возможен: следующий запуск после остановки должен работать без ручных костылей.

5) Ошибки и 422 payload
- Используется общий механизм ошибок (см. отдельный документ `api-errors-and-422-payload-v1.md`).

## Non-goals

- UI realtime и стриминг статусов (это STR-EPIC-04).
- Сложные ACL/роли кроме owner (только владелец).
- Распределённый оркестратор запусков/кластеризация (в рамках v1 — корректный доменный контракт и API).

## Ключевые решения

### 1) Strategy — immutable entity (обновлений нет)

Стратегию нельзя “обновить”: любые изменения — это создание новой стратегии через clone (в т.ч. из шаблона).

Последствия:
- ✅ История изменений естественно выражается как набор версий-стратегий.
- ✅ Упрощаются кэширование/воспроизводимость и аудиторность.
- ⚠️ Нужен удобный clone endpoint (и понятная UX/документация).

### 2) Soft delete вместо физического удаления

`DELETE /strategies/{id}` переводит стратегию в `deleted` (или аналогичное состояние), физически не удаляя.

Последствия:
- ✅ Возможность аудита, восстановления, безопасные внешние ссылки.
- ⚠️ Все list/get use-cases обязаны учитывать deleted.

### 3) Доменно-очевидные правила ownership/visibility — только через use-case

Правило “пользователь видит только свои стратегии и только не-deleted” фиксируется в application layer (use-case), а репозиторий получает уже “скоупленые” параметры (owner_id, include_deleted=false) — без магии в SQL.

Пример набора use-cases:
- `ListMyStrategies(current_user)`
- `GetMyStrategy(strategy_id, current_user)`
- `DeleteMyStrategy(strategy_id, current_user)`
- `CloneMyStrategy(source_strategy_id, overrides, current_user)`

Последствия:
- ✅ Правило читается как бизнес-логика и тестируется unit-тестами без DB.
- ✅ SQL остаётся простым механизмом хранения/выборки.

### 4) Identity используется через порт `CurrentUser`

Strategy контекст не знает про JWT/HTTP. API слой предоставляет адаптер порта `CurrentUser` (или `CurrentUserProvider`) из identity-контекста.

Последствия:
- ✅ Нет копирования JWT логики в strategy.
- ✅ Use-cases тестируются с фейковым current_user.
- ⚠️ Нужен общий контракт пользователя (например `UserId`, `email` опционально).

### 5) Clone-from-template — основной способ “изменения” стратегии

`POST /strategies/clone`:
- копирует индикаторы/их параметры,
- позволяет изменить `instrument_id`/`timeframe` (и другие разрешённые overrides),
- создаёт новую стратегию с новым `strategy_id`, owner = current_user.

Последствия:
- ✅ Единая модель изменений.
- ⚠️ Нужно явно описать список разрешённых overrides и их валидацию.

### 6) Run state machine — доменная, детерминированная, поддерживает повторные запуски

Состояния (минимально необходимые по договорённости):
- `stopped` / `failed` → `run` → `running`
- `running` → `stop` → `stopping` → `stopped`
- “второй run” = следующий запуск после stop: **должен быть возможен**.

Ключевое: `POST /strategies/{id}/run` создаёт новый run (run_id) и инициирует переходы согласно правилам, не “переиспользуя” старый run.

Последствия:
- ✅ Повторные запуски после stop работают естественно.
- ✅ Легче хранить историю run’ов и метрики по ним.
- ⚠️ Нужны ограничения конкурентности (см. следующее решение).

### 7) Конкурентность: один активный run на стратегию + идемпотентность на уровне use-case

Правило: в один момент времени у стратегии не может быть >1 активного run (`run/running/stopping`).

Реализация (в application):
- проверка текущего активного run,
- если уже `running` → вернуть доменную ошибку “already running”,
- если `stopping` → доменная ошибка “cannot start while stopping” (или политика “wait” — но в v1 проще fail-fast),
- операции `run/stop` защищаются транзакцией/optimistic lock (версионность агрегата) либо уникальным индексом “active run”.

Последствия:
- ✅ Предсказуемость, отсутствие гонок.
- ⚠️ Нужно аккуратно оформить в репозитории/хранилище.

### 8) Warmup считается в runner детерминированно по индикаторам

Warmup не “приходит из UI” и не зависит от случайной выборки:
- runner вычисляет warmup, исходя из набора индикаторов стратегии,
- вычисление детерминированное (одинаковая стратегия → одинаковый warmup),
- warmup фиксируется в метаданных run (для трассировки и воспроизводимости).

Последствия:
- ✅ Детерминизм и повторяемость.
- ✅ Меньше ручных ошибок в конфигурации.
- ✅ Используется единая функция `estimate_strategy_warmup_bars(...)` в application layer.

### 9) Детерминированный порядок everywhere

- Списки (strategies/runs/errors) сортируются детерминированно.
- Валидационные ошибки (422) сортируются детерминированно (см. отдельный документ про 422).

Последствия:
- ✅ Стабильные тесты, предсказуемый API.

### 10) Ошибки: доменные ошибки → RoehubError → HTTPException (через общий слой)

Strategy/use-cases возвращают/бросают доменные ошибки, которые мапятся в `RoehubError(code, message, details)`, а затем API слой мапит `RoehubError` в `HTTPException`.

Последствия:
- ✅ Единый контракт ошибок по всему API.
- ✅ API слой остаётся тонким и предсказуемым.
- ⚠️ Требуется общий модуль ошибок (см. отдельный документ).

## Контракты и инварианты

- Стратегия immutable: нет endpoint “update strategy”, любые изменения через clone.
- Пользователь видит/управляет только своими стратегиями (owner).
- Soft delete: deleted стратегии не возвращаются в list/get по умолчанию.
- Один активный run на стратегию в момент времени.
- Повторный run после stop возможен всегда (если стратегия не deleted и нет активного run).
- Warmup вычисляется runner’ом детерминированно из индикаторов.
- Ошибки и 422 payload единообразны и детерминированы.

## Связанные файлы

- `apps/api/routes/strategies.py` — HTTP endpoints Strategy (тонкие)
- `apps/api/wiring/strategy_container.py` — composition root для Strategy
- `src/trading/contexts/strategy/application/use_cases/*.py` — use-cases (ownership, clone, run/stop)
- `src/trading/contexts/strategy/application/ports/current_user.py` — порт `CurrentUser`/`CurrentUserProvider`
- `src/trading/contexts/strategy/domain/*.py` — агрегаты Strategy/Run + доменные ошибки
- `src/trading/contexts/strategy/adapters/outbound/*` — репозитории/хранилища
- `docs/architecture/api/api-errors-and-422-payload-v1.md` — общий контракт ошибок/422 (см. отдельный документ)
- `docs/architecture/indicators/*` — референс по стилю API/ошибок (422)

## Как проверить

```bash
# запускать из корня репозитория

uv run ruff check .
uv run pyright
uv run pytest -q

# обновить индекс документации
uv run python -m tools.docs.generate_docs_index
````

## Риски и открытые вопросы

* Риск: гонки `run/stop` при параллельных запросах → решается транзакцией/optimistic lock/уникальным ограничением “active run”.
* Риск: неверная оценка warmup для сложных индикаторных графов → покрыть unit-тестами “warmup estimator” и контрактными тестами runner’а.
* Открытые вопросы: отсутствуют (все согласовано).

````

**Куда кладём:** `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`



```md
# API errors v1: RoehubError + deterministic 422 payload (v1)

Фиксирует единый контракт ошибок Roehub и общий формат 422 payload для всего API, включая маппинг доменных ошибок в `RoehubError` и перевод `RoehubError` в `HTTPException` на уровне API.

## Цель

- Ввести единый объект ошибки: `RoehubError(code, message, details)` в `src/trading/platform/errors/`.
- Сделать общий 422 payload на уровне API (единый формат, детерминированный порядок ошибок).
- Стандартизировать маппинг: domain/app ошибки → `RoehubError` → HTTP response.

## Контекст

- В indicators/market_data уже есть ожидание “единый 422 payload, детерминированный порядок ошибок”.
- Нужно избежать “зоопарка” исключений и форматов ошибок между контекстами.
- Strategy и другие контексты должны отдавать доменно понятные ошибки, не зависящие от FastAPI.

## Scope

1) `RoehubError` (platform-level)
- структура: `code: str`, `message: str`, `details: dict | None`
- используется как “канонический” контракт ошибки внутри приложения

2) 422 payload (API-level)
- единый JSON-ответ для ошибок валидации/контрактов
- сортировка ошибок детерминирована (по path/loc)

3) Маппинг доменных ошибок
- в каждом контексте: доменные ошибки мапятся в `RoehubError`
- API слой конвертирует `RoehubError` в `HTTPException` (или response) единообразно

## Non-goals

- Полный переход всех legacy ошибок одномоментно (допускается поэтапная миграция).
- RFC7807/ProblemDetails как внешний стандарт (в v1 — внутренний контракт Roehub; при желании можно мапить наружу позже).

## Ключевые решения

### 1) RoehubError — единственный “канонический” объект ошибки

Вводим `RoehubError(code, message, details)` в `src/trading/platform/errors/`.
- `code` — стабильный машинный код (snake/kebab — один стиль выбрать и закрепить)
- `message` — человекочитаемое сообщение
- `details` — структурированные данные (например поля валидации, ids, текущие состояния)

Последствия:
- ✅ Единая форма ошибок между контекстами.
- ✅ Проще тестировать: use-case возвращает/бросает RoehubError-совместимые ошибки без FastAPI.
- ⚠️ Нужно дисциплинированно поддерживать реестр/стабильность `code`.

### 2) Общий 422 payload в API слое, детерминированная сортировка ошибок

Делаем общий модуль в `apps/api/*` (общий для всех роутов), который:
- перехватывает ошибки валидации (Pydantic/FastAPI) и/или доменные “validation” ошибки,
- приводит их к единому payload,
- сортирует ошибки детерминированно.

Рекомендуемый payload (единый для ошибок Roehub и 422, с кодом 422):
```json
{
  "error": {
    "code": "validation_error",
    "message": "Validation failed",
    "details": {
      "errors": [
        { "path": "body.strategy.name", "code": "required", "message": "Field is required" }
      ]
    }
  }
}
````

Правило детерминизма:

* `details.errors[]` сортируется по `path`, затем по `code`, затем по `message` (лексикографически).

Последствия:

* ✅ Стабильные ответы → стабильные тесты и UX.
* ✅ Одинаковый формат для всех контекстов.
* ⚠️ Нужно аккуратно адаптировать raw ошибки FastAPI/Pydantic к `path`.

### 3) Domain/App errors мапятся в RoehubError до API слоя

В application layer (или на границе context-а) вводим явный маппинг:

* доменные исключения/ошибки (например `StrategyAlreadyRunning`, `StrategyNotFound`, `ForbiddenNotOwner`)
* → `RoehubError` с предсказуемыми `code/message/details`

API слой не должен знать доменные типы (только `RoehubError`).

Последствия:

* ✅ API слой тонкий.
* ✅ Доменные ошибки не “просачиваются” в HTTP контракт напрямую.

### 4) API слой мапит RoehubError → HTTPException/Response

В API wiring добавляется единый обработчик, который:

* принимает `RoehubError`,
* выбирает HTTP статус по `code` (таблица маппинга),
* возвращает payload `{ "error": ... }`.

Пример таблицы (фиксируем как контракт):

* `validation_error` → 422
* `not_found` → 404
* `forbidden` → 403
* `conflict` → 409
* `unauthorized` → 401
* `unexpected_error` → 500

Последствия:

* ✅ Единый формат и статусы.
* ⚠️ Требует дисциплины по использованию `code`.

## Контракты и инварианты

* Любая ошибка, уходящая из API, имеет форму `{ "error": { "code", "message", "details" } }`.
* 422 всегда детерминированен: порядок `details.errors[]` стабилен.
* Доменные контексты не импортируют FastAPI/HTTP; они работают с доменными ошибками и/или `RoehubError`.
* API слой не импортирует доменные типы ошибок напрямую — только `RoehubError`.

## Связанные файлы

* `src/trading/platform/errors/__init__.py` — стабильный экспорт ошибок
* `src/trading/platform/errors/roehub_error.py` — `RoehubError(code, message, details)`
* `apps/api/common/errors.py` — обработчики/маппинг RoehubError → HTTP
* `apps/api/main/*.py` или `apps/api/app.py` — регистрация exception handlers FastAPI
* `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md` — использование в Strategy
* `docs/architecture/indicators/*` — референс по существующему ожиданию 422 payload

## Как проверить

```bash
# запускать из корня репозитория

uv run ruff check .
uv run pyright
uv run pytest -q

# отдельный фокус: проверка детерминизма 422
uv run pytest -q -k "validation_error and deterministic"

# обновить индекс документации
uv run python -m tools.docs.generate_docs_index
```

## Риски и открытые вопросы

* Риск: дрейф `code` между контекстами → нужен минимальный реестр/гайдлайн (в `src/trading/platform/errors/README.md` или в docs).
* Риск: сложность маппинга Pydantic loc → `path` → покрыть тестами на типовые кейсы (body/query/path).
* Открытые вопросы: отсутствуют (все согласовано).

````
