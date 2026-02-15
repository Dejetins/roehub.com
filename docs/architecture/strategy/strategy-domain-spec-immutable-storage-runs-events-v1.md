**Расположение:** `docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md`
**Название (H1):** `Strategy v1 — Immutable Spec + Storage + Runs/Events + Migrations Automation`

````md
# Strategy v1 — Immutable Spec + Storage + Runs/Events + Migrations Automation

Документ фиксирует доменную модель Strategy v1 (immutable spec), хранение в Postgres (strategies/runs/events) и правила автоприменения миграций через Alembic.

## Цель

1) Зафиксировать Strategy как **неизменяемую** сущность: любые изменения spec → только созданием новой стратегии.  
2) Зафиксировать минимальную модель хранения и аудита: стратегии, запуски (runs), события (events).  
3) Зафиксировать профессиональный механизм миграций Postgres: Alembic + явные миграции + автоприменение с защитой от гонок.

## Контекст

- Репозиторий использует DDD-структуру `src/trading/contexts/*` и уже имеет Postgres слой для `identity` (явный SQL + минимальный gateway).  
- В `shared_kernel` уже зафиксированы примитивы:
  - `InstrumentId` как доменная идентичность `(market_id, symbol)` и каноничная сериализация `as_dict()`.
  - `Timeframe` со строгой валидацией кода (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`).  
- Цель Milestone 3: заложить корректный фундамент стратегии и исполнения live runner (без backtest/execution).

См. также:
- `docs/architecture/strategy/strategy-milestone-3-epics-v1.md` — пользовательский сценарий и границы Milestone 3.

## Scope

Входит в v1:

- Новый bounded context `src/trading/contexts/strategy/*`:
  - Domain: `Strategy` (immutable spec), `Run` state machine, `Event` (append-only).
  - Application ports: репозитории для strategies/runs/events.
  - Adapters: Postgres persistence (SQL через gateway).
- StrategySpecV1 (schema_version=1) как неизменяемый JSON-спек.
- Детерминированная генерация имени стратегии:
  - базовая часть “человеческая”
  - суффикс — короткий стабильный хэш от `(user_id + spec_json)`
  - формат: `BTCUSDT 1m [MA(20,50)] #A1B2C3D4`
- Теги v1: `symbol`, `market_type`, `timeframe` (derived для фильтрации/индекса).
- Runs:
  - максимум **1 активный run** на стратегию в состояниях `starting|warming_up|running|stopping`
  - хранение checkpoint для продолжения “с места остановки”
- Events:
  - append-only история (truth) + база для realtime/UI
  - `run_id` допускается `NULL` (события уровня стратегии, не привязанные к конкретному запуску)
- Миграции Postgres:
  - Alembic как механизм
  - миграции пишем явно (без ORM/autogenerate)
  - автоприменение через entrypoint `apps/migrations/main.py` (или отдельный compose job)
  - `pg_advisory_lock` на время миграции
  - API/worker стартуют **только после** успешного `upgrade head` (fail-fast)
  - в CI: поднять Postgres → `alembic upgrade head` → тесты

## Non-goals

- Backtest engine (Milestone 4).
- Execution (Milestone 9): ордера, позиции, комисcии, PnL.
- Распределённый scaling live runner >1 instance (дизайн допускает, но реализация v1 может быть single-worker).
- DSL сигналов (v1 — template-based).
- Ротация ключей/операционные runbooks миграций исторических данных вне baseline (в v1 предполагается “baseline уже применён вручную”).

## Ключевые решения

### 1) Strategy immutable: любые изменения через создание новой стратегии

StrategySpec (spec_json) является source-of-truth и **не обновляется**.

Причина:
- воспроизводимость, аудит, отсутствие “тихих” изменений стратегии во времени.

Последствия:
- spec-редактирование в UI/API реализуется как: `create new strategy` + (опционально) `soft-delete old`.
- repository layer запрещает UPDATE spec-полей.
- `is_deleted` допускает UPDATE как исключение (см. решение 2).

### 2) Soft-delete разрешён как единственный UPDATE для strategies

Стратегия immutable по spec, но `is_deleted` можно изменять для soft-delete.

Причина:
- практичность управления списком стратегий (архивирование/удаление без физического DELETE).

Последствия:
- только одно поле допускает обновление: `is_deleted`.
- все остальные поля стратегии (включая `spec_json`) — insert-only.

### 3) Instrument identity и key: используем instrument_key как тег, instrument_id как структуру

- `instrument_key` хранится как каноничная строка: `"{exchange}:{market_type}:{symbol}"`.
- `instrument_id` хранится как JSON, но в v1 достаточно минимальной каноники: `{"market_id": <int>, "symbol": "<BTCUSDT>"}`.

Почему так:
- `InstrumentId` — доменный ID: `(market_id, symbol)`; он стабилен для домена.
- `instrument_key` — удобен для интеграций/логов/streams и строится детерминированно через runtime mapping market_id → (exchange, market_type).

Ссылки:
- `src/trading/shared_kernel/primitives/instrument_id.py` — `InstrumentId.as_dict()`
- `src/trading/contexts/market_data/adapters/outbound/config/instrument_key.py` — `build_instrument_key(...)`

Последствия:
- API/доменные DTO используют `InstrumentId` как “истину” идентичности.
- `instrument_key` — денормализованный тег и удобный индекс.

### 4) Timeframe валидируется примитивом shared_kernel и допускает 1m

`Timeframe` уже проверяет допустимые коды и нормализует `code`.

Ссылка:
- `src/trading/shared_kernel/primitives/timeframe.py`

Последствия:
- в домене/DTO стратегии используем `Timeframe`, а в БД храним строковое `timeframe` (code).
- `1m` в v1 **разрешён** (не запрещён).

### 5) Run state machine v1: starting → warming_up → running → stopping → stopped | failed

Состояния:
- `starting`
- `warming_up`
- `running`
- `stopping`
- `stopped`
- `failed`

Причина:
- warmup в Milestone 3 — реальное поведение; его нужно отражать для UI/метрик/аудита.
- “paused” пока не нужен.

Последствия:
- в domain должны быть unit-тесты переходов (deterministic).
- правило “1 активный run на strategy” enforced на уровне repository/constraint.

### 6) checkpoint_ts_open: прогресс обработки для возобновления

`checkpoint_ts_open` — timestamp последней обработанной базовой свечи (по ts_open), чтобы продолжать без дублей.

Пример:
- последняя обработанная 1m свеча имела `ts_open=2026-02-14T10:37:00Z`
- тогда `checkpoint_ts_open=2026-02-14T10:37:00Z`

Последствия:
- после рестарта worker может “догонять” с checkpoint.
- события/метрики не дублируются.

### 7) Events append-only, run_id nullable

Events — append-only (truth log).  
`run_id` допускает `NULL`, чтобы хранить события уровня стратегии (created/deleted/renamed-by-new-version и т.п.) отдельно от run-событий.

Последствия:
- “лента стратегии” строится простой выборкой по `strategy_id ORDER BY ts`.
- “лента run” строится `WHERE run_id = ... ORDER BY ts`.

### 8) Миграции Postgres: Alembic + явные SQL миграции + autoupgrade + advisory lock

Принимаем Alembic как механизм версионирования схемы, но:
- пишем миграции явно (DDL/SQL), без autogenerate/ORM.

Автоприменение:
- отдельный entrypoint `apps/migrations/main.py` (или compose job) делает `alembic upgrade head`.
- на время миграций берём `pg_advisory_lock`, чтобы исключить гонки.

Fail-fast:
- API/worker стартуют только после успешного применения миграций.

CI:
- поднять Postgres → `alembic upgrade head` → тесты.

## Контракты и инварианты

- StrategySpecV1:
  - `schema_version=1` обязателен в `spec_json`.
  - `spec_json` immutable (insert-only).
- Имя стратегии:
  - детерминированно вычисляется от `(user_id + spec_json)`
  - формат: `<human-part> #<8hex>` (например `#A1B2C3D4`)
- Instrument:
  - `InstrumentId` каноничен как `(market_id, symbol)` и сериализуется через `as_dict()`.
  - `instrument_key` строится детерминированно как `{exchange}:{market_type}:{symbol}`.
- Timeframe:
  - валидируется через `Timeframe` (`code` нормализуется и проверяется).
- Runs:
  - максимум 1 активный run на strategy в состояниях `starting|warming_up|running|stopping`.
- Events:
  - append-only (никаких UPDATE/DELETE в пределах truth log).
  - `run_id` nullable (события уровня стратегии допускаются).
- Soft-delete:
  - единственный допустимый UPDATE в `strategy_strategies` — изменение `is_deleted`.

## Хранилище (Postgres) — минимальный DDL v1

Ниже — целевая структура таблиц (миграции реализуются через Alembic, но DDL здесь фиксирует контракт).

### strategy_strategies

- `strategy_id uuid pk`
- `user_id uuid`
- `name text`
- `instrument_id jsonb` — каноника `InstrumentId.as_dict()`: `{"market_id": <int>, "symbol": "<BTCUSDT>"}`
- `instrument_key text` — `{exchange}:{market_type}:{symbol}`
- `market_type text` — `spot|futures` (derived tag)
- `symbol text` — `BTCUSDT` (derived tag)
- `timeframe text` — `Timeframe.code`
- `indicators_json jsonb` — денормализованный набор индикаторов/params для UI/поиска (опционально, если уже есть в spec)
- `spec_json jsonb` — StrategySpecV1, immutable
- `created_at timestamptz`
- `is_deleted boolean`

Правило immutability:
- UPDATE `spec_json` запрещён на уровне репозитория (и опционально — через trigger/constraint).

### strategy_runs

- `run_id uuid pk`
- `user_id uuid`
- `strategy_id uuid`
- `state text`
- `started_at timestamptz`
- `stopped_at timestamptz null`
- `checkpoint_ts_open timestamptz null`
- `last_error text null`
- `updated_at timestamptz`

### strategy_events

- `event_id uuid pk`
- `user_id uuid`
- `strategy_id uuid`
- `run_id uuid null`
- `ts timestamptz`
- `event_type text`
- `payload_json jsonb`

## Связанные файлы

Shared Kernel:
- `src/trading/shared_kernel/primitives/instrument_id.py` — доменная идентичность инструмента + `as_dict()`
- `src/trading/shared_kernel/primitives/timeframe.py` — валидатор/нормализатор timeframe + bucket alignment API

Market Data (для instrument_key):
- `src/trading/contexts/market_data/adapters/outbound/config/instrument_key.py` — каноничный `instrument_key`

Strategy context (будущая реализация):
- `src/trading/contexts/strategy/` — новый bounded context (domain/application/adapters)
- `src/trading/contexts/strategy/application/ports/repositories/` — contracts для strategies/runs/events repositories
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/` — Postgres adapters

Migrations:
- `migrations/postgres/` — legacy SQL baseline (уже применён вручную в prod)
- `apps/migrations/main.py` — новый entrypoint автоприменения alembic (создать)
- `alembic.ini` + `alembic/` — конфиг и версии миграций (создать)

Документация:
- `docs/architecture/strategy/strategy-milestone-3-epics-v1.md` — общий scope Milestone 3

## Как проверить

```bash
# запускать из корня репозитория

# 1) линтер
uv run ruff check .

# 2) типы
uv run pyright

# 3) тесты
uv run pytest -q

# 4) индекс документации (если добавлялся/обновлялся этот документ)
uv run python -m tools.docs.generate_docs_index
````

## Риски и открытые вопросы

* Риск: “baseline уже применён вручную” — важно корректно перенести текущее состояние БД в Alembic (через stamp), чтобы CI/деплой не пытались переиграть историю.
* Риск: запрет UPDATE на `spec_json` должен быть enforced минимум на уровне repository (триггер — опционально, но требует аккуратного сопровождения).
* Вопрос (на след. шаг): где хранить DSN/конфиг для `apps/migrations/main.py` (env vars vs config файлы) и как это встроить в deploy compose job.

```
```
