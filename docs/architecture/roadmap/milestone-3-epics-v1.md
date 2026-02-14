# Milestone 3 — EPIC map (v1)

Карта EPIC’ов для Milestone 3: Telegram-only регистрация + 2FA-гейтинг ключей + Strategy v1 (immutable) + live runner + realtime streams + telegram notifications.


## Контекст и новые вводные (зафиксировано)

### Identity / доступ
- Пользователь может попасть в систему **только через Telegram auth** (один раз “регистрируется” телеграмом).
- Если пользователь хочет добавить **API keys биржи**, он **обязан включить 2FA** (TOTP).
- Оповещения пользователю (сигналы/вход/выход/ошибки) идут **через Telegram**.

### Strategy (immutable, per-user)
- Shared стратегий **нет**: даже если параметры одинаковые, стратегия всегда принадлежит одному пользователю.
- **Нет версионирования**: `1 стратегия = 1 версия`, менять нельзя. Любое “изменение” = новая стратегия.
- Разрешено “шаблонирование”: пользователь может создать новую стратегию **копированием spec** + сменой `instrument_id/timeframe`.
- Имя стратегии генерируется автоматически: `symbol + market_type + timeframe + indicator_names`.
- У стратегии отдельными тегами фиксируются: `symbol`, `market_type`, `timeframe`.
- Ключ инструмента:
  - `instrument_key: String = "{exchange}:{market_type}:{symbol}"`
  - domain identity инструмента: `(market_id, symbol)` как в shared kernel.

### Run states (автомат и допустимые переходы)
Состояния:
- `created -> run -> running -> stopping -> stopped/failed -> deleted`
Доп. переходы:
- `stopping -> run -> running`
- `stopped/failed -> run -> running`

## Принцип декомпозиции Milestone 3
Milestone 3 делится на 2 трека, которые можно делать параллельно:
1) `identity` (Telegram login + 2FA + exchange keys)
2) `strategy` (PG persistence + API + live worker + realtime output + telegram notify)

---

## Порядок внедрения (боевой, без лишней магии)

1) ID-EPIC-01 → Telegram auth + user bootstrap  
2) ID-EPIC-02 → 2FA (TOTP) + enforce policy  
3) ID-EPIC-03 → Exchange API keys (хранение, только после 2FA)  
4) STR-EPIC-01 → Strategy domain/spec (immutable) + PG DDL + migrations runner  
5) STR-EPIC-02 → Strategy API (CRUD+clone+run state machine) + 422 payloads  
6) STR-EPIC-03 → Strategy live worker (Redis 1m → rollup TF) + warmup/repair(read)  
7) STR-EPIC-04 → Realtime output streams v1 (UI metrics/events)  
8) STR-EPIC-05 → Telegram notifier (best-effort) + политики уведомлений  
9) STR-EPIC-06 → Configs/ops/runbooks/metrics hardening

---

## EPIC’и Milestone 3

### ID-EPIC-01 — Identity v1: Telegram-only registration (login) + user model

**Цель:** пользователь появляется в системе только через Telegram; дальше все запросы API выполняются в контексте `user_id`.

**Scope:**
- Новый bounded context: `src/trading/contexts/identity/*`
- Telegram login flow (на уровне API):
  - валидация Telegram Login Widget payload (или Bot-based handshake — выбрать один и зафиксировать)
  - создание `user_id (UUID)` и привязка `telegram_user_id`
  - хранение `telegram_chat_id` (если уже есть способ получить), иначе — этап 2/3
- Минимальный порт `CurrentUser` для API (контекст пользователя).

**Non-goals:**
- полноценные роли/права/админка
- OAuth провайдеры кроме Telegram

**PG DDL (минимум):**
- `identity_users(user_id uuid pk, telegram_user_id bigint unique, created_at, last_login_at, is_deleted)`
- `identity_telegram_channels(user_id fk, chat_id bigint, is_confirmed bool, confirmed_at)`

**DoD:**
- API умеет “впустить” пользователя только через Telegram auth.
- В любой strategy/backtest endpoint есть стабильный `user_id` в request context.
- Fail-fast: если секреты Telegram не заданы — сервис не стартует (кроме test/dev режимов по флагу).

**Paths:**
- `src/trading/contexts/identity/domain/*`
- `src/trading/contexts/identity/application/ports/*`
- `src/trading/contexts/identity/adapters/inbound/api/*`
- `migrations/pg/0001_identity_v1.sql`

---

### ID-EPIC-02 — 2FA v1 (TOTP): enable/verify + policy “keys require 2FA”

**Цель:** включение 2FA (TOTP) и принудительный гейт для операций с ключами биржи.

**Scope:**
- Генерация секрета TOTP, QR-данные для UI, подтверждение кодом.
- Хранение TOTP секрета **в зашифрованном виде** (envelope encryption; ключ в env/secret manager).
- Политика: `exchange keys` запрещены, пока `two_factor_enabled=true`.

**Non-goals:**
- recovery codes (можно оставить как OQ для следующего milestone)
- device management

**PG DDL:**
- `identity_2fa(user_id pk/fk, totp_secret_enc bytea, enabled bool, enabled_at, updated_at)`

**DoD:**
- `/2fa/setup` → отдает otpauth-uri/qr-data
- `/2fa/verify` → включает 2FA
- Любая операция “keys upsert” возвращает 403/422 без 2FA (единый payload).

---

### ID-EPIC-03 — Exchange API keys v1: storage + validation gates (no execution)

**Цель:** пользователь может сохранить свои API keys, но торговля/execution пока не делается.

**Scope:**
- Модель `ExchangeKey`:
  - `exchange_name` (`binance|bybit`)
  - `market_type` (`spot|futures`) или `market_id` (решение: хранить то, что нужно UI)
  - `label` (optional)
  - `api_key`, `api_secret`, `passphrase` (nullable, зависит от биржи)
  - `permissions` (read-only vs trade) — пока можно хранить как метаданные
- Шифрование секретов.
- Гейт: только при `2FA enabled`.

**Non-goals:**
- проверка ключей реальным запросом к бирже (можно как best-effort, но не обязательное)
- execution gateway

**PG DDL:**
- `identity_exchange_keys(key_id uuid pk, user_id fk, exchange_name text, market_type text, label text, api_key text, api_secret_enc bytea, passphrase_enc bytea, created_at, updated_at, is_deleted)`

**DoD:**
- API: create/list/delete keys
- Без 2FA — операция запрещена
- Логи не содержат секретов (ни при каких ошибках)

---

### STR-EPIC-01 — Strategy v1 domain/spec (immutable) + PG DDL + migrations automation

**Цель:** зафиксировать модель стратегии и хранилище: стратегия неизменяемая; все изменения через создание новой.

**Scope:**
- Новый bounded context: `src/trading/contexts/strategy/*`
- Domain:
  - `Strategy` entity (immutable spec)
  - `StrategySpecV1` (schema_version=1)
  - автогенерация имени (детерминированно)
  - теги: `symbol`, `market_type`, `timeframe`
- Runs:
  - состояние и допустимые переходы (state machine)
- Events:
  - история (truth) + основа для realtime/UI
- PG DDL: стратегии, runs, events, templates/clone lineage (optional).

**Non-goals:**
- backtest engine (Milestone 4)
- execution (Milestone 9)

**Миграции (профессионально, без ручного вмешательства):**
- Принять **Alembic** как механизм миграций, но писать миграции **явно** (без ORM/autogenerate).
- Автоприменение:
  - отдельный entrypoint `apps/migrations/main.py` (или отдельный compose job) делает `alembic upgrade head`
  - защита от гонок: `pg_advisory_lock` на время миграции
  - API/worker стартуют только после успеха миграций (fail-fast)
- В CI: поднимаем Postgres → `alembic upgrade head` → тесты.

**PG DDL (минимум):**
- `strategy_strategies(strategy_id uuid pk, user_id uuid fk, name text, instrument_id jsonb, instrument_key text, timeframe text, indicators_json jsonb, spec_json jsonb, created_at, is_deleted)`
  - `spec_json` неизменяем (только insert), обновления запрещены на уровне repo (и опционально constraint/trigger).
- `strategy_runs(run_id uuid pk, user_id uuid, strategy_id uuid, state text, started_at, stopped_at, checkpoint_ts_open timestamptz, last_error text, updated_at)`
- `strategy_events(event_id uuid pk, user_id uuid, strategy_id uuid, run_id uuid null, ts timestamptz, event_type text, payload_json jsonb)`

**DoD:**
- Domain валидируется unit-тестами (immutability, name generation, state transitions).
- Миграции применяются автоматически в CI и на деплое, без ручных шагов.

---

### STR-EPIC-02 — Strategy API v1: CRUD (immutable) + Clone-from-template + Run control

**Цель:** пользователь через API может создать стратегию, клонировать как шаблон, запустить/остановить.

**Scope:**
- Endpoints (пример):
  - `POST /strategies` (create, immutable)
  - `POST /strategies/clone` (template → new strategy)
  - `GET /strategies` (list by user)
  - `GET /strategies/{id}` (get)
  - `POST /strategies/{id}/run` (создать run + перевести в run/running по правилам)
  - `POST /strategies/{id}/stop` (stopping → stopped)
  - `DELETE /strategies/{id}` (soft delete → deleted)
- Единый 422 payload (как в indicators/market_data), детерминированный порядок ошибок.

**Non-goals:**
- UI realtime (это STR-EPIC-04)
- сложные права доступа (только owner)

**DoD:**
- Невозможно “обновить” стратегию: только новая.
- Clone работает: переносит indicator params, меняет `instrument_id/timeframe`.
- Run state machine соблюдается, включая:
  - `stopped/failed -> run -> running`
  - `stopping -> run -> running`

---

### STR-EPIC-03 — Strategy live worker v1: Redis 1m ingest → rollup TF → evaluate gate (warmup) + repair(read)

**Цель:** запущенные стратегии получают derived свечи (15m/1h/4h/1d) из live 1m и могут испускать events/metrics.

**Scope:**
- Новый процесс: `strategy-live-worker`
- Input:
  - Redis Streams `md.candles.1m.<instrument_key>` (schema v1) из market_data
- Rollup:
  - строго: **только закрытые и полные бакеты**
  - bucket alignment: `Timeframe.bucket_open(ts)` (epoch-aligned UTC)
- Warmup:
  - `warmup_bars` в spec (в барах timeframe стратегии)
  - seed через ClickHouse canonical 1m (ACL read) → rollup → ring buffer
  - пока warmup не набран: evaluation запрещён, но heartbeat/lag публикуются
- Repair(read):
  - если обнаружен gap по 1m (пропущенные минуты), worker дочитывает отсутствующие 1m **из ClickHouse canonical** (read-only)
  - ingestion/REST catchup не запускает (это задача market_data)
- Checkpointing:
  - хранить `checkpoint_ts_open` в `strategy_runs`

**Non-goals:**
- backtest
- торговля/execution

**DoD:**
- Worker устойчив к дублям/out-of-order Redis (idempotency in-memory + checkpoint).
- На одном инструменте и TF получаем стабильные derived свечи и events.
- Fail-fast: worker не стартует без PG/Redis/ClickHouse (если enabled=true).

---

### STR-EPIC-04 — Realtime output v1: Redis Streams для UI (metrics + events)

**Цель:** UI может подписаться на потоки пользователя и видеть метрики/события в реальном времени.

**Scope:**
- Streams:
  - `strategy.metrics.v1.user.<user_id>`
  - `strategy.events.v1.user.<user_id>`
- Payload schema v1 (все значения строками):
  - metrics: `schema_version, ts, strategy_id, run_id, metric_type, value, instrument_key, timeframe`
  - events: `schema_version, ts, strategy_id, run_id, event_type, payload_json`
- Best-effort semantics (как market_data publisher): ошибки Redis не ломают worker.

**DoD:**
- Документ + runbook команд redis-cli для проверки.
- Метрики publish_total/errors_total/duplicates_total.

---

### STR-EPIC-05 — Telegram notifier v1: best-effort adapter + notification policy

**Цель:** отправлять пользователю уведомления о ключевых событиях стратегий через Telegram.

**Scope:**
- Port `TelegramNotifier` в strategy context.
- Реализация:
  - берёт `chat_id` через ACL к identity (или из identity port)
  - отправляет сообщения best-effort (ошибки не роняют pipeline)
- Policy:
  - какие event_type шлём (например `signal`, `trade_open`, `trade_close`, `failed`)
  - rate-limit/anti-spam (минимально: debounce по одинаковым ошибкам)

**Non-goals:**
- сложные шаблоны сообщений, локализация
- guarantee delivery

**DoD:**
- В dev/test можно включить “log-only adapter”.
- В prod — реальная отправка, если identity chat binding подтверждён.

---

### STR-EPIC-06 — Configs/ops: configs/*/strategy.yaml + metrics ports + enable toggles + runbooks

**Цель:** привести strategy к тому же уровню “runtime discipline”, что market_data/indicators.

**Scope:**
- `configs/dev/strategy.yaml`, `configs/prod/strategy.yaml`, `configs/test/strategy.yaml`
- Валидатор + loader (fail-fast)
- Тумблеры:
  - `strategy.api.enabled`
  - `strategy.live_worker.enabled`
  - `strategy.realtime_output.redis_streams.enabled`
  - `strategy.telegram.enabled`
- Metrics port:
  - `strategy.metrics.port` (аналогично другим приложениям)
- Env overrides:
  - секреты через env (`TELEGRAM_BOT_TOKEN`, `PG_DSN`, `CH_DSN`, `REDIS_PASSWORD_ENV` и т.д.)
  - допускается override отдельных scalar (если есть единый механизм в проекте — использовать его)

**DoD:**
- Документы:
  - `docs/architecture/strategy/strategy-runtime-config-v1.md`
  - `docs/runbooks/strategy-live-worker.md`
- Smoke: “поднять стек → создать стратегию → запустить run → увидеть metrics/event в Redis → получить telegram notify (или log-only)”.

---

## Открытые вопросы (фиксируем, но не блокируем Milestone 3)
- ID/OQ-01: какой Telegram flow фиксируем в v1: Login Widget vs Bot handshake?
- ID/OQ-02: recovery для 2FA (backup codes) — в какой milestone?
- STR/OQ-01: запрещаем ли `1m` timeframe в live v1 (скорее да), а в backtest — разрешаем.
