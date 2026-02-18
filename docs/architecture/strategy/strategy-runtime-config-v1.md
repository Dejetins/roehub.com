# Strategy runtime config v1: `configs/*/strategy.yaml` + toggles + metrics port + env overrides

Документ фиксирует STR-EPIC-06: единый runtime-конфиг для Strategy (API + live worker) с fail-fast валидацией, enable-тумблерами, метриками и поддержкой env overrides.

## Цель

Привести Strategy к тому же уровню "runtime discipline", что `market_data`/`indicators`:

- единый человекочитаемый YAML в `configs/<env>/strategy.yaml` (source of truth),
- строгая валидация на старте (fail-fast),
- явные enable-тумблеры для компонент,
- единый порт метрик через конфиг (с CLI override),
- секреты только через env (конфиг хранит имена env переменных, а не значения).

## Контекст

На момент STR-EPIC-06:

- Live worker Strategy уже существует и использует отдельный YAML `configs/*/strategy_live_runner.yaml` с собственным loader/validator: `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py`.
- Metrics port для воркеров задается CLI-флагом `--metrics-port` (как в market_data), а не через YAML:
  - `apps/worker/strategy_live_runner/main/main.py`
  - `apps/worker/market_data_ws/main/main.py`
- Strategy API router сейчас всегда подключается в FastAPI app и не имеет enable toggle: `apps/api/main/app.py`.
- В проекте уже есть паттерн загрузки YAML по env+`ROEHUB_ENV` и scalar overrides через env (env -> YAML -> default):
  - `src/trading/platform/config/indicators_compute_numba.py`
  - `apps/api/wiring/modules/indicators.py`

Цель STR-EPIC-06 — унифицировать запуск/эксплуатацию Strategy без ломки текущих точек входа, поэтому `strategy_live_runner.yaml` не удаляется, а становится alias/shim на `strategy.yaml`.

## Scope

- Новые конфиги:
  - `configs/dev/strategy.yaml`
  - `configs/test/strategy.yaml`
  - `configs/prod/strategy.yaml`
- Loader + валидатор (fail-fast) для `strategy.yaml`.
- Alias/shim поддержка для `configs/*/strategy_live_runner.yaml` (сохранение совместимости с текущими entrypoints).
- Enable тумблеры (фиксированные key-path):
  - `strategy.api.enabled`
  - `strategy.live_worker.enabled`
  - `strategy.realtime_output.redis_streams.enabled`
  - `strategy.telegram.enabled`
- Metrics port:
  - `strategy.metrics.port`
- Env overrides:
  - `ROEHUB_STRATEGY_CONFIG` (путь к YAML)
  - overrides для scalar: enable-тумблеры + metrics port
  - секреты через env: `STRATEGY_PG_DSN`, `TELEGRAM_BOT_TOKEN`, `ROEHUB_REDIS_PASSWORD`, `CH_HOST/CH_PORT/...`.

## Non-goals

- Hot reload конфигурации (конфиг читается на старте процесса).
- Унификация/переименование env ключей на новые DSN-имена (в v1 используем текущие env имена репозитория).
- Добавление метрик в API-процесс (в рамках v1 фиксируется порт метрик для strategy live worker; API метрики можно добавить отдельным эпиком).
- Глобальный "distributed override" механизм для произвольных nested ключей (в v1 override ограничен скалярами, перечисленными в этом документе).

## Ключевые решения

### 1) Source of truth: `configs/*/strategy.yaml`

`configs/<env>/strategy.yaml` — единый источник runtime-настроек Strategy.

Причины:
- операторский формат (читается/редактируется без кода),
- единый файл для включения/отключения компонент и ports,
- меньше дрейфа настроек между процессами.

Последствия:
- все composition roots (API и worker) читают один и тот же конфиг (возможны разные подпути и разные enable-ветки).

### 2) `strategy_live_runner.yaml` остается, но становится alias/shim на `strategy.yaml`

Чтобы не ломать существующие entrypoints/скрипты/докер-команды, файл `configs/<env>/strategy_live_runner.yaml` сохраняется.
При этом он **не дублирует** реальный конфиг, а указывает на `strategy.yaml`.

Минимальный shim-формат v1:

```yaml
version: 1
strategy_live_runner:
  config_ref:
    path: "configs/dev/strategy.yaml"
```

Loader live-worker обязан поддерживать два режима:
- legacy: полный payload `strategy_live_runner: { ... }` (как сейчас),
- shim: `strategy_live_runner.config_ref.path` -> загрузить `strategy.yaml` и извлечь из него нужные секции.

Последствия:
- нет drift между `strategy.yaml` и `strategy_live_runner.yaml`;
- миграция может происходить поэтапно: сначала добавляем `strategy.yaml`, затем переводим `strategy_live_runner.yaml` на shim.

### 3) Разрешение пути конфигурации: CLI/ENV > `ROEHUB_ENV`

В проекте допускаются два механизма выбора конфига:

- CLI `--config` (существующий механизм воркеров),
- env override `ROEHUB_STRATEGY_CONFIG`,
- fallback: `configs/<ROEHUB_ENV>/strategy.yaml` (по аналогии с indicators).

Прецедент для такого поведения: `src/trading/platform/config/indicators_compute_numba.py`.

Последствия:
- запуск в docker/CI может оставаться через CLI;
- локально/на хосте можно управлять файлом без изменения команд.

### 4) Enable toggles определяют поведение процессов (не только wiring)

Фиксируем семантику тумблеров:

- `strategy.api.enabled=false`:
  - Strategy router не подключается в FastAPI app;
  - API продолжает работать для остальных модулей (identity/indicators).

- `strategy.live_worker.enabled=false`:
  - процесс `strategy-live-runner` пишет лог "disabled" и завершается с `exit 0`.

- `strategy.realtime_output.redis_streams.enabled=false`:
  - live worker использует `NoOpStrategyRealtimeOutputPublisher`, даже если Redis параметры указаны.

- `strategy.telegram.enabled=false`:
  - live worker использует `NoOpTelegramNotifier`.

Последствия:
- конфиг становится реальным ops-инструментом управления компонентами;
- исключаются "полузапуски" с неявными включенными фичами.

### 5) `strategy.metrics.port` + CLI override `--metrics-port`

`strategy.metrics.port` задает порт Prometheus endpoint (`/metrics`) для strategy live worker.

Правило приоритета:
- CLI `--metrics-port` override > `strategy.metrics.port` из YAML > default.

Последствия:
- сохраняется существующий паттерн запуска воркеров (как в market_data),
- появляется единый конфиговый дефолт и единая точка согласования ports.

### 6) Secrets через env: YAML хранит только имена env переменных

В v1 не кладем секреты в YAML. В YAML допускается только:
- `password_env` для Redis,
- `bot_token_env` для Telegram,
а сами значения читаются из `os.environ`.

Каноничные env ключи (как уже используется в репозитории):
- Postgres (strategy): `STRATEGY_PG_DSN` (см. `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`, `apps/api/wiring/modules/strategy.py`)
- Telegram token: `TELEGRAM_BOT_TOKEN` (см. wiring identity и strategy)
- Redis password: `ROEHUB_REDIS_PASSWORD` (используется в `configs/*/*.yaml`)
- ClickHouse: `CH_HOST/CH_PORT/CH_USER/CH_PASSWORD/CH_DATABASE/CH_SECURE/CH_VERIFY` (см. `apps/cli/wiring/db/clickhouse.py`)

Последствия:
- безопаснее (секреты не версионируются),
- совместимо с текущими docker-compose подходами.

### 7) Env overrides только для скаляров (строгий whitelist)

В v1 поддерживаем env overrides только для:

- `strategy.api.enabled`
- `strategy.live_worker.enabled`
- `strategy.realtime_output.redis_streams.enabled`
- `strategy.telegram.enabled`
- `strategy.metrics.port`

Рекомендуемые env ключи:
- `ROEHUB_STRATEGY_API_ENABLED`
- `ROEHUB_STRATEGY_LIVE_WORKER_ENABLED`
- `ROEHUB_STRATEGY_REALTIME_OUTPUT_REDIS_STREAMS_ENABLED`
- `ROEHUB_STRATEGY_TELEGRAM_ENABLED`
- `ROEHUB_STRATEGY_METRICS_PORT`

Парсинг boolean должен быть строгим и предсказуемым (например как в `apps/api/wiring/modules/identity.py`: 1/0/true/false/yes/no/on/off).

Последствия:
- минимальная сложность и минимальный риск "магических" overrides,
- достаточно для ops (включить/выключить компонент, поправить порт).

## Контракты и инварианты

- `configs/<env>/strategy.yaml` — source of truth.
- `configs/<env>/strategy_live_runner.yaml` может быть:
  - legacy полным конфигом,
  - или shim-указателем на `strategy.yaml`.
- Конфиг читается на старте процесса; hot reload не поддерживается.
- Валидация fail-fast: при неверной структуре/типах/пустых строках процесс не должен стартовать.
- Secrets не хранятся в YAML; YAML хранит только имена env переменных.
- Enable-тумблеры имеют фиксированные key-path и управляют поведением процессов.
- Порядок приоритетов: CLI override > env override > YAML > default (если применимо).

## Связанные файлы

- `docs/architecture/roadmap/milestone-3-epics-v1.md` — формулировка STR-EPIC-06.
- `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py` — существующий loader/validator `strategy_live_runner.yaml` (будет расширен для shim).
- `apps/worker/strategy_live_runner/main/main.py` — CLI флаги `--config` и `--metrics-port`.
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` — wiring live worker, метрики, fail-fast секретов.
- `apps/api/main/app.py` — место включения/выключения strategy router по `strategy.api.enabled`.
- `apps/api/wiring/modules/strategy.py` — wiring strategy API и fail-fast для Postgres DSN.
- `src/trading/platform/config/indicators_compute_numba.py` — пример env -> YAML -> default и config-path resolution.
- `apps/cli/wiring/db/clickhouse.py` — ClickHouse env settings loader (используется и в strategy worker).

Новые файлы по результату реализации STR-EPIC-06:
- `configs/dev/strategy.yaml` — dev source-of-truth.
- `configs/test/strategy.yaml` — test source-of-truth.
- `configs/prod/strategy.yaml` — prod source-of-truth.
- `docs/runbooks/strategy-live-worker.md` — runbook запуска/проверок/траблшутинга live worker.

## Как проверить

Минимальные проверки (после реализации):

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

# docs index
uv run python -m tools.docs.generate_docs_index --check
```

Smoke (концептуально, детально фиксируется в runbook):

1) Поднять стек (postgres/clickhouse/redis + api + strategy live worker).
2) Создать стратегию через API.
3) Запустить run.
4) Увидеть события/метрики в Redis Streams (`strategy.metrics.v1.user.<user_id>`, `strategy.events.v1.user.<user_id>`).
5) Получить Telegram notify (в prod) или увидеть log-only notify (dev/test).

## Риски и открытые вопросы

- Риск: временно будет существовать legacy `strategy_live_runner.yaml` и новый `strategy.yaml` — критично не допустить drift; shim-формат решает это, но требует аккуратной миграции.
- Риск: конфликт портов метрик при одновременном запуске нескольких процессов; нужен единый порт-реестр в `configs/*/strategy.yaml` и явные defaults.

Открытые вопросы:

- ID/OQ-01: Telegram login flow в v1 уже зафиксирован как **Telegram Login Widget (вариант A)**
  в `docs/architecture/identity/identity-telegram-login-user-model-v1.md`.
- ID/OQ-02: recovery для 2FA (backup codes) в документации v1 оставлен как вопрос на следующий milestone
  (см. `docs/architecture/identity/identity-2fa-totp-policy-v1.md`).
