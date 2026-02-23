# Nginx Gateway v1 -- Same-Origin UI + API routing (WEB-EPIC-02)

Документ фиксирует архитектуру WEB-EPIC-02: Nginx gateway для same-origin доставки Web UI (SSR) и JSON API, чтобы браузерный UI работал без CORS и без cross-origin cookie сложностей.

## Цель

- Дать один origin entrypoint (gateway), который обслуживает:
  - HTML UI (web upstream),
  - JSON API через префикс `/api/*` (api upstream) со strip `/api`.
- Зафиксировать локальный dev сценарий “одна команда поднять стек” для UI разработки.
- Зафиксировать DB bootstrap для dev (identity baseline SQL + alembic upgrade head).

## Контекст

- Web UI реализован как отдельный процесс `apps/web` (SSR + Jinja2 + HTMX) и ожидает:
  - browser-side вызовы identity по `/api/auth/*`,
  - server-side login gate через `GET /api/auth/current-user` (см. `apps/web/main/api_client.py`).
- API реализован как отдельный процесс `apps/api` и экспонирует identity под путями без `/api`:
  - `POST /auth/telegram/login`
  - `POST /auth/logout`
  - `GET /auth/current-user`
  (см. `src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py`).

Следствие:

- Префикс `/api` существует только на gateway уровне. Gateway обязуется стабильно и
детерминированно маппить `/api/*` -> `/*` на api upstream.

См.:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-02.
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md` — WEB-EPIC-01.
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — cookie policy и Telegram login.

## Scope

### 1) Nginx gateway routing

- `/api/*` -> proxy_pass на API upstream со strip префикса `/api`.
  - пример: `/api/auth/current-user` -> API получает `/auth/current-user`.
- `/assets/*` -> static assets из `apps/web/dist`.
- Остальные пути (`/`, `/login`, `/strategies`, ...) -> web upstream.

### 2) Docker compose: профиль `ui`

- Добавляем сервисы `api`, `web`, `gateway` в `infra/docker/docker-compose.yml`.
- Все UI сервисы находятся под `profiles: ["ui"]`, чтобы не менять дефолтный market-data стек.
- На host публикуем только gateway (например `127.0.0.1:8080:80`).
  `api` и `web` живут только внутри compose сети.

### 3) DB bootstrap (dev)

В dev режиме gateway-стек должен уметь подготовить Postgres схемы.

- Identity baseline (legacy SQL): применить `migrations/postgres/*.sql` в `IDENTITY_PG_DSN`.
  Важно: `0004_identity_exchange_keys_v2.sql` не идемпотентен и не должен выполняться повторно
  после успешного апгрейда. Нужен bootstrap runner, который:
  - проверяет текущую schema-version (например по наличию колонок v2),
  - выполняет `0003 -> 0004` только один раз на “чистой” базе,
  - либо fail-fast с понятной ошибкой, если обнаружен v1 schema с непустой таблицей.

- Alembic schema (strategy/backtest/backtest-jobs): запуск migrations runner
  `python -m apps.migrations.main --dsn "$POSTGRES_DSN"` (как в CI) с `upgrade head`.

### 4) Runbooks

Добавляем runbooks:

- запуск локального стека `web+api+gateway+postgres+clickhouse+redis` одной командой,
- DB bootstrap для dev (identity baseline + alembic),
- настройка Telegram Login Widget domain.

## Non-goals

- TLS termination / Let's Encrypt автоматизация.
- SPA и frontend build pipeline (в WEB-EPIC-02 только gateway delivery).
- Realtime (SSE/WebSocket).

## Ключевые решения

### 1) Gateway как отдельный Nginx сервис

Gateway отделён от API и web, чтобы:

- закрепить same-origin semantics,
- вынести path routing и статическую раздачу ассетов,
- не добавлять CORS/credential complexity в браузер.

Последствия:

- `/api/*` является контрактом между browser UI и инфраструктурой.
- API не знает про `/api` и не меняет router paths.

### 2) Strip `/api` реализован через `proxy_pass .../` с trailing slash

Контракт для Nginx (пример):

- `location ^~ /api/ { proxy_pass http://api:8000/; }`

Это гарантирует:

- `/api/<path>` -> `/<path>` на api upstream.

### 3) `/assets/*` отдаёт gateway (из `apps/web/dist`)

Gateway обслуживает `GET /assets/*` напрямую из filesystem слоя внутри gateway image.

Причины:

- deterministic static delivery в prod (без volume mounts),
- web upstream не занимается статикой в той же степени, что gateway.

### 4) `WEB_API_BASE_URL` у web указывает на gateway

В текущей реализации WEB-EPIC-01 server-side login gate дергает путь `/api/auth/current-user`
(см. `apps/web/main/api_client.py`). Поэтому:

- `WEB_API_BASE_URL` в web контейнере должен указывать на gateway base URL (в docker: `http://gateway`).

Это сознательно вводит один дополнительный hop для server-side проверок, но
сохраняет единый префикс `/api` как инфраструктурный контракт.

### 5) Compose `profiles: ["ui"]` для UI сервисов

UI сервисы добавляются в основной compose файл, но выключены по умолчанию.

Причина:

- не ломаем текущий прод deployment market-data стека,
- UI включается явным флагом `--profile ui`.

### 6) Dev Telegram domain: минимальная стратегия

Для Telegram Login Widget требуется домен, связанный с ботом через `@BotFather /setdomain`.

Рекомендуемая dev стратегия (самая простая без DNS):

- использовать туннель (например Cloudflare Tunnel или ngrok) на gateway port,
- установить домен туннеля в `@BotFather /setdomain` для бота.

Ограничение:

- если тот же бот уже “живёт” в проде на домене `roehub.com`, то смена домена для dev
  может временно ломать prod login widget. Митигация: завести отдельного staging/dev бота
  или сделать bot username конфигурируемым в web (отдельный эпик, не WEB-EPIC-02).

## Контракты и инварианты

- **G-001:** `/api/*` на gateway всегда проксируется в API upstream со strip `/api`.
- **G-002:** API router paths не меняются и не включают `/api` префикс.
- **G-003:** `/assets/*` обслуживается gateway из `apps/web/dist`.
- **G-004:** В dev compose на host публикуется только gateway.
- **G-005:** DB bootstrap в dev выполняется fail-fast и не должен повторно ломаться на уже
  инициализированной схеме (особенно для identity exchange keys v2).

## Связанные файлы

Docs:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-02.
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md` — WEB-EPIC-01.
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — Telegram login + JWT cookie.

Web:
- `apps/web/main/app.py` — web SSR app.
- `apps/web/main/api_client.py` — server-side current-user gate (`/api/auth/current-user`).
- `apps/web/dist/**` — статические ассеты.

API:
- `apps/api/main/app.py` — API composition root.
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py` — `/auth/*` endpoints.

Infra:
- `infra/docker/docker-compose.yml` — основной compose (будет расширен профилем `ui`).
- `infra/docker/Dockerfile.market_data` — общий python образ (должен включить `alembic/`, `alembic.ini`, `migrations/`).

Migrations:
- `migrations/postgres/*.sql` — identity baseline.
- `apps/migrations/main.py` — alembic runner.

## Как проверить

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

Manual smoke (после реализации WEB-EPIC-02):

1) `docker compose -f infra/docker/docker-compose.yml --profile ui --env-file <path_to_env> up -d --build`
2) Открыть `http://127.0.0.1:8080/login` и проверить:
   - `POST /api/auth/telegram/login` устанавливает cookie,
   - `/strategies` редиректит на `/login` при 401 и доступен после login.
3) Проверить `curl -i http://127.0.0.1:8080/api/auth/current-user`:
   - без cookie -> 401
   - с cookie -> 200.

## Риски и открытые вопросы

- Риск: identity SQL baseline v2 (0004) не идемпотентен и требует аккуратного bootstrap runner.
- Риск: dev Telegram domain требует туннеля или отдельного staging бота; иначе “bot domain invalid”.
- Риск: отсутствие TLS termination в scope (для prod нужен внешний TLS слой или ручная настройка).
