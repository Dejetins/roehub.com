# Milestone 6 -- EPIC map (v1)

EPIC map для Milestone 6: Web UI v1 (Backtest + Jobs + Strategy + Auth).

Milestone 6 добавляет браузерный UI, который работает через same-origin gateway (Nginx) и
использует уже реализованные JSON API контракты (identity/strategy/backtest/backtest-jobs),
без CORS и без realtime (SSE/WebSocket).

Референс по формату: `docs/architecture/roadmap/milestone-5-epics-v1.md`.

---

## Контекст и новые вводные (зафиксировано)

### Same-origin gateway semantics (фикс)

- Один origin: `https://roehub.com`.
- Nginx gateway маршрутизирует:
  - `/api/*` -> Roehub API upstream, с strip префикса `/api`.
  - все остальные пути -> web upstream (HTML), статические ассеты отдаёт gateway из `apps/web/dist`.
- UI **всегда** вызывает JSON API только по `/api/...`.
  HTML страницы UI обслуживаются web UI роутами без `/api`.

### Web UI architecture (фикс)

- `apps/web` = Python SSR + Jinja2 + HTMX (без React/SPA).
- `apps/web` — HTML facade поверх JSON API:
  - web app не wires `src/trading/**` use-cases напрямую,
  - web app вызывает JSON API по HTTP (internal base URL) и рендерит HTML (страницы и partials).

### Auth (фикс)

- Identity остаётся в API:
  - `POST /auth/telegram/login` (JWT HttpOnly cookie)
  - `POST /auth/logout`
  - `GET /auth/current-user`
- Web login page использует Telegram Login Widget и отправляет JSON payload в `POST /api/auth/telegram/login`.
- Protected pages (web) редиректят на `/login`, если `GET /api/auth/current-user` возвращает 401.
- Cookie policy v1: `SameSite=lax`, `HttpOnly=true`, `Secure=true` в prod (CSRF откладываем).

### Strategy persistence + “Save variant as Strategy” (фикс)

- Стратегии сохраняются через Strategy API в Postgres:
  - таблицы: `strategy_strategies`, `strategy_runs`, `strategy_events` (см. `alembic/versions/20260215_0001_strategy_storage_v1.py`).
  - endpoints: `POST/GET/DELETE /strategies*`, `POST /strategies/clone`.
- Backtest использует ACL reader `StrategyRepositoryBacktestStrategyReader` для saved-mode backtest:
  - читает Strategy через `StrategyRepository`,
  - мапит StrategySpec -> Backtest template snapshot.

Indicator payload shape for StrategySpec v1:

- В `StrategySpecV1.indicators[]` используем каноничный entry:
  - `{"id": "<indicator_id>", "inputs": {...}, "params": {...}}`
- Почему:
  - `StrategySpecV1` валидирует наличие одного из `name|kind|id` (не только `indicator_id`),
  - `StrategyRepositoryBacktestStrategyReader` умеет читать `indicator_id|id|kind|name` и ожидает `inputs/params`.

Save-from-backtest semantics v1:

- UI берёт `variant.payload.indicator_selections[]` из backtest/backtest-jobs responses и создаёт новую StrategySpec v1:
  - `id = indicator_selections[i].indicator_id`
  - `inputs = indicator_selections[i].inputs`
  - `params = indicator_selections[i].params`
- `risk/execution/direction/sizing` НЕ сохраняем в StrategySpec v1; это остаётся параметрами backtest.

### Backtest UI scope (фикс)

- Sync backtest UI: `POST /api/backtests` (small-run) + рендер результатов:
  - top-K variants,
  - `report.table_md` всегда (для top-K),
  - trades только для `top_trades_n`.
- Jobs UI: `POST/GET /api/backtests/jobs*`:
  - create/list/status/top/cancel,
  - polling прогресса и best-so-far top во время `running`,
  - UX учитывает reclaim v1: возможны сброс `stage/progress` и временно stale `/top` snapshot до первой перезаписи.

### Preflight estimate in UI (фикс)

- Перед запуском template-mode backtest UI делает preflight через `POST /api/indicators/estimate`:
  - показывает `total_variants` и `estimated_memory_bytes`,
  - блокирует очевидно невозможные запросы (guards).

### Market-data reference endpoints for UI (фикс)

UI выбирает инструменты из ClickHouse reference tables:

- `market_data.ref_market`
- `market_data.ref_instruments`

Добавляем auth-only API endpoints:

- `GET /market-data/markets`
  - возвращает только enabled рынки,
  - детерминированная сортировка: `market_id ASC`.
- `GET /market-data/instruments?market_id=&q=&limit=`
  - фильтр: только `status='ENABLED' AND is_tradable=1`,
  - `q` = prefix поиск по `symbol`,
  - детерминированная сортировка: `symbol ASC`.

### Postgres migrations bootstrap for UI environments (фикс)

- В Milestone 6 фиксируем “одна команда поднять dev stack” с корректной подготовкой Postgres схем:
  - Identity baseline: применить SQL baseline миграции из `migrations/postgres/*.sql` в `IDENTITY_PG_DSN`.
  - Alembic schema: применить миграции strategy/backtest/backtest-jobs через migrations runner
    `apps/migrations/main.py` в `POSTGRES_DSN` (в простом dev может указывать на тот же инстанс, что и `STRATEGY_PG_DSN`).

---

## Принцип декомпозиции Milestone 6

Milestone 6 делится на 5 логических частей:

1) Gateway/Delivery: Nginx same-origin routing + статические ассеты.
2) Web app: SSR skeleton + layout + auth UX + internal API client.
3) API additions: market-data reference read endpoints (ClickHouse -> API).
4) UI features: strategy CRUD/builder, backtest sync UI, backtest jobs UI.
5) Tests + runbooks.

---

## Порядок внедрения (рекомендуемый)

1) WEB-EPIC-01 -- Web UI skeleton + auth integration.
2) WEB-EPIC-02 -- Nginx gateway (same-origin) + local/prod runbooks.
3) WEB-EPIC-03 -- Market-data reference API endpoints (markets/instruments).
4) WEB-EPIC-04 -- Strategy UI v1 (CRUD + builder) + delete.
5) WEB-EPIC-05 -- Backtest sync UI v1 + preflight estimate + save variant -> strategy.
6) WEB-EPIC-06 -- Backtest jobs UI v1 (create/list/status/top/cancel).
7) WEB-EPIC-07 -- Tests + docs index.

---

## EPIC'и Milestone 6

### WEB-EPIC-01 -- Web UI skeleton v1 (SSR + HTMX) + auth UX

**Цель:** завести `apps/web` как отдельный web upstream, который рендерит HTML страницы и
использует JSON API через `/api/...`, с обязательным login gate.

**Scope:**

- `apps/web`:
  - app factory + router (страницы `/`, `/login`, `/logout`, `/strategies`, `/backtests`, `/backtests/jobs`),
  - базовый layout (navigation, user badge, error banner),
  - HTMX partials для таблиц/форм.
- Auth UX:
  - `/login`: Telegram widget + JS, который делает `POST /api/auth/telegram/login` (JSON) и затем redirect.
  - protected pages: проверка `GET /api/auth/current-user` и redirect на `/login` при 401.
  - `/logout`: вызывает `POST /api/auth/logout` и редиректит на `/login`.
- Internal API client:
  - базовый `api_base_url` из env (например `WEB_API_BASE_URL`),
  - форвардинг `Cookie` заголовка из web request в API request.

**Non-goals:**

- SPA framework.
- UI для 2FA и exchange keys.
- Realtime streams UI.

**DoD:**

- Web app запускается отдельно от API.
- Login/logout/current-user flow работает end-to-end через `/api/*`.
- Protected pages не показывают данные без auth (redirect).

**Paths:**

- `apps/web/**`
- `docs/runbooks/*` (web runbook добавляется в WEB-EPIC-02)

---

### WEB-EPIC-02 -- Nginx gateway v1 (same-origin) + runbooks

**Цель:** обеспечить same-origin доставку UI+API через Nginx gateway, чтобы UI мог работать
без CORS и без cross-origin cookie сложностей.

**Scope:**

- Nginx конфиг:
  - `/api/*` -> proxy_pass на API upstream с strip `/api`.
  - `/assets/*` (или аналогичный префикс) -> static from `apps/web/dist`.
  - остальное -> web upstream.
- Docker compose:
  - добавить сервисы `web` и `gateway` (nginx),
  - обеспечить локальный dev сценарий “одна команда поднять UI+API+gateway+PG+CH+Redis”.
- DB bootstrap (dev):
  - применить `migrations/postgres/*.sql` в `IDENTITY_PG_DSN` (identity tables),
  - запустить `apps/migrations/main.py` для Alembic `upgrade head` в `POSTGRES_DSN`.
- Runbooks:
  - запуск `web+api+gateway` в dev,
  - настройка Telegram widget allowed domains для dev/prod.

**Non-goals:**

- TLS termination/Let’s Encrypt автоматизация (можно отдельным документом/эпиком).

**DoD:**

- Один origin сценарий работает локально и в целевом прод-профиле.
- `/api/*` корректно маршрутизируется на API без изменения API router paths.

**Paths:**

- `infra/docker/**`
- `docs/runbooks/**`

---

### WEB-EPIC-03 -- Market-data reference API v1 (markets + instruments)

**Цель:** дать UI список рынков и инструментов для выбора (dropdown/search) на основании
ClickHouse reference tables.

**Scope:**

- Market-data context:
  - добавить application ports/use-cases для чтения `ref_market` и поиска по `ref_instruments`.
  - ClickHouse adapters: явный SQL, deterministic ordering.
- API:
  - новый router `market-data reference`:
    - `GET /market-data/markets`
    - `GET /market-data/instruments?market_id=&q=&limit=`
  - auth-only через текущий `current_user_dependency`.
- Unit tests:
  - deterministic ordering,
  - фильтры enabled/tradable,
  - поведение `q`/`limit`.

**Non-goals:**

- Enrich ref_instruments из биржевых REST (это отдельный market_data epic).
- Public unauth endpoints.

**DoD:**

- UI может загрузить список markets и сделать поиск symbols без прямого доступа к ClickHouse.
- Запросы детерминированы и покрыты unit тестами.

**Paths:**

- `src/trading/contexts/market_data/application/**`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/**`
- `apps/api/routes/**`
- `apps/api/wiring/modules/**`
- `tests/unit/**`

---

### WEB-EPIC-04 -- Strategy UI v1 (CRUD + visual builder) + delete

**Цель:** пользователь управляет своими стратегиями через UI: list/get/create/clone/delete,
а создание происходит через визуальный builder (без JSON textarea).

**Scope:**

- Pages/components:
  - list strategies (таблица + фильтры по symbol/market_type/timeframe по данным response),
  - strategy details page,
  - create builder:
    - выбрать market/symbol/timeframe (через `/api/market-data/*`),
    - выбрать индикаторы и их scalar params/inputs (через `/api/indicators`),
    - собрать StrategySpec v1 payload и вызвать `POST /api/strategies`.
  - clone flow: `POST /api/strategies/clone`.
  - delete flow: `DELETE /api/strategies/{id}`.

**Non-goals:**

- run/stop UI для live runner.
- realtime streams UI.

**DoD:**

- Можно создать стратегию в UI (builder) и увидеть её в списке.
- Clone и delete работают и отражаются в UI.

**Paths:**

- `apps/web/**`

---

### WEB-EPIC-05 -- Backtest sync UI v1 (POST /backtests) + preflight + save variant

**Цель:** UI позволяет запускать sync small-run backtest (template или saved mode),
показывает top-K результаты и позволяет сохранить вариант как Strategy.

**Scope:**

- Backtest form:
  - template-mode builder (grid axes) + preflight `POST /api/indicators/estimate`.
  - saved-mode: запуск backtest по `strategy_id` + overrides.
  - выбор режима “Run sync” vs “Run as job” (job путь покрывается WEB-EPIC-06).
- Result view:
  - таблица top-K (deterministic order),
  - `report_table_md` рендерится как текст/markdown,
  - trades показываются только для первых `top_trades_n`.
- Save variant:
  - кнопка “Save as Strategy” на выбранном варианте:
    - строит StrategySpec v1 payload из `variant.payload.indicator_selections`,
    - вызывает `POST /api/strategies`.

**Non-goals:**

- История sync backtest запусков (результаты не сохраняются в БД для sync режима).

**DoD:**

- Пользователь запускает sync backtest, видит top-K и может сохранить вариант как Strategy.
- Preflight estimate используется до запуска template-mode.

**Paths:**

- `apps/web/**`

---

### WEB-EPIC-06 -- Backtest Jobs UI v1 (async)

**Цель:** UI позволяет запускать async backtest jobs, видеть прогресс, best-so-far top,
финальные результаты и отменять job.

**Scope:**

- Create job UI:
  - template-mode job create,
  - saved-mode job create,
  - reuse form parts from WEB-EPIC-05.
- Jobs list UI:
  - keyset pagination (`cursor` opaque `base64url(json)`),
  - фильтр по `state`.
- Job status UI:
  - state/stage/progress, timestamps, hashes,
  - error payload для `failed`.
- Best-so-far top UI:
  - polling `GET /api/backtests/jobs/{job_id}/top`,
  - state-dependent policy:
    - `report_table_md` и trades показывать только для `succeeded`.
- Cancel UI:
  - `POST /api/backtests/jobs/{job_id}/cancel` (idempotent).

**Non-goals:**

- realtime push (SSE/WebSocket).

**DoD:**

- Пользователь запускает job, видит прогресс и top результаты во время `running`,
  и видит финальный отчёт после `succeeded`.
- Cancel работает и отражается в UI.

**Paths:**

- `apps/web/**`

---

### WEB-EPIC-07 -- Tests + docs index

**Цель:** закрепить Milestone 6 контракт тестами (минимально) и обновить docs index.

**Scope:**

- Unit tests:
  - API reference endpoints (`/market-data/*`).
  - smoke-level web routes (login gate, basic pages render) без внешних сервисов.
- Docs index:
  - `python -m tools.docs.generate_docs_index`.

**Non-goals:**

- Полные e2e UI тесты (Playwright) в v1.

**DoD:**

- `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` проходят.
- Документация UI/gateway/runbooks добавлена, индекс docs обновлен.

**Paths:**

- `tests/unit/**`
- `docs/architecture/README.md`
