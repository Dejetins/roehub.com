# Web UI -- Backtest sync UI v1 (POST /backtests) + preflight + save variant (WEB-EPIC-05)

Документ фиксирует архитектуру WEB-EPIC-05: UI для синхронного (small-run) backtest запуска,
обязательный preflight в template-mode и сохранение выбранного варианта как Strategy через переход
в strategy builder с prefill.

## Цель

- Пользователь может:
  - запустить sync backtest в template-mode (ad-hoc grid) и в saved-mode (по `strategy_id`),
  - увидеть top-K результаты (таблица + `report_table_md` + trades только для `top_trades_n`),
  - сохранить выбранный вариант как Strategy, открыв `/strategies/new` с предзаполненными полями.

## Контекст

- Web слой: `apps/web` (SSR + Jinja2 + HTMX) с login gate (WEB-EPIC-01).
- Same-origin gateway: browser вызывает JSON API по `/api/*` (WEB-EPIC-02).
- Sync backtest API (source-of-truth): `POST /backtests`.
  Контракт request/response и deterministic ordering описаны в:
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - реализация: `apps/api/routes/backtests.py`, `apps/api/dto/backtests.py`

- Preflight endpoint для template-mode: `POST /indicators/estimate`.
  Реализация: `apps/api/routes/indicators.py`, request DTO `apps/api/dto/indicators.py`.

- Strategy API (сохранение): `POST /strategies`.
  Реализация: `apps/api/routes/strategies.py`.

## Scope

### 1) URL structure (фикс)

- `GET /backtests` — одна protected страница, объединяющая:
  - template-mode builder (grid + preflight + run sync)
  - saved-mode run (strategy select + overrides + run sync)

История запусков не сохраняется. Результаты существуют только в состоянии страницы.

### 2) Browser-side integration pattern (фикс)

- Все backtest запросы выполняются из браузера напрямую в JSON API через `/api/...`:
  - `GET /api/backtests/runtime-defaults`
  - `POST /api/indicators/estimate`
  - `POST /api/backtests`
  - (в дальнейшем) `POST /api/backtests/jobs` для async режима
  - `POST /api/strategies` для сохранения выбранного варианта

Все запросы должны использовать cookie auth:

- `credentials: 'include'`

### 3) Template-mode builder

UI собирает `POST /api/backtests` request с `template` блоком:

- instrument selection: `market_id`/`symbol` через `/api/market-data/*`
- timeframe
- `indicator_grids[]` (compute axes)
- optional advanced: execution/risk_grid/signal_grids/direction/sizing/top_k/preselect/top_trades_n/warmup_bars

Обязательный preflight:

- перед `Run sync` UI вызывает `POST /api/indicators/estimate`.
- пока preflight не выполнен успешно, запуск sync backtest запрещён.

### 4) Saved-mode run

UI выбирает стратегию из `GET /api/strategies` и запускает `POST /api/backtests` с:

- `strategy_id`
- `overrides?` (advanced блок)

Preflight в saved-mode не применяется (endpoint требует indicator grids).

### 5) Result view

UI отображает:

- metadata запуска: mode, instrument_id, timeframe, hashes (`spec_hash` или `grid_request_hash`, `engine_params_hash`).
- таблицу variants (deterministic order, как в API ответе):
  - `total_return_pct`
  - `variant_key`
  - `indicator_variant_key`
  - `report` summary
- `report.table_md` рендерится как markdown -> HTML (см. решение 2).
- trades показываются только для первых `top_trades_n` вариантов и только если API вернул `report.trades`.

### 6) Save variant -> Strategy builder prefill

На каждом варианте есть действие `Save as Strategy`.

Поведение:

1) UI строит prefill payload для strategy builder из backtest variant payload:
   - `instrument_id` + `timeframe` берём из backtest response.
   - `indicators` строим из `variant.payload.indicator_selections[]`:
     - `{"id": indicator_id, "inputs": inputs, "params": params}`
   - `instrument_key`/`market_type` UI берёт из контекста запуска:
     - template-mode: из выбранного market (`market_code`, `market_type`) + `symbol`
     - saved-mode: из выбранной strategy (`spec.instrument_key`, `spec.market_type`)

2) UI сохраняет prefill payload в `sessionStorage` и редиректит на:
   - `/strategies/new?prefill=<prefill_id>`

3) Strategy builder (WEB-EPIC-04) читает `prefill_id`, загружает payload из `sessionStorage` и
   предзаполняет поля.

## Non-goals

- История sync backtest запусков (нет persisted run_id/results).
- Async jobs UI (WEB-EPIC-06).
- Полноценный UI для signal grids catalog (если потребуется, отдельный эпик).

## Ключевые решения

### 1) Одна protected страница `/backtests`

Разделяем template-mode и saved-mode внутри одной страницы.

Последствия:
- проще навигация и меньше маршрутов;
- результаты хранятся в состоянии страницы и теряются при reload.

### 2) `report.table_md` рендерится как markdown -> HTML (с sanitization)

UI рендерит `report.table_md` как HTML, но обязательно с защитой от XSS:

- markdown renderer (client-side) с отключенным raw HTML,
- обязательная sanitization (allowlist tags/attributes).

Причина:
- `table_md` генерируется backend-ом в формате markdown, и UI должен показывать это как таблицу.

Последствия:
- добавляется pinned JS зависимость на markdown renderer и sanitizer (через CDN).

### 3) Preflight обязателен для template-mode (POST /indicators/estimate)

UI должен выполнить preflight до запуска, чтобы:

- заранее показать `total_variants` и `estimated_memory_bytes`,
- предотвратить заведомо невозможные вычисления по guards.

Важно:
- preflight оценивает compute memory/variants по indicator grids и risk axes;
  сигнал-параметры (signal grids) в этом endpoint не учитываются.

### 4) Save variant реализован как переход в strategy builder (prefill), а не прямой `POST /strategies`

Кнопка `Save as Strategy` открывает `/strategies/new` с предзаполнением.

Причины:
- пользователь может отредактировать/дополнить strategy перед сохранением;
- переиспользуем один builder и один UX.

### 5) Prefill transport через `sessionStorage`

Prefill payload переносится между страницами через `sessionStorage` по ключу `prefill_id`.

Причины:
- не раздуваем URL (payload может быть больше query string лимитов),
- не кладём JSON payload в логи proxy/gateway,
- простая реализация без новых backend endpoints.

## Контракты и инварианты

- Все JSON API вызовы из UI идут на `/api/...` с `credentials: 'include'`.
- `POST /api/backtests` mode selection: `strategy_id xor template`.
- Preflight обязателен только для template-mode.
- `limit`/guards semantics обрабатываются и показываются пользователю как 422 (deterministic payload).
- `Save as Strategy` использует индикаторный payload shape:
  - `{"id": "<indicator_id>", "inputs": {...}, "params": {...}}`.

## Связанные файлы

Docs:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-05.
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md` — sync backtest API contract.
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md` — runtime defaults
  endpoint contract for browser prefill.
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` — guards semantics.
- `docs/architecture/apps/web/web-strategy-ui-crud-builder-delete-v1.md` — strategy builder v1.

API:
- `apps/api/routes/backtests.py` — `POST /backtests`.
- `apps/api/dto/backtests.py` — request/response DTO.
- `apps/api/routes/indicators.py` — `POST /indicators/estimate`.
- `apps/api/routes/strategies.py` — `POST /strategies`.

Web:
- `apps/web/main/app.py` — `/backtests` protected route.
- `apps/web/dist/**` — web JS helpers (preflight/run/save/prefill).

## Как проверить

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Manual smoke (через gateway):

1) Открыть `/backtests` после логина.
2) Template-mode: выбрать market/symbol/timeframe, добавить индикаторы.
3) Нажать preflight (estimate), убедиться что run разрешается только после успешного preflight.
4) Запустить sync backtest, увидеть top-K.
5) Нажать `Save as Strategy` на варианте и убедиться что `/strategies/new` предзаполнен.

## Риски и открытые вопросы

- Риск: markdown rendering без sanitization может привести к XSS. Решение: обязательная sanitization.
- Риск: отсутствие signal params defaults в `configs/<env>/indicators.yaml` может приводить к 422 для некоторых индикаторов.
  Митигация v1: UI ограничивает/подсказывает индикаторы, или добавляются server-side defaults в configs (отдельный change).
