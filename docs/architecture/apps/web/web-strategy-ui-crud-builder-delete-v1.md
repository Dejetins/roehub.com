# Web UI -- Strategy UI v1 (CRUD + visual builder) + delete (WEB-EPIC-04)

Документ фиксирует архитектуру WEB-EPIC-04: Strategy UI v1 поверх существующего Strategy JSON API,
в рамках `apps/web` (SSR + Jinja2 + HTMX) с минимальным JS для управления builder state.

## Цель

- Пользователь может:
  - увидеть список своих стратегий,
  - открыть details конкретной стратегии,
  - создать новую стратегию через визуальный builder (без JSON textarea),
  - клонировать стратегию,
  - soft-delete (архивировать) стратегию.

## Контекст

- Web слой: `apps/web` (SSR + Jinja2 + HTMX) уже реализует login gate (WEB-EPIC-01).
- Same-origin доставка обеспечивается gateway (WEB-EPIC-02): browser вызывает JSON API по `/api/...`.
- Strategy API уже реализован и является source-of-truth:
  - `GET /strategies` (owner-only, deterministic ordering)
  - `GET /strategies/{strategy_id}`
  - `POST /strategies` (create immutable)
  - `POST /strategies/clone`
  - `DELETE /strategies/{strategy_id}` (soft delete)
  См. `apps/api/routes/strategies.py`.

- Builder использует reference endpoints:
  - markets/instruments: `/api/market-data/*` (WEB-EPIC-03)
  - indicator registry: `GET /api/indicators`

См.:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-04.
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md` — базовые принципы web.
- `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md` — Strategy API контракт.
- `docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md` — StrategySpec v1 инварианты.
- `docs/architecture/market_data/market-data-reference-api-v1.md` — reference markets/instruments API.

## Scope

### 1) URL structure (фикс)

- `GET /strategies` — list page.
- `GET /strategies/new` — create builder page.
- `GET /strategies/{strategy_id}` — details page.

### 2) Data access pattern (фикс)

- UI вызывает JSON API напрямую из браузера через `/api/...` (cookie auth):
  - `fetch(..., { credentials: 'include' })` или HTMX requests.
- Web SSR страницы не делают server-side create/clone/delete; web отвечает только HTML.

Причина:
- сохраняем “API as source of truth”,
- исключаем дублирование DTO/валидаций на web-слое,
- упрощаем gateway контракт.

### 3) List strategies page

- UI грузит `GET /api/strategies`.
- Таблица строится из response.
- Фильтры (symbol/market_type/timeframe) — client-side (JS) по уже загруженным данным.

Actions:

- `View` -> переход на `/strategies/{id}`.
- `Clone` -> `POST /api/strategies/clone` с payload `{ "source_strategy_id": "..." }`.
- `Delete` -> confirm -> `DELETE /api/strategies/{id}`.

### 4) Strategy details page

- UI грузит `GET /api/strategies/{id}`.
- Показывает:
  - header: name, instrument (market_id/symbol), market_type, timeframe, created_at.
  - список indicator blocks (id + params + inputs).
  - `<details>` блок с raw `spec` JSON (debug/копирование).

Actions:

- `Clone` (как в list).
- `Delete` (confirm).

### 5) Create builder page (visual, без JSON textarea)

Builder собирает StrategySpec v1 payload и вызывает `POST /api/strategies`.

Inputs:

- Instrument:
  - выбрать `market_id` из `GET /api/market-data/markets`.
  - выбрать `symbol` через typeahead `GET /api/market-data/instruments?market_id=&q=&limit=`.
- Timeframe:
  - выбрать из supported `Timeframe` codes (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`).
  - UI валидирует значение, API/домен валидируют повторно.
- Indicators:
  - загрузить registry `GET /api/indicators`.
  - пользователь добавляет indicator blocks и заполняет scalar params/inputs.
  - порядок индикаторов сохраняется (влияет на детерминированный `Strategy.name`).

State management:

- Используем минимальный JS для:
  - add/remove/reorder indicator blocks,
  - хранения текущего builder state в памяти,
  - сборки JSON payload на submit.

Никакого "JSON textarea" пользователю не показываем.

### 6) Indicator payload shape (фикс)

`StrategySpecV1.indicators[]` entry MUST follow the canonical mapping:

- `{"id": "<indicator_id>", "inputs": {...}, "params": {...}}`

Это совместимо с:

- `StrategySpecV1` validation (id|kind|name),
- backtest saved-mode ACL reader (`strategy_repository_reader.py`) и сохранением variants.

### 7) Clone/delete UX policies (фикс)

- Clone v1: без overrides UI (только `source_strategy_id`).
- Delete v1: всегда подтверждение (modal/confirm), т.к. это soft-delete.

## Non-goals

- Run/stop UI для live runner.
- Realtime streams UI.
- Strategy edit/update (Strategy immutable; изменения только через create/clone).

## DoD

- Можно создать стратегию в UI (builder) и увидеть её в списке.
- Clone работает и создаёт новую стратегию.
- Delete работает (soft-delete) и стратегия исчезает из list.
- Все вызовы к backend идут через `/api/...` с `credentials: 'include'`.

## Контракты и инварианты

- UI не вводит `name` стратегии: name генерируется доменом детерминированно.
- UI сохраняет порядок индикаторов, чтобы name был воспроизводим.
- Instrument identity: `instrument_id = {market_id, symbol}`.
- `instrument_key` строится из `market_code` + `symbol` (например `binance:spot:BTCUSDT`).
- `market_type` для StrategySpec v1 берём из выбранного market (`ref_market.market_type`).

## Связанные файлы

Web:
- `apps/web/main/app.py` — routing/login gate.
- `apps/web/templates/**` — templates.
- `apps/web/dist/**` — assets.

API:
- `apps/api/routes/strategies.py` — Strategy endpoints.
- `apps/api/routes/indicators.py` — indicators registry endpoint.
- `apps/api/routes/market_data_reference.py` — reference endpoints.

Domain:
- `src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py` — StrategySpec v1 validation.
- `src/trading/shared_kernel/primitives/timeframe.py` — supported timeframe codes.

## Как проверить

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Manual smoke (через gateway):

1) Открыть `http://127.0.0.1:8080/strategies`.
2) Создать стратегию через `/strategies/new`.
3) Убедиться что стратегия появилась в списке.
4) Нажать Clone и убедиться что появилась новая стратегия.
5) Нажать Delete (confirm) и убедиться что стратегия пропала из списка.

## Риски / открытые вопросы

- При большом числе стратегий client-side фильтрация может стать тяжелой; v1 допускает, позже можно добавить server-side фильтры в API.
- Builder UI требует аккуратной валидации типов params/inputs по данным `/api/indicators`.
