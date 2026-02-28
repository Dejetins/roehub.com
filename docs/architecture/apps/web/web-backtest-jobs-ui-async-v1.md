# Web UI -- Backtest Jobs UI v1 (async) (WEB-EPIC-06)

Документ фиксирует архитектуру WEB-EPIC-06: UI для async backtest jobs (create/status/top/list/cancel)
в `apps/web` с polling (без SSE/WebSocket), deterministic pagination cursor и lazy report policy
для `/top` (summary-only rows + `Load report` on-demand через `variant-report`).

## Цель

- Пользователь может:
  - создать backtest job (template-mode или saved-mode),
  - видеть прогресс и best-so-far top результаты во время исполнения,
  - получить финальные результаты после `succeeded`,
  - отменить job (best-effort cancel),
  - сохранить выбранный вариант как Strategy через существующий builder prefill.

## Контекст

- Jobs API уже реализован и является source-of-truth:
  - `POST /backtests/jobs`
  - `GET /backtests/jobs/{job_id}`
  - `GET /backtests/jobs/{job_id}/top?limit=`
  - `GET /backtests/jobs?state=&limit=&cursor=`
  - `POST /backtests/jobs/{job_id}/cancel`
  См. `docs/architecture/backtest/backtest-jobs-api-v1.md`, `apps/api/routes/backtest_jobs.py`.

- UI работает same-origin через gateway (WEB-EPIC-02): browser calls `/api/*`.
- Web UI уже имеет:
  - sync backtest UI `/backtests` (WEB-EPIC-05) с формой template/saved mode.
  - Strategy builder prefill через `sessionStorage` + `/strategies/new?prefill=...`.

Особенности jobs семантики, влияющие на UI:

- Progress/stage:
  - stage literals: `stage_a|stage_b|finalizing`
  - progress units: `processed_units/total_units` зависят от stage.
- Reclaim v1 = restart attempt:
  - progress может сброситься (`stage_a`, `processed_units=0`),
  - `/top` snapshot может быть временно stale (пока новый attempt не перезапишет snapshot).
  См. `docs/architecture/backtest/backtest-job-runner-worker-v1.md`.

## Scope

### 1) URL structure (фикс)

- `GET /backtests/jobs` — protected jobs list page.
- `GET /backtests/jobs/{job_id}` — protected job details page.

### 2) Job creation UX (фикс)

Job creation происходит на `/backtests` (reuse формы WEB-EPIC-05):

- UI toggle "Run as job" вызывает browser-side `POST /api/backtests/jobs`.
- На успех: redirect на `/backtests/jobs/{job_id}`.

`/backtests/jobs` page содержит ссылку/кнопку "Create job" -> `/backtests`.

### 3) Jobs list UI

- Browser-side loading:
  - `GET /api/backtests/jobs?state=&limit=&cursor=`
- Функции:
  - фильтр по `state` (queued|running|succeeded|failed|cancelled)
  - keyset pagination:
    - cursor opaque `base64url(json)`
    - UI хранит cursor как opaque string и не парсит.

### 4) Job details UI

Секция Status:

- `GET /api/backtests/jobs/{job_id}`:
  - state/stage
  - progress counters
  - timestamps
  - hashes (`request_hash`, `engine_params_hash`, `backtest_runtime_config_hash`, `spec_hash?`)
  - для `failed`: `last_error` + `last_error_json`

Секция Top (best-so-far):

- polling `GET /api/backtests/jobs/{job_id}/top?limit=...`
- default limit: `50` (UI всегда передает limit явно)
- `/top` возвращает только ranking summary rows (`rank`, `variant_key`, `indicator_variant_key`,
  `variant_index`, `total_return_pct`, `payload`) и `report_context`.
- Для выбранной строки UI показывает действие `Load report` и запрашивает
  `POST /api/backtests/variant-report`.
- Загруженный report (`rows/table_md/trades`) кэшируется в браузере по `variant_key`.

Секция Cancel:

- `POST /api/backtests/jobs/{job_id}/cancel` idempotent
- UI обновляет status snapshot после cancel.

### 5) Polling policy (фикс)

- Пока job active (`queued`/`running`):
  - status poll: каждые 2s
  - top poll: каждые 3s (`limit=50`)
- После terminal (`succeeded`/`failed`/`cancelled`):
  - один финальный refresh status/top
  - stop polling.

### 6) Save variant from jobs results (фикс)

На job details в top-таблице добавляем действие `Save as Strategy`:

- строим payload из row `payload.indicator_selections[]` (shape как в sync backtest):
  - `{"id": indicator_id, "inputs": inputs, "params": params}`
- переносим в strategy builder через `sessionStorage` + redirect:
  - `/strategies/new?prefill=...`

Instrument context:

- Для jobs `/top` payload `payload` содержит только variant конфиг, но не market_code.
  Поэтому UI использует стратегию v1:
  - в template-mode job create UI сохраняет выбранный `market_code/market_type/symbol/timeframe`
    в local state и переносит как часть prefill payload.
  - в saved-mode job create UI использует выбранную strategy spec (`instrument_key`, `market_type`).

### 7) Jobs disabled UX (фикс)

Если `backtest.jobs.enabled=false`, endpoints `/backtests/jobs*` не монтируются и UI
может получить 404.

UI policy:

- если jobs create/list/status/top/cancel возвращают 404:
  - показываем "Jobs disabled by config"
  - на `/backtests` выключаем "Run as job"
  - на `/backtests/jobs` показываем explanation + ссылку на `/backtests`.

## Non-goals

- realtime push (SSE/WebSocket).
- Persisted UI state/history beyond API.

## DoD

- Пользователь запускает job из `/backtests`, видит progress/top во время `running`.
- На details можно загрузить report по кнопке `Load report` для конкретной строки.
- Cancel работает и отражается в UI.
- Можно сохранить вариант как Strategy (через builder prefill).

## Ключевые решения

### 1) Два маршрута: list + details

- `/backtests/jobs` = list + pagination/filter
- `/backtests/jobs/{job_id}` = details + polling

Причины:
- проще навигация и shareable deep-link.

### 2) Job creation reuse из `/backtests`

Создание job reuse-ит existing форму WEB-EPIC-05.

Причины:
- избегаем дублирования сложного template/saved request envelope builder.

### 3) Polling вместо realtime

Polling фиксирован как v1 механизм, т.к. SSE/WebSocket не входят в scope.

### 4) UI использует lazy report policy `/top`

UI всегда работает с summary `/top` и загружает report/trades только по explicit `Load report`.

### 5) Jobs toggle маппится на UX "disabled"

Вместо generic 404, UI показывает понятное сообщение.

## Контракты и инварианты

- Все browser-side вызовы используют `/api/*` и `credentials: 'include'`.
- Cursor opaque: UI не декодирует `cursor`.
- Default `/top` limit = 50 и всегда передаётся явно.
- Polling stops on terminal state.
- `Save as Strategy` использует canonical indicator payload shape.

## Связанные файлы

Docs:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-06.
- `docs/architecture/backtest/backtest-jobs-api-v1.md` — jobs endpoints contract.
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md` — stage/progress/reclaim semantics.
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md` — backtests form reuse.

Web:
- `apps/web/main/app.py` — web routes.
- `apps/web/templates/**` — jobs pages.
- `apps/web/dist/**` — jobs UI JS.

API:
- `apps/api/routes/backtest_jobs.py` — jobs router.
- `apps/api/dto/backtest_jobs.py` — jobs DTO/cursor.

## Как проверить

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Manual smoke (через gateway):

1) `/backtests` -> выбрать "Run as job" -> создать job -> redirect в `/backtests/jobs/{job_id}`.
2) На details: проверить polling status/top, прогресс и best-so-far.
3) Для выбранной строки нажать `Load report` и проверить `rows/table_md/trades`.
4) Отменить job и проверить state transition.
5) Нажать `Save as Strategy` и убедиться в prefill `/strategies/new`.

## Риски и открытые вопросы

- Polling может создавать нагрузку при большом числе открытых вкладок; v1 принимает этот риск.
- При reclaim attempt UI может видеть сброс прогресса и stale `/top` snapshot — это ожидаемое v1 поведение.
