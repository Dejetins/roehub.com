# Web UI -- Backtest History and Variant Detail v2

Документ фиксирует web contract для R9-02/R9-03 flow поверх persisted `runs` vocabulary.
После R9-03 public UI использует history-first навигацию, summary-only run page и отдельную
lazy detail page для одного persisted variant.

## Status

- Status: active delivered contract after R9-03 history/summary/detail rollout.
- Depends on:
  - `docs/architecture/backtest/backtest-runs-history-v2.md`
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - `docs/architecture/roadmap/base_refactor_plan.md`
- Compatibility note:
  - legacy `POST /api/backtests/variant-report` остаётся compatibility path;
  - preferred public detail flow использует
    `POST /api/backtests/runs/{run_id}/variant-report`.
  - launch and status UX treat `background_auto` as the canonical background path for new queued
    runs;
  - if a persisted row still shows `background_manual_legacy`, the UI treats it as a
    `compatibility-only` literal for already stored runs rather than a modern selectable path.
  - after R9-03 launch/status/detail navigation is runs-first:
    - `/backtests/history`
    - `/backtests/runs/{run_id}`
    - `/backtests/runs/{run_id}/variants/{variant_key}`
  - legacy `/backtests/jobs*` remains compatibility surface during migration and may still expose
    legacy report/cancel tooling.
- Summary-only note:
  - `/backtests` launch page keeps sync results summary-only and deep-links selected rows into
    `/backtests/runs/{run_id}/variants/{variant_key}` for on-demand detail;
  - `/backtests/runs/{run_id}` stays summary-only and never depends on persisted
    `report_table_md` or `trades_json`;
  - one-variant detail lives only on `/backtests/runs/{run_id}/variants/{variant_key}`.
- A2 additive progress note:
  - history and summary pages continue polling `/api/backtests/runs*`, but now render a real
    `0..100%` progress bar from additive status fields instead of inferring percent from raw
    counters only;
  - active run summary page shows:
    - current `stage`
    - `progress_percent`
    - `eta_seconds` when non-null
    - `execution_profile_mode`;
  - launch page still does not show true sync-inline in-flight progress before response return,
    because this EPIC explicitly keeps request/response transport and does not introduce
    SSE/WebSocket.

## Цель

- Пользователь видит единый history список persisted backtest runs.
- Из summary table run details пользователь открывает detail одного выбранного варианта.
- Browser отправляет только `run_id` и explicit `variant` payload; backend сам восстанавливает
  original request semantics и pinned artifact context исходного run.
- Launch UI must treat `execution_mode=background_auto` as an explicit queued outcome:
  `run_id`, `state`, and `execution_mode` are shown to the user immediately after `POST /backtests`
  instead of keeping an invisible browser-side sync/job toggle.
- `background_manual_legacy` may still appear for legacy persisted rows in history/detail pages,
  but only as a `compatibility-only` literal and not as an active launch choice;
- this compatibility remains necessary because history/detail pages can still open pre-cutover
  queued/running rows until the worker drives them to terminal state.

## Scope

### 1) Routes

- `GET /backtests/history` в `apps/web`
- `GET /backtests/runs/{run_id}` persisted run summary page
- `GET /backtests/runs/{run_id}/variants/{variant_key}` persisted one-variant detail page
- browser API calls:
  - `GET /api/backtests/runs`
  - `GET /api/backtests/runs/{run_id}`
  - `GET /api/backtests/runs/{run_id}/top`
  - `POST /api/backtests/runs/{run_id}/variant-report`

### 2) History page

- Загружает persisted runs list через `GET /api/backtests/runs`.
- Показывает summary metadata:
  - `run_id`
  - `state`
  - `execution_mode`
  - `execution_profile_mode`
  - `market_id`, `symbol`, `timeframe`
  - `requested_top_n`
  - timestamps
  - progress counters + additive `progress_percent/eta_seconds`
    where ETA uses `throughput first -> benchmark fallback second -> null last`
- Не показывает и не кэширует internal hashes (`request_hash`, `engine_params_hash`, etc.).

### 3) Run details page

- Загружает status snapshot через `GET /api/backtests/runs/{run_id}`.
- Загружает summary rows через `GET /api/backtests/runs/{run_id}/top`.
- Показывает additive progress UX:
  - weighted `progress_percent`
  - `eta_seconds`, when available
  - `execution_profile_mode`
  - raw `processed_units/total_units` as supporting counters
- Показывает действия per row:
  - `Open detail` -> `/backtests/runs/{run_id}/variants/{variant_key}`
  - `Save as Strategy` -> existing strategy builder prefill flow
- Первый render сохраняет server order `rank ASC, variant_key ASC`.
- Browser-side local resort:
  - читает approved `contracts.summary.sortable_columns` из
    `GET /api/backtests/runtime-defaults`;
  - переставляет только уже загруженные summary rows;
  - не триггерит новый run и не запрашивает server recompute `/top`.
- `/top` rows содержат только:
  - `rank`
  - `variant_key`
  - `indicator_variant_key`
  - `variant_index`
  - `total_return_pct`
  - `payload`
  - `summary_metrics_json`
  - `best_tp_pct`
  - `best_sl_pct`
- `/top` не содержит persisted `report_table_md`, `trades`, `equity`.

### 4) Variant detail fetch

- Dedicated detail page открывается по persisted `run_id + variant_key`.
- На bootstrap detail page:
  - читает `GET /api/backtests/runs/{run_id}`;
  - читает `GET /api/backtests/runs/{run_id}/top` c `limit=requested_top_n`;
  - находит selected row exact-match по `variant_key`.
- После разрешения row identity detail page отправляет:
  - `POST /api/backtests/runs/{run_id}/variant-report`
- Request body:
  - `variant`
  - `include_trades?`
- UI не отправляет:
  - `time_range`
  - `strategy_id`
  - `template`
  - `overrides`
  - `artifact_slot*`
- Response:
  - `rows`
  - `table_md`
  - `trades?`
- UI показывает:
  - detailed metrics из `rows`
  - markdown table из `table_md`
  - trades table
  - trade-sequenced equity chart, построенный client-side из `trades`

### 5) Client-side state

- `run_id` берётся из route/path и остаётся canonical public identifier.
- `variant` payload берётся из selected `/top` row `payload`.
- Report cache key в браузере:
  - `run_id + variant_key + include_trades`
- Изменение `include_trades=false -> true` считается отдельным fetch.
- Save-as-strategy flow:
  - строит prefill из `payload.indicator_selections[]`;
  - `market_type` / `instrument_key` восстанавливает через `GET /api/market-data/markets`
    + status fields `market_id/symbol/timeframe`;
  - сохраняет payload в `sessionStorage` и редиректит на `/strategies/new?prefill=<id>`.

## Invariants

- Owner policy user-visible и deterministic:
  - missing run -> `404 not_found`
  - foreign existing run -> `403 forbidden`
- Summary table на `/backtests/runs/{run_id}` остаётся trades-free и report-free.
- `/backtests` launch page не materialize'ит report/trades inline; detail lives only on the
  dedicated persisted variant page.
- Detail endpoint пересчитывает ровно один selected variant.
- Detail payload не сохраняется в PG как часть persisted run history.
- Opening detail page не запускает full top-N recompute.
- Missing `variant_key` на detail page даёт explicit browser-side fallback message вместо
  silent fallback на другой variant.
- Legacy `/api/backtests/variant-report` может использоваться только как compatibility path
  для старых клиентов, но новый UI не должен зависеть от full run envelope.

## Related Files

- `apps/api/routes/backtest_runs.py`
- `apps/api/dto/backtest_runs.py`
- `apps/api/routes/backtests.py`
- `apps/api/dto/backtests.py`
- `apps/web/dist/backtest_jobs_ui.js`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`

## How To Verify

```bash
uv run pytest -q tests/unit/apps/api/test_backtest_runs_routes.py tests/unit/apps/api/test_backtest_runs_dto.py tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_history_api_v1.py
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Manual smoke:

1. Открыть history page и перейти в persisted run details.
2. Убедиться, что active run summary page показывает `stage`, `progress_percent`,
   `execution_profile_mode` и `eta_seconds` (если estimate уже есть).
3. Убедиться, что summary table грузится через `/api/backtests/runs/{run_id}/top`.
4. Нажать `Open detail` у строки top table.
5. Проверить, что detail page вызывает `/api/backtests/runs/{run_id}/variant-report` и не шлёт
   full run envelope.
6. Проверить, что `include_trades=true` возвращает `trades`, а `false` — нет.
7. Нажать `Save as Strategy` на summary и detail page и проверить редирект на
   `/strategies/new?prefill=...`.
