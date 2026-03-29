# Web UI -- Backtest History and Variant Detail v2

Документ фиксирует web contract для history/details flow после R7-04:
public UI работает через persisted `runs` vocabulary и открывает detail выбранного варианта
по `run_id + variant`, без отправки полного run envelope обратно с клиента.

## Status

- Status: active target contract after R7-04.
- Depends on:
  - `docs/architecture/backtest/backtest-runs-history-v2.md`
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - `docs/architecture/roadmap/base_refactor_plan.md`
- Compatibility note:
  - legacy `POST /api/backtests/variant-report` остаётся migration path;
  - preferred public detail flow использует
    `POST /api/backtests/runs/{run_id}/variant-report`.
  - after R9-01 the `/backtests` launch page may land on explicit
    `202 Accepted` `background_auto` responses and hand users off to
    `/backtests/jobs/{run_id}` as a compatibility surface until R9-02 history UX is promoted.

## Цель

- Пользователь видит единый history список persisted backtest runs.
- Из summary table run details пользователь открывает detail одного выбранного варианта.
- Browser отправляет только `run_id` и explicit `variant` payload; backend сам восстанавливает
  original request semantics и pinned artifact context исходного run.
- Launch UI must treat `execution_mode=background_auto` as an explicit queued outcome:
  `run_id`, `state`, and `execution_mode` are shown to the user immediately after `POST /backtests`
  instead of keeping an invisible browser-side sync/job toggle.

## Scope

### 1) Routes

- `GET /backtests/history` или equivalent history page route в `apps/web`
- `GET /backtests/runs/{run_id}` или equivalent run details page route
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
  - `market_id`, `symbol`, `timeframe`
  - `requested_top_n`
  - `artifact_slot`, `artifact_slot_generation`, `artifact_manifest_hash`,
    `artifact_asof_date`
- Не показывает и не кэширует internal hashes (`request_hash`, `engine_params_hash`, etc.).

### 3) Run details page

- Загружает status snapshot через `GET /api/backtests/runs/{run_id}`.
- Загружает summary rows через `GET /api/backtests/runs/{run_id}/top`.
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

- По клику `Load report` UI отправляет:
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

### 5) Client-side state

- `run_id` берётся из route/path и остаётся canonical public identifier.
- `variant` payload берётся из selected `/top` row `payload` plus explicit
  `direction_mode/sizing_mode/execution/risk/signal` scalars.
- Report cache key в браузере:
  - `run_id + variant_key + include_trades`
- Изменение `include_trades=false -> true` считается отдельным fetch.

## Invariants

- Owner policy user-visible и deterministic:
  - missing run -> `404 not_found`
  - foreign existing run -> `403 forbidden`
- Detail endpoint пересчитывает ровно один selected variant.
- Detail payload не сохраняется в PG как часть persisted run history.
- Opening detail page не запускает full top-N recompute.
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
2. Убедиться, что summary table грузится через `/api/backtests/runs/{run_id}/top`.
3. Нажать `Load report` у строки top table.
4. Проверить, что browser вызывает `/api/backtests/runs/{run_id}/variant-report` и не шлёт
   full run envelope.
5. Проверить, что `include_trades=true` возвращает `trades`, а `false` — нет.
