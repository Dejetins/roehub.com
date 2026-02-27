---
title: План оптимизации производительности staged backtest (ranking + reporting + UI on-demand) (v1)
version: 1
status: draft
owner: backtest
---

# План оптимизации производительности staged backtest (ranking + reporting + UI on-demand) (v1)

Документ фиксирует пошаговый план изменений для backtest compute path и UI-контрактов без изменения базовой детерминированности перебора вариантов и variant identity.

## Цель

- Убрать из hot path лишние расчеты full-details при ранжировании.
- Сделать ранжирование конфигурируемым по 1-2 метрикам (default: `Return %`).
- Убрать массовую генерацию `rows/table_md` для всего `top_k`; перейти к on-demand report по клику из UI.
- Снизить write amplification и лишние сериализации в jobs path.

## Scope

В этот план входят 6 направлений:

1. Configurable ranking по 1-2 метрикам.
2. Разделение metric-only scoring и details scoring без дублирования расчетов.
3. On-demand отчет (`rows/table_md/trades`) по выбранному варианту из UI.
4. Оптимизация jobs snapshot/finalizing path (меньше повторных writes/serialization).
5. Снижение накладных расходов на variant/cache keys в hot loop.
6. Удаление дублирующих sort/normalize проходов в API mapping.

## Non-goals

- Изменение формул индикаторов, signal rules, или семантики execution engine.
- Изменение детерминированного порядка `variant_key` tie-break и variant enumeration.
- Добавление новых внешних зависимостей.

## Инварианты (обязательно сохранить)

- Детерминированный ranking order: сначала выбранные ranking metrics, затем `variant_key ASC`.
- Детерминированность Stage A/Stage B variant enumeration.
- `variant_key`/`indicator_variant_key` семантика и формат не меняются.
- Default ranking остается `Total Return [%] DESC`.

## Baseline: текущие узкие места

1. `score_variant(...)` фактически вызывает `score_variant_with_details(...)`, поэтому ранжирование всегда считает полный execution outcome.
2. В jobs finalizing есть повторный details-score для persisted rows.
3. Sync response строит `rows + table_md` для каждого `top_k`, даже если пользователь не открывает отчет.
4. Jobs snapshot использует full replace (`delete+insert`) на каждом persist cadence.
5. В hot loops выполняется частая JSON+SHA сериализация для keys/cache keys.
6. Есть дублирующие defensive sort в API mapping при уже отсортированном application response.

## Направление 1: Configurable ranking по 1-2 метрикам

### 1.1 Контракт ranking конфигурации

Добавляем ranking-конфиг на 2 уровнях:

- runtime defaults (`backtest.yaml`)
- request override (`POST /backtests`, `POST /backtests/jobs`)

Предлагаемый shape:

```yaml
backtest:
  ranking:
    primary_metric_default: total_return_pct
    secondary_metric_default: null
```

```json
{
  "ranking": {
    "primary_metric": "total_return_pct",
    "secondary_metric": "max_drawdown_pct"
  }
}
```

### 1.2 Допустимые ranking metrics (v1)

Метрики должны считаться в metric-only execution pass без построения полного report:

- `total_return_pct` (`DESC`) — default.
- `max_drawdown_pct` (`ASC`).
- `return_over_max_drawdown` (`DESC`).
- `profit_factor` (`DESC`).

Если `secondary_metric` не задана: tie-break по `variant_key ASC`.

### 1.3 Изменения DTO/API/UI

- Добавить ranking-блок в request DTO для sync/jobs.
- Добавить ranking defaults в `GET /backtests/runtime-defaults`.
- UI: селекты `primary_metric` и `secondary_metric` в Advanced секции.

### Файлы (направление 1)

- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/dev/backtest.yaml`
- `configs/test/backtest.yaml`
- `configs/prod/backtest.yaml`
- `src/trading/contexts/backtest/application/dto/run_backtest.py`
- `apps/api/dto/backtests.py`
- `apps/api/dto/backtest_jobs.py`
- `apps/api/dto/backtest_runtime_defaults.py`
- `apps/web/templates/backtests.html`
- `apps/web/dist/backtest_ui.js`

## Направление 2: Metric-only scoring без дублирования details

### 2.1 Разделить scorer API

AS-IS:

```python
score_variant(...) -> score_variant_with_details(...)
```

TO-BE:

```python
score_variant_metric(...) -> RankingMetricsV1
score_variant_with_details(...) -> BacktestVariantScoreDetailsV1
```

Ранжирование Stage A/Stage B использует только metric-only path.

### 2.2 Sync path

- Stage A/Stage B: только metric-only.
- Details считаются только для выбранного варианта по explicit report request (см. направление 3).

### 2.3 Jobs path

- Stage B ranking: только metric-only.
- Finalizing не пересчитывает full details для persisted `top_k`.
- Details/report считаются только по on-demand UI/API запросу.

### Файлы (направление 2)

- `src/trading/contexts/backtest/application/ports/staged_runner.py`
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- `src/trading/contexts/backtest/application/services/staged_core_runner_v1.py`
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

## Направление 3: On-demand report по клику из UI

### 3.1 Sync response policy

AS-IS: каждый variant в `top_k` содержит `report.rows + report.table_md (+trades для top_trades_n)`.

TO-BE:

- `POST /backtests` возвращает только ranking/payload summary без full report body.
- Полный отчет запрашивается отдельным endpoint по выбранному варианту.

### 3.2 Новый endpoint для отчета варианта

Добавить endpoint для lazy report build:

- `POST /api/backtests/variant-report`

Вход:

- run context (time_range, instrument/timeframe, warmup, execution context)
- explicit variant payload (`indicator_selections`, `signal_params`, `risk_params`, `execution_params`)

Выход:

- `rows`
- `table_md`
- `trades` (с учетом policy `top_trades_n`/explicit include flag)

### 3.3 Jobs UI policy

Для `/backtests/jobs/{job_id}`:

- `/top` отдает ranking summary без `report_table_md/trades`.
- По клику "Load report" для row вызывается variant-report endpoint.

### 3.4 UI изменения

Sync UI (`/backtests`):

- убрать рендер full report в таблице top-k.
- добавить action-кнопку `Load report`.
- добавить lazy cache в браузере по `variant_key`.

Jobs UI (`/backtests/jobs/{job_id}`):

- убрать auto-render `report_table_md/trades` из top table.
- добавить per-row `Load report`.

### Файлы (направление 3)

- `apps/api/routes/backtests.py`
- `apps/api/routes/backtest_jobs.py`
- `apps/api/dto/backtests.py`
- `apps/api/dto/backtest_jobs.py`
- `src/trading/contexts/backtest/application/services/reporting_service_v1.py`
- `apps/web/templates/backtests.html`
- `apps/web/templates/backtest_job_details.html`
- `apps/web/dist/backtest_ui.js`
- `apps/web/dist/backtest_jobs_ui.js`

## Направление 4: Jobs snapshot/finalizing optimization

### 4.1 Snapshot writes only when frontier changed

Перед `replace_top_variants_snapshot(...)` проверять, изменился ли frontier (`variant_key + ranking_metrics` signature).

### 4.2 Убрать full details из finalizing

Finalizing делает только terminal state/progress и не выполняет массовый report build.

### 4.3 Checkpoint callback optimization

Не материализовать ranked rows/tasks mapping на каждом checkpoint без необходимости persist.

### Файлы (направление 4)

- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `src/trading/contexts/backtest/application/services/staged_core_runner_v1.py`
- `src/trading/contexts/backtest/application/services/job_runner_streaming_v1.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`

## Направление 5: Key/cache hot-path optimization

### 5.1 Variant key build

- сократить число `json.dumps + sha256` в Stage A/Stage B loops.
- переиспользовать преднормализованные payload-компоненты.

### 5.2 Signal cache key

- убрать лишнюю canonical JSON сериализацию для cache key там, где signal params уже нормализованы и immutable.

### Файлы (направление 5)

- `src/trading/contexts/backtest/application/services/grid_builder_v1.py`
- `src/trading/contexts/backtest/application/services/staged_core_runner_v1.py`
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py`
- `src/trading/contexts/indicators/application/dto/variant_key.py`

## Направление 6: Удаление лишних sort/normalize проходов

- убрать повторный sort variants в API mapper, если application DTO уже гарантирует order.
- минимизировать повторные normalized mapping passes на boundary, где payload уже canonical.

### Файлы (направление 6)

- `apps/api/dto/backtests.py`
- `src/trading/contexts/backtest/application/dto/run_backtest.py`

## Порядок внедрения

1. Направление 1 (ranking config + DTO + UI controls).
2. Направление 2 (metric-only scoring split) с сохранением текущего API behavior за feature-flag.
3. Направление 3 (variant-report endpoint + lazy UI) и отключение массового report build.
4. Направление 4 (jobs snapshot/finalizing/checkpoint optimization).
5. Направление 5 и 6 (micro-optimizations и cleanup).

## Совместимость и rollout

- Ввести feature flag `backtest.reporting.eager_top_reports_enabled` (временный, default `false` в dev/test, `true` на переходном этапе в prod).
- После подтверждения UI migration перевести prod default на lazy-only и удалить legacy флаг.
- Сохранить backward-compatible поле `total_return_pct` в variant response даже при multi-metric ranking.

## Что обновить в документации

- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/backtest/backtest-jobs-api-v1.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
- `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`
- `docs/architecture/backtest/README.md` (ссылка на этот план)

## Проверки, чтобы не сломать поведение

```bash
uv run ruff check .
uv run pyright
uv run pytest -q tests/unit/contexts/backtest
uv run pytest -q tests/unit/apps/api/test_backtests*
uv run pytest -q tests/unit/apps/api/test_backtest_jobs*
uv run pytest -q tests/perf_smoke/contexts/backtest
uv run python -m tools.docs.generate_docs_index --check
```

## Критерии готовности

- Ранжирование работает по конфигурируемым 1-2 метрикам, default = `total_return_pct`.
- Stage A/Stage B не считают full details в ranking hot path.
- Sync/jobs UI получает full report только по explicit user action.
- Убраны дублирующие details расчеты в jobs finalizing.
- Snapshot writes и checkpoint serialization сокращены без потери детерминизма.
- Документация и runtime defaults синхронизированы с кодом.
