# Backtest v1 — Bounded Context + Domain + Use-Case Skeleton (Milestone 4 / BKT-EPIC-01)

Документ фиксирует архитектуру BKT-EPIC-01: как вводим bounded context `backtest` (domain/application/ports), какие минимальные контракты (DTO/ошибки) и как он интегрируется с `market_data`/`indicators`/`strategy`, не фиксируя преждевременно детали исполнения/метрик.

## Цель

1) Завести bounded context `src/trading/contexts/backtest/*` в стиле репозитория (DDD layers: domain/application/adapters).  
2) Зафиксировать минимальные доменные сущности/ошибки/инварианты для backtest v1.  
3) Зафиксировать application use-case “run backtest v1” (вход/выход DTO) и общий механизм ошибок через `RoehubError` (deterministic 422 payload).

## Контекст

- Milestone 4 вводит backtest v1 (close-fill) по одному инструменту, с grid вариантами и staged pipeline (Stage A shortlist → Stage B exact → top-K).
- В репозитории уже есть:
  - каноническое чтение 1m свечей: `CanonicalCandleReader` (market_data port)
  - детерминированный dense timeline + NaN holes: `CandleFeed.load_1m_dense(...)` (indicators port + ACL)
  - вычисление индикаторных тензоров по grid: `IndicatorCompute.compute(...)` (indicators port)
  - immutable StrategySpecV1 и явные правила ownership/visibility через use-cases (strategy)
  - единый контракт ошибок API: `RoehubError` + deterministic 422 payload.

Этот EPIC сознательно не реализует:
- backtest execution loop,
- staged grid builder,
- расчёт метрик отчёта,
- API роуты.

## Scope

### 1) Новый bounded context: `src/trading/contexts/backtest/*`

- Domain:
  - минимальные value objects и сущности (backtest request spec, variant identity, placeholders для trade/position/result)
  - доменные ошибки (валидация, forbidden/not-found, бюджетные ограничения как validation)
- Application:
  - DTO входа/выхода “run backtest v1” (saved + ad-hoc template)
  - один use-case: `RunBacktestUseCase.execute(...)`
  - application ports:
    - candles: используем `CandleFeed` из `indicators`
    - indicators compute: используем `IndicatorCompute` из `indicators`
    - strategy loading (saved mode): отдельный порт backtest (ACL) к strategy storage
  - маппинг исключений → `RoehubError` (422/403/404/409/500) по общему стандарту.

### 2) Runtime config v1 (минимум)

Source of truth: `configs/<env>/backtest.yaml`.

Ключи:
- `warmup_bars_default` (default 200)
- `top_k_default` (default 300)
- `preselect_default` (default 20_000)

Разрешение пути конфигурации:
- override: `ROEHUB_BACKTEST_CONFIG`
- fallback: `configs/<ROEHUB_ENV>/backtest.yaml` (env default `dev`)

## Non-goals

- Jobs/progress и сохранение результатов (Milestone 5).
- Intrabar execution (Milestone 7).
- Реализация staged runner, ranking, top-K таблицы и метрик отчёта (BKT-EPIC-04/05/06).
- API endpoints и transport DTO (BKT-EPIC-07).

## Ключевые решения

### 1) Свечи для backtest: используем `CandleFeed` (indicators port), а не `CanonicalCandleReader` напрямую

Backtest use-case читает свечи через `trading.contexts.indicators.application.ports.feeds.CandleFeed`.

Причины:
- `IndicatorCompute` требует `CandleArrays` (dense timeline + NaN holes), и этот контракт уже реализован и протестирован в `MarketDataCandleFeed`.
- это устраняет дублирование “dense timeline builder” внутри backtest.

Последствия:
- backtest не зависит напрямую от ClickHouse/market_data adapters;
- backtest получает детерминированный `CandleArrays` на полуинтервале `[start, end)`.

### 2) Индикаторы: используем `IndicatorCompute` (indicators port) напрямую

Backtest use-case зависит от `trading.contexts.indicators.application.ports.compute.IndicatorCompute`.

Причины:
- библиотека индикаторов и compute engine уже формализованы в bounded context `indicators`.

Последствия:
- backtest не реализует вычисления индикаторов;
- budget/guard ошибки indicators мапятся в backtest как `validation_error` (422).

### 3) Saved strategy mode: ownership/visibility проверяются в backtest use-case (не в API слое)

Если backtest запускается по `strategy_id`, backtest use-case:
- загружает стратегию,
- явно проверяет, что стратегия принадлежит `current_user`,
- работает только с non-deleted snapshot.

Причины:
- согласование с архитектурным правилом Strategy: ownership/visibility — доменно-очевидное правило в use-case (а не “фильтр в SQL” и не обязанность HTTP слоя).

Последствия:
- backtest use-case можно вызывать из разных inbound адаптеров без риска обхода ACL;
- нужен backtest application port для чтения Strategy snapshot (см. следующее решение).

### 4) Интеграция со Strategy: вводим порт backtest-слоя для загрузки strategy snapshot

Внутри backtest application layer вводим порт (название фиксируется в реализации) вида:

- `BacktestStrategyReader.load_any(strategy_id) -> Strategy | None`

Семантика:
- возвращает snapshot по id без owner фильтрации;
- ownership/deleted checks выполняются в backtest use-case.

Причины:
- backtest контекст не должен импортировать strategy adapters.

Последствия:
- реализация порта может быть тонким ACL поверх `StrategyRepository.find_any_by_strategy_id(...)`.

### 5) Use-case DTO: один request DTO поддерживает saved и ad-hoc template режимы

Фиксируем один request DTO v1 с взаимоисключающими полями:

- либо `strategy_id` (saved mode)
- либо `template` (ad-hoc grid mode)

Причины:
- один use-case и один контракт для future API `POST /backtests`.

Последствия:
- transport слой (FastAPI/Pydantic) в BKT-EPIC-07 повторит этот контракт без “расхождений”.

### 6) Variant identity: фиксируем `variant_index` + `variant_key` v1 (sha256 canonical json)

Backtest результаты (в т.ч. top-K) должны возвращать идентичность варианта:

- `variant_index`: позиция в детерминированном порядке
- `variant_key`: стабильный hash, чтобы UI мог:
  - сохранить выбранный вариант как immutable Strategy,
  - ссылаться на вариант в UI без “плавающих” идентификаторов.

Как строится `variant_key`:
- sha256 от canonical JSON payload v1;
- indicator часть должна быть совместима с `indicators` variant_key semantics (`build_variant_key_v1`) и расширяется параметрами backtest (risk/sizing/fees/slippage).

### 7) Runtime config: отдельный `configs/<env>/backtest.yaml` с fail-fast loader/validator

Причины:
- backtest имеет собственные runtime параметры (warmup/top-K/preselect), которые не должны “раствориться” в других конфигах.

Последствия:
- появится loader/validator по паттерну `strategy_runtime_config.py`.

### 8) Ошибки: единый контракт `RoehubError` и deterministic 422 payload

Backtest use-case мапит доменные и интеграционные ошибки в `RoehubError`:

- `validation_error` (422): неверный payload, guard/budget exceeded, invalid range
- `not_found` (404): strategy_id не существует или soft-deleted (не различаем)
- `forbidden` (403): стратегия не принадлежит пользователю
- `conflict` (409): конфликт инвариантов use-case (например, несовместимые поля request)
- `unexpected_error` (500): неизвестные ошибки

## Контракты и инварианты

- Time range: полуинтервал `[start, end)` (shared-kernel `TimeRange`).
- Warmup lookback: `warmup_bars_default` трактуется как **число баров выбранного timeframe** (например 200 баров `1h` = 200 часов).
- Saved strategy: backtest use-case обязан проверить owner и `is_deleted` (если стратегия удалена — `not_found`).
- Детерминизм:
  - порядок materialization grid/variants должен быть детерминирован (фиксируется в BKT-EPIC-04),
  - `variant_key` должен быть воспроизводим.
- Errors:
  - 422 payload детерминированно сортирован как в `apps/api/common/errors.py`.

## Связанные файлы

Roadmap:
- `docs/architecture/roadmap/milestone-4-epics-v1.md` — полный EPIC map Milestone 4.
- `docs/architecture/roadmap/base_milestone_plan.md` — milestone requirements.

Errors:
- `docs/architecture/api/api-errors-and-422-payload-v1.md` — единый контракт ошибок.
- `src/trading/platform/errors/roehub_error.py` — RoehubError.
- `apps/api/common/errors.py` — handlers и deterministic 422.

Indicators (candles + compute):
- `src/trading/contexts/indicators/application/ports/feeds/candle_feed.py` — CandleFeed port.
- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py` — dense timeline + NaN holes.
- `src/trading/contexts/indicators/application/ports/compute/indicator_compute.py` — IndicatorCompute port.
- `src/trading/contexts/indicators/application/dto/compute_request.py` — ComputeRequest.
- `src/trading/contexts/indicators/application/dto/variant_key.py` — indicator variant_key v1 builder.

Strategy (saved mode reference):
- `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md` — ownership/visibility rule.
- `src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py` — StrategyRepository port.

Backtest (будет создано в реализации EPIC):
- `src/trading/contexts/backtest/domain/`
- `src/trading/contexts/backtest/application/dto/`
- `src/trading/contexts/backtest/application/ports/`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/dev/backtest.yaml`, `configs/test/backtest.yaml`, `configs/prod/backtest.yaml`

## Как проверить

После реализации EPIC:

```bash
# запускать из корня репозитория
uv run ruff check .
uv run pyright
uv run pytest -q

# обновить индекс документации
uv run python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: расползание variant_key семантики между indicators и backtest.
  Нужна явная canonicalization политика и переиспользование `indicators` builder как building block.
- Риск: слишком “толстый” request DTO в v1.
  Смягчается тем, что EPIC-01 фиксирует только DTO/контракты, а исполнение/метрики реализуются позднее.
