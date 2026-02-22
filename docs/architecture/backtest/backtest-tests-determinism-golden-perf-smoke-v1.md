# Backtest Tests v1 — Determinism + Golden Fixtures + Perf Smoke (BKT-EPIC-08)

Фиксирует набор тестов v1 для backtest Milestone 4: как гарантируем детерминизм (grid/signals/engine/reporting), как устроены golden fixtures (строковый отчёт), и как запускаем perf-smoke без флапов.

## Цель

- Зафиксировать корректность и детерминизм backtest v1:
  - grid builder (variants + guards),
  - signal aggregation,
  - execution engine (close-fill),
  - reporting (equity/trades + metrics table).
- Добавить “golden” тесты: один и тот же вход -> идентичный строковый отчёт (`report.table_md`) и ключевые числа.
- Добавить perf-smoke тест: небольшой sync grid в пределах guards, без тяжёлых интеграционных прогонов.

## Контекст

- Backtest v1 реализован по staged pipeline v1 (Stage A shortlist -> Stage B top-K):
  - `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- Execution engine v1 (close-fill + fee/slippage + sizing + SL/TP):
  - `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
- Reporting v1 (equity curve + trades + metrics table):
  - `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
- В repo уже есть паттерн perf-smoke тестов для indicators (без строгого SLA):
  - `tests/perf_smoke/contexts/indicators/*`

EPIC-08 вводит контроль “регрессий детерминизма” и базовый контроль “катастрофических” регрессий производительности.

## Scope

1) Unit tests (determinism)

- Grid builder:
  - детерминированный порядок variants,
  - guards (variants/memory),
  - корректная Stage A/Stage B семантика.
- Signals:
  - deterministic rule registry,
  - корректная AND агрегация,
  - NaN/warmup => NEUTRAL,
  - pivot dependencies.
- Execution engine:
  - edge-based entry gating,
  - direction modes,
  - fee/slippage,
  - close-based SL/TP,
  - forced close,
  - profit lock.
- Reporting:
  - equity curve,
  - метрики и форматирование (строго фиксированный порядок и строки).

2) Golden tests (строковый отчёт)

- Two scenarios (фикс):
  - **no-trades**: final_signal == NEUTRAL на всём `target_slice`.
  - **multi-trade**: заранее заданный final_signal c несколькими edge входами/выходами (>=2 trades), чтобы были определены trade stats + drawdowns + ratios.
- Golden тесты выполняются на уровне:
  - `BacktestExecutionEngineV1` + `BacktestReportingServiceV1`
  - входом является **готовый** `final_signal[]` (без indicator_compute), чтобы исключить Numba/compute вариативность.
- Golden фикстуры храним как файлы в репозитории:
  - `tests/unit/contexts/backtest/golden/*.md`
- Проверки golden:
  - `report.table_md` сравнивается 1:1 со строкой из файла,
  - дополнительно (для читаемого дебага) проверяются несколько ключевых чисел/строк (например `Total Return [%]`, `Num. Trades`).

3) Perf smoke (маленький sync grid)

- Добавляем `tests/perf_smoke/contexts/backtest/*`.
- Цель perf-smoke:
  - убедиться, что небольшой staged grid запускается до конца,
  - не нарушает guards,
  - не “взрывается” по памяти,
  - не превращается в патологически медленный сценарий.
- SLA по времени НЕ фиксируем (чтобы не флапать в CI/на разных CPU).
- Допускается очень широкий “катастрофический” лимит (например `< 60s`) как защита от зависаний.

## Non-goals

- Долгие интеграционные тесты на больших периодах/с реальными ClickHouse/Postgres.
- Жёсткие latency SLA в тестах.
- Golden по полному `POST /backtests` (HTTP) — это отдельный уровень (smoke/integration) и не часть v1 golden.

## Ключевые решения

### 1) Golden строим без indicator_compute (2A)

Golden тесты используют детерминированный `final_signal[]` как вход:

- это фиксирует семантику **execution + reporting**,
- исключает влияние Numba/JIT и вариативность compute.

Связанные файлы:
- `src/trading/contexts/backtest/application/services/execution_engine_v1.py`
- `src/trading/contexts/backtest/application/services/reporting_service_v1.py`

### 2) Два golden сценария (1A)

- no-trades: проверяем стабильность “нулевого” отчёта.
- multi-trade: проверяем стабильность trade stats, drawdowns и ratios.

Последствия:
- минимальная цена поддержки,
- достаточное покрытие, чтобы поймать регрессии формул/форматирования.

### 3) Golden fixtures храним как `.md` файлы (3A)

Golden строки хранятся как ASCII `.md` и сравниваются 1:1.

Последствия:
- понятные diff’ы в PR,
- строгая защита от случайных изменений форматирования.

### 4) Perf smoke без строгого SLA + отдельные “complexity invariants” (4A)

Чтобы perf-smoke не флапал:

- не фиксируем tight SLA по времени,
- добавляем unit тесты на детерминированные инварианты сложности, например:
  - Stage B risk expansion НЕ увеличивает число `IndicatorCompute.compute(...)` вызовов (за счёт signal cache в `CloseFillBacktestStagedScorerV1`),
  - количество score вызовов соответствует формуле Stage A и Stage B totals.

Связанные файлы:
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py` (signal cache)
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`

## Контракты и инварианты

- Тесты не должны зависеть от порядка словарей:
  - все сравнения должны опираться на фиксированные ordering rules (`BACKTEST_METRIC_ORDER_V1`, tie-break keys).
- Golden сравнивает строку `report.table_md` 1:1 и должен быть стабилен на разных платформах.
- Perf-smoke должен быть:
  - небольшим (guard-safe),
  - не требовать внешних сервисов,
  - детерминированным (фиксированные candles/signals/params),
  - не флапать по времени.

## Связанные файлы

Документы:
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
- `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`

Код:
- `src/trading/contexts/backtest/application/services/grid_builder_v1.py`
- `src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py`
- `src/trading/contexts/backtest/application/services/execution_engine_v1.py`
- `src/trading/contexts/backtest/application/services/reporting_service_v1.py`
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`

Тесты:
- `tests/unit/contexts/backtest/**`
- `tests/unit/contexts/backtest/golden/*`
- `tests/perf_smoke/contexts/backtest/*`

## Как проверить

```bash
# запускать из корня репозитория
uv run ruff check .
uv run pyright
uv run pytest -q
```

Отдельно (при отладке):

```bash
uv run pytest -q tests/unit/contexts/backtest
uv run pytest -q tests/perf_smoke/contexts/backtest
```

## Риски и открытые вопросы

- Риск: golden тесты могут потребовать обновления при сознательном изменении методики метрик/форматирования. Политика: обновлять golden только вместе с изменением архитектурных документов EPIC-05/06 и с явным diff’ом.
- Открытые вопросы: отсутствуют (все решения согласованы).
