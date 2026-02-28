# Backtest v1 — Reporting: Equity Curve + Trades + Metrics Table (BKT-EPIC-06)

Фиксирует контракт BKT-EPIC-06: как детерминированно строится отчёт backtest v1 (equity curve + trade log + таблица метрик `|Metric|Value|`), включая benchmark и ratios.

## Цель

- Построить детерминированный отчёт по одному варианту backtest v1:
  - trade log,
  - equity curve на close,
  - метрики из Milestone 4 в формате таблицы `|Metric|Value|`.
- Для grid запуска: считать отчёт on-demand для выбранного варианта через `variant-report`.
- Зафиксировать benchmark/ratios семантики v1.
- Зафиксировать детерминированное форматирование значений (строк) в таблице.

## Контекст

- Execution engine v1 (BKT-EPIC-05) уже даёт детерминированное close-fill исполнение и trade log:
  - `src/trading/contexts/backtest/application/services/execution_engine_v1.py`
  - `src/trading/contexts/backtest/domain/entities/execution_v1.py`
- Grid/staged pipeline (BKT-EPIC-04) возвращает только top-K вариантов по `Total Return [%]`:
  - `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
  - `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- Candle timeline (BKT-EPIC-02) фиксирует:
  - warmup vs `target_slice`,
  - правило включения баров по close timestamp: `Start <= bar_close_ts < End`.

EPIC-06 добавляет слой “reporting” поверх существующего execution/scoring:

- ranking в Stage A/Stage B остаётся по `Total Return [%]`;
- sync/reporting flow по умолчанию lazy и не строит отчёты для всего top-K upfront.

## Scope

- Вход в reporting v1 (per variant):
  - warmup-inclusive `CandleArrays`,
  - `target_slice`,
  - `ExecutionParamsV1` + `RiskParamsV1`,
  - trade log (`TradeV1[]`) и/или возможность воспроизвести его через engine.
- Выход v1:
  - equity curve на close (как ряд `equity_close_quote[t]` на `target_slice`),
  - trade log (для варианта),
  - таблица метрик `|Metric|Value|` (строго фиксированный порядок и формат).
- Для grid запуска:
  - отчёт строится по explicit запросу `POST /api/backtests/variant-report`,
  - sync eager reports включаются только через
    `backtest.reporting.eager_top_reports_enabled=true`,
  - trades включаются по `include_trades` (variant-report) или по `top_trades_n` в eager policy.
- Метрики v1 (минимум, фиксированный список):
  - PnL/Return,
  - drawdowns + drawdown durations,
  - coverage/exposure,
  - trade stats,
  - Expectancy + SQN,
  - Sharpe/Sortino/Calmar,
  - benchmark return.

## Non-goals

- UI визуализация (front-end).
- Сохранение результатов в БД (Milestone 5 jobs/history).
- Расширение набора метрик сверх списка Milestone 4.
- “Тяжёлые” отчёты для всех Stage B вариантов в одном sync response.

## Ключевые решения

### 1) Report считается на `target_slice`; Start/End/Duration в таблице — из request time_range

- Все вычисления (equity curve, trades, benchmark, ratios) выполняются только на баровом `target_slice`.
- Поля `Start`, `End`, `Duration` в таблице показывают **request** `time_range` (а не фактические границы баров).

Причины:
- warmup не должен влиять на метрики;
- UI ожидает видеть тот же диапазон, который пользователь запросил.

### 2) Equity curve v1: equity на каждом bar close (после применения действий engine на этом close)

Equity curve `equity_close_quote[t]` строится на close каждого бара `t` в `target_slice`.

Определение equity на баре:

- если позиции нет: `equity = account.available_quote + account.safe_quote`
- если позиция есть: `equity = account.available_quote + account.safe_quote + position_value_quote`

Где `position_value_quote` (v1, детерминированно) считается как:

- long: `qty_base * close_price`
- short (synthetic/margin-style): `entry_quote_amount + unrealized_gross_pnl_quote(close_price)`

Примечание:
- slippage/fees отражаются в equity через entry/exit операции (и через mark-to-market при slippage на entry, если equity оценивается по raw close).

### 3) Метрики v1 и их строгий порядок

Таблица метрик возвращается в строгом порядке (как в Milestone 4):

1. `Start`
2. `End`
3. `Duration`
4. `Init. Cash`
5. `Total Profit`
6. `Total Return [%]`
7. `Benchmark Return [%]`
8. `Position Coverage [%]`
9. `Max. Drawdown [%]`
10. `Avg. Drawdown [%]`
11. `Max. Drawdown Duration`
12. `Avg. Drawdown Duration`
13. `Num. Trades`
14. `Win Rate [%]`
15. `Best Trade [%]`
16. `Worst Trade [%]`
17. `Avg. Trade [%]`
18. `Max. Trade Duration`
19. `Avg. Trade Duration`
20. `Expectancy`
21. `SQN`
22. `Gross Exposure`
23. `Sharpe Ratio`
24. `Sortino Ratio`
25. `Calmar Ratio`

### 4) Методы расчёта: совместимость с `backtesting.py` для Expectancy/SQN + фиксированные ratios

**Trades и trade returns**

- `trade_return_pct = (trade.net_pnl_quote / trade.entry_quote_amount) * 100`.

**Expectancy (v1, совместимо с backtesting.py)**

- `Expectancy = mean(trade_return_pct)` (арифметическое среднее).

**SQN (v1, совместимо с backtesting.py)**

- `SQN = sqrt(N) * mean(trade.net_pnl_quote) / std(trade.net_pnl_quote)`.

**Drawdown (фикс: положительный dd%)**

- `dd_frac[t] = 1 - equity[t] / peak_equity[t]`
- `Max. Drawdown [%] = max(dd_frac) * 100`
- `Avg. Drawdown [%] = mean(dd_frac) * 100` (среднее по всем барам `target_slice`, включая 0)

**Drawdown durations**

- drawdown-эпизод начинается, когда `dd_frac` переходит с `0` в `>0`,
- заканчивается, когда equity возвращается к предыдущему peak (`dd_frac == 0`),
- если recovery не наступил до конца `target_slice`, эпизод закрывается на последнем баре.

`Max. Drawdown Duration` = максимум длительностей эпизодов.

`Avg. Drawdown Duration` = среднее длительностей эпизодов.

**Position Coverage [%]**

- `coverage = 100 * mean(have_position[t])`, где `have_position[t] ∈ {0,1}` для баров `target_slice`.
- В v1 считаем `have_position[t]=1` для баров между `entry_bar_index..exit_bar_index` включительно.

**Gross Exposure**

- `gross_exposure = mean(exposure_frac[t])`, где:
  - если позиции нет: `exposure_frac = 0`
  - если позиция есть: `exposure_frac = entry_quote_amount / equity_close_quote[t]`

**Benchmark Return [%] (фикс v1)**

- buy&hold long без fee/slippage,
- вход: `close[first_target_bar]`, выход: `close[last_target_bar]`,
- `Benchmark Return [%] = (close_last / close_first - 1) * 100`.

**Ratios (фикс v1)**

- строим equity curve по close на `target_slice`,
- ресэмплим equity в 1d (UTC) по last (close),
- считаем дневные доходности `r_d = pct_change(equity_1d)`,
- `risk_free = 0`, annualization = `365`.

Рекомендуемые формулы v1:

- `annual_return = (1 + geometric_mean(r_d))**365 - 1`
- `vol_ann = std(r_d, ddof=1) * sqrt(365)`
- `Sharpe = annual_return / vol_ann`
- `Sortino = annual_return / (sqrt(mean(min(r_d, 0)^2)) * sqrt(365))`
- `Calmar = annual_return / max_dd_frac`

### 5) Детерминированное форматирование таблицы

Reporting v1 возвращает таблицу в виде:

- `rows: list[(metric_name: str, value: str)]` в фиксированном порядке,
- опционально: `table_md: str` (markdown) как дериват от `rows`.

Правила форматирования v1:

- Даты `Start/End`: `isoformat(sep=" ")` в UTC.
- `Duration`: `End - Start` как строка timedelta.
- Числа:
  - целые значения форматируем без `.0`;
  - дробные значения округляем до фиксированного precision (рекомендуем:
    - проценты/экспозиции: 4 знака,
    - ratios: 5 знаков,
    - quote profit: 2 знака) и затем удаляем trailing zeros (`"12.3400" -> "12.34"`).
- Missing/undefined (например, нет trades или недостаточно дневных точек): `N/A`.

Причины:
- UI должен получать стабильные строки для рендера и golden tests;
- формат не должен зависеть от `repr()` pandas/float.

### 6) Политика “не раздувать response”: lazy report + controlled trades payload

В v1 (как минимум):

- `POST /backtests` по умолчанию возвращает только ranking summary (`total_return_pct` + payload).
- Детальный отчёт (`rows/table_md/trades`) строится on-demand через `variant-report`.
- В legacy eager policy trades остаются ограничены `top_trades_n_default`, чтобы payload не разрастался.

Рекомендуемый runtime default:
- `backtest.reporting.top_trades_n_default = 3`.

## Контракты и инварианты

- Метрики считаются только на `target_slice` (warmup исключён).
- Метрики-таблица имеет фиксированный набор и порядок строк.
- Значения в таблице форматируются детерминированно (строки стабильны между запусками).
- Benchmark считается без fee/slippage и использует те же first/last бары `target_slice`.
- Ratios используют 1d resample equity, `risk_free=0`, annualization=365.
- Для grid запуска отчёт строится по explicit `variant-report`; eager mode контролируется
  `backtest.reporting.eager_top_reports_enabled`.

## Связанные файлы

Roadmap/requirements:
- `docs/architecture/roadmap/milestone-4-epics-v1.md` — формулировка BKT-EPIC-06.
- `docs/architecture/roadmap/base_milestone_plan.md` — список метрик v1 и ratios/benchmark семантика.

Backtest docs:
- `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md` — `target_slice` и warmup.
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` — top-K policy.
- `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md` — engine semantics (trades/equity definition).

Backtest code:
- `src/trading/contexts/backtest/application/services/execution_engine_v1.py` — источник trade log и execution semantics.
- `src/trading/contexts/backtest/domain/entities/execution_v1.py` — `TradeV1`/`PositionV1`/`ExecutionOutcomeV1`.
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py` — staged pipeline и deterministic ranking.
- `src/trading/contexts/backtest/application/dto/run_backtest.py` — response DTO (будет расширен для отчёта).
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py` — runtime defaults (будет расширен для `reporting.*`).

## Как проверить

После реализации EPIC-06:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: eager policy может снова сделать sync response тяжёлым на длинных периодах.
  Митигация: default lazy policy (`backtest.reporting.eager_top_reports_enabled=false`).
- Риск: “особые” случаи (нет trades, 0 volatility, короткий период) должны давать детерминированные `N/A`, иначе golden tests будут флапать.
