# Backtest v1 — Execution Engine (Close-Fill) + Fees/Slippage + Sizing + Close-Based SL/TP (BKT-EPIC-05)

Фиксирует контракт BKT-EPIC-05: детерминированное исполнение сделок по close (close-fill) по сигналам `LONG|SHORT|NEUTRAL` с direction modes, position sizing (4 режима), close-based SL/TP, fees/slippage и staged semantics (Stage A без SL/TP, Stage B с SL/TP).

## Цель

- Реализовать backtest execution engine v1, который по одному и тому же входу даёт один и тот же trade log и equity результат.
- Зафиксировать правила close-fill исполнения:
  - порядок событий на одном баре (risk exit -> signal exit/entry -> forced close),
  - reversal на одном `close[t]` (exit -> entry) в `long-short`.
- Зафиксировать accounting v1 для long/short в упрощённой (spot/futures одинаково) модели без leverage/funding/liq.

## Контекст

- Candle timeline (BKT-EPIC-02) уже фиксирует:
  - canonical candles как source of truth,
  - best-effort rollup,
  - warmup lookback,
  - `target_slice` по правилу `Start <= bar_close_ts < End`.
- Signals-from-indicators v1 (BKT-EPIC-03) уже фиксирует:
  - per-indicator `LONG|SHORT|NEUTRAL`,
  - AND aggregation,
  - NaN/warmup => `NEUTRAL`.
- Grid + staged runner + guards (BKT-EPIC-04) уже фиксирует:
  - Stage A без SL/TP (shortlist `preselect`),
  - Stage B с SL/TP (top-K),
  - guards (600k variants, 5 GiB),
  - ranking key: `Total Return [%]`.

Этот документ фиксирует именно execution engine (трейды/позиции/бухгалтерия). Отчёт метрик и форматирование таблицы метрик — EPIC-06.

## Scope

- Close-fill loop:
  - исполнение действий на `close[t]`;
  - допускается `exit -> entry` на одном баре (включая reversal, если разрешено direction_mode).
- Direction modes v1:
  - `long-only`, `short-only`, `long-short`.
- Position constraint v1 (фикс):
  - в каждый момент времени открыта максимум 1 позиция;
  - новая позиция открывается только после закрытия предыдущей;
  - допускается `exit -> entry` на одном `close[t]`.
- Entry gating v1 (фикс):
  - новый вход разрешён только по **новому сигналу** (edge), старый “persisted” сигнал не должен открывать сделки повторно.
- Формат входного сигнала:
  - engine принимает legacy `LONG|SHORT|NEUTRAL` и compact `np.int8` (`0/1/-1`);
  - в hot loop используется compact-представление без per-bar `str(...)` нормализации.
- Sizing v1 (4 режима):
  - `all_in`
  - `fixed_quote` (в v1: capped by available: `min(available, fixed_quote)`)
  - `strategy_compound`
  - `strategy_compound_profit_lock` (параметр `safe_profit_percent`)
- Fees/slippage v1:
  - slippage применяется к fill price,
  - fee применяется на entry и exit,
  - reversal на одном баре = 2 операции = 2 комиссии, slippage на обоих fills.
- SL/TP close-based v1:
  - триггер на `close[t]`, без intrabar,
  - при одновременном SL и TP — приоритет SL.
- Параллельное выполнение по вариантам на CPU (без CUDA) допускается и должно сохранять детерминизм результата.

## Non-goals

- Intrabar fills (touch high/low), OHLC-based execution.
- Leverage, funding, liquidation, borrow/fees modelling.
- Биржевые лоты, tick size, округления quantity.
- Jobs/progress и сохранение результатов (Milestone 5).

## Ключевые решения

### 1) Торгуем и считаем доходность только на `target_slice` (warmup только для compute)

Backtest engine исполняет сделки и считает `Total Return [%]` только на бар-срезе `target_slice`:

- warmup бары присутствуют в `candles` для индикаторов/сигналов,
- но торговля начинается с первого бара `target_slice` и заканчивается последним баром `target_slice`.

Причина:
- метрики/ранжирование не должны “размываться” warmup участком.

Последствия:
- execution engine (или scorer) должен получать `target_slice` явно (не только `candles`).

Связанные файлы:
- `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`
- `src/trading/contexts/backtest/application/services/candle_timeline_builder.py`

### 2) Desired position semantics + entry gating по edge (новый сигнал)

Фиксируем семантику сигналов:

- `LONG` => хотим быть в long
- `SHORT` => хотим быть в short
- `NEUTRAL` => хотим быть flat

Но вход делаем только по **edge** (новый сигнал):

- `enter_long[t] = (final_signal[t] == LONG) and (prev_final_signal != LONG)`
- `enter_short[t] = (final_signal[t] == SHORT) and (prev_final_signal != SHORT)`

Где `prev_final_signal` — значение `final_signal` на предыдущем баре в `target_slice`.
На первом баре `target_slice` считаем `prev_final_signal = NEUTRAL`.

Это решает фиксированное требование v1:
- если позиция закрылась по SL/TP (или по `NEUTRAL`/exit-only), и сигнал остаётся тем же уровнем (persisted), мы НЕ должны открывать новую позицию повторно.

Пример:
- t=10: сигнал стал `LONG` (edge) -> открыли long.
- t=25: сработал SL, сигнал всё ещё `LONG` -> закрыли позицию, но НЕ переоткрываем long.
- t=40: сигнал стал `NEUTRAL`.
- t=41: сигнал снова `LONG` (edge) -> открываем long.

Последствия:
- trade log должен хранить `entry_signal_bar_index` (и/или `entry_signal`) для аудита и дебага.

Связанные файлы:
- `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`

### 3) Direction modes: “запрещённый” сигнал = exit-only

В `long-only`/`short-only` сигнал противоположного направления трактуем как **exit-only**:

- `long-only`: `SHORT` может закрыть long, но не может открыть short.
- `short-only`: `LONG` может закрыть short, но не может открыть long.

Причина:
- ожидаемое UX поведение: “запрещённый” сигнал не игнорируется полностью, но и не создаёт запрещённые сделки.

### 4) Forced close на последнем баре `target_slice`

Если позиция всё ещё открыта на последнем баре `target_slice`, принудительно закрываем её на `close[last]` с применением slippage и exit fee.

Причина:
- детерминированный финал equity и корректный `Total Return [%]` для сравнения вариантов.

### 5) Close-based SL/TP: триггеры по `close[t]`, fill по slippage, приоритет SL

Триггеры проверяем по raw `close[t]`, уровни считаем от `entry_fill_price`:

- long:
  - SL: `close[t] <= entry_fill_price * (1 - sl_pct/100)`
  - TP: `close[t] >= entry_fill_price * (1 + tp_pct/100)`
- short:
  - SL: `close[t] >= entry_fill_price * (1 + sl_pct/100)`
  - TP: `close[t] <= entry_fill_price * (1 - tp_pct/100)`

Если одновременно истинны SL и TP (возможно при `sl_pct=0`), приоритет SL.

### 6) Fee/slippage semantics (фикс v1)

Percent units:
- все проценты — в “процентных пунктах” (human percent), например:
  - `fee_pct = 0.075` означает `0.075%`
  - `slippage_pct = 0.01` означает `0.01%`
  - `sl_pct = 3.0` означает `3%`

Fill price:
- buy fill: `price_fill = close * (1 + slippage_pct/100)`
- sell fill: `price_fill = close * (1 - slippage_pct/100)`

Fee:
- берём комиссию на entry и exit,
- рекомендуемая v1 формула: `fee_quote = abs(qty_base) * price_fill * (fee_pct/100)`.

Reversal на одном баре:
- exit + entry = 2 операции => 2 комиссии,
- slippage применяется к каждому fill.

### 7) Accounting v1: “synthetic/margin-style” short, единообразно для spot/futures

Short в v1 трактуется как синтетический short (как будто маржа доступна), без borrow/funding/liq.

Для детерминизма и симметрии фиксируем accounting:

- стратегия хранит два баланса:
  - `strategy_available_quote`
  - `strategy_safe_quote` (только для profit_lock режима)
- при входе резервируем budget из `strategy_available_quote` (margin-style) и открываем позицию количеством `qty_base`.
- equity в момент времени — детерминированная функция состояния позиции и `close[t]`.

Причина:
- избегаем “cash explosion” модели short-sale и сохраняем симметрию long/short.

### 8) Sizing modes v1: один стартовый параметр `init_cash_quote`

Фиксируем v1: один параметр стартового капитала `init_cash_quote`.

- `all_in` и `strategy_compound` в single-strategy backtest v1 фактически совпадают (оба используют весь `strategy_available_quote`).
- `fixed_quote`: используем `min(available, fixed_quote)`.
- `strategy_compound_profit_lock`: после прибыльной сделки lock’аем часть прибыли в safe.

Profit lock policy v1:

- `safe_profit_percent` (например `30.0` = 30%)
- после закрытия сделки, если `trade_pnl_quote_net > 0`:
  - `locked = trade_pnl_quote_net * safe_profit_percent/100`
  - `strategy_safe_quote += locked`
  - `strategy_available_quote -= locked`

Важно:
- lock выполняется сразу после exit и ДО возможного entry на том же `close[t]` (reversal), чтобы новый entry использовал уже “урезанный” available.

### 9) Дефолты fee/slippage/sizing — source-of-truth в `configs/<env>/backtest.yaml`

Дефолтные параметры исполнения должны приходить из runtime config backtest (fail-fast), а request может их override.

Минимально фиксируем значения v1 (source: `docs/architecture/roadmap/milestone-4-epics-v1.md` и `docs/architecture/roadmap/base_milestone_plan.md`):

- fee defaults: spot `0.075%`, futures `0.1%`
- slippage default: `0.01%`

Рекомендуемая схема ключей в YAML (v1):

```yaml
version: 1
backtest:
  warmup_bars_default: 200
  top_k_default: 300
  preselect_default: 20000
  execution:
    init_cash_quote_default: 10000
    fixed_quote_default: 100
    safe_profit_percent_default: 30.0
    slippage_pct_default: 0.01
    fee_pct_default_by_market_id:
      1: 0.075
      2: 0.1
      3: 0.075
      4: 0.1
```

Причина:
- backtest не должен зависеть от “market_type” как модели другого контекста; market_id уже является стабильным идентификатором.

### 10) Compact signal contract в execution hot path

Для perf-пайплайна фиксируем:

- канонические коды: `NEUTRAL=0`, `LONG=1`, `SHORT=-1`;
- нормализация legacy сигналов делается один раз до цикла исполнения;
- внутри per-bar цикла используется только `np.int8` код, чтобы убрать строковые преобразования из hot path;
- `Total Return [%]` и trade ordering должны оставаться эквивалентными legacy path.

## Контракты и инварианты

- Determinism:
  - один и тот же вход (`candles`, `signals`, `execution_params`, `risk_params`, `direction_mode`, `sizing_mode`) => один и тот же trades/equity.
- Trade order:
  - на одном баре порядок: risk exit -> signal exit/entry -> forced close (если last bar).
  - reversal (если разрешено) выполняется как exit -> entry на одном `close[t]`.
- Entry gating:
  - entry выполняется только по edge (новому сигналу) внутри `target_slice`.
- Direction modes:
  - `long-only`/`short-only`: противоположный сигнал = exit-only.
- SL/TP:
  - close-based, intrabar не используем.
  - при SL и TP одновременно: SL.
- Percent units:
  - все `*_pct` параметры приходят как “проценты” (human percent) и в формулах делятся на 100.

## Связанные файлы

Документы:
- `docs/architecture/roadmap/milestone-4-epics-v1.md` — BKT-EPIC-05 scope и фиксированные семантики fee/slippage.
- `docs/architecture/roadmap/base_milestone_plan.md` — детальная спецификация execution/sizing/profit-lock.
- `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md` — `target_slice` и warmup policy.
- `docs/architecture/backtest/backtest-signals-from-indicators-v1.md` — `LONG|SHORT|NEUTRAL` и AND aggregation.
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` — Stage A/Stage B semantics и ranking key.

Код (существующий):
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py` — staged orchestration (Stage A/Stage B) и ranking по `Total Return [%]`.
- `src/trading/contexts/backtest/application/services/candle_timeline_builder.py` — нормализация диапазона + rollup + `target_slice`.
- `src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py` — signal engine v1.
- `src/trading/contexts/backtest/application/dto/run_backtest.py` — direction/sizing literals и risk grid contracts.
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py` — `variant_key` (включая `signals`/risk/execution).

Код (будет добавлен в реализации EPIC-05):
- `src/trading/contexts/backtest/domain/entities/` — `PositionV1`, `TradeV1`, `AccountStateV1`.
- `src/trading/contexts/backtest/application/services/execution_engine_v1.py` — close-fill execution engine v1.
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py` — реализация `BacktestStagedVariantScorer` на базе engine.
- `configs/<env>/backtest.yaml` — расширение секцией `backtest.execution`.

## Как проверить

После реализации EPIC-05:

```bash
# запускать из корня репозитория
uv run ruff check .
uv run pyright
uv run pytest -q

# если добавлялся/менялся этот документ
python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: performance. Чисто питоновый per-variant loop может быть медленным на 100k+ вариантах; в v1 допускаем параллелизм по CPU и/или Numba-fastpath, но сохраняем детерминизм через stable сортировку результатов.
- Риск: memory. Хранить equity curve/trades для всех Stage B вариантов нельзя; v1 должен хранить полный результат только для top-K (подробный отчёт — EPIC-06).
