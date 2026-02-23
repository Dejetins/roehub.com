Ниже — “профессиональный” план разработки **по модулям** под вашу DDD+OOP/SOLID/EO структуру, но простым языком и **в шагах**, с упором на то, чтобы **каждый этап давал работающий вертикальный результат** и сразу учитывал **2 биржи (Binance/Bybit) × 2 рынка (spot/futures)**.

Я буду писать так, как это обычно делает команда: **Milestone → цель → что делаем → что должно работать → критерии готовности (DoD) → ключевые детали и риски**.

---

# Общие принципы (на весь проект)

## P0. “Один домен — много адаптеров”

- Доменные и прикладные части **не знают**, Binance это или Bybit.
    
- Разница “binance/bybit/spot/futures” живёт в:
    
    - `ref_market` (`market_id`)
        
    - адаптерах `adapters/exchanges/*`
        
    - конфигурации подписок
        

## P1. Вертикальные срезы

Каждый этап заканчивается смоук-сценарием:

- **данные → canonical → индикатор → стратегия → backtest → результат**  
    И только потом усложняем (grid, портфель, intrabar, ML, live execution).
    

## P2. Контракты — по необходимости

Контракт (Protocol/ABC) появляется, когда есть:

- **2 реализации** или **2 потребителя**.
    

## P3. Две скорости кода

- “нормальный” код (EO/SOLID) — в `domain/` и `application/`
    
- ускорение — только в `fastpath/` и только после профилирования
    

---

# Milestone 0 — Скелет приложений + окружение (1–3 дня)

### Цель

Все сервисы запускаются, Grafana/Prometheus/Blackbox уже видят базовые health.

### Что делаем

1. `apps/api`:
    

- `/health` (жив ли процесс)
    
- `/ready` (готов ли: есть коннект к PG/CH)
    

2. `apps/worker`:
    

- цикл воркера (пока может выполнять “заглушечную” job)
    
- heartbeat (метрика/лог)
    

3. `apps/scheduler`:
    

- простая периодическая задача “обновить справочники символов” (пока заглушка)
    

4. `infra/monitoring`:
    

- blackbox check API endpoints
    
- базовые Prometheus alerts “API down”, “worker down”
    

### Результат / DoD

- В Docker/на сервере поднимается весь стек.
    
- В Grafana есть NOC dashboard: API up/down, worker up/down.
    

---

# Milestone 1 — Market Data v1: raw → canonical по 2 биржам × 2 рынкам (3–10 дней)

Это фундамент. Без него дальше всё будет шататься.

## 1.1 Контекст: `contexts/market_data`

### Цель

**Минутные свечи** стабильно попадают в ClickHouse:

- raw_binance_klines_1m
    
- raw_bybit_klines_1m  
    и автоматически через MV — в:
    
- canonical_candles_1m
    

### Что делаем (шаги)

1. **Справочник рынков** (`ref_market`):
    

- фиксируем `market_id` для 4 комбинаций:
    
    - 1 binance spot
        
    - 2 binance futures
        
    - 3 bybit spot
        
    - 4 bybit futures
        

2. **Справочник инструментов** (`ref_instruments`):
    

- джоба/ручной скрипт: загрузить инструменты с каждой биржи по рынку
    
- записать `is_tradable`, `status`, optional meta
    

3. **Ингест**:
    

- Binance: источник свечей (лучше WS, fallback REST)
    
- Bybit: источник свечей
    
- на каждый `market_id` можно задать “список подписок” (вначале whitelist)
    

4. **Вставка в raw**:
    

- запись свечей ровно в свою raw-таблицу
    
- обязательные поля: `market_id, symbol, instrument_key, ... , ingested_at`
    

5. **Проверка MV**:
    

- MV должны писать в canonical без ручного dual-write
    

### Что должно работать (смоук)

- Берём 3–5 символов на каждый market_id, включаем ingestion на 10 минут.
    
- Проверяем:
    
    - raw растёт
        
    - canonical растёт
        
    - lag небольшой
        
    - данные читаются одним запросом из canonical
        

### DoD

- Для каждого `market_id` хотя бы 3 символа идут стабильно 30 минут.
    
- Есть базовый runbook: “что делать если canonical не растёт”.
    

### Ключевые детали (важные)

- `instrument_key` = `"exchange:market_type:symbol"` — удобно дебажить.
    
- В canonical держите ключ `(market_id, symbol, ts_open)`.
    
- Дедуп: ReplacingMergeTree(ingested_at) — **нормально**, что дубли могут жить до merge.
    

---

# Milestone 2 — Indicators v1 (3–7 дней)

## Контекст: `contexts/indicators`

### Цель

UI может спросить:

- какие индикаторы есть
    
- какие параметры у каждого (min/max/step)  
    А compute работает по canonical candles.
    

### Что делаем

1. **Реестр индикаторов (domain)**
    

- `IndicatorDef`: имя, версия, список параметров, типы
    

2. **Use case: list_indicators (application)**
    

- отдаёт JSON для UI
    

3. **Use case: compute_indicator**
    

- вход: `market_id, symbol, timeframe(пока 1m), params, window`
    
- читает canonical
    
- считает серию
    

4. **MVP набор индикаторов**
    

- SMA, EMA, RSI, ATR, BBands (5 штук хватает)
    

### Смоук

- API endpoint: `/indicators` и `/indicators/compute`
    
- Для BTCUSDT: получаем серию SMA(20) на последних 1000 свечах
    

### DoD

- Индикаторы детерминированы (один вход → один выход).
    
- По умолчанию всё в чистом Python/Numpy, fastpath позже.
    

---

## Milestone 3 — Identity + Strategy v1 (immutable) + Live runner + Realtime + Telegram

### 3A — Identity v1 (Telegram-only + 2FA + keys storage)
**Цель:** пользователь входит только через Telegram; ключи биржи только после 2FA.

**Что делаем:**
- Telegram auth → `user_id (UUID)` + `telegram_user_id`
- 2FA (TOTP): setup/verify/enforce
- Exchange API keys: хранение + шифрование + гейт 2FA

**DoD:**
- Без Telegram user не существует.
- Без 2FA ключи не добавить.
- Секреты не логируются.

### 3B — Strategy v1 (immutable per-user) + API
**Цель:** пользователь создаёт стратегии (immutable), клонирует как шаблон, запускает run’ы.

**Что делаем:**
- StrategySpecV1 + детерминированное имя
- Run state machine (включая restart из stopped/failed и stopping→run)
- PG persistence: strategies/runs/events
- API endpoints: create/list/get/clone/run/stop/delete
- 422 payloads unified & deterministic

**DoD:**
- Нельзя “изменить стратегию” — только новая.
- Clone работает как шаблон.

### 3C — Live runner worker + rollup + warmup + repair(read)
**Цель:** стратегии получают derived свечи из live 1m и испускают events/metrics.

**Что делаем:**
- worker читает `md.candles.1m.<instrument_key>` через consumer group
- rollup в TF (`1m` как passthrough; `5m/15m/1h/4h/1d` как derived), только closed+full buckets
- warmup_bars вычисляется в runner детерминированно из `spec.indicators` (алгоритм `numeric_max_param_v1`) и фиксируется в metadata run
- warmup seed из ClickHouse canonical 1m
- gap detection + repair(read) из canonical (без запуска ingestion)

**DoD:**
- derived свечи корректны и детерминированы,
- idempotency на дублях/out-of-order,
- checkpoint в PG.

### 3D — Realtime output (UI streams) + Telegram notify
**Цель:** UI и пользователь видят события/метрики.

**Что делаем:**
- Redis Streams:
  - `strategy.metrics.v1.user.<user_id>`
  - `strategy.events.v1.user.<user_id>`
- Telegram notifier best-effort + политики уведомлений

**DoD:**
- UI может подписаться на streams и видеть heartbeat/lag/events.
- Telegram отправляется при подтверждённом chat binding (или log-only в dev).

---

## Milestone 4 — Backtest v1 (close-fill, single instrument; multi-variant grid)

### Цель

Запустить backtest по **одному инструменту** (Binance/Bybit × spot/futures на уровне данных), где:

- сигнал строится из **индикаторов библиотеки**,
- backtest умеет запускать **набор вариантов** (grid) по комбинациям индикаторов и диапазонам их параметров,
- поддерживаются **режимы направления**: `long-only`, `short-only`, `long-short` (с переворотом),
- поддерживаются **4 режима position sizing**,
- есть **close-based SL/TP** (триггер только по `close`, без intrabar),
- есть **комиссии** и **slippage**, настраиваемые из UI (через API параметры),
- результаты детерминированы и воспроизводимы.

### Что делаем

1) **Bounded context `backtest` (v1)**

- Завести контракты домена и application use-case в `src/trading/contexts/backtest/*`.
- Интеграции:
  - Market data → свечи через порт `CanonicalCandleReader` (источник правды: `canonical_candles_1m`).
  - Rollup таймфреймов по правилам shared-kernel (только закрытые и полные бакеты).
  - Indicators → вычисление рядов на базе существующего `indicators` compute:
    - для backtest v1 нужен режим `V>=1` (grid), чтобы считать индикаторы на диапазонах параметров,
    - допускается стратегия “сначала indicator tensors, затем backtest loop”, если укладывается в бюджет.
  - Strategy → в Milestone 4 допускаются 2 режима запуска:
    - (A) backtest “сохранённой стратегии” (по `strategy_id` / `StrategySpecV1`),
    - (B) backtest “ad-hoc grid” из UI: пользователь выбирает набор индикаторов и их диапазоны параметров (по `configs/prod/indicators.yaml`).

2) **Signal engine v1: “signals-from-indicators”**

- StrategySpec использует `indicators[]` как список активных индикаторов (для saved strategy) или input grid (для ad-hoc backtest).
- Для каждого индикатора вычисляется дискретный сигнал на бар:
  - `LONG` | `SHORT` | `NEUTRAL`.
- Финальный сигнал стратегии строится как AND-политика:
  - `final_long = all(indicator_signal == LONG)`
  - `final_short = all(indicator_signal == SHORT)`
  - если индикаторов > 1, то “лонг/шорт” возникает только при согласии всех.
- Коллизии (детерминированно):
  - если `final_long=true` и `final_short=true` на одном баре → `NEUTRAL` (no-trade) + событие/метрика “conflicting_signals”.

3) **Каталог “как индикатор даёт long/short” (v1 контракт)**

- Расширить/зафиксировать правила сигналов на основе формул индикаторов.
- Источник описания формул индикаторов: `docs/architecture/indicators/indicators_formula.yaml`.
- Вариант реализации v1 (зафиксировать одним решением и соблюдать):
  - либо расширить `docs/architecture/indicators/indicators_formula.yaml` секцией `signals:` для каждого `indicator_id`,
  - либо завести отдельный файл `docs/architecture/indicators/indicators_signals.yaml`, но формально ссылаться на outputs из `indicators_formula.yaml`.

Минимальный контракт для signal rule v1:

- правило должно быть вычислимо **по данным одного индикатора** (без сравнений “индикатор A против индикатора B”),
- правило должно быть вычислимо **на закрытии бара** (используем значение ряда на close-бара),
- NaN-политика: если в моменте нет валидного значения (NaN/warmup) → `NEUTRAL`.

4) **Backtest execution model v1 (close-fill)**

- Таймфрейм: `1m` как база; `5m/15m/1h/4h/1d` как derived (строго по rollup).
- На каждом закрытии бара `t`:
  1) обновляем индикаторы на данных до и включая бар `t`,
  2) считаем `final_long/final_short`,
  3) проверяем SL/TP (close-based) для уже открытой позиции,
  4) при необходимости закрываем позицию на **close[t]** с учётом slippage+fee,
  5) затем (если режим разрешает) открываем/переворачиваем позицию на **close[t]** с учётом slippage+fee.

Warmup lookback (v1):

- по умолчанию `warmup_bars_default = 200` (конфигурируемо)
- если истории недостаточно — начинаем с первого доступного бара

5) **Режимы направления (direction modes) v1**

- `long-only`: разрешены только вход/выход в long.
- `short-only`: разрешены только вход/выход в short.
- `long-short`: разрешены long и short, допускается “переворот” (close текущей позиции и open противоположной) на одном баре.

Примечание v1 (чтобы не блокировать spot):

- short в `spot` трактуется как “синтетический short” (как будто маржа доступна), без borrow/funding/liq. Это осознанное упрощение модели исполнения в Milestone 4.

6) **SL/TP close-based v1**

- SL и TP задаются в процентах (например `sl_pct=1.5`, `tp_pct=3.0`).
- SL/TP предполагаются UI-editable (шаг 0.1%), но в backtest v1 передаются как числа (percent) в API.
- Триггер только по `close`:
  - для long: SL если `close <= entry_price * (1 - sl_pct)`; TP если `close >= entry_price * (1 + tp_pct)`
  - для short: SL если `close >= entry_price * (1 + sl_pct)`; TP если `close <= entry_price * (1 - tp_pct)`
- Коллизии (детерминированно):
  - если в одном баре одновременно SL и TP “истинны” (редко на close-based, но возможно при sl_pct=0) → приоритет SL.

7) **Комиссии и slippage (UI-editable параметры)**

- Fee defaults:
  - `spot_fee_pct_default = 0.075%`
  - `futures_fee_pct_default = 0.1%`
  - параметр изменяемый (UI шаг 0.01%)
- Slippage default:
  - `slippage_pct_default = 0.01%`
  - параметр изменяемый (UI шаг 0.01%)

v1 модель применения (фикс):

- slippage применяется к цене fill;
- комиссия применяется к notional fill;
- обе операции должны быть реализованы симметрично для long/short.

8) **Position sizing v1 (4 режима)**

Требование: заложить механику сразу, даже при backtest на одном инструменте.

Предлагаемые имена режимов:

- `all_in` — использовать весь доступный баланс.
- `fixed_quote` — на каждый вход использовать фиксированный notional в quote (например 100 USDT).
- `strategy_compound` (аналог `fixed_quote_strategy`) — стратегия стартует с выделенным бюджетом (например 100 USDT) и компаундит: следующий вход использует весь текущий баланс стратегии (например 110 после прибыли или меньше после убытка).
- `strategy_compound_profit_lock` (аналог `fixed_quote_strategy + fixed_safe_percent`) — как `strategy_compound`, но часть прибыли фиксируется в “safe balance” и больше не используется стратегией.

Ключевая деталь v1: в домене backtest должна появиться “стратегийная бухгалтерия”:

- `strategy_equity_quote`
- `strategy_safe_quote` (только для profit_lock режима)
- `strategy_available_quote = strategy_equity_quote - strategy_safe_quote`

Profit lock policy v1:

- параметр `safe_profit_percent` (например `30.0` = 30%)
- после *закрытия* каждой сделки:
  - считаем `trade_pnl_quote_net` (уже после slippage+fees),
  - если `trade_pnl_quote_net > 0`, то
    - `locked = trade_pnl_quote_net * safe_profit_percent/100`
    - `strategy_safe_quote += locked`
    - `strategy_equity_quote` остаётся как есть (safe — это “заморозка части equity”)
- входы/перевороты используют только `strategy_available_quote`.

9) **API: запуск backtest (sync small) + grid budgets**

- Backtest запускается из UI, но Milestone 4 фиксирует API и синхронный режим только для **малого периода и ограниченного grid**.
- Большие сетки и длительные периоды должны уходить в jobs/progress (Milestone 5).
- Endpoint (v1): `POST /backtests`.
  - режим A (saved): body содержит `strategy_id`
  - режим B (ad-hoc): body содержит grid template (индикаторы + параметры + оси SL/TP)

Runner v1: staged pipeline:

- Stage A: быстрый прогон по базовому grid (без SL/TP) и shortlist `preselect` (конфиг, default 20_000)
- Stage B: расширение shortlist по SL/TP осям и точный расчёт; возвращаем только top-K

Вход (общее):

- `time_range` (`[start, end)` UTC)
- `direction_mode`
- `position_sizing_mode` + параметры режима
- `market_fee_pct` (default по рынку)
- `slippage_pct` (default)

Вход (grid):

- `indicators[]` как список выбранных индикаторов (из библиотеки) + их grid specs
  - диапазоны параметров и шаги берутся из `configs/prod/indicators.yaml` (или из выбранного `configs/<env>/indicators.yaml` на сервере),
  - запрос может сужать диапазоны, но не расширять их за hard-bounds.
- `risk` как grid оси SL/TP (close-based):
  - `sl` и `tp` задаются диапазонами/шагами (step регулируется)
  - SL/TP входят в комбинаторику Stage B мультипликативно
- guards (v1): используем только лимиты `MAX_VARIANTS_PER_COMPUTE_DEFAULT` и `MAX_COMPUTE_BYTES_TOTAL_DEFAULT`.

Выход:

- возвращаем только top-K вариантов (по умолчанию K=300, конфигурируемо), отсортированных по `Total Return [%]` (desc),
- для каждого варианта в top-K возвращается один отчёт-таблица метрик (см. ниже),
- (опционально) возвращаются trades для выбранных N лучших вариантов (чтобы не раздувать response),
- подтверждение воспроизводимости: `spec_hash` (для saved strategy) или `grid_request_hash` (для ad-hoc) + “engine params hash”.

UX/flow v1:

- пользователь может сохранить стратегию только после завершения backtest по всем вариантам
- backtest response должен содержать достаточно данных, чтобы UI мог сохранить выбранный вариант как immutable Strategy (конкретные параметры индикаторов и risk/sizing)

### Отчёт backtest (метрики v1)

Backtest v1 обязан строить отчёт в виде таблицы `|Metric|Value|` со следующими метриками (минимум):

|Metric|Value|
|---|---|
|Start|...|
|End|...|
|Duration|...|
|Init. Cash|...|
|Total Profit|...|
|Total Return [%]|...|
|Benchmark Return [%]|...|
|Position Coverage [%]|...|
|Max. Drawdown [%]|...|
|Avg. Drawdown [%]|...|
|Max. Drawdown Duration|...|
|Avg. Drawdown Duration|...|
|Num. Trades|...|
|Win Rate [%]|...|
|Best Trade [%]|...|
|Worst Trade [%]|...|
|Avg. Trade [%]|...|
|Max. Trade Duration|...|
|Avg. Trade Duration|...|
|Expectancy|...|
|SQN|...|
|Gross Exposure|...|
|Sharpe Ratio|...|
|Sortino Ratio|...|
|Calmar Ratio|...|

Примечания v1:

- `Benchmark Return [%]`: buy-and-hold long по тому же инструменту на том же `time_range`, **без fee/slippage**.
- `Position Coverage [%]`: доля баров, когда позиция была открыта (long или short), относительно общего числа баров.
- `Gross Exposure`: средняя доля капитала в позиции; для `all_in` обычно близко к coverage, для `fixed_quote` может быть меньше.
- `Expectancy`, `SQN`: вычисляются детерминированно по одной зафиксированной методике (в коде).
- `Sharpe Ratio`, `Sortino Ratio`, `Calmar Ratio` (фикс v1):
  - строим equity curve по закрытиям баров,
  - ресэмплим equity в 1d,
  - считаем дневные доходности,
  - `risk_free = 0`,
  - annualization = `365`.

### Результат / DoD

- Backtest детерминирован: одинаковый вход (saved spec или grid request) + одинаковые параметры backtest → одинаковый результат.
- Поддерживаются режимы направления: `long-only`, `short-only`, `long-short` (с переворотом).
- Поддерживаются 4 режима position sizing (включая profit lock) и они дают ожидаемую “бухгалтерию”.
- SL/TP работают в close-based логике и не используют intrabar.
- Комиссия/Slippage применяются и настраиваемы параметрами (с дефолтами: spot 0.075%, futures 0.1%, slippage 0.01%).
- API позволяет запустить backtest синхронно на малом периоде и ограниченном grid (latency приемлемая; есть guards).
- Результаты **не сохраняются** в БД (v1), но возвращаются как response для UI.

### Ключевые детали и риски

- **Сигналы “из любых индикаторов”**: требование покрыть сигнал-правила для всего списка `configs/prod/indicators.yaml` может существенно расширить объём Milestone 4.
- **Сигнальный DSL**: если правила long/short будут слишком “богатые” (сравнения нескольких outputs, окна, кроссы), быстро получится полноценный DSL — это риск scope creep.
- **Lookahead**: исполнение на `close[t]` при сигнале на `close[t]` допустимо, но нужно фиксировать порядок “exit → entry” и поведение при конфликте long/short.
- **Spot vs futures**: модель исполнения одинаковая (требование), но нужно явно зафиксировать, что funding/ликвидации/маржинальные ограничения — не делаем в v1.

### Что это меняет в roadmap (важно)

Фактически Milestone 4 включает **grid backtest**. Чтобы не перегрузить синхронный API:

- Milestone 4: sync small с жёсткими guards по числу вариантов и длине периода.
- Milestone 5: jobs/progress обязательны для больших сеток.
- Milestone 7: top-k / pruning / batching — отдельный слой оптимизации.

### Кандидаты на вынос в следующий milestone (если не влезает в Milestone 4)

- Полное покрытие signal rules для всех индикаторов из `configs/prod/indicators.yaml`.
- Расширенные правила сигналов (кроссы между индикаторами, сравнения A vs B, сложные композиции) — отдельный milestone как “Signal DSL v2”.
- Сохранение результатов backtest, история запусков, прогресс и “тяжёлые” задачи — Milestone 5 (jobs).

---

# Milestone 5 -- Backtest Jobs v1

---

## Цель

Сделать backtest модуль масштабируемым для одновременных запусков несколькими пользователями:

- большие расчеты не блокируют UI и HTTP (async),
- есть прогресс выполнения и "best-so-far" top результаты во время расчета,
- результаты сохраняются в Postgres и доступны после завершения,
- можно отменить (cancel) выполняющийся backtest,
- система устойчиво переживает рестарт job-runner (минимальный resume).

---

## Контекст

Milestone 4 ввел синхронный (small-run) API `POST /backtests`:
- staged pipeline (Stage A shortlist -> Stage B exact -> top-K),
- deterministic ranking + tie-break,
- отчет `report.table_md` для каждого варианта в top-K,
- trades возвращаются только для `top_trades_n`.

Синхронный режим намеренно ограничен guards и не подходит для больших сеток/длинных периодов.
Milestone 5 добавляет асинхронный путь, который масштабируется через несколько реплик воркера.

Связанные документы:
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md` (sync API, hashes, trades policy)
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` (Stage A/Stage B, guards, tie-break)
- `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md` (table_md + top_trades_n)

---

## Зафиксированные решения Milestone 5

### 1) Jobs делаем только для backtest (не generic platform jobs)

- Все сущности/таблицы/воркер/эндпойнты принадлежат bounded context `backtest`.
- В Milestone 5 не вводим общий "jobs" модуль для других контекстов.

### 2) Persisted output policy v1

- Храним `report_table_md` для всех persisted top вариантов.
- Trades храним/возвращаем только для `top_trades_n` лучших вариантов (как в sync API).

### 3) Cancel входит в Milestone 5

- API позволяет запросить отмену.
- Cancel best-effort: job-runner обязан проверять `cancel_requested_at` на границах батчей и завершать job в состоянии `cancelled`.

---

## Scope

### A) Storage (Postgres) для backtest jobs и результатов

Вводим новую модель хранения (через Alembic миграции) в том же Postgres, где уже живут таблицы Strategy.

Минимальный набор таблиц (v1):

1) `backtest_jobs`
- identity: `job_id (uuid)`, `user_id (uuid)`
- lifecycle:
  - `state`: `queued|running|succeeded|failed|cancelled`
  - `created_at`, `updated_at`
  - `started_at`, `finished_at`
  - `cancel_requested_at`
  - `last_error` (short string) + `last_error_json` (optional)
- request payload:
  - `mode`: `saved|template` (денормализация для удобства list/filter)
  - `request_json` (канонизированный payload v1)
  - `request_hash` (sha256 canonical json)
  - `engine_params_hash` (как в sync), плюс (опционально) `backtest_runtime_config_hash`
- progress:
  - `stage`: `stage_a|stage_b|finalizing` (строго фиксированный enum/strings)
  - `processed_units`, `total_units` (int)
  - `progress_updated_at`
- concurrency/lease:
  - `locked_by` (string, например `<hostname>-<pid>`)
  - `locked_at`
  - `lease_expires_at`
  - `heartbeat_at`
  - `attempt` (int, инкремент при каждом reclaim)

2) `backtest_job_top_variants`
- `job_id`, `rank` (1..K)
- variant identity:
  - `variant_key`
  - `indicator_variant_key` (если требуется для дебага/кешей)
  - `variant_index` (как в sync, если нужно)
- ranking:
  - `total_return_pct` (primary key for ordering)
  - tie-break фиксируется как `variant_key` (asc)
- persisted report:
  - `report_table_md` (ASCII markdown)
  - `trades_json` (NULL для rank > top_trades_n)
- payload для сохранения выбранного варианта:
  - `payload_json` (explicit selections: indicators/signals/risk/execution/direction/sizing)

3) (минимальный resume) `backtest_job_stage_a_shortlist`
- `job_id`
- `shortlist_json` (deterministic list of shortlisted base variants, достаточно для запуска Stage B)

Примечание:
- v1 не обязан хранить equity curve для всех top-500; при необходимости это расширение будущего milestone.

### B) Backtest job-runner worker (многорепличный)

Добавляем новый воркер (entrypoint) `backtest-job-runner`, который:

- забирает jobs из `backtest_jobs` (claim) атомарно через row-level lock:
  - `SELECT ... FOR UPDATE SKIP LOCKED`
  - меняет `state` в `running`, выставляет `locked_by`, `lease_expires_at`, `attempt += 1`
- выполняет расчет батчами и пишет прогресс:
  - Stage A: посчитать base grid без SL/TP, сохранить shortlist
  - Stage B: расширить shortlist по risk осям и посчитать точный score/report
  - поддерживать running top-K heap (K=500, tie-break по `variant_key`)
  - периодически сохранять persisted top-K snapshot (транзакционно)
- поддерживает cancel:
  - перед каждым батчем проверяет `cancel_requested_at`
  - при cancel корректно завершает job как `cancelled` (без "failed")
- поддерживает minimal resume:
  - если воркер умер, lease истекает, другой воркер может reclaim job
  - v1 допускает restart job c начала attempt (детерминированно), но job не должен застревать навсегда в `running`

Ожидаемая масштабируемость:
- несколько реплик воркера обрабатывают разные job'ы параллельно;
- один job в один момент времени исполняется максимум одним воркером.

### C) API: backtest jobs endpoints (async)

Синхронный `POST /backtests` (Milestone 4) сохраняется.
Добавляем отдельные endpoints для jobs (Milestone 5):

1) `POST /backtests/jobs`
- создает job для saved или template режима (request envelope как в `POST /backtests`)
- валидирует:
  - ownership (для saved),
  - guards/лимиты, которые можно проверить до запуска,
  - квоту `max_active_jobs_per_user`.
- response: `{job_id, state, request_hash, engine_params_hash}`

2) `GET /backtests/jobs/{job_id}`
- status + progress + timestamps + hashes.

3) `GET /backtests/jobs/{job_id}/top?limit=500`
- возвращает persisted top результаты (best-so-far во время running; финальные после succeeded/cancelled/failed).
- ordering фикс:
  - `total_return_pct` desc,
  - tie-break `variant_key` asc.

4) `GET /backtests/jobs?state=&limit=&cursor=`
- список "моих" job'ов (owner only) с пагинацией.

5) `POST /backtests/jobs/{job_id}/cancel`
- ставит `cancel_requested_at` (idempotent),
- state transitions:
  - `queued -> cancelled` (если job еще не claimed)
  - `running -> cancelled` (best-effort, при следующей проверке воркером)

Ошибки:
- используем общий контракт `RoehubError` и deterministic 422 payload (как в sync API).
- ownership/visibility проверяем в use-case.

### D) Runtime config (backtest.yaml)

Расширяем `configs/<env>/backtest.yaml` секцией jobs:

- `backtest.jobs.enabled` (toggle)
- `backtest.jobs.top_k_persisted_default` (default 500)
- `backtest.jobs.max_active_jobs_per_user` (default, например 2)
- `backtest.jobs.claim_poll_seconds` (default, например 1)
- `backtest.jobs.lease_seconds` (default, например 60)
- `backtest.jobs.heartbeat_seconds` (default, например 5)

Важно:
- defaults `top_trades_n_default` остаются в `backtest.reporting` (Milestone 4), job-runner использует их же.

### E) Observability + runbook

- Prometheus метрики job-runner: jobs claimed, active, succeeded/failed/cancelled, durations, lease lost, cancels.
- Runbook: запуск/масштабирование воркера, диагностика stuck jobs, проверка API.

---

## Что должно работать (смоук)

1) Два пользователя создают по job через `POST /backtests/jobs` и получают разные `job_id`.
2) Запущены 2-4 реплики `backtest-job-runner`:
   - каждый job claimed ровно одним воркером,
   - jobs считаются параллельно.
3) UI опрашивает `GET /backtests/jobs/{job_id}` и видит прогресс (stage + processed/total).
4) Во время `running` UI получает `GET /backtests/jobs/{job_id}/top` и видит best-so-far результаты.
5) Cancel:
   - UI вызывает `POST /backtests/jobs/{job_id}/cancel`,
   - job завершается как `cancelled`, дальнейшие вычисления прекращаются.
6) Рестарт воркера:
   - если воркер умер во время `running`, job reclaim'ится после истечения lease и в итоге завершается (минимально допустим перезапуск attempt).

---

## DoD

- Async flow реализован: create/status/top/list/cancel.
- Масштабирование через несколько реплик job-runner работает без гонок (row-lock + lease).
- Persisted top-K:
  - хранится как минимум top-500,
  - для всех persisted вариантов хранится `report_table_md`,
  - trades присутствуют только для `top_trades_n` лучших.
- Детерминизм сохранен:
  - один и тот же request + одинаковые runtime defaults -> одинаковые persisted результаты (top ordering + table_md).
- Minimal resume:
  - job не застревает в `running` навсегда,
  - lease/reclaim работает, итоговые данные консистентны.

---

## Кандидаты EPIC'ов (Milestone 5)

Нумерация продолжает Milestone 4 (`BKT-EPIC-01..08`).

1) BKT-EPIC-09 -- Backtest Jobs storage v1 (PG schema + repositories + state machine)
- Alembic миграции + Postgres adapters.
- Domain errors/invariants для job state + ownership.

2) BKT-EPIC-10 -- Backtest job-runner worker v1 (claim/lease/heartbeat + batching + cancel)
- Новый воркер + wiring + метрики.
- Persist progress и top-500 snapshots.

3) BKT-EPIC-11 -- Backtest Jobs API v1
- `POST /backtests/jobs`, `GET status/top/list`, `POST cancel`.
- Unified deterministic errors.

4) BKT-EPIC-12 -- Tests + runbook
- Unit/integration тесты job state, claim/reclaim, cancel.
- Runbook и docs index updates.

---

## Риски и открытые вопросы

- Retention: без политики очистки `backtest_job_top_variants` объем БД будет расти (в v1 можно оставить как риск, а policy сделать отдельным эпиком).
- Write amplification: частые snapshot update top-500 могут нагрузить PG; v1 должен писать батчами (по времени или по N батчей).
- Resume semantics: v1 допускает restart attempt с начала (дороже по CPU), но это проще и надежнее; более тонкий checkpoint Stage B можно вынести в следующий milestone.
---


## Milestone 6 — Web UI v1 (Backtest + Jobs + Strategy + Auth)

**Цель:** можно начать строить и развивать продукт через браузерный UI, используя уже реализованные системы (identity/strategy/backtest/backtest-jobs) без CORS, с same-origin доставкой через reverse-proxy.

**Ключевая топология (фикс):**

- один origin: `https://roehub.com`
- gateway (Nginx) маршрутизирует:
  - UI: `/` → web UI
  - API: `/api/*` → Roehub API (с strip префикса `/api`)
- UI **всегда** вызывает JSON API только по `/api/...` (cookie auth, `credentials=include`).
  HTML страницы UI обслуживаются web UI роутами без `/api`.

**Что делаем:**

- Web UI app:
  - `apps/web` (Python SSR + Jinja2 + HTMX; без React/SPA),
  - сборка статических ассетов в `apps/web/dist` (CSS/JS),
  - UI интегрирует Telegram Login Widget (Variant A) и опирается на HttpOnly JWT cookie.

- Reverse proxy:
  - Nginx gateway в `infra/docker` (и/или отдельный prod конфиг),
  - deterministic routing: `/api/*` всегда уходит на API upstream.
  - default routing: всё остальное (UI HTML) уходит на web upstream; статические ассеты отдаёт gateway из `apps/web/dist`.

- Reference data для выбора инструментов в UI:
  - добавить auth-only API endpoints, которые читают ClickHouse `market_data.ref_market` и `market_data.ref_instruments`:
    - `GET /market-data/markets` — список доступных рынков (market_id, exchange_name, market_type)
    - `GET /market-data/instruments?market_id=&q=&limit=` — поиск/листинг `symbol` (только enabled+tradable)
  - ordering детерминированный (для UI и тестов).

- UI flows v1:
  - identity: login/logout/current-user;
  - strategy: list/get/create/clone + "Save variant as Strategy" из результатов backtest.
    - create UI = визуальный builder (market/symbol/timeframe + выбор индикаторов/параметров по `/api/indicators`), без JSON textarea.
    - "Save variant" сохраняет StrategySpec v1 только в пределах текущего доменного контракта (индикаторы + instrument/timeframe tags); risk/execution/direction/sizing остаются настройками backtest.
  - backtest sync: `POST /backtests` (small-run) + отображение top-K (`report_table_md`, trades только для `top_trades_n`);
  - backtest jobs: create/list/status/top/cancel, polling прогресса и best-so-far top во время `running`.
    Важно для UX: при reclaim attempt возможны сброс `stage/progress` и временно stale `/top` snapshot до первой перезаписи.

- Ops/runbooks:
  - runbook запуска web+api+nginx локально;
  - runbook настройки Telegram (allowed domains) для dev/prod.
  - обеспечить применение Alembic миграций (strategy/backtest/backtest-jobs) перед стартом сервисов через migrations runner (`POSTGRES_DSN`).

**DoD:**

- Пользователь может сделать end-to-end сценарий в UI:
  - login → выбрать market/symbol/timeframe → создать/клонировать strategy → запустить backtest (sync или jobs) → увидеть top результаты (таблица+trades) → сохранить вариант как Strategy → cancel job при необходимости.
- Один origin работает в dev/prod через Nginx gateway, без CORS.
- CSRF-стратегию для mutating endpoints сознательно откладываем (same-origin + `SameSite=lax` в v1).

---

## Milestone 7 — Optimize/Grid + Pruning

**Цель:** сотни тысяч комбинаций параметров → top-500 без OOM.

**Что делаем:**
- lazy grid generator
- batching 1k–10k
- pruning (maxDD/loss-window/limit trades)
- streaming top-k

**DoD:**
- 100k+ комбинаций не падают по памяти,
- валидационные лимиты не дают убить воркера.

---

## Milestone 8 — Backtest v2: intrabar + portfolio + risk

**Цель:** более реалистичная модель исполнения в OHLC + портфель стратегий.

**Что делаем:**
- intrabar fills (touch high/low)
- portfolio engine
- risk sizing, портфельные лимиты

**DoD:**
- портфель не ломает single-strategy pipeline,
- результаты стабильны в рамках модели.

---

## Milestone 9 — ML каркас (optional/parallel)

**Цель:** ML как источник сигналов, совместимый со spec/backtest/jobs.

**Что делаем:**
- feature registry + dataset builder (leakage guards)
- model registry + inference (score→signal)

**DoD:**
- backtest/optimize не требуют переделки.

---

## Milestone 10 — Live execution (контуры/контракты, реализация позже)

**Цель:** заложить правильные границы, не делая реальный трейдинг.

**Что закладываем:**
- execution gateway contracts (place/cancel/reconcile)
- модели ордера/исполнений + идемпотентность

**Что не делаем:**
- реальный трейдинг, ордер-менеджмент, управление риском на бирже

---

## Сквозные инварианты (must-have)
- Identity инструмента: `(market_id, symbol)`; `instrument_key` только trace/debug.
- Хранение свечей: только `canonical_candles_1m`; все TF derived через rollup.
- Derived candles: только closed+full buckets.
- Strategy immutable: любые изменения = новая стратегия.
- Keys: только после 2FA; секреты всегда шифруются и не логируются.
- Best-effort Telegram/Redis publish: ошибки не ломают ingestion/runner.

--- 
