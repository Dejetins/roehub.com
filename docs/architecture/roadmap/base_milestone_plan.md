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
- Milestone 6: top-k / pruning / batching — отдельный слой оптимизации.

### Кандидаты на вынос в следующий milestone (если не влезает в Milestone 4)

- Полное покрытие signal rules для всех индикаторов из `configs/prod/indicators.yaml`.
- Расширенные правила сигналов (кроссы между индикаторами, сравнения A vs B, сложные композиции) — отдельный milestone как “Signal DSL v2”.
- Сохранение результатов backtest, история запусков, прогресс и “тяжёлые” задачи — Milestone 5 (jobs).

---

## Milestone 5 — Jobs + Progress + Top-K

**Цель:** большие расчёты не блокируют UI; есть прогресс и top-результаты.

**Что делаем:**
- job модель в PG + heartbeat/progress
- worker job-runner (батчи)
- API: create/status/progress/top-500

**DoD:**
- resume после перезапуска worker (минимально).

---

## Milestone 6 — Optimize/Grid + Pruning

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

## Milestone 7 — Backtest v2: intrabar + portfolio + risk

**Цель:** более реалистичная модель исполнения в OHLC + портфель стратегий.

**Что делаем:**
- intrabar fills (touch high/low)
- portfolio engine
- risk sizing, портфельные лимиты

**DoD:**
- портфель не ломает single-strategy pipeline,
- результаты стабильны в рамках модели.

---

## Milestone 8 — ML каркас (optional/parallel)

**Цель:** ML как источник сигналов, совместимый со spec/backtest/jobs.

**Что делаем:**
- feature registry + dataset builder (leakage guards)
- model registry + inference (score→signal)

**DoD:**
- backtest/optimize не требуют переделки.

---

## Milestone 9 — Live execution (контуры/контракты, реализация позже)

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
