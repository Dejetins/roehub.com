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

## Milestone 4 — Backtest v1 (close-fill, single strategy)

**Цель:** запустить бектест одной immutable-стратегии по одному инструменту.

**Что делаем:**
- close-fill модель + комиссии + простой slippage
- отчёт: pnl, dd, trades, winrate
- API: запуск backtest (синхронно на малом периоде)

**DoD:**
- результаты воспроизводимы (spec фиксируется),
- время ответа приемлемо на “маленьком тесте”.

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
