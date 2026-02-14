

---

## Требования Milestone 2: Indicators (обновлено)

### 1) Цель и границы

**Цель:** сделать контекст `indicators`, который:

- хранит **библиотеку индикаторов** (registry) с параметрами и **hard bounds**,
    
- позволяет UI получить **полное описание** индикаторов/параметров/диапазонов,
    
- умеет считать индикаторы **матрицами/тензорами** для сетки параметров **на CPU** через **Numba**,
    
- умеет **оценивать размер задачи ДО запуска** (комбинации + память) для всего пользовательского батча,
    
- работает на данных `canonical_candles_*` через ACL (market_data → numpy arrays).
    

**В Milestone 2 не делаем:**

- бектест/исполнение стратегий, staged A/B, top’ы стратегий, джобы, очереди — только индикаторы и оценка бюджета (но закладываем совместимость).
    

---

### 2) Данные на входе (candles)

- Источник: `canonical_candles_*` (через ACL-порт в indicators).
    
- На чтении строим **плотную временную сетку** (uniform timeline) и **вставляем NaN** в пропуски.
    
- Пропуски **не лечим** (не интерполируем), только NaN.
    
- Выход ACL: плотные `np.ndarray` (contiguous) `open/high/low/close/volume` + `ts_open[]`.
    

---

### 3) Индикаторы и сетки параметров

**Каждый индикатор** определяется через `IndicatorDef`:

- `indicator_id` (уникальный; формулы не меняем — новая формула = новый `indicator_id`)
    
- входы (какие series нужны: close/high/low/volume и т.п.)
    
- параметры:
    
    - тип: `int / float / enum`
        
    - **hard_min/hard_max**
        
    - **step/precision задаётся в YAML конфиге** (`configs/*/indicators.yaml`) для каждого параметра/оси, и валидируется против hard bounds
        
    - дефолтные значения/диапазоны для UI (в YAML)
        
    - режим задания: **explicit values** и/или **range(start/stop/step)**
        
- оси тензора (axis definitions), например:
    
    - `SMA`: `window × source`
        
    - `MACD`: `fast × slow × signal × source`
        

**Guard (фикс):**

- `max_variants_per_compute = 600_000` (по умолчанию, конфигурируемо) — **публичный лимит по числу комбинаций для всего батча**.
    
- `max_compute_bytes_total = 5 GiB` (по умолчанию, конфигурируемо) — **публичный лимит по оценке памяти для всего батча**.
    
- Если `variants > max_variants_per_compute` **или** `memory_bytes > max_compute_bytes_total` → 422/validation error (UI должен сузить диапазоны/изменить таймфрейм/период).
    

**Важно (обновлённый подход):**

- Оценка (`/estimate`) делается **по всему батчу сразу** (несколько индикаторов + параметры риск-менеджмента).
    
- В “комбинации” мы также учитываем **SL/TP** (по заданному шагу в конфиге), чтобы заранее отсечь конфиги, которые дальше сделают бектест невозможным по бюджету. При этом **сам бектест в Milestone 2 не реализуем**.
    

---

### 4) Выход вычисления

Результат индикатора возвращается как `IndicatorTensor`:

- `values: np.ndarray[float32]`
    
- `axes: list[AxisDef]` (имя оси, тип, значения оси)
    
- `layout: enum` (`TIME_MAJOR` / `VARIANT_MAJOR`)
    
- `meta`: сведения о warmup, времени расчёта, размере тензора, NaN policy.
    

**Точность:**

- выход `float32` (фикс),
    
- внутри kernels допускается `float64` для сумм/накоплений,
    
- `fastmath=True`, `parallel=True` — разрешено.
    

---

### 5) Layout (решение через бенчмарк)

Пока **не фиксируем**, делаем измерение на ваших данных:

- поддерживаем оба layout-а (`T×V` и `V×T`),
    
- в одном EPIC делаем бенч и по результатам фиксируем дефолт.
    

---

### 6) Numba / производительность / warmup

- Все вычисления в `compute_numba` через `@njit(parallel=True, fastmath=True, cache=True)`.
    
- Обязательный **warmup** при старте процесса (компиляция ключевых kernels).
    
- Конфигурируем:
    
    - `NUMBA_NUM_THREADS` (через конфиг контекста indicators),
        
    - `NUMBA_CACHE_DIR` (чтобы кеш работал в Docker/на сервере).
        

---

### 7) Registry: hybrid (код + YAML)

- **Hard bounds и формальная спецификация** индикатора — в коде (`domain/definitions/*`).
    
- **UI defaults / включенность / дефолтные range/values / шаги (step)** — в YAML (`configs/*/indicators.yaml`).
    
- На старте приложение валидирует YAML против `IndicatorDef`:
    
    - нельзя выйти за hard bounds,
        
    - шаги/диапазоны должны быть валидны (step > 0, range не пустой, значения в пределах bounds).
        

---

### 8) API (Milestone 2)

В API (в `apps/api`) добавляем endpoints:

- `GET /indicators` — список индикаторов и параметров (registry + UI defaults)
    
- `POST /indicators/estimate` — **оценка размера батча**:
    
    - возвращает **только итоговые числа**:
        
        - `total_variants` (количество комбинаций по всему батчу)
            
        - `estimated_memory_bytes` (оценка памяти по всему батчу)
            
    - отказ при превышении:
        
        - `total_variants > 600_000`
            
        - `estimated_memory_bytes > max_compute_bytes_total` (default 5 GiB)
            
    
    _(Важно: в estimate НЕ показываем first/last/preview значений осей; UI нужен только итоговый размер и память.)_
    
- `POST /indicators/compute` — ограниченный compute (с guard 600k + 5GiB) для расчёта **индикаторных тензоров** на CPU/Numba.  
    Большие объёмы/батчи — позже через jobs.
    

---

## Базовый набор индикаторов (фиксируем как “baseline”)

Ниже — **те, которые точно ложатся на движок** (OHLCV → numpy → numba grids), разделены на:

### A) Simple grid (обычно 1–2 оси: window × source)

**MA/сглаживание**

- SMA, EMA, WMA/LWMA, RMA/SMMA, VWMA, MA по HLC3/HL2
    

**Volatility**

- TR, ATR, Rolling Std/Variance, HV(log returns)
    

**Momentum**

- RSI, ROC/Momentum, CCI, Williams %R, TRIX, Fisher Transform
    

**Volume**

- OBV, Volume MA, A/D, CMF, MFI, Rolling VWAP (window) + deviation
    

**Trend/Breakout**

- ADX/DMI, Aroon, Donchian, LinReg slope/channel, Vortex
    

**Structure/Normalization**

- Z-score, Percentile rank, Candle stats (body/wicks, ATR-normalized)
    

### B) Multi-param / complex (делаем через базовые тензоры + индексацию, без взрыва памяти)

- DEMA, TEMA, ZLEMA, HMA
    
- KAMA, ALMA, FRAMA, McGinley
    
- MA envelopes/ribbons
    
- MACD, PPO
    
- Stochastic K%D, Stoch RSI
    
- Ultimate Oscillator, AO
    
- TSI, SMI, Connors RSI, RMI
    
- Bollinger Bands (+Bandwidth/%B)
    
- Chaikin Volatility
    
- Range-based vol estimators (Parkinson, GK, RS, YZ)
    
- EWMA vol (RiskMetrics)
    
- SuperTrend, Keltner, Chandelier Exit
    
- Parabolic SAR
    
- Ichimoku
    
- Heikin-Ashi
    

(Мы не включаем “breadth/рынок целиком”, microstructure, PCA/ADF/KPSS и подобное — это уже другой класс данных/стоимости.)

---

## EPIC’и Milestone 2 (план разработки)

Нотация: `IND-EPIC-XX`.  
Для каждого — цель, что делаем, DoD, куда ложится в репозитории.

---

### IND-EPIC-01 — Domain model + DTO + Ports (скелет контекста)

**Цель:** зафиксировать контрактную модель для registry/grid/tensor.

**Делаем:**

- `IndicatorDef`, `ParamDef`, `AxisDef`, `GridSpec`, `IndicatorTensor`, `Layout`.
    
- Application ports:
    
    - `IndicatorRegistryPort`
        
    - `IndicatorComputePort`
        
    - `CandleFeedPort` (ACL)
        
- DTO для API: request/response для list/estimate/compute.
    

**DoD:**

- Контракты покрыты unit-тестами на валидацию (bounds/step/explicit+range).
    
- Публичные интерфейсы не завязаны на Binance/Bybit.
    

**Paths:**

- `src/trading/contexts/indicators/domain/*`
    
- `src/trading/contexts/indicators/application/dto/*`
    
- `src/trading/contexts/indicators/application/ports/*`
    

---

### IND-EPIC-02 — Registry (кодовые defs + YAML defaults)

**Цель:** UI видит библиотеку индикаторов и дефолтные диапазоны.

**Делаем:**

- `domain/definitions/*.py` по группам (ma/trend/volatility/momentum/volume/structure).
    
- YAML `configs/*/indicators.yaml` (dev/prod/test).
    
- Валидатор: YAML не может выходить за hard bounds, **step валиден и используется для построения сеток**.
    

**DoD:**

- `GET /indicators` отдаёт полный registry (hard bounds + UI defaults).
    
- Ошибки YAML понятные и “fail-fast” на старте.
    

**Paths:**

- `src/trading/contexts/indicators/domain/definitions/*`
    
- `configs/dev/indicators.yaml`, `configs/prod/indicators.yaml`, `configs/test/indicators.yaml`
    
- `apps/api/routes/indicators.py` (+ wiring)
    

---

### IND-EPIC-03 — Grid builder + estimator + guards (600k + 5GiB) **(обновлено)**

**Цель:** уметь раскрывать сетки и заранее оценивать размер задачи для **всего батча**.

**Делаем:**

- раскрытие `explicit values` + `range(start/stop/step)` (**step берётся из YAML-конфига**) для:
    
    - параметров индикаторов,
        
    - параметров риск-контролей (SL/TP) — учитываются в общей комбинаторике.
        
- подсчёт `total_variants` как **итоговое число комбинаций по всему батчу** (перемножение длин осей внутри каждого блока и далее — в единую формулу батча по утверждённой модели),
    
- оценка `estimated_memory_bytes` по батчу (на основании `T` и планируемых тензоров + reserve workspace),
    
- guard:
    
    - `total_variants <= 600k`,
        
    - `estimated_memory_bytes <= max_compute_bytes_total` (default 5GiB),
        
- `POST /indicators/estimate` возвращает **только** `total_variants` и `estimated_memory_bytes` (без preview осей).
    

**DoD:**

- estimator совпадает с фактическим compute (по variant_count и guard-логике),
    
- отказ при превышении лимита 600k или 5GiB.
    

**Paths:**

- `src/trading/contexts/indicators/application/services/grid_builder.py`
    
- `apps/api/routes/indicators.py`
    

---

### IND-EPIC-04 — CandleFeed ACL: dense timeline + NaN holes

**Цель:** индикаторы получают быстрые contiguous arrays и не думают про пропуски.

**Делаем:**

- порт `CandleFeedPort` + реализация в `adapters/outbound/*/market_data_acl`.
    
- функция `load_ohlcv_dense(...)`:
    
    - читает свечи,
        
    - строит плотный таймлайн,
        
    - вставляет NaN в missing.
        

**DoD:**

- На тестовом символе подтверждаем корректность: длина, NaN-дырки, порядок.
    
- Perf: построение dense arrays не становится бутылочным горлом.
    

**Paths:**

- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/*` (или аналогичный уже принятый ACL путь)
    
- тесты: `tests/unit/contexts/indicators/...`
    

---

### IND-EPIC-06 — Реализация группы MA + базовые “строительные блоки”

**Цель:** первый мощный вертикальный результат: MA-тензоры и sources.

**Делаем (Numba grids):**

- SMA/EMA/WMA/LWMA/RMA/VWMA
    
- sources: open/high/low/close + derived HLC3/HL2
    
- базовые multi-param: DEMA/TEMA/ZLEMA/HMA
    
- (если влезает без перегруза) KAMA/ALMA/FRAMA/McGinley + envelopes
    

**DoD:**

- `POST /indicators/compute` считает MA по сетке (например windows 1..200 × 4 sources) ≤ 600k.
    
- Есть тесты сравнения `compute_numba` vs `compute_numpy` на случайных данных.
    
- Perf-smoke: MA grid на реальных свечах проходит в заданных пределах (фиксируем в тесте).
    

**Paths:**

- `compute_numba/kernels/ma.py`
    
- `compute_numpy/*` как oracle
    
- `tests/perf_smoke/test_indicators_ma.py`
    

---

### IND-EPIC-07 — Volatility + Momentum (основа для будущих стратегий)

**Цель:** дать ядро ATR/Std/RSI и продвинутые multi-param осцилляторы.

**Делаем:**

- Vol: TR/ATR, rolling std/var, HV, EWMA vol, range-based estimators, BBands (+Bandwidth/%B)
    
- Momentum: RSI, ROC, CCI, %R, TRIX, Fisher, + multi-param (Stoch, StochRSI, PPO, MACD)
    

**DoD:**

- compute работает для каждого индикатора из списка, grid в пределах guard.
    
- Корректная NaN-зона и NaN propagation.
    
- Perf smoke на 2–3 индикаторах из группы.
    

**Paths:**

- `compute_numba/kernels/volatility.py`
    
- `compute_numba/kernels/momentum.py`
    
- тесты unit + perf_smoke
    

---

### IND-EPIC-08 — Trend + Volume (каналы/пробои/денежный поток)

**Цель:** закрыть трендовые и объёмные признаки.

**Делаем:**

- Trend: ADX/DMI, Aroon, Donchian, LinReg slope/channel, Vortex, SuperTrend, Keltner, Chandelier, SAR, Ichimoku, Heikin-Ashi
    
- Volume: OBV, Volume MA, A/D, CMF, MFI, rolling VWAP, Volume Oscillator, Force Index, Klinger, VPT/PVT, EOM
    

**DoD:**

- Registry отражает все параметры и их bounds/step.
    
- Compute покрыт unit сравнениями на части индикаторов + sanity checks.
    
- Guards не дают взорвать комбинаторику.
    

**Paths:**

- `compute_numba/kernels/trend.py`
    
- `compute_numba/kernels/volume.py`
    

---

### IND-EPIC-09 — Structure/Normalization features

**Цель:** добавить “признаки режима” без ухода в microstructure/market breadth.

**Делаем:**

- Z-score, Percentile rank
    
- Candle stats (body/wicks, ATR-normalized)
    
- distance-to-MA normalized (как multi-param композиция)
    

**DoD:**

- Индикаторы доступны через registry + compute.
    
- Perf smoke на percentile/zscore.
    

**Paths:**

- `compute_numba/kernels/structure.py`
    

---

### IND-EPIC-10 — Layout benchmark + фиксация дефолта

**Цель:** выбрать default layout (TIME_MAJOR или VARIANT_MAJOR) на фактах.

**Делаем:**

- micro-bench на ваших данных:
    
    - MA grid 1..200 × sources
        
    - ATR/RSI grid
        
- замер: wall time, peak RSS, скорость последующего прохода по вариантам.
    

**DoD:**

- В репо есть perf отчет/таблица (в docs) + принято решение “default layout”.
    
- Код поддерживает оба layout-а, но один выбран по умолчанию.
    

**Paths:**

- `tests/perf_smoke/bench_indicators_layout.py`
    
- `docs/architecture/indicators/indicators-overview.md` (секция “Layout decision”)
    

---

### IND-EPIC-11 — Документация + runbooks (по группам)

**Цель:** чтобы это можно было поддерживать и расширять без “магии”.

**Делаем:**

- `docs/architecture/indicators/indicators-overview.md`
    
- docs по группам:
    
    - `indicators-ma.md`, `indicators-volatility.md`, `indicators-momentum.md`, `indicators-trend.md`, `indicators-volume.md`, `indicators-structure.md`
        
- Runbook:
    
    - warmup/JIT задержки
        
    - cache dir / NUMBA_NUM_THREADS
        
    - “почему NaN” и как это влияет
        

**DoD:**

- Есть понятные правила добавления нового индикатора:
    
    - def в коде + YAML defaults + kernel + тесты.
        

---

## Организация файлов (фикс)

Как согласовали:

**Код kernels — группами:**

- `compute_numba/kernels/{_common,ma,trend,volatility,momentum,volume,structure}.py`
    

**Registry defs — группами:**

- `domain/definitions/{ma_defs,trend_defs,volatility_defs,momentum_defs,volume_defs,structure_defs}.py`
    

**Документация — группами:**

- `docs/architecture/indicators/indicators-*.md`
    

---

## Критерий готовности Milestone 2 (общий DoD)

Milestone 2 считается выполненным, если:

1. `GET /indicators` отдаёт полный registry + YAML defaults
    
2. `POST /indicators/estimate`:
    
    - считает **итоговое число комбинаций** по всему батчу,
        
    - считает **оценку памяти** по всему батчу,
        
    - режет по **600k** и по **max_compute_bytes_total (default 5GiB)**,
        
    - **не возвращает preview** (first/last и подобное)
        
3. `POST /indicators/compute` считает тензоры на CPU/Numba для baseline набора
    
4. warmup происходит на старте и не ломается в Docker
    
5. есть perf_smoke тесты и layout decision зафиксирован в docs
    

---

Порядок внедрения по приоритету (какой EPIC делать первым/вторым в плане максимальной пользы для следующего Milestone 3/4) — **обновление минимальное**: остаётся **01→05→02→03→04→06→07→10→08→09→11**.

---
