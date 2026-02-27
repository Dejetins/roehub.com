---

# Indicators — Volatility + Momentum compute (Numba/Numpy) v1

Этот документ является source of truth для **IND-EPIC-07 — Volatility + Momentum (основа для будущих стратегий)** в bounded context `indicators`.

Цель EPIC: расширить compute-движок набором **volatility** и **momentum** индикаторов, которые:

* считаются **тензором по сетке параметров** на CPU через Numba;
* имеют **Numpy oracle** для unit tests (семантика совпадает);
* корректно обрабатывают **NaN holes** (propagate + reset там, где stateful);
* покрыты unit + perf-smoke тестами;
* доступны через уже существующий `POST /indicators/compute` (один индикатор на запрос).

Связанные документы (читать вместе):

* `docs/architecture/indicators/indicators-overview.md`
* `docs/architecture/indicators/indicators-compute-engine-core.md`
* `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
* `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
* `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md`
* `docs/architecture/indicators/indicators-ma-compute-numba-v1.md` (эталон для NaN policy и state reset)
* `docs/architecture/indicators/indicators_formula.yaml` (спецификация формул/outputs для индикаторов)

Связанные ключевые файлы/папки:

* Numba runtime: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
* Numba kernels common: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
* Numba kernels (добавляем):

  * `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py`
  * `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py`
* Numpy oracle (добавляем):

  * `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py`
  * `src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py`
* Grid materialization: `src/trading/contexts/indicators/application/services/grid_builder.py`
* Candle arrays DTO: `src/trading/contexts/indicators/application/dto/candle_arrays.py`
* CandleFeed ACL: `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
* API route: `apps/api/routes/indicators.py`
* API DTO: `apps/api/dto/indicators.py`
* Config defaults: `configs/prod/indicators.yaml` (аналогично `dev/test`)

---

## Scope / Non-goals

### In scope (EPIC-07)

**Volatility**

* `volatility.tr`
* `volatility.atr`
* `volatility.stddev`
* `volatility.variance`
* `volatility.hv` (c `annualization`)
* `volatility.bbands`
* `volatility.bbands_bandwidth`
* `volatility.bbands_percent_b`

**Momentum**

* `momentum.rsi`
* `momentum.roc`
* `momentum.cci`
* `momentum.williams_r`
* `momentum.trix`
* `momentum.fisher`
* multi-param:

  * `momentum.stoch`
  * `momentum.stoch_rsi`
  * `momentum.ppo`
  * `momentum.macd`

**Также входит**

* Numpy oracle для перечисленных индикаторов (семантика совпадает с Numba).
* Unit tests: сравнение `compute_numba` vs `compute_numpy` (включая NaN holes).
* Perf-smoke: минимум 2–3 индикатора из группы (см. раздел Tests).
* Обновление exports через `__init__.py` и документации (`docs/architecture/indicators/README.md`) при необходимости.

### Out of scope (EPIC-07)

* EWMA vol (как отдельный indicator_id), range-based estimators (Parkinson/GK), т.к. их **нет** в текущем `configs/prod/indicators.yaml` (в v1 не добавляем новые идентичности без согласования).
* Расширение API `/indicators/compute` на batch (несколько индикаторов в одном запросе).
* DSL runtime-интерпретатор `indicators_formula.yaml` (формулы — спецификация; kernels пишутся кодом).

---

## Source of Truth

1. **Список индикаторов / параметры / шаги сетки**:

* `configs/prod/indicators.yaml` (и `configs/dev/indicators.yaml`, `configs/test/indicators.yaml`)

2. **Формулы / outputs (спецификация)**:

* `docs/architecture/indicators/indicators_formula.yaml`

Для каждого индикатора, который:

* присутствует в `configs/*/indicators.yaml` **и**
* реализуется в этом EPIC,

**должна существовать запись в `indicators_formula.yaml`** с согласованными ключами (`indicator_id`, `inputs`, `params`, `outputs`).

3. **Input формат и NaN holes**:

* CandleFeed возвращает dense timeline + NaN holes (см. `indicators-candlefeed-acl-dense-timeline-v1.md`).

---

## Ключевые решения

### 1) Один `/compute` = один индикатор (v1)

`POST /indicators/compute` в v1 вычисляет **один** `indicator_id` по grid.
Эта модель повторяет EPIC-06 и упрощает guards/ошибки/перф.

### 2) Sources — часть grid только там, где это задано в YAML defaults

Если в `configs/prod/indicators.yaml` у индикатора задан `inputs.source`, то `source` — **обязательная ось grid**.

* Порядок `explicit.values` сохраняется как в request (без сортировки).
* Допустимые значения `source` должны совпадать с доменной моделью (см. `domain/definitions/*` и `domain/entities/input_series.py`).

Для source-parameterized volatility/momentum индикаторов в compute engine используется grouped-source pipeline:

* `variant_source_labels` группируются по уникальному `source` в deterministic порядке;
* kernel рассчитывается по source-группам на subset-вариантах;
* результаты scatter’ятся обратно в глобальный `(V, T)` без изменения variant ordering.

### 3) In-place heavy kernels (phase 4)

Для наиболее тяжёлых volatility/momentum kernels в variant-циклах используется in-place write path:

* запись результата идёт сразу в `out[variant_index, :]`;
* внутренние helper-ы `*_into_*` используются там, где это снижает число временных векторов;
* публичные entrypoints `compute_momentum_grid_f32(...)` и `compute_volatility_grid_f32(...)`
  сохраняют прежние сигнатуры и поведение.

Phase 4 покрывает следующие индикаторы:

* `volatility.bbands`
* `volatility.bbands_bandwidth`
* `volatility.bbands_percent_b`
* `momentum.stoch_rsi`
* `momentum.macd`
* `momentum.ppo`

### 4) NaN policy (фикс v1, единая для EPIC-07)

Compute **не делает импутацию**. NaN holes из CandleFeed должны корректно “просачиваться” в outputs.

**A) Windowed (rolling) индикаторы**
Примеры: stddev/variance/hv/bbands/roc (если windowed)/cci/williams_r/stoch/stoch_rsi

* `t < window-1` → `NaN` (warmup зона)
* если в окне есть NaN → output NaN
* деление на 0 → NaN (детерминированно)

**B) Stateful EMA-like части (reset-on-NaN)**
Примеры: RSI (через RMA), TRIX (EMA×3), MACD/PPO (EMA fast/slow + signal)

* до первого валидного `x[t]` → NaN
* при первом валидном `x[t]`: seed `y[t] = x[t]`
* если `x[t]` NaN → `y[t]=NaN` и **reset state**
* после reset, при следующем валидном `x[t]` стартуем заново с seed

Эта политика гарантирует, что состояние не “перетягивается” через пропуски.

### 5) `volatility.tr` и `volatility.atr` используют OHLC, а не `source`

* `TR` использует `high/low/close` (prev_close нужен для max-ветки).
* `ATR` = сглаживание TR (v1 фикс: RMA alpha=1/window, с reset-on-NaN).

### 6) Float32 output (фикс)

`IndicatorTensor.values` — `float32`.
Внутри kernels допускаются float64 accumulator’ы (implementation detail), но выход фиксирован как `float32`.

### 7) Guards применяются до больших аллокаций тензора

До расчёта:

* `variants <= max_variants_per_compute` (default 600k)
* `estimated_memory_bytes <= max_compute_bytes_total` (default 5 GiB)

Модель estimate/memory reserve совпадает с EPIC-03/06:

* `reserve = max(64MiB, 20% от (bytes_candles + bytes_outputs))`

---

## Реализуемые индикаторы (v1)

Ниже перечислены индикаторы **строго из** `configs/prod/indicators.yaml`, которые входят в EPIC-07.

### Volatility

#### 1) `volatility.tr`

Параметры: отсутствуют.
Inputs: `high`, `low`, `close`.

Семантика (v1):

* `TR[t] = max(high[t]-low[t], abs(high[t]-close[t-1]), abs(low[t]-close[t-1]))`
* если `high[t]` или `low[t]` NaN → `TR[t]=NaN`
* если `close[t-1]` NaN (например на границе NaN-hole), то используются только валидные ветки:

  * `TR[t] = high[t]-low[t]` (если `high/low` валидны), иначе NaN

Output: `tr` (см. `indicators_formula.yaml`).

#### 2) `volatility.atr(window)`

Params:

* `window: int` (range 5..120 step 1)

Inputs: как TR (`high`, `low`, `close`).
Семантика (v1):

* `TR` как выше
* `ATR` = RMA(TR, window), alpha=`1/window`, reset-on-NaN

Output: `atr`.

#### 3) `volatility.stddev(window, source)`

Params:

* `window: int` (range 5..120 step 1)

Inputs:

* `source` ∈ explicit (из YAML: `close/hlc3/ohlc4/low/high/open`)

Семантика:

* rolling stddev на `source`
* warmup `t < window-1` → NaN
* NaN в окне → NaN

Output: `stddev` (точное имя — по `indicators_formula.yaml`).

#### 4) `volatility.variance(window, source)`

Params:

* `window: int` (range 5..120 step 1)

Inputs: как stddev.
Семантика: rolling variance на `source`, warmup/NaN как выше.

Output: `variance`.

#### 5) `volatility.hv(window, annualization, source)`

Params:

* `window: int` (range 5..120 step 1)
* `annualization: int` explicit `[252, 365]`

Inputs:

* `source` ∈ explicit (из YAML)

Семантика (v1):

* `r[t] = log(source[t] / source[t-1])`
* если `source[t]` или `source[t-1]` NaN или `<= 0` → `r[t]=NaN`
* `hv[t] = stddev(r over window) * sqrt(annualization)`
* warmup на returns: первые значения NaN по определению (минимум 1 лаг + окно)

Output: `hv`.

#### 6) `volatility.bbands(window, mult, source)`

Params:

* `window: int` (range 5..120 step 1)
* `mult: float` (range 0.5..4.0 step 0.01)

Inputs:

* `source` ∈ explicit

Семантика (v1):

* `middle = SMA(source, window)`
* `sigma = stddev(source, window)`
* `upper = middle + mult*sigma`
* `lower = middle - mult*sigma`
* warmup/NaN: как rolling

Outputs: ожидаются как минимум `middle`, `upper`, `lower` (точные ключи — по formula yaml).

#### 7) `volatility.bbands_bandwidth(window, mult, source)`

Params/Inputs: как bbands.

Семантика (v1):

* рассчитывает базовые bbands,
* `bandwidth = (upper - lower) / middle`
* если `middle == 0` или NaN → NaN

Output: `bandwidth` (точное имя — по formula yaml).

#### 8) `volatility.bbands_percent_b(window, mult, source)`

Params/Inputs: как bbands.

Семантика (v1):

* рассчитывает базовые bbands,
* `percent_b = (source - lower) / (upper - lower)`
* если `(upper-lower)==0` или NaN → NaN

Output: `percent_b` (точное имя — по formula yaml).

> Примечание: реализация может (и должна) переиспользовать общий bbands-core и лишь выбирать output.

---

### Momentum

#### 1) `momentum.rsi(window, source)`

Params:

* `window: int` (range 3..120 step 1)

Inputs:

* `source` ∈ explicit

Семантика (v1, через RMA, reset-on-NaN):

* `delta[t] = source[t] - source[t-1]`
* `gain[t] = max(delta, 0)`, `loss[t] = max(-delta, 0)`
* `avg_gain = RMA(gain, window)` (alpha=1/window, reset-on-NaN)
* `avg_loss = RMA(loss, window)` (alpha=1/window, reset-on-NaN)
* `RS = avg_gain / avg_loss`
* `RSI = 100 - 100/(1+RS)`
* если `avg_loss == 0`:

  * если `avg_gain == 0` → RSI = NaN (детерминированно, чтобы не генерить “идеальные” 50/100 на дырках)
  * иначе RSI = 100
* warmup: минимум window + лаг (по факту materialization)

Output: `rsi`.

#### 2) `momentum.roc(window, source)`

Params:

* `window: int` (range 3..120 step 1)

Inputs:

* `source` ∈ explicit

Семантика (v1):

* `roc = 100 * (source[t] / source[t-window] - 1)`
* если `source[t]` или `source[t-window]` NaN или `source[t-window]==0` → NaN
* warmup `t < window` → NaN

Output: `roc`.

#### 3) `momentum.cci(window)`

Params:

* `window: int` (range 5..120 step 1)

Inputs: `high`, `low`, `close` (без `source` оси).
Семантика (v1):

* `tp = (high + low + close) / 3`
* `sma = SMA(tp, window)`
* `mean_dev = mean(|tp - sma| over window)`
* `cci = (tp - sma) / (0.015 * mean_dev)`
* если `mean_dev == 0` или NaN → NaN
* warmup/NaN: rolling policy

Output: `cci`.

#### 4) `momentum.williams_r(window)`

Params:

* `window: int` (range 5..120 step 1)

Inputs: `high`, `low`, `close`.
Семантика (v1):

* `hh = max(high over window)`
* `ll = min(low over window)`
* `%R = -100 * (hh - close) / (hh - ll)`
* если `(hh-ll)==0` или NaN → NaN
* warmup/NaN: rolling policy

Output: `williams_r`.

#### 5) `momentum.trix(window, signal_window, source)`

Params:

* `window: int` (range 5..120 step 1)
* `signal_window: int` (range 3..20 step 1)

Inputs:

* `source` ∈ explicit

Семантика (v1, EMA chain reset-on-NaN):

* `ema1 = EMA(source, window)`
* `ema2 = EMA(ema1, window)`
* `ema3 = EMA(ema2, window)`
* `trix = 100 * (ema3[t] / ema3[t-1] - 1)` (если `ema3[t-1]==0` → NaN)
* `signal = EMA(trix, signal_window)`
* `hist = trix - signal`

Outputs: `trix`, `signal`, `hist` (точные ключи — по formula yaml).

#### 6) `momentum.fisher(window)`

Params:

* `window: int` (range 5..120 step 1)

Inputs: v1 фикс: `close` (без source axis).
Семантика (v1, классическая Fisher Transform):

* нормализация price в [-1, 1] по rolling min/max за window:

  * `v = 2*(x - min)/(max-min) - 1`
  * clamp `v` в `[-0.999, 0.999]` (детерминированно)
* `fisher = 0.5 * ln((1+v)/(1-v))`
* warmup/NaN: rolling policy (если min/max недоступны → NaN)

Output: `fisher`.

#### 7) `momentum.stoch(k_window, smoothing, d_window)`

Params:

* `k_window: int` (range 5..120 step 1)
* `smoothing: int` (range 2..10 step 1)
* `d_window: int` (range 2..10 step 1)

Inputs: `high`, `low`, `close`.
Семантика (v1):

* `hh = max(high over k_window)`
* `ll = min(low over k_window)`
* `k_raw = 100 * (close - ll) / (hh - ll)` (если denom==0 → NaN)
* `k = SMA(k_raw, smoothing)`
* `d = SMA(k, d_window)`
* warmup/NaN: rolling policy на каждом шаге

Outputs: `k`, `d`.

#### 8) `momentum.stoch_rsi(rsi_window, k_window, smoothing, d_window, source)`

Params:

* `rsi_window: int` (range 3..120 step 1)
* `k_window: int` (range 5..120 step 1)
* `smoothing: int` (range 2..10 step 1)
* `d_window: int` (range 2..10 step 1)

Inputs:

* `source` ∈ explicit

Семантика (v1):

* `rsi = RSI(source, rsi_window)` (см. выше)
* `hh = max(rsi over k_window)`
* `ll = min(rsi over k_window)`
* `k_raw = 100 * (rsi - ll) / (hh - ll)` (denom==0 → NaN)
* `k = SMA(k_raw, smoothing)`
* `d = SMA(k, d_window)`
* warmup/NaN: rolling policy + reset semantics у RSI

Outputs: `k`, `d`.

#### 9) `momentum.ppo(fast_window, slow_window, signal_window, source)`

Params:

* `fast_window: int` (range 4..24 step 1)
* `slow_window: int` (range 8..50 step 1)
* `signal_window: int` (range 3..20 step 1)

Inputs:

* `source` ∈ explicit

Семантика (v1, EMA reset-on-NaN):

* `fast = EMA(source, fast_window)`
* `slow = EMA(source, slow_window)`
* `ppo = 100 * (fast - slow) / slow` (если slow==0 → NaN)
* `signal = EMA(ppo, signal_window)`
* `hist = ppo - signal`

Outputs: `ppo`, `signal`, `hist`.

#### 10) `momentum.macd(fast_window, slow_window, signal_window, source)`

Params:

* `fast_window: int` (range 4..24 step 1)
* `slow_window: int` (range 8..50 step 1)
* `signal_window: int` (range 3..20 step 1)

Inputs:

* `source` ∈ explicit

Семантика (v1, EMA reset-on-NaN):

* `fast = EMA(source, fast_window)`
* `slow = EMA(source, slow_window)`
* `macd = fast - slow`
* `signal = EMA(macd, signal_window)`
* `hist = macd - signal`

Outputs: `macd`, `signal`, `hist`.

---

## Execution pipeline (compute v1)

### 1) Request → Grid → Axes

* Вход: `ComputeRequest` (`apps/api/dto/indicators.py`)
* Grid материализуется через:

  * `src/trading/contexts/indicators/application/services/grid_builder.py`
* Результат:

  * `AxisDef` по каждой оси (`window`, `source`, `mult`, `annualization`, …)
  * `variants = product(axis_lengths)`

### 2) CandleFeed → CandleArrays

* Dense candles через CandleFeed ACL:

  * `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
* Получаем dense timeline `[start, end)` на 1m, NaN holes уже расставлены.

### 3) Guards

* применяются до больших аллокаций:

  * variants guard (600k default)
  * memory guard (5 GiB default)
* ошибки: детерминированные payload’ы (422) как в EPIC-03/06.

### 4) Compute (Numba) + Output tensor

* Numba kernels:

  * `compute_numba/kernels/volatility.py`
  * `compute_numba/kernels/momentum.py`
* Output: `IndicatorTensor(values=float32, axes, layout, meta)`.

### 5) Oracle (Numpy)

* Numpy kernels:

  * `compute_numpy/volatility.py`
  * `compute_numpy/momentum.py`
* Используется в unit tests как эталон.

---

## API: POST /indicators/compute (v1)

Endpoint уже существует (EPIC-06). EPIC-07 расширяет набор поддерживаемых `indicator_id`.

Требования:

* Endpoint обязан применять guards и возвращать детерминированные ошибки.
* Если API в v1 отдаёт “компактный ответ” без тензора значений — это допустимо, но compute **должен реально выполняться**, и тесты должны это подтверждать.

---

## Tests

### Unit tests (обязательные)

Покрываем:

* совпадение `compute_numba` vs `compute_numpy` на фиксированном seed,
* NaN policy:

  * warmup зоны,
  * NaN holes и state reset (RSI, TRIX, MACD, PPO),
* dtype/layout invariants:

  * output `float32`,
  * массивы contiguous,
  * axes порядок детерминирован.

Рекомендуемые пути:

* `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volatility_kernels.py` *(new)*
* `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_volatility_oracle.py` *(new)*
* `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_momentum_kernels.py` *(new)*
* `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_momentum_oracle.py` *(new)*
* (при необходимости) расширение `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py`

### Perf-smoke (обязательный)

Минимум 2–3 индикатора:

1. `volatility.atr` (grid по window 5..120)
2. `momentum.rsi` (grid по window 3..120 × sources)
3. `momentum.macd` или `volatility.bbands` (multi-param или multi-output)

Путь:

* `tests/perf_smoke/contexts/indicators/test_indicators_vol_mom.py` *(new)*

Сценарий:

* использовать реальный `CandleFeed` (или фиксированную dense fixture, если feed в CI недоступен),
* grid близкий к guard, но ≤ 600k,
* asserts:

  * не падает,
  * guards не превышены,
  * фиксируем метрику времени/размера (без жёсткого SLA, но стабильный smoke).

---

## Determinism & ordering rules (фикс)

1. Оси grid:

* `explicit.values` сохраняют порядок request (без сортировки).
* `range(start, stop_incl, step)` материализуется строго инклюзивно.

2. Variants ordering:

* порядок осей в `IndicatorDef` определяет порядок перемножения и variant indexing.
* variant enumeration детерминирована при одинаковом request.

3. Candles:

* политика CandleFeed ACL:

  * duplicates: `last-wins` (детерминированно),
  * out-of-range: `ignore` (детерминированно),
  * NaN holes сохраняются.

4. Floating ops:

* избегать `fastmath` там, где NaN-семантика важна (как в EPIC-06 для MA).

---

## Целевое размещение в репозитории

### Numba kernels

* `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py`
* `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py`
* обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/__init__.py`

### Numpy oracle

* `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py`
* `src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py`
* обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/__init__.py`

### Engine wiring

* обновить dispatch в:

  * `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
* добавить warmup-paths в:

  * `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`

### Domain definitions (если требуется)

* `src/trading/contexts/indicators/domain/definitions/volatility.py`
* `src/trading/contexts/indicators/domain/definitions/momentum.py`
  (должны соответствовать ключам из `configs/prod/indicators.yaml`)

### Docs

* добавить ссылку на этот документ в:

  * `docs/architecture/indicators/README.md`

---

## DoD (EPIC-07)

EPIC-07 считается выполненным, если:

1. Для каждого индикатора из списка In scope:

* `POST /indicators/compute` успешно выполняет compute по grid (в пределах guards),
* NaN зона и NaN propagation соблюдены,
* output `float32` и детерминирован.

2. Есть Numpy oracle для каждого индикатора (или для каждого kernel-группового ядра, если общий core) и unit tests “numba vs numpy”.

3. Есть perf-smoke тест минимум на 2–3 индикаторах из группы (ATR + RSI + MACD/BBands).

4. Все quality gates проходят:

* `uv run pytest -q`
* `uv run ruff check .`
* `uv run pyright`
* `uv run python -m compileall -q src`

5. Документация обновлена:

* добавлен этот документ,
* `docs/architecture/indicators/README.md` содержит ссылку,
* публичные экспорты через `__init__.py` стабильны.

---
