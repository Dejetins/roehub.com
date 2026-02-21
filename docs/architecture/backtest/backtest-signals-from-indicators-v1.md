# Backtest v1 — Signals-from-indicators (v1) + AND aggregation (BKT-EPIC-03)

Фиксирует контракт BKT-EPIC-03: как backtest v1 превращает значения индикаторов (primary output) и данные свечей в дискретные сигналы `LONG|SHORT|NEUTRAL`, и как эти сигналы агрегируются AND-политикой для стратегии.

## Цель

- Формально зафиксировать v1 “signal rules catalog”: для каждого `indicator_id` из `configs/prod/indicators.yaml` определить детерминированное правило получения `LONG|SHORT|NEUTRAL`.
- Зафиксировать NaN/warmup semantics: NaN/недостаток истории не создаёт сигнал, а даёт `NEUTRAL`.
- Зафиксировать детерминированную агрегацию сигналов по нескольким индикаторам: AND.

## Контекст

- Backtest v1 в Milestone 4 строится из “любых комбинаций индикаторов” и grid диапазонов параметров.
- `IndicatorCompute` возвращает `IndicatorTensor.values` как float32 тензор с одним “primary output” на `indicator_id` (см. group docs indicators).
- Многие индикаторы по природе “не направленные” (ATR/HV/stddev/variance/TR, bandwidth и т.п.). Чтобы всё равно получать `LONG|SHORT`, v1 допускает параметризуемый `delta rule`.

Важные ограничения v1:

- Это НЕ “богатый DSL” сигналов: не сравниваем индикатор A с индикатором B и не строим сложные деревья формул.
- Rule может использовать:
  - primary output индикатора,
  - candle series (`open/high/low/close/volume`) на том же баре,
  - и, при необходимости, `delta(primary_output, N)` (где `N` задаётся как параметр и может быть grid).

## Scope

- Спецификация signal rules:
  - документируется в `docs/architecture/indicators/indicators_formula.yaml` как секция `signals:` внутри каждого `indicator_id`.
  - Важно: `docs/architecture/indicators/indicators_formula.yaml` остаётся спецификацией; реализация сигналов делается кодом (1 раз) и далее используется.
- NaN/warmup semantics.
- Конфликты и детерминизм.
- Реализация signal rules для всех индикаторов из `configs/prod/indicators.yaml`.

## Non-goals

- “богатый DSL” сигналов, сравнение нескольких индикаторов друг с другом.
- Изменение compute engine `indicators` под multi-output tensors или int-output.
- Портфельные/мульти-инструментальные сигналы.

## Ключевые решения

### 1) Сигнал вычисляется на закрытии каждого бара (bar-close evaluation)

Сигнал — это дискретное значение на каждом закрытом баре: `LONG|SHORT|NEUTRAL`.

Это “событие” в том смысле, что оно вычисляется **после закрытия** бара и может возникать на каждом баре, если правило истинно.

Последствия:

- AND агрегация по нескольким индикаторам практически применима (в отличие от редких “cross-only” событий).
- Execution engine (BKT-EPIC-05) может принимать решение на каждом `close[t]`.

### 2) NaN/warmup semantics v1: NaN -> NEUTRAL

Если невозможно вычислить правило на баре `t` из-за:

- NaN на required series (primary output или candle input),
- недостатка истории для delta/окна,

то signal на баре `t` = `NEUTRAL`.

Последствия:

- пропуски данных и warmup зоны не создают ложных сделок.
- поведение детерминировано.

### 3) Delta rule v1: параметризуемый лаг (per-direction)

Для “не направленных” индикаторов и некоторых фильтров используем `delta`:

- `delta(x, N) = x[t] - x[t-N]`, где `N > 0`.

В контракте v1 параметризуем лаг отдельно для long/short (оба задаются в UI/backtest как значения или диапазоны):

- `long_delta_periods` (целое, задаётся отрицательным числом в UI/конфиге: `-1..-10`, step=1)
- `short_delta_periods` (аналогично)

Семантика (фикс v1):

- `long` условие использует `N_long = abs(long_delta_periods)`
- `short` условие использует `N_short = abs(short_delta_periods)`
- `LONG` если `delta(x, N_long) > 0`
- `SHORT` если `delta(x, N_short) < 0`
- иначе `NEUTRAL`

Если `t-N` вне диапазона или `x[t]`/`x[t-N]` NaN -> `NEUTRAL`.

### 4) Конфликты и детерминизм

- Если одно правило одновременно даёт `LONG` и `SHORT` на одном баре (например при разных лаг-параметрах) -> `NEUTRAL`.
- Агрегация стратегии:
  - `final_long = all(indicator_signal == LONG)`
  - `final_short = all(indicator_signal == SHORT)`
  - Если `final_long && final_short` на одном баре -> `NEUTRAL` + метрика/событие `conflicting_signals`.

### 5) Где живут signal параметры и их grid ranges

Signal параметры (thresholds / delta periods / body_pct_min и т.п.) должны быть доступны для UI как grid (range/explicit) и использоваться при backtest.

Чтобы не дублировать конфиги, v1 фиксирует:

- ranges/steps для signal параметров хранятся в `configs/<env>/indicators.yaml` рядом с defaults индикаторов,
  но в отдельной секции, чтобы не ломать compute grid.

Рекомендованная форма (v1):

```yaml
defaults:
  momentum.rsi:
    inputs:
      source: { mode: explicit, values: ["close"] }
    params:
      window: { mode: range, start: 3, stop_incl: 120, step: 1 }
    signals:
      v1:
        params:
          long_threshold: { mode: range, start: 10, stop_incl: 50, step: 1 }
          short_threshold: { mode: range, start: 50, stop_incl: 90, step: 1 }
```

Примечание: текущий loader `indicators` читает только `inputs/params` и игнорирует прочие ключи.
Backtest/Signals слой вводит свой loader для `defaults.<indicator_id>.signals.*`.

### 6) `signals:` в `docs/architecture/indicators/indicators_formula.yaml` — спецификация, не runtime DSL

Для каждого `indicator_id`, который участвует в backtest, добавляется секция `signals:` с:

 - rule family/type (например `compare_price_to_output`, `threshold_band`, `delta_sign`, `pivot_events`, `candle_body_direction`),
- именами signal параметров (если нужны),
- списком series, которые rule использует (`primary`, `close`, `open`, ...).

Реализация сигналов — в коде backtest (и тестах), а YAML — источник документированного контракта.

### 7) Зависимости signal rules на wrapper индикаторы (pivots)

Для `structure.pivots` v1 фиксирует правило, использующее оба события:

- `pivot_low` (через wrapper `structure.pivot_low`) -> `LONG`
- `pivot_high` (primary output `structure.pivots` или wrapper `structure.pivot_high`) -> `SHORT`

Это означает, что pipeline backtest должен уметь добавить “внутренний” compute dependency на wrapper индикатор при необходимости.

## Контракты и инварианты

- Signal evaluation выполняется на закрытии бара `t` и использует значения series на этом баре.
- `NaN` на required series -> `NEUTRAL`.
- `delta` использует прошлое значение `t-N`, где `N = abs(delta_periods)`.
- Конфликт long/short на одном баре -> `NEUTRAL`.
- AND aggregation по индикаторам детерминирована и не зависит от порядка (indicator list сортируется по `indicator_id` перед агрегацией).

## Каталог правил (v1) — по `indicator_id`

Ниже фиксируем rule family для каждого `indicator_id` из `configs/prod/indicators.yaml`.

Соглашения:

- `primary` = primary output series индикатора (как возвращает `IndicatorCompute` для данного `indicator_id`).
- `price` = цена на баре `t`:
  - если индикатор параметризуется `inputs.source` (как в `configs/prod/indicators.yaml`) — используем выбранный source series (например `close`, `high`, `low`, `open`, `hlc3`, `ohlc4`),
  - иначе используем `close`.

### MA (`ma.*`)

Rule family: `compare_price_to_output`.

- `ma.sma`, `ma.ema`, `ma.wma`, `ma.lwma`, `ma.rma`, `ma.dema`, `ma.tema`, `ma.hma`, `ma.zlema`, `ma.vwma`:
  - `LONG` если `price > primary`
  - `SHORT` если `price < primary`
  - иначе `NEUTRAL`

### Trend (`trend.*`)

- `trend.adx`: `delta_sign(primary)` (использует `long_delta_periods/short_delta_periods`).
- `trend.aroon` (primary `aroon_osc`): `threshold_band(primary)` с параметрами `long_threshold/short_threshold` (int, step=1).
- `trend.chandelier_exit` (primary `chandelier_long`): `compare_price_to_output`.
- `trend.donchian` (primary `donchian_mid`): `compare_price_to_output`.
- `trend.ichimoku` (primary `span_a`): `compare_price_to_output`.
- `trend.keltner` (primary `middle`): `compare_price_to_output`.
- `trend.linreg_slope` (primary `slope`): `sign(primary)` (LONG if >0, SHORT if <0).
- `trend.psar` (primary `psar`): `compare_price_to_output`.
- `trend.supertrend` (primary `supertrend`): `compare_price_to_output`.
- `trend.vortex` (primary `vi_plus`): `threshold_centered(primary, center=1.0)` с параметром `abs_threshold` (float, step из YAML),
  - LONG если `primary > 1.0 + abs_threshold`
  - SHORT если `primary < 1.0 - abs_threshold`

### Volatility (`volatility.*`)

- `volatility.tr`: `delta_sign(primary)`.
- `volatility.atr`: `delta_sign(primary)`.
- `volatility.stddev`: `delta_sign(primary)`.
- `volatility.variance`: `delta_sign(primary)`.
- `volatility.hv`: `delta_sign(primary)`.
- `volatility.bbands` (primary `basis`): `compare_price_to_output`.
- `volatility.bbands_bandwidth`: `delta_sign(primary)`.
- `volatility.bbands_percent_b`: `threshold_band(primary)` с float параметрами `long_threshold/short_threshold`.

### Momentum (`momentum.*`)

- `momentum.rsi`: `threshold_band(primary)` с параметрами `long_threshold/short_threshold` (int).
- `momentum.roc`: `sign(primary)` (LONG if >0, SHORT if <0).
- `momentum.cci`: `threshold_band(primary)` с параметрами `long_threshold/short_threshold` (int).
- `momentum.williams_r`: `threshold_band(primary)` с параметрами `long_threshold/short_threshold` (int; отрицательная шкала допустима).
- `momentum.trix`: `sign(primary)`.
- `momentum.fisher` (primary fisher): `sign(primary)`.
- `momentum.stoch` (primary K): `threshold_band(primary)`.
- `momentum.stoch_rsi` (primary K): `threshold_band(primary)`.
- `momentum.macd` (primary macd line): `sign(primary)`.
- `momentum.ppo` (primary ppo line): `sign(primary)`.

### Volume (`volume.*`)

- `volume.ad_line`: `delta_sign(primary)`.
- `volume.cmf`: `threshold_band(primary)` с float параметрами `long_threshold/short_threshold`.
- `volume.mfi`: `threshold_band(primary)` с int параметрами `long_threshold/short_threshold`.
- `volume.obv`: `delta_sign(primary)`.
- `volume.volume_sma`: `compare_volume_to_output`:
  - LONG если `volume > primary`
  - SHORT если `volume < primary`
- `volume.vwap`: `compare_price_to_output`.
- `volume.vwap_deviation` (primary `vwap_upper`, mean-reversion v1):
  - LONG если `close < primary`
  - SHORT если `close > primary`

### Structure (`structure.*`)

- `structure.candle_stats` (primary `body_pct`): `candle_body_direction` с параметром `min_body_pct` (float):
  - LONG если `body_pct >= min_body_pct` и `close > open`
  - SHORT если `body_pct >= min_body_pct` и `close < open`
- `structure.candle_stats_atr_norm` (primary `body_atr`): `candle_body_direction` с параметром `min_body_atr` (float) и `close/open`.
- `structure.distance_to_ma_norm`: `threshold_band(primary)` с float параметрами `long_threshold/short_threshold`.
- `structure.percent_rank`: `threshold_band(primary)`.
- `structure.pivots`: `pivot_events`:
  - LONG если `structure.pivot_low` finite на баре
  - SHORT если `structure.pivot_high` finite на баре
- `structure.zscore`: `threshold_band(primary)` с float параметрами `long_threshold/short_threshold`.

## Связанные файлы

- `docs/architecture/roadmap/milestone-4-epics-v1.md` — EPIC map Milestone 4, BKT-EPIC-03.
- `configs/prod/indicators.yaml` — список indicator ids и ranges/steps (и место для `defaults.<id>.signals`).
- `docs/architecture/indicators/indicators_formula.yaml` — formula spec и место для `signals:`.
- `docs/architecture/indicators/indicators-overview.md` — общие инварианты NaN policy/determinism.
- `docs/architecture/indicators/indicators-*.md` — primary output mapping и gotchas по группам.
- `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` — контракты backtest v1.

## Как проверить

После реализации EPIC-03:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: для части индикаторов rule family по определению “эвристика” (особенно delta_sign для volatility). Это ок в v1, т.к. стратегия может комбинировать индикаторы, а UI/grid позволяет отсеивать варианты.
- Риск: добавление `defaults.<id>.signals` в `configs/*/indicators.yaml` требует отдельного loader в backtest, чтобы не менять indicators registry contracts.
