# Backtest v1 — Candle Timeline + Rollup + Warmup Policy (BKT-EPIC-02)

Фиксирует контракт BKT-EPIC-02: как backtest v1 строит свечной таймлайн на выбранном timeframe из canonical 1m, как работает best-effort rollup и warmup lookback.

## Цель

- Детерминированно загрузить 1m candles из canonical как source of truth.
- Построить derived timeframe candles (`5m/15m/1h/4h/1d`) детерминированно.
- Зафиксировать warmup lookback policy v1:
  - `warmup_bars_default = 200` (конфигурируемо),
  - если истории недостаточно — стартуем с первого доступного бара (без ошибки),
  - метрики/отчёт считаются только на целевом интервале `[Start, End)`.

## Контекст

- Source of truth по свечам: `market_data.canonical_candles_1m`.
- Чтение canonical 1m выполняется через port `CanonicalCandleReader`.
- Для compute-инфраструктуры индикаторов уже существует ACL `CandleFeed.load_1m_dense(...)`, который:
  - строит плотный 1m таймлайн `[start, end)`,
  - ставит `NaN` в missing 1m candles,
  - возвращает `CandleArrays` (contiguous arrays) для `IndicatorCompute`.

Ограничения backtest v1:

- Пользователь не обязан выравнивать `Start/End` к границам timeframe.
- Derived timeframe candles не должны “пропадать” из-за одной отсутствующей минуты.
- Derived timeframe candles не должны содержать `NaN` (в v1 для backtest).

## Scope

- Canonical 1m load: через `CanonicalCandleReader` как источник правды, но технически — через `CandleFeed.load_1m_dense(...)` (ACL indicators -> market_data) для получения dense 1m + NaN holes.
- Derived rollup: построение `5m/15m/1h/4h/1d` из 1m по epoch-aligned bucket границам (`Timeframe.bucket_open/bucket_close`).
- Warmup lookback:
  - `warmup_bars_default=200` (в барах целевого timeframe; конфигурируемо),
  - если данных не хватает — без ошибки стартуем с первого доступного бара.
- “Target slice”: downstream метрики/отчёт считаются только на `[Start, End)`.

## Non-goals

- Live ingestion/repair (это `market_data` и Strategy live-runner).
- Staged runner, execution engine, reporting metrics, API transport (это BKT-EPIC-04/05/06/07).

## Ключевые решения

### 1) Пользовательские `Start/End` произвольны; backtest нормализует диапазон для 1m load

Пользователь может задавать любой `Start/End`, но для детерминированного 1m load нужен минутный грид.

Фиксируем внутреннюю нормализацию:

- `warmup_duration = warmup_bars * timeframe.duration()`
- `start_1m = floor_to_minute(Start - warmup_duration)`
- `end_1m   = ceil_to_minute(End)`

Где:

- `floor_to_minute`: обрезка до `...:..:00.000Z`.
- `ceil_to_minute`: округление вверх до следующей минуты при необходимости.

Последствия:

- Пользовательский input остаётся “как есть”.
- Внутренний load не превращается в “почти всё NaN” из-за сдвига в секундах.

### 2) Epoch-aligned buckets: derived timeframe выравнивается через `Timeframe.bucket_open`

Derived timeframe строится по epoch-aligned UTC bucket границам:

- `bucket_open = Timeframe.bucket_open(ts)`
- `bucket_close = bucket_open + Timeframe.duration()`

Последствия:

- Derived candle timestamps стабильны и едины во всей системе.
- `Start/End` не “переносят” сетку — сетка фиксирована timeframe.

### 3) Rollup policy v1 для backtest — best-effort (не strict completeness)

Backtest v1 использует best-effort rollup:

- missing 1m внутри бакета не “убивают” бакет,
- derived OHLCV агрегируется по доступным 1m.

Агрегация (по доступным 1m):

- `open`  = `open` первой доступной 1m свечи в бакете (по времени)
- `close` = `close` последней доступной 1m свечи
- `high`  = `max(high)`
- `low`   = `min(low)`
- `volume` = `sum(volume)`

Критерий “1m candle доступна” (v1):

- candle доступна, если `close` является finite (не NaN).

Последствия:

- Derived candle существует даже при missing минуте/нескольких минутах.
- Rollup остаётся детерминированным и тестируемым.

### 4) Empty bucket policy v1: carry-forward + `volume=0`

Если в бакете нет ни одной доступной 1m свечи:

- `open = high = low = close = prev_close`
- `volume = 0`

Где `prev_close` — последняя известная derived `close` до текущего бакета.

Если `prev_close` отсутствует (во всём диапазоне нет данных):

- возвращаем доменную `validation_error` (422): “no market data for requested range”.

Последствия:

- Таймлайн остаётся плотным.
- Derived candles не содержат NaN.

### 5) Warmup lookback v1: `warmup_bars_default=200` баров целевого timeframe

Warmup рассчитывается как:

- `warmup_bars` (default `warmup_bars_default=200`, можно override)
- `warmup_duration = warmup_bars * timeframe.duration()`

Если история инструмента меньше, чем `warmup_duration`:

- не падаем,
- начинаем с первого доступного бара.

Метрики/отчёт downstream считаются только на `[Start, End)` (warmup зона не входит в отчёт).

### 6) Close-fill: бар попадает в целевой интервал по `bar_close_ts`

Так как backtest v1 исполняется на закрытии баров, “целевой” интервал `[Start, End)` для баров интерпретируется по close timestamp:

- бар включается в processing/report slice, если `Start <= bar_close_ts < End`.

## Контракты и инварианты

- Источник правды: canonical 1m (`CanonicalCandleReader`), доступ через `CandleFeed` (ACL) для dense arrays.
- `TimeRange` — полуинтервал `[start, end)`.
- Внутренний 1m load всегда минутно-нормализован (floor/ceil) для совпадения с canonical ts_open.
- Derived buckets epoch-aligned (`Timeframe.bucket_open/bucket_close`).
- Derived rollup v1 (backtest): best-effort + carry-forward; derived OHLCV не содержит NaN.
- Warmup трактуется как число баров целевого timeframe.

## Связанные файлы

- `docs/architecture/roadmap/milestone-4-epics-v1.md` — EPIC map Milestone 4, BKT-EPIC-02.
- `docs/architecture/shared-kernel-primitives.md` — `Timeframe` и bucket alignment.
- `src/trading/shared_kernel/primitives/timeframe.py` — `Timeframe.bucket_open/bucket_close`.
- `src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py` — canonical 1m reader port.
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md` — dense 1m + NaN holes.
- `src/trading/contexts/indicators/application/ports/feeds/candle_feed.py` — `CandleFeed` port.
- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py` — реализация dense 1m.
- `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` — BKT-EPIC-01 (границы контекстов и выбор портов).

## Как проверить

После реализации EPIC-02:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: best-effort rollup + carry-forward меняет поведение сигналов на участках с missing данных (flat candles).
- Риск: контраст со strict completeness rule из shared-kernel. В v1 фиксируем: strict completeness остаётся для Strategy/live-runner; backtest использует best-effort по своему контракту.
