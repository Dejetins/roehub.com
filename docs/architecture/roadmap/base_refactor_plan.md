# Base Refactor Plan: Daily Precompute Artifacts + Backtest Engine v2 (Sync-First)

План перехода backtest на engine v2 с daily precompute артефактами в `.npy`.
ClickHouse используется только в precompute, на hot path backtest данных из CH нет.

Референсы:
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `tests/notebook_tests/06_backtest_compute.ipynb`
- `tests/notebook_tests/05_hit_time_grid.ipynb`
- `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`
- `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`
- `docs/architecture/shared-kernel-primitives.md`
- `configs/prod/indicators.yaml`

---

## Зафиксированные решения

- Engine v2 полностью замещает старый backtest engine (v1 не остается как "альтернатива").
- Приоритет внедрения: сначала sync runner (`POST /backtests`), затем job runner.
- Базовые цены храним только в `1m`.
  Любые таймфреймы для индикаторов/бектеста строятся из `1m` через rollup.
- Храним готовые дискретные сигналы `{-1,0,1}`; сырые значения индикаторов не сохраняем.
- Сигналы храним отдельно (не одним огромным файлом): по инструменту + таймфрейму + `indicator_id`.
- Manifest делаем в YAML (`manifest.yaml`), как человекочитаемый паспорт артефактов.

---

## Система координат (layout артефактов)

Базовая координатная сетка: `exchange / market_type / symbol`.

`asof_date` (UTC, T-1) хранится в `manifest.yaml` и означает: "артефакты обновлены и консистентны до конца этой даты".
Директория артефактов стабильная и переиспользуется; daily precompute делает обновление хвоста, а не пересборку всего датасета.

Workspace layout (v1):

```
artifacts/backtest/v2/
  <exchange>/<market_type>/<symbol>/
    manifest.yaml
    prices/
      prices_1m.npy
      prices_1m_columns.npy
    signals/
      <timeframe>/
        <indicator_id>/
          signals.i8.npy
          manifest.yaml
    hit_times/
      <timeframe>/
        tp_values.npy
        sl_values.npy
        long_tp.u32.npy
        long_sl.u32.npy
        short_tp.u32.npy
        short_sl.u32.npy
        manifest.yaml
```

Пояснения:
- `prices/prices_1m.npy` содержит только 1m OHLCV + time (без сигналов).
- `signals/<timeframe>/<indicator_id>/signals.i8.npy` содержит только сигналы данного индикатора на данном таймфрейме для полного grid (из `configs/prod/indicators.yaml` + signal-params, если есть).
- `hit_times/<timeframe>/*` содержит hit-time grids для TP/SL на данном execution timeframe.
- `asof_date` = последняя полностью материализованная дата (UTC), по политике T-1.

---

## Manifest (YAML): что это и зачем

`manifest.yaml` нужен для:
- детерминированной валидации артефактов (dtype/shape/хэши/совместимость),
- воспроизводимости (какой конфиг/код сгенерировал файлы),
- прозрачного дебага (человеку понятно, что лежит в каталоге).

Manifest не заменяет координатную систему путей: пути и имена файлов должны быть предсказуемыми и без сканирования.

### Root `manifest.yaml` (на уровне `<exchange>/<market_type>/<symbol>/`)

Рекомендуемые поля (v1):

```yaml
schema_version: 1
artifact_kind: backtest_artifacts
artifact_version: v2

identity:
  exchange: binance
  market_type: futures
  symbol: BTCUSDT
  asof_date: "2026-03-05"   # UTC, T-1 (последняя полностью материализованная дата)

time_unit: ms

prices_1m:
  path: prices/prices_1m.npy
  columns_path: prices/prices_1m_columns.npy
  dtype: float32
  shape: [T, C]
  sha256: "..."
  columns_sha256: "..."
  time_columns: [open_time, close_time]
  ohlcv_columns: [open, high, low, close, volume]
  data_range:
    start_open_time_ms: 0
    end_close_time_ms: 0
    rows: T
  update:
    recompute_from_open_time_ms: 0

rollup:
  policy_id: backtest_rollup_best_effort@1

signals:
  policy_id: signals_from_indicators_v1
  timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]

hit_times:
  timeframes: ["1m", "5m", "15m", "1h", "4h", "1d"]

config:
  indicators_yaml_path: configs/prod/indicators.yaml
  indicators_yaml_sha256: "..."
  indicators_formula_yaml_path: docs/architecture/indicators/indicators_formula.yaml
  indicators_formula_yaml_sha256: "..."
  backtest_v2_spec_path: docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md

provenance:
  git_sha: "..."
  created_at_utc: "2026-03-06T12:34:56Z"
  generator_version: backtest-artifacts@1
```

Примечания:
- `policy_id` фиксирует выбранную rollup семантику для backtest artifacts.
  Это важно, потому что strict rollup из shared-kernel и best-effort rollup из backtest v1 дают разные derived свечи.
- Root manifest не обязан перечислять все `indicator_id` и все файлы сигналов; discovery делается по координатам пути.

### Signal `manifest.yaml` (на уровне `signals/<timeframe>/<indicator_id>/`)

Рекомендуемые поля (v1):

```yaml
schema_version: 1
kind: indicator_signals

identity:
  timeframe: "5m"
  indicator_id: "ma.sma"

grid:
  inputs:
    source: ["close", "hlc3", "ohlc4", "low", "high", "open"]
  params:
    window: {mode: range, start: 5, stop_incl: 200, step: 1}
  signal_params:
    v1: {}
  order: ["inputs.source", "params.window"]

signals:
  path: signals.i8.npy
  dtype: int8
  shape: [T_tf, V]
  sha256: "..."
  encoding:
    long: 1
    neutral: 0
    short: -1

timeline:
  time_unit: ms
  cadence_ms: 300000
  start_open_time_ms: 0
  end_close_time_ms: 0
  rows: T_tf

update:
  recompute_lookback_bars: 200
  recompute_from_open_time_ms: 0
```

Смысл `grid.order`: детерминированное соответствие "вариант -> индекс столбца".

Примечание по dtype сигналов:
- `int8` выбран потому, что значения сигналов — трёхзначные `{-1,0,1}` и должны быть mmap-friendly и Numba-friendly.
- "int1" как dtype в numpy нет; битовая упаковка возможна как отдельная оптимизация хранения, но добавляет слой кодирования/декодирования.

### Hit-times `manifest.yaml` (на уровне `hit_times/<timeframe>/`)

```yaml
schema_version: 1
kind: hit_times

identity:
  timeframe: "5m"

tp_values:
  path: tp_values.npy
  dtype: float32
  shape: [N_TP]
  sha256: "..."

sl_values:
  path: sl_values.npy
  dtype: float32
  shape: [N_SL]
  sha256: "..."

tables:
  long_tp:  {path: long_tp.u32.npy,  dtype: uint32, shape: [N_TP, T_tf], sha256: "..."}
  long_sl:  {path: long_sl.u32.npy,  dtype: uint32, shape: [N_SL, T_tf], sha256: "..."}
  short_tp: {path: short_tp.u32.npy, dtype: uint32, shape: [N_TP, T_tf], sha256: "..."}
  short_sl: {path: short_sl.u32.npy, dtype: uint32, shape: [N_SL, T_tf], sha256: "..."}

semantics:
  sentinel_exit_index: T_tf
  monotone_by_level: true
```

---

## Инкрементальное обновление (tail update)

Ежедневный precompute не пересобирает весь датасет. Он:
- пересчитывает хвост начиная с минимального времени из `manifest.yaml`,
- перезаписывает пересекающийся хвост,
- дописывает новые строки.

Где хранится "минимальное время обновления":
- Для цен: `prices_1m.update.recompute_from_open_time_ms` в root `manifest.yaml`.
- Для сигналов: `update.recompute_from_open_time_ms` в `signals/<tf>/<indicator_id>/manifest.yaml`.

Как выбирается хвост для сигналов:
- Для каждого `indicator_id` и `timeframe` manifest хранит `update.recompute_lookback_bars`.
- `recompute_lookback_bars` должен покрывать warmup всех вариантов grid (например, для MA с `window=5..200` это `200` баров на выбранном TF).
- На каждом daily update `recompute_from_open_time_ms` обновляется так, чтобы следующий запуск мог начать пересчет с корректным warmup (обычно: "конец данных минус lookback").

Пример (как я понял твою идею):
- Есть существующий `signals/5m/ma.sma/signals.i8.npy` до `end_close_time_ms`.
- `recompute_lookback_bars=200`.
- При очередном daily update добавились новые бары 5m.
- Мы считаем `recompute_from_open_time_ms = (new_end_close_time_ms - 200 * 5m)` (с выравниванием по bucket-open выбранного TF),
  пересчитываем сигналы на отрезке `[recompute_from, new_end]` для полного grid,
  перезаписываем хвост и дописываем новые строки.

Это обеспечивает, что rolling/lag-логика (окна/дельты/пороговые правила) корректна на границе обновления без полного пересчета истории.

## Milestone R0 — Контракты артефактов (YAML manifests + loader/validator)

Цель: зафиксировать форматы и сделать базовую инфраструктуру чтения/валидации.

EPIC R0-01 — Artifact store API (FS) + path builder
- Ввести единый builder путей по координатам (`exchange/market_type/symbol`).
- Реализовать FS-store: read-only load + list timeframes/indicator_ids (опционально).

EPIC R0-02 — YAML manifests + schema v1
- Ввести root `manifest.yaml`, signal `manifest.yaml`, hit-times `manifest.yaml`.
- Ввести `schema_version` и fail-fast валидацию структуры.

EPIC R0-03 — Validators
- Валидатор `prices_1m.npy`:
  - dtype/shape,
  - обязательные колонки time/OHLCV,
  - монотонность времени.
- Валидатор signal artifacts:
  - dtype `int8`, значения в `{-1,0,1}`,
  - `shape[0] == T_tf`.
- Валидатор hit-time artifacts:
  - dtype `uint32`, формы,
  - монотонность по уровням,
  - sentinel semantics.

Paths (планируемые):
- `src/trading/contexts/backtest_v2/**`
- `src/trading/contexts/backtest_v2/adapters/outbound/artifacts_fs/**`

DoD:
- Можно проверить целостность артефактного каталога одной командой (fail-fast).

---

## Milestone R1 — Precompute: export prices `1m`

Цель: поддерживать базовые цены `1m` как растущий артефакт и ежедневно обновлять только хвост.

EPIC R1-01 — Extract canonical `1m` -> `prices/prices_1m.npy`
- Источник правды: `market_data.canonical_candles_1m` через существующие reader/ACL.
- Сохранить `prices_1m.npy` (OHLCV+time) + `prices_1m_columns.npy`.
- Инкрементально обновлять хвост:
  - старт пересчета берем из `manifest.yaml` (`prices_1m.update.recompute_from_open_time_ms`),
  - перезаписываем пересекающийся хвост и дописываем новые строки до конца `asof_date`.
- Обновить root `manifest.yaml` (`identity.asof_date`, `prices_1m.data_range.*`, `prices_1m.update.*`).

EPIC R1-02 — T-1 policy + scheduler job
- Daily job обновляет артефакты до конца `asof_date = today_utc - 1 day`.

DoD:
- Sync backtest может загрузить `prices_1m.npy` и построить derived candles через rollup.

---

## Milestone R2 — Precompute: export signals (all indicators, per TF, per indicator)

Цель: по ценам `1m` ежедневно предрасчитывать дискретные сигналы `{-1,0,1}` для полного grid всех индикаторов из `configs/prod/indicators.yaml`.

EPIC R2-01 — Rollup reuse/normalize (для compute)
- Переиспользовать существующий rollup (backtest/strategy), при необходимости выделить общий модуль.
- Rollup используется:
  - в precompute (для расчета сигналов и hit-times на TF > 1m),
  - в sync runner (для построения свечей TF на вход engine v2).

EPIC R2-02 — Signal rules engine (v1)
- Реализовать вычисление сигналов по `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`.
- Спецификация rule families живет в docs, runtime реализация в коде.

EPIC R2-03 — Signal params grid
- Поддержать signal params ranges (thresholds/delta periods/etc) в `configs/prod/indicators.yaml` по форме из `backtest-signals-from-indicators-v1.md`.

EPIC R2-04 — Export format
- Для каждого `timeframe` и `indicator_id`:
  - сохранить `signals.i8.npy` shape `[T_tf, V]` (V = число вариантов grid),
  - сохранить `signals/<tf>/<indicator_id>/manifest.yaml` с grid/order + sha256.
  - инкрементально обновлять хвост начиная с `update.recompute_from_open_time_ms`:
    - пересчитываем сигналы на tail-window, достаточном для warmup всех вариантов grid,
    - перезаписываем пересекающийся хвост и дописываем новые строки.

DoD:
- Для любого `indicator_id` из `configs/prod/indicators.yaml` можно загрузить сигналовую матрицу по координатам пути.

---

## Milestone R3 — Precompute: hit-time grids (TP/SL)

Цель: считать hit-time grids и сохранять как mmap-friendly `uint32` таблицы.

EPIC R3-01 — TP/SL values
- Уровни TP/SL фиксируются как явные массивы `tp_values.npy` и `sl_values.npy`.

EPIC R3-02 — Hit-time compute + export
- Вынести из `tests/notebook_tests/05_hit_time_grid.ipynb` в production код:
  - вычисление `long_tp/long_sl/short_tp/short_sl`,
  - export через `np.lib.format.open_memmap`.

EPIC R3-03 — Hit-times manifest + validators
- Записать `hit_times/<tf>/manifest.yaml` и валидировать монотонность.

DoD:
- Engine v2 может загрузить hit-time grids и использовать их для fast grid-search.

---

## Milestone R4 — Backtest Engine v2: Sync runner (замена v1)

Цель: реализовать engine v2 и сделать `POST /backtests` sync путем по артефактам.

EPIC R4-01 — Engine v2 core
- Реализовать алгоритм из:
  - `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
  - `tests/notebook_tests/06_backtest_compute.ipynb`
- Адаптировать входы к формату "prices отдельно, signals отдельно":
  - загружать нужные `signals.i8.npy` для выбранных индикаторов,
  - собирать нужные матрицы/представления в памяти только для конкретного запроса.

EPIC R4-02 — Sync API integration
- `POST /backtests` использует engine v2 и artifacts loader.
- Guards/422 ошибки остаются детерминированными.

EPIC R4-03 — Удаление v1 path
- Удалить/закрыть кодовые пути старого engine v1 после прохождения тестов и golden fixtures.

DoD:
- Один и тот же запрос + один и тот же набор артефактов -> один и тот же результат.
- ClickHouse не используется на hot path sync backtest.

---

## Milestone R5 — Job runner (вторым этапом)

Цель: job runner использует тот же engine v2 и те же артефакты.

EPIC R5-01 — Jobs request snapshot
- В job snapshot хранить:
  - координаты артефакта (`exchange/market_type/symbol`),
  - cutoff `asof_date` (или `end_close_time_ms`) на момент создания job, чтобы результат был воспроизводим даже если артефакты обновятся.

EPIC R5-02 — Worker integration
- Job-runner загружает артефакты и запускает engine v2.

DoD:
- Jobs path дает те же результаты, что и sync, при одинаковых входах.

---

## Milestone R6 — Эксплуатация (retention)

EPIC R6-01 — Retention/GC
- Политика хранения снапшотов (N дней).
