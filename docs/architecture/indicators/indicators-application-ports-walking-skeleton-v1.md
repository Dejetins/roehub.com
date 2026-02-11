# Indicators — Порты application + доменный walking skeleton v1

Этот документ является source of truth для **IND-EPIC-01**.
Он определяет минимальную доменную модель, application DTO и application ports
для bounded context `indicators`.

Область охвата:
- включаются только domain-контракты, DTO и порты;
- без реализации compute (Numba/Numpy kernels), без API wiring, без адаптеров;
- свечи читаются через application port, а не напрямую из инфраструктуры.

## Ключевые решения

### 1) Неизменяемые определения индикаторов и версионирование
Формулы индикаторов неизменяемы. Любое изменение формулы должно приводить к новой
идентичности (например, `sma_v2`) или к явно версионированному идентификатору.

### 2) Жёсткие границы + step для параметров
Каждый параметр имеет объявленный тип (`int`, `float`, `enum`) и правила валидации:
- числовые параметры: `hard_min`, `hard_max`, `step` (`step` должен быть положительным);
- enum-параметры: непустой набор enum-значений, без числовых границ.

### 3) Тензорный output + явный layout
Результат compute — это `IndicatorTensor` с `float32` значениями и явным layout:
- `TIME_MAJOR`
- `VARIANT_MAJOR`

### 4) Guard от комбинаторного взрыва
`variants = product(axis lengths)` ограничивается guard.
Дефолтный guard в этом epic: `600_000` вариантов.
При превышении поток должен завершаться ошибкой валидации grid.

### 5) Политика распространения NaN
`CandleFeed` возвращает плотные 1m-массивы. Отсутствующие свечи представлены через `NaN`
в OHLCV-массивах. Compute должен сохранять/распространять NaN и не должен делать импутацию.

### 6) Контракт на float32 output
`IndicatorTensor.values` должен быть `float32`.
Любая внутренняя временная повышенная точность — detail реализации вне этого epic.

### 7) Warmup в compute-контракте
`IndicatorCompute` должен предоставлять `warmup()` для bootstrap движка.
Это требование уровня контракта даже до появления реализации.

## Обзор доменной модели

### Entities / Value Objects
- `IndicatorId`: стабильный идентификатор индикатора.
- `Layout`: ориентация layout тензора в памяти (`TIME_MAJOR`, `VARIANT_MAJOR`).
- `ParamKind`: тип параметра (`INT`, `FLOAT`, `ENUM`).
- `InputSeries`: поддерживаемые логические метки входных рядов.
- `OutputSpec`: декларация выхода (логические имена/компоненты выхода).
- `ParamDef`: определение параметра с инвариантами.
- `AxisDef`: материализованные значения оси (ровно один из int/float/enum наборов значений).
- `IndicatorDef`: полный контракт индикатора (`inputs`, `params`, `axes`, `output`).

### Specifications
- `GridParamSpec` (контракт): полиморфная спецификация grid-параметра.
- `ExplicitValuesSpec`: явные значения оси.
- `RangeValuesSpec`: инклюзивная материализация `start..stop` с положительным step.
- `GridSpec`: grid-запрос уровня индикатора (`params`, optional `source`, hint по layout).

### Доменные ошибки
- `UnknownIndicatorError`
- `GridValidationError`
- `MissingInputSeriesError`

## Обзор DTO (Application)

### CandleArrays
Плотные 1m-массивы свечей для compute:
- `ts_open: int64[1d]`
- `open/high/low/close/volume: float32[1d]`
- метаданные shared kernel: `MarketId`, `Symbol`, `TimeRange`, `Timeframe`

Инварианты:
- все массивы 1D и одной длины;
- dtype OHLCV — `float32`;
- dtype timestamp — `int64`;
- timestamps стабильно отсортированы.

### ComputeRequest
Входной envelope для compute:
- `candles: CandleArrays`
- `grid: GridSpec`
- `max_variants_guard: int`
- `dtype: "float32"` (фиксированное значение в v1)

### EstimateResult
Выход estimate:
- `indicator_id`
- материализованные `axes`
- `variants`
- `max_variants_guard`

### IndicatorTensor
Выход compute:
- `indicator_id`
- `layout`
- `axes`
- `values: np.ndarray[float32]`
- `meta` (`t`, `variants`, `nan_policy`, optional `compute_ms`)

## Порты application

### IndicatorRegistry
Контракт:
- `list_defs() -> tuple[IndicatorDef, ...]`
- `get_def(indicator_id: IndicatorId) -> IndicatorDef`

Семантика ошибок:
- `get_def` может выбрасывать `UnknownIndicatorError`.

### CandleFeed
Контракт:
- `load_1m_dense(market_id: MarketId, symbol: Symbol, time_range: TimeRange) -> CandleArrays`

Семантика ошибок:
- может выбрасывать `MissingInputSeriesError`, если запрошенный ряд нельзя получить.

### IndicatorCompute
Контракт:
- `estimate(grid: GridSpec, *, max_variants_guard: int) -> EstimateResult`
- `compute(req: ComputeRequest) -> IndicatorTensor`
- `warmup() -> None`

Семантика ошибок:
- `estimate`/`compute` могут выбрасывать `GridValidationError`;
- `compute` может выбрасывать `UnknownIndicatorError` и `MissingInputSeriesError`.

## Целевое размещение в репозитории

### Domain
- `src/trading/contexts/indicators/domain/entities/`
- `src/trading/contexts/indicators/domain/specifications/`
- `src/trading/contexts/indicators/domain/errors/`

### Application
- `src/trading/contexts/indicators/application/dto/`
- `src/trading/contexts/indicators/application/ports/registry/`
- `src/trading/contexts/indicators/application/ports/compute/`
- `src/trading/contexts/indicators/application/ports/feeds/`

### Tests
- `tests/unit/contexts/indicators/domain/`
- `tests/unit/contexts/indicators/application/dto/`

---

## Порты

Контракты портов определяются в application-слое.
Имена и семантика не зависят от конкретной реализации (Numba/Numpy, ClickHouse и т.д.).

### IndicatorRegistry

**Назначение**
`IndicatorRegistry` — источник описаний индикаторов (library).
Используется UI/use-cases для получения списка доступных индикаторов и их hard bounds.

**Контракт**
- `list_defs() -> tuple[IndicatorDef, ...]`
- `get_def(indicator_id: IndicatorId) -> IndicatorDef`

**Семантика**
- список детерминирован и стабилен при фиксированной версии кода;
- `get_def` выбрасывает доменную ошибку `UnknownIndicatorError`, если индикатор не существует.

---

### CandleFeed

**Назначение**
`CandleFeed` — порт чтения канонических свечей из market_data в формате dense arrays.
Порт не зависит от ClickHouse/таблиц/биржи: это детали адаптера.

**Контракт**
- `load_1m_dense(market_id: MarketId, symbol: Symbol, time_range: TimeRange) -> CandleArrays`

**Семантика**
- возвращает свечи для полуинтервала `[time_range.start, time_range.end)`;
- возвращает **плотную** временную сетку для `timeframe=1m`;
- пропуски заполняются NaN в OHLCV;
- SHOULD: массивы отсортированы по времени по возрастанию.

**Инварианты**
- `len(open)==len(close)==...==len(ts_open)`;
- `timeframe` соответствует requested (1m в v1).

---

### IndicatorCompute

**Назначение**
`IndicatorCompute` — порт вычисления индикаторов по свечам и grid.
Реализация может быть на Numba (основная) или Numpy (референсная).

**Контракт**
- `estimate(grid: GridSpec, *, max_variants_guard: int) -> EstimateResult`
- `compute(req: ComputeRequest) -> IndicatorTensor`
- `warmup() -> None`

**Семантика**
- `estimate`:
  - строит фактические оси (`AxisDef`) из grid;
  - считает variants;
  - применяет guard `variants <= max_variants_guard` (иначе ошибка валидации).
- `compute`:
  - должен повторять семантику `estimate` (оси и variants совпадают);
  - возвращает `IndicatorTensor(values=float32, nan_policy=propagate)`.
- `warmup`:
  - прогревает ключевые kernels/код paths (implementation detail);
  - MUST вызываться при старте процесса (composition root решает когда).

**Ошибки**
- `GridValidationError` (выход за bounds, пустые оси, неверный step, variants > guard);
- `UnknownIndicatorError`;
- `MissingInputSeriesError` (если индикатор требует high/low/volume, а candles не содержат — в v1 candles всегда содержит OHLCV, но оставляем ошибку как контракт).

---

## Заметки для будущих EPIC (не входит в v1)
- CachePort (`indicators.application.ports.cache`) появится, когда будет >=2 реализации кеша или >=2 потребителя.
- YAML defaults (`configs/*/indicators.yaml`) валидируются против `IndicatorDef` в EPIC-02.
- Выбор default layout фиксируется в EPIC-10 (benchmark).
- API endpoints (`/indicators`, `/indicators/estimate`, `/indicators/compute`) реализуются позже (EPIC-02/03).

---

## Целевое размещение в репозитории (guidance)

Следующие пути соответствуют текущему дереву репозитория:

- Domain:
  - `src/trading/contexts/indicators/domain/entities/` (IndicatorDef, ParamDef, AxisDef, IndicatorId, Layout)
  - `src/trading/contexts/indicators/domain/specifications/` (GridSpec, GridParamSpec)
  - `src/trading/contexts/indicators/domain/errors/` (UnknownIndicatorError, GridValidationError, ...)

- Application DTO:
  - `src/trading/contexts/indicators/application/dto/` (CandleArrays, ComputeRequest, EstimateResult, IndicatorTensor)

- Порты application:
  - `src/trading/contexts/indicators/application/ports/registry/` (IndicatorRegistry)
  - `src/trading/contexts/indicators/application/ports/compute/` (IndicatorCompute)
  - `src/trading/contexts/indicators/application/ports/feeds/` (CandleFeed) *(если feeds-подпапки нет — создаётся в EPIC-01)*

Примечание: в текущем дереве у `indicators/application/ports/` уже есть `cache/compute/registry/`.
Порт `CandleFeed` в v1 добавляется как новый раздел (`feeds/`), по аналогии с другими контекстами.
