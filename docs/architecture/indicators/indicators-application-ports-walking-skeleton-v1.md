# Indicators — Application Ports + Domain Walking Skeleton v1

This document is the source of truth for **IND-EPIC-01**.
It defines the minimal domain model, application DTOs, and application ports
for the `indicators` bounded context.

Scope:
- include domain contracts, DTOs, and ports only;
- no compute implementation (Numba/Numpy kernels), no API wiring, no adapters;
- candles are read through an application port, not directly from infrastructure.

## Key Decisions

### 1) Immutable indicator definitions and versioning
Indicator formulas are immutable. Any formula change must produce a new identity
(for example `sma_v2`) or an explicit versioned identifier.

### 2) Hard bounds + step for parameters
Each parameter has a declared kind (`int`, `float`, `enum`) and validation rules:
- numeric params: `hard_min`, `hard_max`, `step` (step must be positive);
- enum params: non-empty enum values, no numeric bounds.

### 3) Tensor output + explicit layout
Compute output is an `IndicatorTensor` with `float32` values and explicit layout:
- `TIME_MAJOR`
- `VARIANT_MAJOR`

### 4) Guard against combinatorial explosion
`variants = product(axis lengths)` is limited by guard.
Default guard in this epic: `600_000` variants.
If exceeded, the flow must fail with grid validation error.

### 5) NaN propagation policy
`CandleFeed` returns dense 1m arrays. Missing candles are represented by `NaN`
in OHLCV arrays. Compute must preserve/propagate NaN and does not impute values.

### 6) Float32 output contract
`IndicatorTensor.values` must be `float32`.
Any internal temporary precision is an implementation detail outside this epic.

### 7) Warmup in compute contract
`IndicatorCompute` must expose `warmup()` for engine bootstrap.
This is a contract-level requirement even before implementation exists.

## Domain Model Overview

### Entities / Value Objects
- `IndicatorId`: stable indicator identifier.
- `Layout`: tensor memory/layout orientation (`TIME_MAJOR`, `VARIANT_MAJOR`).
- `ParamKind`: parameter kind (`INT`, `FLOAT`, `ENUM`).
- `InputSeries`: supported logical input series labels.
- `OutputSpec`: output declaration (logical output names/components).
- `ParamDef`: parameter definition with invariants.
- `AxisDef`: materialized axis values (exactly one of int/float/enum values).
- `IndicatorDef`: full indicator contract (`inputs`, `params`, `axes`, `output`).

### Specifications
- `GridParamSpec` (contract): polymorphic parameter grid specification.
- `ExplicitValuesSpec`: explicit axis values.
- `RangeValuesSpec`: inclusive `start..stop` materialization with positive step.
- `GridSpec`: indicator-level grid request (`params`, optional `source`, layout hint).

### Domain Errors
- `UnknownIndicatorError`
- `GridValidationError`
- `MissingInputSeriesError`

## DTO Overview (Application)

### CandleArrays
Dense 1m candle arrays used by compute:
- `ts_open: int64[1d]`
- `open/high/low/close/volume: float32[1d]`
- shared kernel metadata: `MarketId`, `Symbol`, `TimeRange`, `Timeframe`

Invariants:
- all arrays are 1D and equal length;
- OHLCV dtype is `float32`;
- timestamp dtype is `int64`;
- timestamps are stably ordered.

### ComputeRequest
Compute input envelope:
- `candles: CandleArrays`
- `grid: GridSpec`
- `max_variants_guard: int`
- `dtype: "float32"` (v1 fixed value)

### EstimateResult
Estimate output:
- `indicator_id`
- materialized `axes`
- `variants`
- `max_variants_guard`

### IndicatorTensor
Compute output:
- `indicator_id`
- `layout`
- `axes`
- `values: np.ndarray[float32]`
- `meta` (`t`, `variants`, `nan_policy`, optional `compute_ms`)

## Application Ports

### IndicatorRegistry
Contract:
- `list_defs() -> tuple[IndicatorDef, ...]`
- `get_def(indicator_id: IndicatorId) -> IndicatorDef`

Error semantics:
- `get_def` may raise `UnknownIndicatorError`.

### CandleFeed
Contract:
- `load_1m_dense(market_id: MarketId, symbol: Symbol, time_range: TimeRange) -> CandleArrays`

Error semantics:
- may raise `MissingInputSeriesError` when requested series cannot be produced.

### IndicatorCompute
Contract:
- `estimate(grid: GridSpec, *, max_variants_guard: int) -> EstimateResult`
- `compute(req: ComputeRequest) -> IndicatorTensor`
- `warmup() -> None`

Error semantics:
- `estimate`/`compute` may raise `GridValidationError`;
- `compute` may raise `UnknownIndicatorError` and `MissingInputSeriesError`.

## Target Repository Placement

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

## Ports

Контракты port-ов определяются в application-слое.
Имена и семантика не зависят от конкретной реализации (Numba/Numpy, ClickHouse и т.д.).

### IndicatorRegistry

**Purpose**
`IndicatorRegistry` — источник описаний индикаторов (library).
Используется UI/use-cases для получения списка доступных индикаторов и их hard bounds.

**Contract**
- `list_defs() -> tuple[IndicatorDef, ...]`
- `get_def(indicator_id: IndicatorId) -> IndicatorDef`

**Semantics**
- список детерминирован и стабилен при фиксированной версии кода
- `get_def` бросает доменную ошибку `UnknownIndicatorError` если индикатор не существует

---

### CandleFeed

**Purpose**
`CandleFeed` — порт чтения канонических свечей из market_data в формате dense arrays.
Порт не зависит от ClickHouse/таблиц/биржи: это детали адаптера.

**Contract**
- `load_1m_dense(market_id: MarketId, symbol: Symbol, time_range: TimeRange) -> CandleArrays`

**Semantics**
- возвращает свечи для полуинтервала `[time_range.start, time_range.end)`
- возвращает **плотную** временную сетку для `timeframe=1m`
- пропуски заполняются NaN в OHLCV
- SHOULD: массивы отсортированы по времени по возрастанию

**Invariants**
- `len(open)==len(close)==...==len(ts_open)`
- `timeframe` соответствует requested (1m в v1)

---

### IndicatorCompute

**Purpose**
`IndicatorCompute` — порт вычисления индикаторов по свечам и grid.
Реализация может быть на Numba (основная) или Numpy (референсная).

**Contract**
- `estimate(grid: GridSpec, *, max_variants_guard: int) -> EstimateResult`
- `compute(req: ComputeRequest) -> IndicatorTensor`
- `warmup() -> None`

**Semantics**
- `estimate`:
  - строит фактические оси (AxisDef) из grid,
  - считает variants,
  - применяет guard `variants <= max_variants_guard` (иначе ошибка валидации)
- `compute`:
  - должен повторять семантику `estimate` (оси и variants совпадают),
  - возвращает `IndicatorTensor(values=float32, nan_policy=propagate)`
- `warmup`:
  - прогревает ключевые kernels/код paths (implementation detail),
  - MUST вызываться при старте процесса (composition root решает когда).

**Errors**
- `GridValidationError` (выход за bounds, пустые оси, неверный step, variants > guard)
- `UnknownIndicatorError`
- `MissingInputSeriesError` (если индикатор требует high/low/volume, а candles не содержат — в v1 candles всегда содержит OHLCV, но оставляем ошибку как контракт)

---

## Notes for future EPICs (не входит в v1)
- CachePort (`indicators.application.ports.cache`) появится, когда будет >=2 реализации кеша или >=2 потребителя.
- YAML defaults (configs/*/indicators.yaml) валидируются против `IndicatorDef` в EPIC-02.
- Выбор default layout фиксируется в EPIC-10 (benchmark).
- API endpoints (`/indicators`, `/indicators/estimate`, `/indicators/compute`) реализуются позже (EPIC-02/03).

---

## Target repo placement (guidance)

Следующие пути соответствуют текущему дереву репозитория:

- Domain:
  - `src/trading/contexts/indicators/domain/entities/` (IndicatorDef, ParamDef, AxisDef, IndicatorId, Layout)
  - `src/trading/contexts/indicators/domain/specifications/` (GridSpec, GridParamSpec)
  - `src/trading/contexts/indicators/domain/errors/` (UnknownIndicatorError, GridValidationError, ...)

- Application DTO:
  - `src/trading/contexts/indicators/application/dto/` (CandleArrays, ComputeRequest, EstimateResult, IndicatorTensor)

- Application Ports:
  - `src/trading/contexts/indicators/application/ports/registry/` (IndicatorRegistry)
  - `src/trading/contexts/indicators/application/ports/compute/` (IndicatorCompute)
  - `src/trading/contexts/indicators/application/ports/feeds/` (CandleFeed)  *(если feeds-подпапки нет — создаётся в EPIC-01)*

Примечание: в текущем дереве у `indicators/application/ports/` уже есть `cache/compute/registry/`.
Порт `CandleFeed` в v1 добавляется как новый раздел (`feeds/`), по аналогии с другими контекстами.
