# Indicators Overview

`indicators` в Milestone 2 отвечает за детерминированный расчёт индикаторных тензоров и за контракты, которые нужны для безопасного расширения библиотеки индикаторов без "магии".

## Навигация

- Архитектурный индекс: `docs/architecture/indicators/README.md`
- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults/bounds: `configs/prod/indicators.yaml`, `configs/dev/indicators.yaml`, `configs/test/indicators.yaml`
- Групповые документы: `indicators-ma.md`, `indicators-volatility.md`, `indicators-momentum.md`, `indicators-trend.md`, `indicators-volume.md`, `indicators-structure.md`
- Runbooks: `docs/runbooks/indicators-numba-warmup-jit.md`, `docs/runbooks/indicators-numba-cache-and-threads.md`, `docs/runbooks/indicators-why-nan.md`

## Архитектурный поток (registry -> grid -> candle feed -> guards -> compute -> tensor)

1. `IndicatorRegistry` поднимает hard definitions + YAML defaults и валидирует их fail-fast на старте.
2. `GridBuilder` материализует оси (`explicit` / `range`) в детерминированном порядке.
3. `CandleFeed` загружает dense `[start, end)` 1m timeline с NaN-holes.
4. Guards проверяют `variants` и бюджет памяти до больших аллокаций.
5. `NumbaIndicatorCompute` диспетчеризует в kernel-группу (`ma`, `volatility`, `momentum`, `trend`, `volume`, `structure`).
6. Возвращается `IndicatorTensor(values=float32, axes, layout, meta)`.

Ключевые документы по шагам:
- Registry: `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md`
- Grid/guards: `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
- Candle feed ACL: `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
- Compute core: `docs/architecture/indicators/indicators-compute-engine-core.md`

## Инварианты и политики (v1)

### NaN policy

- CandleFeed не лечит пропуски: missing candle -> `NaN` в OHLCV.
- Rolling-window операции: warmup-зона `t < window-1` и `NaN` при любом `NaN` внутри окна.
- Stateful цепочки (EMA/RMA и зависящие индикаторы): reset-on-NaN.
- Деление на 0 не маскируется: результат `NaN`.
- Подробная диагностика: `docs/runbooks/indicators-why-nan.md`.

### Warmup policy и startup fail-fast

- API стартап запускает `ComputeNumbaWarmupRunner`.
- Warmup применяет runtime config (`NUMBA_NUM_THREADS`, `NUMBA_CACHE_DIR`) и проверяет writable cache dir.
- При невалидной конфигурации сервис падает на старте (fail-fast), а не деградирует в рантайме.
- Операционный runbook: `docs/runbooks/indicators-numba-warmup-jit.md`.

### Dtype/layout policy

- `IndicatorTensor.values` всегда `float32`.
- Поддерживаются layout `TIME_MAJOR` и `VARIANT_MAJOR` без изменения семантики.
- Внутренние `float64` accumulator'ы допустимы как implementation detail, но не меняют публичный контракт.

### Детерминизм осей и вариантов

- Порядок групп во всей документации фиксирован: `ma`, `volatility`, `momentum`, `trend`, `volume`, `structure`.
- Внутри группы ID перечисляются лексикографически.
- Материализация осей и variant mapping детерминированы и не зависят от случайных факторов.
- `variant_key v1` semantics не изменяются без отдельного архитектурного решения.

## Source of truth

- Формулы/spec: `docs/architecture/indicators/indicators_formula.yaml`.
- Runtime defaults и рабочие диапазоны: `configs/prod/indicators.yaml`.
- Hard definitions и контракты параметров: `src/trading/contexts/indicators/domain/definitions/`.
- Compute реализация: `src/trading/contexts/indicators/adapters/outbound/compute_numba/` и `src/trading/contexts/indicators/adapters/outbound/compute_numpy/`.

## How to add a new indicator

Ниже обязательный чеклист для изменений без ломки публичных контрактов.

1. Добавить или обновить спецификацию в `docs/architecture/indicators/indicators_formula.yaml`.
2. Добавить/обновить hard definition в `src/trading/contexts/indicators/domain/definitions/<group>.py`.
3. Обновить defaults и bounds во всех окружениях:
   - `configs/prod/indicators.yaml`
   - `configs/dev/indicators.yaml`
   - `configs/test/indicators.yaml`
4. Реализовать Numba kernel в `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/<group>.py`.
5. Реализовать/обновить NumPy oracle в `src/trading/contexts/indicators/adapters/outbound/compute_numpy/<group>.py`.
6. Подключить dispatch в `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`.
7. При необходимости добавить warmup-path в `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`.
8. Добавить/обновить unit tests и, если нужно, perf-smoke:
   - `tests/unit/contexts/indicators/...`
   - `tests/perf_smoke/contexts/indicators/...`
9. Обновить документацию:
   - `docs/architecture/indicators/README.md`
   - `docs/architecture/indicators/indicators-overview.md`
   - `docs/architecture/indicators/indicators-<group>.md`
   - runbooks (`docs/runbooks/indicators-*.md`) при изменении NaN/warmup/runtime поведения.
10. Прогнать quality gates (ruff, pyright, pytest, compileall).

## Group docs (детерминированный порядок)

- MA: `docs/architecture/indicators/indicators-ma.md`
- Volatility: `docs/architecture/indicators/indicators-volatility.md`
- Momentum: `docs/architecture/indicators/indicators-momentum.md`
- Trend: `docs/architecture/indicators/indicators-trend.md`
- Volume: `docs/architecture/indicators/indicators-volume.md`
- Structure: `docs/architecture/indicators/indicators-structure.md`

## Runbooks

- Numba warmup / JIT: `docs/runbooks/indicators-numba-warmup-jit.md`
- Numba cache dir + threads (`NUMBA_NUM_THREADS`, `numba_cache_dir`): `docs/runbooks/indicators-numba-cache-and-threads.md`
- Troubleshooting: why NaN?: `docs/runbooks/indicators-why-nan.md`

## Глоссарий

- Grid: материализованные оси параметров/inputs для одного `indicator_id`.
- Axis: одно измерение grid (например `window`, `source`, `mult`).
- Variant: одна комбинация значений по всем осям.
- Warmup: первичный JIT compile kernels на старте процесса.
- Reset-on-NaN: при NaN входе stateful серия сбрасывает состояние и стартует заново на следующем валидном значении.
- Oracle: эталонная NumPy-реализация для сравнения с Numba в unit tests.
