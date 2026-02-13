# Runbook — Troubleshooting: why NaN?

## Назначение

Документ объясняет, откуда берутся `NaN` в `indicators` и как отличить корректную политику от дефекта.

Связанный контекст:
- `docs/architecture/indicators/indicators-overview.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`

## Основные причины NaN

1. Warmup окна: для rolling-индикаторов первые точки (`t < window-1`) всегда `NaN`.
2. NaN holes из CandleFeed: пропущенные свечи в dense timeline дают `NaN` во входах.
3. Reset-on-NaN для stateful цепочек (EMA/RMA/PSAR/SuperTrend и др.).
4. Деление на ноль (например, `sum(volume)==0`, `range==0`, `atr==0`).
5. Недостаточный горизонт данных для lag/shift/confirm-окон.

## Пошаговая диагностика

### 1) Проверить, что indicator_id корректно объявлен

```bash
rg -n "<indicator_id>" docs/architecture/indicators/indicators_formula.yaml
rg -n "<indicator_id>" configs/prod/indicators.yaml
```

Ожидание:
- ID есть в formula spec.
- для боевого использования есть defaults в `configs/prod/indicators.yaml`.

### 2) Проверить тип NaN-паттерна

- NaN только в начале ряда и длина совпадает с warmup -> обычно норма.
- NaN "островами" внутри ряда -> обычно входные пропуски или reset stateful цепочки.
- NaN на всём диапазоне -> обычно отсутствует нужная входная серия или постоянный div-by-zero.

### 3) Проверить входные данные на NaN holes

```bash
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/feeds/test_market_data_acl_candle_feed.py
```

Ожидание:
- тесты подтверждают ожидаемую dense+NaN семантику.

### 4) Проверить конкретную группу индикаторов

```bash
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_ma_kernels.py
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volatility_kernels.py
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_momentum_kernels.py
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volume_kernels.py
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_structure_kernels.py
```

Ожидание:
- поведение NaN совпадает с зафиксированной политикой в тестах и EPIC docs.

### 5) Проверить, не сломан ли runtime warmup/config

```bash
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_runtime_wiring.py
```

Ожидание:
- нет ошибок warmup/runtime config.

## Частые ситуации

- `bbands_percent_b` или `cmf` возвращают NaN на отдельных интервалах:
  - проверьте нулевой знаменатель (`upper-lower`, `sum(volume)`).
- `rsi`/`macd` дают длинную NaN-полосу после дырки:
  - это ожидаемый reset-on-NaN + повторный warmup.
- `pivots` дают поздние значения:
  - это confirm-semantics (`left/right`), не ошибка.

## FAQ

Q: Можно ли автоматически заменять NaN на 0 внутри compute?
A: Нет. В v1 NaN policy фиксирована и должна быть прозрачной, чтобы не скрывать проблемные участки данных.

Q: Почему NumPy и Numba дают одинаковый NaN-паттерн?
A: NumPy oracle используется как эталон семантики, поэтому расхождение считается багом.

Q: Можно ли убрать NaN в начале ряда?
A: Только уменьшив окно/лаг параметров или изменив time range. Это не runtime defect.
