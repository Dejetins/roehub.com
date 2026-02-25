# Backtest v1 — Grid Builder + Staged Runner + Sync Guards (BKT-EPIC-04)

Фиксирует контракт BKT-EPIC-04: как детерминированно строится grid вариантов для sync backtest v1 (Stage A/Stage B), какие guards применяются, и почему API возвращает только top-K результатов.

## Цель

- Построить детерминированный grid вариантов backtest v1 из диапазонов/шагов `configs/<env>/indicators.yaml` (compute params + signal params) и risk-осей SL/TP из request.
- Зафиксировать staged pipeline v1: Stage A (base grid без SL/TP) -> shortlist `preselect` -> Stage B (expand по SL/TP) -> top-K.
- Зафиксировать guards для sync режима:
  - `MAX_VARIANTS_PER_COMPUTE_DEFAULT = 600_000`
  - `MAX_COMPUTE_BYTES_TOTAL_DEFAULT = 5 GiB`
- Зафиксировать output policy v1: в ответе возвращаем только top-K, сортировка стабильна и воспроизводима.

## Контекст

- Milestone 4 вводит sync backtest v1 по одному инструменту, close-fill, с multi-variant grid (параметры индикаторов + signal params + SL/TP) и рейтингом по `Total Return [%]`.
- В репозитории уже существует детерминированная materialization/estimate инфраструктура в контексте `indicators`:
  - `GridBuilder` + `BatchEstimator` + guards (`600k`, `5 GiB`) в `src/trading/contexts/indicators/application/services/grid_builder.py`.
- Signal rules и источник signal params ranges фиксируются в BKT-EPIC-03:
  - `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`.
- Backtest v1 уже имеет skeleton контракты variant identity (`variant_key`) и runtime defaults (`top_k_default`, `preselect_default`):
  - `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md`.

Ключевое ограничение sync v1:

- мы не возвращаем все варианты; мы возвращаем только top-K.
- мы не делаем advanced pruning/streaming top-k (это Milestone 6).

## Scope

- Grid variants generation:
  - materialization `explicit`/`range(start/stop_incl/step)` из YAML defaults и/или request overrides,
  - декартово произведение по параметрам выбранных индикаторов (subsets не перебираем),
  - signal params участвуют в base grid (Stage A),
  - детерминированный порядок вариантов.
- Risk grid:
  - SL/TP входят как оси grid в Stage B и увеличивают варианты мультипликативно,
  - шаги SL/TP задаются в request (UI управляет step),
  - SL/TP значения трактуются как проценты: `3.0 == 3%`,
  - разрешено отключать SL и/или TP отдельным флагом.
- Staged runner config v1:
  - Stage A: base grid (индикаторы + signal params; без SL/TP) -> shortlist `preselect` по `Total Return [%]`.
  - Stage B: expand shortlist по SL/TP -> точный расчёт метрик -> top-K.
- Guards для sync:
  - variants guard: `<= 600_000` (Stage A и Stage B),
  - memory guard: `<= 5 GiB` (оценка compute-памяти для candles + indicator tensors).
- Output policy v1:
  - API возвращает только top-K (default `top_k_default=300`),
  - `preselect_default=20_000` (Stage A shortlist),
  - ranking key: `Total Return [%]` (desc), tie-break: stable `variant_key`.

## Update 2026-02-25 (Perf Phase 3)

С 2026-02-25 staged scoring path реализован через shared core и batched tensor prepare:

- Sync и jobs используют общий `BacktestStagedCoreRunnerV1` для Stage A/Stage B ranking.
- `CloseFillBacktestStagedScorerV1.prepare_for_grid_context(...)` заранее вычисляет indicator tensors по natural Stage-A grid и переиспользует их в Stage A и Stage B.
- `IndicatorCompute.compute(...)` больше не вызывается per-variant в Stage A/Stage B hot path при доступном prepared context.
- Кооперативная отмена добавлена на границах длинных циклов (Stage A/Stage B checkpoints) через `BacktestRunControlV1` (sync) и lease/cancel checks (jobs).

Guard budgets разделены по режимам:

- Sync wiring (`apps/api/wiring/modules/backtest.py`) использует half-budget:
  - `max_variants_per_compute = floor(backtest.guards.max_variants_per_compute / 2)`;
  - `max_compute_bytes_total = floor(backtest.guards.max_compute_bytes_total / 2)`.
- Jobs wiring (`apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`) использует full budget из `backtest.guards.*`.

## Non-goals

- Перебор подмножеств индикаторов (subsets).
- Асинхронные jobs/progress и сохранение результатов (Milestone 5).
- Streaming top-K/pruning/beam search (Milestone 6).
- Интрабарное исполнение SL/TP (Milestone 7).

## Ключевые решения

### 1) Источник grid ranges: request overrides + fallback на `configs/<env>/indicators.yaml` (1C)

Backtest принимает grid specs из request, но может строить их по defaults из `configs/<env>/indicators.yaml`:

- если UI прислал полный `GridSpec`/signal params spec -> используем его,
- если UI прислал только список индикаторов (или частичный payload) -> достраиваем отсутствующие оси из YAML defaults.

Это позволяет:

- держать серверный YAML как source-of-truth для допустимых диапазонов,
- не блокировать UX (UI может постепенно переезжать от “use defaults” к “custom grid”).

Последствия:

- нужен loader в backtest/сигнальном слое для `defaults.<indicator_id>.signals.v1.params.*` (см. BKT-EPIC-03).
- валидация осей compute остаётся в `indicators.GridBuilder` (hard bounds/step/enum).

### 2) Signal params входят в base grid (Stage A), но не влияют на memory estimate индикаторов

Варианты backtest зависят от:

- compute параметров индикаторов (как считается primary output),
- signal параметров (как primary output превращается в `LONG|SHORT|NEUTRAL`).

Stage A строит base grid именно по этим двум слоям.

При этом размер indicator tensors зависит только от compute grid (inputs/params), поэтому memory estimate v1:

- учитывает candles + indicator tensors (как в `BatchEstimator`),
- не умножается на signal params и SL/TP оси.

Ссылка: `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`.

### 3) Staged pipeline v1: guard применяем к фактической нагрузке Stage A и Stage B (а не к full expanded grid)

Stage A реально исполняет base grid (без SL/TP). Stage B реально исполняет только shortlist, расширенный по SL/TP.

Поэтому variants guard проверяем так:

- `stage_a_variants = base_variants_total` (compute x signals) и требуем `<= MAX_VARIANTS_PER_COMPUTE_DEFAULT`.
- `stage_b_variants = shortlist_len * sl_variants * tp_variants` и требуем `<= MAX_VARIANTS_PER_COMPUTE_DEFAULT`.

Где:

- `shortlist_len = min(preselect, stage_a_variants)`.
- если SL/TP отключены флагами, соответствующая ось имеет длину `1`.

Почему не проверяем `stage_a_variants * sl_variants * tp_variants`:

- этот full product никогда не вычисляется в v1 staged pipeline.

### 4) Детерминизм variants и stable ordering

Фиксируем детерминизм на трёх уровнях:

1) **Materialization осей**
   - `explicit`: порядок значений сохраняется как в request.
   - `range`: значения материализуются детерминированно по формуле индекса (`start + i*step`, inclusive).

2) **Декартово произведение**
   - набор индикаторов фиксирован пользователем; subsets не перебираем.
   - порядок индикаторов для grid считаем детерминированным (рекомендуемый выбор: сортировка по `indicator_id`).

3) **Stable sort в staged pipeline**
   - Stage A shortlist сортируется стабильно по `Total Return [%] desc`, tie-break по `base_variant_key`.
   - Stage B top-K сортируется стабильно по `Total Return [%] desc`, tie-break по `variant_key`.

Tie-break по ключу нужен, чтобы результат не зависел от порядка обхода/параллелизма при одинаковом `Total Return [%]`.

### 5) Variant keys: `indicator_variant_key` остаётся compute-only, а `backtest.variant_key` включает `signals`

Фиксируем (утверждено):

- `indicators.build_variant_key_v1(...)` остаётся compute-only:
  - инструмент + timeframe + явные compute selections (inputs/params) по каждому `indicator_id`.
- `backtest.build_backtest_variant_key_v1(...)` расширяется и включает отдельное поле `signals`:

Пример payload v1 (canonical JSON перед sha256):

```json
{
  "schema_version": 1,
  "indicator_variant_key": "<sha256>",
  "direction_mode": "long-short",
  "sizing_mode": "all_in",
  "signals": {
    "momentum.rsi": {"long_threshold": 30, "short_threshold": 70},
    "trend.adx": {"long_delta_periods": -5, "short_delta_periods": -10}
  },
  "risk": {
    "sl_enabled": true,
    "sl_pct": 3.0,
    "tp_enabled": false,
    "tp_pct": null
  },
  "execution": {
    "fee_pct": 0.075,
    "slippage_pct": 0.01
  }
}
```

Требования к детерминизму:

- ключи словарей `signals` сортируются детерминированно (по `indicator_id`, затем по имени параметра),
- значения scalars должны быть JSON-совместимыми (`int|float|str|bool|null`),
- `signals` НЕ встраивается в `indicator_variant_key`, чтобы не смешивать bounded contexts.

### 6) Guards и ошибки (sync)

Guards v1 фиксированы и переиспользуются из `indicators`:

- `MAX_VARIANTS_PER_COMPUTE_DEFAULT = 600_000`
- `MAX_COMPUTE_BYTES_TOTAL_DEFAULT = 5 GiB`

Политика ошибок:

- При превышении guards возвращаем `422` (`validation_error`) с понятным machine-readable `details`.
- Рекомендуемый payload (детали):
  - `error: "max_variants_per_compute_exceeded" | "max_compute_bytes_total_exceeded"`
  - `stage: "stage_a" | "stage_b" | "preflight"`
  - численные поля (`total_variants`, `estimated_memory_bytes`, лимиты).

Семантика memory estimate:

- Используем `BatchEstimator` как верхнюю оценку по `(T, indicator_variants_i)`.
- `T` должен соответствовать внутреннему диапазону расчёта (включая warmup), чтобы не занижать оценку.

## Контракты и инварианты

- Base grid (Stage A) включает:
  - compute params/inputs по выбранным индикаторам,
  - signal params (как отдельные оси),
  - без SL/TP (risk disabled).
- Risk grid (Stage B) включает SL/TP оси (проценты, `3.0 == 3%`), шаги задаёт request; SL/TP можно отключить флагом.
- Guards (sync):
  - `stage_a_variants <= 600_000`, иначе 422.
  - `stage_b_variants <= 600_000`, иначе 422.
  - `estimated_memory_bytes <= 5 GiB` (или конфиг), иначе 422.
- Output policy:
  - Stage A формирует shortlist длины `min(preselect, stage_a_variants)`.
  - Stage B возвращает `min(top_k, stage_b_variants)` результатов.
  - Результаты отсортированы детерминированно по `Total Return [%] desc`, tie-break по `variant_key asc`.
- Variant identity:
  - `indicator_variant_key` детерминированно идентифицирует compute selections,
  - `variant_key` детерминированно идентифицирует полный вариант backtest, включая `signals` и risk flags/значения.

## Связанные файлы

- `docs/architecture/roadmap/milestone-4-epics-v1.md` — EPIC map Milestone 4, BKT-EPIC-04.
- `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` — variant identity и runtime defaults.
- `docs/architecture/backtest/backtest-signals-from-indicators-v1.md` — signal params и их хранение в `configs/<env>/indicators.yaml`.
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md` — materialization/estimate/guards (600k/5GiB).
- `configs/prod/indicators.yaml` — compute ranges/steps; место для `defaults.<id>.signals.v1.params`.
- `configs/prod/backtest.yaml` — `top_k_default`, `preselect_default`, `warmup_bars_default`.
- `src/trading/contexts/indicators/application/services/grid_builder.py` — guard defaults + estimator.
- `src/trading/contexts/indicators/application/dto/variant_key.py` — `indicator_variant_key` builder.
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py` — `variant_key` builder (расширяется полем `signals`).

## Как проверить

```bash
# запускать из корня репозитория
uv run ruff check .
uv run pyright
uv run pytest -q

# обновить/проверить индекс документации
uv run python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: memory estimate учитывает только candles + indicator tensors (как в `indicators`), но не учитывает возможные крупные буферы backtest engine. В v1 снижаем риск большим reserve и консервативным `max_compute_bytes_total`.
- Риск: `configs/prod/indicators.yaml` пока не содержит `defaults.<id>.signals.v1.params`; требуется добавление и отдельный loader в backtest, чтобы не ломать indicators registry.
- Вопрос (на EPIC-07): точный API контракт `POST /backtests` (как именно передаются risk axes specs и enable flags) и формат возвращаемого результата (какие метрики, какие поля нужны UI для сохранения StrategySpec).
