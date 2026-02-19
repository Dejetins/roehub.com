# Milestone 4 — EPIC map (v1)

Карта EPIC'ов для Milestone 4: Backtest v1 (close-fill) по одному инструменту с multi-variant grid (комбинации индикаторов + диапазоны параметров), direction modes, 4 режима position sizing, close-based SL/TP, комиссии/slippage и расширенный отчёт метрик.

Этот документ декомпозирует Milestone 4 из `docs/architecture/roadmap/base_milestone_plan.md` в набор EPIC'ов, которые можно реализовывать последовательно.

## Контекст и новые вводные (зафиксировано)

### Execution model (v1)

- **Close-fill**: на закрытии бара пересчитываем индикаторы/сигналы и исполняем сделки на `close[t]`.
- **One instrument**: backtest работает по одному инструменту за запуск.
- **Spot + futures**: одинаковая упрощённая модель исполнения; funding/liq/borrow не моделируем.

### Warmup lookback (v1)

- Делаем lookback по истории по умолчанию: `warmup_bars_default = 200`.
- Warmup должен быть конфигурируемым.
- Если истории недостаточно (инструмент начинается позже, чем `Start - warmup_bars_default`) — начинаем с первого доступного бара.

### Direction modes (v1)

- `long-only`
- `short-only`
- `long-short` (с переворотом)

### Position sizing (v1)

4 режима:

- `all_in`
- `fixed_quote`
- `strategy_compound`
- `strategy_compound_profit_lock` (c параметром `safe_profit_percent`)

### Risk (v1)

- SL/TP: close-based, без intrabar.

### Fees/slippage (v1)

- Fee defaults: spot `0.075%`, futures `0.1%` (UI шаг `0.01%`).
- Slippage default: `0.01%` (UI шаг `0.01%`).

### Signals-from-indicators (v1)

- Каждый индикатор выдаёт дискретный сигнал `LONG|SHORT|NEUTRAL`.
- Signal rules фиксируются в `docs/architecture/indicators/indicators_formula.yaml` (секция `signals:` для каждого `indicator_id`).
- В Milestone 4 покрываем signal rules для всех индикаторов, доступных в `configs/prod/indicators.yaml`.
- Финальный сигнал стратегии: AND по всем индикаторам.
- Конфликт `final_long && final_short` на одном баре: `NEUTRAL` (no-trade).

### Grid backtest (v1)

- Backtest запускается не только для одной стратегии, но и для **grid**:
  - пользователь выбирает набор индикаторов,
  - параметры варьируются в диапазонах/шагах из `configs/prod/indicators.yaml`.
- Перебор подмножеств индикаторов (subsets) **не делаем**: grid варьирует только параметры выбранного набора.
- SL/TP входят в grid как оси (варианты растут мультипликативно; шаги настраиваются).
- Runner v1: staged pipeline как в примере:
  - Stage A: быстрый прогон по базовому grid (без SL/TP) и shortlist `preselect`.
  - Stage B: расширение shortlist по SL/TP осям и точный расчёт; возвращаем только top-K.
- В результирующей таблице возвращаем только top-K (default `top_k_default=300`, конфигурируемо).
- Ranking key: `Total Return [%]` (по убыванию).

Guards v1 (фикс): используем только два лимита из indicators grid builder:

- `MAX_VARIANTS_PER_COMPUTE_DEFAULT = 600_000`
- `MAX_COMPUTE_BYTES_TOTAL_DEFAULT = 5 GiB`

### Report (v1)

- Отчёт строится как таблица `|Metric|Value|` с фиксированным набором метрик.
- Benchmark: buy&hold long без fee/slippage.
- Sharpe/Sortino/Calmar: по дневным доходностям equity (resample 1d), risk-free=0, annualization=365.

---

## Принцип декомпозиции Milestone 4

Milestone 4 делится на 3 логических трека:

1) **Backtest engine**: свечи/rollup, исполнение (direction/sizing/SLTP/fees), детерминизм.
2) **Signals layer**: правила `indicator -> LONG|SHORT|NEUTRAL` и AND-агрегация.
3) **Grid + API + Report**: построение вариантов, guards sync, API контракт и вычисление метрик.

---

## Порядок внедрения (без лишней магии)

1) BKT-EPIC-01 → Backtest context skeleton + domain/contracts + wiring points
2) BKT-EPIC-02 → Candle timeline: read canonical 1m + rollup TF + warmup policy
3) BKT-EPIC-03 → Indicator signal rules spec (docs + contracts) + full coverage for `configs/prod/indicators.yaml`
4) BKT-EPIC-04 → Grid builder + staged config (Stage A/Stage B) + guards + top-K policy
5) BKT-EPIC-05 → Execution engine (close-fill): direction/sizing/SLTP/fees/slippage (CPU parallel)
6) BKT-EPIC-06 → Reporting v1: equity curve + trade log + metrics table + benchmark + ratios
7) BKT-EPIC-07 → API v1: `POST /backtests` (saved strategy + ad-hoc grid) + 422 errors
8) BKT-EPIC-08 → Tests: determinism + golden fixtures + perf smoke

---

## EPIC'и Milestone 4

### BKT-EPIC-01 — Backtest bounded context v1: domain + use-case skeleton

**Цель:** завести `backtest` как bounded context с минимальными доменными сущностями/ошибками и application use-case, не фиксируя преждевременно все детали реализации.

**Scope:**
- `src/trading/contexts/backtest/domain/*`: сущности и value objects (позиция/сделка/параметры backtest), ошибки, инварианты.
- `src/trading/contexts/backtest/application/use_cases/*`: один use-case “run backtest v1”.
- Порты (application): чтение свечей, вычисление индикаторов, загрузка strategy spec (для saved strategy режима).
- Runtime config v1 (минимум):
  - `warmup_bars_default` (default 200)
  - `top_k_default` (default 300)
  - `preselect_default` (default 20_000)

**Non-goals:**
- jobs/progress, сохранение результатов (Milestone 5)
- intrabar execution (Milestone 7)

**DoD:**
- Появляется структура контекста `backtest` по стилю остальных контекстов.
- Определены типы входа/выхода use-case (DTO) и единый механизм ошибок (422).

**Paths:**
- `src/trading/contexts/backtest/*`
- `docs/architecture/api/api-errors-and-422-payload-v1.md`

---

### BKT-EPIC-02 — Candle timeline v1: canonical 1m read + rollup TF + warmup

**Цель:** детерминированно получить свечи для backtest на выбранном timeframe, используя canonical 1m как источник правды, и определить warmup-правила для индикаторов/сигналов.

**Scope:**
- Canonical 1m — source of truth: чтение через `CanonicalCandleReader`, в backtest используем `CandleFeed.load_1m_dense(...)` (ACL indicators -> market_data) для получения dense 1m + NaN holes.
- Пользователь задаёт произвольные `Start/End` (без требования выравнивания). Внутри backtest:
  - `start_1m = floor_to_minute(Start - warmup_duration)`
  - `end_1m = ceil_to_minute(End)`
- Derived TF (`5m/15m/1h/4h/1d`) строим по epoch-aligned bucket границам через `Timeframe.bucket_open/bucket_close` (UTC).
- Rollup v1 для backtest — best-effort:
  - missing 1m внутри бакета не “убивают” бакет,
  - derived OHLCV агрегируется по доступным 1m (NaN минуты игнорируются),
  - полностью пустой бакет → carry-forward: `OHLC=prev_close`, `volume=0`,
  - derived candles не содержат NaN.
- Warmup lookback policy:
  - `warmup_bars_default = 200` (конфигурируемо, в барах целевого timeframe),
  - если истории недостаточно — начинаем с первого доступного бара (без ошибки),
  - метрики/отчёт downstream считаем только на целевом интервале `[Start, End)` по правилу `Start <= bar_close_ts < End`.

**Non-goals:**
- live ingestion/repair (это market_data и Strategy runner)

**DoD:**
- Свечи на выходе backtest одинаковы при повторе запроса.
- Warmup policy описана и тестируема.
- Rollup policy описана и тестируема (missing minutes не удаляют бакет; carry-forward работает).

**Paths:**
- `src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py`
- `src/trading/shared_kernel/primitives/timeframe.py`
- `src/trading/contexts/indicators/application/ports/feeds/candle_feed.py`
- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
- `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`

---

### BKT-EPIC-03 — Signals-from-indicators v1: signal rules catalog + AND aggregation

**Цель:** формально зафиксировать и реализовать минимальную систему, где каждый индикатор из библиотеки может дать `LONG|SHORT|NEUTRAL`, а стратегия агрегирует сигнал AND'ом.

**Scope:**
- Спецификация signal rules:
  - хранится в `docs/architecture/indicators/indicators_formula.yaml` как секция `signals:` внутри каждого `indicator_id`,
  - NaN/warmup semantics,
  - конфликты и детерминизм.
- Реализация сигналов для всех индикаторов, доступных в `configs/prod/indicators.yaml`.

**Non-goals:**
- “богатый DSL” сигналов, сравнение нескольких индикаторов друг с другом

**DoD:**
- Есть документированный контракт signal rules.
- Backtest может посчитать финальные `final_long/final_short` по любому выбранному набору индикаторов из `configs/prod/indicators.yaml`.

**Paths:**
- `configs/prod/indicators.yaml`
- `docs/architecture/indicators/indicators_formula.yaml`
- `docs/architecture/indicators/indicators-overview.md`

---

### BKT-EPIC-04 — Grid builder v1: variants generation + sync guards

**Цель:** построить variants (grid) для backtest из диапазонов/шагов `configs/prod/indicators.yaml` и ввести guards для синхронного backtest.

**Scope:**
- Построение variants:
  - materialization диапазонов,
  - декартово произведение по параметрам выбранных индикаторов,
  - не перебираем подмножества индикаторов (фиксируем набор, который выбрал пользователь),
  - детерминированный порядок вариантов.
- Risk grid:
  - SL/TP входят как оси grid (stage B) и увеличивают число вариантов мультипликативно.
  - шаги SL/TP задаются в request (UI управляет step).
- Staged runner config:
  - Stage A: базовый grid (только индикаторы/их параметры; без SL/TP) → shortlist `preselect` по `Total Return [%]`.
  - Stage B: расширение shortlist по SL/TP-осям → точный расчёт метрик → top-K.
- Guards для sync:
  - `MAX_VARIANTS_PER_COMPUTE_DEFAULT` (600k)
  - `MAX_COMPUTE_BYTES_TOTAL_DEFAULT` (5 GiB)

Output policy v1:

- В API возвращаем только top-K результатов по grid (по умолчанию `top_k_default=300`, конфигурируемо).
- Ranking key: `Total Return [%]` (по убыванию).
- `preselect_default=20_000` (конфигурируемо; Stage A shortlist).

**Non-goals:**
- продвинутый pruning/streaming top-k (Milestone 6)

**DoD:**
- При превышении guards возвращается понятная 422 ошибка.
- Grid variants детерминированы и воспроизводимы.
- В ответе sync grid возвращается только top-K (K из конфига), и порядок результатов детерминирован (при одинаковом ranking ключе).
- Staged pipeline детерминирован:
  - Stage A shortlist сортируется стабильно,
  - Stage B top-K сортируется стабильно,
  - при равенстве `Total Return [%]` используется фиксированный tie-break (например стабильный variant key).

**Paths:**
- `configs/prod/indicators.yaml`
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
- `src/trading/contexts/indicators/application/services/grid_builder.py` (guard defaults)

---

### BKT-EPIC-05 — Backtest execution engine v1 (close-fill)

**Цель:** реализовать детерминированное исполнение сделок на close по сигналам, с fees/slippage, direction modes, position sizing и close-based SL/TP.

**Scope:**
- Close-fill loop.
- Direction modes: `long-only`, `short-only`, `long-short`.
- Position sizing: 4 режима + safe-profit lock policy.
- SL/TP close-based.
- Fee/slippage применение.

Staged semantics v1:
- Stage A: исполняем сигнальную стратегию без SL/TP (для отбора shortlist).
- Stage B: исполняем с SL/TP (close-based) и считаем полный отчёт.

Compute model v1:
- grid backtest выполняется параллельно на CPU по множеству вариантов (без CUDA)
- цель: уметь прогнать grid в пределах guards и вернуть только top-K

Position constraint v1 (фикс):
- одновременно может быть только одна позиция; новая позиция открывается только после закрытия предыдущей
- допускается закрытие по SL/TP и открытие новой позиции на этом же `close[t]` (exit → entry)

Fee/slippage semantics (фикс v1):
- buy fill: `price_fill = close * (1 + slippage_pct)`
- sell fill: `price_fill = close * (1 - slippage_pct)`
- комиссия берётся на entry и exit (round-trip = 2 комиссии)
- при reversal в одном баре: close+open = 2 комиссии (и slippage применяется к обоим fill)

**Non-goals:**
- intrabar fills
- leverage/funding/liq modelling

**DoD:**
- Один и тот же вход → один и тот же trades/equity результат.
- Reversal детерминирован (close+open на одном баре, если разрешено режимом).

**Paths:**
- `src/trading/contexts/backtest/domain/*`
- `src/trading/contexts/backtest/application/use_cases/*`

---

### BKT-EPIC-06 — Reporting v1: equity curve + trades + metrics table

**Цель:** строить отчёт по каждому варианту в формате `|Metric|Value|` и вычислять весь список метрик (включая benchmark и ratios) детерминированно.

**Scope:**
- Сбор trade log.
- Построение equity curve.
- Метрики:
  - PnL/Return, DD, coverage/exposure,
  - trade stats,
  - Expectancy, SQN,
  - Sharpe/Sortino/Calmar (по согласованной методике),
  - Benchmark return.

Benchmark/ratios semantics (фикс v1):
- benchmark: buy&hold long без fee/slippage, вход на `close[first]`, выход на `close[last]`
- ratios: по дневным доходностям equity (resample 1d), `risk_free=0`, annualization=365

**Non-goals:**
- UI визуализация (это фронтенд)

**DoD:**
- Отчёт стабилен и совпадает с утверждённой методикой расчёта метрик.
- Форматирование значений детерминировано.
- Для grid запуска отчёты формируются как минимум для всех вариантов, попавших в top-K.

**Paths:**
- `docs/architecture/roadmap/base_milestone_plan.md` (Milestone 4 report requirements)

---

### BKT-EPIC-07 — API v1: `POST /backtests` (saved strategy + ad-hoc grid)

**Цель:** предоставить API, который UI может вызывать из “после бектеста” и из “сохранённые стратегии”, с синхронным small-run и понятными 422 ошибками.

**Scope:**
- Endpoint v1:
  - `POST /backtests`
    - режим A (saved): body содержит `strategy_id` + overrides (например SL/TP grid, fees, slippage, sizing)
    - режим B (ad-hoc): body содержит grid template (индикаторы + параметры + оси SL/TP) без сохранения Strategy
- Flow requirement: пользователь может сохранить стратегию только после завершения backtest по всем вариантам; UI сохраняет выбранный вариант из top-K.
- Ответ backtest должен содержать достаточно данных, чтобы UI мог сохранить выбранный вариант как StrategySpec (например: список индикаторов и конкретные параметры варианта + параметры risk/sizing).
- Auth: только owner (через `current_user`).
- Unified 422 errors.

Output policy v1:
- для grid запуска API возвращает только top-K (по умолчанию K=300, конфигурируемо)
- ranking key: `Total Return [%]`

**Non-goals:**
- асинхронные jobs (Milestone 5)

**DoD:**
- API контракт зафиксирован и покрыт тестами.
- Ошибки детерминированы и не “плывут” между версиями.

**Paths:**
- `apps/api/routes/backtests.py`
- `docs/architecture/api/api-errors-and-422-payload-v1.md`

---

### BKT-EPIC-08 — Tests v1: determinism + golden fixtures + perf smoke

**Цель:** зафиксировать корректность и детерминизм backtest по набору “золотых” тестов, и защититься от случайных регрессий производительности.

**Scope:**
- Unit tests: grid builder, signal aggregation, execution loop.
- Golden tests: один и тот же вход даёт идентичный отчёт (строковый) и ключевые числа.
- Perf smoke: небольшой sync grid в пределах guards.

**Non-goals:**
- долгие интеграционные тесты на больших периодах

**DoD:**
- Тесты проходят стабильно и не зависят от порядка словарей/платформы.

**Paths:**
- `tests/unit/contexts/backtest/*`
- `tests/perf_smoke/*`
