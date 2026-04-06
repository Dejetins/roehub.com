# Final Backtest Refactor Plan v2

Статус: approved architecture baseline  
Дата фиксации: 2026-03-24  
Область: `backtest`, `indicators`, `market_data`, artifact precompute/runtime

## 1. Цель рефакторинга

Цель рефакторинга: радикально ускорить backtest execution за счёт переноса вычислительно дорогой подготовки данных из sync/job runtime в отдельный precompute pipeline и замены текущего per-variant/per-bar исполнения на artifact-backed runtime kernels.

Что должно быть достигнуто:

- hot path sync и jobs не должны обращаться к ClickHouse;
- hot path sync и jobs не должны запускать `IndicatorCompute.compute(...)`;
- runtime не должен строить rollup свечей на лету;
- runtime не должен выполнять Python loop по каждому бару для каждого варианта;
- ranking и selection должны опираться на precomputed signals, price arrays и `1m hit-times`;
- запуск backtest не должен требовать отдельной кнопки `Estimate preflight`; проверка лимитов должна происходить автоматически при старте run;
- каждый пользовательский запуск должен сохраняться как persisted run и быть доступен из `Backtest history`;
- итоговые результаты должны сохраняться как summary-only table top-N без persisted trades/report bodies;
- архитектура должна оставаться реалистичной для текущего репозитория и поддерживать инкрементальную миграцию.

## 2. Контекст и исходные документы

Документ заменяет и уточняет ранее предложенный подход из [base_refactor_plan.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/base_refactor_plan.md).

Основные изученные документы:

- [base_refactor_plan.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/base_refactor_plan.md)
- [backtest-compute-notebook-algorithm-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md)
- [backtest-signals-from-indicators-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-signals-from-indicators-v1.md)
- [backtest-candle-timeline-rollup-warmup-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md)
- [backtest-api-post-backtests-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [backtest-job-runner-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)
- [backtest-job-runner-worker-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)
  — historical / compatibility reference for legacy EPIC-10 wording
- [backtest-execution-engine-close-fill-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md)
- [backtest-bounded-context-domain-use-case-skeleton-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md)
- [backtest-grid-builder-staged-runner-guards-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md)
- [backtest-refactor-perf-plan-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-refactor-perf-plan-v1.md)
- [backtest-staged-ranking-reporting-perf-optimization-plan-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-staged-ranking-reporting-perf-optimization-plan-v1.md)
- [indicators_formula.yaml](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/indicators/indicators_formula.yaml)
- [indicators-compute-engine-core.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/indicators/indicators-compute-engine-core.md)
- [indicators-grid-compute-perf-optimization-plan-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md)
- [configs/prod/indicators.yaml](/Users/daniildegtyarev/Projects/roehub.com/configs/prod/indicators.yaml)

Основные изученные кодовые точки:

- [run_backtest.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest.py)
- [run_backtest_job_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py)
- [staged_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/staged_runner_v1.py)
- [staged_core_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/staged_core_runner_v1.py)
- [close_fill_scorer_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py)
- [execution_engine_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/execution_engine_v1.py)
- [grid_builder_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/grid_builder_v1.py)
- [candle_timeline_builder.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/candle_timeline_builder.py)
- [indicators_yaml_defaults_provider.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/adapters/outbound/defaults/indicators_yaml_defaults_provider.py)
- [engine.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py)
- [market_data_candle_feed.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py)
- [canonical_candle_reader.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py)

## 2A. R0 implementation references

Эти документы фиксируют baseline артефакты, которые должны существовать до начала runtime cutover:

- [backtest-v2-benchmarks.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-v2-benchmarks.md)
- [web-backtest-runtime-defaults-endpoint-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md)

Они не отменяют этот final plan, а делают его исполнимым на уровне benchmark protocol и frozen contract surface.

## 3. Подтверждённые наблюдения о текущей реализации

### 3.1 Runtime всё ещё зависит от ClickHouse

Текущий sync/jobs path строит candle timeline через:

- `RunBacktestUseCase`
- `BacktestCandleTimelineBuilder`
- `MarketDataCandleFeed`
- `ClickHouseCanonicalCandleReader`

То есть runtime всё ещё тянет dense `1m` candles из ClickHouse и делает rollup в запросе.

### 3.2 Основной bottleneck уже не indicator compute

Даже если indicator tensors уже batched/materialized, текущий runtime остаётся медленным из-за:

- построения `final_signal` по каждому варианту;
- Python loop по каждому бару внутри execution engine;
- Stage A и Stage B scoring по одному варианту за раз.

### 3.3 Текущий signal contract уже трёхсостоянийный

Backtest signal в текущем контракте — это не бинарный event, а дискретное состояние на каждом баре:

- `SHORT = -1`
- `NEUTRAL = 0`
- `LONG = 1`

Это already aligned с compact int8 storage и не требует перехода на `NaN` или float-based signal arrays.

### 3.4 Full zoo precompute в текущем `indicators.yaml` непрактичен без чистки

Суммарная compute-cardinality текущего `configs/prod/indicators.yaml` была оценена как `8,225,252` вариантов.  
Основной взрыв давали 11 тяжёлых индикаторов. После их полного удаления остаётся около `49,310` compute-rows.

## 4. Проблемы текущей реализации

### 4.1 Hot path слишком длинный

Текущий путь:

`API/job -> request decode -> build candles -> query CH -> rollup -> compute/prepare -> build signals -> execute per variant -> rank/report`

Проблема в том, что слишком много тяжёлой работы делается прямо в runtime.

### 4.2 Дорогая execution semantics

`BacktestExecutionEngineV1` работает как stateful close-fill engine с Python per-bar replay.  
Это слишком дорого для больших variant grids.

### 4.3 Слишком много ненужной гибкости в runtime

Сейчас runtime страдает от сочетания:

- широкого zoo индикаторов;
- большого количества timeframes;
- отсутствия жёсткой artifact policy;
- необходимости поднимать compute semantics в каждом run.

### 4.4 Недостаточно жёсткие storage/runtime contracts

Без фиксированных правил:

- какие TF поддерживаются;
- какие indicator_id вообще живут в продукте;
- как кодируются сигналы;
- как публикуются артефакты;
- как pin’ится artifact version на время job;

невозможно получить детерминированный и быстрый runtime.

### 4.5 Текущий launch/results UX мешает производительности и продуктовой логике

Сейчас в v1/v1-docs есть несколько неудачных решений:

- template-mode требует отдельный ручной `Estimate preflight` перед реальным запуском;
- sync run не является persisted entity и теряется после reload страницы;
- нет единой `Backtest history`;
- результаты исторически смешивают ranking summary и detail/trades concerns;
- нет отдельной канонической страницы варианта, где можно быстро пересчитать один выбранный вариант и построить график со сделками;
- `top_k=300` пришит к старой staged модели и не совпадает с желаемым продуктовым UX `top 100` / configurable top-N.

## 5. Зафиксированные архитектурные решения

Ниже перечислены решения, которые считаются утверждёнными и не требуют дополнительного выбора.

### 5.1 Общая модель

- Engine v2 становится целевой production архитектурой для backtest.
- bounded context `backtest` сохраняется; v2 реализуется внутри него новыми модулями.
- Sync runner внедряется первым, job runner переключается следом.

### 5.2 Artifact publishing

- Используются два слота: `slot_a` и `slot_b`.
- Активный слот определяется через маленький pointer file `current.yaml`.
- Новый publish всегда строится в неактивном слоте.
- После полной валидации `current.yaml` атомарно переключается на новый слот.
- Если неактивный слот ещё pinned активными job-ами, publish блокируется и не начинается.
- Старый активный слот не переписывается in-place.

### 5.2A Operational execution model

- Production rebuild/publish выполняется отдельным сервисом artifact precompute/publish, а не
  inline внутри API или `backtest-job-runner`.
- Deployment target этого сервиса: Mac Studio native backend.
- Instrument universe source-of-truth: `market_data.ref_instruments`.
- Scheduled daily run обходит все enabled+tradable trading pairs из актуального snapshot
  `market_data.ref_instruments`.
- Manual ad-hoc mode допускается для bootstrap initial slot и для точечного rebuild одного
  инструмента, но использует тот же publish contract и тот же whole-slot validation.
- Scheduled execution anchored to `Europe/Moscow` и запускается ежедневно в `03:05`.
- Prod `artifact_root` должен быть стабильным host data path вне repo checkout; относительный
  путь внутри checkout допускается только как dev/test convenience.
- Сервис обязан держать host-level lock, исключающий concurrent rebuild/publish для одного и того
  же symbol root.
- Сервис обязан публиковать Prometheus metrics и structured logs, достаточные для ответов на
  вопросы:
  - был ли daily run запущен;
  - сколько инструментов обработано/пропущено/упало;
  - был ли publish blocked lock/pin/validation failure;
  - когда был последний successful publish;
  - сколько `1m` bars реально переписано в incremental mode по `prices`, `mappings`,
    `signals`, `hit_times`.

### 5.3 Timeframes

Backtest request timeframes:

- `15m`
- `30m`
- `1h`
- `2h`
- `4h`
- `6h`
- `8h`
- `1d`
- `2d`
- `3d`

Запрещены как request timeframe:

- `1m`
- `5m`

При этом `1m` остаётся внутренней базой для:

- source prices;
- `1m hit-times`;
- mapping request timeframe bars в minute execution space.

### 5.4 Risk execution semantics

Принята новая семантика:

- signal evaluation выполняется на request timeframe;
- TP/SL execution живёт только на `1m`;
- используется `1m hit-time only`.

Это означает, что Stage B больше не обязан воспроизводить старый close-fill semantics на выбранном TF.

### 5.5 Signal storage

- Signals хранятся как `np.int8`.
- Кодировка:
  - `SHORT = -1`
  - `NEUTRAL = 0`
  - `LONG = 1`
- `NaN` не используется как runtime/storage sentinel.
- Layout signal matrices: `[V, T_tf]`.

### 5.6 Indicators policy

Из всех систем полностью удаляются:

- `momentum.stoch_rsi`
- `trend.ichimoku`
- `volatility.bbands`
- `volatility.bbands_bandwidth`
- `volatility.bbands_percent_b`
- `momentum.macd`
- `momentum.ppo`
- `trend.chandelier_exit`
- `volume.vwap_deviation`
- `trend.keltner`
- `trend.supertrend`

Полное удаление означает:

- удаление из `configs/*/indicators.yaml`;
- удаление из indicator definitions/registry;
- удаление из compute kernels;
- удаление из signal rules/docs;
- удаление из API/UI выбора;
- удаление из тестов;
- удаление из архитектурных документов.

### 5.7 `signals.v1.params`

- `signals.v1.params` должны быть добавлены в `configs/*/indicators.yaml`.
- В initial v2 они поддерживаются только как default values.
- Full signal-grid expansion в initial v2 запрещён.
- Request с non-default `signals.v1.params` должен получать deterministic reject.

### 5.8 Модель запуска, результатов и истории

- `POST /backtests` становится единым входом создания persisted run.
- Пользователь больше не нажимает отдельную кнопку `Estimate preflight`; сервер всегда делает preflight автоматически при старте run.
- Если request проходит sync budgets, run выполняется inline и завершается как persisted succeeded run.
- Если request не проходит sync budgets, но проходит full background budgets, run автоматически создаётся как background run и попадает в history без отдельного ручного переключения пользователя.
- Если request не проходит и full budgets, возвращается deterministic `422`.
- Все запуски хранятся в истории, независимо от того, были они inline или background.
- Public UX больше не делит сущности на “sync result” и “job result”; для пользователя существует единый `backtest run`.
- Итоговый результат run всегда хранится как одна summary table `top N`.
- `N` по умолчанию равен `100`, но должен конфигурироваться через runtime config и при необходимости меняться на `500` или другой server-approved cap.
- В summary result не сохраняются и не отдаются trades.
- Подробный расчёт варианта выполняется только лениво по выбранной строке таблицы.
- Detail page варианта всегда пересчитывает только один вариант по его явным параметрам и по pinned artifact slot исходного run.
- Hash-поля сохраняются как внутренний reproducibility contract, но не показываются пользователю в UI.

### 5.9 Ranking и сортировка результатов

Runtime обязан поддерживать выбор ranking metric для отбора `top N`.

Рекомендованный набор ranking metrics:

- `total_return_pct` (`DESC`)
- `max_drawdown_pct` (`ASC`)
- `return_over_max_drawdown` (`DESC`)
- `profit_factor` (`DESC`)
- `sharpe_trades` (`DESC`)
- `win_rate_pct` (`DESC`)

Рекомендованный набор summary columns, по которым UI может локально пересортировать уже найденный `top N` без нового расчёта:

- `total_return_pct`
- `max_drawdown_pct`
- `return_over_max_drawdown`
- `profit_factor`
- `sharpe_trades`
- `win_rate_pct`
- `trade_count`
- `avg_trade_ret_pct`
- `avg_trade_exec_bars`
- `exposure_pct`
- `best_tp_pct`
- `best_sl_pct`

Важно:

- ranking metric выбирает, **какие** варианты попадают в `top N`;
- сортировка по любой колонке в UI переставляет местами только уже найденные `top N` строк и не запускает новый run.

### 5.10 Выбор source series пользователем

- Пользователь должен иметь возможность выбирать несколько source series для индикаторов через checkbox/multi-select UI.
- Поддерживаются все source values, разрешённые для конкретного indicator_id в `configs/*/indicators.yaml`.
- На уровне request это остаётся explicit axis `inputs.source`.
- На уровне UI runtime defaults должны отдавать список разрешённых source values для каждого indicator_id, чтобы форма не хардкодила `close/open/hlc3/ohlc4/...`.

## 6. Целевая схема взаимодействия модулей

```text
Client/UI
  -> POST /backtests / job create
  -> backtest use cases
  -> artifact runtime v2
     -> current.yaml resolver
     -> slot manifest loader
     -> prices loader
     -> signal matrix loader
     -> signal aggregator kernel
     -> trade compactor kernel
     -> 1m risk-exit kernel
     -> metrics kernel
  -> ranking/reporting

Daily precompute worker
  -> read current.yaml
  -> choose inactive slot
  -> load canonical 1m once
  -> materialize prices/1m
  -> build/update 1m hit-times
  -> for one timeframe at a time:
     -> derive/write rolled_prices as prices/<tf>
     -> open timeframe session
     -> build mappings/<tf>
     -> materialize signals/<tf>/<indicator_id> in bounded chunks
     -> close timeframe session
  -> finalize manifests
  -> validate
  -> atomically switch current.yaml
```

Граница ответственности:

- `market_data` и `indicators` нужны precompute pipeline;
- runtime v2 должен обходиться без них в hot path;
- `backtest` становится главным потребителем precomputed artifacts.

### 6.1 R12 clarification: stable outputs, changed execution model

R12 не меняет published artifact contract. Он уточняет, как именно эти артефакты должны
строиться.

- Stable outputs:
  - layout `artifacts/backtest/v2/...` не меняется;
  - `signals/<tf>/<indicator_id>/signals.i8.npy` не меняется;
  - `axis_order: [variant, time]` не меняется;
  - `current.yaml` / manifest schemas не меняются.
- Changed execution model:
  - artifact precompute должен быть stage-oriented, а не giant tensor-first;
  - canonical model uses `timeframe-scoped execution`: один `current_timeframe` session, затем
    детерминированный переход к следующему timeframe;
  - bounded chunk execution is required for signal materialization;
  - follow-up executor work must be driven by explicit `execution_policy` and `ChunkPlanner`
    contracts.
- Unchanged public/runtime semantics:
  - public `indicators` runtime may keep its tensor-first contract and hot-path guards;
  - offline artifact precompute must not inherit that giant in-memory dense tensor model as the
    recommended architecture.

### 6.2 Why tensor-first is not the target precompute model

The old tensor-first narrative is not recommended for artifact precompute because it keeps too many
large buffers alive at once and couples offline slot building to runtime-oriented indicator memory
guards. The target state is bounded Mac Studio execution: load canonical `1m` once, process one
timeframe session at a time, flush chunked writes eagerly, and only then move to the next
timeframe.

## 7. Артефакты: структура хранения

Целевой layout:

```text
artifacts/backtest/v2/
  <exchange>/<market_type>/<symbol>/
    current.yaml
    slot_a/
      manifest.yaml
      prices/
        1m/
          open_time.i64.npy
          close_time.i64.npy
          ohlcv.f32.npy
        15m/
          open_time.i64.npy
          close_time.i64.npy
          ohlcv.f32.npy
        30m/
        1h/
        2h/
        4h/
        6h/
        8h/
        1d/
        2d/
        3d/
      signals/
        15m/
          ma.ema/
            signals.i8.npy
            manifest.yaml
          ma.sma/
            signals.i8.npy
            manifest.yaml
        30m/
        1h/
        ...
      mappings/
        15m/
          bar_open_1m_idx.u32.npy
          bar_close_1m_idx.u32.npy
        30m/
        1h/
        ...
      hit_times/
        1m/
          tp_values.f32.npy
          sl_values.f32.npy
          long_tp.u32.npy
          long_sl.u32.npy
          short_tp.u32.npy
          short_sl.u32.npy
          manifest.yaml
    slot_b/
      ...
```

### 7.1 Что хранится в `prices`

Для каждого TF:

- `open_time.i64.npy`
- `close_time.i64.npy`
- `ohlcv.f32.npy`

`ohlcv.f32.npy` хранит только numeric OHLCV columns.  
Timestamps не смешиваются с float columns в одном homogeneous массиве.

### 7.2 Что хранится в `signals`

Для каждого `timeframe + indicator_id`:

- отдельный `signals.i8.npy`;
- отдельный `manifest.yaml`.

Shape:

```text
[V, T_tf]
```

где:

- `V` — число variant rows этого индикатора на этом TF;
- `T_tf` — число баров этого timeframe.

### 7.3 Что хранится в `mappings`

Для каждого request timeframe:

- `bar_open_1m_idx.u32.npy`
- `bar_close_1m_idx.u32.npy`

Они нужны, чтобы быстро переводить позицию бара request timeframe в minute execution space.

### 7.4 Что хранится в `hit_times`

Только `1m hit-time tables`.

Не допускается хранение отдельного набора hit-times для каждого TF.  
Вся risk execution semantics унифицируется через `1m`.

## 8. Manifest contracts

### 8.1 `current.yaml`

Минимально содержит:

```yaml
schema_version: 1
active_slot: slot_a
slot_generation: 42
asof_date: "2026-03-24"
manifest_sha256: "..."
published_at_utc: "2026-03-24T02:00:00Z"
```

Назначение:

- определить активный слот;
- зафиксировать версию артефактов для воспроизводимости;
- дать runtime и jobs стабильную identity published dataset.

### 8.2 Root `manifest.yaml`

Должен содержать:

- identity (`exchange`, `market_type`, `symbol`);
- `asof_date`;
- список поддержанных TF;
- список поддержанных indicator_id;
- signal encoding contract;
- hash/shape/dtype на ключевые файлы;
- version/policy identifiers;
- config provenance;
- generator version;
- `slot_generation`.

### 8.3 Per-indicator signal manifest

Должен содержать:

- `indicator_id`;
- `timeframe`;
- deterministic axis order;
- grid description;
- `signals.v1.params defaults`;
- `shape`, `dtype`, `sha256`;
- rows count;
- timeline coverage.

## 9. Правила для performance-critical path

Внутри sync/job runtime v2 запрещены:

- ClickHouse queries;
- `IndicatorCompute.compute(...)`;
- runtime rollup из `1m`;
- YAML parsing;
- filesystem scanning;
- unicode signal arrays;
- Python per-bar execution loops;
- debug materialization для всех variants;
- hash recomputation на hot path.

Внутри sync/job runtime v2 обязательны:

- `np.load(..., mmap_mode='r')` или эквивалентный mmap loader;
- fixed numeric metadata, уже прочитанные из manifest;
- chunked loading subset variant rows;
- kernels на contiguous numeric buffers;
- materialization detailed trades только для top-K shortlist.

## 10. Что меняется по слоям

## 10.1 Backtest application/use cases

Файлы:

- [run_backtest.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest.py)
- [run_backtest_job_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py)

Изменения:

- убрать зависимость sync/jobs hot path от `CandleFeed` и `IndicatorCompute`;
- подключить единый runtime facade `BacktestEngineV2Runtime`;
- pin’ить `current.yaml` identity на время всего run/job;
- прокидывать в response/job metadata `artifact_slot`, `slot_generation`, `asof_date`, `manifest_hash`.
- превратить `POST /backtests` из ephemeral sync run в create-and-execute persisted run;
- добавить use-case’ы для history list, run details, run top table и lazy single-variant detail page.

## 10.2 Backtest runtime services

Старые файлы, переводимые в legacy/compat слой:

- [staged_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/staged_runner_v1.py)
- [staged_core_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/staged_core_runner_v1.py)
- [close_fill_scorer_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py)
- [execution_engine_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/execution_engine_v1.py)
- [grid_builder_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/grid_builder_v1.py)
- [candle_timeline_builder.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/candle_timeline_builder.py)

Новые файлы/модули:

- `src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py`
- `src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py`
- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/backtest_engine_v2_runtime.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py`

## 10.3 Backtest defaults/config loading

Файл:

- [indicators_yaml_defaults_provider.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/adapters/outbound/defaults/indicators_yaml_defaults_provider.py)

Изменения:

- сохраняется parsing `signals.v1.params`;
- вводится enforcement режима `default-only` для v2 initial;
- backtest defaults provider должен уметь детерминированно reject non-default signal params.

## 10.4 Indicators context

Удаляются indicator definitions:

- `src/trading/contexts/indicators/domain/definitions/momentum.py`
- `src/trading/contexts/indicators/domain/definitions/trend.py`
- `src/trading/contexts/indicators/domain/definitions/volatility.py`
- `src/trading/contexts/indicators/domain/definitions/volume.py`

Удаляются compute implementations:

- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py`

Обновляются registry/wiring:

- `src/trading/contexts/indicators/adapters/outbound/registry/yaml_indicator_registry.py`
- `src/trading/contexts/indicators/domain/definitions/__init__.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`

## 10.5 Market data context

Файлы:

- [market_data_candle_feed.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py)
- [canonical_candle_reader.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py)

Изменения:

- эти модули остаются нужны только precompute pipeline;
- runtime v2 не должен через них проходить.

## 10.6 Persisted run storage и history layer

Базовый pragmatic вариант:

- физически переиспользовать существующее семейство PG таблиц `backtest_jobs`, `backtest_job_top_variants`, `backtest_job_stage_a_shortlist`;
- логически превратить их из “storage только для async jobs” в canonical persisted run storage для всех запусков.

Это решение выбрано потому, что:

- storage уже существует;
- state/progress/lease/snapshot semantics уже реализованы;
- история запусков по объёму относительно дешева, если хранить только summary rows;
- не нужен второй параллельный persistence stack.

Что должно измениться:

- `backtest_jobs` логически становится таблицей всех persisted runs;
- inline runs тоже создают запись в этом storage;
- background runs используют тот же storage, но с другим `execution_mode`;
- `backtest_job_top_variants` хранит только summary payload и summary metrics;
- `report_table_md` и `trades_json` перестают быть частью persisted results contract v2 и должны быть удалены или оставлены постоянно `NULL` на переходный период.

Минимальные дополнительные поля persisted run list contract:

- `execution_mode` (`sync_inline | background_auto | background_manual_legacy`)
- `market_id`
- `symbol`
- `timeframe`
- `requested_top_n`
- `ranking_primary_metric`
- `ranking_secondary_metric`
- `artifact_slot`
- `artifact_slot_generation`
- `artifact_manifest_hash`

`backtest_job_top_variants` должен хранить:

- `rank`
- `variant_key`
- `variant_index`
- `payload_json`
- `summary_metrics_json`
- `best_tp_pct`
- `best_sl_pct`

И не должен хранить:

- полный report body;
- trades list;
- equity curve.

## 11. Изменения в API и контрактах

### 11.1 `POST /backtests`

Поведение:

- request shape сохраняется;
- `timeframe in {"1m", "5m"}` -> deterministic validation error;
- удалённые `indicator_id` -> deterministic validation error;
- non-default `signals.v1.params` -> deterministic validation error.
- сервер всегда выполняет внутренний preflight автоматически;
- ручной `POST /indicators/estimate` перед запуском больше не является обязательной пользовательской стадией;
- при успешном preflight создаётся persisted run record;
- если request проходит sync budgets, run исполняется inline;
- если request не проходит sync budgets, но проходит full background budgets, run автоматически переводится в background execution;
- если request не проходит full budgets, возвращается deterministic `422`.

Response metadata добавляет:

- `run_id`
- `state`
- `execution_mode`
- `engine_version`
- `artifact_slot`
- `artifact_slot_generation`
- `artifact_asof_date`
- `artifact_manifest_hash`

Response payload policy:

- для `succeeded` inline run ответ может сразу содержать summary table `top N`;
- для background run ответ может быть `202 Accepted` с persisted `run_id`, после чего UI должен перейти/обновить history entry;
- hashes и служебные runtime identifiers присутствуют в API/internal storage, но не обязаны отображаться в UI.

### 11.2 History и run retrieval API

Нужен единый public contract для history, поверх persisted run storage:

- `GET /backtests/runs`
- `GET /backtests/runs/{run_id}`
- `GET /backtests/runs/{run_id}/top`
- `POST /backtests/runs/{run_id}/cancel`

Назначение:

- `GET /backtests/runs` — вкладка `Backtest history`;
- `GET /backtests/runs/{run_id}` — metadata/status одного run;
- `GET /backtests/runs/{run_id}/top` — summary-only таблица результатов;
- `POST /backtests/runs/{run_id}/cancel` — отмена background run.

Допустим временный compatibility layer:

- старые `/backtests/jobs*` endpoints могут существовать как legacy alias на период миграции;
- целевой public UX должен говорить не “jobs”, а “history/runs”.

### 11.3 Single-variant detail contract

Нужен отдельный lazy detail endpoint/flow для страницы выбранной стратегии.

Целевой контракт:

- detail page открывается по persisted run + variant identity;
- backend не читает готовые persisted trades, а заново считает **ровно один** вариант;
- расчёт detail page использует:
  - pinned artifact slot исходного run,
  - исходный time range,
  - explicit variant payload из summary row.

Result detail payload должен содержать:

- подробную статистику варианта;
- equity curve / chart series;
- список сделок;
- сделки, размеченные на графике.

И не должен:

- сохраняться в PG как часть history по умолчанию;
- пересчитывать заново весь `top N`.

### 11.4 Background execution contract

Job payload и persisted results должны сохранять:

- pinned artifact slot;
- slot generation;
- manifest hash;
- as-of date.

Это нужно для воспроизводимости и безопасного чтения артефактов во время долгого job.

### 11.5 Runtime defaults endpoint

Должен явно отдавать:

- разрешённые request TF;
- поддержанные indicator ids;
- режим `signals.v1.params: default-only`;
- версию execution semantics: `signal_tf + 1m_risk`.
- `top_n_default`;
- `top_n_max`;
- ranking metrics, доступные для отбора `top N`;
- список sortable summary columns;
- доступные `inputs.source` values по каждому indicator_id для checkbox/multi-select UI.

## 12. Data flow и execution model

### 12.1 Signal layer

Для каждого выбранного индикатора runtime:

1. находит нужный `signals.i8.npy`;
2. вычисляет row index для выбранного набора compute params;
3. читает ровно нужную row или chunk rows;
4. агрегирует per-indicator signals в `final_signal`.

### 12.2 Stage A

Stage A выполняется без SL/TP.  
Он нужен для дешёвой предварительной фильтрации вариантов.

Алгоритм:

1. взять per-indicator rows;
2. построить strategy-level `final_signal`;
3. найти edges `neutral -> long`, `neutral -> short`, `long -> neutral`, `short -> neutral`, `long -> short`, `short -> long`;
4. построить compact trade list на request timeframe;
5. посчитать fast metrics для ranking.

### 12.3 Stage B

Stage B использует shortlist из Stage A и добавляет risk execution.

Алгоритм:

1. каждый entry bar request timeframe маппится в `1m` через `bar_close_1m_idx`;
2. из compact trade list выполняется fast monotone TP/SL grid-search по `1m hit-times`, а не полный replay всех ячеек;
3. signal exit определяется через следующий relevant bar на request timeframe, затем тоже маппится в `1m`;
4. tie-break фиксируется по notebook rules:
   - signal exit выигрывает при равенстве бара с TP/SL;
   - SL выигрывает при равенстве бара между TP и SL;
5. после выбора лучшей TP/SL ячейки выполняется точный replay только этой ячейки;
6. финальные метрики считаются по compact trade list и exact replay лучшей ячейки.

### 12.4 Что именно переносится из `tests/notebook_tests/06_backtest_compute.ipynb`

В текущей версии документа ранее не хватало явной фиксации того, какие именно вычислительные принципы и правила SP/SL берутся из ноутбука `06_backtest_compute.ipynb`.

Канонический R5-02 entrypoint для этого transfer scope теперь вынесен в
`docs/architecture/backtest/backtest-runtime-kernels-v2.md`.
Эта секция остаётся architecture baseline summary, а детальные notebook anchors живут в
`docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`.

Ниже фиксируется точный scope переноса.

Из notebook **переносится как обязательный production contract**:

1. Разделение `signal timeline` и `execution timeline`.
2. Маппинг `signal bars -> execution entry indices`.
3. Построение `compact trade list` вместо полного bar-by-bar replay.
4. Единая функция логики выхода одной сделки, эквивалентная `evaluate_trade_factor`.
5. Fast TP/SL grid-search через монотонные hit-time таблицы и difference-буферы в log-space.
6. Точный replay только для выбранной лучшей ячейки TP/SL после fast grid-search.
7. Подсчёт метрик поверх compact trade list, а не поверх полного execution replay на весь grid.

Из notebook **не переносится буквально как системный контракт**:

1. pair-specific логика подбора SMA/EMA пар;
2. notebook-specific prefilters `top_frac_side`, `min_confirm`, `top_frac_pairs`, `single_score_chunked`;
3. жёстко зашитая комбинация `signal_tf=1h`, `exec_tf=5m`;
4. конкретный формат `prices_and_signals_5m.npy` из research-папки;
5. исследовательские DataFrame-таблицы и ranking колонок ноутбука как публичный API.

То есть notebook становится для v2 **источником runtime kernel semantics**, а не literal
orchestration contract продукта; иначе говоря, это `not a literal notebook orchestration script`.

### 12.5 Подробный алгоритм, который должен быть перенят из notebook

#### 12.5.1 Маппинг signal bars в execution entries

Цель маппинга: вход должен происходить не на signal bar, а на следующем доступном execution bar после закрытия signal bar.

Правило:

1. Если `signal timeline == execution timeline`, то:
   - `entry_exec_idx = signal_bar_idx + 1`
   - если индекс вышел за границу, используется sentinel `T_exec`.
2. Если `signal timeline != execution timeline`, то:
   - `entry_time_ms = signal_close_time_ms + 1`
   - `entry_exec_idx = searchsorted(exec_open_time_ms, entry_time_ms, side="left")`
   - если индекс вышел за границу, используется sentinel `T_exec`.

Практический смысл:

- сделка не может открываться на том же баре, по которому был вычислен сигнал;
- lookup TP/SL не должен иметь lookahead на баре входа;
- signal logic и execution logic остаются развязанными.

#### 12.5.2 Построение compact trade list

После того как построен `final_signal`, runtime не должен идти по всем барам и всем TP/SL ячейкам.

Вместо этого из signal timeline строится compact trade list со структурой:

- `entry_exec_idx`
- `direction`
- `sig_exit_exec_idx`

Смысл полей:

- `entry_exec_idx` — execution индекс, где позиция открывается;
- `direction` — `+1` long, `-1` short;
- `sig_exit_exec_idx` — execution индекс, где позиция должна закрыться по сигналу, либо `T_exec` как sentinel “signal exit отсутствует”.

Правила построения:

1. Пустые бары и повторение того же направления новую сделку не создают.
2. Первый подтверждённый вход открывает текущую позицию.
3. Если пришёл противоположный сигнал:
   - текущая сделка закрывается по сигналу на `entry_exec_idx` нового подтверждения;
   - сразу открывается новая сделка в противоположную сторону.
4. Если до конца ряда позиция остаётся открытой:
   - она записывается как сделка с `sig_exit_exec_idx = T_exec`.

Это ровно тот принцип, который в ноутбуке заменяет полный replay trade-state на компактный список сделок.

#### 12.5.3 Единая логика выхода одной сделки

В production runtime должна существовать одна canonical функция выхода сделки, эквивалентная notebook-функции `evaluate_trade_factor`.

Она обязана соблюдать следующие правила.

Правило 1. Поиск TP/SL начинается с `entry_exec + 1`.

Это критично: TP/SL не имеет права “сработать” на баре входа, иначе появляется lookahead.

Правило 2. Между TP и SL выбирается более ранний hit.

Если hit произошёл на одном и том же execution баре:

- приоритет имеет `SL`.

То есть tie-break между TP и SL фиксируется как `SL wins`.

Правило 3. Signal-exit имеет приоритет над TP/SL при совпадении execution бара.

Если:

- `sig_exit_exec_idx < T_exec`
- и `sig_exit_exec_idx <= tp_sl_exec_idx`

то побеждает выход по сигналу.

То есть tie-break между:

- signal exit
- TP/SL hit

фиксируется как `signal exit wins on equal bar`.

Правило 4. Если раньше произошёл TP/SL, берётся фактор соответствующего уровня.

То есть при TP/SL exit runtime не пересчитывает вручную intra-bar price, а использует precomputed factor для этой TP/SL ячейки.

Правило 5. Если ни signal exit, ни TP/SL не случились:

- при `close_on_end = 1` сделка форсированно закрывается на конце ряда;
- при `close_on_end = 0` сделка считается незакрытой и в scoring не даёт закрытого trade factor.

Для v2 фиксируется:

- `close_on_end = 1` как production default для notebook-derived Stage B kernels.

#### 12.5.4 Формула доходности сделки

Notebook использует не просто “закрыть по цене и умножить на equity”, а конкретную факторную модель сделки.

Для long при signal/end exit:

```text
pf_long = exit_price / entry_price
```

Для short при signal/end exit фиксируется x1 USDT ROI модель:

```text
pf_short = max(0, 2 - exit_price / entry_price)
```

Это правило должно быть отражено в production runtime явно.  
Менять его без отдельного продуктового решения нельзя, потому что оно меняет:

- результат grid-search;
- ranking;
- итоговые метрики;
- parity с notebook-derived expectations.

Для TP/SL exits используются предрассчитанные факторы уровня:

- `long_tp_eq = 1 + tp`
- `long_sl_eq = 1 - sl`
- `short_tp_eq = 1 + tp`
- `short_sl_eq = 1 - sl`

Комиссия в notebook-принципе применяется снаружи к gross factor:

```text
fee_two_sides = (1 - fee_rate)^2
net_factor = fee_two_sides * pf
```

Это правило тоже должно быть сохранено.

#### 12.5.5 Fast TP/SL grid-search через monotone hit-times

Ключевая идея ноутбука, которую обязательно нужно перенять:  
мы не гоняем каждую сделку по каждой ячейке TP/SL grid отдельным replay.

Вместо этого:

1. для каждой сделки берётся `start = entry_exec_idx + 1`;
2. из `hit_long_tp`, `hit_long_sl`, `hit_short_tp`, `hit_short_sl` получаются времена первого срабатывания уровней;
3. используется монотонность hit-time по уровню;
4. вклад одной сделки раскладывается сразу на регионы TP/SL grid через difference-буферы.

Обязательные буферы fast kernel:

- `row_diff`
- `col_diff`
- `rect_diff`

Обязательные примитивы:

- `add_row_range`
- `add_col_range`
- `add_rect`
- `lower_bound_ge_hit`
- `first_equal_hit`

Смысл:

- если сигнал закрывает сделку раньше, чем часть TP/SL уровней успевает сработать, эти ячейки образуют прямоугольную область одинакового вклада;
- оставшиеся области делятся на TP-dominant и SL-dominant регионы;
- после этого вклад всех сделок суммируется через prefix sums.

Это и есть главный алгоритмический перенос из notebook, который заменяет дорогой полный replay для каждой `(tp_i, sl_i)` ячейки.

#### 12.5.6 Зачем нужен точный replay лучшей ячейки после fast kernel

После fast grid-search нельзя безоговорочно считать найденный максимум финальным `best_ret`.

Причина:

- в short-модели возможен `log(0)` и notebook использует технический `NEG_LARGE`;
- fast kernel нужен для быстрого поиска лучшей ячейки, но не для финальной “канонической” цифры return.

Поэтому production runtime обязан:

1. сначала найти `best_tp_idx` и `best_sl_idx` fast kernel’ом;
2. затем пересчитать только эту лучшую ячейку точным trade-list replay;
3. уже этот replay считать источником истины для final metrics.

Это обязательное правило, а не опциональная оптимизация.

#### 12.5.7 Метрики поверх compact trade list

Из notebook также переносится принцип подсчёта метрик:

- сначала выбирается лучшая TP/SL ячейка;
- потом метрики считаются только по compact trade list для этой ячейки.

Обязательные метрики ядра:

- `total_ret`
- `max_dd`
- `trades`
- `winrate`
- `avg_trade_ret`
- `avg_trade_exec_bars`
- `exposure`
- `sharpe`

Важная деталь notebook:

- `sharpe` считается по сделкам, а не по барам;
- annualization использует `trades_per_year`, а не `bars_per_year_exec`.

Это тоже должно быть явно отражено в v2 runtime docs и тестах.

### 12.6 Что это означает для нашего финального дизайна

Итоговый production runtime v2 должен строиться не по формуле “Stage B = взять earliest hit и на этом остановиться”, а по более точной схеме из notebook:

1. построить `final_signal`;
2. построить compact trade list;
3. выполнить fast monotone TP/SL grid-search на hit-time таблицах;
4. выбрать лучшую ячейку;
5. сделать точный replay только по этой ячейке;
6. посчитать финальные метрики.

То есть главное изменение принципа расчёта по сравнению с v1 такое:

- v1 мыслит “variant -> full execution replay -> metrics”;
- v2 notebook-derived мыслит “variant -> compact trade list -> fast TP/SL cell search -> exact replay of best cell -> metrics”.

Именно эта схема должна считаться целевой для v2.

## 13. Пример алгоритма на простом языке

Пример: `BTCUSDT`, timeframe `1h`, стратегия из `ma.ema(window=50)` и `momentum.rsi(window=14)`.

Что происходит:

1. runtime открывает `current.yaml` и фиксирует, например, `slot_a`;
2. из `slot_a/prices/1h/*` загружается timeline;
3. из `slot_a/signals/1h/ma.ema/signals.i8.npy` выбирается одна row;
4. из `slot_a/signals/1h/momentum.rsi/signals.i8.npy` выбирается одна row;
5. runtime строит общий `final_signal`;
6. когда на `1h` баре возникает вход, runtime находит minute index соответствующего закрытия этого бара;
7. от этой минуты ищется, когда на `1m` впервые сработает TP или SL;
8. одновременно отслеживается следующее `1h` изменение сигнала;
9. кто наступил раньше, тот и закрывает сделку;
10. после этого строятся trade metrics и ranking.

Смысл этой схемы:

- сигналы живут на удобном аналитическом TF;
- risk execution живёт на единой minute base;
- runtime не тратит время на повторный compute и rollup.

## 13A. Запуск, результаты и history: пользовательский сценарий

### 13A.1 Запуск backtest

Пользователь заполняет одну форму `/backtests`:

- инструмент;
- timeframe;
- indicator set;
- source series по индикаторам через checkbox/multi-select;
- ranking metric;
- желаемый `top N` в пределах server runtime config.

После нажатия `Run backtest`:

1. backend автоматически выполняет preflight;
2. если sync budgets проходят, run стартует сразу;
3. если sync budgets не проходят, но background budgets проходят, run автоматически создаётся как background run;
4. если run невозможен и по full budgets, пользователь получает deterministic validation error.

Пользователь не взаимодействует с отдельной кнопкой `Estimate preflight`.

### 13A.2 Таблица результатов

После завершения run пользователь всегда получает одну summary table:

- `top N` строк;
- одна строка = один вариант;
- trades в таблицу не входят.

Таблица должна:

- по умолчанию быть упорядочена по выбранной ranking metric;
- уметь пересортировываться локально по любой доступной summary metric;
- не запускать новый расчёт при пересортировке.

### 13A.3 История запусков

Каждый запуск сохраняется и попадает в `Backtest history`.

History entry должна позволять:

- открыть старый run без повторного расчёта grid;
- увидеть metadata, state и summary table;
- открыть detail page отдельного варианта;
- при background run видеть progress и terminal state.

### 13A.4 Detail page варианта

Когда пользователь кликает в конкретную строку summary table:

1. открывается отдельная страница варианта;
2. backend заново считает только этот один вариант;
3. используются explicit variant params из summary row;
4. используется pinned artifact slot исходного run;
5. пользователю показываются:
   - график equity/price;
   - входы/выходы сделок на графике;
   - подробная статистика;
   - список сделок.

Этот lazy detail расчёт не сохраняется как отдельный persisted run result.

### 13A.5 Сохранение в избранное / strategy library

Из summary table и из detail page пользователь должен иметь возможность сохранить вариант в избранное/strategy library.

Важно:

- это отдельное действие пользователя;
- это не часть persisted run history;
- для этого используются explicit variant params, а не внутренние hash identifiers.

## 14. Оценка хранения и почему отказались от `1m/5m` request timeframes

После удаления 11 тяжёлых индикаторов:

- остаётся примерно `49,310` compute rows.

Если signals хранить как `int8` для разрешённых TF:

- `15m` -> около `1.609 GiB / год / инструмент`
- `30m` -> около `0.805 GiB`
- `1h` -> около `0.402 GiB`
- `2h` -> около `0.201 GiB`
- `4h` -> около `0.101 GiB`
- `6h` -> около `0.067 GiB`
- `8h` -> около `0.050 GiB`
- `1d` -> около `0.017 GiB`
- `2d` -> около `0.008 GiB`
- `3d` -> около `0.006 GiB`

Итого:

- примерно `3.27 GiB / год / инструмент` только на signals.

Это признано допустимым компромиссом.  
Если оставить `1m` и `5m` как request timeframes, storage становится слишком дорогим и практического смысла в них для целевого backtest v2 нет.

## 15. Что нельзя делать

- нельзя оставлять ClickHouse в sync/job hot path;
- нельзя оставлять runtime rollup из `1m`;
- нельзя оставлять Python per-bar execution loop для Stage B;
- нельзя переписывать активный slot in-place;
- нельзя разрешать `1m` и `5m` в backtest requests;
- нельзя хранить timestamps внутри `float32` prices array;
- нельзя возвращать к жизни удалённые индикаторы;
- нельзя silently fallback на old engine без feature-flag и явного решения;
- нельзя расширять `signals.v1.params` в full grid в initial v2.

## 16. План миграции по этапам

### Этап 1. Документы и продуктовые ограничения

- создать этот final doc;
- обновить `base_refactor_plan.md` как superseded;
- обновить API/docs/runtime-defaults semantics;
- зафиксировать список удаляемых индикаторов;
- зафиксировать список разрешённых TF.
- зафиксировать новый persisted run/history contract и auto-preflight launch semantics.

### Этап 2. Чистка indicator zoo

- удалить 11 индикаторов из config, registry, docs, kernels, tests, API/UI;
- добавить `signals.v1.params` defaults для оставшихся поддержанных indicator ids.

### Этап 3. Artifact store contracts

- реализовать slot layout;
- реализовать `current.yaml`;
- реализовать manifest schemas;
- реализовать slot pinning policy для jobs.

### Этап 3A. Persisted run storage generalization

- расширить existing PG job storage до persisted run history model;
- убрать persisted report/trades из top results storage;
- добавить summary metrics storage для top rows;
- добавить denormalized columns, нужные для history list и filters.

### Этап 4. Precompute pipeline

- bootstrap initial slot, если для symbol root ещё нет valid `current.yaml` и published slot;
- daily load canonical `1m` candles;
- rollup prices for allowed TF;
- compute signals for remaining zoo;
- build mappings;
- build `1m hit-times`;
- validate and publish.

После bootstrap steady-state policy должна быть такой:

- `prices`, `mappings`, `signals` не пересчитываются full-history по умолчанию, а используют
  bounded incremental rebuild по explicit `lookback_policy.*_tail_bars_1m`;
- `hit_times/1m` тоже должны использовать bounded incremental rebuild по
  `lookback_policy.hit_times_tail_bars_1m`, а не обязательный full recompute на каждый daily run;
- если reuse prerequisites нарушены (missing files, manifest drift, config drift, grid drift),
  для конкретного symbol root выполняется deterministic full rebuild с тем же publish contract.

### Этап 5. Runtime kernels

- signal row loading;
- aggregation kernel;
- compact trade builder;
- `1m risk exit` kernel;
- metrics kernel.

### Этап 6. Sync cutover

- подключить v2 runtime к `POST /backtests`;
- встроить auto-preflight внутрь `POST /backtests`;
- реализовать auto-fallback `sync -> background`;
- прогнать parity/perf suites;
- включить behind feature flag;
- затем сделать default.

### Этап 7. Jobs cutover

- подключить тот же runtime к background run worker;
- внедрить pinned slot semantics;
- прогнать long-running job safety tests;
- включить v2 path.

### Этап 7A. History и detail page

- сделать UI tab `Backtest history`;
- сделать run summary page на persisted results;
- сделать lazy single-variant detail page;
- убрать обязательный manual preflight из UI;
- убрать trades из top results table.

### Этап 8. Legacy cleanup

- убрать старый hot path из production usage;
- оставить только controlled legacy fallback на переходный период;
- затем убрать и его.

## 17. Риски и меры контроля

### 17.1 Риск: артефакты занимают слишком много диска

Меры:

- запретить `1m/5m` request TF;
- удалить 11 тяжёлых индикаторов;
- держать `signals.v1.params` в режиме default-only;
- вводить storage budgets и checks на publish.

### 17.2 Риск: race между publish и jobs

Меры:

- slot pinning;
- publish block, если неактивный слот ещё занят job-ами;
- `current.yaml` switch только после полной валидации.

### 17.2A Риск: history storage начнёт раздуваться

Меры:

- хранить только summary top-N rows;
- не хранить trades/equity/report bodies в persisted results;
- пересчитывать detail page лениво только для одного варианта;
- при необходимости вводить retention/archival policy отдельно, не смешивая её с runtime contract.

### 17.3 Риск: сильное расхождение semantics с v1

Меры:

- документировать новую semantics явно;
- добавить отдельные golden fixtures для v2;
- не обещать old close-fill parity там, где теперь `1m risk execution`.

### 17.4 Риск: combinatorial explosion вернётся через signal params

Меры:

- initial v2 = default-only `signals.v1.params`;
- signal-grid expansion обсуждать только отдельным документом и отдельными budget measurements.

### 17.5 Риск: auto-fallback в background будет восприниматься как “магия”

Меры:

- response должен явно возвращать `execution_mode`;
- UI должен явно показывать, что run переведён в background;
- history entry должна появляться сразу;
- cancel/status semantics должны быть едиными для всех background runs.

## 18. Критерии готовности

Система считается готовой, когда выполняются все условия:

- sync/jobs hot path делает `0` ClickHouse queries;
- sync/jobs hot path делает `0` вызовов `IndicatorCompute.compute(...)`;
- `1m` и `5m` requests детерминированно отклоняются;
- удалённые indicator ids детерминированно отклоняются;
- non-default `signals.v1.params` детерминированно отклоняются;
- ручной preflight больше не нужен для запуска;
- каждый запуск появляется в `Backtest history`;
- результаты persisted history хранят только summary top-N rows;
- top results table не содержит persisted trades;
- detail page одного варианта пересчитывается лениво и использует pinned artifact slot;
- Stage A и Stage B работают на artifacts-only runtime;
- long-running jobs воспроизводимы через pinned slot identity;
- perf на representative workloads существенно лучше текущего v1;
- docs, tests, API, config и runtime contracts приведены в консистентное состояние.

## 19. Связанные документы, которые нужно создать или обновить

Update note after R10-01 / R10-02:

- R10-01 закрыл production hot-path cutover без silent legacy fallback.
- R10-02 должен оставить один canonical doc set для artifact-backed runtime, runs-first UX и
  `summary-only` persisted history.
- После этого final plan остаётся semantic source-of-truth, а незакрытым handoff остаётся только
  R10-03 perf/runbook closure.
- R10-03 closure не добавляет новый runtime/API surface и должна завершаться только через:
  - deterministic test closure;
  - benchmark protocol c `0 CH calls on hot path` и
    `0 IndicatorCompute.compute(...) calls on hot path`;
  - explicit rollout / rollback runbook.

### Создать

- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/runbooks/backtest-rollout-rollback.md`
- `docs/architecture/backtest/backtest-runs-history-v2.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

### Обновить

- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`
- `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md` (historical / compatibility reference)
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
- `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
- `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md`
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- `docs/architecture/apps/web/web-backtest-sync-ui-preflight-save-variant-v1.md`
- `docs/architecture/apps/web/web-backtest-jobs-ui-async-v1.md`
- `docs/architecture/indicators/indicators_formula.yaml`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md`
- `docs/architecture/indicators/indicators-trend.md`
- `docs/architecture/indicators/indicators-momentum.md`
- `docs/architecture/indicators/indicators-volatility.md`
- `docs/architecture/indicators/indicators-volume.md`
- `configs/prod/indicators.yaml`

## 20. Краткий итог

Новая архитектура backtest v2 строится вокруг простой идеи:

- всё тяжёлое считаем заранее;
- runtime читает только готовые arrays;
- сигналы живут на выбранном TF;
- risk execution живёт только на `1m`;
- публикация артефактов делается через два слота и pointer file;
- продукт сознательно отказывается от слишком тяжёлых индикаторов и мелких request timeframes;
- запуск всегда делает auto-preflight и создаёт persisted run;
- если sync budgets не проходят, но background budgets проходят, run автоматически уходит в background и всё равно попадает в history;
- результаты хранятся как summary-only top-N table;
- trades и графики считаются только лениво для выбранного варианта;
- пользователь не видит внутренние reproducibility hashes и работает через runs/history/detail pages;
- initial v2 сознательно ограничивает `signals.v1.params` до default-only, чтобы не вернуть combinatorial explosion.

Это не “самая абстрактно красивая” архитектура, но это реалистичная и производительно оправданная архитектура для данного репозитория и задачи ускорения backtest.
