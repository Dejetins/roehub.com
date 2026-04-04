---
title: План доработки и ускорения backtest runtime v1
version: 1
status: draft
owner: backtest
---

# План доработки и ускорения backtest runtime v1

Статус: proposed follow-up roadmap after R10-03  
Дата фиксации: 2026-04-04  
Область: `backtest`, `apps/api`, `apps/web`, artifact precompute/runtime, persisted runs UX

## 1. Зачем нужен этот документ

Текущий production backtest уже перешёл на artifact-backed runtime:

- sync/jobs path читает published artifacts, а не ClickHouse;
- Stage A / Stage B уже работают на `prices/<tf>`, `signals/<tf>`, `mappings/<tf>`, `hit_times/1m`;
- результаты живут в persisted runs storage (`backtest_jobs`, `backtest_job_top_variants`,
  `backtest_job_stage_a_shortlist`);
- есть history/status/top/detail UX.

Но текущий runtime всё ещё проигрывает notebook-пайплайну по orchestration speed:

- внутри одного run почти нет реальной multi-core загрузки;
- Stage B остаётся слишком последовательным;
- запуск может быть слишком тяжёлым для `sync_inline`;
- runtime пока не использует staged pruning в духе notebook;
- UX пока не показывает полноценный progress bar `0..100%` с примерным ETA.

Цель этого документа: описать реалистичный путь, как ускорить backtesting **без потери
важных результатов по умолчанию**, сохранив один exact scorer как source of truth и добавив
гибридные ускорители только там, где они действительно оправданы.

## 2. Что считаем уже реализованным baseline

Вне scope этого follow-up плана:

- artifact store v2 с `current.yaml`, slot pinning и whole-slot validation;
- artifact precompute/publish pipeline;
- artifact-backed Stage A / Stage B kernels;
- persisted runs/history/detail API;
- summary-only top rows;
- strict runtime defaults / request contract / slot-pinned context.

Основные reference документы:

- [base_refactor_plan.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/base_refactor_plan.md)
- [backtest-refactor-final-plan-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-refactor-final-plan-v2.md)
- [backtest-runtime-kernels-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- [backtest-compute-notebook-algorithm-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md)
- [backtest-precompute-runner-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-precompute-runner-v2.md)
- [backtest-runs-history-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [backtest-job-runner-worker-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)
- [06_backtest_compute.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/06_backtest_compute.ipynb)

## 3. Ключевые решения этого follow-up плана

### 3.1 Один exact scorer остаётся каноническим source of truth

Нельзя строить новую систему так, чтобы у каждого family был свой "почти отдельный backtest engine".

Правило:

- один общий orchestrator;
- один общий exact final scorer;
- любые heuristic layers и family plugins могут только **предлагать shortlist/proposals**;
- финальная оценка retained candidates всегда идёт через exact kernels.

Это сохраняет:

- поддерживаемость;
- сравнимость результатов;
- единый benchmark harness;
- предсказуемость rollout.

### 3.2 Heuristics разрешены, но только явно и умеренно

Approximate search допускается, но не как silent default для всех запросов.

Правило:

- exact path остаётся canonical baseline;
- heuristic path сначала вводится как explicit execution profile;
- rollout идёт только после benchmark gates по recall / overlap / latency / memory.

### 3.3 Warm feature cache нужен, но только для row-local deterministic features

Это хорошее предложение, но есть важное ограничение:

- в artifact cache надо класть только такие features, которые зависят от самой signal row и
  artifact timeline;
- туда не надо класть pair-specific или runtime-policy-specific эвристики.

Хорошие кандидаты:

- activity / nonzero count;
- direction balance;
- edge count proxy;
- transition count;
- mean hold proxy;
- dense/sparse profile;
- simple no-risk row summary, если она не зависит от TP/SL grid.

Плохие кандидаты:

- pair-only proxy score;
- ranking policy, завязанная на конкретный UI mode;
- runtime-threshold-dependent feature values.

### 3.4 Plugin timeout должен быть budget-based, а не fixed hardcoded

Фиксированный timeout вроде `500ms` выглядит просто, но плохо переживает разные symbol/timeframe
масштабы.

Вместо этого:

- plugin получает небольшой planning budget из execution profile;
- при timeout/error orchestrator пишет warning и идёт по universal path;
- после `N` failures plugin попадает под circuit breaker до конца текущего run.

### 3.5 Request-shape control должен сначала переводить run в background, а не просто отвергать его

Если запрос валиден, но тяжёлый, это не повод ломать UX жёстким reject, если runtime ещё может
обработать его в background.

Правило:

- hard reject только при реальном violation guard / contract;
- heavy-but-valid requests по умолчанию классифицируются в `background_auto`.

## 4. Целевая архитектура

Итоговая схема должна выглядеть так:

```text
POST /backtests
  -> request normalization
  -> cost model
  -> execution profile selection
  -> slot-pinned artifact context
  -> row feature loading (warm cache)
  -> universal shortlist or family plugin proposals
  -> exact Stage A / Stage B on survivors
  -> persisted summary rows
  -> progress/ETA updates for UX
```

Ключевые блоки:

1. `Universal exact core`
- быстрый exact runtime без изменения semantics;
- parallel Stage B;
- лучшая locality / меньше Python overhead.

2. `Universal conservative shortlist`
- generic staged pruning для любых indicator blocks;
- работает через row-local cached features и cheap runtime features;
- не заменяет exact scorer, а сокращает число кандидатов.

3. `Pluggable family accelerators`
- optional proposal plugins;
- family может дать stronger shortlist/proxy;
- при отсутствии plugin или при ошибке path автоматически деградирует в universal mode.

4. `ExecutionProfile`
- одна явная сущность, которая решает, как именно исполняется run.

## 5. Обязательный UX contract: progress bar 0..100% + ETA

Это не nice-to-have, а обязательная часть follow-up плана.

Пользователь должен видеть:

- текущую стадию run;
- прогресс в процентах от `0` до `100`;
- примерное оставшееся время;
- execution mode (`sync_inline`, `background_auto`, `hybrid_*`, `exact_*`).

### Что это значит простыми словами

Сейчас пользователь видит "что-то считается".  
Нужно, чтобы он видел:

- `Stage A shortlist: 32%`
- `Estimated remaining: ~18s`
- `Execution profile: exact_parallel`

или:

- `Hybrid shortlist: 68%`
- `Exact evaluation: pending`
- `Estimated remaining: ~1m 40s`

### Как это считать

Прогресс должен быть не fake-spinner, а real weighted progress:

- Stage A units;
- Stage B units;
- finalizing units.

Нужны:

- stage weights по execution profile;
- `processed_units / total_units`;
- short-term throughput estimate;
- historical benchmark fallback для ETA, если run только стартовал.

Важно для следующих milestone:

- stage weights не должны надолго остаться отдельной read-model таблицей вне
  `ExecutionProfile`;
- до включения benchmark-based ETA нужно свести progress/ETA semantics к одному source of truth,
  чтобы profile weights не жили отдельно от profile contract/config.

### Какие файлы и документы трогаем

Ожидаемые кодовые точки:

- [backtest_job.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/domain/entities/backtest_job.py)
- [backtest_runs.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/dto/backtest_runs.py)
- [backtest_runs.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/routes/backtest_runs.py)
- [backtest_runs_ui.js](/Users/daniildegtyarev/Projects/roehub.com/apps/web/dist/backtest_runs_ui.js)
- [backtest_ui.js](/Users/daniildegtyarev/Projects/roehub.com/apps/web/dist/backtest_ui.js)
- [backtest_run_summary.html](/Users/daniildegtyarev/Projects/roehub.com/apps/web/templates/backtest_run_summary.html)
- [backtests.html](/Users/daniildegtyarev/Projects/roehub.com/apps/web/templates/backtests.html)
- [backtest_runs_history_api_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py)

Ожидаемые docs:

- [backtest-runs-history-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [web-backtest-history-and-variant-detail-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md)
- [backtest-job-runner-worker-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)

## 6. Execution profile как явная сущность

Это правильное предложение, его надо зафиксировать сразу.

Пример shape:

```python
@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    mode: Literal[
        "exact_small",
        "exact_parallel",
        "hybrid_conservative",
        "hybrid_family",
    ]
    shortlist_config: ShortlistConfig
    parallelism: ParallelismConfig
    feature_flags: Mapping[str, bool]
    planning_budget_ms: int
```

Зачем это нужно:

- проще тестировать;
- проще benchmark'ить;
- проще rollout через config/flags;
- проще объяснять UX, почему один run идёт inline, а другой в hybrid background mode.

Ожидаемые точки:

- [backtest_runtime_config.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py)
- [artifact_runtime_plan_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py)
- новый модуль `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- [backtests.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/dto/backtests.py)
- [backtest_runtime_defaults.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/dto/backtest_runtime_defaults.py)

## 7. Milestones и EPICs

Ниже предложен порядок внедрения. Он намеренно разделяет:

- безопасные exact improvements;
- cache/foundation work;
- heuristic layers;
- family plugins;
- adaptive selector.

### Milestone A. Foundation: profiles, progress, benchmarks

#### EPIC A1. ExecutionProfile и launch classification

Что делаем простыми словами:

- описываем run не набором скрытых if/else, а явным execution profile;
- маленькие run идут exact inline;
- тяжёлые, но валидные, уходят в background или parallel profile.

Пример:

- `120 variants` -> `exact_small`
- `3_000 variants` -> `exact_parallel`
- `40_000 variants` -> `hybrid_conservative` или `background_auto`

Кодовые точки:

- [backtest_runtime_config.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py)
- [artifact_runtime_plan_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py)
- [backtest_runs_api_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py)
- [backtest.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/wiring/modules/backtest.py)
- [backtest.yaml](/Users/daniildegtyarev/Projects/roehub.com/configs/prod/backtest.yaml)

Документы:

- [backtest-api-post-backtests-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [backtest-runs-history-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)

#### EPIC A2. Progress/ETA contract для UX

Что делаем простыми словами:

- вводим stage-aware progress model;
- UI показывает нормальный progress bar `0..100%`;
- ETA строится из текущего throughput + benchmark fallback.

Пример:

- `Stage A 45% / ETA 12s`
- `Stage B 80% / ETA 5s`
- `Finalizing 95% / ETA <1s`

Кодовые точки:

- [backtest_job.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/domain/entities/backtest_job.py)
- [backtest_runs.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/dto/backtest_runs.py)
- [backtest_runs_ui.js](/Users/daniildegtyarev/Projects/roehub.com/apps/web/dist/backtest_runs_ui.js)
- [backtests.html](/Users/daniildegtyarev/Projects/roehub.com/apps/web/templates/backtests.html)
- [backtest_run_summary.html](/Users/daniildegtyarev/Projects/roehub.com/apps/web/templates/backtest_run_summary.html)

Документы:

- [backtest-runs-history-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [web-backtest-history-and-variant-detail-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md)

#### EPIC A3. Benchmark corpus и rollout gates

Что делаем:

- фиксируем benchmark suite, на котором потом валидируем и exact, и hybrid profiles;
- отдельно мерим edge cases.

Обязательные benchmark slices:

- small grids;
- medium grids;
- huge grids;
- low-activity signals;
- high-correlation families;
- multi-block strategies;
- small-grid overhead от hybrid layer;
- memory footprint при beam search.

Ожидаемые файлы:

- новый doc `docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md`
- новые tests/perf fixtures рядом с существующими benchmark fixtures
- [backtest-v2-benchmarks.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-v2-benchmarks.md)

### Зафиксированные follow-up constraints после фактического Milestone A

После реализации A1/A2/A3 нужно считать явно открытыми два follow-up пункта.

#### A-Follow-up-1. Progress/ETA profile weights должны вернуться в единый profile contract

Что фактически получилось сейчас:

- typed `ExecutionProfile` уже есть;
- но stage weights для `progress_percent` / `eta_seconds` живут отдельной таблицей в read path.

Почему это важно:

- сейчас появились два semantic source of truth для profile behavior:
  - сам `ExecutionProfile`;
  - отдельная mapping-таблица progress weights;
- пока это допустимо для Milestone A, но в `Milestone B` и особенно в `Milestone F` это начнёт
  мешать adaptive selector, benchmark-based ETA и profile rollout.

Что обязательно учесть дальше:

- в одном из следующих exact/foundation EPIC нужно перенести stage weights в profile contract
  или в другой единый config-driven source of truth;
- benchmark/history ETA fallback нельзя строить на постоянной основе поверх второй независимой
  таблицы весов;
- любые новые profile literals или adaptive policy не должны требовать отдельного ручного
  обновления progress mapping вне profile layer.

Куда это относится:

- [backtest_runs_history_api_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py)
- [execution_profile_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py)
- [backtest_runtime_config.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py)

#### A-Follow-up-2. Benchmark `exact_baseline` и active runtime exact default пока не совпадают

Что фактически получилось сейчас:

- benchmark corpus использует `exact_parallel` как `exact_baseline` evidence anchor;
- active runtime-enabled default profile пока остаётся `exact_small`.

Почему это важно:

- это не bug, пока corpus используется как evidence surface;
- но в `Milestone B/F` это может стать источником путаницы между:
  - benchmark baseline;
  - current active default exact mode;
  - future adaptive selector decisions.

Что обязательно учесть дальше:

- до rollout benchmark-based ETA, adaptive selector и automatic profile promotion нужно явно
  решить одно из двух:
  - либо `exact_parallel` остаётся benchmark baseline независимо от active default,
    и это продолжает документироваться как intentional distinction;
  - либо corpus baseline и runtime default realign-ятся вместе одним изменением;
- нельзя молча смешивать benchmark evidence anchor и текущий active exact mode в одном и том же
  decision path.

Куда это относится:

- [backtest-runtime-acceleration-benchmarks-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md)
- [backtest-v2-benchmarks.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-v2-benchmarks.md)
- [backtest_runtime_acceleration_benchmark_corpus_v1.json](/Users/daniildegtyarev/Projects/roehub.com/tests/perf_smoke/contexts/backtest/fixtures/backtest_runtime_acceleration_benchmark_corpus_v1.json)
- [backtest.yaml](/Users/daniildegtyarev/Projects/roehub.com/configs/prod/backtest.yaml)

### Milestone B. Universal exact acceleration

Это первый milestone, который должен дать быстрый выигрыш при нулевом semantic drift.

#### EPIC B1. Parallel Stage B

Что делаем простыми словами:

- exact Stage B перестаёт быть почти одноядерным;
- один coordinator process;
- несколько worker processes;
- все читают artifacts readonly через memmap;
- final merge deterministic.

Пример:

- вместо `1 process x 50s` получаем `6 workers x ~12-18s`, не меняя итоговые winners.

Кодовые точки:

- [artifact_backed_stage_b_scorer_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py)
- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)
- [risk_exit_kernel_1m.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py)
- [metrics_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/metrics_kernel.py)

Документы:

- [backtest-runtime-kernels-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- [backtest-job-runner-worker-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-worker-v1.md)

#### EPIC B2. Better memmap locality и loader discipline

Что делаем:

- меньше случайных row reads;
- лучше chunk order;
- меньше reopen/reseek overhead;
- subset loading под profile-aware access patterns.

Кодовые точки:

- [price_arrays_loader.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py)
- [signal_matrix_loader.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py)
- [artifact_slot_resolver.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py)

#### EPIC B3. Lower Python overhead / object churn

Что делаем:

- меньше transient objects в planning and scoring loops;
- больше tuple/array based payloads;
- меньше repeated normalization and mapping churn.

Кодовые точки:

- [artifact_runtime_plan_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py)
- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)
- [contracts.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/contracts.py)

#### EPIC B4. Better request-shape control

Что делаем:

- контролируем, чтобы run исполнялся ровно по request shape;
- тяжёлые, но валидные runs уводим в `background_auto` раньше;
- hard reject оставляем только для настоящих guard violations.

Кодовые точки:

- [artifact_runtime_plan_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py)
- [backtest_runs_api_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py)
- [backtest_ui.js](/Users/daniildegtyarev/Projects/roehub.com/apps/web/dist/backtest_ui.js)

Документы:

- [backtest-api-post-backtests-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [web-backtest-runtime-defaults-endpoint-v1.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md)

### Milestone C. Warm row feature cache

#### EPIC C1. Precompute signal feature artifacts

Что делаем простыми словами:

- считаем row-local features один раз при precompute;
- сохраняем рядом с сигналами;
- runtime, notebook и plugins используют один и тот же feature source.

Пример artifact layout:

```text
signals/1h/ma.ema/signals.i8.npy
signal_features/1h/ma.ema/features.f32.npy
signal_features/1h/ma.ema/manifest.yaml
```

Или компактный вариант:

```text
signal_features/<tf>/<indicator_id>/features.f32.npy  # [V, F]
```

Кодовые точки:

- [artifact_precompute_runner.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py)
- [contracts.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/contracts.py)
- новый loader `src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py`
- [signal_rules_engine_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py) только если нужен deterministic feature derivation from final row

Документы:

- [backtest-precompute-runner-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-precompute-runner-v2.md)
- [backtest-artifact-store-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-artifact-store-v2.md)
- [backtest-compute-notebook-algorithm-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md)

#### EPIC C2. Runtime access to cached features

Что делаем:

- runtime сначала открывает cached features;
- если feature artifact отсутствует и profile его не требует, exact path остаётся рабочим;
- для heuristic profiles cached features становятся preferred input.

Кодовые точки:

- новый module `signal_features_loader_v2.py`
- [artifact_runtime_plan_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py)
- [stage_a_shortlist_builder_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py)

### Milestone D. Universal conservative shortlist

Это первый approximate milestone. Он должен быть opt-in.

#### EPIC D1. Generic row scorer

Что делаем:

- считаем cheap row score для любого indicator block;
- используем cached features + cheap runtime-derived stats.

Пример:

- из `12_000` row candidates оставляем `1_500`, но не только по raw score, а с diversity buckets.

Новый код:

- `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py`

#### EPIC D2. Hierarchical shortlist builder

Это должно быть новым модулем, а не перегрузкой уже существующего exact Stage A builder.

Что делаем:

- per-block shortlist;
- diversified retain;
- beam combine across blocks;
- exact scorer видит уже survivors.

Пример:

- block A: 2000 rows -> 150 retained
- block B: 1500 rows -> 120 retained
- beam combine keeps 600 partial combos
- exact Stage B receives only final 300 survivors

Новый код:

- `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`

Документы:

- новый doc `docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md`
- [backtest-runtime-kernels-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)

#### EPIC D3. Rollout gates для approximate mode

Обязательные acceptance gates:

- top-1 recall vs exact baseline;
- top-10 overlap vs exact baseline;
- отдельные slices для low-activity и high-correlation;
- small-grid overhead;
- memory footprint.

Рекомендуемый стартовый барьер rollout:

- top-1 recall >= 99% на baseline corpus;
- top-10 overlap >= 90%;
- low-activity slice top-1 recall >= 97%;
- no unacceptable regression on small-grid latency.

Числа могут быть пересмотрены после первых benchmark runs, но thresholds должны быть explicit.

### Milestone E. Pluggable family accelerators

#### EPIC E1. Plugin contract

Что делаем:

- family plugin может предложить row shortlist / pair shortlist / proxy score;
- final exact scorer не меняется.

Пример контракта:

```python
class FamilyAccelerationPlugin(Protocol):
    def propose(
        self,
        *,
        context: RuntimePlanningContextV2,
        profile: ExecutionProfile,
    ) -> ProposalResult:
        ...
```

#### EPIC E2. Failure handling

Обязательные правила:

- optional plugin budget;
- timeout/error -> warning + universal fallback;
- circuit breaker до конца run после repeated failures.

Новый код:

- `src/trading/contexts/backtest/application/services/v2/family_plugins/registry_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/contracts_v2.py`
- `src/trading/contexts/backtest/application/services/v2/family_plugins/circuit_breaker_v2.py`

#### EPIC E3. First plugin

Первый plugin имеет смысл делать только после D-милестоуна и benchmark harness.

Кандидат:

- MA-family

Но важно:

- plugin не должен становиться special backtest engine;
- он должен лишь улучшать proposal layer.

Документы:

- новый doc `docs/architecture/backtest/backtest-family-accelerators-v1.md`

### Milestone F. Adaptive selector и controlled rollout

#### EPIC F1. Cost-model based profile selection

Что делаем:

- orchestrator по cost model выбирает profile;
- profile selection использует:
  - grid cardinality;
  - estimated Stage A / Stage B work;
  - memory budget;
  - runtime mode (`sync` vs background);
  - plugin availability.

#### EPIC F2. Feature flags и environment rollout

Что делаем:

- сначала `dev`;
- затем `test/prod shadow`;
- затем opt-in prod;
- затем selective default for large runs.

Кодовые точки:

- [backtest_runtime_config.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py)
- [backtest.yaml](/Users/daniildegtyarev/Projects/roehub.com/configs/prod/backtest.yaml)
- [backtest.py](/Users/daniildegtyarev/Projects/roehub.com/apps/api/wiring/modules/backtest.py)

## 8. Рекомендуемая последовательность внедрения

Вот порядок, который я считаю правильным:

1. `Milestone A`
- profiles
- progress/ETA
- benchmark corpus

2. `Milestone B`
- exact acceleration
- parallel Stage B
- request classification

3. `Milestone C`
- warm row feature cache
- без включения heuristics в runtime by default

4. `Milestone D`
- universal conservative shortlist
- только под feature flag / opt-in profile

5. `Milestone E`
- family plugin contract
- первый plugin

6. `Milestone F`
- adaptive selector
- rollout based on evidence

## 9. Что считаем успехом

План считается успешно реализованным, если выполняются одновременно следующие условия:

1. Small exact runs остаются exact и не становятся заметно медленнее.
2. Medium/large exact runs ускоряются за счёт parallel Stage B и lower overhead.
3. Hybrid profiles дают существенный speedup на больших grids.
4. Approximate modes не теряют winners "случайно" чаще, чем допускают rollout gates.
5. UX показывает честный progress `0..100%` и reasonable ETA.
6. Поддерживаемость не деградирует:
   - один exact scorer,
   - одна runtime orchestration surface,
   - plugins только proposal-layer.
7. Progress/ETA weights и profile semantics не живут в двух независимых местах.
8. Benchmark baseline и active exact default различаются только если это явно задокументировано и
   осознанно используется в rollout logic.

## 10. Anti-patterns, которых надо избежать

- Не делать heuristics silent default без benchmark rollout.
- Не делать у каждого family свой почти независимый scorer.
- Не тащить runtime-specific proxy features в precomputed artifact cache.
- Не вводить fixed hardcoded plugin timeout без учета execution budget.
- Не смешивать request-shape validation с безусловным reject heavy runs.
- Не перегружать existing exact Stage A builder approximate-логикой; для hybrid path нужен
  отдельный shortlist builder.
- Не держать profile semantics в двух разных таблицах, если обе влияют на rollout, ETA или
  adaptive selector.
- Не смешивать benchmark evidence anchor и active runtime default без явного решения в docs/config.

## 11. Простой итог

Если сказать совсем коротко, правильная целевая модель такая:

- exact runtime остаётся основой;
- его сначала ускоряем без изменения результата;
- затем добавляем cached row features;
- затем добавляем осторожный universal hybrid shortlist;
- затем разрешаем family plugins как proposal layer;
- всё это показываем пользователю через нормальный progress bar и ETA.

Это и есть наилучший компромисс между:

- скоростью;
- качеством результата;
- поддерживаемостью;
- возможностью постепенно расширять систему новыми indicator families.
