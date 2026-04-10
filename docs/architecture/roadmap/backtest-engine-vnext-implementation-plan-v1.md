---
title: План переустройства backtest engine vNext
version: 1
status: draft
owner: backtest
---

# План переустройства backtest engine vNext

Статус: proposed implementation roadmap for notebook-derived engine redesign  
Дата фиксации: 2026-04-10  
Область: `backtest`, `apps/api`, `apps/web`, `apps/worker`, artifact precompute/runtime, persisted runs UX

## 1. Зачем нужен этот документ

Текущий production backtest уже перешёл на artifact-backed runtime и уже умеет:

- исполнять sync и background runs через один shared engine;
- читать `prices/<tf>`, `signals/<tf>`, `mappings/<tf>`, `hit_times/1m` из published artifacts;
- сохранять persisted runs, summary-only top rows и progress/history surfaces;
- использовать exact Stage B scorer с `1m hit-times` и fast TP/SL search.

Но текущий hot path всё ещё архитектурно не соответствует целевой скорости:

- Stage A shortlist строится слишком дорогим exact-first breadth pass;
- expensive compute начинается слишком рано, до staged narrowing;
- пользовательская surface перегружена устаревшими knobs;
- default launch path всё ещё несёт historical baggage, которое уже не нужно продукту.

Этот документ фиксирует реалистичный план, как:

- заменить current Stage A hot path на notebook-derived universal engine;
- оставить sync и `backtest-job-runner` на одном shared runtime;
- сохранить exact scorer как source of truth;
- убрать лишние публичные поля и лишнюю user-facing сложность;
- перевести engine на понятную staged модель, из которой потом можно делать последовательные
  machine-readable prompts без конфликтов.

## 2. Что считаем уже реализованным baseline

Вне scope этого roadmap:

- artifact store v2 с slot pinning и `current.yaml`;
- artifact precompute/publish pipeline;
- shared sync/background runtime orchestration;
- persisted runs/history/detail API;
- summary-only top rows в persisted storage;
- current Stage B exact scorer на `1m hit-times`;
- `direction_mode`, `sizing_mode`, `primary_metric` как допустимая product surface;
- on-demand variant detail endpoint как отдельная пользовательская операция.

Основные reference документы и файлы:

- [Final Backtest Refactor Plan v2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-refactor-final-plan-v2.md)
- [План доработки и ускорения backtest runtime v1](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md)
- [Backtest Runtime Kernels V2](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- [Backtest Job Runner v2 -- claimed background worker для persisted runs](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)
- [Backtest API v1 — `POST /backtests`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-api-post-backtests-v1.md)
- [tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb)

Historical reference only:

- [tests/notebook_tests/06_backtest_compute.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/06_backtest_compute.ipynb)

Для этого roadmap canonical experimental anchor один:

- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`

`06_backtest_compute.ipynb` больше не должен использоваться как основной implementation anchor
для новых prompts. Он остаётся только как historical precedent, если нужно объяснить происхождение
отдельных kernel-паттернов.

## 3. Ключевые решения этого roadmap

### 3.1 Новый redesign anchor

Основной экспериментальный и смысловой anchor для будущего engine redesign:

- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`

Именно он должен описывать:

- универсальные prefilter-паттерны;
- работу с `15m` signal timeline и `1m` execution timeline;
- hit-time based risk search;
- reference-vs-fast self-check;
- bounded exact evaluation на shortlisted candidates.

### 3.2 Переносим только универсальные kernel-паттерны

Из notebook подхода в engine переносится только следующее:

- `trade-list-first design`;
- `prefilter before exact path`;
- `hit-time tables`;
- `fast monotone TP/SL kernel`;
- `reference-vs-fast self-check`.

Не переносится pair-specific orchestration как product contract.

Новый engine должен оставаться generic для `N-indicator` search, а не превращаться в
специализированный runtime только под старый `SMA + EMA` notebook shape.

### 3.3 `compact trade representation` остаётся только внутренним exact payload

Здесь фиксируется важное различие:

- engine MAY строить внутреннее compact trade representation как hidden exact intermediate form;
- engine MUST NOT materialize full user-facing trades/report bodies в default launch path;
- пользовательские trades строятся только по явному on-demand запросу для выбранного варианта.

Иными словами:

- `trade-list-first` это internal engine pattern;
- `full trades list` это не часть default run result и не часть массового launch path.

### 3.4 Exact scorer остаётся единым source of truth

Нельзя построить новую систему так, чтобы:

- prefilter жил отдельно;
- sync path жил отдельно;
- job-runner жил отдельно;
- а final exact scorer различался между режимами.

Правило:

- один shared planner;
- один shared sync/background engine;
- один exact Stage B scorer как финальная authority;
- любые heuristic layers only narrow the search and never replace the final exact authority.

### 3.5 Public launch surface упрощается

Публичная advanced surface должна быть уменьшена.

Оставляем как user-facing control:

- `direction_mode`
- `sizing_mode`
- `primary_metric`
- `top_k` / top-N semantics
- обычные execution/risk settings

Убираем из публичного user-facing input:

- `secondary_metric`
- `warmup_bars`
- `top_trades_n`

### 3.6 `primary_metric` остаётся, `secondary_metric` удаляется

Пользователь по-прежнему должен иметь право выбрать, по какой метрике сортировать `top N`.

Но ranking contract упрощается:

- один `primary_metric`;
- deterministic tie-break сохраняется;
- `secondary_metric` больше не участвует ни в public request, ни в defaults, ни в runtime
  ordering contract.

### 3.7 `warmup_bars` становится только internal derived value

Пользователь не должен задавать `warmup_bars`.

Правило:

- `warmup_bars` вычисляется внутри engine детерминированно из effective indicator requirements;
- поле исчезает из public API/UI/defaults surface;
- при необходимости derived warmup MAY оставаться только internal/debug metadata.

### 3.8 `top_trades_n` удаляется

`top_trades_n` был нужен старому запуску, где trades могли materialize-иться для части top rows.

В новой модели:

- launch path summary-only;
- trades only on demand;
- массовая materialization trades для top rows не нужна.

Следовательно:

- `top_trades_n` удаляется из public API/UI/defaults/runtime normalization;
- default launch path больше не содержит массового trade materialization behavior.

### 3.9 Дополнительные user-facing filters сейчас не вводим

Фильтры вида:

- `max_drawdown_pct <= X`
- `profit_factor >= Y`
- `trade_count >= Z`

не включаются в redesign baseline.

Причина:

- они усложняют request contract;
- они усложняют ranking semantics;
- они усложняют request hash и persisted reproducibility contract;
- они не являются необходимыми, чтобы сначала сделать быстрый engine core.

Такие constraints MAY появиться отдельно позже, но не должны быть частью первого набора prompts.

### 3.10 Wider TP/SL grid это отдельная, но обязательная зависимость

Production artifacts должны получить более широкий TP/SL grid.

Это отдельная ветка работы, но она является обязательной dependency для полного rollout нового
engine, потому что:

- fast monotone TP/SL kernel должен работать на canonical published grid;
- runtime не должен опираться на ad hoc runtime scans для production exact path там, где grid
  должен быть artifact-backed.

### 3.11 `signal_features` пока не делаем обязательным контрактом

Первый redesign этап не должен зависеть от обязательного `signal_features` cache.

Рекомендация этого roadmap:

- prefilter сначала строится на signal matrices и дешёвых runtime proxy features;
- `signal_features` остаётся optional accelerator surface;
- делать его canonical mandatory artifact можно только отдельным решением после стабилизации core.

### 3.12 `StageANoRiskMetricsV2` больше не должен быть hot-path authority

Текущий Stage A exact breadth pass опирается на large no-risk metrics surface.

В новом engine:

- этот слой не должен быть главным driver hot path;
- если часть логики останется, она должна быть сильно сужена;
- он MAY выжить как debug/reference helper, но не как canonical breadth-first scorer.

### 3.13 Progress semantics наружу остаётся стабильной

Внутри engine могут появиться:

- row prefilter;
- combo prefilter;
- exact candidate frontier;

Но наружу для history/UI желательно сохранить стабильный vocabulary:

- `stage_a`
- `stage_b`
- finalizing

То есть internal sub-stages допустимы, но публичную persisted progress semantics ломать не нужно.

## 4. Current -> Target replacement matrix

### 4.1 Runtime anchor

Current:

- docs и reasoning всё ещё часто опираются на `06_backtest_compute.ipynb`.

Target:

- canonical experimental anchor становится `01_run_322_btcusdt_1h_artifact_probe.ipynb`.

### 4.2 Stage A shortlist

Current:

- [BacktestStageAShortlistBuilderV2.build_shortlist](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py)
  выполняет expensive exact-first breadth pass.

Target:

- Stage A превращается в staged narrowing pipeline:
  - single-row prefilter
  - combo proxy prefilter
  - exact evaluation only on retained candidates

### 4.3 Stage A no-risk breadth scoring

Current:

- `build_compact_trade_list_v2(...)` + `compute_no_risk_metrics_v2(...)`
  участвуют в широком Stage A hot path.

Target:

- large no-risk metrics object больше не является main breadth scorer;
- compact trade representation строится только после prefilter для retained exact candidates.

### 4.4 Trades in default launch path

Current:

- historical design still carries `top_trades_n` semantics and top-row trade materialization traces.

Target:

- launch path остаётся summary-only;
- full trades считаются только по on-demand variant detail request.

### 4.5 Ranking contract

Current:

- public/UI/runtime surface поддерживает `primary_metric` + `secondary_metric`.

Target:

- остаётся только `primary_metric`.

### 4.6 Warmup

Current:

- public request/UI/defaults surface всё ещё содержит `warmup_bars`.

Target:

- `warmup_bars` исчезает из public contract и вычисляется только internal-но.

### 4.7 `top_trades_n`

Current:

- поле всё ещё живёт в DTOs, defaults, UI и request normalization.

Target:

- поле удаляется полностью из active public/runtime contract.

### 4.8 Additional filters

Current:

- explicit constraint filters как first-class public input отсутствуют.

Target:

- baseline это сохраняет: redesign не добавляет их в первой волне.

## 5. Почему внедрение начинается с Milestone A

Этот roadmap намеренно начинается с documentation and contract reset, а не с code patching.

Причина:

- сначала надо зафиксировать новый canonical anchor и новый product/runtime vocabulary;
- только после этого безопасно удалять публичные поля, менять Stage A contract и выпускать prompts;
- иначе разные prompts начнут одновременно опираться на старый `06_...` notebook, старый
  `secondary_metric` surface и старую `top_trades_n` semantics.

То есть `Milestone A` здесь это foundation, а не “документация ради документации”.

## 6. Целевая архитектура

Итоговая схема должна выглядеть так:

```text
POST /backtests
  -> request normalization
  -> shared runtime planning
  -> artifact-backed signal loading
  -> row prefilter
  -> combo proxy prefilter
  -> retained exact candidates
  -> internal compact trade representation
  -> exact Stage B TP/SL search on hit-times
  -> summary-only top N persistence

POST /backtests/runs/{run_id}/variant-report
  -> selected winner only
  -> exact trade materialization on demand
  -> report/trades for one chosen variant

backtest-job-runner
  -> same shared planner
  -> same prefilter + exact pipeline
  -> same exact Stage B scorer
  -> persisted progress/history semantics unchanged
```

Ключевые правила:

1. Sync и job-runner используют один engine.
2. Prefilter никогда не является final scoring authority.
3. Exact Stage B scorer остаётся canonical result authority.
4. `compact trade representation` является internal payload, а не default user artifact.
5. Full trades не считаются массово при launch.
6. Публичная advanced surface после redesign становится проще, а не сложнее.

## 7. Milestones и EPICs

Ниже порядок внедрения, по которому потом можно делать последовательные prompts.

### Milestone A. Foundation: canonical redesign docs и vocabulary reset

Это первый milestone. Его задача: перенести source of truth на новый anchor notebook и зафиксировать
новую product/runtime vocabulary до начала code migration.

#### EPIC A1. Новый canonical redesign doc

Что делаем простыми словами:

- создаём отдельный canonical architecture doc для нового engine redesign;
- фиксируем там новый anchor notebook, internal compact trade representation и summary-only
  launch path.

Что должно быть реализовано:

- создать новый canonical doc для engine redesign;
- описать current -> target replacement matrix;
- зафиксировать решения по `primary_metric`, `secondary_metric`, `warmup_bars`,
  `top_trades_n`, trades-on-demand.

Документы:

- создать:
  - `docs/architecture/backtest/backtest-engine-vnext.md`

#### EPIC A2. Re-anchor runtime docs away from `06_backtest_compute.ipynb`

Что делаем простыми словами:

- перестаём ссылаться на `06_backtest_compute.ipynb` как на canonical implementation anchor;
- переносим active docs на `01_run_322...`.

Что должно быть реализовано:

- `backtest-runtime-kernels-v2.md` обновлён как current-state contract с explicit note, что
  redesign planning anchor moved to `01_run_322...`;
- docs с notebook-derived semantics не должны представлять `06_...` как новый basis для prompts.

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
  - `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
  - при необходимости: `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`

#### EPIC A3. Public-contract redesign vocabulary

Что делаем простыми словами:

- заранее фиксируем, какие поля остаются публичными, а какие уходят.

Что должно быть реализовано:

- docs явно говорят:
  - `primary_metric` остаётся
  - `secondary_metric` удаляется
  - `warmup_bars` удаляется
  - `top_trades_n` удаляется
  - trades only on demand

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`
  - `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
  - `docs/runbooks/backtest-job-runner.md`

### Milestone B. Public contract simplification

Это первый кодовый milestone. Он убирает лишние user-facing knobs, не меняя ещё core compute
semantics.

#### EPIC B1. Удалить `secondary_metric`

Что делаем простыми словами:

- ranking остаётся configurable, но только по одной метрике.

Что должно быть реализовано:

- request DTO больше не принимает `ranking.secondary_metric`;
- runtime defaults больше не публикуют `secondary_metric_default`;
- UI больше не рендерит `secondary_metric`;
- runtime ordering contract использует один `primary_metric` + stable tie-break.

Кодовые точки:

- обновить:
  - `apps/api/dto/backtests.py`
  - `apps/api/dto/backtest_runtime_defaults.py`
  - `apps/api/dto/backtest_runs.py`
  - `apps/api/dto/backtest_jobs.py`
  - `apps/api/routes/backtests.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
  - `apps/web/templates/backtests.html`
  - `apps/web/templates/backtest_run_summary.html`
  - `apps/web/dist/backtest_ui.js`
  - `apps/web/dist/backtest_runs_ui.js`
  - `apps/api/wiring/modules/backtest.py`

#### EPIC B2. Удалить `warmup_bars` из public input

Что делаем простыми словами:

- пользователь больше не задаёт warmup руками;
- runtime выводит его сам.

Что должно быть реализовано:

- `warmup_bars` исчезает из `POST /backtests` request surface;
- runtime использует derived internal warmup value;
- history/status/detail payload не обязаны публиковать `warmup_bars` наружу как user-facing field.

Кодовые точки:

- обновить:
  - `apps/api/dto/backtests.py`
  - `apps/api/dto/backtest_runtime_defaults.py`
  - `apps/api/dto/backtest_runs.py`
  - `apps/api/dto/backtest_jobs.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`
  - `apps/web/templates/backtests.html`
  - `apps/web/dist/backtest_ui.js`
  - `apps/web/dist/backtest_jobs_ui.js`
  - `apps/api/wiring/modules/backtest.py`

#### EPIC B3. Удалить `top_trades_n`

Что делаем простыми словами:

- trades больше не материализуются массово при launch;
- параметр `top_trades_n` становится ненужным.

Что должно быть реализовано:

- `top_trades_n` исчезает из request, defaults, UI и run normalization;
- launch/result pipeline не содержит массового trade materialization behavior;
- variant-report/detail path остаётся единственным местом для full trades materialization.

Кодовые точки:

- обновить:
  - `apps/api/dto/backtests.py`
  - `apps/api/dto/backtest_runtime_defaults.py`
  - `apps/api/dto/backtest_runs.py`
  - `apps/api/dto/backtest_jobs.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py`
  - `apps/web/templates/backtests.html`
  - `apps/web/dist/backtest_ui.js`
  - `apps/api/wiring/modules/backtest.py`

#### EPIC B4. Явно сохранить разрешённую advanced surface

Что делаем простыми словами:

- после удаления лишних knobs фиксируем, что именно остаётся допустимым пользователю.

Что должно быть реализовано:

- docs и runtime defaults согласованы по:
  - `direction_mode`
  - `sizing_mode`
  - `primary_metric`
  - execution settings
  - risk grid settings

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`

### Milestone C. Artifact dependency expansion for wider TP/SL grid

Новый engine зависит от более широкой canonical TP/SL grid. Эта работа выделяется отдельно,
но должна быть выполнена до полного rollout нового exact path.

#### EPIC C1. Расширить artifact TP/SL grid

Что делаем простыми словами:

- precompute должен публиковать более широкий набор TP/SL levels.

Что должно быть реализовано:

- canonical TP/SL grid расширен в precompute config;
- `hit_times/1m` публикуются для новой grid;
- old narrow grid не остаётся hidden production assumption.

Кодовые точки:

- обновить:
  - `configs/dev/backtest_artifacts.yaml`
  - `configs/test/backtest_artifacts.yaml`
  - `configs/prod/backtest_artifacts.yaml`
  - `src/trading/contexts/backtest/application/services/v2/hit_times_compute_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_precompute_coordinator.py`

#### EPIC C2. Publish/validation contract for wider grid

Что делаем простыми словами:

- publish pipeline и manifests должны валидировать новую grid как часть canonical artifact
  contract.

Что должно быть реализовано:

- manifest shape и validation contract описывают новую grid;
- perf/memory budget documented and tested.

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-precompute-runner-v2.md`
  - `docs/architecture/backtest/backtest-v2-benchmarks.md`

#### EPIC C3. Runtime loaders must stay grid-agnostic

Что делаем простыми словами:

- runtime не должен быть зашит под старые `5 x 5` TP/SL arrays.

Что должно быть реализовано:

- Stage B loaders и kernels работают от artifact manifest/arrays, а не от hardcoded grid size;
- tests cover wider grid shapes.

Кодовые точки:

- обновить:
  - `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`

### Milestone D. Universal prefilter before exact path

Это главный redesign milestone. Он заменяет current Stage A exact-first breadth pass на staged
prefilter pipeline.

#### EPIC D1. Single-row prefilter

Что делаем простыми словами:

- сначала дешёво оцениваем каждую signal row отдельно;
- только хороший поднабор row-ов проходит дальше.

Что должно быть реализовано:

- deterministic row-local prefilter for each indicator family;
- row ranking опирается на cheap proxy stats и signal matrices;
- first cut не зависит от mandatory `signal_features`.

Кодовые точки:

- активировать/адаптировать:
  - `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`

#### EPIC D2. Combo proxy prefilter

Что делаем простыми словами:

- после row prefilter быстро оцениваем combinations грубым proxy score;
- exact path видит только survivors.

Что должно быть реализовано:

- combo proxy layer добавлен как canonical Stage A narrowing step;
- retained frontier deterministic and stable;
- tie-break ordering explicit and reproducible.

Кодовые точки:

- активировать/адаптировать:
  - `src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`

#### EPIC D3. New Stage A planner budgets

Что делаем простыми словами:

- planner должен понимать не только raw cartesian size, но и retained frontier shape.

Что должно быть реализовано:

- `ExecutionProfile` и planner оперируют стадиями:
  - row prefilter
  - combo prefilter
  - exact retained candidates
- progress units и ETA weights можно рассчитать без изменения public stage vocabulary.

Кодовые точки:

- обновить:
  - `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`

### Milestone E. Internal exact representation and Stage B handoff

Этот milestone закрепляет internal exact payload и убирает старый Stage A hot path как breadth
authority.

#### EPIC E1. `compact trade representation` только для retained candidates

Что делаем простыми словами:

- compact trade representation строится не для всего grid, а только для exact survivors.

Что должно быть реализовано:

- current breadth pass over all variants removed from hot path;
- compact trade representation becomes hidden exact intermediate payload after prefilter;
- default launch path still does not materialize full user trades.

Кодовые точки:

- существенно переписать:
  - `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
  - `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`

#### EPIC E2. Stage B consumes the new retained frontier

Что делаем простыми словами:

- existing exact Stage B scorer остаётся authority, но получает новых survivors из redesign Stage A.

Что должно быть реализовано:

- fast monotone TP/SL kernel сохранён как canonical exact risk search;
- Stage B принимает retained exact candidates from new pipeline;
- final summary metrics строятся после exact winning cell.

Кодовые точки:

- обновить:
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
  - `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
  - `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`

#### EPIC E3. Reference-vs-fast self-check

Что делаем простыми словами:

- быстрый kernel должен уметь сверяться с медленным reference path на bounded subset.

Что должно быть реализовано:

- explicit slow reference evaluator for bounded subset;
- explicit fast evaluator;
- deterministic parity checks в test/perf/debug surface;
- self-check не живёт в default production hot path.

Кодовые точки:

- обновить или создать:
  - `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
  - `src/trading/contexts/backtest/application/services/v2/stage_b_golden_fixtures_v2.py`
  - `tests/perf_smoke/contexts/backtest/*`
  - `tests/unit/contexts/backtest/application/services/v2/*`

### Milestone F. Shared sync / job-runner cutover

Новый engine должен быть включён в один shared execution path для sync и background runs.

#### EPIC F1. Sync launch cutover

Что делаем простыми словами:

- `POST /backtests` начинает использовать новый prefilter-first exact pipeline.

Что должно быть реализовано:

- sync launch, summary response и persisted run creation используют новый engine path;
- no fallback to current Stage A exact-first breadth path.

Кодовые точки:

- обновить:
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
  - `apps/api/routes/backtests.py`

#### EPIC F2. Claimed worker cutover

Что делаем простыми словами:

- `backtest-job-runner` использует тот же redesigned engine, что и sync.

Что должно быть реализовано:

- worker path не diverges from sync path;
- persisted progress and snapshots continue to work;
- default launch/result contracts не расширяются.

Кодовые точки:

- обновить:
  - `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
  - `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
  - `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`

#### EPIC F3. Progress/ETA semantics stability

Что делаем простыми словами:

- внутренняя staged модель меняется, но наружу UI/history остаются понятными.

Что должно быть реализовано:

- public progress stages stay stable;
- internal sub-stages map cleanly into existing persisted progress contract;
- no new browser-visible confusion around stages.

Кодовые точки:

- обновить:
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`
  - `apps/api/dto/backtest_runs.py`
  - `apps/web/dist/backtest_runs_ui.js`
  - `apps/web/templates/backtest_run_summary.html`

### Milestone G. On-demand trades/detail only

Этот milestone закрепляет, что full trades не являются частью default launch path.

#### EPIC G1. Detail endpoint as the only place for trades materialization

Что делаем простыми словами:

- trades и report bodies считаются только для выбранного пользователем варианта.

Что должно быть реализовано:

- variant-report/detail path clearly owns full trade materialization;
- launch path and summary persistence remain summary-only.

Кодовые точки:

- обновить:
  - `apps/api/routes/backtests.py`
  - `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
  - `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
  - `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

#### EPIC G2. Remove historical top-row trades assumptions

Что делаем простыми словами:

- убираем из docs/runtime любые active assumptions, что trades materialize-ятся для top rows по
  умолчанию.

Что должно быть реализовано:

- active docs no longer describe top-row trade materialization in launch flow;
- historical notes MAY remain only as compatibility reference.

Документы:

- обновить:
  - `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
  - `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md` where wording must stay aligned

### Milestone H. Cleanup, tests, docs index, prompt-ready closure

Это финальный milestone. Он закрывает cleanup и делает план пригодным для prompt-by-prompt
исполнения.

#### EPIC H1. Remove stale public/runtime vocabulary

Что делаем простыми словами:

- удаляем остаточные mentions старых полей и старых assumptions.

Что должно быть реализовано:

- active code/docs/tests no longer refer to:
  - `secondary_metric`
  - `warmup_bars` as user-facing input
  - `top_trades_n`
- historical references clearly marked as historical.

#### EPIC H2. Deterministic tests and perf-smoke closure

Что делаем простыми словами:

- новый engine должен иметь test surface, которая ловит semantic drift и perf regressions.

Что должно быть реализовано:

- unit tests for row prefilter, combo prefilter, compact exact payload, Stage B fast kernel;
- perf smoke for representative bounded searches;
- reference-vs-fast parity checks;
- worker/sync integration tests for new contract.

Ожидаемые тестовые поверхности:

- `tests/unit/contexts/backtest/application/services/v2/*`
- `tests/unit/contexts/backtest/application/use_cases/*`
- `tests/unit/apps/api/*`
- `tests/perf_smoke/contexts/backtest/*`

#### EPIC H3. Docs index and roadmap closure

Что делаем простыми словами:

- делаем новый plan discoverable и готовим его как source for machine-readable prompts.

Что должно быть реализовано:

- docs index updated;
- roadmap file finalised;
- prompt generation может безопасно идти milestone-by-milestone без vocabulary drift.

## 8. Какие файлы, вероятнее всего, будут созданы

Ожидаемо создать:

- `docs/architecture/backtest/backtest-engine-vnext.md`
- дополнительные test/perf-smoke files for prefilter/self-check if existing coverage surface is insufficient

## 9. Какие файлы, вероятнее всего, будут существенно переписаны

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `apps/api/dto/backtests.py`
- `apps/api/dto/backtest_runtime_defaults.py`
- `apps/web/templates/backtests.html`
- `apps/web/dist/backtest_ui.js`

## 10. Какие документы обязательно нужно будет синхронизировать

- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/backtest/backtest-job-runner-v2.md`
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
- `docs/runbooks/backtest-job-runner.md`

## 11. Non-goals

Этот roadmap не вводит:

- новый публичный selector execution profile в `POST /backtests`;
- обязательный `signal_features` contract в первой волне redesign;
- новые user-facing constraint filters;
- возврат к legacy ClickHouse/runtime recompute path;
- разные compute engines для sync и worker;
- full trade materialization в launch path;
- сохранение `secondary_metric`, `warmup_bars`, `top_trades_n` как active public knobs.

## 12. Критерии успеха

Roadmap считается успешно реализованным, когда одновременно выполнено следующее:

1. active redesign docs ссылаются на `01_run_322...` как на canonical experimental anchor;
2. `secondary_metric`, `warmup_bars`, `top_trades_n` удалены из active public surface;
3. новый Stage A больше не является exact-first breadth scorer для всего grid;
4. prefilter before exact path работает как canonical runtime behavior;
5. exact Stage B scorer остаётся final authority;
6. trades materialize-ятся только on demand;
7. sync и job-runner используют один redesigned engine path;
8. perf smoke подтверждает заметный разрыв в лучшую сторону относительно current Stage A hot path;
9. новый roadmap можно безопасно разложить на последовательные prompts без конфликтующих
   трактовок.

