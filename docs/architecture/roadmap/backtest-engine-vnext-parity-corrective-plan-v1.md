---
title: Корректирующий план достижения notebook-parity для backtest engine vNext
version: 1
status: draft
owner: backtest
---

# Корректирующий план достижения notebook-parity для backtest engine vNext

Статус: proposed corrective roadmap after live `NR2` benchmark verification  
Дата фиксации: 2026-04-12  
Область: `backtest`, `apps/api`, `apps/worker`, planner/runtime kernels, persisted shortlist, benchmark/perf-smoke

## 1. Зачем нужен этот документ

Документ [План достижения notebook-parity производительности для backtest engine vNext](./backtest-engine-vnext-notebook-parity-plan-v1.md)
зафиксировал целевую performance contract.

После выполнения milestone/prompts `24-36` живой benchmark для canonical `NR2` shape показал, что
repository ещё не достиг notebook-parity:

- canonical backend shape для `run_id=f7d2c378-bca2-46fe-b5a6-47062fb75140` всё ещё идёт по
  тяжёлому пути и не укладывается в target runtime;
- worker/background path всё ещё содержит локальные дефекты;
- Stage A всё ещё реализован слишком общим runtime shape по сравнению с notebook anchor.

Этот документ нужен для узкой corrective program. Он не открывает новый redesign и не меняет
базовую архитектурную цель. Он фиксирует только те доработки, которые теперь подтверждены живой
проверкой как необходимые для реального сближения backend с notebook.

## 2. Зафиксированные факты по живому benchmark

Canonical no-risk notebook anchor:

- `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`

Canonical backend request shape:

- `run_id=f7d2c378-bca2-46fe-b5a6-47062fb75140`

Зафиксированные измерения на `macstudio` при одинаковом thread budget:

- notebook anchor при `NUMBA_NUM_THREADS=4`: около `8.01s`, `1` Python process, peak RSS около
  `1.37 GB`;
- текущий backend rerun того же request shape:
  - route: `background_auto`;
  - failure after about `85.25s`;
  - Stage A alone: about `83.54s`;
  - failure in Stage B persistence because `profit_factor=Infinity` was serialized into JSON.

Следствие:

- текущий backend остаётся примерно `10x` медленнее notebook even before successful completion;
- текущий backend still does not satisfy the accepted `NR2` parity contract;
- `perf_smoke` contract and docs were not enough to prove live parity on the benchmark host.

## 3. Что считаем подтверждёнными root causes

Ниже перечислены только причины, подтверждённые живым benchmark и кодом.

### 3.1 Sync launch budgets всё ещё не согласованы с no-risk workload

Current sync launch still routes the canonical `NR2` shape to `background_auto`, because planner
launch-budget checks still use broad raw workload signals:

- raw `stage_a_variants_total`;
- broad `stage_b_variants_total`;
- legacy-style memory estimate.

Это видно в planner/runtime selection surface:

- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`

Проблема не в том, что launch budget “маленький”. Проблема в том, что canonical `NR2` no-risk
shape уже не соответствует старой budget model, но launch routing всё ещё живёт по ней.

### 3.2 Stage A всё ещё итерирует raw grid вместо narrowed frontier

Current Stage A does row prefilter, but then still walks the raw Stage A grid and only afterwards
filters chunk variants against the retained frontier.

Это видно в:

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`

Для `f7d2` это означает:

- notebook exact path реально работает по narrowed pools (`142 x 142 = 20164` combos);
- backend Stage A всё ещё проходит через raw `345744` Stage A variants.

Это уже само по себе создаёт большой breadth/orchestration gap.

### 3.3 GenericRowScorerV2 остаётся в parity hot path

Current row prefilter for Stage A still uses the universal scored object path:

- typed row payload creation;
- optional `signal_features` resolution;
- bucketed `GenericRowScorePayloadV2`;
- sorted deterministic scorer output.

Это реализовано в:

- `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`

Этот scorer полезен как universal framework и audit surface, но он слишком тяжёлый для canonical
parity hot path. Notebook anchor uses a much cheaper matrix-first proxy ranking path.

### 3.4 Worker path не переносит Stage A no-risk exact result

Persisted Stage A shortlist currently keeps only `stage_a_indexes`, without the exact no-risk
payload needed to let worker resume and finalize no-risk runs cheaply.

Это реализовано в:

- `src/trading/contexts/backtest/domain/entities/backtest_job_results.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

Следствие:

- sync and worker no-risk paths are not behaviorally equivalent;
- worker no-risk finalization cannot faithfully reuse the exact Stage A result contract.

### 3.5 Stage B persistence still serializes non-finite metrics

Current metrics contract still allows values like `Infinity` to flow into persisted summary JSON.

Это видно в:

- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- persistence adapters under
  `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/`

Это локальный correctness bug. Он не объясняет весь `10x` gap, но он currently blocks successful
benchmark completion and therefore blocks honest closure.

### 3.6 Что уже не считаем текущей root cause

Ниже перечислено то, что больше не должно рассматриваться как основной объясняющий фактор для
canonical `NR2` gap:

- проблема больше не сводится к “надо просто поднять `NUMBA_NUM_THREADS`”;
- current live `NR2` rerun не показал process fan-out как главный источник slowdown: measured
  worker tree stayed effectively single-process during the failing rerun;
- current retained frontier уже не должен хранить full `final_signal_row` как основной active
  contract, поэтому прежняя версия этой проблемы больше не является главным объяснением `10x`
  gap;
- raw numeric kernels themselves уже достаточно близки к notebook shape; основной remaining gap
  сейчас находится в launch routing, breadth iteration shape, Python-heavy row prefilter, and
  worker resume contract.

Это важно зафиксировать, чтобы corrective program не ушла снова в ложные направления вроде
очередного thread tuning или возврата process fan-out как default strategy.

## 4. Что считаем достаточным scope corrective program

Этот corrective roadmap intentionally ограничен пятью рабочими направлениями:

1. `Stage A narrowed-frontier iteration`
2. `removal of GenericRowScorer from parity hot path`
3. `sync launch budget alignment with no-risk workload`
4. `persistence of Stage A no-risk exact result through worker path`
5. `finite-metric JSON sanitization`

Если эти пять направлений не будут закрыты, canonical `NR2` backend shape останется либо
архитектурно тяжелее notebook, либо формально недостоверной из-за broken persistence/runtime
contract.

## 5. Важное ограничение про “100% уверенность”

В этом документе `100% уверенность` трактуется только в одной честной форме:

- для каждого corrective milestone есть benchmark gate;
- milestone не считается завершённым, если gate не пройден;
- итоговый corrective program не считается завершённым, если canonical live benchmark on the
  benchmark host всё ещё не проходит target gates.

То есть документ не обещает результат “по идее”. Он фиксирует программу изменений, которая должна
быть доказана live benchmark execution.

## 6. Целевое runtime состояние после corrective program

### 6.1 Для canonical `NR2`

Canonical `NR2` run должен иметь такую форму:

- request stays on `sync_inline`;
- planner classifies it using no-risk-aware launch budgeting;
- Stage A works on narrowed retained frontier, not on raw grid iterator;
- row prefilter hot path does not use `GenericRowScorerV2`;
- final no-risk ranking is resolved directly from Stage A exact results;
- `stage_b_execution_mode = bypassed_no_risk`;
- no worker/process fan-out is used;
- persisted summary rows contain only finite JSON-safe metrics.

### 6.2 Для worker parity

Если тот же no-risk shape всё же исполняется как background job:

- worker receives persisted Stage A exact no-risk result in a compact deterministic form;
- worker no-risk finalization stays equivalent to sync finalization;
- no expensive generic replay is required just because the run resumed from persisted state.

## 7. Пошаговый план реализации

## EPIC C0. Live benchmark authority hardening

### Цель

Перед фиксом runtime необходимо закрепить live benchmark truth как blocking authority, а не только
synthetic corpus/tests.

### Что делаем

- добавить explicit live benchmark runner/harness для canonical `NR2` and `RG-TTR` host runs;
- явно хранить:
  - wall clock;
  - peak RSS;
  - `max_python_processes_seen`;
  - `stage_b_execution_mode`;
  - `exact_replay_count`;
  - `numba_threads_used`;
- сделать live benchmark output обязательной частью corrective closure.

### Основные файлы

- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- `src/trading/contexts/backtest/application/services/v2/notebook_parity_benchmark_corpus_v2.py`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`

### Acceptance gate

- repository must distinguish `synthetic contract validation` from `live host measurement`;
- corrective closure cannot pass without explicit live benchmark capture for canonical `NR2`.

## EPIC C1. Sync launch budget alignment with no-risk workload

### Цель

Сделать так, чтобы canonical `NR2` shape больше не уходил в `background_auto` purely because the
planner still evaluates it with broad raw-grid launch budgets.

### Что заменяем

Current launch-budget decision logic for requested sync profile in:

- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`

### Что делаем

- ввести explicit no-risk-aware sync launch budgeting;
- requested sync profile must use narrowed no-risk workload evidence rather than raw
  `stage_a_variants_total` as the blocking budget signal;
- memory estimate for canonical no-risk class must stop using legacy broad tensor estimate when
  runtime shape no longer matches it;
- sync planner decision must align with the actual no-risk terminal path.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `configs/dev/backtest.yaml`
- `configs/test/backtest.yaml`
- `configs/prod/backtest.yaml`

### Acceptance gate

- canonical `NR2` benchmark request must stay `sync_inline`;
- no-risk sync launch must no longer be rejected by raw-grid launch-budget math.

## EPIC C2. Stage A narrowed-frontier iteration

### Цель

Убрать главный breadth/orchestration gap между backend and notebook by making Stage A iterate the
retained frontier itself instead of iterating the full raw grid and filtering later.

### Что заменяем

Current raw-grid batch loop in:

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`

### Что делаем

- after row prefilter, materialize deterministic narrowed row pools explicitly;
- enumerate Stage A combo chunks from retained local row pools rather than from
  `grid_context.iter_stage_a_variants()` over the full raw grid;
- keep chunk order deterministic and stable;
- preserve same shortlist semantics and tie-breaks.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py`

### Acceptance gate

- Stage A runtime shape for canonical `NR2` must operate on narrowed frontier cardinality close to
  notebook shape, not on the full raw `stage_a_variants_total`;
- Stage A live timing for canonical `NR2` must drop materially before any thread tuning.

## EPIC C3. Remove GenericRowScorer from parity hot path

### Цель

Убрать universal row scorer from the canonical parity hot path and replace it with a numeric,
matrix-first row prefilter path closer to the notebook implementation.

### Что заменяем

Current row prefilter dependency on:

- `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`

### Что делаем

- оставить `GenericRowScorerV2` как optional audit/debug/universal path;
- для canonical parity Stage A path использовать:
  - nonzero count;
  - vectorized/matrix proxy score;
  - deterministic top-fraction retain;
- optional `signal_features` must not be required for parity hot path.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`

### Acceptance gate

- canonical `NR2` Stage A row prefilter must no longer allocate/score universal row objects on
  the main hot path;
- live `NR2` benchmark must show single-process numeric prefilter behavior close to notebook.

## EPIC C4. Persist Stage A no-risk exact result through worker path

### Цель

Сделать sync and worker no-risk finalization contract-equivalent.

### Что заменяем

Current persisted shortlist shape, which stores only ordered `stage_a_indexes`.

### Что делаем

- persisted Stage A shortlist for no-risk class must include enough compact exact result to let
  worker finalize without recomputing the generic path;
- payload must stay compact and deterministic;
- contract must remain additive and backward-readable where needed;
- worker resume path must reuse that exact no-risk result directly.

### Основные файлы

- `src/trading/contexts/backtest/domain/entities/backtest_job_results.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- corresponding migration / storage schema files if contract expansion requires them

### Acceptance gate

- worker no-risk finalization must not depend on rebuilding generic Stage A/Stage B data that was
  already computed;
- sync and worker no-risk benchmark outputs must match for canonical `NR2`.

## EPIC C5. Finite-metric JSON sanitization

### Цель

Убрать persistence failures on non-finite metrics and make summary-only persisted rows always
JSON-safe.

### Что заменяем

Current metrics serialization path that allows `Infinity` to flow into persisted JSON.

### Что делаем

- define one explicit finite-metric normalization rule for persisted summary metrics;
- sanitize `profit_factor`, `return_over_max_drawdown`, and any other non-finite values before
  JSON serialization;
- keep ranking semantics explicit and test-covered.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py`

### Acceptance gate

- canonical `NR2` backend benchmark must complete successfully;
- persisted rows and job snapshots must never fail on `Infinity` / `NaN`.

## EPIC C6. Final live benchmark closure

### Цель

Закрыть corrective program не по docs, а по фактическому benchmark result on the benchmark host.

### Что делаем

- rerun canonical `NR2` backend vs notebook on the same host and same thread budget;
- rerun canonical `RG-TTR` backend vs notebook on the same host and same thread budget;
- publish the final live benchmark capture;
- update runtime docs only if live gates actually pass.

### Основные файлы

- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/architecture/backtest/backtest-engine-vnext.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- live benchmark capture helpers under `tests/perf_smoke/contexts/backtest/`

### Acceptance gate

- canonical `NR2` backend runtime `<= 1.18x` notebook runtime on equal thread budget;
- canonical `NR2` peak RSS `<= 1.35x` notebook peak RSS on equal thread budget;
- canonical `NR2` stays `sync_inline`, `single-process`, `bypassed_no_risk`;
- canonical `RG-TTR` keeps in-process default and passes its own parity gates;
- `RG-ALT` remains functionally correct and must not regress by more than `10%`.

## 8. Что intentionally не входит в corrective scope

- не открываем новый product redesign;
- не возвращаем старый `exact_parallel` Stage B process fan-out как default answer;
- не добавляем новые user-facing filters;
- не добавляем новые публичные knobs вместо фикса hot path;
- не считаем, что “достаточно повысить thread count”.

## 9. Порядок реализации

Порядок должен быть только таким:

1. `C0` live benchmark authority
2. `C1` sync launch budget alignment
3. `C2` narrowed-frontier Stage A iteration
4. `C3` remove GenericRowScorer from parity hot path
5. `C4` persist Stage A no-risk exact result through worker path
6. `C5` finite-metric JSON sanitization
7. `C6` final live benchmark closure

Причина:

- без `C0` нельзя честно доказывать результат;
- без `C1` canonical `NR2` всё ещё будет уходить в worker path;
- без `C2` и `C3` backend Stage A останется слишком широкой и слишком Python-heavy;
- без `C4` worker path останется контрактно неверной;
- без `C5` benchmark может продолжать формально падать даже после ускорения;
- только после этого имеет смысл делать closure.

## 10. Документы, которые должны быть синхронизированы

- `docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md`
- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/architecture/backtest/backtest-engine-vnext.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-job-runner-v2.md`
- `docs/architecture/backtest/backtest-runs-history-v2.md`

## 11. Итоговое правило этого документа

Corrective program считается завершённой только тогда, когда:

- canonical live `NR2` benchmark реально проходит на benchmark host;
- canonical live `RG-TTR` benchmark реально проходит на benchmark host;
- docs описывают именно live-delivered runtime truth;
- backend speed claim опирается на измерение, а не на synthetic corpus/tests only.
