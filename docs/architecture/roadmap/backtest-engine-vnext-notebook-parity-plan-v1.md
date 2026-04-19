---
title: План достижения notebook-parity производительности для backtest engine vNext
version: 1
status: draft
owner: backtest
---

# План достижения notebook-parity производительности для backtest engine vNext

Статус: proposed benchmark-gated implementation roadmap; historical performance roadmap after `v2` umbrella cutover
Дата фиксации: 2026-04-12  
Область: `backtest`, `apps/api`, `apps/worker`, runtime kernels, artifact-backed scoring, benchmark/perf-smoke

Исполнительный статус:

- этот документ сохраняется как historical performance-rationale and benchmark-shape reference;
- remaining execution scope и closure authority перенесены в
  `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`;
- benchmark gates, если они остаются актуальными, должны читаться через `v2`, а не как
  самостоятельная параллельная программа работ.

## 1. Зачем нужен этот документ

Предыдущий roadmap [План переустройства backtest engine vNext](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md)
зафиксировал общий redesign engine.

Этот документ уже не про общий redesign, а про более узкую и жёсткую цель:

- приблизить production backtest engine к notebook speed на одном и том же host;
- убрать текущие архитектурные причины, из-за которых backend остаётся сильно медленнее notebook;
- сделать это без возврата к legacy runtime и без раздвоения sync/background engines;
- зафиксировать именно такой план работ, который не опирается на надежду, а опирается на обязательные benchmark gates.

Этот документ нужен потому, что текущая ситуация уже измерена и хорошо локализована:

- backend run [f7d2c378-bca2-46fe-b5a6-47062fb75140](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb) завершился примерно за `181.3s`;
- run-specific notebook [02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb) на том же `macstudio` завершился:
  - за `5.63s` при `12` numba threads;
  - за `7.54s` при `4` numba threads;
- значит backend сейчас проигрывает не из-за “мало ядер”, а из-за неверной computational shape runtime.

## 2. Важное ограничение про “100% уверенность”

Ни один архитектурный документ не может честно обещать “мы гарантированно получим 85-90% от notebook speed” до того, как изменения будут реализованы и измерены.

Единственная честная форма “100% уверенности” в этом контексте такая:

- каждый milestone имеет обязательный benchmark gate;
- milestone не считается завершённым, если gate не выполнен;
- следующий milestone не начинается, если предыдущий не прошёл gate;
- итоговый rollout не считается успешным, если canonical benchmark classes не достигли целевой скорости.

То есть этот документ intentionally фиксирует не “набор идей”, а benchmark-driven stop-the-line program.

## 3. Что известно с высокой уверенностью уже сейчас

### 3.1 Текущий backend проигрывает notebook не из-за сырых данных

Run-specific notebook [02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb)
использует те же published artifacts:

- `prices/15m`
- `prices/1m`
- `mappings/15m`
- `signals/15m/ma.dema`
- `signals/15m/ma.hma`

И при этом завершает тот же search shape на том же host за секунды, а не за минуты.

Следовательно:

- bottleneck не в `npy` / `mmap`;
- bottleneck не в самом объёме grid как таковом;
- bottleneck в runtime orchestration, retained payload contract и Stage B flow.

### 3.2 Для `f7d2...` current backend shape всё ещё архитектурно неверна

Для этого run:

- `stage_a_variants_total = 345744`
- `preselect_used = 20000`
- `risk_total = 1`
- `best_tp_pct = NULL`, `best_sl_pct = NULL`

Это означает:

- run является no-risk class;
- current backend всё равно тянет generic Stage B machinery;
- Stage A сохраняет слишком тяжёлый retained frontier;
- downstream exact path остаётся тяжелее, чем должен быть для no-risk default search.

### 3.3 Process-parallel Stage B сейчас ухудшает memory shape

Current `exact_parallel` profile в prod включает:

- `stage_b_workers > 1`
- `parallel_stage_b_enabled = true`

и runtime реально поднимает `ProcessPoolExecutor` в
[artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py).

Это создаёт наблюдаемую в production картину:

- `4-5` Python processes;
- большой RSS в каждом процессе;
- поведение, которое резко отличается от notebook-style single-process multi-thread kernel path.

### 3.4 Current retained frontier слишком тяжёлая

Current Stage A хранит `_RetainedExactCandidateV2.final_signal_row` в
[stage_a_shortlist_builder_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py).

Для длинных runs это означает:

- гигантские `int8` buffers per retained survivor;
- дорогой `np.stack(...)` перед exact scoring;
- лишний retained payload hand-off в Stage B;
- memory pressure, которой в notebook вообще нет.

### 3.5 Current no-risk class всё ещё не использует правильный terminal path

Для no-risk runs current engine всё ещё не заканчивает computation там, где должен.

Правильная notebook-like shape для no-risk класса такая:

1. row prefilter
2. combo proxy prefilter
3. exact no-risk trade-list-first scoring
4. final top-K

Current backend вместо этого всё ещё протаскивает shortlisted rows через generic downstream machinery.

## 4. Что считаем canonical benchmark anchors

### 4.1 Canonical no-risk anchor

- [02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb)

Этот notebook является canonical reference для:

- two-indicator no-risk class;
- `15m` signal timeline + `1m` execution timeline;
- row prefilter;
- combo proxy prefilter;
- trade-list-first exact no-risk scoring;
- single-process multi-thread exact path.

### 4.2 Canonical risk-grid anchor

- [01_run_322_btcusdt_1h_artifact_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb)

Этот notebook является canonical reference для:

- wider `N-indicator` search shape;
- hit-time based risk search;
- fast monotone TP/SL kernel;
- reference-vs-fast self-check;
- bounded exact evaluation after prefilter.

### 4.3 Historical reference only

- [06_backtest_compute.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/06_backtest_compute.ipynb)

Этот notebook больше не должен использоваться как основной implementation anchor.

## 5. Целевая performance contract

### 5.1 Главный принцип сравнения

Все сравнения backend vs notebook делаются:

- на одном и том же host;
- на одном и том же artifact slot;
- на одном и том же request shape;
- при одинаковом thread budget.

Сравнения вида:

- notebook on `12` threads vs backend on `4` threads

не считаются валидными acceptance benchmarks.

### 5.2 Canonical target classes

#### Class NR2

No-risk two-indicator class:

- two indicators;
- no-risk;
- `primary_metric = total_return_pct`;
- exact no-risk final ranking;
- canonical anchor: [02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb)

#### Class RG-TTR

Risk-grid class with total-return ranking:

- `N-indicator` staged search;
- artifact-backed TP/SL grid;
- `primary_metric = total_return_pct`;
- canonical anchor: [01_run_322_btcusdt_1h_artifact_probe.ipynb](/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb)

#### Class RG-ALT

Risk-grid class with alternative `primary_metric`:

- `max_drawdown_pct`
- `return_over_max_drawdown`
- `profit_factor`
- `sharpe_trades`
- `win_rate_pct`

Этот класс остаётся поддерживаемым функционально, но notebook-parity target first wave фиксируется
только для `NR2` и `RG-TTR`.

### 5.3 Performance gates

#### Mandatory gate for `NR2`

Для canonical no-risk benchmark:

- backend end-to-end runtime MUST be `<= 1.18x` notebook runtime on the same thread budget;
- backend peak RSS MUST be `<= 1.35x` notebook peak RSS on the same thread budget;
- backend MUST stay single-process by default for this class.

#### Mandatory gate for `RG-TTR`

Для canonical risk-grid total-return benchmark:

- backend end-to-end runtime MUST be `<= 1.18x` notebook runtime on the same thread budget;
- backend MUST keep Stage B in-process by default for benchmark-sized workloads;
- backend MUST avoid per-worker RSS explosion from process fan-out in the default path.

#### Functional gate for `RG-ALT`

Для alternative ranking metrics:

- correctness MUST remain intact;
- runtime MUST NOT regress against the pre-cutover baseline by more than `10%` on the canonical benchmark host;
- notebook-parity for this class is explicitly not part of the first acceptance bar.

## 6. Зафиксированные архитектурные решения этого performance plan

### 6.1 No-risk class должна финализироваться в Stage A

Если run является no-risk class, engine не должен проходить через тяжёлый generic Stage B path.

Правило:

- row prefilter
- combo proxy prefilter
- exact no-risk trade-list-first batch scoring
- final top-K and summary rows

Это и есть terminal path для no-risk class.

Следствие:

- generic Stage B orchestration MUST be bypassed for no-risk class;
- no-risk ranking MUST be resolved из exact Stage A metrics;
- process fan-out для no-risk class запрещён по умолчанию.

### 6.2 Retained frontier больше не хранит full `final_signal_row`

Current retained frontier contract должен быть заменён.

Запрещается:

- хранить `final_signal_row` per retained survivor;
- stack-ить тысячи full signal rows для exact retained pass;
- переносить full signal rows дальше как internal payload.

Разрешённый target:

- streaming exact scoring per retained chunk;
- retain only minimal shortlist state;
- для risk path retain only compact trade payload for shortlisted finalists.

### 6.3 Stage A exact work должен быть streaming, а не two-phase retained replay

Current shape:

1. build retained frontier
2. retain full signal rows
3. replay retained exact scoring later

Target shape:

1. row prefilter
2. combo proxy chunk
3. exact batch immediately for retained chunk
4. merge final/top heaps immediately

То есть Stage A exact work должен стать streaming pipeline, а не deferred retained replay.

### 6.4 Retained payload для risk path должен быть compact-trade-first

Если downstream risk path требует retained payload, он должен состоять из:

- compact trade arrays;
- exact no-risk metrics if already computed;
- minimal variant addressing data;

но не из full `final_signal_row`.

### 6.5 Retained payload не должен автоматически выключать cheap Stage B path

Current contract, где retained payload сам по себе включает `force exact Stage B`, должен быть изменён.

Правило:

- retained payload MAY accelerate exact replay;
- retained payload MUST NOT disable cheap total-return grid ranking by itself;
- fast Stage B grid search must remain available whenever ranking semantics allow it.

### 6.6 Exact replay в Stage B должен быть отложен до finalist scope

Current backend тратит слишком много exact work на shortlist breadth.

Target:

- fast monotone grid kernel ranks shortlisted candidates;
- exact replay runs only for finalist scope;
- summary-only top rows are built from final exact winners, not from full exact breadth.

### 6.7 Default Stage B path должен быть single-process batched kernel path

По умолчанию Stage B должен работать так же по shape, как notebook:

- один Python process;
- batched kernels;
- multi-thread Numba where appropriate;
- no process fan-out for moderate canonical workloads.

`ProcessPoolExecutor` допускается только как explicit fallback path после benchmark proof.

### 6.8 `stage_a_workers` и `max_numba_threads` тюнятся только после architecture cutover

Правило:

- сначала меняется computational shape;
- потом включается tuning thread budgets;
- config-only tuning не считается решением architecture gap.

## 7. Current -> Target replacement matrix

### 7.1 Stage A retained frontier

Current:

- `_RetainedExactCandidateV2.final_signal_row`
- deferred retained replay

Target:

- no full signal-row retention
- streaming exact scoring per retained chunk
- shortlist heap merges exact metrics directly

### 7.2 No-risk class

Current:

- no-risk runs still flow through generic downstream runtime layers

Target:

- no-risk runs terminate in Stage A
- Stage B is bypassed

### 7.3 Stage B fast path

Current:

- retained payload forces exact Stage B for retained candidates

Target:

- retained payload can coexist with cheap Stage B total-return path
- exact replay is finalist-only

### 7.4 Stage B parallelism

Current:

- default `exact_parallel` profile fans out spawned Python workers

Target:

- in-process batched Stage B is default
- process fan-out is fallback-only and benchmark-gated

### 7.5 Memory shape

Current:

- large retained signal buffers
- scorer snapshot duplication across spawned workers

Target:

- bounded compact payload only
- no worker RSS explosion in the default path

## 8. Milestones

### Milestone A. Benchmark Contract and Perf Harness

#### A1. Зафиксировать canonical benchmark protocol

Что делаем:

- оформляем canonical benchmark matrix для `NR2`, `RG-TTR`, `RG-ALT`;
- фиксируем host, slot, thread-budget rules;
- фиксируем exact acceptance gates.

Что создаём/обновляем:

- [backtest-v2-benchmarks.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-v2-benchmarks.md)
- [backtest-engine-vnext.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-engine-vnext.md)
- этот roadmap

Что реализуем в коде:

- benchmark/perf-smoke runner for canonical runtime shapes;
- repeatable runner for backend vs notebook parity comparison on `macstudio`-class host.

Ожидаемые file touches:

- `/Users/daniildegtyarev/Projects/roehub.com/tests/perf/` new benchmark runner files
- `/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`
- `/Users/daniildegtyarev/Projects/roehub.com/tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`

Acceptance gate:

- canonical benchmark classes reproducibly runnable;
- thread-budget normalization written down and executable;
- current baseline numbers stored as explicit comparison points.

### Milestone B. Stage A Streaming Frontier Cutover

#### B1. Убрать full `final_signal_row` из retained frontier

Что делаем:

- удаляем retained contract, который хранит full signal rows;
- retained frontier becomes proxy-only addressing plus minimal metadata.

Что меняем:

- [stage_a_shortlist_builder_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py)

Связанные файлы:

- [signal_aggregator_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py)
- [trade_compactor_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py)

Acceptance gate:

- no retained candidate object stores full `final_signal_row`;
- Stage A peak memory on `NR2` drops materially and measurably versus current baseline;
- correctness fixtures unchanged.

#### B2. Сделать Stage A exact scoring streaming per chunk

Что делаем:

- exact no-risk work runs immediately after combo proxy chunk selection;
- final heap merges exact metrics directly;
- remove deferred retained replay over stacked signal rows.

Что меняем:

- [stage_a_shortlist_builder_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py)
- [trade_compactor_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py)

Acceptance gate:

- Stage A no longer does `np.stack(...)` over retained signal-row frontier for exact evaluation;
- Stage A benchmark for `NR2` is materially closer to notebook baseline before touching Stage B.

#### B3. Перевести Stage A frontier path на batched kernels everywhere, где это нужно

Что делаем:

- audit and remove remaining Python-heavy frontier hot spots;
- add Numba-parallel consensus/aggregation and batch merge helpers where benchmark proves value.

Что меняем:

- [signal_aggregator_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py)
- [stage_a_shortlist_builder_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py)
- [trade_compactor_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py)

Acceptance gate:

- Stage A CPU utilization becomes notebook-like on canonical benchmark host;
- `NR2` backend Stage A runtime is close enough that Stage B becomes the next visible bottleneck.

### Milestone C. No-Risk Direct Finalization

#### C1. Ввести explicit no-risk terminal path

Что делаем:

- no-risk class finalizes after Stage A exact scoring;
- generic Stage B is not entered for this class.

Что меняем:

- [artifact_runtime_plan_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py)
- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)
- [artifact_backed_stage_b_scorer_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py)

Связанные use cases:

- [run_backtest.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest.py)
- [run_backtest_job_runner_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py)

Acceptance gate:

- `NR2` backend does not spawn Stage B workers;
- `NR2` backend end-to-end runtime meets `<= 1.18x` notebook target on the same thread budget;
- `NR2` backend peak RSS meets `<= 1.35x` notebook target.

#### C2. Сохранить поддержку всех no-risk primary metrics

Что делаем:

- direct no-risk finalization still supports:
  - `total_return_pct`
  - `max_drawdown_pct`
  - `return_over_max_drawdown`
  - `profit_factor`
  - `sharpe_trades`
  - `win_rate_pct`

Что меняем:

- [trade_compactor_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py)
- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)

Acceptance gate:

- no-risk alternative `primary_metric` values remain functionally supported;
- no-risk alt metrics do not re-enable generic Stage B by accident.

### Milestone D. Risk-Grid Stage B Notebook-Parity Cutover

#### D1. Retained payload becomes compact-trade-first only

Что делаем:

- risk path retains compact-trade payload only for shortlisted base variants;
- retained payload MAY include already computed no-risk metrics;
- retained payload MUST NOT include full `final_signal_row`.

Что меняем:

- [trade_compactor_kernel.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py)
- [stage_a_shortlist_builder_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py)
- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)

Acceptance gate:

- retained payload memory shape is compact-trade based;
- risk-grid benchmark no longer carries Stage A retained signal-row memory overhead.

#### D2. Fast Stage B total-return path must stay enabled with retained payload

Что делаем:

- retained payload no longer forces exact Stage B for `primary_metric = total_return_pct`;
- fast monotone TP/SL search remains canonical ranking path for breadth scoring.

Что меняем:

- [artifact_backed_stage_b_scorer_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py)
- [risk_exit_kernel_1m.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py)

Acceptance gate:

- retained payload does not disable cheap Stage B total-return ranking;
- exact replay is not performed for every shortlisted candidate anymore.

#### D3. Exact replay becomes finalist-only

Что делаем:

- fast grid ranking for shortlisted candidates;
- exact replay and expensive metric finalization only for finalist scope.

Что меняем:

- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)
- [artifact_backed_stage_b_scorer_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py)

Acceptance gate:

- Stage B breadth cost drops to near-notebook shape for `RG-TTR`;
- exact replay count is bounded by finalist scope and observable in tests/metrics.

### Milestone E. Single-Process Stage B Default and Parallel Fallback Policy

#### E1. Сделать in-process batched Stage B default

Что делаем:

- default Stage B path for benchmark-sized workloads becomes single-process batched kernel path;
- `ProcessPoolExecutor` is no longer default behavior for `exact_parallel`.

Что меняем:

- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)
- [execution_profile_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py)
- [configs/prod/backtest.yaml](/Users/daniildegtyarev/Projects/roehub.com/configs/prod/backtest.yaml)

Acceptance gate:

- canonical `RG-TTR` benchmark runs without default process fan-out;
- benchmark host no longer shows `4-5` large Python worker processes by default.

#### E2. Оставить process fan-out только как explicit fallback path

Что делаем:

- spawned Stage B workers are allowed only above explicit workload thresholds;
- fallback path remains opt-in and benchmark-proven.

Что меняем:

- [artifact_runtime_core_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py)
- [backtest-runtime-kernels-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- [backtest-engine-vnext.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-engine-vnext.md)

Acceptance gate:

- fallback path documented and bounded;
- default benchmark classes do not rely on process fan-out.

### Milestone F. Thread-Budget Tuning After Architecture Cutover

#### F1. Re-tune `stage_a_workers` and `max_numba_threads`

Что делаем:

- tune thread budgets only after Milestones B-E are complete;
- align sync/job runtime thread budget with proven benchmark settings.

Что меняем:

- [numba_runtime_v1.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/numba_runtime_v1.py)
- [execution_profile_v2.py](/Users/daniildegtyarev/Projects/roehub.com/src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py)
- [configs/prod/backtest.yaml](/Users/daniildegtyarev/Projects/roehub.com/configs/prod/backtest.yaml)

Acceptance gate:

- tuning improves or preserves benchmark targets;
- no tuning change is allowed to hide an unresolved architectural bottleneck.

### Milestone G. Tests, Docs, and Rollout Closure

#### G1. Perf-smoke and regression harness

Что делаем:

- add canonical perf-smoke for `NR2` and `RG-TTR`;
- add memory-shape assertions where feasible;
- add debug traces for exact replay counts and Stage B execution mode.

Ожидаемые file touches:

- `/Users/daniildegtyarev/Projects/roehub.com/tests/perf/`
- `/Users/daniildegtyarev/Projects/roehub.com/tests/unit/contexts/backtest/`

Acceptance gate:

- perf-smoke can detect regressions against notebook-parity target;
- rollout no longer depends on manual guesswork.

#### G2. Docs alignment

Что обновляем:

- [backtest-engine-vnext.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-engine-vnext.md)
- [backtest-runtime-kernels-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runtime-kernels-v2.md)
- [backtest-job-runner-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)
- [backtest-runs-history-v2.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-runs-history-v2.md)
- [backtest-v2-benchmarks.md](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-v2-benchmarks.md)

Acceptance gate:

- docs describe the new default compute shape accurately;
- docs no longer imply that generic Stage B exact replay is required for no-risk class.

## 9. Что этот план сохраняет, а что нет

### Сохраняем

- один shared sync/background runtime;
- artifact-backed engine;
- row prefilter;
- combo proxy prefilter;
- trade-list-first internal exact representation;
- hit-time tables;
- fast monotone TP/SL kernel;
- reference-vs-fast self-check;
- public support for `direction_mode`, `sizing_mode`, `primary_metric`.

### Не сохраняем как baseline

- full `final_signal_row` retained frontier;
- forced exact Stage B because “retained payload exists”;
- process-fanned Stage B as default runtime shape;
- no-risk runs flowing through heavy generic Stage B;
- acceptance by intuition without benchmark gates.

## 10. Non-goals

- Не обещать notebook-parity для `RG-ALT` в первой волне.
- Не возвращаться к legacy v1 staged runner.
- Не вводить отдельный engine для sync и отдельный engine для jobs.
- Не считать, что config tuning заменяет architecture cutover.
- Не пытаться достичь целей только за счёт повышения числа потоков.

## 11. Короткая рекомендуемая последовательность prompt-цепочки

1. Benchmark contract and perf harness
2. Remove retained `final_signal_row`
3. Streaming Stage A exact scoring
4. No-risk Stage A direct finalization
5. Compact-trade-only retained payload for risk path
6. Keep Stage B fast path enabled with retained payload
7. Finalist-only exact replay
8. In-process Stage B default
9. Process fallback policy
10. Thread-budget tuning
11. Perf-smoke/docs closure

## 12. Итоговое правило принятия результата

Этот performance program считается успешным только если одновременно верно следующее:

- `NR2` backend reaches notebook-parity target on the same thread budget;
- `RG-TTR` backend reaches notebook-parity target on the same thread budget;
- no-risk class no longer enters heavy Stage B by default;
- default Stage B no longer explodes into multiple large Python worker processes for canonical workloads;
- perf-smoke can catch regression automatically.

Если хотя бы один из этих пунктов не выполнен, rollout не считается завершённым.
