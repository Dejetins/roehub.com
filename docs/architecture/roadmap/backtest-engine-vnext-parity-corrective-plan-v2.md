---
title: Корректирующий план достижения notebook-parity для backtest engine vNext (v2)
version: 2
status: draft
owner: backtest
---

# Корректирующий план достижения notebook-parity для backtest engine vNext (v2)

Статус: proposed corrective roadmap after live `NR2` rerun verification; execution master-plan  
Дата фиксации: 2026-04-15  
Область: `backtest`, `apps/api`, `apps/worker`, execution-profile topology, parity runtime
contract, Stage A exact kernels, risk-grid parity closure, public API/UI contract, persisted parity
state, compatibility invariants, artifact dependency status, benchmark/perf-smoke

## 1. Зачем нужен этот документ

Документ [Корректирующий план достижения notebook-parity для backtest engine vNext](./backtest-engine-vnext-parity-corrective-plan-v1.md)
зафиксировал первую corrective program после live benchmark failure.

После выполнения memory-focused исправлений и нового живого rerun canonical `NR2` shape стало
понятно, что repository действительно исправил часть симптомов, но не приблизился к notebook
настолько, насколько требует target contract:

- idle baseline API больше не раздут до multi-GB состояния;
- request-local peak RSS заметно снизился;
- sync run now stays `sync_inline` and effectively single-process;
- но runtime всё ещё исполняется как `hybrid_conservative` shared shortlist path;
- top result всё ещё расходится с notebook anchor не на проценты, а на completely different winner.

Этот документ нужен как вторая corrective program. В отличие от `v1`, он фиксирует уже не набор
локальных hot-path дефектов, а архитектурный разрыв между:

- service topology, которая сейчас жёстко закрепляет canonical sync launch за
  `hybrid_conservative`,
- и notebook-parity целью, которой нужен parity-first no-risk exact path.

Документ intentionally не открывает новый product redesign. Он описывает только тот набор
доработок, который теперь нужен, чтобы backend service пришёл либо к target parity, либо к
максимально близкому состоянию, подтверждённому live benchmark на benchmark host.

Начиная с этой редакции документ также становится единственным исполняемым master-plan для
remaining scope по:

- `docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md`
- `docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md`
- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md`

То есть unresolved scope из этих документов больше не должен исполняться “рядом” как отдельная
программа. Если обязательство остаётся актуальным, оно должно быть либо перенесено сюда как
explicit epic/gate, либо зафиксировано здесь как completed prerequisite с подтверждающей ссылкой.

## 2. Зафиксированные факты по живому benchmark

Canonical no-risk notebook anchor:

- `tests/notebook_tests/new_engine/02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb`

Canonical backend rerun:

- `run_id=472df61a-8229-4153-a51d-077657e8481b`

Зафиксированные измерения на `macstudio` при одинаковом thread budget (`NUMBA_NUM_THREADS=4`):

- service:
  - `execution_mode=sync_inline`
  - `execution_profile_mode=hybrid_conservative`
  - `artifact_slot=slot_a`
  - `artifact_asof_date=2026-04-14`
  - `wall_clock_seconds = 18.887235915999554`
  - `cpu_time_seconds = 25.814602943999997`
  - `peak_rss_bytes = 16409231360` (`~16.41 GB`)
  - `peak_cpu_percent_sum = 185.5`
  - `max_python_processes_seen = 1`
  - top-1 `total_return_pct = -92.35799778293095`
- notebook:
  - `wall_clock_seconds = 7.808417250000275`
  - `cpu_time_seconds = 18.775969`
  - `peak_rss_bytes = 1374126080` (`~1.37 GB`)
  - `peak_cpu_percent_sum = 387.7`
  - `max_python_processes_seen = 1`
  - top-1 `total_return_pct = 1621.7322019157828`

Top-1 strategy divergence:

- service winner:
  - `ma.dema`: `source=close`, `window=71`
  - `ma.hma`: `source=high`, `window=184`
- notebook winner:
  - `ma.dema`: `source=hlc3`, `window=145`
  - `ma.hma`: `source=high`, `window=127`

Derived ratios:

- `wall_clock_ratio = 2.418830258590355x`
- `peak_rss_ratio = 11.941576248956718x`
- `peak_cpu_ratio = 0.47846272891410885x`

Что улучшилось по сравнению с предыдущим diagnostic benchmark того же дня:

- cold API baseline RSS after restart упал примерно с `~9.73 GB` до `~0.21 GB`;
- service peak RSS упал примерно с `~22.32 GB` до `~16.41 GB`;
- service path remains single-process and no longer looks like worker/process-fanout problem.

Что не улучшилось:

- wall time всё ещё далека от notebook;
- CPU saturation всё ещё сильно ниже notebook;
- top result всё ещё semantically divergent;
- canonical service path всё ещё живёт на `hybrid_conservative`, а не на notebook-shaped exact
  path.

Следствие:

- текущее состояние больше нельзя объяснять только memory hot spots;
- remaining gap теперь подтверждён как topology/runtime-contract problem, а не только as-if
  micro-optimization debt.

## 3. Как сейчас реально связаны service layers

Ниже зафиксирован текущий service chain для canonical sync launch.

### 3.1 `POST /backtests` sync wrapper насильно закрепляет internal `hybrid_conservative`

Sync launch wrapper в:

- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`

прямо inject-ит internal `execution_profile_mode=hybrid_conservative` в runtime payload для
`POST /backtests`.

Это означает:

- canonical sync launch уже не выбирает parity path independently;
- public API transport не знает об этом profile;
- persisted sync metadata становится truthful только для уже выбранного universal hybrid path.

### 3.2 `RunBacktestUseCase` строит обычный runtime plan, а потом заменяет его на reduced plan

Основной sync orchestration в:

- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`

делает:

1. timeline build,
2. planner build,
3. conditional branch:
   - если profile uses hierarchical shortlist runtime,
   - runtime plan replaces itself with `hierarchical_shortlist_builder.build_runtime_plan(...)`,
4. downstream scorer + Stage A shortlist + no-risk finalization continue already from the reduced
   plan.

То есть service не идёт по notebook-shaped exact plan и потом узко оптимизирует hot path. Он уже
на верхнем уровне переходит в другой runtime contract.

### 3.3 `hybrid_conservative` по собственному контракту является approximate universal path

Документ:

- `docs/architecture/backtest/README.md`

и код:

- `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`

фиксируют, что `hybrid_conservative` — это:

- universal shortlist runtime,
- diversified retention,
- hierarchical combine,
- reduced runtime plan for survivors,
- exact scorer only for survivors.

Это полезный approximate rollout path, но он не эквивалентен notebook parity path. По design он
имеет право потерять часть exact frontier до final scorer.

### 3.4 Planner / adaptive-selector model всё ещё смешивает rollout profile и parity goal

Execution profile surface живёт в:

- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`

Сейчас эти слои рассматривают:

- `exact_small`
- `exact_parallel`
- `hybrid_conservative`
- `hybrid_family`

как общий profile catalog. Для canonical no-risk service launch нет отдельного parity-first
profile contract. Поэтому service topology вынуждена переиспользовать rollout profile, который
изначально решал другую задачу.

### 3.5 Stage A exact internals всё ещё строят слишком широкий dense runtime state

Даже после memory fixes Stage A retained exact path остаётся broad and dense:

- `_retain_indicator_rows(...)` грузит весь per-indicator row pool:
  `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `load_chunk_runtime_inputs(...)` materializes chunk signal rows:
  `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `signal_aggregator_kernel.py` allocates dense `[indicator, variant, time]` cube and output
  matrix:
  `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- retained exact batch still builds internal dense trade-list-first state:
  `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
  and `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`

Это уже не тот catastrophic idle baseline bug, который был раньше. Это structural memory shape of
the exact retained frontier.

### 3.6 No-risk finalization уже bypasses generic Stage B, но приходит туда с неправильным input

Runtime plan correctly classifies no-risk class as:

- `stage_b_execution_mode = bypassed_no_risk`
- `stage_b_process_fallback_threshold = none`

through:

- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`

Но этот bypass приходит уже после shortlist/reduced-plan decisions. Поэтому проблема сейчас не в
самом Stage B branch, а в том, какие candidates дошли до no-risk finalization.

### 3.7 Worker/job-runner повторяет тот же reduced runtime contract

Background runner in:

- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`

mirror-ит тот же conditional branch на hierarchical shortlist runtime.

Следствие:

- sync и worker сейчас разделяют один и тот же divergence source;
- worker cannot serve as “true exact parity fallback” for the same no-risk class;
- even after persistence fixes worker path inherits the same reduced-plan semantics.

## 4. Что считаем подтверждёнными root causes сейчас

### 4.1 Canonical sync launch всё ещё жёстко закреплён за approximate runtime

Главный root cause теперь не “planner чуть ошибся”. Главный root cause в том, что service sync
wrapper deliberately injects `hybrid_conservative` into the canonical path.

Пока это решение активно, canonical `NR2` не может стать notebook-like by local tuning only.

### 4.2 Execution-profile topology сейчас архитектурно смешивает две разные цели

Текущая profile model пытается одним и тем же surface обслужить:

- exact runtime classification,
- hybrid rollout,
- adaptive selector recommendations,
- and canonical sync launch.

Для notebook parity это плохое решение, потому что approximate rollout profile и parity-first
exact service path должны быть разными contracts.

### 4.3 `HierarchicalShortlistRuntimePlan` по design не гарантирует notebook winner preservation

Даже если numerical row scoring стал дешевле и deterministic, сам reduced-plan contract остаётся
approximate:

- retained row pools narrow candidates before exact no-risk scoring;
- hierarchical combine reduces Stage A search space;
- exact final scorer sees only survivors.

Поэтому top-1 divergence на `472df...` — не случайный residual bug, а ожидаемый риск текущей
архитектуры.

### 4.4 Exact retained frontier всё ещё materializes too much dense state

После того как idle baseline bug and one trade-batch spike были исправлены, remaining memory gap
переехал в persistent structural shape:

- full row-pool loads,
- dense indicator cube,
- dense `final_signal[V, T_signal]`,
- dense retained exact batch.

Это и держит:

- высокий request-local RSS,
- page-fault / memcpy pressure,
- и слабую загрузку CPU relative to notebook.

### 4.5 Service and worker still share the same divergence contract

Даже если sync path будет слегка ускоряться, worker path не станет parity-correct automatically,
потому что он повторяет тот же reduced-plan branch.

Поэтому corrective program должна править не только sync hot path, но и общий parity contract
между sync и worker.

### 4.6 Что уже не считаем текущей root cause

Ниже перечислено то, что больше не должно рассматриваться как главный remaining gap:

- loader lifecycle / idle baseline memory problem — materially improved and no longer the main
  blocker;
- previous compact trade batch pathological allocation — partially fixed and no longer explains
  the whole gap by itself;
- `GenericRowScorerV2` as the canonical Stage A hot path — больше не main culprit for `NR2`;
- process fan-out — fresh canonical rerun remained effectively single-process;
- `Infinity` JSON persistence bug — больше не объясняет semantic divergence and memory/runtime
  gap.

Это важно, чтобы новая corrective program не ушла снова в локальные hot-spot patches вместо
service-topology split.

## 5. Что считаем достаточным scope этого master-plan

Начиная с этой редакции документ intentionally фиксирует полный remaining scope всей программы
`backtest engine vNext`, который ещё нужен для честного closure.

Эта master-program intentionally ограничена одиннадцатью рабочими направлениями:

1. `canonical service-topology split away from hybrid_conservative`
2. `planner / execution-profile / adaptive-selector topology split`
3. `first-class parity runtime plan for canonical no-risk exact class`
4. `notebook-shaped Stage A parity pipeline`
5. `pair-first no-risk exact kernel and memory-shape collapse`
6. `sync / worker parity contract and persisted parity state`
7. `benchmark observability foundation`
8. `public contract and UX closure`
9. `risk-grid parity closure`
10. `compatibility and correctness invariants`
11. `artifact dependency status`

Если хотя бы одно из этих направлений не будет закрыто, то master-plan не считается завершённым:
backend service либо останется approximate by design, либо будет “быстрее”, но не parity-correct,
либо сохранит незакрытые продуктовые/контрактные хвосты из старых roadmap’ов.

## 6. Целевое runtime состояние после master-plan closure

### 6.1 Для canonical `NR2`

Canonical `NR2` run должен иметь такую форму:

- request stays `sync_inline`;
- sync `POST /backtests` no longer hard-pins `hybrid_conservative`;
- planner classifies the request into an internal parity-first no-risk exact class;
- runtime plan explicitly carries narrowed retained row pools and narrowed combo cardinality;
- canonical path bypasses hierarchical shortlist runtime completely;
- Stage A enumerates exact narrowed combos directly, in notebook-equivalent deterministic order;
- no-risk exact evaluation does not materialize broad dense cube/batch state for the full retained
  frontier;
- `stage_b_execution_mode = bypassed_no_risk`;
- `max_python_processes_seen = 1`;
- top-1 winner and top-level metrics match notebook anchor for the same artifact slot and thread
  budget.

### 6.2 Для worker parity

Если тот же no-risk class исполняется как background job:

- worker receives the same parity runtime classification, not `hybrid_conservative`;
- worker uses the same narrowed parity runtime plan;
- persisted parity state is sufficient to resume no-risk exact finalization without re-entering
  approximate universal shortlist logic;
- sync and worker outputs stay contract-equivalent.

### 6.3 Для canonical `RG-TTR`

Canonical risk-grid total-return run должен иметь такую форму:

- Stage A retains compact, deterministic survivors instead of broad signal-row replay payloads;
- retained payload for the risk path is compact-trade-first and does not carry full dense
  `final_signal_row` state;
- Stage B keeps cheap total-return grid ranking enabled even when retained payload exists;
- exact replay is bounded by finalist scope only;
- benchmark-sized workloads stay `in_process` and `single-process` by default;
- process fan-out exists only as explicit fallback above documented workload thresholds;
- top-level metrics and ranking stay comparable to the notebook anchor on equal thread budget.

### 6.4 Public contract and compatibility invariants

После closure master-plan должен одновременно сохранять следующие инварианты:

- public launch surface keeps only `ranking.primary_metric` as the ranking selector;
- `secondary_metric`, `warmup_bars`, and `top_trades_n` are removed from active
  request/defaults/UI/history contracts;
- launch path remains `summary-only`;
- trades, report rows, and report tables are materialized only on explicit on-demand
  `variant-report` flow;
- public progress vocabulary remains `stage_a`, `stage_b`, `finalizing` even if internal
  sub-stages change;
- supported alternative `primary_metric` values remain functionally correct for both no-risk and
  risk-grid paths;
- persisted summary/job payloads remain finite and JSON-safe.

### 6.5 Целевые benchmark gates

Primary target for canonical `NR2`:

- top-1 winner identical to notebook anchor on equal thread budget;
- `wall_clock_ratio <= 1.18x`;
- `peak_rss_ratio <= 1.35x`;
- `max_python_processes_seen = 1`;
- `execution_mode = sync_inline`;
- `stage_b_execution_mode = bypassed_no_risk`.

Near-target fallback for `NR2`, acceptable only as explicit interim milestone and not as final
closure:

- top-1 winner identical to notebook anchor;
- `wall_clock_ratio <= 1.30x`;
- `peak_rss_ratio <= 2.00x`;
- same single-process no-risk runtime shape.

Primary target for canonical `RG-TTR`:

- notebook-equivalent top result and top-level metrics on equal thread budget;
- `wall_clock_ratio <= 1.18x`;
- Stage B stays `in_process` and `single-process` by default;
- retained payload remains compact-trade-first;
- exact replay count is bounded by finalist scope and observable in benchmark output.

Functional target for `RG-ALT`:

- correctness remains intact for shipped alternative `primary_metric` values;
- runtime does not regress by more than `10%` versus the accepted pre-cutover live baseline on the
  benchmark host;
- alternative metrics do not accidentally force a heavier runtime class when the ranking semantics
  do not require it.

## 7. Пошаговый план реализации

## EPIC D0. Canonical service-topology split

### Цель

Убрать главное архитектурное препятствие: canonical sync `POST /backtests` must stop forcing the
approximate `hybrid_conservative` rollout profile into the no-risk parity path.

### Что заменяем

Current internal sync-wrapper pinning in:

- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`

### Что делаем

- ввести отдельный internal parity-first execution profile or runtime class for canonical no-risk
  exact service launches, for example `exact_no_risk_parity`;
- убрать hard pinning `hybrid_conservative` from sync wrapper for that class;
- keep `hybrid_conservative` as approximate rollout/runtime-acceleration surface only;
- persist truthful internal metadata so history/status/docs no longer present canonical parity runs
  as `hybrid_conservative`.

### Основные файлы

- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `apps/api/dto/backtest_runs.py`
- `apps/api/dto/backtests.py`

### Acceptance gate

- fresh canonical `NR2` service run must no longer persist
  `execution_profile_mode=hybrid_conservative`;
- canonical sync launch must still stay `sync_inline`.

## EPIC D1. Planner / selector topology split for parity class

### Цель

Развести approximate rollout selection and parity-first exact classification into different
decisions.

### Что заменяем

Current shared selection topology in:

- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`

### Что делаем

- add explicit canonical no-risk parity classification evidence:
  - disabled-risk single-cell class,
  - low indicator-block cardinality,
  - narrowed retained-row evidence,
  - notebook-shaped cost units;
- adaptive selector must not recommend or force approximate hybrid runtime for this class;
- launch budgets for parity class must be evaluated against its own exact narrowed workload, not
  against hybrid rollout heuristics or legacy broad-grid evidence;
- keep rollout benchmarks for `hybrid_conservative` separate and reviewable.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/services/v2/adaptive_selector_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_adaptive_selector_v2.py`
- `tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py`

### Acceptance gate

- planner must expose deterministic evidence that canonical `NR2` belongs to the parity-first
  no-risk exact class;
- sync launch acceptance for that class must no longer depend on `hybrid_conservative` rollout
  policy.

## EPIC D2. First-class parity runtime plan

### Цель

Сделать notebook-shaped no-risk exact plan first-class runtime contract instead of reduced plan
derived from hybrid shortlist runtime.

### Что заменяем

Current `effective_runtime_plan = hierarchical_shortlist_builder.build_runtime_plan(...)` branch
for canonical no-risk service launches.

### Что делаем

- add a dedicated parity runtime-plan type carrying:
  - retained per-indicator row pools,
  - narrowed combo cardinality,
  - deterministic combo ordering,
  - explicit no-risk exact execution shape metadata;
- stop representing canonical no-risk runtime as generic “reduced plan”;
- expose debug counters that can be compared directly with notebook anchor:
  - retained rows per indicator,
  - narrowed combo total,
  - exact replay count,
  - no-risk finalization counters.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `src/trading/contexts/backtest/application/services/v2/notebook_parity_benchmark_corpus_v2.py`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`

### Acceptance gate

- canonical `NR2` must no longer depend on `BacktestHierarchicalShortlistBuilderV2` to build its
  active runtime plan;
- runtime plan counters for the same slot/time range must be comparable to the notebook anchor,
  not to hybrid rollout evidence.

## EPIC D3. Notebook-shaped Stage A parity pipeline

### Цель

Вернуть canonical no-risk service path к notebook-equivalent candidate semantics before final
no-risk exact scoring.

### Что заменяем

Current canonical dependence on hierarchical shortlist reduction semantics.

### Что делаем

- bypass `BacktestHierarchicalShortlistBuilderV2` entirely for the parity-first no-risk class;
- keep numeric row prefilter, but make retained-row contract parity-owned rather than hybrid-owned;
- enumerate exact narrowed indicator-pair combos directly from retained row pools;
- preserve notebook-equivalent deterministic order and tie-break semantics;
- make top-result parity a blocking gate for this phase.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `tests/unit/contexts/backtest/application/services/v2/test_stage_a_shortlist_builder_v2.py`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`

### Acceptance gate

- canonical `NR2` service top-1 winner must match notebook winner on the same host / slot /
  thread budget;
- remaining gap after this phase may still be memory/perf-related, but not winner-identity
  related.

## EPIC D4. Pair-first no-risk exact kernel and memory-shape collapse

### Цель

Убрать основной remaining request-local memory gap and raise CPU saturation by replacing broad
dense retained-frontier internals with pair-first/blockwise exact evaluation.

### Что заменяем

Current dense retained-frontier internals in:

- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`

### Что делаем

- introduce parity kernel path for canonical 2-indicator no-risk class that:
  - evaluates retained indicator pairs directly,
  - avoids full dense `[indicator, variant, time]` cube allocation for the parity path,
  - avoids full dense retained `final_signal[V, T_signal]` batch for large survivor blocks,
  - uses bounded blockwise workspaces sized by pair-block cardinality;
- keep the generic dense kernels for universal exact/hybrid paths where they are still useful;
- make parity kernel path selectable only for the parity-first no-risk exact class.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `tests/unit/contexts/backtest/application/services/v2/test_signal_aggregator_kernel_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_trade_compactor_kernel_v2.py`

### Acceptance gate

- canonical `NR2` peak RSS must drop materially from current `~16.41 GB` toward notebook scale;
- intermediate gate: `peak_rss_ratio <= 3.0x` and `wall_clock_ratio <= 1.60x`;
- CPU saturation must move materially upward toward notebook behavior.

## EPIC D5. Sync / worker parity contract and persisted parity state

### Цель

Сделать parity-first no-risk service path and worker path contract-equivalent instead of letting
worker fall back to the older shared hybrid reduction semantics.

### Что заменяем

Current worker reuse of the same reduced-plan branch and incomplete persisted parity context.

### Что делаем

- persist compact parity runtime state sufficient to resume canonical no-risk exact evaluation;
- worker must reconstruct or reuse the same parity runtime plan, not `hybrid_conservative`;
- keep payload additive, deterministic, and backward-readable where needed;
- ensure sync and worker no-risk outputs are bitwise or tolerance-equivalent on canonical anchors.

### Основные файлы

- `src/trading/contexts/backtest/domain/entities/backtest_job_results.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- corresponding migration / storage schema files if contract expansion requires them

### Acceptance gate

- background execution of the same canonical no-risk class must produce the same winner and top
  metrics as sync execution;
- worker must not re-enter hybrid shortlist reduction for the parity-first no-risk class.

## EPIC D6. Benchmark observability foundation

### Цель

Сделать future regressions immediately observable и подготовить measured-truth surface, на которой
потом можно честно закрывать весь master-plan.

### Что делаем

- expose parity-critical runtime-shape fields directly in persisted run/benchmark surfaces:
  - effective parity profile,
  - `stage_b_execution_mode`,
  - `stage_b_process_fallback_threshold`,
  - retained row counts,
  - narrowed combo count,
  - exact replay count;
- rerun canonical `NR2` service vs notebook on equal thread budget;
- rerun canonical `RG-TTR` service vs notebook on equal thread budget;
- rerun worker parity validation for the same class if background path stays supported;
- keep docs/fixtures blocked on fresh live capture until all remaining epics are closed.

### Основные файлы

- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- `tests/perf_smoke/contexts/backtest/fixtures/backtest_notebook_parity_benchmark_corpus_v1.json`
- `src/trading/contexts/backtest/application/services/v2/notebook_parity_benchmark_corpus_v2.py`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

### Acceptance gate

- benchmark surfaces must expose live runtime-shape fields for `NR2` and `RG-TTR` without relying
  on code-level inference;
- repository must have repeatable equal-thread-budget capture flow ready before contract-closure
  epics are declared complete.

## EPIC D7. Public contract and UX closure

### Цель

Вернуть в master-plan обязательный public-contract closure from implementation roadmap, чтобы
parity fix не оставил рядом несинхронизированные API/UI/defaults/history surfaces.

### Что заменяем

Current leftover public-contract ambiguity around:

- `secondary_metric`
- `warmup_bars`
- `top_trades_n`
- mixed launch-vs-detail trade materialization assumptions

### Что делаем

- remove `secondary_metric` from active request DTOs, runtime defaults, UI, and persisted history
  vocabulary;
- remove `warmup_bars` from public input and keep it as internal derived/debug-only value;
- remove `top_trades_n` from active request/defaults/UI/normalization surfaces;
- keep launch payloads and persisted top rows `summary-only`;
- keep trades/report bodies only on explicit run-scoped `variant-report` path;
- sync API, web defaults, history/detail docs, and browser behavior so they all describe the same
  public contract.

### Основные файлы

- `apps/api/dto/backtests.py`
- `apps/api/dto/backtest_runtime_defaults.py`
- `apps/api/dto/backtest_runs.py`
- `apps/api/dto/backtest_jobs.py`
- `apps/api/routes/backtests.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_api_v1.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
- `apps/web/templates/backtests.html`
- `apps/web/templates/backtest_run_summary.html`
- `apps/web/dist/backtest_ui.js`
- `apps/web/dist/backtest_runs_ui.js`
- `apps/web/dist/backtest_jobs_ui.js`
- `apps/api/wiring/modules/backtest.py`
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`
- `docs/architecture/backtest/README.md`

### Acceptance gate

- active public launch contract accepts only `ranking.primary_metric` as ranking selector;
- `secondary_metric`, `warmup_bars`, and `top_trades_n` disappear from active API/UI/defaults and
  persisted run-history vocabulary;
- launch path stays `summary-only`, and full trades/report bodies stay on explicit `variant-report`
  flow only.

## EPIC D8. Risk-grid parity closure

### Цель

Вернуть в master-plan обязательный `RG-TTR` scope, чтобы notebook-parity closure не ограничивался
только canonical no-risk path.

### Что заменяем

Current partially closed risk-grid runtime contract spread across:

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`

### Что делаем

- keep retained payload for the risk path compact-trade-first only;
- retained payload must not disable cheap Stage B total-return grid ranking;
- exact replay must become finalist-only and observable in runtime counters/tests;
- single-process `in_process` Stage B becomes the default for benchmark-sized `RG-TTR` workloads;
- process fan-out remains fallback-only above explicit documented workload thresholds;
- rerun canonical `RG-TTR` live benchmark on equal thread budget and keep it as blocking closure
  gate.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `configs/prod/backtest.yaml`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

### Acceptance gate

- canonical `RG-TTR` benchmark reaches its accepted parity gate on equal thread budget;
- retained payload for the risk path stays compact-trade-first and does not carry full signal-row
  state;
- exact replay count is bounded by finalist scope;
- default benchmark-sized risk-grid runtime no longer relies on process fan-out.

## EPIC D9. Compatibility and correctness invariants

### Цель

Явно зафиксировать те correctness and compatibility obligations, которые нельзя считать
“вторичными” относительно parity, потому что без них closure будет недостоверным.

### Что заменяем

Current partial handling of:

- `RG-ALT` functional guardrails
- alternative `primary_metric` support
- finite-metric JSON persistence
- progress vocabulary stability

### Что делаем

- keep shipped alternative `primary_metric` values functionally supported for both no-risk and
  risk-grid classes;
- keep `RG-ALT` as correctness-first class with explicit non-regression gate;
- define one explicit finite-metric normalization rule for persisted summary rows, job snapshots,
  and benchmark output surfaces;
- sanitize `Infinity`, `-Infinity`, and `NaN` before JSON serialization where persistence/export
  contracts require finite values;
- preserve public progress vocabulary `stage_a`, `stage_b`, `finalizing` even if internal
  sub-stages and counters become more detailed.

### Основные файлы

- `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_core_v2.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py`
- `apps/api/dto/backtest_runs.py`
- `tests/unit/contexts/backtest/application/services/v2/test_metrics_kernel_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`
- `tests/perf_smoke/contexts/backtest/test_backtest_notebook_parity_perf_smoke_v1.py`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`

### Acceptance gate

- `RG-ALT` remains functionally correct and does not regress runtime by more than `10%` versus the
  accepted live baseline on the benchmark host;
- persisted rows, job snapshots, and exported benchmark/report surfaces remain finite and
  JSON-safe;
- public progress semantics stay stable even after internal runtime-stage refactoring.

## EPIC D10. Artifact dependency status

### Цель

Убрать ambiguity вокруг wider TP/SL artifact dependency, чтобы master-plan не оставлял открытым
вопрос, должен ли он ещё внедряться или уже является completed prerequisite.

### Что заменяем

Current ambiguity inherited from the older implementation roadmap about whether:

- wider canonical `hit_times/1m` TP/SL grids are already delivered;
- Stage B loaders/kernels are already grid-agnostic;
- parity closure must still include this rollout or only validate its presence.

### Что делаем

- explicitly fix in this master-plan that wider TP/SL artifact grids and grid-agnostic Stage B
  loaders are treated as `completed prerequisite`, not as a new unresolved epic;
- cite the confirming authority for that prerequisite in active docs/tests:
  - `docs/architecture/backtest/README.md`
  - `docs/architecture/backtest/README.md`
  - `tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py`
  - `tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py`
- require any contradiction discovered during `RG-TTR` closure to reopen this dependency as a
  blocker before final acceptance.

### Основные файлы

- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py`
- `tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py`

### Acceptance gate

- this roadmap explicitly states that wider TP/SL artifact grids and grid-agnostic runtime loaders
  are already delivered prerequisites;
- repository docs/tests cited above remain sufficient to confirm that status for master-plan
  readers and reviewers;
- if live `RG-TTR` closure contradicts that prerequisite, the contradiction is treated as a
  blocking defect, not as silent roadmap drift.

## 8. Что intentionally не входит в scope

- не открываем новый product redesign outside the canonical parity path;
- не возвращаем `hybrid_conservative` как default explanation for canonical no-risk service runs;
- не считаем thread tuning главным решением;
- не расширяем public `POST /backtests` transport новыми user-facing profile knobs;
- не возвращаем `secondary_metric`, `warmup_bars`, или `top_trades_n` как active public knobs;
- не переписываем family-plugin / `hybrid_family` rollout beyond what is necessary to stop it
  interfering with parity classification;
- не делаем broad docs rewrite outside directly touched runtime contracts.

## 9. Порядок реализации

Порядок должен быть только таким:

1. `D0` canonical service-topology split
2. `D1` planner / selector topology split
3. `D2` first-class parity runtime plan
4. `D3` notebook-shaped Stage A parity pipeline
5. `D4` pair-first no-risk exact kernel and memory-shape collapse
6. `D5` sync / worker parity contract
7. `D6` benchmark observability foundation
8. `D7` public contract and UX closure
9. `D8` risk-grid parity closure
10. `D9` compatibility and correctness invariants
11. `D10` artifact dependency status validation

Причина:

- без `D0` service остаётся жёстко закреплён за approximate runtime;
- без `D1` parity class продолжит зависеть от rollout profile semantics;
- без `D2` canonical path всё ещё будет жить как reduced plan instead of first-class runtime;
- без `D3` top-result parity не может стать blocking contract;
- без `D4` memory and CPU symptoms останутся слишком далеки от notebook;
- без `D5` worker path останется отдельной divergence surface;
- без `D6` нельзя честно измерять closure для `NR2` и `RG-TTR`;
- без `D7` останутся несинхронизированные API/UI/defaults/history contracts из старого redesign
  scope;
- без `D8` master-plan будет закрывать только no-risk path, но не canonical risk-grid parity
  target;
- без `D9` останутся correctness/compatibility holes, которые сломают доверие к closure;
- без `D10` останется ambiguity, delivered ли wider TP/SL prerequisite или он был silently skipped.

## 10. Документы, которые должны быть синхронизированы

- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v1.md`
- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`
- `docs/architecture/roadmap/backtest-engine-vnext-implementation-plan-v1.md`
- `docs/architecture/roadmap/backtest-engine-vnext-notebook-parity-plan-v1.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`
- `docs/architecture/apps/web/web-backtest-history-and-variant-detail-v2.md`

## 11. Итоговое правило этого документа

Этот master-plan считается завершённым только тогда, когда одновременно выполнены все
условия:

- canonical service `NR2` path больше не живёт на `hybrid_conservative`;
- canonical `NR2` top-1 winner совпадает с notebook anchor при equal thread budget;
- canonical `RG-TTR` path keeps notebook-equivalent top result on equal thread budget while
  staying `in_process`, `single-process` by default, and finalist-only for exact replay;
- runtime shape остаётся `sync_inline`, `single-process`, `bypassed_no_risk`;
- `NR2` and `RG-TTR` либо реально проходят свои target gates, либо для `NR2` документированно
  зафиксирован near-target interim state, который не выдаётся за final closure;
- sync and worker no-risk outputs больше не расходятся контрактно;
- active public launch contract keeps only `ranking.primary_metric`; `secondary_metric`,
  `warmup_bars`, and `top_trades_n` удалены из active API/UI/defaults/history surfaces;
- launch path остаётся `summary-only`, а trades/report bodies materialize-ятся только on-demand
  через explicit `variant-report`;
- `RG-ALT` remains functionally correct and within its non-regression runtime envelope;
- persisted summary/job/benchmark payloads remain finite and JSON-safe;
- public progress semantics remain `stage_a`, `stage_b`, `finalizing`;
- wider TP/SL artifact dependency explicitly recorded here as completed prerequisite and not
  contradicted by live `RG-TTR` closure evidence;
- docs описывают именно live-delivered service truth, а не intended architecture.
