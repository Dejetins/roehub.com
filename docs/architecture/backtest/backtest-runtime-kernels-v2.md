# Backtest Runtime Kernels V2 (R5-02 contract / R6-01 loader-context boundary for `signal_tf + 1m_risk`)

Этот документ фиксирует канонический production contract для Stage A / Stage B runtime kernels
после shipped artifact-backed cutover, не меняя R5-01 artifact contracts.

Для активного redesign surface canonical target anchor теперь находится в
`docs/architecture/backtest/backtest-engine-vnext.md` и опирается на
`tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`.
`tests/notebook_tests/06_backtest_compute.ipynb` остаётся только historical semantics reference,
а не активным implementation anchor для новых prompts.

Статус: `Milestone R5 / EPIC R5-02`, `Milestone R6 / EPIC R6-01 + R6-02 + R6-03 + R6-04`,
`Milestone R10 / EPIC R10-01 production hot-path cutover`  
Следующий этап handoff: `R10-03 perf/runbook closure`

## Status

- Status: active canonical production runtime contract after R10-01 and R10-02.
- Supersedes as hot-path description:
  - `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`
  - `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
  - `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- Compatibility note:
  - legacy v1 modules remain import-stable only where migration boundaries still need them;
  - production launch, claimed worker execution, and run-scoped lazy detail do not silently
    fallback to legacy runtime orchestration.
  - target redesign vocabulary and future cutover planning now live in
    `docs/architecture/backtest/backtest-engine-vnext.md`; this document remains the current
    shipped runtime contract until a later cutover prompt updates it explicitly.

Связанные документы:

- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/backtest-engine-vnext.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md`
- `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `tests/notebook_tests/new_engine/01_run_322_btcusdt_1h_artifact_probe.ipynb`
- `tests/notebook_tests/05_hit_time_grid.ipynb`

Historical notebook references only:

- `tests/notebook_tests/06_backtest_compute.ipynb`

## Роль документа

- Это главный entrypoint для R5-02 и будущего R6 implementation path.
- Notebook остаётся semantics source, но production runtime трактует его как
  `notebook-derived kernel semantics` и `not a literal notebook orchestration script`.
- Для target redesign decisions, public vocabulary cleanup, и будущих cutover prompts canonical
  handoff идёт через `docs/architecture/backtest/backtest-engine-vnext.md`, а не через
  `06_backtest_compute.ipynb`.
- R5-01 остаётся immutable input boundary: runtime читает только shipped `1m hit-times`,
  `prices/<tf>`, `prices/1m`, `mappings/<tf>` и `signals/<tf>/<indicator_id>`.
- R6-01 уже реализует runtime-side artifact loading primitives:
  `artifact_slot_resolver.py`, `price_arrays_loader.py`, `signal_matrix_loader.py`.
- R6-02 уже реализует Stage A artifact-backed kernels и additive shortlist bridge:
  `signal_aggregator_kernel.py`, `trade_compactor_kernel.py`,
  `stage_a_shortlist_builder_v2.py`.
- R6-03 уже реализует Stage B artifact-backed risk kernels и additive scorer bridge:
  `risk_exit_kernel_1m.py`, `metrics_kernel.py`,
  `artifact_backed_stage_b_scorer_v2.py`.
- Sync и background starts теперь обязаны делить один immutable `slot-pinned context` contract,
  а не расходиться по разным pointer/discovery paths.
- Документ не вводит новые API payloads, новые request TF или новые persisted storage contracts.
- Launch/persistence flows остаются `summary-only`; full user-facing trades/report bodies для
  выбранного варианта по-прежнему относятся к on-demand detail surfaces, а не к default runtime
  result.

## Канонический словарь

| Термин | Канонический смысл |
|---|---|
| `signal timeline` | Request timeframe timeline, где строится `final_signal` и фиксируются подтверждения стратегии. |
| `execution timeline` | Canonical `1m` timeline, где живут `1m hit-times` и исполняются risk exits. |
| `compact trade list` | Упорядоченный список сделок `[(entry_exec_idx, direction, sig_exit_exec_idx)]` без полного bar-by-bar replay. |
| `signal bar` | Один бар `signal timeline`, закрытие которого разрешает вычислить следующее действие стратегии. |
| `entry_exec` | Индекс первого execution bar после закрытия signal bar; для artifact-backed runtime это `bar_close_1m_idx + 1` с sentinel fallback. |
| `sig_exit_exec` | Execution index следующего противоположного подтверждения, либо `sentinel_index`, если signal exit отсутствует. |
| `sentinel_index` | Индекс `T_exec`, означающий “событие не произошло до конца execution timeline”. |

## Transfer Matrix

| Notebook concept | Production contract | Target v2 module | Status |
|---|---|---|---|
| Pair confirmations on request TF | Deterministic signal aggregation on `signal timeline` with output value set `{-1, 0, 1}` | `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py` | Implemented in R6-02 |
| `build_trade_list_for_pair` | `compact trade list` with deterministic ordering and sentinel-based signal exits | `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py` | Implemented in R6-02 |
| `evaluate_trade_factor` over hit tables | `1m hit-times` risk-exit resolution on `execution timeline` | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | Implemented in R6-03 |
| Monotone diff-buffer decomposition | `fast TP/SL grid search` over precomputed `1m hit-times` | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | Implemented in R6-03 |
| Best-cell verification replay | `exact replay of best TP/SL cell` only after fast search converges | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | Implemented in R6-03 |
| Notebook summary metrics after replay | `metrics over compact trades` for ranking and final summary | `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py` | Implemented in R6-03 |

## Artifact Dependencies By Stage

| Stage | Required inputs | Produced contract |
|---|---|---|
| Stage A | `prices/<signal_tf>/*`, `signals/<signal_tf>/<indicator_id>/signals.i8.npy`, `mappings/<signal_tf>/bar_close_1m_idx.u32.npy`, optional `signal_features/<signal_tf>/<indicator_id>/features.f32.npy` warm cache | `final_signal`, deterministic edges, `compact trade list`, shortlist-ready no-risk summaries |
| Stage B | Stage A `compact trade list`, `prices/1m/*`, `hit_times/1m/manifest.yaml`, `hit_times/1m/*.npy` | best TP/SL cell, exact replay of best TP/SL cell, final `metrics over compact trades` |

## R6-01 / R6-03 implemented boundary

R6-01 закрывает runtime bootstrap/loaders boundary, R6-02 добавляет Stage A kernels и
artifact-backed shortlist bridge, а R6-03 добавляет Stage B risk kernels поверх shipped
`1m hit-times` и additive artifact-backed scorer bridge для sync/background runtime.

Что уже зафиксировано кодом:

- один `slot-pinned context` с полями `artifact_slot`, `slot_generation`,
  `artifact_asof_date`, `artifact_manifest_hash`;
- sync path резолвит active slot из strict `current.yaml`;
- background path резолвит тот же identity shape из persisted job pin metadata;
- оба path читают один и тот же root `manifest.yaml` и downstream manifests без directory
  scanning;
- arrays открываются только через `np.load(..., mmap_mode='r')` и `allow_pickle=False`;
- runtime fail-fast reject'ит drift по `path`, `dtype`, `shape`, `axis_order`, `timeline`,
  `slot_generation`, `asof_date`.
- Stage A runtime может работать по `artifacts-only inputs` через
  `BacktestStageAShortlistBuilderV2`, не переоткрывая artifact identity ad hoc;
- subset row loading для `signals/<tf>/<indicator_id>/signals.i8.npy` используется по
  выбранным variant rows, а не через full matrix materialization;
- runtime loaders могут переиспользовать уже валидированные mmap payloads для
  `prices/<tf>`, `mappings/<tf>`, `hit_times/1m` и `signals/<tf>/<indicator_id>` внутри одного
  pinned run вместо повторного `np.load(...)` на каждый internal call;
- additive `signal_features/<tf>/<indicator_id>` могут открываться тем же pinned runtime как
  optional warm-cache surface для future hybrid/plugin work, но exact `exact_small` /
  `exact_parallel` path не должен использовать их для shortlist pruning или score drift;
- D1 foundation modules
  `generic_row_scorer_v2.py` и `diversified_retention_v2.py`
  могут использовать этот warm-cache surface вместе с cheap row-local runtime stats, но остаются
  reusable detached primitives и не встраиваются в live exact runtime path до следующего
  milestone;
- contiguous explicit signal row tuples могут быть internal-normalized до slice-view, если это
  не меняет deterministic row ordering и subset semantics;
- `chunked variant processing` обязано давать тот же shortlist result, что и non-chunked path.
- `risk_exit_kernel_1m.py` резолвит one-trade exits по `entry_exec_idx`,
  `sig_exit_exec_idx`, `sentinel_index` и shipped `1m hit-times`;
- fast TP/SL search использует monotone / diff-buffer decomposition и не делает naive full replay
  для каждой ячейки;
- `exact replay of best TP/SL cell` ограничен только выбранной winning cell;
- `metrics_kernel.py` считает deterministic Stage B ranking/summary fields и строит
  details-compatible outcome только для retained exact replay;
- exact Stage B runner может stream'ить task enumeration в canonical order и не обязан
  materialize'ить весь intermediate task tuple заранее, если winner ordering и `variant_key`
  semantics не меняются;
- artifact-backed Stage B scorer после R10-01 является mandatory production runtime contract для
  sync launch, claimed background execution и run-scoped lazy detail при валидном
  `slot-pinned context`;
- production orchestration больше не возвращается к legacy close-fill fallback и не строит live
  candle timelines через ClickHouse.

## Milestone B exact acceleration note

B1 оставляет финальную exact semantics неизменной и ограничивается in-process exact-core
оптимизациями без request classification.

B2 активирует executable `exact_parallel` semantics для уже resolved execution profiles:

- Stage B может исполняться через `spawn`-based worker processes;
- каждый worker заново открывает pinned artifacts readonly через mmap и не делит mutable runtime
  state с coordinator process;
- coordinator merge происходит в canonical chunk order, поэтому completion order worker'ов не
  влияет на winners, persisted ordering или checkpoint frontier;
- `exact_small` остаётся serial exact path;
- active runtime default по-прежнему не меняется автоматически и request classification для
  `exact_parallel` остаётся вне scope до следующего EPIC.

Разрешённые exact-only internal optimizations для B1/B2:

- reuse deterministic row/array plans across repeated `compute_index` / `signal_index` groups;
- reuse already validated mmap payloads inside one pinned runtime instance;
- process-parallel Stage B только для уже resolved profiles с
  `feature_flags.parallel_stage_b_enabled = true`;
- использовать existing `exact_baseline` и `small_grid_overhead` benchmark vocabulary как
  evidence surface, не меняя active runtime default и не conflating `exact_baseline` anchor with
  rollout policy.

## P2 conservative parallelism tuning

После того как `stage_a_workers` стал реальным runtime knob, production-style tuning фиксируется
как явный contract, а не как скрытая надежда на глобальный `numba.set_num_threads(...)`.

- `backtest.cpu.max_numba_threads=4` остаётся общим ceiling для `dev`, `test`, и `prod`, чтобы
  Stage A не oversubscribe'ил типичные 4-vCPU среды по умолчанию;
- `exact_small` остаётся serial profile: `stage_a_workers=1`, `stage_b_workers=1`;
- `exact_parallel` использует полный ceiling для breadth/exact workload:
  `stage_a_workers=4`, `stage_b_workers=4`;
- `hybrid_conservative` отдаёт Stage A полный ceiling, но держит Stage B чуть уже после
  shortlist narrowing: `stage_a_workers=4`, `stage_b_workers=3`;
- `hybrid_family` остаётся самым узким shipped hybrid profile:
  `stage_a_workers=3`, `stage_b_workers=2`.

Эти значения выбраны консервативно: Stage A теперь действительно масштабируется по профилю, но
hybrid Stage B получает уже narrowed frontier, поэтому worker counts intentionally stay below the
`exact_parallel` cap вместо агрессивного max-out по всем режимам.

## Milestone D hybrid shortlist note

Milestone D добавляет первый approximate runtime path, но не меняет canonical default:

- `generic_row_scorer_v2.py` и `diversified_retention_v2.py` дают reusable universal
  foundation primitives;
- `hierarchical_shortlist_builder_v2.py` остаётся отдельным module и не смешивает hybrid
  orchestration с `stage_a_shortlist_builder_v2.py`;
- hybrid path разрешён только для explicitly opted-in
  `execution_profile.mode = hybrid_conservative`;
- public launch routing не получает новый selector и по-прежнему остаётся exact-first;
- final scoring authority остаётся за existing exact Stage B scorer.

Практическая граница:

- exact Stage A builder по-прежнему canonical для `exact_small` и `exact_parallel`;
- hybrid runtime может уменьшать Stage A / Stage B candidate space только до передачи
  survivors в exact scorer;
- benchmark gates для `top_1_recall`, `top_10_overlap`, `low_activity`,
  `high_correlation`, `small_grid_overhead`, `memory_footprint` описаны в
  `docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md`.

Что остаётся вне scope после R10-01:

- EPIC R10-02 full docs synchronization scope;
- EPIC R10-03 perf benchmark closure и runbook expansion.

## Stage A Contract

Stage A существует для batch-oriented работы на `signal timeline` без risk replay по каждой ячейке.

Обязательные обязанности:

1. Загрузить deterministic subset signal rows по уже выбранным variant keys.
2. Собрать один `final_signal` на request timeframe без pair-specific notebook prefilters.
3. Выделить входы/выходы стратегии из `final_signal`.
4. Смаппить каждый signal entry в `execution timeline` через local `bar_close_1m_idx`.
5. Построить `compact trade list` без TP/SL replay.
6. Посчитать deterministic no-risk metrics для shortlist/ranking без Stage B risk kernels.
7. Поддерживать `chunked variant processing` без drift относительно reference path.

Обязательные правила:

- `signal timeline` и `execution timeline` считаются разными концептами даже тогда, когда в
  research notebook они временно совпадали.
- В artifact-backed runtime request TF остаётся `signal timeline`, а `1m` остаётся
  `execution timeline`.
- `signal_aggregator_kernel.py` использует explicit consensus AND policy:
  long только когда все выбранные indicator rows дают `+1`,
  short только когда все выбранные indicator rows дают `-1`,
  иначе `final_signal = 0`.
- Повторное подтверждение в той же стороне не создаёт новую сделку.
- Противоположное подтверждение закрывает текущую сделку по `sig_exit_exec` и сразу открывает
  новую.
- Незакрытая до конца позиция получает `sig_exit_exec = sentinel_index`.
- В `long-only` и `short-only` режимах запрещённый противоположный сигнал работает только как
  signal exit и не открывает новую позицию.
- Для shortlist ordering tie-break должен быть explicit и stable:
  ranking payload сортируется детерминированно, а при полном равенстве метрик сохраняется
  `base_variant_key ASC`.

## R10-01 production hot-path cutover

R10-01 закрывает production reachability legacy hot-path orchestrators для покрытого scope:

- `src/trading/contexts/backtest/application/use_cases/run_backtest.py` больше не исполняет
  sync launch и lazy detail через `candle_timeline_builder.py` или `staged_runner_v1.py`;
- `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py` больше не
  исполняет claimed background runs через `grid_builder_v1.py` или `staged_core_runner_v1.py`;
- production paths используют только
  `artifact_runtime_timeline_v2.py`, `artifact_runtime_plan_v2.py`,
  `stage_a_shortlist_builder_v2.py`, `artifact_runtime_core_v2.py`,
  `artifact_backed_stage_b_scorer_v2.py`;
- legacy v1 modules остаются compatibility-only для stable imports и non-production migration
  boundaries, но не являются implicit runtime fallback.

## R8-01 worker cutover

R8-01 не меняет kernel semantics, но меняет orchestration boundary для claimed background runs:

- `RunBacktestJobRunnerV1` стартует только из persisted pin metadata и вызывает
  `resolve_pinned_context(...)` вместо live `current.yaml` discovery;
- worker использует pinned `prices/<signal_tf>` для warmup-aware grid guard candles и не строит
  live candle timeline через ClickHouse;
- worker hot path не вызывает `IndicatorCompute.compute(...)`; artifact-backed Stage A / Stage B
  читают только shipped signals, mappings, `prices/1m` и `1m hit-times`;
- lifecycle/persistence contract не меняется: `queued -> running -> succeeded|failed|cancelled`,
  snapshots и terminal rows остаются summary-only (`report_table_md=NULL`, `trades_json=NULL`).

## Stage B Contract

Stage B добавляет risk execution поверх Stage A output и использует только shipped R5-01
artifacts, а не runtime recompute.

Обязательные обязанности:

1. Использовать `compact trade list` как единственный вход trade-state.
2. Выполнить `fast TP/SL grid search` поверх `1m hit-times`.
3. Найти лучшую TP/SL ячейку без полного replay всего grid.
4. Выполнить `exact replay of best TP/SL cell`.
5. Посчитать финальные `metrics over compact trades`.

R6-03 shipped boundary:

- `TP/SL lookup starts at entry_exec + 1` явно enforced в runtime kernel;
- `signal exit wins on equal bar`;
- `SL wins TP tie`;
- Stage B runtime remains `grid-agnostic`: local kernels read `tp_values`, `sl_values`, and
  table widths from shipped `hit_times/1m` artifact arrays/manifests instead of fixed table
  literals;
- `close_on_end = 1` остаётся explicit notebook-derived default;
- ranking hot path может использовать fast `total_return_pct` lookup только для pinned
  artifact-backed Stage B scorer и только при primary=`total_return_pct`, secondary=`None`;
- любой другой Stage B ranking/details path делает exact replay только для уже выбранной explicit
  cell / retained variant.

Канонический Stage B flow:

```text
load stage_a_output
  -> map entries to execution timeline
  -> fast TP/SL grid search on 1m hit-times
  -> exact replay of best TP/SL cell
  -> metrics over compact trades
```

## Детерминированные boundary rules

### Entry mapping

- Generic rule: сделка открывается на первом execution bar строго после закрытия signal bar.
- Artifact-backed form: `entry_exec = bar_close_1m_idx + 1`.
- Если индекс вышел за границу, используется `sentinel_index == T_exec`.

### Exit precedence

- `1m hit-times` tables themselves remain same-bar-inclusive lookup artifacts; runtime chooses
  the lookup start explicitly.
- TP/SL lookup starts at `entry_exec + 1`.
- `signal exit wins on equal bar`.
- `SL wins TP tie`.

### Additional runtime rules

- `sig_exit_exec` — это execution index следующего противоположного подтверждения, а не signal
  bar index.
- Если TP/SL происходит раньше `sig_exit_exec`, сделка закрывается по precomputed factor.
- Если TP/SL не произошло, а `sig_exit_exec < sentinel_index`, сделка закрывается по signal exit.
- Если ни TP/SL, ни signal exit не произошли, runtime использует `close_on_end = 1` как
  notebook-derived default для Stage B.

## Module-Level Boundaries For R6

### `signal_aggregator_kernel.py`

- Вход: subset-loaded signal rows по выбранным variants и deterministic aggregation policy.
- Выход: `final_signal[V, T_signal]` c value set `{-1, 0, 1}`.
- Каноническая функция:
  - `aggregate_final_signal_rows_v2(selected_signal_rows=...)`
- Deterministic ordering:
  - indicator matrices обходятся в sorted order по `indicator_id`;
  - shape drift и invalid signal values fail-fast reject'ятся до hot loop.
- Не должен:
  - читать файлы;
  - знать о TP/SL grid;
  - переносить notebook pair-specific heuristics.

### `trade_compactor_kernel.py`

- Вход: `final_signal`, `bar_close_1m_idx`, `sentinel_index`.
- Выход: `compact trade list` с полями `entry_exec_idx`, `direction`, `sig_exit_exec_idx`.
- Канонические функции:
  - `build_compact_trade_list_v2(...)`
  - `compute_no_risk_metrics_v2(...)`
  - `no_risk_metrics_to_ranking_payload_v2(...)`
- `entry_exec_idx` вычисляется как `bar_close_1m_idx + 1` с sentinel fallback.
- `sig_exit_exec_idx` равен execution index противоположного подтверждения либо
  `sentinel_index`, если signal exit не наступил.
- No-risk metric contract для shortlist включает:
  - `total_return_pct`
  - `max_drawdown_pct`
  - `return_over_max_drawdown`
  - `profit_factor`
  - `sharpe_trades`
  - `trade_count`
  - `win_rate_pct`
  - `avg_trade_ret_pct`
  - `avg_trade_exec_bars`
  - `exposure_pct`
- D1 generic shortlist foundation поверх этих metrics может публиковать separate row-score
  explanation payload и diversity buckets, но не меняет canonical exact ordering/winners до
  explicit rollout wiring milestone.
- Не должен:
  - делать risk replay;
  - зависеть от `1m hit-times`;
  - менять ordering variants/trades недетерминированно.

### `risk_exit_kernel_1m.py`

- Вход: `compact trade list`, `prices/1m`, `1m hit-times`, TP/SL grids.
- Выход: best cell indices, exact exit facts, replay payload for metrics.
- Не должен:
  - recompute `1m hit-times`;
  - зависеть от notebook file layout;
  - становиться orchestration facade для всего runtime.

### `metrics_kernel.py`

- Вход: exact replay payload по лучшей ячейке и deterministic trade ordering.
- Выход: ranking fields и финальные summary metrics.
- Approved runtime ranking literals:
  - `total_return_pct DESC`
  - `max_drawdown_pct ASC`
  - `return_over_max_drawdown DESC`
  - `profit_factor DESC`
  - `sharpe_trades DESC`
  - `win_rate_pct DESC`
- Deterministic final ordering для retained rows обязан оставаться
  `ranking metrics -> variant_key ASC`.
- Не должен:
  - делать DataFrame/report formatting;
  - materialize full trade bodies для всех variants;
  - переопределять exit semantics.

## Что явно не переносится из notebook

- pair-specific prefilters `top_frac_side`, `min_confirm`, `top_frac_pairs`;
- research-only staged ranking таблицы и exploratory DataFrame outputs;
- literal `prices_and_signals_5m.npy` layout как production storage contract;
- жёстко пришитые `signal_tf=1h` и `exec_tf=5m`;
- notebook control flow с environment flags, plotting и ad-hoc self-check cells.

Иными словами, production runtime переносит reusable kernel boundaries, но не переносит notebook
как orchestration script.

## Relationship To R5-01 And R5-03

- R5-01 уже shipped и materialize'ит strict `1m hit-times`, на которые обязан опираться этот
  contract.
- R5-02 фиксирует production transfer semantics и boundaries для R6 implementation.
- R5-03 остаётся отдельным milestone для golden fixtures и не подменяет runtime implementation,
  но теперь публикует executable validation baseline:
  - unit fixture catalog:
    `tests/unit/contexts/backtest/application/services/v2/fixtures/stage_b_golden_fixtures_v2.json`
  - perf-smoke reference manifest:
    `tests/perf_smoke/contexts/backtest/fixtures/r5_stage_b_golden_cases.json`
  - executable contract tests:
    `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`
- Эти golden fixtures фиксируют `entry mapping request TF -> 1m`, `TP/SL earliest hit`,
  `earliest signal-exit mapping`, tie-break rules, `exact best-cell replay` и
  `metrics over compact trades` без notebook execution в CI.

## R5-03 Verification Baseline

Для будущих R6 kernels canonical verification path теперь такой:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
```

Важно:

- `tests/perf_smoke/contexts/backtest/fixtures/r0_parity_scope.json` сохраняет
  `stage_b_signal_tf_1m_risk_reference` как `reference-only` R0 marker;
- отдельный `r5_stage_b_golden_cases.json` делает change explicit и version-controlled;
- runtime/API/storage contracts по-прежнему не меняются до R6 cutover.
