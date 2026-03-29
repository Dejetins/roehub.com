# Backtest Runtime Kernels V2 (R5-02 contract / R6-01 loader-context boundary for `signal_tf + 1m_risk`)

Этот документ фиксирует канонический production contract для Stage A / Stage B runtime kernels,
который переносит notebook-derived kernel semantics из
`tests/notebook_tests/06_backtest_compute.ipynb` в generic runtime boundaries, не меняя shipped
R5-01 artifact contracts.

Статус: `Milestone R5 / EPIC R5-02`, `Milestone R6 / EPIC R6-01 + R6-02`  
Следующие этапы реализации: `Milestone R6 / EPIC R6-03 + R6-04`

Связанные документы:

- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/backtest/backtest-precompute-runner-v2.md`
- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `tests/notebook_tests/06_backtest_compute.ipynb`
- `tests/notebook_tests/05_hit_time_grid.ipynb`

## Роль документа

- Это главный entrypoint для R5-02 и будущего R6 implementation path.
- Notebook остаётся semantics source, но production runtime трактует его как
  `notebook-derived kernel semantics` и `not a literal notebook orchestration script`.
- R5-01 остаётся immutable input boundary: runtime читает только shipped `1m hit-times`,
  `prices/<tf>`, `prices/1m`, `mappings/<tf>` и `signals/<tf>/<indicator_id>`.
- R6-01 уже реализует runtime-side artifact loading primitives:
  `artifact_slot_resolver.py`, `price_arrays_loader.py`, `signal_matrix_loader.py`.
- R6-02 уже реализует Stage A artifact-backed kernels и additive shortlist bridge:
  `signal_aggregator_kernel.py`, `trade_compactor_kernel.py`,
  `stage_a_shortlist_builder_v2.py`.
- Sync и background starts теперь обязаны делить один immutable `slot-pinned context` contract,
  а не расходиться по разным pointer/discovery paths.
- Документ не вводит новые API payloads, новые request TF или новые persisted storage contracts.

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
| `evaluate_trade_factor` over hit tables | `1m hit-times` risk-exit resolution on `execution timeline` | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | R5-01 input implemented, R6 kernel planned |
| Monotone diff-buffer decomposition | `fast TP/SL grid search` over precomputed `1m hit-times` | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | R5-01 input implemented, R6 kernel planned |
| Best-cell verification replay | `exact replay of best TP/SL cell` only after fast search converges | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | Planned for R6 |
| Notebook summary metrics after replay | `metrics over compact trades` for ranking and final summary | `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py` | Planned for R6 |

## Artifact Dependencies By Stage

| Stage | Required inputs | Produced contract |
|---|---|---|
| Stage A | `prices/<signal_tf>/*`, `signals/<signal_tf>/<indicator_id>/signals.i8.npy`, `mappings/<signal_tf>/bar_close_1m_idx.u32.npy` | `final_signal`, deterministic edges, `compact trade list`, shortlist-ready no-risk summaries |
| Stage B | Stage A `compact trade list`, `prices/1m/*`, `hit_times/1m/manifest.yaml`, `hit_times/1m/*.npy` | best TP/SL cell, exact replay of best TP/SL cell, final `metrics over compact trades` |

## R6-01 / R6-02 implemented boundary

R6-01 закрывает runtime bootstrap/loaders boundary, а R6-02 добавляет только Stage A kernels и
artifact-backed shortlist bridge.

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
- `chunked variant processing` обязано давать тот же shortlist result, что и non-chunked path.

Что остаётся вне scope после R6-02:

- Stage B risk execution kernels из R6-03;
- ranking/top-N runtime materialization из R6-04;
- full cutover с legacy scorer/execution paths на v2 runtime kernels.

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

## R6-02 shipped Stage A runtime bridge

R6-02 не заменяет весь runtime целиком. Он добавляет отдельный additive bridge:

- `src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`
  materialize'ит Stage A shortlist из `artifacts-only inputs`;
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py` и
  `src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py`
  подключают этот builder только когда есть валидный `slot-pinned context`;
- `src/trading/contexts/backtest/application/services/staged_runner_v1.py` остаётся legacy
  orchestration facade и использует v2 Stage A path additively;
- если v2 builder/context недоступен, sync/background flow продолжает использовать legacy
  Stage A scorer loop без изменения публичных imports.

## Stage B Contract

Stage B добавляет risk execution поверх Stage A output и использует только shipped R5-01
artifacts, а не runtime recompute.

Обязательные обязанности:

1. Использовать `compact trade list` как единственный вход trade-state.
2. Выполнить `fast TP/SL grid search` поверх `1m hit-times`.
3. Найти лучшую TP/SL ячейку без полного replay всего grid.
4. Выполнить `exact replay of best TP/SL cell`.
5. Посчитать финальные `metrics over compact trades`.

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
  - `trade_count`
  - `win_rate_pct`
  - `avg_trade_ret_pct`
  - `avg_trade_exec_bars`
  - `exposure_pct`
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
