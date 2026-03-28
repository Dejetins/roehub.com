# Backtest Runtime Kernels V2 (R5-02 contract for `signal_tf + 1m_risk`)

Этот документ фиксирует канонический production contract для Stage A / Stage B runtime kernels,
который переносит notebook-derived kernel semantics из
`tests/notebook_tests/06_backtest_compute.ipynb` в generic runtime boundaries, не меняя shipped
R5-01 artifact contracts.

Статус: `Milestone R5 / EPIC R5-02`  
Следующий этап реализации: `Milestone R6`

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
| Pair confirmations on request TF | Deterministic signal aggregation on `signal timeline` with output value set `{-1, 0, 1}` | `src/trading/contexts/backtest/application/services/v2/signal_aggregator_kernel.py` | Planned for R6 |
| `build_trade_list_for_pair` | `compact trade list` with deterministic ordering and sentinel-based signal exits | `src/trading/contexts/backtest/application/services/v2/trade_compactor_kernel.py` | Planned for R6 |
| `evaluate_trade_factor` over hit tables | `1m hit-times` risk-exit resolution on `execution timeline` | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | R5-01 input implemented, R6 kernel planned |
| Monotone diff-buffer decomposition | `fast TP/SL grid search` over precomputed `1m hit-times` | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | R5-01 input implemented, R6 kernel planned |
| Best-cell verification replay | `exact replay of best TP/SL cell` only after fast search converges | `src/trading/contexts/backtest/application/services/v2/risk_exit_kernel_1m.py` | Planned for R6 |
| Notebook summary metrics after replay | `metrics over compact trades` for ranking and final summary | `src/trading/contexts/backtest/application/services/v2/metrics_kernel.py` | Planned for R6 |

## Artifact Dependencies By Stage

| Stage | Required inputs | Produced contract |
|---|---|---|
| Stage A | `prices/<signal_tf>/*`, `signals/<signal_tf>/<indicator_id>/signals.i8.npy`, `mappings/<signal_tf>/bar_close_1m_idx.u32.npy` | `final_signal`, deterministic edges, `compact trade list`, shortlist-ready no-risk summaries |
| Stage B | Stage A `compact trade list`, `prices/1m/*`, `hit_times/1m/manifest.yaml`, `hit_times/1m/*.npy` | best TP/SL cell, exact replay of best TP/SL cell, final `metrics over compact trades` |

## Stage A Contract

Stage A существует для batch-oriented работы на `signal timeline` без risk replay по каждой ячейке.

Обязательные обязанности:

1. Загрузить deterministic subset signal rows по уже выбранным variant keys.
2. Собрать один `final_signal` на request timeframe без pair-specific notebook prefilters.
3. Выделить входы/выходы стратегии из `final_signal`.
4. Смаппить каждый signal entry в `execution timeline`.
5. Построить `compact trade list` без TP/SL replay.

Обязательные правила:

- `signal timeline` и `execution timeline` считаются разными концептами даже тогда, когда в
  research notebook они временно совпадали.
- В artifact-backed runtime request TF остаётся `signal timeline`, а `1m` остаётся
  `execution timeline`.
- Повторное подтверждение в той же стороне не создаёт новую сделку.
- Противоположное подтверждение закрывает текущую сделку по `sig_exit_exec` и сразу открывает
  новую.
- Незакрытая до конца позиция получает `sig_exit_exec = sentinel_index`.

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

- Вход: signal rows по выбранным variants и deterministic aggregation policy.
- Выход: `final_signal[V, T_signal]` c value set `{-1, 0, 1}`.
- Не должен:
  - читать файлы;
  - знать о TP/SL grid;
  - переносить notebook pair-specific heuristics.

### `trade_compactor_kernel.py`

- Вход: `final_signal`, `bar_close_1m_idx`, `sentinel_index`.
- Выход: `compact trade list` с полями `entry_exec_idx`, `direction`, `sig_exit_exec_idx`.
- Не должен:
  - делать risk replay;
  - считать финальные metrics;
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
- R5-03 остаётся отдельным milestone для golden fixtures и не подменяется этим документом.
