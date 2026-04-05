# Backtest Hybrid Shortlist Runtime v1

Документ фиксирует первый approximate runtime path для `Milestone D`:
универсальный conservative shortlist, который уменьшает Stage A / Stage B workload, но остаётся
строго opt-in и не меняет canonical exact default.

## Status

- Status: active `Milestone D / EPIC D2+D3` architecture contract.
- Scope:
  - `hybrid_conservative` only;
  - universal row scoring + diversified retention + hierarchical shortlist;
  - explicit rollout gates against exact baseline;
  - exact Stage B scorer remains the final source of truth.
- Non-goals:
  - no public `POST /backtests` field for profile selection;
  - no automatic hybrid routing for ordinary requests;
  - no `hybrid_family` / family-plugin behavior here;
  - no adaptive selector in this document.

Связанные документы:

- `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-runtime-acceleration-benchmarks-v1.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
- `docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md`

## Зачем нужен hybrid path

Exact runtime остаётся каноническим, но на больших grids он дорог:

- приходится просматривать много row candidates в каждом indicator block;
- затем строить слишком широкое Stage A / Stage B enumeration space;
- и только в конце exact scorer отбрасывает очевидно слабые комбинации.

`hybrid_conservative` добавляет отдельный pre-exact path:

1. cheap row scoring,
2. diversified per-block retain,
3. hierarchical combine,
4. exact Stage B scoring only for survivors.

Идея не в том, чтобы заменить exact scorer, а в том, чтобы уменьшить объём работы до него.

## Execution boundary

Hybrid runtime может исполняться только когда одновременно выполнены все условия:

- resolved `execution_profile.mode == "hybrid_conservative"`;
- `feature_flags.runtime_enabled = true`;
- `feature_flags.heuristic_shortlist_enabled = true`.

Важно:

- public `POST /backtests` contract не получает нового поля;
- browser не выбирает profile самостоятельно;
- default server routing остаётся exact-first;
- hybrid path разрешён только через internal-only `execution_profile_mode` override,
  который не входит в request-hash semantics и нужен для tests/manual wiring/persisted internal
  metadata.

## Runtime flow

### 1. Row scoring

Модуль:

- `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`

Для каждого candidate row scorer строит deterministic payload из:

- cached `signal_features`, если они есть;
- cheap row-local runtime stats, если feature artifact отсутствует;
- explicit score breakdown:
  - `activity_ratio`
  - `direction_balance`
  - `transition_count`
  - `active_span_ratio`

### 2. Diversified retention

Модуль:

- `src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py`

Retention не является простым `top-N by raw score`.
Он удерживает survivors по explicit diversity buckets, чтобы approximate shortlist не
схлопывался в один узкий correlated cluster.

Базовые bucket axes:

- `activity_band`
- `direction_band`
- `transition_band`

### 3. Hierarchical shortlist builder

Модуль:

- `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`

Это отдельный module, а не перегрузка
`src/trading/contexts/backtest/application/services/v2/stage_a_shortlist_builder_v2.py`.

Он делает:

- per-block retained rows;
- explicit audit-friendly block results;
- hierarchical combine / reduced Stage A enumeration;
- reduced runtime plan, который всё ещё совместим с canonical exact downstream scorer.

### 4. Exact final scoring

После conservative shortlist runtime не строит новый heuristic Stage B scorer.
Он передаёт survivors в уже существующий exact path:

- `src/trading/contexts/backtest/application/services/v2/artifact_backed_stage_b_scorer_v2.py`

Следствие:

- final winners и final metrics всё ещё принадлежат canonical exact scorer;
- hybrid path меняет только candidate set до final exact evaluation.

## Relationship to exact runtime

Exact runtime остаётся главным контрактом:

- `exact_small` и `exact_parallel` остаются canonical defaults;
- `stage_a_shortlist_builder_v2.py` остаётся canonical exact Stage A builder;
- hybrid path не должен silently подменять exact path;
- если internal hybrid opt-in отсутствует, runtime должен остаться exact.

Практически это означает:

- ordinary `POST /backtests` launches продолжают идти по exact routing;
- benchmark corpus сравнивает hybrid against exact baseline, а не наоборот;
- любые rollout decisions для hybrid должны приниматься только по benchmark evidence.

## Benchmark and rollout gates

Source of truth:

- `tests/perf_smoke/contexts/backtest/fixtures/backtest_runtime_acceleration_benchmark_corpus_v1.json`
- `src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py`
- `tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py`

Обязательные gates для `hybrid_conservative`:

- `top_1_recall >= 0.99`
- `top_10_overlap >= 0.90`
- `low_activity` top-1 recall `>= 0.97`
- `high_correlation` diversity evidence:
  - minimum `2` distinct winners
- `small_grid_overhead`:
  - wall-clock ratio `<= 1.25x`
- `memory_footprint`:
  - peak traced memory ratio `<= 1.10x`

Эти thresholds не делают hybrid default.
Они только фиксируют минимальный evidence bar для safe rollout discussion.

## Files that define the current contract

- `src/trading/contexts/backtest/application/services/v2/hierarchical_shortlist_builder_v2.py`
- `src/trading/contexts/backtest/application/services/v2/generic_row_scorer_v2.py`
- `src/trading/contexts/backtest/application/services/v2/diversified_retention_v2.py`
- `src/trading/contexts/backtest/application/services/v2/execution_profile_v2.py`
- `src/trading/contexts/backtest/application/services/v2/artifact_runtime_plan_v2.py`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `tests/unit/contexts/backtest/application/services/v2/test_hierarchical_shortlist_builder_v2.py`
- `tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py`

## Invariants

- hybrid path is opt-in only;
- public API contract remains unchanged;
- internal profile metadata stays out of request-hash semantics;
- exact scorer remains final authority for survivors;
- benchmark corpus remains the common evidence surface for exact vs hybrid comparison.
