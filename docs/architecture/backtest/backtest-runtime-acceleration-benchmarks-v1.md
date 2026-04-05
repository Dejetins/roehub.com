# Backtest Runtime Acceleration Benchmarks v1

Статус: active benchmark corpus for Milestone A / EPIC A3 and later exact + hybrid rollout milestones  
Область: `backtest` runtime follow-up after A1/A2  
Связанные документы:
- `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`

## Зачем нужен этот документ

A3 не меняет production runtime behavior.
Он добавляет один явный benchmark corpus, чтобы следующие milestone сравнивали exact, hybrid и
plugin rollout на одном и том же наборе deterministic slices.

Корпус обязан говорить на уже зафиксированном словаре:

- `execution_profile_mode`
- `stage_a`, `stage_b`, `finalizing`
- `exact_baseline`
- lightweight harness без hard machine-dependent thresholds

## Канонические артефакты

- corpus fixture:
  `tests/perf_smoke/contexts/backtest/fixtures/backtest_runtime_acceleration_benchmark_corpus_v1.json`
- typed loader / validator:
  `src/trading/contexts/backtest/application/services/v2/benchmark_corpus_v2.py`
- exact baseline / zero-call perf gates:
  `tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py`
- small-grid lightweight harness:
  `tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py`
- hybrid rollout gates:
  `tests/perf_smoke/contexts/backtest/test_backtest_hybrid_shortlist_rollout_v2.py`
- Stage B golden alignment:
  `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`

## Slice catalog

| `slice_id` | Baseline `execution_profile_mode` | Candidate profile | Rollout scope | Stage focus | Main evidence |
|---|---|---|---|---|---|
| `exact_baseline` | `exact_parallel` | none | `exact_only` | `stage_a`, `stage_b`, `finalizing` | existing `r0_benchmark_scenarios.json` + `r5_stage_b_golden_cases.json` |
| `low_activity` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b` | sparse synthetic slice + `earliest_signal_exit_mapping` anchor |
| `high_correlation` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b` | correlation-sensitive synthetic slice + exact best-cell replay anchor |
| `small_grid_overhead` | `exact_small` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b` | lightweight sync-sized synthetic slice consumed by staged-runner perf smoke |
| `memory_footprint` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b`, `finalizing` | wide retained-survivor synthetic slice anchored to background-sized baseline |
| `medium_grids` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b`, `finalizing` | roadmap explicit medium-grid slice reusing `large-run` plus benchmark ETA fallback envelope |
| `huge_grids` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b`, `finalizing` | roadmap explicit huge-grid slice reusing `background-run` plus benchmark ETA fallback envelope |
| `multi_block` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b`, `finalizing` | additive synthetic multi-block workload slice used for reviewable rollout evidence and ETA fallback |

## Как корпус используется сейчас

- `exact_baseline` переиспользует approved R0/R5 baseline вместо создания второго exact reference.
- `exact_baseline=exact_parallel` остаётся benchmark evidence anchor и не должен silently
  подменять runtime `default_execution_profile=exact_small`, который используется launch/runtime
  defaults в production B3 classification.
- `small_grid_overhead` уже читает committed corpus metadata из
  `test_backtest_staged_runner_perf_smoke.py`, поэтому small sync harness больше не держит
  отдельный hardcoded shape.
- `medium_grids`, `huge_grids`, и `multi_block` закрывают недостающие roadmap edge slices
  additively, не меняя старые `slice_id` и не создавая второй benchmark corpus.
- ETA fallback для persisted runs history теперь использует тот же committed corpus через
  startup-loaded typed metadata (`eta_fallback`), а не request-path fixture reads.
- `test_backtest_hybrid_shortlist_rollout_v2.py` читает те же `slice_id` и
  `rollout_gates`, поэтому hybrid rollout evidence не создаёт второй ad-hoc benchmark set.
- `test_backtest_adaptive_selector_rollout_v2.py` связывает committed env rollout config с теми
  же evidence anchors: `small_grid_overhead` защищает `exact_small` / small sync runs, а
  `memory_footprint` и existing hybrid/plugin rollout tests оправдывают только selective default
  for large runs.
- `test_stage_b_golden_fixtures_v2.py` проверяет, что corpus не расходится с canonical Stage B
  case order.
- `test_backtest_r0_baseline_perf_smoke.py` проверяет byte-stable serialization и completeness
  всего benchmark corpus.

## F2 rollout mapping

F2 intentionally keeps benchmark evidence, selector policy, and active defaults separate:

- `exact_baseline=exact_parallel` остаётся evidence anchor и не становится runtime default.
- `small_grid_overhead` объясняет, почему даже при `adaptive selector=active` small sync runs
  должны оставаться `exact_small`.
- `medium_grids` и `huge_grids` делают roadmap buckets explicit without conflating the benchmark
  anchor with the active runtime default exact profile.
- `multi_block` keeps multi-block strategy evidence explicit and machine-readable instead of
  relying on implicit reuse of unrelated large-run slices.
- `test_backtest_hybrid_shortlist_rollout_v2.py` + `memory_footprint` дают evidence surface для
  `hybrid_conservative` selective default only on large runs.
- `test_backtest_family_plugin_rollout_v2.py` доказывает proposal-layer viability для pure
  `ma.` requests, но этого пока недостаточно для unconditional live default; поэтому
  `hybrid_family` остаётся narrower rollout path (`shadow` by default in committed prod config).

То есть benchmark corpus по-прежнему не является runtime selector input.
Он только фиксирует reviewable evidence, по которому env config может безопасно оставаться
`shadow`, перейти в `opt_in`, затем в `active`, или откатиться обратно.

## Что корпус сейчас намеренно не делает

- не включает automatic hybrid selection для обычных launch requests;
- не включает `hybrid_family` / family plugin execution;
- не вводит benchmark pass/fail thresholds по абсолютным machine-dependent wall-clock numbers;
- не меняет active production default exact profile и не управляет public `POST /backtests`
  routing;
- не меняет public runtime API или persisted storage contracts.

Milestone D note:

- foundation-only modules `generic_row_scorer_v2.py` и `diversified_retention_v2.py` могут уже
  использовать vocabulary slices `low_activity` и `high_correlation` для explanation / diversity
  benchmarking;
- `hierarchical_shortlist_builder_v2.py` и
  `test_backtest_hybrid_shortlist_rollout_v2.py` теперь используют этот же corpus для explicit
  hybrid rollout gates;
- сам corpus по-прежнему не является launch selector и не трактует наличие shortlist knobs как
  автоматическую rollout activation.

То есть текущий CI проверяет:

- shape, ordering и fixture linkage;
- existing exact baseline anchors;
- explicit hybrid rollout gates для `hybrid_conservative`;
- но не превращает эти gates в automatic production routing policy.

## Repro protocol

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py \
  tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
```

Если future milestone расширяет rollout gates или candidate profiles, он должен:

1. переиспользовать существующие `slice_id`;
2. расширять corpus additive fields, а не вводить новый ad-hoc benchmark set;
3. оставлять `exact_baseline` каноническим source of truth для final scoring comparison.
