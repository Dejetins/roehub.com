# Backtest Runtime Acceleration Benchmarks v1

Статус: active benchmark corpus for Milestone A / EPIC A3  
Область: `backtest` runtime follow-up after A1/A2  
Связанные документы:
- `docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md`
- `docs/architecture/backtest/backtest-v2-benchmarks.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
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
- Stage B golden alignment:
  `tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py`

## Slice catalog

| `slice_id` | Baseline `execution_profile_mode` | Candidate profile | Rollout scope | Stage focus | Main evidence |
|---|---|---|---|---|---|
| `exact_baseline` | `exact_parallel` | none | `exact_only` | `stage_a`, `stage_b`, `finalizing` | existing `r0_benchmark_scenarios.json` + `r5_stage_b_golden_cases.json` |
| `low_activity` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b` | sparse synthetic slice + `earliest_signal_exit_mapping` anchor |
| `high_correlation` | `exact_parallel` | `hybrid_family` | `plugin_rollout` | `stage_a`, `stage_b` | correlated-family synthetic slice + exact best-cell replay anchor |
| `small_grid_overhead` | `exact_small` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b` | lightweight sync-sized synthetic slice consumed by staged-runner perf smoke |
| `memory_footprint` | `exact_parallel` | `hybrid_conservative` | `hybrid_rollout` | `stage_a`, `stage_b`, `finalizing` | wide retained-survivor synthetic slice anchored to background-sized baseline |

## Как корпус используется сейчас

- `exact_baseline` переиспользует approved R0/R5 baseline вместо создания второго exact reference.
- `small_grid_overhead` уже читает committed corpus metadata из
  `test_backtest_staged_runner_perf_smoke.py`, поэтому small sync harness больше не держит
  отдельный hardcoded shape.
- `test_stage_b_golden_fixtures_v2.py` проверяет, что corpus не расходится с canonical Stage B
  case order.
- `test_backtest_r0_baseline_perf_smoke.py` проверяет byte-stable serialization и completeness
  всего benchmark corpus.

## Что корпус пока намеренно не делает

- не вводит heuristic shortlist logic;
- не вводит family plugin execution;
- не вводит benchmark pass/fail thresholds по абсолютным wall-clock numbers;
- не меняет public runtime API или persisted storage contracts.

То есть текущий CI проверяет shape, ordering, fixture linkage и уже существующие zero-call gates,
а не rollout thresholds для будущих hybrid/plugin paths.

## Repro protocol

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_stage_b_golden_fixtures_v2.py \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py \
  tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
```

Если future milestone добавляет rollout gates по recall / overlap / diversity / memory, он должен:

1. переиспользовать существующие `slice_id`;
2. расширять corpus additive fields, а не вводить новый ad-hoc benchmark set;
3. оставлять `exact_baseline` каноническим source of truth для final scoring comparison.
