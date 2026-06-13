# Stage 13 TP/SL Block Autotune: handoff report for external recommendation review

Дата отчета: 2026-06-14.

Репозиторий: `/Users/daniildegtyarev/Projects/roehub.com`.

Итоговый commit с зафиксированной evidence: `61bd720d Record rejected Stage 13 TP/SL block autotune gate`.

## 1. Короткий вывод

Stage 13 был выполнен как production gate для TP/SL full-grid cell backend, но итоговый статус: `rejected`.

Причина отказа: `no_shape_met_stage_13_service_wall_gate`.

Лучший кандидат по суммарному service wall: `64x128`, но он не прошел обязательный минимум `>=15%` ускорения на каждом TP/SL heavy workload:

- `tp_sl_grid/arity_6/long_only`: `+27.558%` против Stage 09 `64x64`.
- `tp_sl_grid/arity_6/long_short_reversal`: `-0.086%` против Stage 09 `64x64`.

Вывод: block-shape autotune улучшает `long_only`, но почти не влияет или ухудшает `long_short_reversal`. Поэтому Stage 13 не разблокирует production default для TP/SL backend и не разблокирует Stage 14.

## 2. Что проверялось

Цель Stage 13: проверить, можно ли превратить accepted opt-in Stage 09 TP/SL backend `matrix_cell_tp_sl_v1` с block shape `64x64` в production candidate за счет выбора более удачной формы cell-block.

Проверялись формы:

- `64x64` как accepted Stage 09 control.
- `128x32`.
- `32x128`.
- `128x64`.
- `64x128`.

Обязательные строки:

- `tp_sl_grid/arity_6/long_only`.
- `tp_sl_grid/arity_6/long_short_reversal`.

Сравнение по производительности делалось против Stage 09 accepted opt-in `64x64`, а не против rejected candidates.

## 3. Acceptance criteria

Stage мог быть принят только если одновременно выполнены условия:

1. Та же top-N identity/order.
2. Те же `best_tp`, `best_sl` и метрики в допустимой точности.
3. Не меняются `variant_hash`, ranking, fees/slippage, sizing, TP/SL tie-breaking.
4. Service wall улучшается минимум на `15%` против Stage 09 `64x64` на каждой обязательной строке.
5. Peak RSS не ухудшается больше чем на `10%`.
6. Benchmark проходит API-runner, parity, instrumentation, memory, scheduler, lazy cache, legacy path, dead-code и docs-drift проверки.

## 4. Evidence sources

Главная Stage 13 evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/`

Ключевые файлы:

- `benchmark_results.json`: aggregate machine-readable report.
- `benchmark_summary.md`: short human-readable report.
- `current_exact/`: current exact TP/SL control.
- `stage_09_accepted_64x64_64x64/`: Stage 09 accepted `64x64` control.
- `candidate_shape_128x32/`, `candidate_shape_32x128/`, `candidate_shape_128x64/`, `candidate_shape_64x128/`: candidate runs.
- `candidate_shape_*_rerun/`: diagnostic rerun history for runs affected by queue/backlog race.

Stage 05+12 production-default baseline для контроля текущего no-risk production default:

`docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_stage13_baseline/`

## 5. Benchmark environment

Host: `MacStudioDaniil`.

Benchmark checkout commit during remote runs: `8508592a857a2229c9c0c70922eeb9e3f44678d9`.

Remote checkout dirty state was intentionally recorded because Stage 13 harness files were copied onto the Mac Studio checkout before local preservation commit:

```text
M scripts/backtest/run_api_runner_benchmark_parity.py
?? scripts/backtest/run_stage_13_tp_sl_block_autotune_gate.py
?? tests/unit/contexts/backtest/application/services/v2/test_stage_13_tp_sl_block_autotune_gate.py
```

Runtime/env:

- Env file: `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Artifact config: `configs/prod/backtest_artifacts.yaml`.
- Artifact root: `/opt/roehub/state/backtest_artifacts/v2`.
- `NUMBA_NUM_THREADS=12`.
- `ROEHUB_BACKTEST_HEAVY_CONCURRENCY=1`.
- `ROEHUB_BACKTEST_LIGHT_CONCURRENCY=0`.
- `top_n=50`.
- `benchmark_top_k=5`.
- `vmmap` / physical-footprint observation was disabled in the hot path for clean timing.

## 6. Stage 05+12 production-default baseline

This baseline was run before Stage 13 to confirm the accepted production default state for no-risk rows:

- Stage 05 for `risk.mode=none`, arity 6.
- Stage 12 for `risk.mode=none`, arity 7.

Result: `pass`.

Parity: `4/4`.

Performance ratios against May 2 reference timings:

| Job | Current exact s | May2/reference s | Ratio |
| --- | ---: | ---: | ---: |
| `none/arity_6/long_only` | `1.054` | `15.694` | `14.890x` |
| `none/arity_6/long_short_reversal` | `3.086` | `15.365` | `4.980x` |
| `none/arity_7/long_only` | `5.268` | `138.755` | `26.337x` |
| `none/arity_7/long_short_reversal` | `17.685` | `136.630` | `7.726x` |

This baseline is not the main TP/SL comparison target. It only confirms the checkout/runtime contains the accepted Stage 05+12 production-default code before TP/SL Stage 13 evidence is interpreted.

## 7. Stage 13 aggregate decision

Aggregate decision:

```json
{
  "status": "rejected",
  "reason": "no_shape_met_stage_13_service_wall_gate",
  "controls_pass": true,
  "missing_payloads": [],
  "crashed_runs": [],
  "best_shape": {
    "shape": "64x128",
    "run_pass": true,
    "accepted": false,
    "total_service_wall_s": 37.43682479101699,
    "min_service_wall_improvement_vs_stage_09": -0.0008608471383446322,
    "max_memory_peak_regression_vs_stage_09": -0.1787037037037037
  }
}
```

Interpretation:

- Controls passed.
- No missing payloads.
- No crashed runs.
- All candidate runs completed.
- The best total service wall shape was `64x128`.
- It was rejected because the minimum per-row improvement was negative on reversal.

## 8. Shape matrix

| Shape | Job | Service wall s | vs Stage 09 | vs current exact | Peak RSS bytes | Memory vs Stage 09 | Top parity | Trade-cell evals/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `64x64` | `tp_sl_grid/arity_6/long_only` | `27.060185` | `0.000%` | `30.009%` | `775028736` | `0.000%` | pass | `4018766` |
| `64x64` | `tp_sl_grid/arity_6/long_short_reversal` | `17.818542` | `0.000%` | `-14.231%` | `619315200` | `0.000%` | pass | `5913590` |
| `128x32` | `tp_sl_grid/arity_6/long_only` | `20.399236` | `24.615%` | `47.238%` | `652214272` | `-15.846%` | pass | `5436983` |
| `128x32` | `tp_sl_grid/arity_6/long_short_reversal` | `17.816628` | `0.011%` | `-14.219%` | `642318336` | `3.714%` | pass | `5914425` |
| `32x128` | `tp_sl_grid/arity_6/long_only` | `20.303614` | `24.969%` | `47.485%` | `485343232` | `-37.377%` | pass | `5461722` |
| `32x128` | `tp_sl_grid/arity_6/long_short_reversal` | `17.879099` | `-0.340%` | `-14.620%` | `622247936` | `0.474%` | pass | `5892723` |
| `128x64` | `tp_sl_grid/arity_6/long_only` | `19.990900` | `26.124%` | `48.294%` | `626966528` | `-19.104%` | pass | `5586523` |
| `128x64` | `tp_sl_grid/arity_6/long_short_reversal` | `18.153907` | `-1.882%` | `-16.381%` | `486326272` | `-21.474%` | pass | `5818388` |
| `64x128` | `tp_sl_grid/arity_6/long_only` | `19.602944` | `27.558%` | `49.297%` | `491175936` | `-36.625%` | pass | `5683400` |
| `64x128` | `tp_sl_grid/arity_6/long_short_reversal` | `17.833881` | `-0.086%` | `-14.330%` | `508641280` | `-17.870%` | pass | `5911723` |

## 9. Current exact vs Stage 09 observation

Current exact control service wall:

- `long_only`: `38.662587s`.
- `long_short_reversal`: `15.598649s`.

Stage 09 accepted `64x64` service wall:

- `long_only`: `27.060185s`.
- `long_short_reversal`: `17.818542s`.

Important observation:

- Stage 09 `64x64` is faster than current exact on `long_only` by about `30.009%`.
- Stage 09 `64x64` is slower than current exact on `long_short_reversal` by about `14.231%`.

This suggests the TP/SL cell-block backend has asymmetric behavior:

- It helps long-only full-grid scoring.
- It does not help the reversal row under this workload and may be structurally bounded by different dominant work.

## 10. Correctness and parity

For completed candidate/control runs:

- top sample identity/order matched current exact and Stage 09 where available;
- `best_tp` / `best_sl` behavior stayed stable;
- metrics matched within accepted tolerance;
- no TP/SL tie-breaking change was accepted;
- no default backend or selector change was enabled.

Stage 13 therefore produced useful correctness evidence, but not accepted production acceleration evidence.

## 11. Memory result

Memory gate: peak RSS must not worsen by more than `10%` versus Stage 09.

All candidate shapes passed memory gate.

The best shape `64x128` had:

- max memory peak regression vs Stage 09: `-17.870%`.
- This is a memory improvement, not a regression.

Memory is not the blocking reason for rejection.

## 12. What changed in the repository

The preservation commit added/updated:

- Stage 13 aggregate gate harness:
  - `scripts/backtest/run_stage_13_tp_sl_block_autotune_gate.py`
- API benchmark harness support for current-exact Stage 13 controls:
  - `scripts/backtest/run_api_runner_benchmark_parity.py`
- Unit tests:
  - `tests/unit/contexts/backtest/application/services/v2/test_stage_13_tp_sl_block_autotune_gate.py`
- Stage 13 evidence:
  - `docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/`
- Stage 05+12 baseline evidence:
  - `docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_stage13_baseline/`
- Documentation:
  - `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`
  - `docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md`
  - `docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md`
  - `docs/architecture/README.md`

No production runtime default change was accepted.

No push or deploy was performed.

## 13. Verification that passed

Local verification after implementation:

```bash
uv run ruff check scripts/backtest/run_api_runner_benchmark_parity.py scripts/backtest/run_stage_13_tp_sl_block_autotune_gate.py tests/unit/contexts/backtest/application/services/v2/test_stage_13_tp_sl_block_autotune_gate.py
```

Result: passed.

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_stage_13_tp_sl_block_autotune_gate.py tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py
```

Result: `15 passed`.

Broader focused service test run during the same work:

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2
```

Result: `262 passed`.

Docs/index checks:

```bash
python -m tools.docs.generate_docs_index --check
git diff --check
git diff --cached --check
```

Result: passed.

## 14. Contract impact

Production/runtime behavior: `none`.

Public API contract: `none`.

Port contract: `none`.

DTO schema: `none`.

Persisted schema: `none`.

Config schema/defaults: `none`.

Request hash/cache key/persistence identity: `none`.

Benchmark/rollout gate: `compatible-change`.

Documentation/evidence: `compatible-change`.

## 15. Practical interpretation

The Stage 13 experiment did not fail because of correctness, memory, or instability.

It failed because the optimization target was not the global dominant cost across both required TP/SL rows.

The data indicates:

1. Larger/asymmetric block shapes improve `long_only` materially.
2. `long_short_reversal` is effectively flat or worse for every tested shape.
3. `trade_cell_evals_per_sec` is already around `5.9M/s` for reversal on most shapes.
4. The reversal workload may be bounded by a different part of the algorithm than simple block shape.
5. A selector that routes only `long_only` to the candidate shape could be tempting, but it would not satisfy the original Stage 13 production gate unless the policy explicitly allows per-row backend selection and separately handles reversal.

## 16. Current blocked state

Stage 13 status: `rejected`.

Stage 14 status: `blocked`.

Reason Stage 14 is blocked:

Stage 14 was intended to build a TP/SL monotonic cell kernel on top of an accepted Stage 13 or equivalent TP/SL production gate. Since Stage 13 did not produce an accepted TP/SL production candidate, Stage 14 cannot proceed as originally defined.

The TP/SL backend remains:

- accepted as Stage 09 opt-in/internal evidence;
- not accepted as production default;
- not allowed as a new selector/default path from this Stage 13 evidence.

## 17. Questions for the next model

Please analyze the evidence and recommend the next technically defensible step.

Important constraints:

1. Exact TP/SL semantics must be preserved unless explicitly reframed as a product decision.
2. SL-wins tie rule must not change.
3. top-N identity/order, `best_tp`, `best_sl`, metrics and ranking must remain stable for exact mode.
4. Rejected methods must not be reintroduced as accepted speedups without new evidence.
5. A recommendation should separate:
   - exact production-safe optimization;
   - telemetry-only learning stage;
   - approximate/coarse product-mode decision.

Specific questions:

1. What is the likely dominant cost in `tp_sl_grid/arity_6/long_short_reversal`, given that block shape changes do not improve it?
2. Should the next stage be a diagnostic telemetry stage focused on reversal internals rather than another implementation attempt?
3. Would a per-row or per-strategy selector be safe, or would it create unacceptable contract/operability complexity?
4. Is there a better exact-safe TP/SL kernel direction than block-shape autotune, for example monotonic classification, hit-time precomputation, window reuse, or early abandon?
5. What minimum new counters should be added before attempting another TP/SL optimization?
6. Should Stage 14 be redefined as a repair/replacement gate for reversal, or should a new Stage 13R be opened first?

## 18. Recommended framing for the next recommendation

Do not recommend enabling `64x128` globally. It fails the required reversal row.

Do not recommend accepting Stage 13 as production gate. It is rejected.

A useful recommendation should probably be one of:

- open a Stage 13R diagnostic/repair stage focused on reversal cost attribution;
- open a telemetry-only accepted-for-learning stage to locate reversal bottlenecks;
- define a narrower per-row selector policy only if contract and rollout rules are explicit;
- redesign Stage 14 so it has a valid accepted/replacement baseline and does not depend on the rejected Stage 13 decision.

