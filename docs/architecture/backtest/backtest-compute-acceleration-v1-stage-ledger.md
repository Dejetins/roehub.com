# Backtest Compute Acceleration v1 Stage Ledger

Журнал фиксирует execution state для staged-ускорения backtest compute. Любой
production-affecting backend stage может двигаться дальше только при
`next_iteration_allowed: true`.

## Правило Движения

- `planned`: stage описан, но не выполнен.
- `running`: stage реализуется или benchmark выполняется.
- `blocked`: нет сопоставимого benchmark evidence или есть correctness/performance blocker.
- `accepted_for_learning`: shadow/instrumentation stage дал полезные данные, но не включает production backend.
- `accepted`: stage прошел correctness, speed, memory и contract gates.
- `rejected`: stage не принят; изменения не должны оставаться в production runtime.

## Git / Main Delivery Rule

Все implementation stages выполняются на ветке `main`. Если checkout не на
`main`, executor должен остановиться и записать blocker, если пользователь
явно не разрешил другой branch.

После `accepted` stage executor должен обновить ledger/evidence/docs, выполнить
required gates, добавить в Git только scoped stage files и создать commit на
ветке `main`. Запись stage должна включать branch, scoped paths, commit SHA и
`push/deploy` статус. По умолчанию для этого плана `push/deploy: not performed`.

`accepted_for_learning` stages коммитятся в `main`, только если shadow/telemetry
code, docs или evidence являются durable handoff для следующих stages. Такой
commit не разблокирует production `on` mode без отдельного `accepted` speed gate.

Для `blocked` или `rejected` stages production runtime changes не коммитятся.
Допускается коммитить только ledger/evidence/docs, фиксирующие blocker или
rejection, если это нужно для durable history.

## Historical Reference Evidence

Это не новая baseline-запись, а последний сохраненный ориентир до Stage 00.
Он оставлен для сравнения с предыдущим чистым состоянием runtime.

| Evidence | Job | Exact current s | Reference s | Ratio | Service wall s | Result |
|---|---|---:|---:|---:|---:|---|
| `2026-05-14_iteration_15_api_runner_clean_arity6_cpu_memory` | `none/arity_6/long_only` | 15.968 | 15.694 | 0.983 | 16.673 | speed pass, memory fail |
| `2026-05-14_iteration_15_api_runner_clean_arity6_cpu_memory` | `none/arity_6/long_short_reversal` | 15.810 | 15.365 | 0.972 | 16.320 | speed pass, memory fail |
| `2026-05-14_iteration_15_api_runner_clean_arity6_cpu_memory` | `tp_sl_grid/arity_6/long_only` | 16.566 | 17.446 | 1.053 | 18.284 | pass |
| `2026-05-14_iteration_15_api_runner_clean_arity6_cpu_memory` | `tp_sl_grid/arity_6/long_short_reversal` | 15.504 | 16.204 | 1.045 | 16.967 | pass |

Rejected experiment:

| Evidence | Decision |
|---|---|
| `2026-05-15_iteration_16_quality_gate_ranking_exact_arity6_cpu_memory` | `rejected`; partial no-risk exact speedup was offset by service wall regression and result-shape breakage |

## Stage 00 Current Baseline

Evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/`

Mac Studio run at commit `d9bfa5811e3f5bccab9fb2635166f97e43f100bb`.

Overall gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `performance.pass` | true |
| `parity.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Current heavy timings:

| Job | Exact current s | Exact May2 s | Ratio | Service wall s | Result |
|---|---:|---:|---:|---:|---|
| `none/arity_6/long_only` | 15.704 | 15.694 | 0.999 | 22.032 | pass |
| `none/arity_6/long_short_reversal` | 15.111 | 15.365 | 1.017 | 15.358 | pass |
| `tp_sl_grid/arity_6/long_only` | 17.206 | 17.446 | 1.014 | 38.605 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 15.367 | 16.204 | 1.054 | 15.733 | pass |

### Stage 00 Verification - 2026-06-06

Stage 00 remains accepted on current checkout
`6dcb62dc918a98564abec9554ae575187b32fa39`.

The Mac Studio benchmark evidence was not refreshed because the scoped backtest
runtime and benchmark harness diff from evidence commit
`d9bfa5811e3f5bccab9fb2635166f97e43f100bb` to current `HEAD` is empty for:

```bash
git diff --name-only d9bfa5811e3f5bccab9fb2635166f97e43f100bb..HEAD -- \
  scripts/backtest \
  apps/api/dto/backtests.py \
  apps/api/dto/ui_backtests.py \
  apps/api/routes/backtests.py \
  apps/api/routes/ui_backtests.py \
  apps/api/wiring/modules/backtest.py \
  apps/api/wiring/modules/ui_backtests.py \
  apps/worker/backtest_job_runner \
  src/trading/contexts/backtest
```

Benchmark accounting was re-validated locally:

```bash
uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/local_accounting_validation.json
```

Result: accounting validation passed; Stage 00 evidence remains comparable for
the backtest compute acceleration baseline. Contract impact: `none`.
`next_iteration_allowed` remains `true` for Stage 01 only.

## Stage Ledger

| Stage | Status | Scope | Evidence | Decision | next_iteration_allowed |
|---:|---|---|---|---|---|
| 00 | accepted | Refresh current heavy baseline on Mac Studio before code changes | `benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/` | Baseline accepted; re-verified on checkout `6dcb62dc918a98564abec9554ae575187b32fa39`; scoped backtest runtime/harness diff from evidence commit is empty; performance, parity, memory, lazy cache, legacy path, accounting and docs drift gates passed | true |
| 01 | planned | Add instrumentation counters without behavior changes | planned | Allowed by Stage 00; not started | false |
| 02 | planned | Row/signature telemetry shadow | planned | Pending Stage 01 | false |
| 03 | planned | Runtime bitset pack shadow | planned | Pending Stage 02 | false |
| 04 | planned | `matrix_bitset_no_risk_v1` for `none/arity_2..3/long_only` | planned | Pending Stage 03 | false |
| 05 | planned | No-risk `long_short_reversal` and arity 6 heavy rows | planned | Pending Stage 04 | false |
| 06 | planned | Consensus signature cache | planned | Pending Stage 05 | false |
| 07 | planned | Sidecar/test bitset artifacts generated outside publisher: planned generator/helper, explicit sidecar path, `signals_pos_bits.u64.npy`, `signals_neg_bits.u64.npy`, `signal_row_hashes.u64.npy`, `unique_signal_row_ids.u32.npy`, `duplicate_signal_row_ids.u32.npy`, `matrix_sidecar_manifest.json`; no `backtest_artifacts` publisher/precompute or canonical manifest changes | planned | Pending Stage 06 | false |
| 08 | planned | TP/SL selected-cell shadow with by-entry hit-times layout or selected by-entry arrays; sidecar-only if persisted for testing, no publisher/manifest changes without a separate approved plan | planned | Pending Stage 07 | false |
| 09 | planned | `matrix_cell_tp_sl_v1` full grid blocks | planned | Pending Stage 08 | false |
| 10 | planned | Exact-safe high-arity pruning | planned | Pending Stage 09 | false |
| 11 | planned | Lazy detail reuse of sparse trade tape | planned | Pending Stage 10 | false |

## Stage Acceptance Requirements

Каждый stage должен записать evidence directory, команды проверки, сравнение с
Stage 00 или ближайшим принятым stage, correctness result, memory result,
contract impact и финальное решение. Если хотя бы один required gate ниже не
закрыт, stage остается `blocked` или `rejected`, а `next_iteration_allowed`
остается `false`.

| Stage | Required implementation boundary | Required acceptance evidence | Required rejection/block rule |
|---:|---|---|---|
| 00 | No code changes; refresh current heavy baseline only | Mac Studio API-runner benchmark, accounting validation, heavy rows for `none` and `tp_sl_grid`, artifact/request hashes, memory cleanup and legacy-path absence | Block if baseline evidence is missing or not comparable |
| 01 | Add instrumentation counters only | Overhead <= 1%, no result hash/top-N drift, counters present for artifact load, pack, combo/proxy/exact/top assembly, rows/candidates/trades/cells | Reject if instrumentation changes scoring, ranking, request identity or service wall materially |
| 02 | Row/signature telemetry shadow only | Duplicate-row potential, signature counts, collision-safety check, no pruning, no top-N drift | Block if dedup identity expansion is not specified or collision handling is ambiguous |
| 03 | Runtime bitset pack shadow only | `+1/0/-1` parity, padding validation, word-count validation, pack timing, memory peak/cleanup, sample consensus parity | Block if bitset masks cannot reproduce current consensus semantics |
| 04 | `matrix_bitset_no_risk_v1` for `none/arity_2..3/long_only` | Exact parity against current backend, top-N shape/hash or bounded metric diff, Stage 00/MVP-row speedup, service wall no regression, fees/slippage/sizing parity for scoped modes | Reject if speedup is only microbenchmark-level or top-N identity/order drifts |
| 05 | No-risk reversal and arity 6 heavy rows | All reversal transitions covered, arity-6 heavy rows faster than Stage 00, service wall/memory no regression, public result shape unchanged | Block if reversal semantics are inferred rather than proven by cases |
| 06 | Consensus signature cache | Cache hit-rate, collision-safe keying, deterministic merge/tie policy, exact parity, accepted speedup over Stage 05 or Stage 00 comparable rows | Reject if cache changes ranking or only improves a non-hot sub-timer |
| 07 | Sidecar/test bitset artifacts outside publisher | Generated `.npy` files plus `matrix_sidecar_manifest.json`, source manifest/hash validation, dtype/shape/padding/duplicate-map validation, absent-sidecar fallback, load-vs-pack timing, no canonical publisher/manifest changes | Block if any canonical `manifest.yaml`, `current.yaml`, active slot or publisher/precompute path is changed |
| 08 | TP/SL selected-cell shadow | `tp_count <= 8`, `sl_count <= 8`, selected by-entry arrays or counters, SL-wins tie proof, trade-boundary and fees/slippage parity, no production top-N feed | Block if selected-cell parity is incomplete or hit-times layout is not measured |
| 09 | `matrix_cell_tp_sl_v1` full grid blocks | Full grid parity, cell-block size recorded, `tp_sl_exact_scoring` speedup, trade-cell eval/sec, memory peak/cleanup, service wall no regression | Reject if selected-cell speedup does not hold on full grid or memory dominates |
| 10 | Exact-safe high-arity pruning | Exact-safe proof for pruning rule, arity 7/10 bounded evidence, no approximate beam in default path, parity on retained candidates, speedup and no result-shape drift | Block if pruning can remove a valid top candidate or requires product-level approximation approval |
| 11 | Lazy detail reuse of sparse trade tape | Selected variant materialization latency benchmark, cache identity parity, no bulk top-N scoring change, fallback to current lazy materialization | Reject if it improves perceived latency while invalidating persisted/lazy detail identity |

## Cross-Stage Acceptance Rules

- No stage may advance only on a local kernel/microbenchmark result; API-runner
  service wall and memory evidence must not regress.
- Acceptance speedup must use the same current-service pipeline boundary unless
  the evidence is explicitly labeled diagnostic. Sidecar generation, sidecar load,
  warmup/cache state, request semantics, candidate set and top-N/persistence shape
  must be recorded so the new path does not get a hidden advantage.
- While publisher/precompute changes are out of scope, sidecar-dependent speedup
  may be `accepted_for_learning`, but it cannot by itself enable production `on`
  mode or unlock a production-affecting stage.
- Shadow or telemetry stages may be `accepted_for_learning`, but they do not
  unlock production `on` mode.
- Any change to public API, DB schema, canonical artifact manifests, request hash,
  `variant_hash`, TP/SL tie-breaking, fees/slippage or sizing semantics requires
  an explicit plan update before implementation continues.
- Accepted or accepted-for-learning stages must leave a scoped commit on `main`
  after evidence and ledger updates. The ledger must record commit SHA, branch,
  scoped paths and whether push/deploy was performed.
- Approximate beam search, GPU-first rewrite, publisher-level bitset artifacts and
  mark-to-market drawdown acceleration are outside this ledger unless a separate
  approved plan adds them.

## Stage 00 Baseline Command

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_00_current_baseline

uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_00_current_baseline/local_accounting_validation.json
```

Acceptance:

- required heavy jobs succeed or blockers are recorded explicitly;
- `performance.pass`, `parity.pass`, `legacy_path_absence.pass` and
  `docs_drift_audit.pass` are recorded;
- memory failures are allowed only as inherited baseline risks and must not
  worsen in later stages;
- the summary includes exact `service_wall_clock_s`, `exact_scoring`,
  `tp_sl_exact_scoring`, CPU samples, artifact hashes and git state.
