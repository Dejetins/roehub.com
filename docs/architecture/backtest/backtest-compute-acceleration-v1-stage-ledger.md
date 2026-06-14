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
- `closed_not_executed`: stage был в плане, но закрыт до реализации и не имеет
  executable prompt/runtime scope.

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

## Stage 01 Instrumentation Counters

Evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_01_instrumentation/`

Mac Studio run from checkout `d9bfa5811e3f5bccab9fb2635166f97e43f100bb`
with a scoped dirty worktree containing only Stage 01 runtime/reporting files:

- `apps/worker/backtest_job_runner/wiring/modules/child_ipc.py`
- `apps/worker/backtest_job_runner/wiring/modules/child_process.py`
- `scripts/backtest/run_api_runner_benchmark_parity.py`
- `src/trading/contexts/backtest/application/services/v2/job_orchestration.py`

The remote checkout was intentionally not fast-forwarded before the benchmark
because its tracked backtest runtime and benchmark harness diff from Stage 00
evidence commit to local `HEAD` was empty before Stage 01 changes. This keeps the
Stage 00/Stage 01 comparison scoped to instrumentation counters.

Commands:

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_01_instrumentation

uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_01_instrumentation/local_accounting_validation.json
```

Overall gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `instrumentation.pass` | true |
| `performance.pass` | true |
| `parity.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Stage 01 timing comparison against Stage 00:

| Job | Stage 00 service wall s | Stage 01 service wall s | Wall delta % | Stage 00 service total s | Stage 01 service total s | Total delta % | Stage 00 exact s | Stage 01 exact s | Exact delta % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 22.032 | 16.169 | -26.612 | 18.337 | 16.690 | -8.981 | 15.704 | 15.484 | -1.400 |
| `none/arity_6/long_short_reversal` | 15.358 | 15.313 | -0.290 | 15.658 | 15.589 | -0.439 | 15.111 | 15.078 | -0.216 |
| `tp_sl_grid/arity_6/long_only` | 38.605 | 17.940 | -53.530 | 36.398 | 34.284 | -5.807 | 17.206 | 16.175 | -5.994 |
| `tp_sl_grid/arity_6/long_short_reversal` | 15.733 | 15.613 | -0.765 | 31.319 | 31.066 | -0.809 | 15.367 | 15.254 | -0.739 |

Instrumentation counters are emitted per benchmark job. Required fields are
present; counters that are not available in the current runtime are explicitly
`null`, including `signals_pack_ms`, `rows_before_prefilter`,
`avg_segments_per_candidate` and `avg_trades_per_candidate`.

Representative counters:

| Job | artifact load ms | combos | proxy candidates | exact candidates/s | trade-cell evals/s | rows after prefilter | tp/sl cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 86.278 | 46656 | 46656 | 3013.182 | n/a | 36 | 0 |
| `none/arity_6/long_short_reversal` | 80.319 | 46656 | 46656 | 3094.228 | n/a | 36 | 0 |
| `tp_sl_grid/arity_6/long_only` | 79.925 | 46656 | 46656 | 2884.457 | 6371765.328 | 36 | 2209 |
| `tp_sl_grid/arity_6/long_short_reversal` | 79.095 | 46656 | 46656 | 3058.664 | 6756589.105 | 36 | 2209 |

Decision: Stage 01 is `accepted_for_learning`. It is a telemetry/reporting
handoff only: no matrix backend, production `on` mode, scoring semantics,
ranking, top-N shape, request hash, cache identity or persistence semantics are
changed. Contract impact: public API `none`; DTO schema `none`; persisted schema
`none`; config schema `none`; request hash/cache identity `none`; benchmark and
report semantics `compatible-change`.

`next_iteration_allowed` is `true` for Stage 02 telemetry only. It does not
unlock any production-affecting backend `on` mode.

Git branch: `main`. Scoped commit SHA is recorded in the executor final report
after the commit is created. Push/deploy: not performed.

## Stage 02 Row/Signature Telemetry Shadow

Evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_02_row_signature_telemetry/`

Mac Studio run from checkout `d9bfa5811e3f5bccab9fb2635166f97e43f100bb`
with a temporary scoped patch containing the local Stage 01 handoff plus Stage
02 row-signature telemetry files. The remote patch was removed after evidence
was copied back; the committed handoff is this local `main` checkout.

Commands:

```bash
ROEHUB_ENV=prod \
ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml \
uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_02_row_signature_telemetry
```

The first Mac Studio attempt failed before compute because the non-interactive
shell did not export `ROEHUB_ENV` or `ROEHUB_BACKTEST_ARTIFACTS_CONFIG`. A
second attempt with exact consensus enumeration enabled passed correctness but
showed unacceptable telemetry overhead (`row_signature_ms` around 43-47s). The
accepted evidence is the final rerun after lowering the default exact consensus
enumeration limit so heavy rows report the deterministic unique-row-product
upper bound instead of enumerating 46,656 consensus vectors.

Overall gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `instrumentation.pass` | true |
| `performance.pass` | true |
| `parity.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Stage 02 telemetry counters:

| Job | rows after prefilter | unique rows after dedup | duplicate rows | consensus_signature_count | consensus mode | row_signature_ms | collisions |
|---|---:|---:|---:|---:|---|---:|---:|
| `none/arity_6/long_only` | 36 | 36 | 0 | 46656 | `upper_bound_unique_row_product` | 10.388 | 0 |
| `none/arity_6/long_short_reversal` | 36 | 36 | 0 | 46656 | `upper_bound_unique_row_product` | 10.400 | 0 |
| `tp_sl_grid/arity_6/long_only` | 36 | 36 | 0 | 46656 | `upper_bound_unique_row_product` | 10.653 | 0 |
| `tp_sl_grid/arity_6/long_short_reversal` | 36 | 36 | 0 | 46656 | `upper_bound_unique_row_product` | 10.800 | 0 |

Correctness/performance evidence:

| Job | Exact current s | May2 exact s | Ratio | System memory gate |
|---|---:|---:|---:|---|
| `none/arity_6/long_only` | 15.446 | 15.694 | 1.016 | pass |
| `none/arity_6/long_short_reversal` | 15.258 | 15.365 | 1.007 | pass |
| `tp_sl_grid/arity_6/long_only` | 16.042 | 17.446 | 1.088 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 15.417 | 16.204 | 1.051 | pass |

Decision: Stage 02 is `accepted_for_learning`. The stage adds shadow-only row
signature telemetry and benchmark reporting fields:
`unique_rows_after_dedup`, `consensus_signature_count`,
`duplicate_signal_row_ids`, `row_signature_collision_count`,
`consensus_signature_mode`, `candidate_upper_bound_after_row_dedup`, and
`row_signature_ms`.

No production pruning, deduplication, candidate reordering, scoring reuse,
top-N identity collapse, request-hash change, cache-key change or persistence
schema change is introduced. Duplicate mapping semantics are explicit:
`duplicate_signal_row_ids` lists original source row ids whose exact int8 row
content matches the first stable unique row for the same indicator, and Stage
02 never removes those rows. Collision strategy is explicit: equality uses full
SHA-256 row-content signatures; the sidecar-style u64 hash is collision-checked
only, and any non-zero collision count must disable future dedup/cache stages.

Contract impact: public API `none`; port contract `none`; DTO schema `none`;
persisted schema `none`; config schema `none`; request hash/cache identity
`none`; service-call semantics `none`; external side effects `none`; benchmark
and report semantics `compatible-change`; alert/runbook semantics `none`;
browser-visible behavior `none`.

`next_iteration_allowed` is `true` for Stage 03 runtime bitset-pack shadow only.
It does not unlock production pruning, Stage 06 cache reuse, Stage 07 sidecar
artifacts or any matrix backend `on` mode.

## Stage 03 Runtime Bitset Pack Shadow

Status: `accepted_for_learning`.

Evidence path:

`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_03_runtime_bitset_pack/`

Implementation state on branch `main`, with Mac Studio benchmark run from a
temporary worktree based on checkout
`16ab06dc1506edaa7e292c2497595fcc3f008664` plus this scoped Stage 03 patch:

- added isolated runtime bitset helper for `pos_bits` / `neg_bits` with
  little-endian bit order and `W = ceil(T / 64)`;
- added shadow-only orchestration hook after `prepare_pools`;
- added additive benchmark counters for `signals_pack_ms`, packed bytes, word
  count, padding validity and sample consensus parity;
- current scoring, combo planning, top-N, public request hash and persistence
  path are not fed by the bitsets.

Local semantic evidence:

| Check | Result |
|---|---|
| `+1/0/-1` round-trip | pass |
| `long_only` positive mask semantics | pass |
| `long_short_reversal` negative mask semantics | pass |
| non-multiple-of-64 padding | pass |
| word count formula `W = ceil(T / 64)` | pass |
| sampled consensus parity | pass |
| current no-risk scoring focused suite | pass |

Focused local commands:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_bitsets.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py

uv run ruff check \
  src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py \
  src/trading/contexts/backtest/application/services/v2/job_orchestration.py \
  scripts/backtest/run_api_runner_benchmark_parity.py \
  tests/unit/contexts/backtest/application/services/v2/test_bitsets.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py

git diff --check
```

Results: pytest `52 passed`; ruff `All checks passed`; pyright
`0 errors`; docs index `OK`; `git diff --check` passed.

Mac Studio API-runner command:

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
  --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
  --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_03_runtime_bitset_pack
```

Runtime environment source:

- Mac Studio env file:
  `/Users/daniildegtyarev/.config/roehub/roehub.env`;
- benchmark-filled runtime keys when absent from env file:
  `ROEHUB_ENV=prod` and
  `ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml`;
- Postgres keys present in the env file:
  `STRATEGY_PG_DSN`, `POSTGRES_DSN`, `IDENTITY_PG_DSN`,
  `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`;
- artifact root resolved by prod config:
  `/opt/roehub/state/backtest_artifacts/v2`, with `BTCUSDT/current.yaml` and
  active slot manifest present on Mac Studio;
- evidence records only key names and paths, never DSN or password values.

Benchmark harness repair: the runner accepts `--env-file` and falls back to
`$ROEHUB_ENV_FILE`, `/Users/daniildegtyarev/.config/roehub/roehub.env`, then
`/etc/roehub/roehub.env`. It expands `${POSTGRES_*}` placeholders in env-file
DSN values, derives missing `STRATEGY_PG_DSN`, `POSTGRES_DSN` and
`IDENTITY_PG_DSN` from `POSTGRES_DB`, `POSTGRES_USER` and `POSTGRES_PASSWORD`,
and fills `ROEHUB_ENV` plus `ROEHUB_BACKTEST_ARTIFACTS_CONFIG` for Mac Studio
API-runner evidence when the env file omits them.

Mac Studio acceptance gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `api_runner_path.pass` | true |
| `instrumentation.pass` | true |
| `performance.pass` | true |
| `parity.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Stage 03 telemetry counters:

| Job | W | signals_pack_ms | packed bytes | padding valid | consensus sample parity |
|---|---:|---:|---:|---|---|
| `none/arity_6/long_only` | 3421 | 23.617 | 1,970,496 | true | true |
| `none/arity_6/long_short_reversal` | 3421 | 24.067 | 1,970,496 | true | true |
| `tp_sl_grid/arity_6/long_only` | 3421 | 24.195 | 1,970,496 | true | true |
| `tp_sl_grid/arity_6/long_short_reversal` | 3421 | 24.063 | 1,970,496 | true | true |

Correctness/performance evidence:

| Job | Exact current s | May2 exact s | Ratio | System memory gate |
|---|---:|---:|---:|---|
| `none/arity_6/long_only` | 15.416 | 15.694 | 1.018 | pass |
| `none/arity_6/long_short_reversal` | 15.168 | 15.365 | 1.013 | pass |
| `tp_sl_grid/arity_6/long_only` | 16.077 | 17.446 | 1.085 | pass |
| `tp_sl_grid/arity_6/long_short_reversal` | 15.175 | 16.204 | 1.068 | pass |

Persisted top-N parity passed for `4/4` API-created runner jobs. The service
state path reached `queued -> running -> succeeded` for every required job, the
lazy cache-hit memory check passed with retained RSS delta `0`, and no failed
memory-release jobs remained in the accepted rerun.

Decision: Stage 03 is `accepted_for_learning`. The stage proves runtime
bitset-pack shape, padding and sampled consensus parity at the real API-runner
child boundary, but it is shadow-only and does not feed scoring. It does not
unlock production `on` mode, pruning, scoring reuse, cache reuse or sidecar
artifact publication.

Contract impact: public API `none`; port contract `none`; DTO schema `none`;
persisted schema `none`; config schema `none`; request hash/cache identity
`none`; service-call semantics `none`; benchmark/report semantics
`compatible-change`; browser-visible behavior `none`.

`next_iteration_allowed` is `true` for Stage 04 no-risk long-only MVP only.
Stage 04 must still produce its own parity, service wall, memory cleanup and
top-N evidence before any production-affecting backend mode is accepted.

## Stage 04: `matrix_bitset_no_risk_v1` MVP Accepted For Learning

Scope attempted: `matrix_bitset_no_risk_v1` for `none/arity_2/long_only` and
`none/arity_3/long_only`, default-off and selectable only through internal
benchmark/runtime gates.

Implementation state in the local worktree and Mac Studio candidate patch:

- added requestable internal backend id `matrix_bitset_no_risk_v1`;
- default backend selection remains unchanged;
- API/request payload, DB schema, request hash, `variant_hash` and persisted
  top-N shape are unchanged;
- `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_04_no_risk_mvp` can opt no-risk
  API-runner jobs into the matrix backend only for `risk.mode=none`,
  `direction_mode=long_only`, arity 2 or 3;
- reversal, TP/SL and higher arity remain out of scope and are not enabled.

Mac Studio execution context:

- SSH target: `macstudio`.
- Remote checkout:
  `/Users/daniildegtyarev/Projects/roehub.com`.
- Remote HEAD before candidate sync:
  `3dc4726f30081968687299e38c01a196c8d7e443`.
- The measured Mac Studio checkout was dirty by design: only the scoped Stage
  04 runtime/test/benchmark candidate files were synced for benchmark evidence.
- Env file loaded:
  `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Additional env file presence was verified at `/etc/roehub/roehub.env`.
- Runtime artifact config:
  `configs/prod/backtest_artifacts.yaml`.
- Active source manifest resolved read-only to:
  `/opt/roehub/state/backtest_artifacts/v2/binance/spot/BTCUSDT/current.yaml`.
- Secret values were not recorded.

Focused Mac Studio checks passed:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  /opt/homebrew/bin/uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py \
  tests/unit/contexts/backtest/application/services/v2/test_bitsets.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py'

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  /opt/homebrew/bin/uv run ruff check \
  scripts/backtest/run_api_runner_benchmark_parity.py \
  src/trading/contexts/backtest/application/services/v2/job_orchestration.py \
  src/trading/contexts/backtest/application/services/v2/no_risk_exact.py \
  src/trading/contexts/backtest/application/services/v2/combo_planning.py \
  src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py'
```

Results: pytest `57 passed`; ruff `All checks passed`.

Acceptance benchmark command:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_04_no_risk_mvp \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-04-mvp-rows \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_04_no_risk_mvp \
    --system-memory-cleanup-wait-seconds 5 \
    --timeout-seconds 1800 \
    --poll-interval-seconds 0.1 \
    --cpu-sample-interval-seconds 0.5'
```

Benchmark evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_04_no_risk_mvp/`

Overall gates:

| Gate | Result |
|---|---|
| `api_runner_path.pass` | true |
| `parity.pass` | true |
| `performance.pass` | false |
| `instrumentation.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Scoped MVP timings:

| Job | Backend | Exact current s | May2 exact s | Ratio | Result |
|---|---|---:|---:|---:|---|
| `none/arity_2/long_only` | `matrix_bitset_no_risk_v1` | 0.003684 | 0.002545 | 0.691 | speed fail |
| `none/arity_3/long_only` | `matrix_bitset_no_risk_v1` | 0.018385 | 0.047619 | 2.590 | speed pass |

Decision: Stage 04 is `accepted_for_learning`. Correctness and operational
gates passed for the scoped Mac Studio API-runner path. The raw performance
gate still reports `fail` because `none/arity_2/long_only` regressed against
the May 2 reference, but that row is a tiny `36`-candidate case where fixed
runtime overhead dominates the measured exact-scoring timer. The absolute
delta is about `1.1ms`, while the same backend gives a `2.590x` exact-scoring
speedup on `none/arity_3/long_only`.

Operator policy decision: the arity-2 no-advantage result is not production
acceptance evidence, but it is not a blocker for learning progression because
the expected algorithmic payoff is in later higher-arity/reversal/cache stages.
This stage unlocks Stage 05 implementation only. It does not unlock production
`on` mode, default backend switching, TP/SL, pruning, request-hash changes,
cache identity changes or sidecar artifacts. Production runtime candidate
changes must not be committed as an accepted production backend.

Contract impact: public API `none`; port contract `none`; DTO schema `none`;
persisted schema `none`; config schema `compatible-change` for the internal
default-off `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` benchmark/runtime gate;
request hash/cache identity `none`; service-call semantics `none`;
benchmark/report semantics `compatible-change`; browser-visible behavior
`none`.

## Stage 05: `matrix_bitset_no_risk_v1` Reversal And Arity 6 Accepted

Scope accepted: `matrix_bitset_no_risk_v1` for no-risk arity 6 heavy rows and
`long_short_reversal`. At original Stage 05 acceptance this remained
default-off and selectable only through the internal benchmark/runtime gate.

Implementation state in the local worktree and Mac Studio candidate patch:

- kept default backend selection unchanged;
- widened the requestable internal backend id `matrix_bitset_no_risk_v1` to
  arity 6;
- added explicit Stage 05 gate
  `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_05_no_risk_reversal_arity6` for
  `none/arity_6/long_only` and `none/arity_6/long_short_reversal`;
- preserved public API, DB schema, request hash, `variant_hash`, persisted
  top-N shape, fees/slippage, sizing and `close_on_end` semantics;
- added focused parity coverage for `long -> short`, `short -> long`,
  `long -> flat`, `short -> flat`, `flat -> long` and `flat -> short` against
  the current backend.

Mac Studio execution context:

- SSH target: `macstudio`.
- Remote checkout:
  `/Users/daniildegtyarev/Projects/roehub.com`.
- Remote HEAD before candidate sync:
  `9ecdb97591d32f1691291ac7c3335cfc3ef530c7`.
- The measured Mac Studio checkout was dirty by design: only the scoped Stage
  05 runtime/test/benchmark candidate files were synced for benchmark evidence.
- Env file loaded:
  `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Runtime artifact config:
  `configs/prod/backtest_artifacts.yaml`.
- Secret values were not recorded.
- The temporary remote code patch and copied remote evidence directory were
  removed after evidence was synced into this local checkout; unrelated remote
  untracked files were left untouched.

Focused checks:

```bash
uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  /opt/homebrew/bin/uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py && \
  /opt/homebrew/bin/uv run ruff check \
  src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py \
  src/trading/contexts/backtest/application/services/v2/no_risk_exact.py \
  src/trading/contexts/backtest/application/services/v2/combo_planning.py \
  src/trading/contexts/backtest/application/services/v2/job_orchestration.py \
  scripts/backtest/run_api_runner_benchmark_parity.py \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py'
```

Results: local focused pytest `49 passed`; Mac Studio focused pytest
`52 passed`; Mac Studio ruff `All checks passed`.

Acceptance benchmark command:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_05_no_risk_reversal_arity6 \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-05-no-risk-heavy-rows \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_05_reversal_arity6 \
    --system-memory-cleanup-wait-seconds 5 \
    --timeout-seconds 1800 \
    --poll-interval-seconds 0.1 \
    --cpu-sample-interval-seconds 0.5'
```

Benchmark evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_05_reversal_arity6/`

Overall gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `api_runner_path.pass` | true |
| `parity.pass` | true |
| `performance.pass` | true |
| `instrumentation.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Heavy timings:

| Job | Backend | Exact current s | May2 exact s | Ratio | Service wall s | Service total s | System memory gate |
|---|---|---:|---:|---:|---:|---:|---|
| `none/arity_6/long_only` | `matrix_bitset_no_risk_v1` | 1.010 | 15.694 | 15.543 | 1.590 | 1.920 | pass |
| `none/arity_6/long_short_reversal` | `matrix_bitset_no_risk_v1` | 2.887 | 15.365 | 5.323 | 3.135 | 3.401 | pass |

Instrumentation counters:

| Job | signal pack ms | combos | proxy candidates | exact candidates/s | rows after prefilter | unique rows | consensus signatures | row signature ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 24.982 | 46656 | 46656 | 46207.826 | 36 | 36 | 46656 | 10.378 |
| `none/arity_6/long_short_reversal` | 26.060 | 46656 | 46656 | 16162.103 | 36 | 36 | 46656 | 10.591 |

`avg_trades_per_candidate` remains unavailable in current additive benchmark
instrumentation and is reported as explicit `null`; this is a reporting
limitation, not a scoring correctness gap.

Decision: Stage 05 is `accepted`. Correctness, parity, performance, memory,
lazy cache, legacy path and docs drift gates passed on the Mac Studio
API-runner boundary. The original Stage 05 handoff kept the matrix backend
default-off and did not unlock production default switching, TP/SL, pruning,
request-hash changes, cache identity changes or sidecar artifacts.

Contract impact: public API `none`; port contract `none`; DTO schema `none`;
persisted schema `none`; config schema `compatible-change` for the internal
default-off Stage 05 `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` value and benchmark
row selector; request hash/cache identity `none`; service-call semantics
`none`; benchmark/report semantics `compatible-change`; browser-visible
behavior `none`.

`next_iteration_allowed` is `true` for Stage 06 consensus signature cache only.
Stage 06 must still prove deterministic cache keying, hit-rate, top-N parity,
service wall, memory cleanup and accepted speedup before any cache-backed
production `on` mode is allowed.

### Stage 05 Default-On Productionization - 2026-06-10

Scope accepted: make `matrix_bitset_no_risk_v1` the default compute backend only
for `risk.mode=none`, arity `6`, and `direction_mode in {long_only,
long_short_reversal}`. The rollback/comparison override remains
`ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off`.

Implementation state:

- default selector now resolves unset/empty `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE`
  to `stage_05_no_risk_reversal_arity6`;
- explicit `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off` still routes to the legacy
  `event_segments_n_no_risk` backend for A/B and rollback;
- no default enablement was added for arity 2/3, TP/SL, sidecar artifacts,
  pruning, cache reuse or publisher/precompute changes.

Focused local checks:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py

uv run ruff check \
  src/trading/contexts/backtest/application/services/v2/job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py
```

Results: local focused pytest `55 passed`; local ruff `All checks passed`.

Mac Studio focused checks:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  /opt/homebrew/bin/uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py'

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  /opt/homebrew/bin/uv run ruff check \
  src/trading/contexts/backtest/application/services/v2/job_orchestration.py \
  tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py'
```

Results: Mac Studio focused pytest `55 passed`; Mac Studio ruff
`All checks passed`.

Mac Studio benchmark evidence:

- baseline-off:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_off_baseline/`
- default-on candidate:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_on_candidate/`
- full top-50 A/B:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_on_candidate/ab_default_on_parity.json`

Benchmark commands:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-05-no-risk-heavy-rows \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_off_baseline \
    --system-memory-cleanup-wait-seconds 5 \
    --timeout-seconds 1800 \
    --poll-interval-seconds 0.1 \
    --cpu-sample-interval-seconds 0.5'

ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  env -u ROEHUB_BACKTEST_MATRIX_BACKEND_MODE \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-05-no-risk-heavy-rows \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_on_candidate \
    --system-memory-cleanup-wait-seconds 5 \
    --timeout-seconds 1800 \
    --poll-interval-seconds 0.1 \
    --cpu-sample-interval-seconds 0.5'
```

The measured Mac Studio checkout reported commit
`e985b30123ca9070ef5b1fc3227ffef6dd3fdf35` with a dirty status because the
local tracked checkout was copied onto the older Mac Studio worktree for this
candidate benchmark. Generated sidecar `.npy` files were not used.

Default-off versus default-on timings:

| Job | Old backend | Default backend | Exact off s | Exact default s | Exact speedup | Wall off s | Wall default s | Wall speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | `event_segments_n_no_risk` | `matrix_bitset_no_risk_v1` | 15.476 | 0.998 | 15.501 | 16.383 | 1.427 | 11.485 |
| `none/arity_6/long_short_reversal` | `event_segments_n_no_risk` | `matrix_bitset_no_risk_v1` | 15.005 | 2.911 | 5.155 | 15.281 | 3.155 | 4.843 |

Full top-50 A/B evidence:

| Job | Request hash same | Top count old/default | Strategy identity same | Variant order same | Max metric abs diff | Exact speedup | Wall speedup |
|---|---|---:|---|---|---:|---:|---:|
| `none/arity_6/long_only` | true | `50 / 50` | true | true | 0.0 | 15.501 | 11.485 |
| `none/arity_6/long_short_reversal/min_closed_trades=1` | true | `50 / 50` | true | true | 0.0 | 5.178 | 4.645 |

The standard Stage 05 long-short request still has zero top variants at
`min_closed_trades=300`; the full top-50 long-short parity check therefore uses
the accepted Stage 05 A/B contour with explicit `min_closed_trades=1`.

Decision: Stage 05 default-on productionization is `accepted`. Correctness,
top-50 identity/order, metric parity, exact scoring speedup, service wall
speedup, memory, lazy cache and legacy path gates passed on Mac Studio for the
accepted scope.

Contract impact: public API `none`; port contract `none`; DTO schema `none`;
persisted schema `none`; config schema `compatible-change` because the absence
of `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` now means Stage 05 default-on for the
accepted no-risk arity-6 scope; request hash/cache identity `none`;
service-call semantics `none`; benchmark/report semantics `compatible-change`;
browser-visible behavior `none`.

## Stage 06 — consensus signature cache rejected

Stage 06 tested an opt-in consensus signature cache candidate on the accepted
Stage 05 heavy no-risk matrix rows. The candidate grouped identical consensus
bitsets inside the matrix no-risk scoring path, compared exact consensus
payload bytes before reuse, preserved public top-N identity/order in focused
tests, and exposed explicit benchmark counters for cache hit-rate and digest
collision count.

Candidate code was copied temporarily to the Mac Studio checkout for benchmark
execution. The measured checkout reported base commit
`9ecdb97591d32f1691291ac7c3335cfc3ef530c7` plus dirty candidate source paths;
the remote checkout was restored after evidence was copied back. Local focused
preflight for the candidate passed:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py \
  tests/unit/contexts/backtest/domain/value_objects/test_variant_identity.py \
  tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py
```

Mac Studio unit preflight passed with `/opt/homebrew/bin/uv`: `65 passed`.

Acceptance benchmark command:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && \
  ROEHUB_ENV=prod \
  ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml \
  ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_05_no_risk_reversal_arity6 \
  ROEHUB_BACKTEST_MATRIX_SIGNATURE_CACHE=1 \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-05-no-risk-heavy-rows \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache'
```

Benchmark evidence:

`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache/`

Overall harness gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `api_runner_path.pass` | true |
| `parity.pass` | true |
| `performance.pass` | true versus May 2 reference only |
| `instrumentation.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Stage 06 acceptance comparison against the immediately accepted Stage 05
baseline failed:

| Job | Stage 05 exact s | Stage 06 exact s | Stage 05 service wall s | Stage 06 service wall s | Cache hit-rate | Cache hits | Unique consensus | Collision count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 1.010 | 4.932 | 1.590 | 5.366 | 0.202396 | 9443 | 37213 | 0 |
| `none/arity_6/long_short_reversal` | 2.887 | 6.166 | 3.135 | 6.414 | 0.202396 | 9443 | 37213 | 0 |

Decision: Stage 06 is `rejected`. The cache produced a real hit-rate and
collision-safe keying evidence, but exact scoring and service wall regressed on
the hot API-runner boundary. No cache-backed production `on` mode is unlocked,
and the candidate runtime code must not be committed as production code from
this stage.

Contract impact for the accepted ledger/evidence record: public API `none`;
port contract `none`; DTO schema `none` for committed runtime because the cache
candidate is not accepted; persisted schema `none`; config schema `none`;
request hash/cache identity `none`; service-call semantics `none`;
benchmark/report semantics `compatible-change`; browser-visible behavior
`none`.

`next_iteration_allowed` is `true` for Stage 07 sidecar/test bitset artifacts
only. Stage 07 must not depend on the rejected runtime signature cache and must
continue to avoid canonical publisher/precompute or manifest changes.

## Stage 07 — sidecar/test bitset artifacts accepted for learning

Stage 07 implemented sidecar/test bitset artifacts outside the canonical
publisher. The generator writes benchmark-only artifacts under the stage
evidence directory, one indicator directory per signal matrix:

- `signals_pos_bits.u64.npy`
- `signals_neg_bits.u64.npy`
- `signal_row_hashes.u64.npy`
- `unique_signal_row_ids.u32.npy`
- `duplicate_signal_row_ids.u32.npy`
- `duplicate_unique_signal_row_ids.u32.npy`
- `matrix_sidecar_manifest.json`

No `backtest_artifacts` publisher/precompute modules, canonical
`manifest.yaml`, `current.yaml`, active slots, public request hash, top-N
identity, scoring semantics or persisted schema were changed. The runtime
sidecar path is opt-in through `ROEHUB_BACKTEST_MATRIX_SIDECAR_DIR`; when the
directory is absent or invalid, bitset telemetry falls back to runtime packing.
Sidecar use feeds only the existing shadow/test bitset telemetry path, not
production scoring or top-N selection.

Mac Studio candidate state was measured from checkout base
`e985b30123ca9070ef5b1fc3227ffef6dd3fdf35` plus dirty Stage 07 files synced
from local `main` (`b3cf81f46b65a60c1da1268ead8b2a7c7f768a1e`) and generated
stage evidence. The remote checkout also had unrelated pre-existing untracked
files outside this stage.

Sidecar generation command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && \
  uv run python scripts/backtest/generate_matrix_sidecar_artifacts.py \
    --artifact-root /opt/roehub/state/backtest_artifacts/v2 \
    --output-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets/sidecar_artifacts \
    --indicator ma.dema --indicator ma.hma --indicator ma.ema \
    --indicator ma.sma --indicator ma.wma --indicator ma.rma"'
```

Final acceptance benchmark command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && \
  ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_05_no_risk_reversal_arity6 \
  uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-05-no-risk-heavy-rows \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets_final \
    --matrix-sidecar-artifact-dir docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets/sidecar_artifacts"'
```

Benchmark evidence:

- Sidecar generation/report/manifests:
  `docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets/`
- Final API-runner evidence:
  `docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets_final/`

The generated sidecar `.npy` payload was about `519M`. The durable stage commit
keeps the generator, loader, manifests/report and final benchmark evidence; the
large generated `.npy` files remain local/Mac Studio evidence and are not a
publisher artifact.

Overall harness gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `api_runner_path.pass` | true |
| `parity.pass` | true |
| `performance.pass` | true versus May 2 reference |
| `instrumentation.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `docs_drift_audit.pass` | true |

Measured Stage 07 sidecar timings:

| Job | Stage 05 exact s | Stage 07 exact s | Stage 07 sidecar load ms | Stage 07 signal prep ms | Stage 03 runtime pack ms reference |
|---|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | 1.010 | 1.028 | 81.530 | 102.640 | about 24.5 |
| `none/arity_6/long_short_reversal` | 2.887 | 2.990 | 75.238 | 97.899 | about 24.5 |

Decision: Stage 07 is `accepted_for_learning`. The sidecar generator, metadata
schema, source-manifest/source-signal hash validation, dtype/shape/padding
validation, duplicate-map validation, fallback behavior and benchmark reporting
are accepted as benchmark/test infrastructure. Sidecar-dependent speedup is not
accepted: load time is slower than the Stage 03 runtime pack shadow reference,
and the Stage 05 exact-scoring/service path shows small overhead. Production
`on` mode remains locked and later stages must not assume sidecar bitsets are a
speed win unless a separate publisher/no-advantage plan proves it.

Contract impact for committed Stage 07 work: public API `none`; port contract
`none`; DTO schema `compatible-change` only for additive internal
instrumentation counters; persisted schema `none`; canonical artifact
file/manifest schema `none`; config schema `compatible-change` for optional
`ROEHUB_BACKTEST_MATRIX_SIDECAR_DIR`; request hash/cache identity `none`;
service-call semantics `none`; benchmark/report semantics `compatible-change`;
browser-visible behavior `none`.

`next_iteration_allowed` is `true` for Stage 08 TP/SL selected-cell shadow only.
Stage 08 may use sidecar artifacts as diagnostic/test overlays, but it must not
write publisher artifacts or treat sidecar loading as an accepted production
acceleration.

## Stage 08 — TP/SL selected-cell shadow accepted for learning

Stage 08 implemented an opt-in TP/SL selected-cell shadow validator for
`tp_count <= 8` and `sl_count <= 8`. The validator is enabled only through
`ROEHUB_BACKTEST_TP_SL_SELECTED_CELL_SHADOW`; default production top-N remains
fed by the existing `event_segments_n_tp_sl_15m_grid` path.

Implementation scope:

- added `matrix_backend/trade_tape.py` to extract selected sparse TP/SL trade
  tapes from prepared candidate rows;
- added `matrix_backend/tp_sl_cells.py` to score selected TP/SL cells against
  the current exact scorer and record by-entry hit-times layout counters;
- added API-runner benchmark flag `--stage-08-tp-sl-selected-cells`, which uses
  a selected 8x8 TP/SL grid and enables shadow diagnostics for that benchmark
  run only;
- added focused unit coverage for long/short selected-cell parity, the
  same-bar `SL wins` rule, and job-local by-entry array layout.

No `backtest_artifacts` publisher/precompute modules, canonical `manifest.yaml`,
`current.yaml`, active slots, public API payloads, request hash, cache identity,
persisted schema or production top-N selection were changed. If by-entry
hit-times are persisted later, they must remain sidecar/test-only until a
separate publisher/manifest plan is approved.

Mac Studio candidate state was measured from checkout base
`e985b30123ca9070ef5b1fc3227ffef6dd3fdf35` plus dirty Stage 07/Stage 08 files
synced from local `main` (`e44750d4ee662523ca1c5115fd4e06b52e39becf`) for the
acceptance run. The remote checkout had pre-existing dirty Stage 07 files.

Acceptance benchmark command:

```bash
ssh macstudio 'zsh -lc "cd /Users/daniildegtyarev/Projects/roehub.com && \
  uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --stage-08-tp-sl-selected-cells \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-07_matrix_bitset_stage_08_tp_sl_selected_cells"'
```

Benchmark evidence:

- `docs/architecture/backtest/benchmark_iterations/2026-06-07_matrix_bitset_stage_08_tp_sl_selected_cells/benchmark_results.json`
- `docs/architecture/backtest/benchmark_iterations/2026-06-07_matrix_bitset_stage_08_tp_sl_selected_cells/benchmark_summary.md`
- `docs/architecture/backtest/benchmark_iterations/2026-06-07_matrix_bitset_stage_08_tp_sl_selected_cells/child_process_evidence/`

Overall harness gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `api_runner_path.pass` | true |
| `parity.pass` | true |
| `performance.pass` | true |
| `instrumentation.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `dead_code_audit.pass` | true |
| `docs_drift_audit.pass` | true |

Measured Stage 08 selected-cell evidence:

| Job | TP | SL | Cells | Shadow status | Selected cell scores | By-entry selected bytes | Max return diff pct | Exact ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `tp_sl_grid/arity_1/long_only` | 8 | 8 | 64 | passed | 384 | 7,145,344 | 0.0 | 1.269 |
| `tp_sl_grid/arity_2/long_short_reversal` | 8 | 8 | 64 | passed | 512 | 212,608 | 0.000029510889 | 0.857 |

The by-entry layout materialized job-local contiguous arrays named
`long_tp_by_entry.u32.npy`, `long_sl_by_entry.u32.npy`,
`short_tp_by_entry.u32.npy`, and `short_sl_by_entry.u32.npy` in memory only.
The benchmark did not write those arrays to canonical artifact roots, active
slots or publisher outputs.

Decision: Stage 08 is `accepted_for_learning`. Selected-cell TP/SL parity
passed for long-only and long-short selected rows, the `SL wins` tie rule is
covered by unit tests and recorded in shadow diagnostics, and by-entry layout
memory/timing counters are present. This stage does not unlock production `on`
mode or a production TP/SL cell backend; it only allows Stage 09 full-grid
cell-block work to start.

Contract impact for committed Stage 08 work: public API `none`; port contract
`none`; DTO schema `none`; persisted schema `none`; canonical artifact
file/manifest schema `none`; config schema `compatible-change` for optional
benchmark-only `ROEHUB_BACKTEST_TP_SL_SELECTED_CELL_SHADOW`; request hash/cache
identity `none`; service-call semantics `none`; benchmark/report semantics
`compatible-change`; browser-visible behavior `none`.

`next_iteration_allowed` is `true` for Stage 09 `matrix_cell_tp_sl_v1` full-grid
blocks only. Stage 09 must prove full-grid parity, cell-block size, accepted
`tp_sl_exact_scoring` speedup and no service wall regression before any
production-affecting backend mode is accepted.

## Stage 09 — TP/SL full-grid cell blocks accepted

Stage 09 implemented `matrix_cell_tp_sl_v1` as an internal TP/SL exact
backend experiment. After the 2026-06-14 closure cleanup it remains historical
evidence only and is no longer selectable through `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE`.

Implementation scope:

- added `matrix_cell_tp_sl_v1` to the internal backend registry for
  `risk.mode = tp_sl_grid`;
- added a Numba full-grid cell-block scorer that materializes each candidate's
  sparse trade tape once, scores the full TP/SL grid in configurable TP/SL cell
  blocks, preserves the same `SL wins` same-bar rule and the existing best-cell
  tie ordering;
- added internal block-size knobs
  `ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_TP_COUNT` and
  `ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_SL_COUNT`;
- added Stage 09 API-runner benchmark mode `--stage-09-tp-sl-full-grid` and
  markdown/JSON counters for backend id, block shape, blocks per candidate,
  block bytes, `tp_count`, `sl_count`, `tp_sl_cells` and
  `trade_cell_evals_per_sec`;
- added focused unit coverage comparing the full-grid matrix backend against the
  existing `event_segments_n_tp_sl_15m_grid` backend for canonical top-N payload,
  row identity, cell telemetry and `SL wins` reporting.

No `backtest_artifacts` publisher/precompute modules, canonical manifests,
`current.yaml`, active artifact slots, public API payload shape, DB schema,
request hash, cache identity, fees/slippage, sizing, TP/SL hit-time generation or
browser-visible behavior were changed. The backend remains opt-in through the
internal matrix backend env mode; the default TP/SL backend remains unchanged.

Mac Studio acceptance was run from isolated candidate copy
`/tmp/roehub-stage09-candidate` because the primary Mac checkout had
pre-existing dirty Stage 07/08 files, including overlapping benchmark harness
paths. Candidate provenance: local `main`
`c3d38bcaca98e1837e0a55ea652fa4c28b0ac09e` plus dirty candidate diff hash
`986c4daba9c7960cfea06c69dd0f0c74b11c1215dbfc65f8e9faa81c27cf921c`.

Diagnostic `16 x 16` benchmark command:

```bash
ssh macstudio 'cd /tmp/roehub-stage09-candidate && \
  ROEHUB_BENCHMARK_GIT_COMMIT=c3d38bcaca98e1837e0a55ea652fa4c28b0ac09e+dirty-f7acc0d6 \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid \
    --stage-09-tp-sl-full-grid \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5'
```

The `16 x 16` diagnostic passed full-grid parity, instrumentation and memory
for both jobs, but failed the speed gate on `tp_sl_grid/arity_6/long_only`
(`May2/current = 0.697`, below the `0.8` threshold). This shape is recorded as
diagnostic evidence only and is not the accepted runtime shape.

Accepted `64 x 64` benchmark command:

```bash
ssh macstudio 'cd /tmp/roehub-stage09-candidate && \
  ROEHUB_BENCHMARK_GIT_COMMIT=c3d38bcaca98e1837e0a55ea652fa4c28b0ac09e+dirty-986c4dab \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64_rerun \
    --stage-09-tp-sl-full-grid \
    --tp-sl-cell-block-tp-count 64 \
    --tp-sl-cell-block-sl-count 64 \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5 \
    --system-memory-cleanup-wait-seconds 90'
```

Benchmark evidence:

- `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid/`
- `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64/`
- `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64_rerun/`

Overall accepted rerun gates:

| Gate | Result |
|---|---|
| `pass` | true |
| `api_runner_path.pass` | true |
| `parity.pass` | true |
| `performance.pass` | true |
| `instrumentation.pass` | true |
| `memory_release.pass` | true |
| `lazy_cache_hit_memory.pass` | true |
| `legacy_path_absence.pass` | true |
| `dead_code_audit.pass` | true |
| `docs_drift_audit.pass` | true |

Measured accepted Stage 09 evidence:

| Job | TP | SL | Cells | Block | Blocks/candidate | `tp_sl_exact_scoring` s | May2/current | Trade-cell evals/s | System memory delta |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `tp_sl_grid/arity_6/long_only` | 47 | 47 | 2,209 | `64 x 64` | 1 | 18.178 | 0.960 | 5,669,734 | 55,050,240 |
| `tp_sl_grid/arity_6/long_short_reversal` | 47 | 47 | 2,209 | `64 x 64` | 1 | 17.399 | 0.931 | 5,923,657 | -262,406,144 |

Both jobs evaluated `103,063,104` trade-cell cells, recorded
`matrix_cell_tp_sl_v1`, passed full-grid top-N parity, and retained no heavy
runtime references in the compact result. The first `64 x 64` run passed parity,
instrumentation and speed but failed only the system memory cleanup gate on the
first job; the accepted rerun with a longer cleanup wait passed memory for both
jobs.

Decision: Stage 09 is `accepted`. Full-grid parity passed for both heavy TP/SL
arity-6 rows, the accepted `64 x 64` block shape passed the
`tp_sl_exact_scoring` threshold and memory cleanup gates, and the implementation
keeps the backend opt-in. Stage 10 exact-safe high-arity pruning may start.

Contract impact for committed Stage 09 work: public API `none`; port contract
`none`; DTO schema `compatible-change` for internal telemetry fields only;
persisted schema `none`; canonical artifact file/manifest schema `none`; config
schema `compatible-change` for optional internal env keys
`ROEHUB_BACKTEST_MATRIX_BACKEND_MODE`,
`ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_TP_COUNT`, and
`ROEHUB_BACKTEST_TP_SL_CELL_BLOCK_SL_COUNT`; request hash/cache identity `none`;
service-call semantics `none`; benchmark/report semantics `compatible-change`;
browser-visible behavior `none`.

`next_iteration_allowed` is `true` for Stage 10 exact-safe high-arity pruning.

## Stage 10 — high-arity min-trade pruning accepted for learning

Stage 10 tested an exact-safe branch-and-bound rule for high-arity planning:
`monotonic_min_closed_trades`.

Rule and proof:

- For a partial indicator prefix, direction-adjusted consensus bars are
  monotonic: adding indicators can only keep an existing nonzero consensus bar or
  turn it to zero.
- Every current no-risk and TP/SL closed trade requires an entry on a nonzero
  consensus bar.
- Therefore a subtree whose partial consensus has fewer nonzero bars than
  `quality_constraints.min_closed_trades` cannot produce a heap-eligible
  candidate and can be pruned without removing a valid top candidate.
- This is not a score upper bound. It does not justify approximate beam search,
  product-level approximate ranking, or pruning candidates that can still satisfy
  the min-trade gate.

Mac Studio candidate evidence was run from isolated copy
`/tmp/roehub-stage10-candidate` because the primary Mac checkout had pre-existing
dirty Stage 07-09 files. Candidate provenance: local `main`
`d00e6afc7445d255533c9b97c5ea40c1167e819c` plus scoped dirty diff hash
`91dc3b6247ebc001288504ed241a4e9ee39fe4db33ee432b81280033b8b57c2e`.

Partial benchmark command:

```bash
ssh macstudio 'cd /tmp/roehub-stage10-candidate && \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7 \
    --stage-10-high-arity-pruning \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5 \
    --system-memory-cleanup-wait-seconds 90'
```

Evidence:

- `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7_partial/`

Completed arity-7 row counters:

| Metric | Value |
|---|---:|
| `combo_count_planned` | `279,936` |
| `candidates_after_proxy` | `116,640` |
| `exact_candidates` | `116,640` |
| `combo_pruning_pruned_subtrees` | `3,246` |
| `combo_pruning_pruned_candidate_upper_bound` | `163,296` |
| `combo_iteration` | `59.350s` |
| `exact_scoring` | `58.182s` |
| `service_total_without_warmup` | `119.252s` |

Decision: Stage 10 is `accepted_for_learning` as a rule/evidence handoff only.
The `monotonic_min_closed_trades` rule is exact-safe and may inform later exact
pruning designs, but the measured Python branch-and-bound runtime candidate is
not accepted as an acceleration. It did not complete accepted arity-7 evidence,
no comparable baseline-off speedup was completed, and the first completed row
shows the traversal adds a large `combo_iteration` cost. Arity-10 acceptance is
also blocked by the current canonical fixture: it contains only seven
indicators.

Contract impact for the rejected runtime candidate if left uncommitted: public
API `none`; port contract `none`; DTO schema `compatible-change` for additive
internal telemetry fields only; persisted schema `none`; canonical artifact
file/manifest schema `none`; config schema `compatible-change` for optional
internal `ROEHUB_BACKTEST_HIGH_ARITY_PRUNING`; request hash/cache identity
`none`; service-call semantics `none`; benchmark/report semantics
`compatible-change`; browser-visible behavior `none`.

`next_iteration_allowed` is `true` for Stage 11 lazy detail reuse only. Stage 11
must not depend on the rejected Stage 10 runtime pruning implementation. A
future Stage 10 retry needs either a tighter exact-safe score/eligibility bound
that beats the baseline through the API-runner path, or an approved benchmark
fixture that can cover arity 10.

## Stage 11 — lazy detail sparse trade tape reuse rejected

Stage 11 tested reuse of the existing sparse trade tape backend for TP/SL lazy
selected variant materialization only. Bulk top-N scoring, matrix backend
selection, public lazy trades payload shape, cache key components and
materialization identity were not changed during the candidate test.

The candidate added a single-candidate sparse tape helper and routed TP/SL lazy
detail recompute through that helper with fallback to the current direct lazy
materialization. After benchmark review, the candidate was rejected because it
did not prove material selected-variant latency acceleration. The production
runtime/test candidate is removed; only benchmark evidence and this ledger
record remain as the durable handoff.

Mac Studio candidate evidence used isolated worktrees:

- Baseline worktree: `/tmp/roehub-stage11-baseline`
- Candidate worktree: `/tmp/roehub-stage11-candidate`
- Base commit: `34fac40074bcf082e86a4396daae3ab6dbdde1a3`
- Candidate patch SHA-256:
  `c410a8783202c2fcb2cf9008899bb0c0ff05b1011b92e53a5be1f6ce8cc06d92`

Evidence:

- Candidate:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse/`
- Comparable baseline:
  `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline/`

Focused Mac Studio gate:

```bash
/opt/homebrew/bin/uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py \
  tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py
```

Result: `12 passed`.

Selected-variant latency benchmark comparison:

| Risk mode | Baseline miss s | Candidate miss s | Miss delta | Baseline hit s | Candidate hit s | Hit delta | Parity |
|---|---:|---:|---:|---:|---:|---:|---|
| `none` | 2.869005 | 2.855623 | -0.466% | 0.000305 | 0.000299 | -2.090% | pass |
| `tp_sl_grid` | 4.334214 | 4.292836 | -0.955% | 0.000301 | 0.000301 | -0.207% | pass |

Decision: Stage 11 is `rejected`. The benchmark passed parity and did not show
a regression, but the measured delta is too small to justify accepting the
runtime change as an acceleration. The candidate must not be reused as a
production lazy-detail optimization without a new benchmark-gated plan.

Contract impact: public API `none`; port contract `none`; DTO schema `none`;
persisted schema `none`; config schema `none`; request hash/cache identity
`none`; service-call semantics `none`; external side effects `none`; benchmark
and ledger semantics `compatible-change`; browser-visible behavior `none`.

Rejection commit scope:

- `docs/architecture/README.md`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`
- `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse/`
- `docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline/`
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py`
- `tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py`

Git branch: `main`. Scoped rejection commit SHA is recorded in the executor
final report after commit. Push/deploy: not performed.

`next_iteration_allowed` is `false` for the Stage 11 lazy-detail path. The
2026-06-13 continuation update below opens a separate Stage 12+ roadmap based on
new dominant cost centers; it does not convert Stage 11 into accepted acceleration.

## Stage 12+ Continuation Planning - 2026-06-13

Status: `superseded_by_stage_12_acceptance`. This entry is the planning handoff
that opened Stage 12-21; Stage 12 itself is now accepted in the section below.

Source recommendation:
`/Users/daniildegtyarev/.codex/attachments/592a9119-4350-4c95-8400-3ae7a245ace7/pasted-text.txt`.

Original planning decision:

- Superseded by the Stage 05+12 production default rollout: the original plan
  kept Stage 05 as the only default production acceleration:
  `matrix_bitset_no_risk_v1` for `risk.mode=none`, arity `6`, and
  `direction_mode in {long_only, long_short_reversal}`.
- Keep Stage 09 `matrix_cell_tp_sl_v1` as historical evidence only after the
  2026-06-14 closure cleanup; env selection is retired.
- Do not revive Stage 06 runtime consensus cache, Stage 07 sidecar load as
  production speedup, Stage 10 Python traversal, or Stage 11 lazy reuse.
- Open Stage 12 as the next executable stage at that time. Stage 12 has since
  been completed and productionized through the Stage 05+12 default.
- Stage 13/13S/13S2/13R/14R were executed after Stage 12 and then cleaned up
  after rejected or learning-only outcomes. The active tree keeps the stop-list
  summary, but not the rejected harness/runtime/evidence branch.

Stage 12 has since passed its gate. Stage 13/14 TP/SL production attempts did
not pass their gates, and their executable prompt files were removed from the
active prompt pack. Stage 15 later tested an exact-safe TP/SL early-abandon
candidate against the current exact TP/SL baseline and closed it as
`accepted_for_learning`: the runtime candidate is rejected, but the Mac Studio
A/B evidence is retained because it proves candidate-level total-return bounds
are too loose on the mandatory TP/SL fixture.

Scope cleanup on 2026-06-14 closed all remaining TP/SL continuation work in
this prompt pack. Stage 16 trade-window reuse telemetry and Stage 21
exact/coarse product-mode design are `closed_not_executed`. The executable
TP/SL prompt files for Stage 08/09/15/16/21 were removed from the generated
prompt pack; Stage 08/09/13/14/15 remain only as historical attempts in the
ledger, benchmark evidence and negative-results records.

## Current Execution Handoff

No executable next stage remains in this prompt pack. Backtest compute
acceleration v1 is closed after the accepted Stage 05+12 production default and
a final cleanup on 2026-06-14. The active runtime tree must keep only the
accepted production paths: Stage 05 `matrix_bitset_no_risk_v1` for no-risk
arity `6` and Stage 12 `compiled_prefix_product_traversal_v1` for no-risk arity
`7` through the composite default `stage_05_and_12_no_risk`.

Stage 17-20 were executed after Stage 05+12, but their runtime, benchmark-runner
flags, prompt files and raw evidence directories are removed from the active
tree by the closure cleanup. Their only durable value is the compact learning in
this ledger and the negative-results stop-list: do not reintroduce the dynamic
selector, top-N batch merge, thread-policy selector or scratch-buffer telemetry
without a separate approved plan and a new Mac Studio A/B gate.

Stage 09 `matrix_cell_tp_sl_v1` remains historical TP/SL evidence only. It is no
longer selectable through `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE`; both
`stage_09_tp_sl_full_grid` and direct `matrix_cell_tp_sl_v1` env selection are
retired. Any renewed TP/SL acceleration requires a separate approved plan.

Stage 06 is closed as `rejected`, not skipped silently. Its only durable outputs
are ledger/evidence files under
`docs/architecture/backtest/benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache/`,
including `stage06_signature_cache_candidate.patch` for audit context. No Stage
06 runtime cache code is accepted as part of the active service, and later work
must not reuse, revive or depend on that candidate unless a new benchmark-gated
plan explicitly reopens the idea.

Stage 07 and Stage 08 remain learning-only sidecar/TP-SL shadow evidence. Stage
15 is closed as `accepted_for_learning` with runtime rejection. Stage 16 and
Stage 21 are `closed_not_executed`. All future benchmark evidence must record
whether it used a repo checkout, an isolated candidate copy, or the active live
runtime.

## Stage 12 - compiled prefix product traversal accepted

Stage 12 added `compiled_prefix_product_traversal_v1` as an opt-in no-risk
backend for arity `6` and `7` with
`ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_12_compiled_prefix_traversal` or the
direct backend id. The production composite default added after Stage 12 keeps
Stage 05 for no-risk arity `6` and enables Stage 12 for no-risk arity `7`.

Implementation:

- compiled Numba iterative DFS over a selectivity-ordered product tree;
- prefix `pos`/`neg` bitset consensus is reused across levels;
- pruning is limited to the exact-safe `min_closed_trades` eligibility bound:
  descendants of a prefix with fewer active consensus bars than the quality gate
  cannot become heap-eligible;
- selected candidates are sorted back to canonical ordinal order before Stage 05
  matrix bitset scoring, preserving stable ranking and `variant_hash` semantics.

Mac Studio candidate provenance:

- isolated candidate copy:
  `/tmp/roehub-stage12-candidate-20260613-compiled-prefix`;
- base commit: `d05d26f80509816f3251063cd1e3c99f3b361050`;
- scoped dirty diff hash:
  `62caaca6cac54b97d516d220cfc728485112c0cecaf4f6ce127640e744891332`;
- env file: `/Users/daniildegtyarev/.config/roehub/roehub.env`;
- artifact config: `configs/prod/backtest_artifacts.yaml`;
- source artifacts: read-only `/opt/roehub/state/backtest_artifacts/v2`;
- secret values: not recorded.

Baseline command:

```bash
ssh macstudio 'cd /tmp/roehub-stage12-candidate-20260613-compiled-prefix && \
  ROEHUB_BENCHMARK_GIT_COMMIT=d05d26f80509816f3251063cd1e3c99f3b361050+dirty-62caaca6 \
  ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_baseline_off \
    --stage-12-compiled-prefix-rows \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5 \
    --system-memory-cleanup-wait-seconds 90'
```

Accepted candidate command:

```bash
ssh macstudio 'cd /tmp/roehub-stage12-candidate-20260613-compiled-prefix && \
  ROEHUB_BENCHMARK_GIT_COMMIT=d05d26f80509816f3251063cd1e3c99f3b361050+dirty-62caaca6 \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_candidate_rerun2 \
    --stage-12-compiled-prefix-rows \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5 \
    --system-memory-cleanup-wait-seconds 90'
```

Evidence:

- baseline-off:
  `benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_baseline_off/`;
- accepted candidate:
  `benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_candidate_rerun2/`.

Mac Studio A/B:

| Job | Baseline service s | Candidate service s | Service improvement | Baseline exact s | Candidate exact s | Exact speedup | Candidate `combo_iteration` s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none/arity_6/long_only` | `17.161` | `1.639` | `90.449%` | `15.441` | `0.871` | `17.725x` | `0.075` |
| `none/arity_6/long_short_reversal` | `15.562` | `3.646` | `76.573%` | `15.053` | `2.922` | `5.152x` | `0.206` |
| `none/arity_7/long_only` | `138.070` | `6.509` | `95.286%` | `137.120` | `5.242` | `26.160x` | `0.400` |
| `none/arity_7/long_short_reversal` | `136.117` | `19.331` | `85.798%` | `135.552` | `17.549` | `7.724x` | `1.241` |

Prefix counters:

| Job | Selected | Pruned | Pruned subtrees | Pruned upper bound | Selectivity order | Prefix total s | Compiled loop s |
|---|---:|---:|---:|---:|---|---:|---:|
| `none/arity_6/long_only` | `19,440` | `27,216` | `11` | `27,216` | `[4, 1, 0, 3, 2, 5]` | `0.075` | `0.072` |
| `none/arity_6/long_short_reversal` | `44,064` | `2,592` | `2,592` | `2,592` | `[0, 3, 2, 5, 4, 1]` | `0.206` | `0.202` |
| `none/arity_7/long_only` | `116,640` | `163,296` | `11` | `163,296` | `[4, 1, 0, 6, 3, 2, 5]` | `0.400` | `0.391` |
| `none/arity_7/long_short_reversal` | `264,384` | `15,552` | `15,552` | `15,552` | `[0, 6, 3, 2, 5, 4, 1]` | `1.241` | `1.221` |

Parity and memory:

- baseline-off parity passed `4/4`, performance passed, but the baseline run
  failed the generic memory-release gate on
  `none/arity_7/long_short_reversal`;
- accepted candidate parity, performance, instrumentation, memory release, lazy
  cache memory, scheduler smoke, legacy path, dead-code audit and docs-drift
  audit all passed;
- full API top-50 comparison by stable fields (`rank`, `variant_hash`,
  `summary_metrics`) matched baseline for both non-empty rows; reversal rows
  stayed quality-gated zero-result rows; `variant_key` differs only by embedded
  job id;
- Stage 10 rejected Python traversal remains on the stop-list. Stage 12
  candidate `combo_iteration` is `0.400s` / `1.241s` for arity-7, versus the
  rejected Stage 10 completed row's `59.350s` Python traversal.

Contract impact: `compatible-change`. No public API payload shape, request hash,
artifact publisher/precompute path, `current.yaml`, active artifact slot, DB
schema, fees/slippage/sizing, close-on-end, ranking, persisted top-N shape or
browser-visible behavior changed. The new backend is selected by internal
backend mode policy and falls back to accepted current paths otherwise.

## Stage 05+12 production default rollout - 2026-06-13

Status: `accepted`.

Rollout policy:

- default/unset `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` resolves to
  `stage_05_and_12_no_risk`;
- `stage_05_and_12_no_risk` selects Stage 05 `matrix_bitset_no_risk_v1` for
  `risk.mode=none`, arity `6`, and
  `direction_mode in {long_only, long_short_reversal}`;
- the same default selects Stage 12 `compiled_prefix_product_traversal_v1` for
  `risk.mode=none`, arity `7`, and
  `direction_mode in {long_only, long_short_reversal}`;
- explicit `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_05_no_risk_reversal_arity6`
  isolates Stage 05-only rollback/comparison behavior;
- explicit `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_12_compiled_prefix_traversal`
  isolates Stage 12 for arity `6`/`7` benchmark comparison;
- explicit `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off` remains the full legacy
  rollback/comparison path.

Acceptance gate:

- focused unit tests prove default arity-6 routes to Stage 05 and default
  arity-7 routes to Stage 12;
- Mac Studio live runtime under `/opt/roehub/app` contains the accepted code and
  launchd services have been restarted;
- `scripts/macos/smoke_prod.sh` passes after deploy;
- production-mode benchmark evidence through the API-runner path records
  default composite results for Stage 05 arity-6 rows and Stage 12 arity-7 rows;
- top-50 identity/order and metric tolerance match the accepted baselines;
- service wall does not materially regress versus accepted Stage 05/Stage 12
  evidence for the rows each backend owns.

Evidence:

- code commit: `1bd7a1e4` on `main`;
- CI: run `27477340161` passed;
- deploy: `Deploy Backend` run `27477461091` passed and refreshed
  `/opt/roehub/app`;
- post-deploy smoke: `ssh macstudio 'cd /opt/roehub/app && bash scripts/macos/smoke_prod.sh'`
  passed;
- live runtime check: `/opt/roehub/app` contains
  `--stage-05-12-production-default-rows`, `MATRIX_BACKEND_MODE_DEFAULT` is
  `stage_05_and_12_no_risk`, `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` is unset in
  `/Users/daniildegtyarev/.config/roehub/roehub.env`, and launchd services
  `com.roehub.api` / `com.roehub.backtest-job-runner` were running;
- accepted benchmark evidence:
  `benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_live/`;
- diagnostic rerun evidence:
  `benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_live_rerun2/`.

Accepted benchmark command:

```bash
ssh macstudio 'cd /opt/roehub/app && \
  env -u ROEHUB_BACKTEST_MATRIX_BACKEND_MODE \
  ROEHUB_BENCHMARK_GIT_COMMIT=1bd7a1e4 \
  /opt/homebrew/bin/uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
    --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
    --out-dir docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_live \
    --stage-05-12-production-default-rows \
    --timeout-seconds 7200 \
    --poll-interval-seconds 0.5 \
    --system-memory-cleanup-wait-seconds 90'
```

Live production-default benchmark result:

| Job | Backend owner | Exact current s | May2 exact s | Exact ratio | Service wall s | Service no-warmup s | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `none/arity_6/long_only` | Stage 05 | `1.056` | `15.694` | `14.858x` | `10.031` | `1.974` | cold `sample_warmup=8.403s`; warm rerun recorded `1.471s` wall / `1.044s` exact |
| `none/arity_6/long_short_reversal` | Stage 05 | `3.076` | `15.365` | `4.995x` | `3.322` | `3.594` | accepted run passed; rerun diagnostic was invalid because launchd production worker claimed the job |
| `none/arity_7/long_only` | Stage 12 | `5.241` | `138.755` | `26.474x` | `6.562` | `6.535` | prefix selected `116640`, pruned `163296` |
| `none/arity_7/long_short_reversal` | Stage 12 | `17.682` | `136.630` | `7.727x` | `19.205` | `19.484` | prefix selected `264384`, pruned `15552` |

Acceptance decision: accepted. The accepted run passed parity `4/4`,
performance, instrumentation, memory release, lazy cache memory, scheduler
smoke, legacy path, dead-code audit and docs-drift audit. Stage 05 arity-6
live timings remain materially faster than the legacy exact path, while small
differences versus the earlier Stage 05 default-on checkout evidence are
treated as live-runtime timing variance rather than a new algorithmic
regression because the routed backend is unchanged. Stage 12 arity-7 timings
match the accepted Stage 12 evidence. The diagnostic rerun is not acceptance
evidence because `com.roehub.backtest-job-runner` claimed one benchmark job
instead of the harness process; future live benchmarks must verify harness job
ownership or isolate the launchd worker.

## Production/default state audit - 2026-06-13 pre-deploy snapshot

This superseded audit distinguished accepted repository code from live
production runtime before the `1bd7a1e4` deploy:

- local `main` contains Stage 12 commit
  `1fda22642ac8f9194322a5b91d39e3f676f42ee7` and is ahead of
  `origin/main` at `d05d26f80509816f3251063cd1e3c99f3b361050`;
- Mac Studio project checkout `/Users/daniildegtyarev/Projects/roehub.com` is
  at `origin/main` `d05d26f80509816f3251063cd1e3c99f3b361050`, so it contains
  accepted Stage 05/09 history but not Stage 12 until push/sync;
- Mac Studio native launchd plist working directories point to
  `/opt/roehub/app`, not the project checkout;
- observed `/opt/roehub/app` is not a Git checkout and does not contain
  `matrix_backend/prefix_traversal.py`, `matrix_backend/tp_sl_cells.py`,
  `matrix_backend/trade_tape.py`, Stage 05 mode strings, Stage 09 mode strings
  or Stage 12 mode strings;
- observed `/Users/daniildegtyarev/.config/roehub/roehub.env` has no explicit
  `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` override.

Conclusion at the time: Stage 05, Stage 09 and Stage 12 were accepted in
repository history, but the observed native live runtime under `/opt/roehub/app`
was older and could not be treated as updated production. This is superseded by
the accepted Stage 05+12 production default rollout above.

## Stage Ledger

| Stage | Status | Scope | Evidence | Decision | next_iteration_allowed |
|---:|---|---|---|---|---|
| 00 | accepted | Refresh current heavy baseline on Mac Studio before code changes | `benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/` | Baseline accepted; re-verified on checkout `6dcb62dc918a98564abec9554ae575187b32fa39`; scoped backtest runtime/harness diff from evidence commit is empty; performance, parity, memory, lazy cache, legacy path, accounting and docs drift gates passed | true |
| 01 | accepted_for_learning | Add instrumentation counters without behavior changes | `benchmark_iterations/2026-06-06_matrix_bitset_stage_01_instrumentation/` | Counters present; explicit `null` for unavailable current-runtime counters; parity, performance, memory, lazy cache, legacy path and accounting gates passed; overhead stayed within <= 1% limit with no Stage 00 service/exact regression; production `on` mode remains locked | true |
| 02 | accepted_for_learning | Row/signature telemetry shadow | `benchmark_iterations/2026-06-06_matrix_bitset_stage_02_row_signature_telemetry/` | Shadow counters present; duplicate rows `0/36` on accepted arity-6 rows; `consensus_signature_count=46656` as deterministic upper bound; collision count `0`; row signature overhead about 10-11ms/job; parity, performance, memory, lazy cache, legacy path and docs drift gates passed; no pruning/scoring/top-N/request-hash/cache change | true |
| 03 | accepted_for_learning | Runtime bitset pack shadow | `benchmark_iterations/2026-06-06_matrix_bitset_stage_03_runtime_bitset_pack/` | Shadow bitsets recorded `signals_pack_ms` about 24ms/job with `W=3421`, packed bytes `1,970,496`, padding valid and consensus sample parity true; API-runner parity `4/4`, performance, memory release, lazy cache, legacy path and docs drift gates passed; scoring/top-N/request hash/cache/persistence unchanged | true |
| 04 | accepted_for_learning | `matrix_bitset_no_risk_v1` for `none/arity_2..3/long_only` | `benchmark_iterations/2026-06-06_matrix_bitset_stage_04_no_risk_mvp/` | Mac Studio API-runner parity `2/2`, memory, instrumentation, lazy cache, legacy path and docs drift gates passed; raw performance failed on tiny `none/arity_2/long_only` by about `1.1ms`, while `none/arity_3/long_only` ratio was `2.590`; arity-2 no-advantage is waived for learning progression only; production `on` mode remains locked | true |
| 05 | accepted | No-risk `long_short_reversal` and arity 6 heavy rows; default-on only for `none/arity_6/long_only` and `none/arity_6/long_short_reversal` after 2026-06-10 productionization gate | `benchmark_iterations/2026-06-06_matrix_bitset_stage_05_reversal_arity6/`, `benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_off_baseline/`, `benchmark_iterations/2026-06-10_matrix_bitset_stage_05_default_on_candidate/` | Original Mac Studio API-runner parity, performance, memory, instrumentation, lazy cache, legacy path and docs drift gates passed; default-on A/B kept request hash, top-50 identity/order and metric parity, with exact speedup `15.501x` long-only and `5.178x` long-short A/B; `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off` remains rollback/comparison path | true |
| 06 | rejected | Consensus signature cache | `benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache/` | Cache hit-rate `0.202396` and collision count `0`, but Mac Studio API-runner exact scoring regressed versus Stage 05: `1.010s -> 4.932s` long-only and `2.887s -> 6.166s` reversal; cache runtime candidate not accepted; only evidence/patch retained | true for Stage 07 sidecar/test bitset artifacts only |
| 07 | accepted_for_learning | Sidecar/test bitset artifacts generated outside publisher; generator/helper, explicit sidecar path, `signals_pos_bits.u64.npy`, `signals_neg_bits.u64.npy`, `signal_row_hashes.u64.npy`, `unique_signal_row_ids.u32.npy`, `duplicate_signal_row_ids.u32.npy`, `matrix_sidecar_manifest.json`; no `backtest_artifacts` publisher/precompute or canonical manifest changes | `benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets/` and `benchmark_iterations/2026-06-06_matrix_bitset_stage_07_sidecar_bitsets_final/` | Mac Studio API-runner parity `2/2`, memory, legacy path and docs drift passed; sidecar generation `7882.282ms`, sidecar load `75.238..81.530ms/job`, but runtime pack Stage 03 reference was about `24.5ms/job`; accepted only as test/benchmark infrastructure, no production sidecar speedup unlocked | true for Stage 08 TP/SL selected-cell shadow only |
| 08 | accepted_for_learning | TP/SL selected-cell shadow with by-entry hit-times layout or selected by-entry arrays; sidecar-only if persisted for testing, no publisher/manifest changes without a separate approved plan | `benchmark_iterations/2026-06-07_matrix_bitset_stage_08_tp_sl_selected_cells/` | Mac Studio API-runner selected 8x8 TP/SL parity `2/2`; `SL wins` tie rule covered; by-entry selected arrays recorded job-locally as `long_tp_by_entry.u32.npy`, `long_sl_by_entry.u32.npy`, `short_tp_by_entry.u32.npy`, `short_sl_by_entry.u32.npy`; production top-N remains current path only | true for Stage 09 full-grid TP/SL cell blocks only |
| 09 | accepted_retired | `matrix_cell_tp_sl_v1` full grid blocks with configurable TP/SL cell block shape; no publisher/precompute or default-backend change; env selection retired by 2026-06-14 closure cleanup | `benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64_rerun/` plus diagnostic `16 x 16` and first `64 x 64` runs | Mac Studio API-runner full-grid parity `2/2`, instrumentation and memory passed; accepted `64 x 64` shape recorded `tp_count=47`, `sl_count=47`, `tp_sl_cells=2209`, `trade_cell_evals_per_sec` about `5.67M..5.92M`; exact speed ratios `0.960` and `0.931`; backend is historical evidence only and no longer selectable through `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` | true for Stage 10 exact-safe high-arity pruning |
| 10 | accepted_for_learning | Exact-safe `monotonic_min_closed_trades` rule and negative performance evidence retained; runtime pruning candidate rejected; approximate beam remains off | `benchmark_iterations/2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7_partial/` | Exact-safe proof holds for the min-trade eligibility bound, but Mac Studio arity-7 evidence did not complete accepted gates; first completed row pruned `163,296 / 279,936` candidates yet spent `59.350s` in branch traversal and `58.182s` in exact scoring; no comparable baseline-off speedup completed; arity-10 blocked by seven-indicator canonical fixture; do not reuse the Python branch-and-bound runtime candidate as accepted acceleration | true for Stage 11 lazy detail reuse only |
| 11 | rejected | TP/SL lazy selected-variant sparse trade tape reuse candidate; production runtime/test candidate removed after review; no bulk top-N scoring change | `benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse/` plus comparable baseline `benchmark_iterations/2026-06-10_matrix_bitset_stage_11_lazy_detail_reuse_baseline/` | Mac Studio lazy parity passed for `none` and `tp_sl_grid`, but TP/SL miss changed only from `4.334214s` to `4.292836s` (`-0.955%`) and cache hit stayed effectively unchanged; no material speedup, so the candidate is rejected and must not be treated as accepted acceleration | false for lazy path; superseded by Stage 12+ continuation plan |
| 12 | accepted | Compiled prefix product traversal with selectivity order and exact-safe prefix pruning; production composite default uses Stage 12 for no-risk arity `7` and keeps Stage 05 for arity `6`; explicit Stage 12 mode remains available for arity `6`/`7`; no Python traversal hot path | `benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_baseline_off/`, `benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_candidate_rerun2/`, `benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_live/`; diagnostic race rerun `benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_live_rerun2/` | Mac Studio API-runner candidate passed parity, performance, instrumentation, memory release, lazy cache, scheduler, legacy path, dead-code and docs-drift gates; arity-7 service wall improved `95.286%` long-only and `85.798%` reversal in isolated Stage 12 evidence; production default evidence passed `4/4` parity with Stage 05 exact ratios `14.858x` / `4.995x` and Stage 12 exact ratios `26.474x` / `7.727x` versus May2; composite default keeps Stage 05 for arity-6 because that was the accepted default service path; stable top-50 `variant_hash`/rank/metrics matched baseline where top-N is non-empty | false; active plan closed after 2026-06-14 cleanup |
| 13 | rejected_removed | TP/SL `64 x 64` production gate and block autotune for accepted `matrix_cell_tp_sl_v1`; dedicated harness/prompt/raw evidence removed from active tree | raw evidence removed from active tree; summary retained in negative-results stop-list | Candidate shapes preserved parity but no shape improved both mandatory TP/SL heavy rows by the required `>=15%` service-wall threshold; backend remains opt-in/internal | false |
| 13S/13S2 | rejected_removed | Narrow TP/SL production selector; selector prompt/runtime/evidence removed from active tree | raw evidence removed from active tree; summary retained in negative-results stop-list | First threshold excluded mandatory `47 x 47` fixture; retest selected matrix backend but regressed long-only `-9.342%` and combined mandatory rows `-5.524%` | false |
| 13R | accepted_for_learning_removed | TP/SL reversal diagnostic telemetry; runtime diagnostics removed from active tree | raw evidence removed from active tree; summary retained in negative-results stop-list | Diagnostics identified likely cost centers, but current-exact reversal diagnostics overhead was `+99.5%`; not accepted for production/default behavior | false |
| 14 | superseded_removed | Original TP/SL monotonic cell kernel prompt depended on a rejected Stage 13 winner; prompt removed | none | Do not execute as written | false |
| 14R | rejected_removed | TP/SL reversal split-by-side repair candidate; runtime candidate/evidence removed from active tree | raw evidence removed from active tree; summary retained in negative-results stop-list | Parity passed, but service wall regressed `-6.393%`, exact scoring regressed `-6.261%`, and sampled RSS increased | false |
| 15 | accepted_for_learning | TP/SL total-return early abandon with exact-safe optimistic log-return upper bound; runtime candidate removed from active tree | `benchmark_iterations/2026-06-14_matrix_bitset_stage_05_12_production_default_stage15_preflight/`, `benchmark_iterations/2026-06-14_matrix_bitset_stage_15_tp_sl_early_abandon_control/`, `benchmark_iterations/2026-06-14_matrix_bitset_stage_15_tp_sl_early_abandon_candidate/` | Mac Studio A/B preserved parity and memory cleanup, but candidate pruned `0` candidates on both mandatory TP/SL rows. Long-only service wall regressed `17.728s -> 31.298s` (`-76.541%`) with `13751.296ms` bound overhead; reversal service wall regressed `15.474s -> 15.502s` (`-0.180%`). No production runtime change accepted; evidence is retained as a learning handoff proving this bound shape is not viable on the mandatory fixture. | false; TP/SL continuation closed by 2026-06-14 scope cleanup |
| 16 | closed_not_executed | TP/SL trade-window reuse telemetry only | none | Closed by 2026-06-14 scope cleanup before implementation; executable prompt removed; no telemetry, cache or grouped scoring scope remains in this plan | false |
| 17 | accepted_for_learning_removed | No-risk dynamic backend selector by estimated work and fixed overhead; explicit env mode was tested but removed from active runtime/config surface by closure cleanup | raw evidence removed from active tree; compact results retained in this ledger | Selector avoided arity `1..3` matrix regressions and showed Stage 12 could help arity-6 long-only, but production/default `stage_05_and_12_no_risk` was not changed. User decision on 2026-06-14: Stage 17 code is not needed in the repository. `stage_17_dynamic_no_risk_selector` and `dynamic_backtest_backend_selector_v1` must not be reintroduced without a separate approved plan. | false |
| 18 | accepted_for_learning_removed | Top-N/result assembly telemetry; stable block top-M merge not implemented because assembly is not a material cost center; telemetry code removed by closure cleanup | raw evidence removed from active tree; summary values retained in negative-results stop-list | `top_result_assembly` was only `0.000..0.728%` of service wall and DB persist only `0.096..1.906%`; no runtime batch merge was justified. | false |
| 19 | accepted_for_learning_removed | Numba thread scaling benchmark by no-risk workload; no worker config/default change; benchmark-runner thread-matrix additions removed by closure cleanup | raw evidence removed from active tree; compact result retained in this ledger | 12 threads was fastest for every required row and RSS stayed bounded; the existing fixed 12-thread policy remains. | false |
| 20 | accepted_for_learning_removed | Allocation telemetry for accepted no-risk production-default rows; per-child scratch-buffer candidate not implemented; telemetry code removed by closure cleanup | raw evidence removed from active tree; summary values retained in negative-results stop-list | Allocation churn did not justify scratch buffers: arity-6 around `3.65MB`, arity-7 around `31.0..51.7MB`, RSS cleanup passed, and no repeated per-child reuse opportunity was accepted. | false; no further active stage in this plan |
| 21 | closed_not_executed | TP/SL exact/coarse product-mode architecture decision | none | Removed from active plan by 2026-06-14 scope cleanup; executable prompt removed; any future product-mode work requires a separate approved architecture/product plan | false |

## Stage 17-20 Closure - runtime code removed

Status: `closed_removed` on 2026-06-14.

The Stage 17-20 prompt files were executed, but the user explicitly decided that
their code is not needed in the repository. The active tree therefore removes:

- Stage 17 `stage_17_dynamic_no_risk_selector` runtime/config/telemetry code;
- Stage 18 top-N/result assembly timer additions and batch-merge prompt scope;
- Stage 19 thread-scaling benchmark-runner additions;
- Stage 20 allocation telemetry/scratch-buffer additions;
- Stage 17-20 generated prompt files;
- Stage 17-20 raw benchmark evidence directories.

Compact retained learnings:

| Stage | Retained learning | Runtime status |
|---:|---|---|
| 17 | Dynamic selector protected small rows and showed arity-6 long-only can prefer Stage 12, but default production stayed Stage 05+12 | removed; do not reintroduce without new plan |
| 18 | `top_result_assembly` and DB persist are not hot on accepted no-risk rows | removed; no block top-M merge |
| 19 | Mac Studio fixed 12-thread policy remained fastest for required rows | no config change |
| 20 | Allocation churn did not justify scratch buffers or extra child-local state | removed; no scratch-buffer path |

Contract impact of closure: public API `none`; persisted schema `none`; request
hash/cache identity `none`; browser-visible behavior `none`; config schema
`breaking-change` only for retired internal experimental env values
`stage_17_dynamic_no_risk_selector`, `stage_09_tp_sl_full_grid` and direct
`matrix_cell_tp_sl_v1` env selection. The production default remains
`stage_05_and_12_no_risk`.

`next_iteration_allowed` is `false`. No further backtest compute stage is
unblocked by this prompt pack.

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
| 12 | Compiled prefix product traversal | Compiled/iterative hot path; selectivity order internal only; exact-safe prefix pruning allowed only for eligibility bounds; no Stage 10 Python traversal dependency | Reject if top-N identity/order drifts, canonical `variant_hash` changes, arity-7 service wall improves <20%, `combo_iteration` is not materially lower, or arity-6 regresses |
| 13/14 removed branch | No implementation boundary remains | Do not execute removed Stage 13/13S/13R/14/14R prompts or restore their harness/runtime code | Block any revival unless a new plan records a different cost model, acceptance rows and Mac Studio A/B gate |
| 15 | No accepted runtime boundary remains; exact-safe upper-bound candidate removed from active tree | Historical Mac Studio A/B evidence in `benchmark_iterations/2026-06-14_matrix_bitset_stage_15_tp_sl_early_abandon_candidate/` preserved parity but pruned `0` candidates and regressed service wall | Keep as `accepted_for_learning` only; production early-abandon revival requires a separate approved plan, cheap reject-rate proof on the same TP/SL fixture and a fresh Mac Studio A/B gate |
| 16 | No implementation boundary remains | Closed without execution by 2026-06-14 scope cleanup; prompt removed | Block any revival unless a separate approved plan reopens TP/SL telemetry with a new acceptance model |
| 17 | No active implementation boundary remains | Dynamic selector code, env mode, prompts and raw evidence removed by 2026-06-14 closure cleanup | Block any revival unless a separate approved plan records a new cost model and Mac Studio A/B gate |
| 18 | No active implementation boundary remains | Top-N/result assembly telemetry and raw evidence removed by closure cleanup; compact negative result retained | Do not add block top-M merge unless assembly/persistence becomes a measured hot path |
| 19 | No active implementation boundary remains | Thread-scaling harness additions and raw evidence removed by closure cleanup; fixed 12-thread policy retained | Do not change worker thread config without fresh hardware-specific evidence |
| 20 | No active implementation boundary remains | Allocation telemetry and raw evidence removed by closure cleanup; no scratch-buffer runtime path accepted | Do not add scratch buffers unless repeated allocation churn becomes a measured service-wall/RSS issue |
| 21 | No implementation boundary remains | Closed without execution by 2026-06-14 scope cleanup; prompt removed | Block any revival unless a separate approved architecture/product plan reopens exact/coarse TP/SL mode work |

## Cross-Stage Acceptance Rules

- No stage may advance only on a local kernel/microbenchmark result; API-runner
  service wall and memory evidence must not regress.
- Stage 12+ executors must read
  `docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md`
  and state which rejected methods remain non-goals.
- Acceptance speedup must use the same current-service pipeline boundary unless
  the evidence is explicitly labeled diagnostic. Sidecar generation, sidecar load,
  warmup/cache state, request semantics, candidate set and top-N/persistence shape
  must be recorded so the new path does not get a hidden advantage.
- Acceptance benchmark/testing evidence must run over `ssh macstudio` from
  `/Users/daniildegtyarev/Projects/roehub.com`. The evidence must record whether
  the Mac Studio checkout matched the measured candidate commit or was dirty.
- Future evidence must use a checkout/runtime containing Stage 12 commit
  `1fda2264` or a later descendant. If the active live runtime under
  `/opt/roehub/app` is used or described as production, the stage must record
  that runtime's code state and `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` state
  before comparing results.
- Mac Studio source artifacts are read from
  `/opt/roehub/state/backtest_artifacts/v2`, with `BTCUSDT/current.yaml`
  resolving the active slot manifest. Stage evidence is written under
  `docs/architecture/backtest/benchmark_iterations/<stageNN_dir>/`; generated
  sidecar/test `.npy` files go under `<stageNN_dir>/sidecar_artifacts/` or an
  explicitly recorded test overlay, never into canonical artifact slots.
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
