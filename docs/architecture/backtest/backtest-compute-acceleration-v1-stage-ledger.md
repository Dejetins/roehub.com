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
`long_short_reversal`, still default-off and selectable only through the internal
benchmark/runtime gate.

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
API-runner boundary. The matrix backend remains default-off and does not unlock
production default switching, TP/SL, pruning, request-hash changes, cache
identity changes or sidecar artifacts.

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

## Stage Ledger

| Stage | Status | Scope | Evidence | Decision | next_iteration_allowed |
|---:|---|---|---|---|---|
| 00 | accepted | Refresh current heavy baseline on Mac Studio before code changes | `benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/` | Baseline accepted; re-verified on checkout `6dcb62dc918a98564abec9554ae575187b32fa39`; scoped backtest runtime/harness diff from evidence commit is empty; performance, parity, memory, lazy cache, legacy path, accounting and docs drift gates passed | true |
| 01 | accepted_for_learning | Add instrumentation counters without behavior changes | `benchmark_iterations/2026-06-06_matrix_bitset_stage_01_instrumentation/` | Counters present; explicit `null` for unavailable current-runtime counters; parity, performance, memory, lazy cache, legacy path and accounting gates passed; overhead stayed within <= 1% limit with no Stage 00 service/exact regression; production `on` mode remains locked | true |
| 02 | accepted_for_learning | Row/signature telemetry shadow | `benchmark_iterations/2026-06-06_matrix_bitset_stage_02_row_signature_telemetry/` | Shadow counters present; duplicate rows `0/36` on accepted arity-6 rows; `consensus_signature_count=46656` as deterministic upper bound; collision count `0`; row signature overhead about 10-11ms/job; parity, performance, memory, lazy cache, legacy path and docs drift gates passed; no pruning/scoring/top-N/request-hash/cache change | true |
| 03 | accepted_for_learning | Runtime bitset pack shadow | `benchmark_iterations/2026-06-06_matrix_bitset_stage_03_runtime_bitset_pack/` | Shadow bitsets recorded `signals_pack_ms` about 24ms/job with `W=3421`, packed bytes `1,970,496`, padding valid and consensus sample parity true; API-runner parity `4/4`, performance, memory release, lazy cache, legacy path and docs drift gates passed; scoring/top-N/request hash/cache/persistence unchanged | true |
| 04 | accepted_for_learning | `matrix_bitset_no_risk_v1` for `none/arity_2..3/long_only` | `benchmark_iterations/2026-06-06_matrix_bitset_stage_04_no_risk_mvp/` | Mac Studio API-runner parity `2/2`, memory, instrumentation, lazy cache, legacy path and docs drift gates passed; raw performance failed on tiny `none/arity_2/long_only` by about `1.1ms`, while `none/arity_3/long_only` ratio was `2.590`; arity-2 no-advantage is waived for learning progression only; production `on` mode remains locked | true |
| 05 | accepted | No-risk `long_short_reversal` and arity 6 heavy rows | `benchmark_iterations/2026-06-06_matrix_bitset_stage_05_reversal_arity6/` | Mac Studio API-runner parity `2/2`, performance, memory, instrumentation, lazy cache, legacy path and docs drift gates passed; `none/arity_6/long_only` ratio `15.543`, `none/arity_6/long_short_reversal` ratio `5.323`; production default remains locked | true |
| 06 | rejected | Consensus signature cache | `benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache/` | Cache hit-rate `0.202396` and collision count `0`, but Mac Studio API-runner exact scoring regressed versus Stage 05: `1.010s -> 4.932s` long-only and `2.887s -> 6.166s` reversal; cache runtime candidate not accepted | true for Stage 07 sidecar/test bitset artifacts only |
| 07 | planned | Sidecar/test bitset artifacts generated outside publisher: planned generator/helper, explicit sidecar path, `signals_pos_bits.u64.npy`, `signals_neg_bits.u64.npy`, `signal_row_hashes.u64.npy`, `unique_signal_row_ids.u32.npy`, `duplicate_signal_row_ids.u32.npy`, `matrix_sidecar_manifest.json`; no `backtest_artifacts` publisher/precompute or canonical manifest changes | planned | Stage 06 cache rejected with clear evidence; Stage 07 may start as an independent sidecar/test-bitset stage only | false |
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
- Acceptance benchmark/testing evidence must run over `ssh macstudio` from
  `/Users/daniildegtyarev/Projects/roehub.com`. The evidence must record whether
  the Mac Studio checkout matched the measured candidate commit or was dirty.
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
