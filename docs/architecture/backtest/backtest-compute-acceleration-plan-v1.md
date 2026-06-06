# Backtest Compute Acceleration Plan v1

Документ фиксирует staged-план ускорения compute-ядра backtest service по линии
`row/signature dedup -> bitset artifacts -> bitset consensus -> sparse trade tape -> TP/SL cell blocks`.

## Статус

План внедрения перед кодовыми изменениями. Никакой новый backend не считается
разрешенным к production path, пока для него не записано сопоставимое benchmark
evidence на Mac Studio и в журнале stages не выставлено `next_iteration_allowed: true`.

## Цель

Снизить wall-clock тяжелых backtest jobs без изменения пользовательского контракта:

- `POST /backtests/jobs`, persisted top-N, readable `variant_key` и stable
  `variant_hash` остаются прежними;
- scoring semantics для `risk.mode = none` и `risk.mode = tp_sl_grid` остаются exact;
- новый compute backend включается только staged: `off -> shadow -> on`;
- каждая оптимизация двигается дальше только после доказанного ускорения на той же
  нагрузке, hardware class, artifact set, request semantics и cache/warmup policy.

## Контекст

Текущий trusted runtime уже artifact-backed: child process читает `.npy` artifacts,
делает prepare pools, ordinal combo streaming, proxy filter, exact scoring и top-N.
Рекомендации из приложенного исследования сходятся с текущими benchmark docs:
главное узкое место находится в compute child, особенно в `exact_scoring` и
`tp_sl_exact_scoring`.

Последний чистый зафиксированный heavy benchmark текущей API-runner архитектуры:

| Evidence | Workload | Важные результаты |
|---|---|---|
| `2026-05-14_iteration_15_api_runner_clean_arity6_cpu_memory` | `BTCUSDT` / `15m` / arity 6 / 12 Numba threads / `top_n=50` | performance `pass`, parity `pass`, memory overall `fail` for no-risk jobs |
| `2026-05-15_iteration_16_quality_gate_ranking_exact_arity6_cpu_memory` | experimental quality gate / ranking-only exact | rejected: partial exact speedup did not produce accepted end-to-end improvement and changed result shape |

Iteration 16 is a hard guardrail for this plan: локальный выигрыш внутри одного
timer недостаточен, если full service boundary, top-N shape, parity или memory gate
регрессируют.

## Что не входит

- GPU-first rewrite.
- Approximate search как default exact backend.
- Изменение public API, DB schema, persisted job identity или public route keys.
- Переписывание UI `/backtests`.
- Смена hit-times semantics или TP/SL tie-breaking. Текущий rule: when TP and SL
  hit on the same bar, SL wins.
- Ускорение worker queue throughput как primary objective. Lazy lane и worker
  priority можно планировать отдельно после compute-core stages.
- Изменение `backtest_artifacts` publisher/precompute, `manifest.yaml`,
  `current.yaml` или active-slot publish pipeline на первых compute stages.
  Новые `.npy` сначала генерируются отдельно как benchmark/test sidecar.
- Full pair cache как default production artifact или default runtime path.
  Pair cache допускается только как bounded/shadow optimization после отдельного
  memory benchmark.
- RMQ/sparse table для drawdown. Текущий no-risk drawdown считается по closed
  equity на trade close; если контракт изменится на mark-to-market drawdown,
  это будет отдельный stage с новой correctness моделью.

## Целевая архитектура

Снаружи orchestration остается прежним:

```text
prepare_pools
  -> combo planning
  -> backend scoring
  -> top result assembly
  -> persist top-N
```

Внутри scoring добавляется optional backend family:

```text
application/services/v2/matrix_backend/
  bitsets.py
  row_signatures.py
  combo_blocks.py
  consensus.py
  trade_tape.py
  no_risk_score.py
  tp_sl_cells.py
  topn.py
  validation.py
```

Planned backend ids:

| Backend | Risk mode | Initial scope | Role |
|---|---|---|---|
| `matrix_bitset_no_risk_v1` | `none` | arity 2/3, then arity 1..10 | blockwise bitset consensus plus sparse no-risk scoring |
| `matrix_cell_tp_sl_v1` | `tp_sl_grid` | selected cell blocks, then full grid | sparse trade tape plus TP/SL cell-block scoring |

Backend selector is additive:

```yaml
backtest_compute:
  matrix_backend:
    mode: off  # off | shadow | on
    candidate_block_size: 4096
    tp_block_size: 16
    sl_block_size: 16
    dedup_signatures: true
    hit_times_layout: by_entry
    sidecar_artifact_dir: null  # benchmark/test only; canonical publisher is unchanged
    max_pair_cache_rows: 0  # disabled by default; research/shadow only
```

`shadow` mode computes bounded samples and parity/hash evidence but does not feed
production top-N. `on` is allowed only after stage acceptance.

## Артефакты Из Рекомендации

Рекомендованные `.npy` additions делятся на три группы. Они не меняют public API
и не должны попадать в public request hash, пока semantics результатов не меняются.

### Signal Bitset And Dedup Artifacts

Stage 07 должен проверить эти файлы как sidecar/test artifacts, сгенерированные
отдельно из текущих canonical artifacts. `backtest_artifacts` publisher,
`manifest.yaml`, `current.yaml` и active slots на этом этапе не меняются.

| Artifact | Purpose | Stage gate |
|---|---|---|
| `signals_pos_bits.u64.npy` | Bitset where signal row is `+1` | Source hash validation plus runtime pack-cost reduction |
| `signals_neg_bits.u64.npy` | Bitset where signal row is `-1` | Same as positive bits |
| `signal_row_hashes.u64.npy` | Stable row signatures for dedup/cache keys | No top-N identity collapse without expansion back to original rows |
| `unique_signal_row_ids.u32.npy` | Unique signal row ids after dedup | Deterministic ordering and sidecar metadata validation |
| `duplicate_signal_row_ids.u32.npy` | Duplicate-to-unique mapping | Public variant identity expansion stays exact |

Each timeframe has its own sidecar bitset artifacts. A sidecar metadata file must
record source canonical `manifest.yaml` hash, source `signals.i8.npy` hash, `T`,
`W = ceil(T / 64)`, padding policy and timeframe/market/symbol identity. Stage 07
is optional for production: if runtime packing is already cheap enough, sidecar
artifacts can stay benchmark-only and never move into publisher.

### Sidecar Artifact Strategy

New `.npy` files are generated outside the canonical artifact publisher for now.
This keeps Stage 07 focused on benchmark evidence and avoids adding manifest
schema risk before the compute speedup is proven.

| Area | Stage 07 rule |
|---|---|
| Canonical `backtest_artifacts` publisher/precompute | No code changes |
| Canonical `manifest.yaml` / `current.yaml` | No changes and no in-place rewrite |
| Planned sidecar generator | `scripts/backtest/generate_matrix_sidecar_artifacts.py` or equivalent benchmark helper |
| Planned sidecar metadata | `matrix_sidecar_manifest.json` with source hashes, artifact schema version, shapes, dtypes and padding |
| Runtime loading | Explicit benchmark/test `sidecar_artifact_dir`; fallback to runtime packing when sidecar is absent |
| Evidence location | Under the stage benchmark evidence directory or another explicitly recorded test overlay path |

Stage 07 acceptance requires sidecar generator tests, source-hash validation,
shape/dtype validation, deterministic duplicate mapping checks, runtime fallback
when sidecar files are absent, and benchmark evidence that sidecar loading is
faster than runtime packing on comparable rows. Publisher tests are not required
because publisher behavior is deliberately unchanged.

### TP/SL Hit-Times By-Entry Artifacts

Stage 08-09 must validate hit-times memory layout before enabling
`matrix_cell_tp_sl_v1`:

| Artifact | Purpose |
|---|---|
| `long_tp_by_entry.u32.npy` | Long-side TP hit index by `entry_idx, tp_idx` |
| `long_sl_by_entry.u32.npy` | Long-side SL hit index by `entry_idx, sl_idx` |
| `short_tp_by_entry.u32.npy` | Short-side TP hit index by `entry_idx, tp_idx` |
| `short_sl_by_entry.u32.npy` | Short-side SL hit index by `entry_idx, sl_idx` |

If full by-entry artifacts are too large or IO-heavy, the accepted alternative is
job-local selected arrays, for example `selected_long_tp[entry_idx, selected_tp_idx]`
and `selected_long_sl[entry_idx, selected_sl_idx]`. That alternative must still
prove speedup on the same Stage 00 workload and must preserve the SL-wins tie rule.
If by-entry hit-times are persisted for testing, they follow the same sidecar
strategy: no publisher changes, explicit sidecar path and source-hash metadata.
Promotion into canonical publisher artifacts is deferred to a separate plan after
compute speedup is proven.

### Pair Cache Policy

Recommendation allowed an optional arity-2 pair cache, but full pair cache is not
part of the default plan because memory can dominate the child process. It can be
tested only under these constraints:

- `max_pair_cache_rows > 0` is opt-in and disabled by default;
- cache is child-local or bounded mmap/LRU, never an unbounded publisher artifact;
- build only after dedup/prefilter or for `R <= threshold`;
- benchmark must record memory peak, cleanup, cache hit-rate and service wall;
- failure to prove end-to-end speedup records the experiment as `rejected`, not as
  a partial success.

## Recommendation Coverage Audit

| Recommendation | Coverage in this plan |
|---|---|
| Blockwise matrix / bitset / sparse-event backend | Covered by Stages 03-05 and 08-09 |
| Deduplicate signal rows before scoring | Stage 02 telemetry found no duplicate signal rows on accepted arity-6 rows; Stage 06 runtime cache was tested and rejected; Stage 07 sidecar artifacts may still record duplicate maps for validation/identity expansion |
| Exact signal bitset `.npy` artifacts | Covered explicitly in Stage 07 sidecar artifact list |
| Consensus signature cache | Covered by Stage 06 and rejected by Mac Studio evidence; do not make later stages depend on the rejected runtime cache candidate |
| Sparse trade tape for selected candidates | Covered by Stages 04-05 and Stage 11 reuse |
| TP/SL by-entry hit-times artifacts | Covered explicitly in Stages 08-09 |
| TP/SL cell-block scoring | Covered by Stages 08-09 |
| Optional pair cache | Research/shadow only; disabled by default |
| Exact-safe high-arity pruning | Covered by Stage 10 |
| MVP-1 no-risk arity 2/3 long-only | Covered by Stage 04 |
| MVP-2 no-risk long-short reversal | Covered by Stage 05 |
| MVP-3 TP/SL selected cells before full grid | Covered by Stage 08 before Stage 09 |
| Signal semantics tests | Covered by the correctness matrix below |
| Trade boundary tests | Covered by the correctness matrix below |
| Fees/slippage and sizing parity | Covered by the correctness matrix below |
| Float determinism and stable top-N tie-break | Covered by the correctness matrix below |
| Minimum block-scoring backend API | Covered by planned `matrix_backend` modules and Stage 04/09 gates |
| Beam search / approximate search | Not default; requires explicit product approval |
| Dense `all_combos x all_bars x all_tp x all_sl` tensor | Explicitly rejected |
| RMQ/sparse table for drawdown | Not applicable to current closed-equity drawdown contract |
| GPU-first rewrite | Deferred; CPU-first only |
| Worker/lazy priority | Separate perceived-latency concern; only Stage 11 reuses compute artifacts |
| Publisher-level bitset artifacts | Intentionally deferred; Stage 07 uses sidecar artifacts first per current rollout constraint |

## Benchmark Model

### Acceptance workload

Stage 00 must refresh the current heavy baseline before any production-affecting
backend change:

```bash
uv run python scripts/backtest/run_api_runner_benchmark_parity.py \
  --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
  --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_00_current_baseline

uv run python scripts/backtest/validate_benchmark_accounting.py \
  --out docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_00_current_baseline/local_accounting_validation.json
```

Runtime environment source for API-runner benchmarks:

| Runtime | Env source | Notes |
|---|---|---|
| Mac Studio native launchd | `/Users/daniildegtyarev/.config/roehub/roehub.env` | This file is outside the repository and contains the real values used by `infra/macos/launchd/com.roehub.api.plist` and `infra/macos/launchd/com.roehub.backtest-job-runner.plist`. |
| Docker/backend compose | `/etc/roehub/roehub.env` via `ROEHUB_ENV_FILE`; template `infra/docker/.env.example` | Compose derives `IDENTITY_PG_DSN`, `POSTGRES_DSN` and `STRATEGY_PG_DSN` from `POSTGRES_DB`, `POSTGRES_USER` and `POSTGRES_PASSWORD`. |

`scripts/backtest/run_api_runner_benchmark_parity.py` accepts `--env-file` and
falls back to `$ROEHUB_ENV_FILE`,
`/Users/daniildegtyarev/.config/roehub/roehub.env`, then
`/etc/roehub/roehub.env`. For Mac Studio native benchmark evidence, the harness
must also run with `ROEHUB_ENV=prod` and
`ROEHUB_BACKTEST_ARTIFACTS_CONFIG=configs/prod/backtest_artifacts.yaml`; when
the env file omits those keys, the benchmark harness fills them and records
only the key names and path. Evidence may record which keys are present, but
must not print DSN or password values.

Acceptance benchmark/testing evidence must run on Mac Studio over SSH:

```bash
ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && <benchmark-or-test-command>'
```

Local runs are preflight diagnostics only unless a stage explicitly marks a
check as local-only. Before SSH testing, the Mac Studio checkout must contain
the exact candidate code being measured; the evidence must record commit SHA or
dirty state. Do not benchmark a different runtime copy without recording it.

Canonical source artifacts for these benchmarks are read-only:

| Purpose | Path |
|---|---|
| Mac Studio source artifact root | `/opt/roehub/state/backtest_artifacts/v2` |
| BTCUSDT active pointer | `/opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml` |
| Active slot manifest | resolved from `BTCUSDT/current.yaml` |

Stage evidence must be saved under
`docs/architecture/backtest/benchmark_iterations/<stageNN_dir>/`. Generated
sidecar/test `.npy` files must be saved under
`docs/architecture/backtest/benchmark_iterations/<stageNN_dir>/sidecar_artifacts/`
or another explicitly recorded test overlay. They must not be written into the
canonical artifact root, `current.yaml`, active slots or publisher outputs.

Current Stage 00 evidence is recorded at:

`docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/`

Required heavy rows:

| Job | Required |
|---|---:|
| `none/arity_6/long_only` | yes |
| `none/arity_6/long_short_reversal` | yes |
| `tp_sl_grid/arity_6/long_only` | yes |
| `tp_sl_grid/arity_6/long_short_reversal` | yes |

Additional MVP rows are required when the stage targets no-risk matrix backend:

| Job | Purpose |
|---|---|
| `none/arity_2/long_only` | compare with specialized current backend |
| `none/arity_3/long_only` | first generic bitset consensus target |
| `none/arity_3/long_short_reversal` | reversal transition parity |

### Metrics

Every stage record must include:

- `exact_scoring` and, for risk-on, `tp_sl_exact_scoring`;
- `service_wall_clock_s` / `service_total_without_warmup`;
- `prepare_pools_core`, `build_exact_context`, `combo_iteration`,
  `proxy_filter`, `heap_update`, `artifact_load_ms`;
- `signals_pack_ms`, `unique_rows_after_dedup`, `consensus_signature_count`,
  `rows_before_prefilter`, `rows_after_prefilter`, `combo_count_planned`,
  `candidates_after_proxy`, `exact_candidates`, `avg_segments_per_candidate`,
  `avg_trades_per_candidate`, `tp_count`, `sl_count`, `tp_sl_cells`,
  `exact_candidates_per_sec`, `trade_cell_evals_per_sec` when applicable;
- child CPU samples, Numba threads, warmup policy, artifact manifest hashes and
  request hashes.

### Gate Rule

An optimization stage passes only when all are true:

- correctness parity passes: top-N shape, result hash or bounded metric diff,
  self-check and route/API shape when applicable;
- target hot stage is faster than its Stage 00 current baseline by at least 10%
  on median measured comparable rows, or by at least 5% with repeated-run variance
  evidence showing the delta is outside timing noise;
- `service_total_without_warmup` and API-runner wall do not regress;
- memory cleanup does not regress relative to Stage 00;
- no public API, persistence identity, cache identity, TP/SL tie-breaking or
  ranking semantics changed silently;
- the ledger row records `next_iteration_allowed: true`.

If a stage is only instrumentation or shadow validation, it may be marked
`accepted_for_learning`, but it must not enable a production backend and must not
unlock a later `on` stage unless the later stage proves speedup.

### No-Advantage Benchmark Policy

Experimental code may live in a separate shadow/test branch of the implementation,
but accepted speedup must not get a benchmark advantage over the current service.

| Area | Fairness rule |
|---|---|
| Pipeline boundary | Acceptance evidence must run through the same service boundary as the current path: artifact context, array loading, request slicing, `prepare_pools`, combo planning/proxy, backend scoring, top-N assembly and persistence shape. Module-level measurements are diagnostic only. |
| Request and fixture | Same symbol, timeframe, time range, indicators, risk mode, execution settings, ranking, `top_n`, request hash policy and artifact manifest compatibility policy. |
| Candidate set | New backend must consume the same candidate stream after the current proxy unless the stage is explicitly about exact-safe pruning; pruning stages must prove no valid top candidate is removed. |
| Warmup/cache | Same warmup policy, Numba thread settings, cold/warm cache labeling and repeated-run policy. A warm sidecar load cannot be compared to a cold current artifact load as acceptance evidence. |
| Sidecar `.npy` | `sidecar_generate_ms` and `sidecar_load_ms` must be recorded separately. `sidecar_load_ms` is included in service wall when sidecar is used. `sidecar_generate_ms` may be excluded only when the result is labeled precomputed/sidecar diagnostic, not production acceptance. |
| Production claim | Because publisher changes are out of scope for the current plan, a speedup that depends on pre-generated sidecar files can be `accepted_for_learning`, but cannot by itself enable production `on` mode. Production acceptance requires either runtime generation included in the measured path or a later approved publisher/precompute plan. |
| Result contract | Public top-N shape, `variant_key`, `variant_hash`, ranking, fees/slippage, sizing and TP/SL tie-breaking must match the current service. |

If any fairness rule is not met, the result can be recorded as directional
evidence only and cannot set `next_iteration_allowed: true` for a production path.

### Git / Main Branch Delivery

Все stage prompts выполняются из checkout ветки `main`. Если активная ветка не
`main`, executor должен остановиться и явно сообщить blocker, если пользователь
заранее не разрешил другой branch для конкретного stage.

После успешного `accepted` stage executor должен:

- обновить stage ledger, evidence и связанные docs до финального состояния;
- выполнить required quality gates и benchmark/evidence gates;
- добавить в Git только scoped stage files, без unrelated worktree changes;
- создать commit на ветке `main` с stage-specific message;
- записать commit SHA, scoped paths и `push/deploy: not performed` в финальный
  отчет и stage ledger.

`accepted_for_learning` shadow/telemetry stages тоже должны быть сохранены в
`main`, если их код, docs или evidence являются durable handoff для следующих
stages. При этом ledger обязан сохранить статус `accepted_for_learning` и явно
указать, что production `on` mode не разблокирован.

`blocked` и `rejected` stages не должны коммитить production runtime changes.
Допускается сохранить в `main` только ledger/evidence/docs, фиксирующие blocker
или rejection, если это нужно для durable history. Push/deploy не является частью
этих stage prompts и выполняется только по отдельному user request или delivery
prompt.

### Correctness Acceptance Matrix

Every stage that changes scoring, candidate pruning, sidecar loading or top-N
merge must prove the relevant rows below before it can be `accepted`.

| Surface | Required checks | Applies to stages |
|---|---|---|
| Signal semantics | `+1`, `0`, `-1`, `long_only`, `long_short_reversal`, neutral handling, opposite signal handling, `close_on_end` | 03-06, 10 |
| Reversal transitions | `long -> short`, `short -> long`, `long -> flat`, `short -> flat`, `flat -> long`, `flat -> short` | 05, 10 |
| Trade boundaries | entry/exit bar, `15m -> 1m` mapping, open/close execution, last bar, empty signal, single-bar trade | 04-05, 08-10 |
| TP/SL tie-breaking | `tp_hit_idx == sl_hit_idx` follows the current conservative rule: SL wins | 08-09 |
| Fees and slippage | entry fee, exit fee, long/short formula and slippage direction match the current backend | 04-05, 08-10 |
| Sizing scope | Stage 04 starts with current accepted all-in/fixed-equity semantics in `none`; broader sizing modes require explicit parity rows before rollout | 04-05, 08-10 |
| Float determinism | deterministic block merge with stable tie-break; no top-N drift from float accumulation noise | 04-06, 09-10 |
| Variant identity | dedup/cache/sidecar expansion preserves public `variant_key`, stable `variant_hash`, ranking order and persisted top-N shape | 02, 04-07, 10 |
| Sidecar safety | source manifest hash, source `signals.i8.npy` hash, shape, dtype, padding and duplicate map validate before use; missing sidecar falls back to runtime packing | 07 |
| Cell blocks | selected cells first, then full grid; `16 x 16`, `32 x 8` or `8 x 32` cell blocks must be recorded with memory and timing evidence | 08-09 |
| High arity pruning | only exact-safe monotonic pruning or branch-and-bound in default path; approximate beam remains off unless separately approved | 10 |

## План Внедрения

| Stage | Scope | Acceptance |
|---:|---|---|
| 00 | Current heavy baseline on Mac Studio | Current timings and memory gates recorded; no code changes; `next_iteration_allowed` only after evidence exists |
| 01 | Instrumentation counters | Adds counters only; benchmark overhead <= 1%; no result hash drift |
| 02 | Row/signature telemetry shadow | Records duplicate row and consensus-signature potential; no pruning; no top-N drift |
| 03 | Runtime bitset pack shadow | Packs `trade_T` into `pos_bits` / `neg_bits`; sample consensus parity; pack overhead measured |
| 04 | `matrix_bitset_no_risk_v1` for `none/arity_2..3/long_only` | Exact parity plus accepted speedup on target rows; service wall no regression |
| 05 | No-risk reversal and arity 6 | `long_short_reversal` transitions and arity 6 parity; accepted heavy-row speedup |
| 06 | Consensus signature cache | Rejected: cache hit-rate was real, but exact scoring and service wall regressed versus Stage 05; no runtime cache code is accepted |
| 07 | Sidecar bitset artifacts | May proceed independently of Stage 06 cache; generate `signals_pos_bits.u64.npy`, `signals_neg_bits.u64.npy`, `signal_row_hashes.u64.npy`, `unique_signal_row_ids.u32.npy`, `duplicate_signal_row_ids.u32.npy` outside publisher; source-hash validation and runtime pack cost removed or reduced |
| 08 | TP/SL selected-cell shadow | `tp_count <= 8`, `sl_count <= 8`; SL tie rule parity; by-entry hit-times layout counters or selected by-entry arrays |
| 09 | `matrix_cell_tp_sl_v1` full grid blocks | Full request grid exact parity; accepted `tp_sl_exact_scoring` speedup and no service wall regression |
| 10 | Exact-safe high-arity pruning | Only monotonic/exact-safe pruning in default path; approximate beam remains explicit non-default mode |
| 11 | Lazy detail reuse of sparse trade tape | Selected variant materialization latency benchmark; separate UX/perceived-latency gate |

## Планируемые Файлы И Артефакты По Stages

| Stage | Planned code/config | Planned evidence |
|---:|---|---|
| 00 | none | `benchmark_iterations/<date>_matrix_bitset_stage_00_current_baseline/` |
| 01 | DTO telemetry fields, benchmark renderer updates | `stage_01_instrumentation/benchmark_results.json` |
| 02 | `row_signatures.py`, telemetry-only hook | duplicate/signature potential report |
| 03 | `bitsets.py`, shadow validation hook | bitset parity sample report |
| 04-05 | `consensus.py`, `trade_tape.py`, `no_risk_score.py` | no-risk benchmark and parity evidence |
| 06 | no accepted runtime files; rejected candidate retained only as evidence patch under `benchmark_iterations/2026-06-06_matrix_bitset_stage_06_signature_cache/` | cache hit-rate/regression evidence |
| 07 | planned sidecar generator `scripts/backtest/generate_matrix_sidecar_artifacts.py` or equivalent benchmark helper; sidecar loader in matrix backend; outputs `signals_pos_bits.u64.npy`, `signals_neg_bits.u64.npy`, `signal_row_hashes.u64.npy`, `unique_signal_row_ids.u32.npy`, `duplicate_signal_row_ids.u32.npy`, `matrix_sidecar_manifest.json`; no `backtest_artifacts` publisher/precompute changes | sidecar source-hash validation, runtime fallback and benchmark evidence |
| 08-09 | `tp_sl_cells.py`, by-entry hit-times layout validation for `long_tp_by_entry.u32.npy`, `long_sl_by_entry.u32.npy`, `short_tp_by_entry.u32.npy`, `short_sl_by_entry.u32.npy` or job-local selected arrays; no publisher/manifest modules unless a later separate publisher plan is approved | selected/full grid TP/SL benchmark evidence |
| 10 | exact-safe pruning planner | arity 7/10 bounded-search evidence |
| 11 | lazy materialization backend adapter | lazy trades benchmark evidence |

## Контракты И Влияние

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | Existing routes and payload meanings stay unchanged |
| Port contract | `compatible-change` | New backend selector/config and telemetry may add optional fields |
| DTO schema | `compatible-change` | Telemetry/counter fields are additive |
| Persisted schema | `none` initially | Bitset artifacts are file artifacts, not DB schema |
| Canonical artifact file/manifest schema | `none` initially | Stage 07 sidecar files do not change canonical `manifest.yaml`, `current.yaml` or active-slot artifacts |
| Test sidecar artifact metadata | `compatible-change` | New benchmark/test-only `matrix_sidecar_manifest.json` records source hashes, shapes, dtypes and schema version |
| Config schema | `compatible-change` | Optional `backtest_compute.matrix_backend` block, explicit sidecar path for benchmark/test runs and disabled-by-default pair-cache controls |
| Request hash / cache identity | `none` | Result-affecting backend must not enter public request hash unless semantics change, which is out of scope |
| Service-call semantics | `none` | Same API/runner/child process topology |
| Logs/metrics/report semantics | `compatible-change` | Additive benchmark and telemetry counters |
| Alert/runbook semantics | `compatible-change` | Only if new production backend metrics become alertable |
| Benchmark gate | `compatible-change` | Adds `next_iteration_allowed` and current-baseline gating |
| Browser-visible behavior | `none` | UI may get results faster, but workflow and defaults stay unchanged |

## Операционные Аспекты

- Benchmark acceptance is Mac Studio only, matching existing backtest benchmark
  policy.
- Child process remains the isolation boundary for heavy compute and memory cleanup.
- Runtime backend mode must be visible in telemetry and benchmark evidence.
- Sidecar artifacts are generated from the canonical active artifacts and are
  addressed by explicit benchmark/test path. Canonical publisher output stays
  unchanged, old manifests stay readable and runtime packing remains the fallback.
- Logs may include backend ids, timings, row counts and hashes; they must not log
  secrets or full unbounded payloads.
- Rollback is config-based: set matrix backend mode to `off` and fall back to
  existing `event_segments_*` backends; sidecar files can be deleted without
  touching canonical artifact slots.

## Журнал Выполнения

Durable stage ledger:

`docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Each row must record:

- stage status;
- code/config/docs touched;
- benchmark command and evidence directory;
- baseline/current/new timing;
- correctness result;
- memory result;
- contract impact;
- Git branch, scoped staged/committed paths and commit SHA when the stage is
  `accepted` or `accepted_for_learning`;
- push/deploy handoff status, normally `not performed` for this plan;
- `next_iteration_allowed`.

## Открытые Риски

- Local dense matrix design is explicitly rejected: `all_combos x all_bars x all_tp x all_sl`
  is not a viable production shape.
- Dedup can accidentally change public top-N if variant identity expansion and
  deterministic tie-breaking are not preserved.
- TP/SL full grid still has inherent `candidate x trade x cell` cost; cell-blocks
  reduce constants but do not remove the need for pruning/finalist strategy later.
- Arity 7/10 exhaustive search may remain impractical without exact-safe pruning;
  approximate beam search requires explicit product approval and separate contract.
- Full pair cache can regress memory and wall-clock even when isolated pair lookup
  gets faster; it is excluded from default production path until bounded evidence
  proves otherwise.
- RMQ/sparse table is intentionally not planned for current closed-equity drawdown;
  adding mark-to-market drawdown acceleration would be a contract change.
