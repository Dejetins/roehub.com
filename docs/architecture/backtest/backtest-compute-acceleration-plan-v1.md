# Backtest Compute Acceleration Plan v1

Документ фиксирует staged-план ускорения compute-ядра backtest service по линии
`row/signature dedup -> bitset artifacts -> bitset consensus -> sparse trade tape -> TP/SL cell blocks -> fused compiled traversal/scoring`.

## Статус

План внедрения и последующих continuation stages. Никакой новый backend не
считается разрешенным к production path, пока для него не записано сопоставимое
benchmark evidence на Mac Studio и в журнале stages не выставлено
`next_iteration_allowed: true`.

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

Continuation update от 2026-06-13 добавляет Stage 12+ после negative review.
Текущие accepted baselines:

| Baseline | Scope | Evidence |
|---|---|---|
| Stage 05 default-on | `risk.mode=none`, arity `6`, `long_only` / `long_short_reversal`; production composite default keeps this path for arity `6`; rollback/comparison через `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=off` | `2026-06-10_matrix_bitset_stage_05_default_off_baseline/`, `2026-06-10_matrix_bitset_stage_05_default_on_candidate/` |
| Stage 09 opt-in | `risk.mode=tp_sl_grid`, arity `6`, full-grid cell blocks, accepted shape `64 x 64`; still internal/opt-in | `2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64_rerun/` |
| Stage 10 learning only | exact-safe min-trade rule is valid, but Python traversal is not accepted acceleration | `2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7_partial/` |
| Stage 12 production/default for arity 7, opt-in for arity 6 | `risk.mode=none`, arity `7`, `long_only` / `long_short_reversal` through production composite default; explicit `ROEHUB_BACKTEST_MATRIX_BACKEND_MODE=stage_12_compiled_prefix_traversal` still runs compiled prefix for arity `6` and `7` benchmark/comparison | `2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_baseline_off/`, `2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_candidate_rerun2/`, `2026-06-13_matrix_bitset_stage_05_12_production_default_live/` |

Новые stages должны сравниваться с ближайшим accepted baseline, а не с rejected
candidate. Stage 12+ нельзя делать зависимыми от Stage 06 cache, Stage 07
sidecar load, Stage 10 Python traversal или Stage 11 lazy reuse.

После принятия Stage 12 baseline делится на два уровня:

- accepted-code baseline: checkout содержит commit `1fda2264` или более новый и
  использует production composite default `stage_05_and_12_no_risk`: Stage 05 для
  no-risk arity `6`, Stage 12 для no-risk arity `7`;
- live-production baseline: runtime, который реально обслуживает API/runner,
  содержит тот же accepted code и его env/default state явно записан в evidence.
  Текущий accepted live-production evidence: commit `1bd7a1e4`, runtime
  `/opt/roehub/app`, env override unset, command flag
  `--stage-05-12-production-default-rows`, evidence
  `2026-06-13_matrix_bitset_stage_05_12_production_default_live/`.

Нельзя называть benchmark "updated production baseline", если измерялась только
копия репозитория, а фактический live runtime не был обновлен и smoke-verified.

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
- Runtime consensus signature cache из Stage 06, Python high-arity traversal из
  Stage 10, lazy detail sparse reuse из Stage 11 и sidecar `.npy` load как
  production speedup. Эти методы можно вернуть только как новый benchmark-gated
  stage с другим dominant cost center.
- Approximate TP/SL coarse grid как замена exact semantics. Это может быть
  только отдельный product/UX режим после явного approval и маркировки как
  approximate.
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
  trade_tape.py
  no_risk_score.py
  prefix_traversal.py
  tp_sl_cells.py
```

Planned backend ids:

| Backend | Risk mode | Initial scope | Role |
|---|---|---|---|
| `matrix_bitset_no_risk_v1` | `none` | accepted default only for arity 6 long-only/reversal | blockwise bitset consensus plus sparse no-risk scoring |
| `matrix_cell_tp_sl_v1` | `tp_sl_grid` | accepted opt-in full grid, shape `64 x 64` | sparse trade tape plus TP/SL cell-block scoring |
| `compiled_prefix_product_traversal_v1` | `none` | accepted production composite default for arity 7 and explicit opt-in for arity 6/7 product-form pools | fused compiled prefix traversal, selectivity ordering, exact-safe prefix pruning and scoring handoff |
| `tp_sl_monotonic_cell_kernel_v1` | `tp_sl_grid` | arity 6 full grid | monotonic TP/SL cell classification within accepted cell-block backend |
| `dynamic_backtest_backend_selector_v1` | `none`, `tp_sl_grid` | exact accepted modes only | choose current vs matrix/cell backend by estimated work and measured overhead, not arity alone |

Backend selector is additive:

```yaml
backtest_compute:
  matrix_backend:
    mode: stage_05_and_12_no_risk  # off | stage_05_and_12_no_risk | stage_05_no_risk_reversal_arity6 | stage_12_compiled_prefix_traversal | stage_09_tp_sl_full_grid | shadow/on aliases where supported
    candidate_block_size: 4096
    tp_block_size: 16
    sl_block_size: 16
    dedup_signatures: true
    hit_times_layout: by_entry
    sidecar_artifact_dir: null  # benchmark/test only; canonical publisher is unchanged
    max_pair_cache_rows: 0  # disabled by default; research/shadow only
    dynamic_selector:
      enabled: false
      min_estimated_bit_ops: null
      thread_policy: fixed
    tp_sl:
      monotonic_cell_kernel: false
      early_abandon_total_return: false
      approximate_coarse_grid: false
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
| Stop-list после failed stages | Covered by `backtest-compute-acceleration-negative-results-v1.md`; Stage 12+ prompts must read it and list the rejected methods as non-goals |
| Compiled prefix product traversal | Stage 12; must be compiled/iterative hot path, not Python recursion/traversal |
| Exact-safe prefix pruning inside traversal | Stage 12; may use eligibility upper bounds such as active bars, possible closed trades and exposure, not score/ranking upper bounds unless separately proven |
| Selectivity-based indicator dimension order | Stage 12; compute order may change, but public variant order, `variant_hash`, result assembly and tie-break must remain canonical |
| TP/SL full-grid `64 x 64` production candidate and block autotune | Stage 13 rejected and removed from active tree; summary retained only in the negative-results stop-list |
| TP/SL selective production selector and reversal repair | Stage 13S/13S2/13R/14R rejected or learning-only, then removed from active tree; do not continue this branch |
| TP/SL monotonic cell kernel | Original Stage 14 superseded by failed Stage 13/14 branch; no executable prompt remains |
| TP/SL total-return early abandon | Stage 15 is unblocked after Stage 13/14 cleanup; it must use the current exact TP/SL baseline on a Stage 05+12 production-default checkout and must not restore removed Stage 13/14 code |
| TP/SL reusable trade-window telemetry | Stage 16 remains blocked; no grouped/cache work without a fresh compiled grouping plan |
| Dynamic backend selector by estimated work | Stage 17 may be reopened only for accepted no-risk backends, or after a new accepted TP/SL production path exists |
| Top-N/result assembly batch reduction | Stage 18; first measures assembly timers, then optional stable block top-M merge if assembly is hot |
| Numba thread scaling by workload | Stage 19; benchmark matrix first, config update only after service-wall evidence |
| Allocation reuse / per-child scratch buffers | Stage 20; starts with allocation telemetry, then per-child scratch only; no global cross-job cache |
| Exact/coarse TP/SL product modes | Stage 21; architecture/product admission policy only unless exact semantics remain unchanged or product approves approximate mode |

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

For stages after Stage 12 productionization, the minimum accepted-code baseline
is the composite default mode `stage_05_and_12_no_risk`: Stage 05 for no-risk
arity `6`, Stage 12 for no-risk arity `7`. No-risk stages that overlap prefix
traversal or no-risk scoring must compare against this production default plus
the explicit rollback/comparison modes. TP/SL stages keep Stage 09/current exact
as their TP/SL comparison paths, but the checkout/runtime for both control and
candidate must still include Stage 12 code or explicitly record why it does not.
If the benchmark claims "production" rather than "repo checkout" evidence, it
must also verify the active runtime path, env file path and
`ROEHUB_BACKTEST_MATRIX_BACKEND_MODE` state. For no-risk production-default
rows, use `--stage-05-12-production-default-rows`. Acceptance evidence is valid
only when each heavy job is claimed by the benchmark harness process; if the
live `com.roehub.backtest-job-runner` launchd service claims a benchmark job,
record that run as diagnostic and rerun with isolation or explicit claim
verification.

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

Additional continuation rows are required for Stage 12+ when the stage targets
the corresponding cost center:

| Job | Required when | Purpose |
|---|---|---|
| `none/arity_7/long_only` | Stage 12 | prove `combo_iteration` is materially lower on the high-arity product fixture |
| `none/arity_7/long_short_reversal` | Stage 12 if fixture/request is available | prove reversal semantics under fused traversal |
| `tp_sl_grid/arity_6/long_only` | Stages 13-16, 18-20 | compare with Stage 09 accepted `64 x 64` opt-in baseline |
| `tp_sl_grid/arity_6/long_short_reversal` | Stages 13-16, 18-20 | compare with Stage 09 accepted `64 x 64` opt-in baseline under reversal |
| `none/arity_1..3` | Stage 17 | prove dynamic selector avoids small-workload regressions |
| Stage-specific thread matrix | Stage 19 | compare `NUMBA_NUM_THREADS=1,2,4,6,8,12` with same request and artifact set |

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
- Stage 12 prefix traversal counters: `prefix_nodes_visited`,
  `prefix_nodes_reused`, `prefix_pruned_subtrees`, `prefix_pruned_candidate_upper_bound`,
  `selectivity_order`, `combo_iteration_candidates_per_sec` and compiled-loop
  entry/exit timing;
- Blocked TP/SL experimental counters remain non-active unless a new accepted
  TP/SL plan reopens them: `early_abandon_cells`, `early_abandon_candidates`,
  `tp_sl_total_trade_windows`, `tp_sl_unique_trade_windows`,
  `tp_sl_trade_window_reuse_ratio`,
  `tp_sl_weighted_reuse_by_cell_count`,
  `tp_sl_top_reused_window_count`,
  `tp_sl_cache_candidate_savings_estimate`;
- Stage 17-20 overhead counters: backend selector decision/reason, estimated bit
  ops, estimated trade-cell ops, `heap_update_ms`, `top_result_proxy_fill_ms`,
  `variant_hash_ms`, `canonical_params_build_ms`, `payload_json_ms`,
  `db_persist_ms`, `matrix_buffers_allocated`, `matrix_buffer_bytes`,
  `cell_metric_buffer_bytes`, `trade_tape_buffer_bytes`, `temporary_array_count`,
  `scratch_buffer_reuse_count`;
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
- Stage 12+ stages must compare against the nearest accepted baseline:
  production composite default for no-risk arity `6`/`7`, Stage 05-only and
  Stage 12-only modes when isolating each accepted backend, and Stage 09/current
  exact for TP/SL full-grid rows on a Stage 05+12 production-default
  checkout/runtime;
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
| Fused prefix traversal | compiled/iterative traversal only; same candidate identity, canonical variant order, stable `variant_hash`, top-N identity/order and no Python object allocation in hot path | 12 |
| Selectivity order | compute-order reordering is internal only; output order and tie-break use canonical order | 12 |
| TP/SL autotune and monotonic kernel | same top-N, `best_tp`, `best_sl`, metric tolerance, full-grid exact semantics and SL-wins tie rule | 13-14 |
| TP/SL early abandon | exact-safe upper bound proof for `total_return_pct desc`; disabled for unsupported rankings/sizing modes | 15 |
| Trade-window reuse telemetry | counters only unless a later plan accepts compiled grouping; no Python dict cache in production hot path | 16 |
| Dynamic selector | selector decisions are logged, deterministic and do not change public result identity; arity 1/2/3 no-regression is mandatory | 17 |
| Top-N/result assembly | stable tie-break by ranking metric, `variant_hash` and combo ordinal; persisted top-N shape unchanged | 18 |
| Thread scaling | no oversubscription, same request/artifacts per run, best thread count does not regress service wall or memory | 19 |
| Allocation reuse | per-child scratch buffers only, no global cross-job cache, cleanup and RSS peak do not regress | 20 |
| Product TP/SL coarse mode | exact mode remains default; approximate/coarse mode requires explicit product approval and visible mode semantics | 21 |

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
| 12 | Compiled prefix product traversal | Accepted 2026-06-13 and productionized through composite default: Stage 05 remains default for `none/arity_6`, `compiled_prefix_product_traversal_v1` becomes default for `none/arity_7`; explicit Stage 12 mode still supports arity `6`/`7`; stable top-50 `variant_hash`/rank/metrics matched baseline |
| 13 | TP/SL block-shape production gate | Rejected and removed from active tree; not executable |
| 13S/13S2 | TP/SL selective selector | Rejected and removed from active tree; not executable |
| 13R | TP/SL reversal diagnostics | Learning-only result retained in negative-results; runtime diagnostics removed from active tree |
| 14/14R | TP/SL monotonic/split-by-side repair | Superseded/rejected and removed from active tree; not executable |
| 15 | TP/SL total-return early abandon | Unblocked; exact-safe upper-bound proof and Mac Studio A/B versus current exact TP/SL baseline required |
| 16 | TP/SL trade-window reuse telemetry | Blocked until a new TP/SL telemetry/repair plan is accepted; no cache/grouped work |
| 17 | Dynamic backend selector | Blocked for TP/SL; may be reopened only for accepted no-risk backends with a new scoped prompt |
| 18 | Top-N/result assembly batch reduction | Assembly timers measured first; optional stable block top-M merge only if assembly is hot and top-N identity/order is unchanged |
| 19 | Thread scaling benchmark | `NUMBA_NUM_THREADS=1,2,4,6,8,12` matrix; worker config change only after service-wall evidence |
| 20 | Allocation reuse and scratch buffers | Allocation telemetry first; per-child scratch buffers only if service wall or RSS improves without cleanup regression |
| 21 | TP/SL exact/coarse mode architecture decision | Product-visible approximate mode policy, admission cost model and exact refine rules; no exact default change without separate approval |

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
| 12 | `prefix_traversal.py` registered as `compiled_prefix_product_traversal_v1`; no Python recursion in hot path; production composite default mode `stage_05_and_12_no_risk` | `benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_baseline_off/`, `benchmark_iterations/2026-06-13_matrix_bitset_stage_12_compiled_prefix_traversal_candidate_rerun2/`, `benchmark_iterations/2026-06-13_matrix_bitset_stage_05_12_production_default_live/` |
| 13/13S/13S2/13R/14R | none; rejected branch removed from active tree | no raw evidence retained in active tree; compact stop-list in `backtest-compute-acceleration-negative-results-v1.md` |
| 15 | exact-safe early-abandon implementation in the current exact TP/SL scorer plus a dedicated benchmark selector if missing; no Stage 13/14 runtime, prompt or harness restore | `benchmark_iterations/<date>_matrix_bitset_stage_15_tp_sl_early_abandon/` |
| 16 | blocked; no telemetry/grouping/cache work until a new TP/SL plan is accepted | none |
| 17 | blocked for TP/SL; optional future no-risk selector requires a new scoped prompt | none |
| 18 | top-N/result assembly timers and optional stable block top-M merge | `benchmark_iterations/<date>_matrix_bitset_stage_18_topn_batch_reduction/` |
| 19 | thread-scaling benchmark harness/report; config update only if accepted | `benchmark_iterations/<date>_matrix_bitset_stage_19_thread_scaling/` |
| 20 | allocation counters and per-child scratch buffers if accepted | `benchmark_iterations/<date>_matrix_bitset_stage_20_allocation_reuse/` |
| 21 | architecture/product decision record for exact vs coarse TP/SL modes | `docs/architecture/backtest/` ADR plus optional benchmark cost model |

## Контракты И Влияние

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | Existing routes and payload meanings stay unchanged |
| Port contract | `compatible-change` | New backend selector/config and telemetry may add optional fields |
| DTO schema | `compatible-change` | Telemetry/counter fields are additive |
| Persisted schema | `none` initially | Bitset artifacts are file artifacts, not DB schema |
| Canonical artifact file/manifest schema | `none` initially | Stage 07 sidecar files do not change canonical `manifest.yaml`, `current.yaml` or active-slot artifacts |
| Test sidecar artifact metadata | `compatible-change` | New benchmark/test-only `matrix_sidecar_manifest.json` records source hashes, shapes, dtypes and schema version |
| Config schema | `compatible-change` | Optional matrix backend modes/env overrides, explicit sidecar path for benchmark/test runs and disabled-by-default pair-cache controls |
| Worker/thread config | `compatible-change` | Stage 19 may add workload-specific thread policy only after benchmark evidence; fixed current behavior remains rollback |
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
- Stage 12+ feature flags/env overrides must support default-off/default-on A/B
  benchmark comparisons. A feature cannot become default unless the ledger records
  the rollback override and accepted comparison path.
- Repo/main acceptance is not the same as live production activation. Evidence
  must record whether the measured path is the Mac Studio project checkout, an
  isolated benchmark copy, or the active live runtime under `/opt/roehub/app`.
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
- Arity 10 exhaustive search may remain impractical even after Stage 12; any
  broader high-arity expansion still needs exact-safe pruning and comparable
  evidence. Approximate beam search requires explicit product approval and a
  separate contract.
- Compiled prefix traversal is production default only for no-risk arity `7` in
  the composite mode. Arity `6` remains Stage 05 by default unless a later
  selector/default gate proves Stage 12 is also better for the end-to-end arity-6
  service path.
- TP/SL monotonic and early-abandon stages can only apply to exact-safe surfaces;
  unsupported rankings must fall back to current exact scoring.
- Dynamic selector mistakes can silently erase Stage 12 no-risk wins, Stage 05
  rollback/default behavior, or re-enable arity 2/3 regressions, so selector
  telemetry and rollback override are mandatory.
- Thread scaling may be hardware-specific; accepted Mac Studio thread policy must
  not be generalized to different hardware without fresh evidence.
- Full pair cache can regress memory and wall-clock even when isolated pair lookup
  gets faster; it is excluded from default production path until bounded evidence
  proves otherwise.
- RMQ/sparse table is intentionally not planned for current closed-equity drawdown;
  adding mark-to-market drawdown acceleration would be a contract change.
