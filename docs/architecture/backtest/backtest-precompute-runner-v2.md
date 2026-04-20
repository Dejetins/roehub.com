# Backtest Precompute Runner V2 (R2-03 / R2-04 / R3-01 / R3-02 / R3-03 / R3-04 / R4-01 / R4-02 / R4-03 / R5-01 / R6-01)

Статус: `Milestone R2 / EPIC R2-03 + R2-04`, `Milestone R3 / EPIC R3-01 + R3-02 + R3-03 + R3-04`, `Milestone R4 / EPIC R4-01 + R4-02 + R4-03`, `Milestone R5 / EPIC R5-01`, `Milestone R6 / EPIC R6-01`

## Status

- Status: active canonical precompute/publish contract after R10-02 docs synchronization.
- D10 prerequisite note:
  - `wider TP/SL artifact grids` plus `grid-agnostic Stage B loaders` are a
    `completed prerequisite` for
    `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`;
  - this document is part of the confirming evidence set for that prerequisite together with
    `docs/architecture/backtest/backtest-runtime-kernels-v2.md`,
    `tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py::test_backtest_artifact_precompute_runner_v2_builds_widened_hit_times_manifest_shapes`,
    `tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py::test_backtest_artifact_precompute_runner_v2_materializes_hit_times_and_full_validation_passes`,
    `tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py::test_target_indicator_variant_counts_match_narrowed_catalog_per_env_v2`,
    `tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py::test_slice_hit_times_to_execution_window_v2_accepts_widened_artifact_grid`,
    `tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py::test_artifact_backed_stage_b_scorer_v2_resolves_widened_grid_risk_indexes`,
    and `tests/unit/contexts/backtest/application/services/v2/test_risk_exit_kernel_1m_v2.py::test_search_risk_cells_total_return_fast_v2_matches_bruteforce_exact_replay`;
  - `RG-TTR contradiction reopens blocker`: if live `RG-TTR` closure contradicts this
    prerequisite, the dependency must be reopened as a blocking defect explicitly rather than
    silently drifting roadmap status.
- Canonical runtime relationship:
  - precompute produces only artifact inputs for the active `signal_tf + 1m_risk` runtime;
  - sync launch, claimed worker, and run-scoped lazy detail consume the same published slot
    family through v2 loaders;
  - `signals.v1.params` remain `default-only` and do not expand into signal-grid runtime
    combinatorics.
- Compatibility note:
  - stage-specific prices+mappings publish remains an explicit helper only;
  - active production docs no longer describe live ClickHouse rollup or runtime recompute as
    equivalent alternatives.

Документ фиксирует контракт precompute/publish слоя, который:

- строит inactive slot в `artifacts/backtest/v2`;
- пишет strict manifests для root / signals / hit_times;
- выполняет fail-fast validation до switch `current.yaml`;
- сохраняет published artifact layout stable, even when the internal execution model changes;
- явно отделяет offline artifact-precompute execution semantics от публичных/runtime semantics
  `indicators`;
- оставляет runtime только fixed metadata reads без schema inference и без hash recomputation;
- после R6-01 передаёт runtime достаточно metadata для shared `slot-pinned context` bootstrap в
  sync и background paths без filesystem discovery.

Основные документы:

- `docs/architecture/backtest/backtest-artifact-store-v2.md`
- `docs/architecture/backtest/backtest-runtime-kernels-v2.md`
- `docs/architecture/backtest/backtest-compute-notebook-algorithm-v2.md`
- `docs/architecture/roadmap/base_refactor_plan.md`
- `docs/architecture/roadmap/backtest-refactor-final-plan-v2.md`
- `docs/architecture/roadmap/backtest-engine-vnext-parity-corrective-plan-v2.md`

## Config Inputs (R2-04)

Precompute/publish слой читает strict `configs/<env>/backtest_artifacts.yaml` contract.

Из него берутся:

- `artifact_root` для path builder / loader wiring;
- `validation_plan` для config-driven `ArtifactSlotValidationSpecV2`;
- `hit_times_grid` как source-of-truth TP/SL levels contract;
- `slot_policy`, `publish_schedule`, `lookback_policy`, `validation_budgets` как fail-fast
  validated pipeline settings.
- `validation_plan.signal_artifacts` может быть либо explicit list of
  `{timeframe, indicator_id}`, либо machine-readable literal `all_supported_v1`, который
  раскрывается в полный registry `supported_indicator_ids_for_signals_v1()` по всем
  `ARTIFACT_SIGNAL_TIMEFRAMES_V2`.
- `validation_budgets.max_hit_times_cells` ограничивает steady-state incremental
  `hit_times/1m` rebuild.
- `validation_budgets.max_hit_times_cells_full_rebuild` используется для первого bootstrap пустого
  symbol root и для explicit `--full-rebuild`, когда bounded incremental budget заведомо слишком
  мал для full-history `hit_times`.
- R12 documentation contract добавляет к strict config shape additive subsection
  `execution_policy`; checked-in R11 config files могут ещё не содержать эти keys, но follow-up
  implementation epics обязаны wire'ить их именно в `configs/<env>/backtest_artifacts.yaml`
  without changing artifact layout or public `/backtests*` contracts.
- signal artifact materialization intentionally does not inherit the public/runtime
  `configs/<env>/indicators.yaml -> compute.numba.max_compute_bytes_total` ceiling; offline
  precompute wiring uses a dedicated compute adapter with effectively-unbounded total compute
  budget, while hot-path API/runtime guards stay unchanged.

R2-04 intentionally keeps these settings отдельно от `configs/<env>/backtest.yaml`, чтобы
runtime request defaults и artifact pipeline knobs не смешивались в одном контракте.

## R13-01 signal-defaults right-sizing

R13-01 narrows only the heaviest non-`ma.*` artifact-precompute grids in
`configs/<env>/indicators.yaml`.

- `all_supported_v1` remains the canonical signal target literal; no indicator ids are removed.
- `signals.v1.params` stay `default-only`.
- `ma.*` defaults stay frozen.
- If an indicator already supports `inputs.source`, artifact defaults must keep the full canonical
  source catalog: `close`, `hlc3`, `ohlc4`, `low`, `high`, `open`.
- `prod` and `dev` should stay aligned; `test` may stay smaller only for deterministic unit-test
  speed, but must keep the same targeted indicator ids and canonical source catalog where source is
  already supported.

Compact before/after guidance for the worst offenders:

| indicator_id | previous compute variants | narrowed compute variants |
| --- | ---: | ---: |
| `momentum.trix` | `12528` | `192` |
| `momentum.stoch` | `9396` | `30` |
| `trend.adx` | `5336` | `24` |
| `volatility.hv` | `1392` | `96` |

## R12 execution-model clarification

R12 фиксирует, что precompute execution model меняется без изменения published artifact output
contract.

- Stable output contract:
  - `signals/<tf>/<indicator_id>/signals.i8.npy` остаётся published signal path;
  - `axis_order: [variant, time]` остаётся canonical signal matrix layout;
  - root/signal/hit-times manifests, slot layout и `current.yaml` publish semantics не меняются.
- Changed offline execution model:
  - artifact precompute больше не должен трактоваться как giant tensor-first materialization для
    каждого target timeframe/indicator pair;
  - canonical model теперь stage-oriented и built around `timeframe-scoped execution`;
  - один `current_timeframe` session открывается, полностью отрабатывается и закрывается до
    перехода к следующему timeframe.
- Unchanged public/runtime `indicators` semantics:
  - public `IndicatorCompute.compute(...)` и его tensor-first guards остаются отдельным contract;
  - offline artifact precompute может использовать другой bounded orchestration layer и не обязан
    наследовать giant in-memory dense tensor model из runtime/API indicator compute.

### Short glossary

| Term | Meaning |
| --- | --- |
| `stage` | Крупный deterministic pipeline segment with stable code ids: `canonical_prices`, `hit_times`, repeated `timeframe_session`, `root_manifest`, `publish` |
| `rolled_prices` | Deterministic per-timeframe `prices/<tf>` family derived once immediately before that timeframe enters `timeframe_session` |
| `timeframe session` | Lifetime одного open target timeframe inside precompute: load derived prices once, process all indicators on that timeframe, flush writes, release memory |
| `chunk` | Contiguous row range for one `(indicator_id, timeframe)` signal matrix |
| `worker` | Один bounded signal worker process, владеющий только своим chunk-local compute buffers |
| `publish` | Sequence `build inactive slot -> validate whole slot -> atomically switch current.yaml` |

### Operator-friendly pipeline

1. Load canonical `1m` once from `market_data.canonical_candles_1m FINAL`.
2. Materialize `prices/1m`.
3. Build or update `hit_times/1m` while only canonical `1m` scope is live.
4. For one timeframe at a time derive and write `rolled_prices` as `prices/<tf>`.
5. Open a `timeframe session` for that `current_timeframe`.
6. Inside that session build `mappings/<tf>` and materialize all configured
   `signals/<tf>/<indicator_id>/signals.i8.npy` targets, including `all_supported_v1`.
7. Eagerly write signal chunks to disk via `np.memmap`, then close the timeframe session.
8. After all timeframe sessions finish, finalize manifests and run whole-slot validation.
9. Publish by atomically switching `current.yaml`.

### Architecture matrix

| Stage | Inputs | Outputs | Memory owner | Parallelism scope |
| --- | --- | --- | --- | --- |
| `canonical_prices` | canonical CH `1m` columns | `prices/1m/*` | runner process | single stage |
| `hit_times` | materialized `prices/1m/ohlcv.f32.npy`, TP/SL grid | `hit_times/1m/*` | runner process | single stage in canonical scope |
| `rolled_prices` | materialized `prices/1m` + one target timeframe | `prices/<tf>/*` | owner of the soon-to-open timeframe session | one timeframe at a time |
| `timeframe_session` | one `rolled_prices` target + signal target list | `mappings/<tf>/*`, `signals/<tf>/<indicator_id>/signals.i8.npy` | session owner for one `current_timeframe` | one open timeframe session at a time |
| `signal_chunk` | one `(indicator_id, timeframe)` + contiguous variant rows | row slice inside `signals/<tf>/<indicator_id>/signals.i8.npy` | one worker process | bounded by `signal_worker_processes` |
| `manifests_publish` | completed slot files | `manifest.yaml`, `current.yaml` | runner + publisher | no data parallelism |

### Why the old tensor-first model is not recommended here

The old narrative implicitly suggested reusing dense `IndicatorCompute` full-target tensors for each
artifact target. That is not the recommended architecture for artifact precompute because it:

- couples offline signal materialization to runtime/API memory guards that solve a different
  problem;
- keeps multiple large target-specific buffers alive longer than needed;
- scales poorly on Mac Studio once `all_supported_v1` is combined with long histories and many
  target timeframes;
- hides ownership boundaries, which makes swap/compressed-memory incidents harder to diagnose.

The recommended future state is bounded stage execution, not a bigger giant tensor.

### `execution_policy` contract

`execution_policy` is an additive subsection of `configs/<env>/backtest_artifacts.yaml` reserved for
offline precompute orchestration only.

```yaml
backtest_artifacts:
  execution_policy:
    max_open_timeframe_sessions: 1
    signal_worker_processes: 4
    signal_worker_memory_budget_bytes: 2147483648
    signal_chunk_rows_min: 32
    signal_chunk_rows_max: 256
```

Field contract:

- `max_open_timeframe_sessions`
  - positive integer;
  - deterministic default: `1`;
  - meaning: upper bound on concurrently open target timeframe sessions;
  - current canonical production meaning: only `1` is recommended, because R12 architecture is
    explicitly one `current_timeframe` at a time;
  - implementations that cannot preserve the same memory bounds for values `> 1` must fail-fast.
- `signal_worker_processes`
  - positive integer;
  - deterministic default: `4`;
  - meaning: fixed upper bound of concurrent signal worker processes inside one timeframe session;
  - must never imply unbounded per-target fan-out.
- `signal_worker_memory_budget_bytes`
  - positive integer;
  - deterministic default: `2147483648` (`2 GiB`);
  - meaning: per-worker ceiling for one signal chunk compute job, including chunk-local scratch
    buffers and the writable output slice.
- `signal_chunk_rows_min`
  - positive integer;
  - deterministic default: `32`;
  - meaning: smallest acceptable contiguous variant-row batch for a chunk job.
- `signal_chunk_rows_max`
  - positive integer;
  - deterministic default: `256`;
  - meaning: largest acceptable contiguous variant-row batch for a chunk job.

Deterministic validation expectations:

- `execution_policy` must reject missing required fields, extra keys, duplicates and non-integer
  values.
- `signal_chunk_rows_min <= signal_chunk_rows_max` is mandatory.
- `max_open_timeframe_sessions >= 1`, `signal_worker_processes >= 1`,
  `signal_worker_memory_budget_bytes >= 1` are mandatory.
- No hidden host-derived defaults are allowed once the section is wired in code.
- No other scalar knobs are required for R12; if more tuning is needed, the follow-up epic must
  justify why the five fields above were insufficient.

### R12 implementation contract surface

R12-01 wires the coordinator/execution-policy foundation, and R12-02 completes the chunked
artifact-only signal stage without changing the published artifact layout or
`PublishBacktestArtifactsV2UseCase` semantics.

- Typed runtime config:
  - `BacktestArtifactExecutionPolicyRuntimeConfig`
  - `ArtifactPrecomputeExecutionPolicyV2`
- Coordinator and lifecycle ownership:
  - `ArtifactPrecomputeCoordinatorV2`
  - `ArtifactTimeframeSessionV2`
- Chunk planning and artifact-only signal execution:
  - `ChunkPlannerV2`
  - `ArtifactSignalChunkPlanningRequestV2`
  - `ArtifactSignalChunkJobV2`
  - `DeterministicSignalChunkPlannerV2`
- Typed stage/progress DTOs:
  - `ArtifactPrecomputeStageInputV2`
  - `ArtifactPrecomputeStageOutputV2`
  - `ArtifactPrecomputeStageResultV2`
  - `ArtifactPrecomputeProgressEventV2`

Deterministic stage order in code:

1. `canonical_prices`
2. `hit_times`
3. repeated `timeframe_session` in canonical timeframe order (`15m` .. `3d`)
4. `root_manifest`

Contract expectations:

- `ArtifactPrecomputeCoordinatorV2` is the single owner of stage ordering and structured progress
  events.
- `ArtifactTimeframeSessionV2` is the single owner of one open `current_timeframe`; when
  `max_open_timeframe_sessions=1`, nested sessions must fail-fast.
- `ArtifactCanonicalPriceExportResultV2` carries additive `stage_results` summaries for later
  metrics/logging work, while existing publish diagnostics continue to aggregate into
  `stage_rebuild_stats` / `tail_rebuild_bars`.

### `ChunkPlanner` contract

`ChunkPlanner` is the deterministic planner for signal materialization inside one timeframe session.

Inputs:

- `indicator_id`;
- `timeframe`;
- `timeline_bar_count` for the final `signals/<tf>/<indicator_id>/signals.i8.npy`;
- `variant_count` for that `(indicator_id, timeframe)` target;
- `estimated_bytes_per_row`;
- `worker_memory_budget_bytes`;
- `signal_chunk_rows_min`;
- `signal_chunk_rows_max`;
- canonical variant row order for this target, as already fixed by manifest/grid contracts.

Planning algorithm:

1. Compute `budget_cap_rows = floor(signal_worker_memory_budget_bytes / estimated_bytes_per_row)`.
2. Fail-fast if `estimated_bytes_per_row <= 0`.
3. Compute `effective_chunk_rows_min = min(signal_chunk_rows_min, variant_count)`.
4. Fail-fast if `budget_cap_rows < effective_chunk_rows_min`, because the configured minimum chunk
   size cannot fit in the worker budget.
5. Set
   `chunk_rows = min(signal_chunk_rows_max, variant_count, max(effective_chunk_rows_min, budget_cap_rows))`.
6. Emit contiguous row ranges in canonical order:
   - chunk `0` -> `[0, chunk_rows)`
   - chunk `1` -> `[chunk_rows, 2 * chunk_rows)`
   - ...
   - final chunk may be shorter but must keep the same row ordering.

Outputs:

- ordered chunk jobs with:
  - `indicator_id`
  - `timeframe`
  - `chunk_index`
  - `chunk_count`
  - `row_start_inclusive`
  - `row_end_exclusive`
  - `chunk_rows`

Determinism guarantees:

- one chunk is always a bounded subset of variant rows for one `(indicator_id, timeframe)`;
- chunks never mix multiple indicators or multiple timeframes;
- chunk execution may complete out of order, but write ownership stays attached to the original
  row range, so final matrix ordering and manifest ordering do not change;
- reconstructing all chunks in `chunk_index` order yields the same `axis_order: [variant, time]`
  matrix as a single non-chunked materialization.

Worked example:

- `variant_count = 1200`
- `estimated_bytes_per_row` fits `budget_cap_rows = 96`
- `signal_chunk_rows_min = 32`
- `signal_chunk_rows_max = 64`
- therefore `chunk_rows = min(64, 1200, max(32, 96)) = 64`
- number of jobs = `ceil(1200 / 64) = 19`
- jobs `0..17` own `64` rows each, job `18` owns the final `48` rows.

### Memory ownership and worker model

Memory ownership is explicit and must be released in the same order on every run:

- canonical source arrays:
  - owned by the main runner;
  - read-only;
  - loaded once for the symbol root and reused across later stages.
- one current timeframe session:
  - owns derived `prices/<tf>` references, `mappings/<tf>` build state and the currently open
    signal target descriptors for exactly one `current_timeframe`;
  - must be closed before the next timeframe is opened.
- per-worker chunk buffers:
  - owned by exactly one worker process;
  - may hold only chunk-local indicator values, scratch arrays and one writable output slice;
  - must be released immediately after the chunk flushes.
- on-disk signal destination:
  - owned by `signals/<tf>/<indicator_id>/signals.i8.npy`;
  - written eagerly through `np.memmap`;
  - workers may touch only their assigned row range.

Mac Studio worker model:

- bounded worker pool only; no unbounded per-target parallelism;
- `max_open_timeframe_sessions` must cap how many timeframe sessions are simultaneously alive;
- `signal_worker_processes` must cap chunk workers inside that single session;
- the intended optimization target is throughput without swap/compressed-memory blowups, not
  maximal instantaneous fan-out;
- mandatory close semantics before moving to the next timeframe:
  - flush and close `np.memmap`
  - join/stop signal workers
  - release session-local arrays
  - clear `current_timeframe`

### Progress observability contract

Operator-facing observability is split into coarse metrics and fine-grained structured logs.

- Prometheus answers whether the overall publish cycle is healthy and whether `rewritten_tail_bars`
  stay bounded.
- Structured logs answer where the runner is currently spending time.
- `artifact_precompute_chunk_finished` adds `completed_chunks_total`, while the enclosing
  `timeframe_session` stage result carries both `completed_chunks_total` and
  `completed_indicators_total`.

Minimal structured log events:

- `artifact_precompute_stage_started`
- `artifact_precompute_stage_finished`
- `artifact_precompute_chunk_started`
- `artifact_precompute_chunk_finished`

Minimal structured log fields:

- `stage`
- `current_timeframe`
- `current_indicator_id`
- `chunk_index`
- `chunk_count`
- `row_start_inclusive`
- `row_end_exclusive`
- `chunk_rows`
- `completed_chunks_total`
- `completed_indicators_total` (in `timeframe_session` completion details)

These fields are mandatory for distinguishing a long bootstrap from a normal daily tail rebuild:

- bootstrap should show long `canonical_prices`, `hit_times`, and repeated
  `timeframe_session current_timeframe=<tf>` progress with large `rewritten_tail_bars`;
- steady-state rebuild should show one open `current_timeframe` at a time and bounded chunked
  progress inside it.

## Operational execution topology

- Precompute/publish orchestration lives in a dedicated artifact service on Mac Studio native
  backend and is not triggered inline by API requests or by `backtest-job-runner`.
- Scheduled mode is anchored to `Europe/Moscow` and runs daily at `03:05`.
- Instrument universe source-of-truth is `market_data.ref_instruments`; scheduled mode processes
  all enabled+tradable pairs from the latest snapshot.
- Manual mode may target one explicit `(exchange, market_type, symbol)` or an explicit subset, but
  must use the same inactive-slot build, whole-slot validation, and atomic `current.yaml` switch.
- Shared orchestration entrypoint is `PublishBacktestArtifactsV2UseCase`; manual CLI and the later
  daily scheduler must call the same use-case instead of wiring precompute/publish services
  separately.
- Manual operator entrypoint is
  `uv run python -m apps.cli.main.main backtest-artifact-publish --exchange <exchange> --market-type <market_type> --symbol <symbol> [--full-rebuild]`.
- The shared result contract returns deterministic per-target diagnostics with
  `publish_mode in {bootstrap, incremental, full_rebuild}`, old/new slot identity, and whole-slot
  validation summary.
- Manual CLI additionally emits stage progress logs
  `event=artifact_precompute_stage_started|artifact_precompute_stage_finished`, while Prometheus
  counters on `backtest-artifact-publisher` remain service-level and do not count one-off CLI
  executions.
- Manual CLI and the scheduled publisher both use the same artifact-precompute-only indicators
  compute wiring, so full-registry signal materialization is not blocked by the public
  `max_compute_bytes_total` guard from `indicators.yaml`.
- Prod `artifact_root` must be a stable host data path outside repo checkout; relative
  checkout-local roots remain acceptable only for dev/test wiring.
- Production wiring fixes `artifact_root` at `/opt/roehub/state/backtest_artifacts/v2`;
  dev/test may continue to use repo-local `artifacts/backtest/v2`.
- Service execution must be protected by host-level locking so overlapping rebuild/publish runs do
  not mutate the same inactive slot concurrently.
- Service observability is part of the contract. Minimal Prometheus set:
  - `backtest_artifact_publish_runs_total{status}`
  - `backtest_artifact_publish_duration_seconds`
  - `backtest_artifact_publish_symbols_total{status}`
  - `backtest_artifact_publish_blocked_total{reason}`
  - `backtest_artifact_publish_last_success_unixtime`
  - `backtest_artifact_tail_rebuild_bars_total{stage}`

## Bootstrap and incremental rebuild policy

- Если для symbol root ещё нет valid `current.yaml` и published slot, runner обязан выполнить
  bootstrap full build и создать initial published identity через обычный publish contract.
- После bootstrap daily rebuild не должен по умолчанию выполнять full-history recompute для всех
  стадий.
- `prices`, `mappings`, `signals` используют bounded incremental rebuild по:
  - `lookback_policy.price_tail_bars_1m`
  - `lookback_policy.mapping_tail_bars_1m`
  - `lookback_policy.signal_tail_bars_1m`
- `hit_times/1m` должны использовать такой же bounded incremental rebuild по
  `lookback_policy.hit_times_tail_bars_1m`.
- `hit_times/1m` budget policy разделяется по режимам:
  - bootstrap пустого symbol root и explicit `full_rebuild` используют
    `validation_budgets.max_hit_times_cells_full_rebuild`;
  - steady-state incremental rebuild uses `validation_budgets.max_hit_times_cells`.
- widened-grid budgeting is part of the canonical contract:
  - total strict table cells = `timeline_bar_count * (2 * len(tp_values) + 2 * len(sl_values))`;
  - canonical `tp_values = [0.5, 1.0, ..., 50.0]` (`100` levels) and
    `sl_values = [0.5, 1.0, ..., 25.0]` (`50` levels) therefore produce `300` cells per `1m`
    bar;
  - default `hit_times_tail_bars_1m = 20_000` yields `6_000_000` cells
    (about `22.9 MiB` of raw `uint32` table bytes);
  - `max_hit_times_cells_full_rebuild = 1_500_000_000` aligns bootstrap/full rebuild with the
    existing `5_000_000`-bar validation ceiling
    (about `5.6 GiB` of raw `uint32` table bytes before allocator/runtime overhead).
- Runner result contract обязан публиковать explicit stage-level stats для `prices`, `mappings`,
  `signals`, `hit_times`:
  - `reused_prefix_bars`
  - `rewritten_tail_bars`
  - `tp_level_count`
  - `sl_level_count`
  - `table_cell_count`
  - scheduler/prometheus aggregation по-прежнему строится из per-stage `rewritten_tail_bars` через
    `backtest_artifact_tail_rebuild_bars_total{stage}`.
- Если reuse prerequisites нарушены для конкретного stage или symbol root, выполняется
  deterministic full rebuild только для этого symbol root, после чего whole-slot validation и
  publish semantics остаются неизменными.

## Область ответственности

Precompute runner v2 обязан:

- писать файлы только в inactive slot;
- использовать deterministic paths из R2-01;
- писать root `manifest.yaml`;
- писать per-indicator `signals/<tf>/<indicator_id>/manifest.yaml`;
- additive-писать `signal_features/<tf>/<indicator_id>/manifest.yaml` для новых signal targets;
- писать `hit_times/1m/manifest.yaml`;
- указывать в manifests fixed runtime metadata:
  - `dtype`
  - `shape`
  - `axis_order`
  - `sha256`
  - `provenance`
  - `slot_generation`
  - `timeline` coverage
- завершать publish только после whole-slot validation.

### R4-01 / R4-02 / R4-03 signal boundary

На этапе R4-01 precompute layer получил explicit signal-rules engine contract.
На этапе R4-02 этот contract стал source-of-truth для real signal artifact materialization.

Это означает:

- indicator outputs детерминированно преобразуются в compact `int8` signals `{-1,0,1}`;
- `inputs.source` трактуется явно для rule families, где price сравнивается с indicator output;
- `signals.v1.params` остаются strictly `default-only` и берутся только из
  `configs/<env>/indicators.yaml`;
- required candle-series for each signal target still come from the indicators compute contract;
  catalog-wide tests must keep hard definitions aligned with the real compute path so precompute
  never requests undeclared fixed inputs;
- zero-axis signal targets `structure.candle_stats`, `volatility.tr`, `volume.ad_line`,
  `volume.obv` are valid even when `compute_defaults(...)` is absent in YAML:
  runner derives a deterministic single-row `GridSpec` from the hard indicator definition with
  `Layout.VARIANT_MAJOR`, while axis-bearing indicators still fail fast on missing defaults;
- для каждого explicit target из `backtest_artifacts.validation_plan.signal_artifacts`
  runner обязан писать:
  - `signals/<tf>/<indicator_id>/signals.i8.npy`
  - `signals/<tf>/<indicator_id>/manifest.yaml`
  - `signal_features/<tf>/<indicator_id>/features.f32.npy`
  - `signal_features/<tf>/<indicator_id>/manifest.yaml`
- если config использует `signal_artifacts: all_supported_v1`, explicit target set равен полному
  registry всех signal-capable indicators, а не сокращённому operator-curated subset;
- root manifest обязан публиковать real `signals.supported_timeframes`,
  `signals.supported_indicator_ids` и `signals.manifests`;
- signal manifest может additively ссылаться на `signal_features` того же `(timeframe,
  indicator_id)`, но старые слоты без этой ссылки остаются publish/runtime-compatible;
- signal features derive only from the already materialized signal row and its final timeline
  length; pair-specific, TP/SL-specific и runtime-threshold-dependent fields запрещены;
- после R4-03 rebuild обязан выводить bounded per-target signal window из
  `lookback_policy.signal_tail_bars_1m`, а затем materialize'ить только
  `prefix + rebuilt_tail` по time axis;
- prefix reuse разрешён только при strict reuse-check:
  - target уже перечислен в root `signals.manifests`
  - existing `manifest.yaml` и `signals.i8.npy` существуют
  - `rows_count`, `timeline`, `variant_key_version`, `variant_keys_sha256`,
    `signals.v1.params = default-only` и file `sha256` не дрейфуют
- missing target files могут переводить target в deterministic full rebuild, но manifest/data
  drift при reuse attempt обязан fail-fast с stable diagnostics;
- R4-04 propagation `source` в runtime payloads теперь закрывается downstream-контрактами:
  `GET /backtests/runtime-defaults`, persisted jobs `/top` payloads и explicit
  `variant-report` payloads.

### R5-01 `hit_times/1m` boundary

На этапе R5-01 precompute runner materialize'ит strict `hit_times/1m` family из уже
artifact-backed `prices/1m.ohlcv`.

Это означает:

- `backtest_artifacts.hit_times_grid` становится source-of-truth для `tp_values` и `sl_values`;
- canonical widened grid for `configs/{dev,test,prod}/backtest_artifacts.yaml` is explicit:
  - `tp_levels_pct = [0.5, 1.0, ..., 50.0]` (`100` levels);
  - `sl_levels_pct = [0.5, 1.0, ..., 25.0]` (`50` levels);
- runner обязан писать real files:
  - `hit_times/1m/tp_values.f32.npy`
  - `hit_times/1m/sl_values.f32.npy`
  - `hit_times/1m/long_tp.u32.npy`
  - `hit_times/1m/long_sl.u32.npy`
  - `hit_times/1m/short_tp.u32.npy`
  - `hit_times/1m/short_sl.u32.npy`
  - `hit_times/1m/manifest.yaml`
- `sentinel_index` обязан равняться `timeline_bar_count`, а таблицы обязаны оставаться
  bounded-by-sentinel и monotone by level;
- root manifest больше не должен публиковать placeholder hash для `hit_times`, если slot построен
  этим R5-01 path;
- runtime читает `hit_times/1m` только по strict manifest metadata, без recompute и discovery.
- daily rebuild policy для `hit_times/1m` должна быть bounded, а не full-history by default:
  - source-of-truth lookback: `lookback_policy.hit_times_tail_bars_1m`;
  - merge strategy: `prefix + rebuilt_tail`;
  - missing existing files, grid drift или manifest drift переводят symbol root в deterministic
    full rebuild;
  - unchanged prefix columns must stay byte-stable between repeated daily runs;
- на этом boundary ответственность precompute слоя заканчивается: `signal timeline`,
  `execution timeline`, `compact trade list`, `fast TP/SL grid search`,
  `exact replay of best TP/SL cell` и `metrics over compact trades` описываются отдельно в
  `docs/architecture/backtest/backtest-runtime-kernels-v2.md`;
- precompute runner materialize'ит inputs для `signal_tf + 1m_risk`, но не становится notebook
  orchestration layer.

### R3-01 / R3-02 prices stage

На этапах R3-01 / R3-02 precompute runner получает отдельную обязанность:

- материализовать canonical source-of-truth export для `prices/1m/*` в inactive slot;
- затем построить из materialized `prices/1m/*` только разрешённые request TF:
  - `15m`
  - `30m`
  - `1h`
  - `2h`
  - `4h`
  - `6h`
  - `8h`
  - `1d`
  - `2d`
  - `3d`
- писать для каждого TF:
  - `prices/<tf>/open_time.i64.npy`
  - `prices/<tf>/close_time.i64.npy`
  - `prices/<tf>/ohlcv.f32.npy`
- использовать source table `market_data.canonical_candles_1m` только для canonical `1m`
  export через existing `CanonicalCandleReader` contract;
- precompute fast path читает `market_data.canonical_candles_1m FINAL` columnar arrays напрямую в
  numeric numpy payload, чтобы не создавать миллионы `CandleWithMeta` / `datetime` объектов во
  время full-history bootstrap;
- строить rollup только из artifact-backed `prices/1m`, без ClickHouse reads на runtime hot path;
- поддерживать deterministic tail update по
  `backtest_artifacts.lookback_policy.price_tail_bars_1m`;
- строить `mappings/<tf>/bar_open_1m_idx.u32.npy` и
  `mappings/<tf>/bar_close_1m_idx.u32.npy` только из artifact-backed `prices/1m` и
  `prices/<tf>`;
- поддерживать deterministic tail update для mappings по
  `backtest_artifacts.lookback_policy.mapping_tail_bars_1m`;
- никогда не мутировать active slot in place.
- Independent `prices/<tf>` rollups и `mappings/<tf>` builds могут выполняться параллельно, но
  manifest ordering и whole-slot publish contract остаются deterministic.

Tail update semantics для R3-01 / R3-02 / R3-03:

- если inactive slot ещё не содержит valid `prices/1m`, выполняется full build по заданному
  `TimeRange [start, end)`, затем full rollup для всех allowed request TF;
- если `prices/1m` уже существует в inactive slot, runner переиспользует prefix внутри requested
  range и reread'ит только tail overlap длиной `price_tail_bars_1m`;
- для rolled `prices/<tf>` prefix reuse считается от bucket, в который попадает reread-tail start;
- для `mappings/<tf>` prefix reuse считается до последнего request-TF бара, чей `close_time`
  остаётся строго левее первого `1m` bar open, попавшего в mapping-tail window;
- mapping rebuild обязан сохранять `dtype=uint32`, `shape=[T_tf]`,
  `bar_open_1m_idx <= bar_close_1m_idx` и exact price correspondence;
- merge policy фиксирована как `prefix + rebuilt_tail`, без best-effort dedup/coercion;
- identical source candles + identical config/request должны давать byte-stable `.npy` и
  `manifest.yaml`.

### R4-03 signal tail-update semantics

Signal rebuild для explicit configured targets обязан быть локальным и deterministic:

- source-of-truth для bounded signal tail rebuild:
  - `lookback_policy.signal_tail_bars_1m`
  - target timeframe duration
  - finite compute context, выведенный из materialized grid axes
  - finite lag/default-only context из `signals.v1.params`
- effective tail window считается в target bars и используется только для explicit configured
  `(timeframe, indicator_id)` targets;
- compute window строится локально внутри precompute runner internals без filesystem discovery;
- merge policy фиксирована как `prefix + rebuilt_tail`, без hidden dedup/coercion;
- merged matrix обязана оставаться strict:
  - `dtype: int8`
  - `shape: [V, T_tf]`
  - `axis_order: [variant, time]`
  - value set `{-1,0,1}`
- per-indicator manifest после merge обязан обновлять:
  - `rows_count`
  - `timeline`
  - `signals.sha256`
  - provenance inputs с `lookback_policy.signal_tail_bars_1m`,
    `effective_target_tail_bars` и `rebuild_strategy = prefix + rebuilt_tail`
- correctness proof для long-window targets обязана учитывать explicit `warmup`:
  - effective compute window может быть шире, чем `effective_target_tail_bars`;
  - shipped incremental result должен совпадать с deterministic full rebuild, даже если naive
    tail cut без warmup дал бы другой `signals.i8.npy`.
- root `signals` catalog обязан оставаться deterministic:
  - `signals.supported_timeframes` deduplicated in canonical timeframe order
  - `signals.supported_indicator_ids` in lexical order
  - `signals.manifests` ordered by `(timeframe, indicator_id)`
- identical source candles + identical config/request + identical generated timestamp должны
  давать byte-stable `signals/<tf>/<indicator_id>/signals.i8.npy` и related manifests.

Rollup contract для R3-02:

- bucket boundaries считаются только через `Timeframe.bucket_open/bucket_close`;
- materialize'ятся только fully covered epoch-aligned buckets;
- partial leading/trailing buckets детерминированно отбрасываются;
- `open_time` / `close_time` пишутся отдельно от `ohlcv`;
- root manifest обязан содержать metadata и coverage для `1m` и всех rolled request TF.

Precompute runner v2 не должен:

- мутировать active slot in place;
- discover'ить содержимое через directory scanning;
- делать dynamic schema discovery;
- переносить expensive hash validation в runtime hot path.

## Manifest outputs

### Root `manifest.yaml`

Root manifest обязан фиксировать:

- `identity` (`exchange`, `market_type`, `symbol`);
- `slot`, `slot_generation`, `asof_date`;
- `prices[]` с metadata для `open_time`, `close_time`, `ohlcv`;
- `mappings[]` с metadata для `bar_open_1m_idx`, `bar_close_1m_idx`;
- `signals.supported_timeframes`;
- `signals.supported_indicator_ids`;
- `signals.manifests[]` с `manifest_path` и `manifest_sha256`;
- `hit_times.manifest_path` и `manifest_sha256`;
- `signal_encoding`:
  - `dtype: int8`
  - `axis_order: [variant, time]`
  - `value_set: [-1, 0, 1]`
- `provenance`.

R3-01 / R3-02 / R3-03 placeholder strategy до materialization следующих stage:

- `prices[]` содержит свежие strict sections для `1m` и всех allowed request TF;
- `mappings[]` может оставаться пустым до R3-03;
- `signals` фиксируется как explicit empty catalog
  (`supported_timeframes=[]`, `supported_indicator_ids=[]`, `manifests=[]`) до R4-02;
- `hit_times` обязан оставаться explicit fixed-path reference
  `hit_times/1m/manifest.yaml`, но до R5-01 допускается placeholder
  `manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"`;
- `signal_encoding` остаётся fixed even when `signals.manifests` is empty.

R3-03 mapping contract:

- `mappings[]` больше не placeholder и обязан содержать non-empty strict sections для:
  - `15m`
  - `30m`
  - `1h`
  - `2h`
  - `4h`
  - `6h`
  - `8h`
  - `1d`
  - `2d`
  - `3d`
- для каждого section обязательны metadata:
  - `path`
  - `dtype`
  - `shape`
  - `axis_order`
  - `sha256`
- validator обязан подтверждать:
  - bounds within `[0, T_1m)`
  - monotonicity
  - `bar_open_1m_idx <= bar_close_1m_idx`
  - `prices/1m.open_time[bar_open_1m_idx] == prices/<tf>.open_time`
  - `prices/1m.close_time[bar_close_1m_idx] == prices/<tf>.close_time`

R4-02 replaces the root-manifest signal placeholder for explicit configured targets:

- `signals.supported_timeframes` must equal the deduplicated ordered timeframes from
  `signals.manifests`;
- `signals.supported_indicator_ids` must equal the lexical ordered indicator ids from
  `signals.manifests`;
- `signals.manifests` must be ordered deterministically by timeframe contract then
  `indicator_id`;
- `signals.manifests` remains explicit configured-target metadata; directory scanning is not a
  supported source of truth;
- root manifest keeps `hit_times/1m` as an explicit placeholder reference only until R5-01.

### Per-indicator signal manifest

Каждый `signals/<tf>/<indicator_id>/manifest.yaml` обязан фиксировать:

- `indicator_id`, `timeframe`;
- `signals.path = signals/<tf>/<indicator_id>/signals.i8.npy`;
- `signals.dtype = int8`;
- `signals.shape = [V, T_tf]`;
- `signals.axis_order = [variant, time]`;
- `signals.sha256`;
- `rows_count = V`;
- `timeline` coverage, совпадающий с root `prices/<tf>.coverage`;
- `signal_value_set: [-1, 0, 1]`;
- `grid.variant_key_version: 1`;
- `grid.variant_keys_sha256`;
- `grid.signals_v1_params_defaults` из strict `signals.v1.params = default-only`;
- `provenance`.

Optional C1 extension in the same manifest:

- `signal_features.manifest_path = signal_features/<tf>/<indicator_id>/manifest.yaml`;
- `signal_features.manifest_sha256`;
- absence of this field remains valid for old slots and must not block exact runtime reads.

R4-03 provenance additions for per-indicator signal manifests:

- `inputs_sha256` must include `lookback_policy.signal_tail_bars_1m`;
- `inputs_sha256` must include the effective target tail budget derived for the target;
- `inputs_sha256` must include `rebuild_strategy = prefix + rebuilt_tail`.

### Additive signal-features manifest

Каждый `signal_features/<tf>/<indicator_id>/manifest.yaml` при наличии обязан фиксировать:

- `indicator_id`, `timeframe`, `slot`, `slot_generation`, `asof_date`;
- `features.path = signal_features/<tf>/<indicator_id>/features.f32.npy`;
- `features.dtype = float32`;
- `features.shape = [V, 6]`;
- `features.axis_order = [variant, feature]`;
- `features.sha256`;
- `rows_count = V`;
- fixed `feature_names` order:
  - `nonzero_count`
  - `long_count`
  - `short_count`
  - `activity_ratio`
  - `direction_balance`
  - `transition_count`
- `provenance`.

C1 runtime neutrality:

- precompute writes this family as a warm row-cache foundation only;
- shortlist, scoring, heuristic profiles and adaptive selector behavior remain unchanged in this
  milestone.

### `hit_times/1m/manifest.yaml`

`hit_times/1m/manifest.yaml` обязан фиксировать:

- `timeline_bar_count`;
- `sentinel_index`;
- `tp_values` и `sl_values`;
- `tables.long_tp|long_sl|short_tp|short_sl`;
- `monotonicity: non_decreasing_by_level`;
- `provenance`.

## Validator responsibilities

Whole-slot validator обязан идти в фиксированном порядке:

1. root manifest schema + root contract;
2. price arrays;
3. mapping arrays;
4. signal manifest refs + signal manifests + `signals.i8.npy`;
5. optional signal-feature refs + `signal_features/<tf>/<indicator_id>/manifest.yaml` +
   `features.f32.npy` when the owning signal manifest declares them;
6. hit-times manifest ref + hit-times manifest + `tp/sl` grids + tables.

Для каждого artifact family validator обязан проверять:

- exact required keys / no unsupported drift;
- expected path literal;
- file existence;
- file `sha256`;
- `dtype`;
- `shape`;
- `axis_order`.

Дополнительно:

- prices:
  - `open_time` strict monotonicity
  - `close_time` strict monotonicity
  - `close_time > open_time`
  - timeline coverage metadata
- mappings:
  - non-decreasing indexes
  - `bar_open_1m_idx <= bar_close_1m_idx`
  - mapping bounds относительно `1m`
  - exact correspondence с materialized `prices/1m` и `prices/<tf>`
- signals:
  - signal value set `{-1,0,1}`
  - `shape=[V,T_tf]`
  - timeline equality с root price coverage
  - deterministic root catalog ordering and hash/path correspondence
- signal_features:
  - validation is optional/additive and runs only when signal manifest declares the family
  - `shape=[V,6]`
  - `feature_names` fixed and ordered
  - values must be finite
- hit_times:
  - `tp/sl` grids strictly increasing
  - tables bounded by sentinel
  - hit-time monotonicity by level.

## Publish interaction

R3-01 / R3-02 / R3-03 сами по себе не делают slot publish-ready.
R3-04 делает publish-ready только stage `prices + mappings`, если validation scope выбран явно и
config-driven:

- `price_timeframes` и `mapping_timeframes` берутся из `backtest_artifacts.validation_plan`;
- `signal_artifacts = ()`;
- `require_hit_times_manifest = false`.

После R4-02 full validation spec уже может требовать real `signals` и успешно проходить, если
root catalog и per-indicator manifests materialized для explicit configured targets.
После R5-01 full validation spec может также требовать real `hit_times/1m`, если slot построен
через актуальный precompute runner path. Отдельный R3-04 prices+mappings stage helper по-прежнему
должен оставаться explicit и выставлять `require_hit_times_manifest = false`.

Runner обязан работать только в порядке:

1. resolve `current.yaml`;
2. precheck inactive slot pin guard;
3. rebuild inactive slot;
4. validate whole slot по strict manifests и explicit validation spec, полученному из
   `backtest_artifacts.validation_plan`;
5. atomically switch `current.yaml`.

Для R3-04 рекомендуется отдельный config-driven derivation:

- взять `price_timeframes` из `validation_plan`;
- взять `mapping_timeframes` из `validation_plan`;
- принудительно выставить `signal_artifacts = ()`;
- принудительно выставить `require_hit_times_manifest = false`.

Если validation вернула хотя бы одну error diagnostic:

- publish завершается без pointer switch;
- `current.yaml` остаётся прежним;
- оператор получает stable `code/message/diagnostics`.

## Runtime contract after publish

После успешного publish runtime может:

- читать root manifest один раз;
- использовать fixed `dtype/shape/axis_order`;
- выбирать ровно нужные signal manifests и `signals.i8.npy`;
- читать `hit_times/1m` без recompute metadata.

После R6-01 runtime additionally обязан:

- собирать `slot-pinned context` только из strict `current.yaml` или persisted pin metadata;
- выравнивать identity fields `artifact_slot`, `slot_generation`, `artifact_asof_date`,
  `artifact_manifest_hash` между sync и background startup;
- читать `prices/<tf>`, `signals/<tf>/<indicator_id>/signals.i8.npy`,
  `mappings/<tf>/bar_open_1m_idx.u32.npy`,
  `mappings/<tf>/bar_close_1m_idx.u32.npy` и `hit_times/1m/manifest.yaml`
  только по explicit paths, уже перечисленным в manifests;
- открывать arrays через `np.load(..., mmap_mode='r')` и `allow_pickle=False`;
- reject'ить contract drift по `path`, `dtype`, `shape`, `axis_order`, `timeline`.

Runtime не должен:

- повторно считать `sha256`;
- вычислять `shape` или `axis_order` по соглашениям из имени файла;
- сканировать slot для discovery.
