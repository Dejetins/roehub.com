# Бектест: текущая реализация и контракты

Статус: active; сверено с репозиторием 2026-09-06.

Artifact-backed runtime реализован: API создаёт persisted jobs, отдельный worker
исполняет расчёт и lazy trades materialization, Web UI показывает результаты.
Прежняя запись о runtime-compute reset больше не описывает текущий код.

## Текущие источники

- [Контракт artifact runtime v1](backtest-service-artifact-runtime-v1.ru.md).
- [API routes](../../../apps/api/routes/backtests.py).
- [Backtest application services](../../../src/trading/contexts/backtest/application/services/).
- [Artifact publisher/precompute](../../../src/trading/contexts/backtest_artifacts/).
- [Job worker](../../../apps/worker/backtest_job_runner/).
- [Installation service manifest](../../../configs/installation/runtime-service-manifest.json).
- [Self-hosted эксплуатация](../../runbooks/offline-release-installation.md).

Наличие реализации и тестов не подтверждает развёртывание. Репозиторий не выбирает
действующий production host; запуск и runtime proof требуют конкретной установки.

## Исторические материалы

- [План ввода job runner](backtest-job-runner-production-plan-v1.md).
- [Начальный prompt pack Iteration 0/1](backtest-service-implementation-prompt-pack-iteration-0-1.md).
- [Benchmark records](benchmark_iterations/README.md).
- [Результаты неудачных ускорений](backtest-compute-acceleration-negative-results-v1.md).

Планы, ledgers и прежние требования к Mac Studio сохраняют историю решений и
измерений. Они не выбирают текущую задачу или deployment target. Новый benchmark
фиксирует revision, окружение, fixture, hashes и сопоставимый baseline; старые
числа нельзя переносить на другое оборудование без повторного измерения.

## Benchmark accounting and full top-N evidence (2026-09-18)

This section defines corrected API-runner evidence, schema
`backtest_api_runner_compute_memory_parity_v2`. Historical records are immutable;
they have not been remeasured. Historical notebook stage sums, iteration 6/7
service totals and pre-v2 orchestration totals are not comparable elapsed totals.
A change from the old overlapping sum to the corrected interval is not a speedup.

### Timing boundaries

All `stage_timings` values are **seconds**; instrumentation fields ending in `_ms`
are milliseconds. `exact_diagnostics.timing_accounting.schema` is
`orchestration_elapsed_v2` for corrected orchestration evidence.

| Field | Boundary and interpretation |
| --- | --- |
| `service_wall_clock_s` | Monotonic interval from `execute` after request/risk extraction to the return of top-result assembly. Includes artifact preparation, signatures/bitsets and their GC, hit-time loading when applicable, sample warmup, combo planning, exact execution and assembly. |
| `sample_warmup` | Nested monotonic interval inside that same call: limited-row preparation, warmup planning/scoring and warmup's own GC. `warmup=not_run` means no warmup interval was executed; missing evidence for a claimed measured warmup is unavailable, not zero. |
| `service_total_without_warmup` | Exactly `service_wall_clock_s - sample_warmup` when warmup ran, otherwise the measured wall interval. Never a stage sum. |
| `prepare_pools_core`, `prepare_pools_total` when supplied | Parent diagnostics; preparation subsegments overlap these totals. Pool-result construction and array copies may be outside a narrower preparation timer while still inside orchestration wall time. |
| `exact_scoring`, `tp_sl_exact_scoring` | TP/SL aliases for the same scoring interval; never add both. Internal exact subsegments may overlap scoring. |
| `top_result_assembly` | Assembly's narrower measured loop; the outer orchestration interval also includes summary-hash construction after that loop. Missing stage evidence stays absent. |
| `persist_top_n_io` | Not measured by orchestration and no longer emitted as a fabricated zero. A separate persistence harness may supply its own IO interval; it is outside orchestration elapsed. |

The outer interval **excludes** subsequent exact-diagnostic serialization,
instrumentation result construction, the final `finally` deletion/GC, child IPC
serialization and transport, Python startup/imports, worker startup/queue/claim,
parent persistence and HTTP. Earlier diagnostics/GC executed before assembly are
included. Execution, cleanup order and funding behavior are unchanged. Child
process envelope, runner wall time and persistence must retain their own labels;
none is an alias of this pre-cleanup interval.

No additive decomposition or residual is inferred from overlapping stages.
`unattributed_seconds=null` explicitly means not assessed. Negative, nonfinite or
out-of-bound warmup intervals are rejected; report validation distinguishes
missing/legacy evidence (`not_assessed`) from inconsistent intervals (`failed`).
The API-runner performance gate requires corrected accounting. Its historical
speed ratios still use exact-scoring stages, never the corrected service total.
No realistic performance baseline is established by accounting regression tests.

`validate_benchmark_accounting.py` validates the historical notebook `runs` /
`request` schema and stage vocabulary. It does **not** validate API-runner JSON,
full-N parity, monotonic intervals or production performance. Its historical
stage-addition rules and records are preserved, not reinterpreted as v2 elapsed.

### Complete reference and transport checks

Current `GET /backtests/jobs/{job_id}/top` is **unpaginated** and returns every
persisted top row (`BacktestTopVariantsResponse.items`); `/jobs` list pagination
is a different contract. The benchmark reads the complete `/top` response.
An unexpected cursor, `has_more` or pagination envelope is unsupported/incomplete
and cannot pass. A future paginated `/top` contract requires a corresponding
reader change before acceptance; the current reader never treats a first page
as complete.

Each reference run may provide `full_top_reference` with:

- `schema: backtest_full_top_reference_v1`, `complete: true`, `requested_top_n`
  (1–50 for this bounded harness), `expected_count` and all `items`.
- `provenance.method`: `independent_expectations` for independently prescribed
  small fixtures, or `trusted_baseline` for a separately identified trusted run;
  nonempty `source` and `revision`. Candidate output must never generate its oracle.
- `context`: exact matching nonempty `request_hash`, `engine_params_hash`,
  `backtest_runtime_config_hash`, `artifact_manifest_hash`. These bind normalized
  request, quality filters, ranking/direction/tie policy, numerical engine policy,
  runtime configuration and artifact identity. This strict full-reference format
  does not infer historical-prefix compatibility from a different manifest hash.
- `risk_mode`, boolean `funding_included` and a complete `metric_names` manifest.
  The manifest must match the current `NO_RISK_METRIC_NAMES` or
  `TP_SL_EXACT_METRIC_NAMES`, plus funding metrics when included. Every reference
  row must have exactly those keys, and the child telemetry metric profile must
  agree. A truncated oracle is `not_assessed`, even if API loses the same metrics;
  a missing API metric against a complete oracle is a demonstrated mismatch.
- Each item includes `rank`, stable `variant_hash`, `canonical_variant_params`,
  complete `summary_metrics`, `best_tp_pct`, and `best_sl_pct`.

Comparison follows the reference's exact canonical order and checks cardinality,
duplicate identities, every rank, both stable hash and full canonical parameters,
all metric keys/values and selected TP/SL percentages. Job-scoped readable
`variant_key`, links and job IDs are excluded; stable semantic identity is retained.
Fewer rows than requested, including zero, are valid only with a complete matching
reference. An empty-quality flag alone is insufficient.

Metric tolerance is the existing absolute `1e-5`, with no relative tolerance;
identity parameters and rank are exact. Missing keys mismatch. The existing
persisted nonfinite policy maps NaN and infinities to JSON null; null equals null,
never zero. Numeric strings do not equal numbers. This evidence cannot distinguish
nonfinite values already normalized to null at persistence.

The harness opts into `ROEHUB_BACKTEST_BENCHMARK_FULL_TOP=1` **only in its runner
environment**. The parent evidence writer exports at most 50 compact rows from
child IPC, with actual count and completeness; larger outputs remain incomplete.
Normal production diagnostics retain their five-row sample and the public API,
persisted row schemas and execution defaults remain unchanged.

`full_reference` proves parity against the identified independent oracle;
`transport_consistency` checks assembly/child versus API and is **not** independent
financial correctness. Partial historical top-five comparisons remain visible.
Quality filters and shadow modes cannot skip the required full reference or
turn its absence into acceptance. The report records requested, available (when
known), expected, retrieved and compared counts, completeness, reasons and status.

`passed` means assessed agreement; `failed` means a demonstrated mismatch;
`not_assessed` means a missing, truncated or incompatible prerequisite. Both
non-passing states prevent aggregate acceptance, and no required jobs is also
`not_assessed`. JSON and Markdown expose these states. Default command exit is 1
unless all required gates pass; `--no-fail-on-threshold` is explicitly
`diagnostic_only`, may exit 0, and never changes a failed/unassessed acceptance flag.

Compatibility: public API, result identity/ranking, funding, persistence and
execution defaults are `none`; additive bounded child evidence is
`compatible-change`. Benchmark acceptance/evidence is an intentional
`breaking-change` within the F03/F04 correction: legacy partial references and
invalid totals cannot satisfy v2 acceptance. Consumers must inspect schema/status
and supply a complete compatible oracle; old records remain historical evidence.
