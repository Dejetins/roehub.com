# Backtest Jobs v1 -- Job-Runner Worker (claim/lease + streaming batches + cancel) (BKT-EPIC-10)

Документ фиксирует архитектурный контракт EPIC-10: новый воркер `backtest-job-runner`, который забирает queued jobs из Postgres, выполняет backtest батчами, пишет progress и best-so-far top-K snapshots, корректно обрабатывает cancel и lease/reclaim.

## Цель

- Добавить отдельный worker-процесс для async backtest jobs (без блокировки API/UI).
- Зафиксировать execution flow по стадиям (`stage_a -> stage_b -> finalizing`) с детерминированными правилами.
- Зафиксировать правила lease/heartbeat/cancel/reclaim, чтобы job не застревали в `running`.

## Контекст

- EPIC-09 уже добавил storage/domain/ports для jobs:
  - `backtest_jobs`
  - `backtest_job_top_variants`
  - `backtest_job_stage_a_shortlist`
- Sync backtest Milestone 4 (`RunBacktestUseCase`) сейчас ориентирован на in-memory staged flow и small runs.
- Для больших jobs нужен отдельный orchestration слой, который:
  - не materialize-ит сразу весь Stage A/Stage B,
  - работает батчами,
  - периодически сохраняет persisted best-so-far snapshot.

Утвержденные решения для EPIC-10 (зафиксировано):

- Архитектура worker: **отдельный streaming orchestrator** (не расширение `BacktestStagedRunnerV1` как главного движка jobs).
- Reclaim/resume v1: **restart attempt с начала** (даже если есть `stage_a_shortlist`).
- Snapshot cadence: писать persisted top-K при выполнении **любого** из условий:
  - прошло `snapshot_seconds`,
  - обработано `snapshot_variants_step` новых Stage-B вариантов.
- Finalizing: детали (`report_table_md`/trades) считаются только для persisted cap:
  - `persisted_k = min(request.top_k, backtest.jobs.top_k_persisted_default)`.
- Parallelism v1: один job исполняется последовательно (intra-job parallelism = 1).
  `backtest.jobs.parallel_workers` существует в runtime config, но в v1 воркер его не использует (зарезервирован под будущее).

## Update 2026-02-25 (Perf Phase 3)

С 2026-02-25 worker использует тот же scoring core, что и sync path:

- Stage A/Stage B исполняются через `BacktestStagedCoreRunnerV1`.
- Ranking/tie-break semantics между sync и jobs унифицированы в одном коде:
  - Stage A: `total_return_pct DESC`, `base_variant_key ASC`;
  - Stage B: `total_return_pct DESC`, `variant_key ASC`.
- Перед стадиями scorer подготавливает batched indicator tensors (`prepare_for_grid_context(...)`), что убирает per-variant compute из hot path.
- CPU лимит из runtime config (`backtest.cpu.max_numba_threads`) применяется в worker attempt через `numba.set_num_threads(...)`.

## Scope

### 1) Новый worker процесс и wiring

Добавляем новый entrypoint:

- `apps/worker/backtest_job_runner/main/main.py`

И отдельный wiring модуль:

- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`

Wiring обязан собрать:

- runtime config (`backtest.jobs.*` + result-affecting defaults),
- `IndicatorCompute`,
- candle feed/reader для canonical candles,
- репозитории EPIC-09:
  - `BacktestJobRepository`,
  - `BacktestJobLeaseRepository`,
  - `BacktestJobResultsRepository`.

Toggle semantics:

- если `backtest.jobs.enabled=false`, worker пишет структурированный log (`component=backtest-job-runner`, `status=disabled`) и завершает процесс с `exit 0`.

### 2) Claim loop (polling + lease)

Worker loop:

1. Sleep/poll по `claim_poll_seconds`.
2. `claim_next(now, locked_by, lease_seconds)`.
3. Если job отсутствует -- продолжить polling.
4. Если job claimed -- запустить `process_job(...)`.

Инварианты:

- Claim atomic через SQL `FOR UPDATE SKIP LOCKED`.
- Один job одновременно исполняет только один worker owner (`locked_by`).
- `locked_by` формат: `<hostname>-<pid>` (можно добавить suffix instance id).

### 3) Execution flow per job

Source-of-truth payload:

- worker читает только `job.request_json` (+ `spec_payload_json` для saved, если нужно для построения effective template),
- worker не зависит от текущего состояния saved strategy в strategy storage.

#### 3.1 Stage A (streaming shortlist)

Цель Stage A:

- посчитать базовые варианты (без SL/TP),
- получить deterministic shortlist top `preselect`.

Контракт:

- Ranking key Stage A:
  - `total_return_pct DESC`,
  - tie-break `base_variant_key ASC`.
- Progress:
  - `stage = stage_a`,
  - `total_units = stage_a_variants_total`,
  - `processed_units` увеличивается монотонно.
- Persist shortlist:
  - сохранить `BacktestJobStageAShortlist` через `save_stage_a_shortlist(...)`.

Примечание v1:

- На reclaim attempt Stage A пересчитывается с начала; сохраненный shortlist нужен для observability/диагностики и будущего усиления resume.

#### 3.2 Stage B (streaming expanded scoring + running top-K)

Цель Stage B:

- расширить shortlisted base-варианты по risk axes,
- поддерживать running top-K без полного in-memory списка всех expanded variants.

Контракт:

- `persisted_k = min(request.top_k, top_k_persisted_default)`.
- Ranking key Stage B:
  - `total_return_pct DESC`,
  - tie-break `variant_key ASC`.
- Worker держит только bounded структуру top-K (heap/эквивалент) + минимально нужный payload для финализации.
- Progress:
  - `stage = stage_b`,
  - `total_units = stage_b_variants_total`,
  - `processed_units` монотонно увеличивается.
- Snapshot writes во время running:
  - через `replace_top_variants_snapshot(...)`,
  - trigger: elapsed `snapshot_seconds` **или** processed delta >= `snapshot_variants_step`.

Snapshot payload policy во время `running`:

- `report_table_md = NULL`,
- `trades_json = NULL`,
- сохраняются ranking fields + `payload_json` для top rows.

#### 3.3 Finalizing (только succeeded)

Finalizing запускается только если job не cancelled и не failed.

Шаги:

1. Зафиксировать progress `stage=finalizing, processed_units=0, total_units=1`.
2. Сохранить terminal snapshot без eager report bodies:
   `report_table_md=NULL`, `trades_json=NULL`.
3. Перезаписать snapshot в `backtest_job_top_variants` в summary-only формате.
4. `finish(..., next_state="succeeded")`.

Полный report/trades для выбранного варианта загружается отдельно через
`POST /api/backtests/variant-report` (lazy policy).

### 4) Cancel semantics

Cancel best-effort:

- Worker проверяет `cancel_requested_at` на границах батчей (до next batch compute и перед finalizing).
- Если cancel обнаружен:
  - worker завершает job через `finish(..., next_state="cancelled")`,
  - `report_table_md` не вычисляется и не сохраняется,
  - уже сохраненный ranking snapshot допускается (best-so-far without report bodies).

### 5) Failure semantics

При ошибке выполнения:

- worker формирует короткий `last_error`,
- формирует `last_error_json` в RoehubError-like shape `{code,message,details}`,
- переводит job в `failed` через `finish(...)`.

Требование:

- traceback хранится только в логах воркера, не в БД/API payload.

### 6) Lease lost behavior (fail-fast)

Любая lease-guarded операция может вернуть `None`/`False` (owner mismatch или lease expired).

Если это произошло:

- worker **немедленно** прекращает обработку текущего job,
- никаких следующих writes по этому job не делает,
- пишет structured log `event=lease_lost`.

Это обязательный split-brain guard.

### 7) Reclaim semantics (minimal resume v1)

- Если worker упал и lease истек, другой worker может reclaim job через `claim_next(...)`.
- v1 политика: restart attempt с начала flow (Stage A -> Stage B -> Finalizing).
- Цель: простая, надежная консистентность без сложного cursor resume в Stage B.

Наблюдаемое поведение (важно для UI/операций):

- При reclaim attempt `stage/progress` могут сброситься к `stage_a` и `processed_units=0`.
- Persisted `/top` snapshot от предыдущей попытки может оставаться в БД и быть видимым до первой перезаписи snapshot в новой попытке.

## Детерминизм и инварианты

- Один и тот же `request_json` + `engine_params_hash` + `backtest_runtime_config_hash` -> один и тот же итоговый ordering persisted top rows.
- Stage A tie-break фиксирован: `base_variant_key ASC`.
- Stage B tie-break фиксирован: `variant_key ASC`.
- Snapshot replace policy фиксирована: full replace (delete+insert contract) через results repository.
- Jobs `/top` остаётся summary-only для всех состояний, включая `succeeded`.
- Все timestamps и hash literals остаются UTC/sha256 contracts из EPIC-09.

## Observability

Минимальные метрики worker:

- counters:
  - `backtest_job_runner_claim_total`
  - `backtest_job_runner_succeeded_total`
  - `backtest_job_runner_failed_total`
  - `backtest_job_runner_cancelled_total`
  - `backtest_job_runner_lease_lost_total`
- histograms:
  - job duration (from claim to terminal)
  - stage duration (`stage_a`, `stage_b`, `finalizing`)
- gauges:
  - active claimed jobs per process (обычно 0/1 для v1 single-job loop)

Логирование:

- structured logs с обязательными полями: `job_id`, `attempt`, `locked_by`, `stage`, `event`.

## Non-goals

- Распараллеливание одного job между несколькими worker instances.
- Cursor-level resume Stage B (checkpoint внутри expanded stream).
- Retention/cleanup политики таблиц jobs/results.
- Realtime push (SSE/WebSocket) прогресса из worker.

## DoD

- Воркep `backtest-job-runner` запускается и корректно работает при `enabled=true`.
- При `enabled=false` процесс завершает `exit 0` и не делает claim.
- 2-4 реплики воркера обрабатывают jobs параллельно без двойного исполнения одного job.
- Cancel переводит job в `cancelled` (не в `failed`) и останавливает дальнейший compute.
- При падении воркера job reclaim'ится и завершается terminal state.
- Для `succeeded` persisted rows остаются summary-only; отчёты грузятся on-demand через `variant-report`.

## Связанные файлы

Roadmap/docs:

- `docs/architecture/roadmap/base_milestone_plan.md`
- `docs/architecture/roadmap/milestone-5-epics-v1.md`
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
- `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`

Current implementation references:

- `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
- `src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py`
- `src/trading/contexts/backtest/application/ports/backtest_job_repositories.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_lease_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`

Target implementation paths (EPIC-10):

- `apps/worker/backtest_job_runner/main/main.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `src/trading/contexts/backtest/application/use_cases/*` (job-runner orchestrator/use-case)
- `src/trading/contexts/backtest/application/services/*` (streaming staged runner pieces)

## Как проверить

После реализации EPIC-10:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

Целевые тесты (минимум):

```bash
uv run pytest -q tests/unit/contexts/backtest
uv run pytest -q tests/unit/apps/worker
```

## Риски

- Write amplification: частые snapshot replace top-K могут нагружать Postgres; контролируется `snapshot_seconds`/`snapshot_variants_step`.
- CPU overhead reclaim: restart-from-beginning увеличивает стоимость recovery для больших jobs.
- Drift risk: если future API/DTO изменит shape `request_json`, нужен строгий backward-compatible reader в worker orchestrator.
