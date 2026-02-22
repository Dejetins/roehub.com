# Backtest Jobs v1 -- Storage (Postgres) + Repositories + State Machine (BKT-EPIC-09)

Документ фиксирует архитектурный контракт EPIC-09: схему хранения backtest jobs в Postgres, state machine, репозиторные порты и правила детерминизма для последующих EPIC-10/11.

## Цель

- Добавить устойчивое Postgres-хранилище для async backtest jobs.
- Зафиксировать минимальный доменный контракт job lifecycle (`state/stage/progress/lease/cancel`).
- Зафиксировать детерминированные правила snapshots/hashes/ordering, чтобы worker и API работали независимо и предсказуемо.

## Контекст

- Milestone 4 реализовал sync `POST /backtests` без history/jobs storage.
- Milestone 5 требует multi-user параллельные запуски, прогресс, cancel и best-so-far результаты.
- В репозитории уже есть паттерны:
  - явный SQL через minimal gateway (strategy/identity),
  - Alembic миграции,
  - deterministic sorting и unified `RoehubError`.

Фиксы, уже утвержденные для Milestone 5 и обязательные для EPIC-09:

- Jobs только внутри bounded context `backtest`.
- Runtime DSN (API + worker): `STRATEGY_PG_DSN`.
- Migration DSN: `POSTGRES_DSN`.
- Saved-mode job обязан хранить `spec_hash` и `spec_payload` snapshot.
- `engine_params_hash` считается при создании job из effective direction/sizing/execution.
- `backtest_runtime_config_hash` включает только result-affecting секции и должен включать `jobs.top_k_persisted_default`.

## Scope

### 1) Postgres schema v1

Вводим 3 таблицы (Alembic):

1. `backtest_jobs`
2. `backtest_job_top_variants`
3. `backtest_job_stage_a_shortlist`

#### 1.1 `backtest_jobs`

Минимальные поля:

- identity:
  - `job_id uuid primary key`
  - `user_id uuid not null`
- mode/lifecycle:
  - `mode text not null` (`saved|template`)
  - `state text not null` (`queued|running|succeeded|failed|cancelled`)
  - `created_at`, `updated_at` (`timestamptz not null`)
  - `started_at`, `finished_at` (`timestamptz null`)
  - `cancel_requested_at` (`timestamptz null`)
- request/reproducibility:
  - `request_json jsonb not null` (canonical effective payload)
  - `request_hash text not null` (sha256 hex)
  - `spec_hash text null` (sha256 hex; обязателен для `mode=saved`)
  - `spec_payload_json jsonb null` (обязателен для `mode=saved`)
  - `engine_params_hash text not null` (sha256 hex)
  - `backtest_runtime_config_hash text not null` (sha256 hex)
- progress:
  - `stage text not null` (`stage_a|stage_b|finalizing`)
  - `processed_units int not null default 0`
  - `total_units int not null default 0`
  - `progress_updated_at timestamptz null`
- lease/concurrency:
  - `locked_by text null` (`<hostname>-<pid>`)
  - `locked_at timestamptz null`
  - `lease_expires_at timestamptz null`
  - `heartbeat_at timestamptz null`
  - `attempt int not null default 0`
- failure payload:
  - `last_error text null`
  - `last_error_json jsonb null` (RoehubError-like object: `code/message/details`)

Минимальные constraints/indexes:

- checks:
  - `mode` in (`saved`,`template`)
  - `state` in (`queued`,`running`,`succeeded`,`failed`,`cancelled`)
  - `stage` in (`stage_a`,`stage_b`,`finalizing`)
  - `attempt >= 0`, `processed_units >= 0`, `total_units >= 0`
  - `request_json/spec_payload_json/last_error_json` имеют JSON object shape (когда не null)
  - saved/template consistency:
    - `mode='saved' -> spec_hash is not null and spec_payload_json is not null`
    - `mode='template' -> spec_hash is null and spec_payload_json is null`
- indexes:
  - list: `(user_id, state, created_at desc, job_id desc)`
  - queue claim FIFO: `(state, created_at asc, job_id asc)`
  - reclaim: `(state, lease_expires_at asc, created_at asc, job_id asc)`
  - active quota helper (partial): `(user_id)` where `state in ('queued','running')`

#### 1.2 `backtest_job_top_variants`

Минимальные поля:

- `job_id uuid not null` (FK -> `backtest_jobs.job_id`)
- `rank int not null` (1..K)
- `variant_key text not null`
- `indicator_variant_key text not null`
- `variant_index int not null`
- `total_return_pct double precision not null`
- `payload_json jsonb not null`
- `report_table_md text null`
- `trades_json jsonb null`
- `updated_at timestamptz not null`

Минимальные constraints/indexes:

- primary/unique:
  - `primary key (job_id, rank)`
  - `unique (job_id, variant_key)`
- checks:
  - `rank > 0`, `variant_index >= 0`
  - `payload_json` object shape
  - `trades_json` is null or json array
- index:
  - `idx_backtest_job_top_variants_job_rank (job_id, rank)`

Примечание по `report_table_md`:

- Для `succeeded` job поле заполняется на этапе `finalizing`.
- Для `running`/`failed`/`cancelled` допускается только `NULL`.

#### 1.3 `backtest_job_stage_a_shortlist`

Минимальные поля:

- `job_id uuid primary key` (FK -> `backtest_jobs.job_id`)
- `stage_a_indexes_json jsonb not null` (array[int], deterministic order)
- `stage_a_variants_total int not null`
- `risk_total int not null`
- `preselect_used int not null`
- `updated_at timestamptz not null`

Минимальные constraints:

- `stage_a_variants_total > 0`, `risk_total > 0`, `preselect_used > 0`
- `stage_a_indexes_json` JSON array shape

### 2) Domain/app contracts (jobs)

Добавляем backtest job-модель в `contexts/backtest`:

- state machine:
  - `queued -> running|cancelled`
  - `running -> succeeded|failed|cancelled`
  - terminal states финальны
  - `queued -> failed` запрещен
- stage literals:
  - `stage_a`, `stage_b`, `finalizing`
- error payload:
  - `last_error`: короткая строка
  - `last_error_json`: `{code, message, details}`

### 3) Application ports

В application layer вводим 3 порта:

1. `BacktestJobRepository`
   - create/get/list/cancel
   - count active (`queued + running`) для quota check

2. `BacktestJobLeaseRepository`
   - claim next job (queued first, plus reclaim expired running)
   - heartbeat/extend lease
   - finish (`succeeded|failed|cancelled`) только условно под активным lease

3. `BacktestJobResultsRepository`
   - replace best-so-far top-K snapshot
   - persist/read Stage A shortlist payload

### 4) Postgres adapters and deterministic SQL

Обязательные SQL правила:

- Без ORM, только явный SQL.
- Claim query использует `FOR UPDATE SKIP LOCKED`.
- Claim order FIFO:
  - `ORDER BY created_at ASC, job_id ASC`.
- List query uses keyset pagination:
  - `ORDER BY created_at DESC, job_id DESC`
  - cursor payload: `{created_at, job_id}`.
- Все worker-side UPDATE/INSERT при `running` выполняются условно по lease owner:
  - `job_id = :job_id`
  - `locked_by = :locked_by`
  - `lease_expires_at > :now`.

Для `backtest_job_top_variants` фиксируем snapshot write policy v1:

- replace whole snapshot в транзакции:
  1) `DELETE FROM backtest_job_top_variants WHERE job_id=:job_id`
  2) bulk `INSERT` нового top-K

### 5) Runtime config (`backtest.jobs.*`)

Расширяем `configs/<env>/backtest.yaml`:

- `backtest.jobs.enabled`
- `backtest.jobs.top_k_persisted_default`
- `backtest.jobs.max_active_jobs_per_user`
- `backtest.jobs.claim_poll_seconds`
- `backtest.jobs.lease_seconds`
- `backtest.jobs.heartbeat_seconds`
- `backtest.jobs.snapshot_seconds` (optional with strict positive validation when present)
- `backtest.jobs.snapshot_variants_step` (optional with strict positive validation when present)
- `backtest.jobs.parallel_workers` (default 1)

Validation policy:

- jobs section и обязательные ключи валидируются fail-fast (strict required).
- fallback defaults для обязательных jobs keys не допускаются.

### 6) `backtest_runtime_config_hash` (result-affecting only)

Добавляем deterministic hash из canonical JSON с включением только полей, влияющих на результат:

- `backtest.warmup_bars_default`
- `backtest.top_k_default`
- `backtest.preselect_default`
- `backtest.execution.*`
- `backtest.reporting.*`
- `backtest.jobs.top_k_persisted_default`

Из hash намеренно исключаем operational knobs:

- `enabled`, `max_active_jobs_per_user`, `claim_poll_seconds`, `lease_seconds`, `heartbeat_seconds`, `snapshot_seconds`, `snapshot_variants_step`, `parallel_workers`.

## Non-goals

- Retention/cleanup старых jobs и результатов.
- Generic jobs framework для других bounded contexts.
- Точный Stage B cursor-checkpoint (v1 допускает restart attempt).

## Ключевые решения

### 1) `request_json` хранит effective payload, а не сырой transport body

Храним payload после merge и нормализации (включая effective defaults), чтобы job был воспроизводим независимо от будущих изменений API mapping/runtime defaults.

Пример:
- если позже изменится default `safe_profit_percent`, старый job всё равно считается по зафиксированному payload.

### 2) Saved-mode snapshot обязателен

Для `mode=saved` job creation обязана сохранить `spec_hash` и `spec_payload_json`.

Если snapshot не удалось получить:
- job не создаем,
- возвращаем deterministic 422/500 по контракту ошибки.

### 3) Claim semantics: FIFO + reclaim expired running

Claim берет сначала старейшие `queued` jobs (FIFO). Дополнительно разрешен reclaim `running` jobs с истекшим lease.

Это дает:
- предсказуемую очередь,
- отсутствие вечных stuck jobs после падения воркера.

### 4) Lease-owner conditional writes -- обязательный инвариант

Все worker writes при `running` допускаются только для текущего lease owner.

Если условный update вернул 0 строк:
- lease потерян,
- worker обязан остановить обработку job.

### 5) Error payload в БД совместим с RoehubError

`last_error_json` хранится в форме `{code,message,details}`.

Traceback в БД не храним; он остается только в логах.

### 6) Stage A shortlist фиксируем компактно (`stage_a_indexes_json`)

Для minimal resume достаточно deterministic списка Stage A индексов и метаданных (`stage_a_variants_total`, `risk_total`, `preselect_used`).

Это уменьшает объем хранения и не ломает воспроизводимость.

## Контракты и инварианты

- Один и тот же effective payload + одинаковый runtime result-affecting hash -> один и тот же `request_hash`.
- `state` и `stage` принимают только фиксированные literals.
- `queued -> failed` запрещен.
- Active jobs per user считаются как `queued + running`.
- Keyset list order фиксирован: `created_at DESC, job_id DESC`.
- Claim order фиксирован: `created_at ASC, job_id ASC`.
- `backtest_job_top_variants` для `cancelled/failed` может содержать best-so-far ranking, но `report_table_md` остается `NULL`.
- `report_table_md`/`trades_json` финализируются только для `succeeded`.

## Связанные файлы

Roadmap/docs:

- `docs/architecture/roadmap/base_milestone_plan.md` -- Milestone 5 requirements.
- `docs/architecture/roadmap/milestone-5-epics-v1.md` -- EPIC-09 scope.
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md` -- hashes and deterministic API contract.
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` -- Stage A/Stage B semantics.
- `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md` -- reporting policy and `top_trades_n`.

Runtime/config:

- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/dev/backtest.yaml`
- `configs/test/backtest.yaml`
- `configs/prod/backtest.yaml`

Persistence/migrations patterns:

- `alembic/versions/20260215_0001_strategy_storage_v1.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/gateway.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_run_repository.py`

Target implementation paths (EPIC-09):

- `alembic/versions/*_backtest_jobs_v1.py`
- `src/trading/contexts/backtest/domain/*`
- `src/trading/contexts/backtest/application/ports/*`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/*`

## Как проверить

После реализации EPIC-09:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

Дополнительно (целевые тесты):

```bash
uv run pytest -q tests/unit/contexts/backtest
uv run pytest -q tests/unit/apps/api
```

## Риски и открытые вопросы

- Риск: write amplification при частом replace snapshot top-K. Митигатор: `snapshot_seconds`/`snapshot_variants_step` в worker (EPIC-10).
- Риск: рост объема таблиц jobs/results без retention. В Milestone 5 это осознанный non-goal.
- Риск: strict required jobs config потребует синхронного обновления всех `configs/<env>/backtest.yaml`.
