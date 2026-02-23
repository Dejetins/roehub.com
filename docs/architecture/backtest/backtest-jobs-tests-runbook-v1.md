# Backtest Jobs v1 -- Tests + Runbook (BKT-EPIC-12)

Документ фиксирует контракт EPIC-12: набор unit/smoke тестов для Backtest Jobs v1 (детерминизм ordering/lease/cancel/cursor/errors) и минимальный runbook для эксплуатации `backtest-job-runner`.

## Цель

- Зафиксировать “неплавающий” контракт Milestone 5 на уровне тестов: claim/lease/reclaim, cancel, deterministic ordering, cursor transport, state-dependent output.
- Дать минимальный runbook, чтобы воркер можно было безопасно запускать/масштабировать и диагностировать stuck/failed jobs.

## Контекст

- Milestone 5 добавляет async backtest путь через jobs storage + worker + API:
  - storage/state machine: `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
  - worker: `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
  - API: `docs/architecture/backtest/backtest-jobs-api-v1.md`
- В v1:
  - lease/reclaim обязательны (job не должен зависать в `running` навсегда),
  - cancel best-effort (`queued -> cancelled` сразу; `running` — на границах батчей),
  - `/top` во время `running` — best-so-far snapshot без `report_table_md`/trades,
  - reclaimed job делает restart attempt (возможны сброс `stage/progress`, а persisted `/top` может оставаться видимым до первой перезаписи snapshot).

## Scope

### 1) Unit tests (контрактные)

- Domain/state machine invariants:
  - saved-mode требует `spec_hash` + `spec_payload_json`.
  - запрещен переход `queued -> failed`.
  - claim/lease поля валидны только для `running`.
  - cancel idempotent и соблюдает `queued` vs `running` семантику.

- Postgres repositories (SQL shape + deterministic clauses):
  - claim использует `FOR UPDATE SKIP LOCKED` и FIFO ordering.
  - list использует keyset pagination `ORDER BY created_at DESC, job_id DESC` и tuple predicate.
  - lease-guarded writes используют `(job_id, locked_by, lease_expires_at > now)`.
  - snapshot replace делает delete+insert в одном statement/CTE.
  - Stage-A shortlist хранится через `ON CONFLICT (job_id)` upsert под lease guard.

- Runtime config:
  - `backtest.jobs.*` секция strict-required и валидируется fail-fast.
  - `backtest_runtime_config_hash` включает `backtest.jobs.top_k_persisted_default` и игнорирует operational knobs.

- Worker (use-case уровень):
  - stage progress semantics (`stage_a`, `stage_b`, `finalizing`).
  - snapshot cadence OR (`snapshot_seconds` или `snapshot_variants_step`).
  - cancel detection на границах батчей останавливает compute и переводит job в `cancelled`.
  - lease-lost прекращает дальнейшие writes и не делает terminal finish.
  - succeeded finalizing записывает `report_table_md` для всех persisted rows и trades только для `top_trades_n`.

- API (routes/DTO/wiring):
  - jobs endpoints монтируются только при `backtest.jobs.enabled=true`.
  - owner-only policy: `403` для чужой существующей job, `404` для отсутствующей.
  - cursor transport `base64url(json)` с deterministic canonical JSON.
  - `/top` соблюдает state-dependent policy по `report_table_md`/trades.

### 2) Smoke-level tests (без внешних сервисов)

- Cursor round-trip и edge-cases (`encode/decode`).
- Детерминированность форматирования list response (`items` + `next_cursor`).

### 3) Runbook: `backtest-job-runner`

Минимальный runbook фиксирует:

- как запустить воркер (локально и/или в docker compose),
- какие env/конфиги обязательны (`STRATEGY_PG_DSN`, ClickHouse settings, `ROEHUB_ENV/ROEHUB_BACKTEST_CONFIG`),
- toggle semantics (`backtest.jobs.enabled=false` -> worker `exit 0`),
- метрики (`/metrics`) и базовые сигналы здоровья,
- диагностику stuck jobs:
  - `running` с истекшим lease -> reclaim после TTL,
  - рост `attempt`, сброс `stage/progress` на restart attempt,
  - `failed` payload (`last_error`, `last_error_json`) без traceback в БД,
- как безопасно проверять cancel и lease-lost случаи.

## Non-goals

- Тяжелые интеграционные тесты с реальным Postgres/ClickHouse.
- E2E UI сценарии.
- Оптимизация производительности job-runner (это отдельный milestone/epic).

## Ключевые решения

### 1) Контрактные тесты вместо интеграционных

Мы фиксируем форму SQL (SKIP LOCKED, ORDER BY, lease predicates) и state machine инварианты на unit-уровне, без зависимости от реальной БД.

Последствия:
- быстрый и стабильный CI;
- риски реальных lock/timeout/DDL особенностей остаются за рамками v1 и покрываются отдельным e2e/integration прогоном при необходимости.

### 2) Runbook как часть определения готовности

Runbook обязателен, потому что lease/reclaim/cancel семантика без операционных подсказок плохо диагностируется по одним только endpoint’ам.

## Контракты и инварианты

- Claim atomic: `SELECT ... FOR UPDATE SKIP LOCKED`.
- Reclaim v1 = restart attempt (допускаются сбросы прогресса; `/top` может показывать snapshot предыдущей попытки до перезаписи).
- Lease-guarded writes: все worker-side записи условны по `(job_id, locked_by, lease_expires_at > now)`.
- `/top` details: `report_table_md`/trades доступны только для `succeeded` job.
- Cursor: opaque `base64url(json)`.

## Связанные файлы

- `docs/architecture/roadmap/milestone-5-epics-v1.md` — EPIC-12 в Milestone 5.
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md` — storage/SQL invariants.
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md` — worker flow + lease/cancel.
- `docs/architecture/backtest/backtest-jobs-api-v1.md` — API contract.

- `tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py` — state machine инварианты.
- `tests/unit/contexts/backtest/adapters/test_postgres_backtest_job_repositories.py` — SQL shape assertions.
- `tests/unit/contexts/backtest/adapters/test_backtest_runtime_config.py` — jobs config + runtime hash.
- `tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py` — worker use-case контракт.
- `tests/unit/apps/test_backtest_job_runner_main.py` — worker entrypoint (disabled -> exit 0).
- `tests/unit/apps/api/test_backtest_jobs_routes.py` — jobs endpoints контракт.
- `tests/unit/apps/api/test_backtest_jobs_dto.py` — cursor codec edge-cases.

## Как проверить

```bash
uv run ruff check .
uv run pyright

uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

## Риски и открытые вопросы

- Риск: unit-only SQL shape tests не ловят реальные проблемы блокировок/планов запросов в Postgres.
- Риск: при reclaim restart attempt UI может видеть “скачки” прогресса и/или stale `/top` snapshot (это допустимо v1 и должно быть явно отражено в runbook/UX).
