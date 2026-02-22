# Milestone 5 -- EPIC map (v1)

Карта EPIC'ов для Milestone 5: Backtest Jobs v1.

Milestone 5 добавляет асинхронный (job-based) путь запуска backtest, который:
- масштабируется по пользователям через несколько реплик job-runner,
- дает прогресс и best-so-far top результаты во время исполнения,
- сохраняет результаты в Postgres и позволяет получить их после завершения,
- поддерживает cancel,
- не застревает навсегда при рестарте job-runner (lease/reclaim + минимальный resume).

Референс по формату: `docs/architecture/roadmap/milestone-4-epics-v1.md`.

---

## Контекст и новые вводные (зафиксировано)

### Jobs scope (фикс)

- Jobs делаем только для backtest (не generic platform jobs).

### Runtime DSN + migrations DSN (фикс)

- Runtime (API + worker): `STRATEGY_PG_DSN`.
- Migrations runner: `POSTGRES_DSN` (в проде указывает на тот же Postgres инстанс).

### Enable toggle semantics (фикс)

- `backtest.jobs.enabled=false`:
  - jobs endpoints не монтируются,
  - worker пишет "disabled" и завершает `exit 0`.

### Request snapshot + hashes (фикс)

- `request_json` всегда содержит все, что влияет на результат.
- Saved-mode job дополнительно содержит:
  - `spec_hash`,
  - и (желательно) снапшот `spec_payload` на момент создания job.
- `request_hash` = sha256 от canonical JSON (`sort_keys`, `separators=(",", ":")`, `ensure_ascii=true`).
- `engine_params_hash` считаем при создании job из effective direction/sizing/execution после применения runtime defaults.
- `backtest_runtime_config_hash` = хэш только result-affecting секций:
  - `warmup/top_k/preselect`,
  - `execution`,
  - `reporting`.

### Error persistence policy (фикс)

- Для `failed` job сохраняем:
  - `last_error` (короткая строка),
  - `last_error_json` (RoehubError-подобный payload: `code/message/details`).
- Traceback хранится только в логах воркера/API (не в БД и не в ответах API).

### Persisted output policy v1 (фикс + уточнение)

- Persisted cap: `top_k <= top_k_persisted_default`.
- Persist сохраняем `min(top_k, cap)`.
- Ranking:
  - `total_return_pct` desc,
  - tie-break `variant_key` asc.

Сохранение отчета:
- `report_table_md` сохраняется для всех persisted top вариантов только для `succeeded` job (на этапе `finalizing`).
- trades сохраняются/возвращаются только для `top_trades_n` лучших и только для `succeeded` job.
- Для `running`/`cancelled`/`failed` `report_table_md` отсутствует (NULL) и не возвращается как содержимое отчета.

### Cancel (фикс)

- API позволяет запросить отмену.
- Cancel best-effort:
  - `queued -> cancelled` сразу,
  - `running -> cancel_requested_at` (воркер проверяет cancel на границах батчей).

### Lease/reclaim semantics (фикс)

- Claim atomic через `SELECT ... FOR UPDATE SKIP LOCKED`.
- Lease/heartbeat обязательны.
- При потере lease воркер обязан fail-fast остановиться.
- Все write операции (progress/top variants/final state) делаем условно по:
  `(job_id, locked_by, lease_expires_at > now())`.

### Progress semantics (фикс)

- Progress stage-specific:
  - `stage_a`: units = base variants.
  - `stage_b`: units = expanded variants.
  - `finalizing`: `processed_units=0, total_units=1` (или эквивалентный фиксированный шаг).

### Parallelism policy (фикс)

- По умолчанию внутренняя параллельность одного job = 1 (масштабируемся репликами воркера).
- Конфиг для parallel_workers добавляем сразу.

### Guards (фикс)

- В Milestone 5 используем те же guards, что и в sync:
  - `MAX_VARIANTS_PER_COMPUTE_DEFAULT = 600_000`
  - `MAX_COMPUTE_BYTES_TOTAL_DEFAULT = 5 GiB`

---

## Принцип декомпозиции Milestone 5

Milestone 5 делится на 4 логических части:

1) Storage/Domain: таблицы + репозитории + state machine + hashes.
2) Worker: claim/lease + staged batching + cancel + finalizing.
3) API: endpoints + ownership + pagination + deterministic errors.
4) Tests + runbook.

---

## Порядок внедрения (рекомендуемый)

1) BKT-EPIC-09 -- Storage/Domain/Config для jobs.
2) BKT-EPIC-10 -- Worker job-runner.
3) BKT-EPIC-11 -- API jobs endpoints.
4) BKT-EPIC-12 -- Tests + runbook + docs index.

---

## EPIC'и Milestone 5

### BKT-EPIC-09 -- Backtest Jobs storage v1 (PG schema + repositories + state machine)

**Цель:** добавить Postgres-хранилище для job'ов backtest и зафиксировать минимальный доменный контракт (state/stage/progress/lease/cancel), чтобы воркер и API могли работать независимо и детерминированно.

**Scope:**
- Alembic миграция:
  - `backtest_jobs`
  - `backtest_job_top_variants`
  - `backtest_job_stage_a_shortlist`
- Domain/app contracts в `contexts/backtest`:
  - job state machine (`queued|running|succeeded|failed|cancelled`)
  - stage literals (`stage_a|stage_b|finalizing`)
  - deterministic request snapshot (`request_json`) + `request_hash`
  - хранение `engine_params_hash` и `backtest_runtime_config_hash`
  - ошибка выполнения: `last_error` + `last_error_json` (RoehubError-подобный payload)
- Application ports:
  - `BacktestJobRepository` (create/get/list/cancel)
  - `BacktestJobLeaseRepository` (claim/heartbeat/finish conditional)
  - `BacktestJobResultsRepository` (persist best-so-far top-K + stage_a shortlist)
- Postgres adapters:
  - явный SQL (без ORM) + детерминированный `ORDER BY`
  - claim query включает `FOR UPDATE SKIP LOCKED`
  - keyset pagination для list: `ORDER BY created_at DESC, job_id DESC` + cursor `{created_at, job_id}`
  - все UPDATE/INSERT для running job выполняются условно по `(job_id, locked_by, lease_expires_at > now())`
- Runtime config:
  - расширить `configs/<env>/backtest.yaml` секцией `backtest.jobs.*`:
    - `enabled`
    - `top_k_persisted_default`
    - `max_active_jobs_per_user`
    - `claim_poll_seconds`
    - `lease_seconds`
    - `heartbeat_seconds`
    - `snapshot_seconds` и/или `snapshot_variants_step`
    - `parallel_workers` (default 1)
  - обновить loader/validator `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`.
  - добавить вычисление `backtest_runtime_config_hash` (result-affecting sections only).

**Non-goals:**
- retention/cleanup политика для старых job'ов и результатов.
- generic jobs framework для других контекстов.
- точный checkpoint Stage B (cursor resume) -- допускаем restart attempt.

**DoD:**
- Есть миграция и строгие constraints/indexes (FK, unique `(job_id, rank)`, индексы по `(user_id, state)` и т.п.).
- Репозитории покрывают минимум: create/get/list/cancel + claim/heartbeat/finish.
- SQL запросы содержат явный deterministic ordering и SKIP LOCKED для claim.
- Loader backtest.yaml валидирует jobs секцию fail-fast.

**Paths:**
- `alembic/versions/*_backtest_jobs_v1.py`
- `src/trading/contexts/backtest/domain/*` (job entities/value_objects/errors)
- `src/trading/contexts/backtest/application/ports/*` (job repositories)
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/*`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/dev/backtest.yaml`, `configs/test/backtest.yaml`, `configs/prod/backtest.yaml`

---

### BKT-EPIC-10 -- Backtest job-runner worker v1 (claim/lease/heartbeat + batching + cancel)

**Цель:** реализовать новый воркер, который выполняет queued backtest jobs батчами, пишет прогресс и best-so-far top результаты, поддерживает cancel и не допускает stuck jobs (lease/reclaim).

**Scope:**
- Новый процесс-воркер: `backtest-job-runner`.
- Wiring:
  - загрузка backtest runtime config (включая jobs toggles),
  - построение `IndicatorCompute` (warmup),
  - candle feed через canonical ClickHouse reader (как в sync backtest),
  - Postgres job repositories через `STRATEGY_PG_DSN`.
- Claim loop:
  - polling с `claim_poll_seconds`,
  - claim atomic (SKIP LOCKED) + lease setup.
- Execution flow (per job):
  - использовать `job.request_json` как source-of-truth (включая saved-mode `spec_payload` snapshot).
  - Stage A:
    - streaming scoring base variants,
    - deterministic shortlist top `preselect` (tie-break `base_variant_key`),
    - persist `stage_a_indexes` + метаданные (`stage_a_variants_total`, `risk_total`, `preselect_used`).
  - Stage B:
    - streaming scoring expanded variants батчами,
    - running top-K heap (K = min(top_k, top_k_persisted_default)), tie-break `variant_key`,
    - snapshot persisted top-K не чаще, чем `snapshot_seconds` и/или `snapshot_variants_step`.
  - Finalizing (только для succeeded):
    - повторно пересчитать финальный top-K через details scorer,
    - построить `report_table_md` для всех top вариантов,
    - построить trades только для `top_trades_n`,
    - записать `report_table_md`/`trades_json` в `backtest_job_top_variants` и завершить job.
- Cancel:
  - проверка `cancel_requested_at` на границах батчей,
  - `running -> cancelled` без сохранения `report_table_md`.
- Failures:
  - при ошибке исполнения job переводится в `failed` и сохраняет `last_error`/`last_error_json`,
  - traceback только в логах.
- Lease lost:
  - при условном update failure (или явной проверке lease) воркер прекращает обработку job и не пишет дальше.
- Observability:
  - Prometheus counters/histograms: claim/success/fail/cancel, durations, lease_lost.

**Non-goals:**
- масштабирование одного job на несколько воркеров.
- “точный” resume Stage B по cursor; допускаем restart attempt.

**DoD:**
- Можно запустить 2-4 реплики воркера: jobs обрабатываются параллельно и без двойного исполнения.
- Cancel работает и не приводит к failed.
- При падении воркера job reclaim'ится и в итоге завершается (succeeded/failed/cancelled).
- Для succeeded job persisted top результаты содержат `report_table_md` (и trades только для top_trades_n).

**Paths:**
- `apps/worker/backtest_job_runner/main/main.py`
- `apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py`
- `docs/runbooks/backtest-job-runner.md` (добавляется в EPIC-12)

---

### BKT-EPIC-11 -- Backtest Jobs API v1

**Цель:** предоставить UI набор endpoints для async backtest jobs: create/status/top/list/cancel, с deterministic ошибками и owner-only доступом.

**Scope:**
- Endpoints:
  - `POST /backtests/jobs` (создать job; request envelope как в `POST /backtests`)
- `GET /backtests/jobs/{job_id}` (status + progress + hashes)
  - `GET /backtests/jobs/{job_id}/top?limit=...` (best-so-far; на succeeded включает `report_table_md`)
  - `GET /backtests/jobs?state=&limit=&cursor=` (keyset pagination)
  - `POST /backtests/jobs/{job_id}/cancel` (idempotent)
- Enable toggle:
  - если `backtest.jobs.enabled=false` endpoints не монтируются.
- Ownership:
  - все операции только для owner (проверка в use-case).
- Валидации:
  - `top_k <= top_k_persisted_default`
  - `top_trades_n <= top_k`
  - quota: `max_active_jobs_per_user`, где active = queued + running.
- Хэши в API:
  - `request_hash`, `engine_params_hash`, `backtest_runtime_config_hash`.
- Ошибки выполнения:
  - для `failed` job status endpoint возвращает `last_error` + `last_error_json`.
- Output semantics:
  - на `running` `GET /top` возвращает ranking + payload; `report_table_md` отсутствует.
  - на `succeeded` `GET /top` возвращает `report_table_md` для всех top результатов; trades только для `top_trades_n`.
  - на `cancelled`/`failed` `report_table_md` отсутствует.
- Ошибки:
  - использовать `RoehubError` + deterministic 422 payload.

**Non-goals:**
- realtime push прогресса (SSE/WebSocket).
- UI.

**DoD:**
- Контракт endpoints зафиксирован и покрыт unit тестами.
- Детерминированная сортировка и пагинация.
- Ошибки не "плывут" и используют единый 422 payload.

**Paths:**
- `apps/api/routes/backtests.py` (расширение router) и/или `apps/api/routes/backtest_jobs.py`
- `apps/api/dto/*` (job request/response models)
- `apps/api/wiring/modules/backtest.py` (wiring jobs use-cases + toggle)
- `src/trading/contexts/backtest/application/use_cases/*` (jobs use-cases)

---

### BKT-EPIC-12 -- Tests + runbook (jobs)

**Цель:** закрепить Milestone 5 контракт тестами (детерминизм ordering/lease/cancel) и добавить минимальный runbook для эксплуатации job-runner.

**Scope:**
- Unit tests:
  - repositories (SQL shape assertions: SKIP LOCKED, ORDER BY, conditional updates),
  - job state machine invariants + cancel idempotency,
  - API router toggle (`backtest.jobs.enabled=false` не монтирует endpoints),
  - worker entrypoint: disabled -> `exit 0`.
- Smoke-level tests (без внешних сервисов):
  - deterministic cursor pagination payload formatting.
- Runbook:
  - `docs/runbooks/backtest-job-runner.md` (как запустить воркер, какие env нужны, как смотреть stuck jobs).
- Docs index:
  - обновить индекс: `python -m tools.docs.generate_docs_index`.

**Non-goals:**
- Тяжелые интеграционные тесты с реальным Postgres/ClickHouse.

**DoD:**
- `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` проходят.
- Документация и runbook добавлены, индекс docs обновлен.

**Paths:**
- `tests/unit/contexts/backtest/*`
- `tests/unit/apps/api/*`
- `tests/unit/apps/*`
- `docs/runbooks/backtest-job-runner.md`
