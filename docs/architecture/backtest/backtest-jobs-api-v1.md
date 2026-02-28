# Backtest Jobs API v1 -- create/status/top/list/cancel (BKT-EPIC-11)

Документ фиксирует архитектурный контракт EPIC-11: HTTP API для асинхронных backtest jobs (create/status/top/list/cancel) с owner-only доступом, deterministic пагинацией и единым RoehubError-контрактом.

## Цель

- Дать UI полный набор endpoints для async backtest workflow без polling по внутренним таблицам:
  - создать job,
  - получить status/progress,
  - читать best-so-far top,
  - листать историю jobs,
  - отправлять cancel.
- Зафиксировать правила детерминизма (hashes, ordering, cursor, error payload), чтобы API и worker оставались согласованными.

## Контекст

- EPIC-09 уже ввел storage/domain/ports для jobs (`backtest_jobs`, `backtest_job_top_variants`, `backtest_job_stage_a_shortlist`).
- EPIC-10 уже реализовал worker execution flow (`stage_a -> stage_b -> finalizing`), lease/reclaim и persisted top snapshot semantics.
- Sync endpoint `POST /backtests` уже задает request envelope и deterministic API error policy.
- Milestone 5 фиксирует для EPIC-11:
  - jobs toggle (`backtest.jobs.enabled`),
  - owner-only доступ,
  - quota и limit validations,
  - hashes в API,
  - `failed` payload (`last_error` + `last_error_json`),
  - `/top` state-dependent output policy.

## Scope

- Endpoints v1:
  - `POST /backtests/jobs`
  - `GET /backtests/jobs/{job_id}`
  - `GET /backtests/jobs/{job_id}/top?limit=...`
  - `GET /backtests/jobs?state=&limit=&cursor=`
  - `POST /backtests/jobs/{job_id}/cancel`
- Используем тот же auth dependency, что и `POST /backtests`.
- Все операции owner-only на уровне use-case (не через "скрывающие" SQL-фильтры).
- Валидации EPIC-11:
  - `top_k <= backtest.jobs.top_k_persisted_default`
  - `top_trades_n <= top_k`
  - quota: active jobs per user (`queued + running`) < `backtest.jobs.max_active_jobs_per_user`
- Deterministic list pagination через keyset cursor `(created_at DESC, job_id DESC)`.
- Deterministic API errors через `RoehubError` и existing global error handlers.

## Non-goals

- Realtime push прогресса (`SSE`/`WebSocket`).
- Изменение worker/state-machine/storage контрактов EPIC-09/10.
- Retention/cleanup политика jobs/results.

## Ключевые решения

### 1) Route composition: jobs API в backtest модуле, но отдельным router builder

Добавляем отдельный builder для jobs endpoints (`build_backtest_jobs_router`) и подключаем его в backtest API composition рядом с `build_backtests_router`.

Причины:

- разделяем sync API и jobs API без раздувания одного файла route-мэппинга,
- сохраняем единый bounded-context вход (`backtest`),
- повторно используем общие DTO/error/current-user паттерны.

### 2) Toggle semantics: endpoints физически не монтируются при `backtest.jobs.enabled=false`

- Если toggle выключен, routes `/backtests/jobs*` отсутствуют в FastAPI routing table.
- `POST /backtests` остается доступным.

Это соответствует Milestone 5 контракту "не монтировать jobs endpoints".

### 3) Owner-only policy с явным `403` (подтверждено)

Для operations по `job_id` (`status/top/cancel`) используем двухшаговую проверку в use-case:

1. Read job без owner filter.
2. Если job есть, но `job.user_id != current_user.user_id` -> `forbidden (403)`.

Если job отсутствует -> `not_found (404)`.

Так сохраняем явную семантику "существует, но не твой ресурс".

### 4) Create flow сохраняет canonical snapshot + reproducibility hashes

`POST /backtests/jobs`:

- принимает тот же request envelope, что и `POST /backtests` (`BacktestsPostRequest`),
- валидирует mode contract (`strategy_id xor template`) и saved-only overrides,
- в saved mode читает strategy snapshot для ownership + `spec_hash/spec_payload_json`,
- формирует canonical `request_json` для воспроизводимости,
- считает и сохраняет:
  - `request_hash`
  - `engine_params_hash`
  - `backtest_runtime_config_hash`
- создает queued job (`state=queued`, `stage=stage_a`, progress=0, no lease fields).

`request_json` и saved snapshot вместе обязаны содержать все данные, влияющие на результат.

### 5) Cursor contract: opaque `base64url(json)` (подтверждено)

Для `GET /backtests/jobs`:

- внутренняя cursor VO: `BacktestJobListCursor {created_at, job_id}`,
- transport format: opaque string `base64url(canonical_json(cursor_payload))`,
- canonical JSON: `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=True`.

Плюсы:

- UI не зависит от внутренних полей keyset cursor,
- можно безопасно эволюционировать payload (versioning) без ломки клиента.

### 6) `/top` output policy: summary-only + context for lazy report

`GET /backtests/jobs/{job_id}/top` возвращает rows из `backtest_job_top_variants` (`rank ASC, variant_key ASC`) с полями ranking + payload всегда.

Дополнительно endpoint возвращает `report_context` (run-context для
`POST /api/backtests/variant-report`), чтобы UI мог загрузить report/trades
по явному действию `Load report` для выбранного `variant_key`.

### 7) Cancel endpoint idempotent и возвращает status payload (подтверждено)

`POST /backtests/jobs/{job_id}/cancel`:

- `queued -> cancelled` сразу,
- `running -> cancel_requested_at` (best-effort),
- terminal states возвращаются без изменений.

Endpoint всегда возвращает актуальный snapshot job status (не `204`), чтобы UI не делал лишний `GET`.

### 8) Error model: только RoehubError-коды + deterministic 422 details

Ошибки routes/use-cases маппятся в canonical коды:

- `validation_error` (422)
- `not_found` (404)
- `forbidden` (403)
- `conflict` (409)
- `unexpected_error` (500)

Порядок validation errors остается deterministic через `apps/api/common/errors.py`.

## Endpoint contracts v1

### 1) `POST /backtests/jobs`

Request:

- тот же envelope, что и `POST /backtests`:
  - `time_range`
  - `strategy_id xor template`
  - `overrides?` (saved-only)
  - `warmup_bars?`, `top_k?`, `preselect?`, `top_trades_n?`

Доп. валидации jobs API:

- `top_k <= backtest.jobs.top_k_persisted_default`
- `top_trades_n <= top_k`
- active quota per user.

Response (`201 Created`):

- job status snapshot:
  - `job_id`, `mode`, `state`, `stage`
  - timestamps (`created_at`, `updated_at`, ...)
  - progress counters
  - hashes (`request_hash`, `engine_params_hash`, `backtest_runtime_config_hash`)
  - `spec_hash` (saved mode only)

### 2) `GET /backtests/jobs/{job_id}`

Response (`200 OK`):

- полный status/progress snapshot job,
- hashes (`request_hash`, `engine_params_hash`, `backtest_runtime_config_hash`),
- для `failed` дополнительно:
  - `last_error`
  - `last_error_json` (`code/message/details`).

### 3) `GET /backtests/jobs/{job_id}/top?limit=...`

Request params:

- `limit` optional; default = `backtest.jobs.top_k_persisted_default`.
- validation: `limit > 0` and `limit <= backtest.jobs.top_k_persisted_default`.

Response (`200 OK`):

- `job_id`, `state`, `report_context`, `items[]`.
- `items[]` rows ordered by rank:
  - `rank`, `variant_key`, `indicator_variant_key`, `variant_index`, `total_return_pct`, `payload`.
- `report_context` содержит поля run-context (`time_range`, `strategy_id xor template`,
  `overrides?`, `warmup_bars?`, `include_trades`) для on-demand report endpoint.

### 4) `GET /backtests/jobs?state=&limit=&cursor=`

Request params:

- `state` optional enum: `queued|running|succeeded|failed|cancelled`.
- `limit` optional, default `50`, max `250`.
- `cursor` optional opaque base64url string.

Response (`200 OK`):

- `items[]` (deterministic order: `created_at DESC, job_id DESC`),
- `next_cursor` (opaque string or `null`).

Каждый item содержит summary поля, достаточные для списка jobs:

- `job_id`, `mode`, `state`, `stage`,
- `created_at`, `updated_at`, `started_at`, `finished_at`, `cancel_requested_at`,
- `processed_units`, `total_units`.

### 5) `POST /backtests/jobs/{job_id}/cancel`

Response (`200 OK`):

- status snapshot после попытки cancel (idempotent).

## Контракты и инварианты

- Jobs endpoints доступны только при `backtest.jobs.enabled=true`.
- Все operations owner-only; чужая существующая job -> `403`, отсутствующая -> `404`.
- List ordering фиксирован: `created_at DESC, job_id DESC`.
- `/top` ordering фиксирован: `rank ASC, variant_key ASC`.
- `request_hash/engine_params_hash/backtest_runtime_config_hash` всегда возвращаются в status payload.
- `failed` status всегда включает `last_error + last_error_json`.
- `/top` не содержит eager `report_table_md`/`trades`; report details читаются через `variant-report`.

## Wiring и целевые implementation paths

API routes/DTO:

- `apps/api/routes/backtest_jobs.py` -- новый jobs router builder.
- `apps/api/routes/backtests.py` -- подключение jobs router в backtest module boundary.
- `apps/api/dto/backtest_jobs.py` -- request/response модели jobs API + cursor codec.
- `apps/api/dto/__init__.py` -- re-export новых DTO helpers.

Backtest API wiring:

- `apps/api/wiring/modules/backtest.py`:
  - сборка jobs use-cases,
  - загрузка `backtest.jobs.*`,
  - conditional router mounting по `jobs.enabled`.

Application layer:

- `src/trading/contexts/backtest/application/use_cases/create_backtest_job.py`
- `src/trading/contexts/backtest/application/use_cases/get_backtest_job_status.py`
- `src/trading/contexts/backtest/application/use_cases/get_backtest_job_top.py`
- `src/trading/contexts/backtest/application/use_cases/list_backtest_jobs.py`
- `src/trading/contexts/backtest/application/use_cases/cancel_backtest_job.py`
- `src/trading/contexts/backtest/application/use_cases/errors.py` -- job-specific error builders/mapping.

## Связанные файлы

Roadmap/docs:

- `docs/architecture/roadmap/milestone-5-epics-v1.md`
- `docs/architecture/roadmap/base_milestone_plan.md`
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md`

Current implementation references:

- `apps/api/routes/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `apps/api/main/app.py`
- `apps/api/common/errors.py`
- `apps/api/dto/backtests.py`
- `src/trading/contexts/backtest/application/ports/backtest_job_repositories.py`
- `src/trading/contexts/backtest/domain/value_objects/backtest_job_cursor.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/persistence/postgres/backtest_job_results_repository.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`

## Как проверить

После реализации EPIC-11:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index --check
```

Целевые тесты (минимум):

```bash
uv run pytest -q tests/unit/apps/api
uv run pytest -q tests/unit/contexts/backtest
```

## Риски

- Риск: drift между логикой materialization effective request в create-use-case и decode path в worker. Митигатор: единый canonical helper и unit tests с round-trip payload.
- Риск: при `jobs.enabled=true` и некорректном DB wiring API стартует, но endpoints нерабочие. Митигатор: fail-fast startup validation для jobs dependencies.
- Риск: большие `/top` payload при высоком persisted cap. Митигатор: limit validation + optional pagination extension в будущем.
