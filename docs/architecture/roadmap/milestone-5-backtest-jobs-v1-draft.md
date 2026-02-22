# Milestone 5 -- Backtest Jobs v1

---

## Цель

Сделать backtest модуль масштабируемым для одновременных запусков несколькими пользователями:

- большие расчеты не блокируют UI и HTTP (async),
- есть прогресс выполнения и "best-so-far" top результаты во время расчета,
- результаты сохраняются в Postgres и доступны после завершения,
- можно отменить (cancel) выполняющийся backtest,
- система устойчиво переживает рестарт job-runner (минимальный resume).

---

## Контекст

Milestone 4 ввел синхронный (small-run) API `POST /backtests`:
- staged pipeline (Stage A shortlist -> Stage B exact -> top-K),
- deterministic ranking + tie-break,
- отчет `report.table_md` для каждого варианта в top-K,
- trades возвращаются только для `top_trades_n`.

Синхронный режим намеренно ограничен guards и не подходит для больших сеток/длинных периодов.
Milestone 5 добавляет асинхронный путь, который масштабируется через несколько реплик воркера.

Связанные документы:
- `docs/architecture/backtest/backtest-api-post-backtests-v1.md` (sync API, hashes, trades policy)
- `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` (Stage A/Stage B, guards, tie-break)
- `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md` (table_md + top_trades_n)

---

## Зафиксированные решения Milestone 5

### 1) Jobs делаем только для backtest (не generic platform jobs)

- Все сущности/таблицы/воркер/эндпойнты принадлежат bounded context `backtest`.
- В Milestone 5 не вводим общий "jobs" модуль для других контекстов.

### 2) Persisted output policy v1

- Храним `report_table_md` для всех persisted top вариантов.
- Trades храним/возвращаем только для `top_trades_n` лучших вариантов (как в sync API).

### 3) Cancel входит в Milestone 5

- API позволяет запросить отмену.
- Cancel best-effort: job-runner обязан проверять `cancel_requested_at` на границах батчей и завершать job в состоянии `cancelled`.

---

## Scope

### A) Storage (Postgres) для backtest jobs и результатов

Вводим новую модель хранения (через Alembic миграции) в том же Postgres, где уже живут таблицы Strategy.

Минимальный набор таблиц (v1):

1) `backtest_jobs`
- identity: `job_id (uuid)`, `user_id (uuid)`
- lifecycle:
  - `state`: `queued|running|succeeded|failed|cancelled`
  - `created_at`, `updated_at`
  - `started_at`, `finished_at`
  - `cancel_requested_at`
  - `last_error` (short string) + `last_error_json` (optional)
- request payload:
  - `mode`: `saved|template` (денормализация для удобства list/filter)
  - `request_json` (канонизированный payload v1)
  - `request_hash` (sha256 canonical json)
  - `engine_params_hash` (как в sync), плюс (опционально) `backtest_runtime_config_hash`
- progress:
  - `stage`: `stage_a|stage_b|finalizing` (строго фиксированный enum/strings)
  - `processed_units`, `total_units` (int)
  - `progress_updated_at`
- concurrency/lease:
  - `locked_by` (string, например `<hostname>-<pid>`)
  - `locked_at`
  - `lease_expires_at`
  - `heartbeat_at`
  - `attempt` (int, инкремент при каждом reclaim)

2) `backtest_job_top_variants`
- `job_id`, `rank` (1..K)
- variant identity:
  - `variant_key`
  - `indicator_variant_key` (если требуется для дебага/кешей)
  - `variant_index` (как в sync, если нужно)
- ranking:
  - `total_return_pct` (primary key for ordering)
  - tie-break фиксируется как `variant_key` (asc)
- persisted report:
  - `report_table_md` (ASCII markdown)
  - `trades_json` (NULL для rank > top_trades_n)
- payload для сохранения выбранного варианта:
  - `payload_json` (explicit selections: indicators/signals/risk/execution/direction/sizing)

3) (минимальный resume) `backtest_job_stage_a_shortlist`
- `job_id`
- `shortlist_json` (deterministic list of shortlisted base variants, достаточно для запуска Stage B)

Примечание:
- v1 не обязан хранить equity curve для всех top-500; при необходимости это расширение будущего milestone.

### B) Backtest job-runner worker (многорепличный)

Добавляем новый воркер (entrypoint) `backtest-job-runner`, который:

- забирает jobs из `backtest_jobs` (claim) атомарно через row-level lock:
  - `SELECT ... FOR UPDATE SKIP LOCKED`
  - меняет `state` в `running`, выставляет `locked_by`, `lease_expires_at`, `attempt += 1`
- выполняет расчет батчами и пишет прогресс:
  - Stage A: посчитать base grid без SL/TP, сохранить shortlist
  - Stage B: расширить shortlist по risk осям и посчитать точный score/report
  - поддерживать running top-K heap (K=500, tie-break по `variant_key`)
  - периодически сохранять persisted top-K snapshot (транзакционно)
- поддерживает cancel:
  - перед каждым батчем проверяет `cancel_requested_at`
  - при cancel корректно завершает job как `cancelled` (без "failed")
- поддерживает minimal resume:
  - если воркер умер, lease истекает, другой воркер может reclaim job
  - v1 допускает restart job c начала attempt (детерминированно), но job не должен застревать навсегда в `running`

Ожидаемая масштабируемость:
- несколько реплик воркера обрабатывают разные job'ы параллельно;
- один job в один момент времени исполняется максимум одним воркером.

### C) API: backtest jobs endpoints (async)

Синхронный `POST /backtests` (Milestone 4) сохраняется.
Добавляем отдельные endpoints для jobs (Milestone 5):

1) `POST /backtests/jobs`
- создает job для saved или template режима (request envelope как в `POST /backtests`)
- валидирует:
  - ownership (для saved),
  - guards/лимиты, которые можно проверить до запуска,
  - квоту `max_active_jobs_per_user`.
- response: `{job_id, state, request_hash, engine_params_hash}`

2) `GET /backtests/jobs/{job_id}`
- status + progress + timestamps + hashes.

3) `GET /backtests/jobs/{job_id}/top?limit=500`
- возвращает persisted top результаты (best-so-far во время running; финальные после succeeded/cancelled/failed).
- ordering фикс:
  - `total_return_pct` desc,
  - tie-break `variant_key` asc.

4) `GET /backtests/jobs?state=&limit=&cursor=`
- список "моих" job'ов (owner only) с пагинацией.

5) `POST /backtests/jobs/{job_id}/cancel`
- ставит `cancel_requested_at` (idempotent),
- state transitions:
  - `queued -> cancelled` (если job еще не claimed)
  - `running -> cancelled` (best-effort, при следующей проверке воркером)

Ошибки:
- используем общий контракт `RoehubError` и deterministic 422 payload (как в sync API).
- ownership/visibility проверяем в use-case.

### D) Runtime config (backtest.yaml)

Расширяем `configs/<env>/backtest.yaml` секцией jobs:

- `backtest.jobs.enabled` (toggle)
- `backtest.jobs.top_k_persisted_default` (default 500)
- `backtest.jobs.max_active_jobs_per_user` (default, например 2)
- `backtest.jobs.claim_poll_seconds` (default, например 1)
- `backtest.jobs.lease_seconds` (default, например 60)
- `backtest.jobs.heartbeat_seconds` (default, например 5)

Важно:
- defaults `top_trades_n_default` остаются в `backtest.reporting` (Milestone 4), job-runner использует их же.

### E) Observability + runbook

- Prometheus метрики job-runner: jobs claimed, active, succeeded/failed/cancelled, durations, lease lost, cancels.
- Runbook: запуск/масштабирование воркера, диагностика stuck jobs, проверка API.

---

## Что должно работать (смоук)

1) Два пользователя создают по job через `POST /backtests/jobs` и получают разные `job_id`.
2) Запущены 2-4 реплики `backtest-job-runner`:
   - каждый job claimed ровно одним воркером,
   - jobs считаются параллельно.
3) UI опрашивает `GET /backtests/jobs/{job_id}` и видит прогресс (stage + processed/total).
4) Во время `running` UI получает `GET /backtests/jobs/{job_id}/top` и видит best-so-far результаты.
5) Cancel:
   - UI вызывает `POST /backtests/jobs/{job_id}/cancel`,
   - job завершается как `cancelled`, дальнейшие вычисления прекращаются.
6) Рестарт воркера:
   - если воркер умер во время `running`, job reclaim'ится после истечения lease и в итоге завершается (минимально допустим перезапуск attempt).

---

## DoD

- Async flow реализован: create/status/top/list/cancel.
- Масштабирование через несколько реплик job-runner работает без гонок (row-lock + lease).
- Persisted top-K:
  - хранится как минимум top-500,
  - для всех persisted вариантов хранится `report_table_md`,
  - trades присутствуют только для `top_trades_n` лучших.
- Детерминизм сохранен:
  - один и тот же request + одинаковые runtime defaults -> одинаковые persisted результаты (top ordering + table_md).
- Minimal resume:
  - job не застревает в `running` навсегда,
  - lease/reclaim работает, итоговые данные консистентны.

---

## Кандидаты EPIC'ов (Milestone 5)

Нумерация продолжает Milestone 4 (`BKT-EPIC-01..08`).

1) BKT-EPIC-09 -- Backtest Jobs storage v1 (PG schema + repositories + state machine)
- Alembic миграции + Postgres adapters.
- Domain errors/invariants для job state + ownership.

2) BKT-EPIC-10 -- Backtest job-runner worker v1 (claim/lease/heartbeat + batching + cancel)
- Новый воркер + wiring + метрики.
- Persist progress и top-500 snapshots.

3) BKT-EPIC-11 -- Backtest Jobs API v1
- `POST /backtests/jobs`, `GET status/top/list`, `POST cancel`.
- Unified deterministic errors.

4) BKT-EPIC-12 -- Tests + runbook
- Unit/integration тесты job state, claim/reclaim, cancel.
- Runbook и docs index updates.

---

## Риски и открытые вопросы

- Retention: без политики очистки `backtest_job_top_variants` объем БД будет расти (в v1 можно оставить как риск, а policy сделать отдельным эпиком).
- Write amplification: частые snapshot update top-500 могут нагрузить PG; v1 должен писать батчами (по времени или по N батчей).
- Resume semantics: v1 допускает restart attempt с начала (дороже по CPU), но это проще и надежнее; более тонкий checkpoint Stage B можно вынести в следующий milestone.
