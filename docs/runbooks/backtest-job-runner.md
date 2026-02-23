# Ранбук backtest job runner

Ранбук для worker-процесса `backtest-job-runner`, который используется в Backtest Jobs v1.

## 1) Область и ссылки

Этот ранбук покрывает:
- запуск и toggles
- обязательные переменные окружения
- метрики и логи
- диагностику зависших jobs
- поведение cancel и lease-lost

Архитектурные ссылки:
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
- `docs/architecture/backtest/backtest-jobs-api-v1.md`

## 2) Обязательное окружение

Минимально обязательные переменные для runtime worker:
- `STRATEGY_PG_DSN` (runtime Postgres DSN для jobs storage)
- `ROEHUB_ENV` (`dev`, `test` или `prod`)

Опциональный override пути к конфигу:
- `ROEHUB_BACKTEST_CONFIG` (путь к `backtest.yaml`)

Переменная для миграций (в runtime worker не используется, используется migration runner):
- `POSTGRES_DSN`

Переменные ClickHouse (используются в wiring candle reader):
- `CH_HOST`
- `CH_PORT`
- `CH_DATABASE`
- `CH_USER` (или `CLICKHOUSE_USER`)
- `CH_PASSWORD` (или `CLICKHOUSE_PASSWORD`)
- `CH_SECURE` (`0` или `1`)
- `CH_VERIFY` (`0` или `1`)

## 3) Команды запуска

Локальный запуск (dev-конфиг):

```bash
export STRATEGY_PG_DSN='postgresql://user:pass@127.0.0.1:5432/roehub'
export ROEHUB_ENV='dev'
uv run python -m apps.worker.backtest_job_runner.main.main --config configs/dev/backtest.yaml --metrics-port 9204
```

Локальный запуск с выбором конфига через env:

```bash
export STRATEGY_PG_DSN='postgresql://user:pass@127.0.0.1:5432/roehub'
export ROEHUB_ENV='prod'
export ROEHUB_BACKTEST_CONFIG='configs/prod/backtest.yaml'
uv run python -m apps.worker.backtest_job_runner.main.main
```

## 4) Семантика toggle

Если `backtest.jobs.enabled=false` в runtime-конфиге:
- worker пишет лог `component=backtest-job-runner status=disabled`
- процесс завершается с кодом `0`
- claim loop не запускается

Такое поведение ожидаемо и безопасно для maintenance window.

## 5) Сигналы здоровья

Endpoint метрик:

```bash
curl -fsS http://127.0.0.1:9204/metrics | head
```

Основные counters:
- `backtest_job_runner_claim_total`
- `backtest_job_runner_succeeded_total`
- `backtest_job_runner_failed_total`
- `backtest_job_runner_cancelled_total`
- `backtest_job_runner_lease_lost_total`

Основные histograms и gauges:
- `backtest_job_runner_job_duration_seconds`
- `backtest_job_runner_stage_duration_seconds`
- `backtest_job_runner_active_claimed_jobs`

Ключевые поля логов для мониторинга:
- `job_id`
- `attempt`
- `locked_by`
- `stage`
- `event`

## 6) Диагностика зависших jobs

### 6.1 Найти running jobs с истекшим lease

```sql
SELECT
  job_id,
  state,
  stage,
  processed_units,
  total_units,
  locked_by,
  lease_expires_at,
  attempt,
  updated_at
FROM backtest_jobs
WHERE state = 'running'
ORDER BY lease_expires_at ASC, created_at ASC, job_id ASC;
```

Если `lease_expires_at < now()`, reclaim ожидаем. Claim SQL использует `FOR UPDATE SKIP LOCKED`.

### 6.2 Семантика reclaim в v1

При попытке reclaim:
- worker может перезапустить job с `stage_a`
- `processed_units` и `stage` могут сбрасываться
- `attempt` увеличивается

Наблюдаемое поведение `/top`:
- предыдущие сохранённые строки могут оставаться видимыми до первой перезаписи в новой попытке
- такая временная stale-выдача `/top` ожидаема в v1

### 6.3 Проверки shortlist Stage A и snapshot

```sql
SELECT job_id, stage_a_variants_total, risk_total, preselect_used, updated_at
FROM backtest_job_stage_a_shortlist
WHERE job_id = '00000000-0000-0000-0000-000000000000';
```

```sql
SELECT job_id, rank, variant_key, report_table_md, trades_json, updated_at
FROM backtest_job_top_variants
WHERE job_id = '00000000-0000-0000-0000-000000000000'
ORDER BY rank ASC, variant_key ASC;
```

## 7) Ранбук отмены (cancel)

Отправить cancel:

```bash
curl -fsS -X POST -b cookies.txt \
  http://127.0.0.1:8000/backtests/jobs/<job_id>/cancel
```

Ожидаемое поведение:
- job в `queued`: сразу `cancelled`
- job в `running`: best-effort, отмена происходит на границах батчей

Проверить статус:

```bash
curl -fsS -b cookies.txt http://127.0.0.1:8000/backtests/jobs/<job_id>
```

Проверить политику по top-строкам:

```bash
curl -fsS -b cookies.txt "http://127.0.0.1:8000/backtests/jobs/<job_id>/top?limit=10"
```

Для jobs, которые не в `succeeded`, `report_table_md` и `trades` не возвращаются.

## 8) Ранбук lease-lost

Симптомы:
- в логах worker есть `event=lease_lost`
- растёт `backtest_job_runner_lease_lost_total`

Ожидаемое поведение:
- worker, потерявший lease, немедленно перестаёт писать по этой job
- terminal finish write этим экземпляром worker не выполняется
- другой worker может reclaim-нуть job и продолжить

Полезная проверка:

```sql
SELECT
  job_id,
  state,
  locked_by,
  lease_expires_at,
  attempt,
  updated_at
FROM backtest_jobs
WHERE job_id = '00000000-0000-0000-0000-000000000000';
```

## 9) Smoke для API list cursor

`GET /backtests/jobs` возвращает opaque `next_cursor` в формате `base64url(json)`.

Round trip smoke:
1. вызвать `GET /backtests/jobs?limit=25`
2. скопировать `next_cursor` из ответа
3. вызвать `GET /backtests/jobs?limit=25&cursor=<next_cursor>`
4. проверить детерминированный порядок `created_at DESC, job_id DESC`

## 10) Частые сбои и действия

- Нет `STRATEGY_PG_DSN`: startup падает сразу, задайте переменную и перезапустите.
- Некорректные значения `CH_*`: startup падает в loader настроек ClickHouse.
- Jobs отключены toggle-ом: проверьте конфиг `backtest.jobs.enabled=false`.
- Растёт failed counter: проверьте `last_error` и `last_error_json` в `backtest_jobs`.
