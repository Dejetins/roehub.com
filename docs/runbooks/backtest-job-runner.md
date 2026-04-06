# Ранбук backtest job runner

Ранбук для worker-процесса `backtest-job-runner`, который используется как canonical claimed
background worker для persisted runs.

## Status

- Status: active operational runbook for canonical `backtest-job-runner` v2 service surface.
- Canonical architecture reference:
  - [`docs/architecture/backtest/backtest-job-runner-v2.md`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/backtest-job-runner-v2.md)
- Compatibility note:
  - queued/running rows могут иметь `execution_mode=background_auto` или
    `execution_mode=background_manual_legacy`;
  - claimed hot path не должен обращаться к ClickHouse и не должен вызывать
    `IndicatorCompute.compute(...)`;
  - persisted summary rows остаются summary-only:
    `report_table_md=NULL`, `trades_json=NULL`.

## 1) Область и ссылки

Этот ранбук покрывает:
- запуск и toggles
- обязательные переменные окружения
- метрики и логи
- диагностику зависших jobs
- поведение cancel и lease-lost

Архитектурные ссылки:
- `docs/architecture/backtest/backtest-job-runner-v2.md` (canonical)
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md` (historical / compatibility-only)
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
uv run python -m apps.worker.backtest_job_runner.main.main --config configs/dev/backtest.yaml --metrics-port 9204 --instance-index 0
```

Локальный запуск с выбором конфига через env:

```bash
export STRATEGY_PG_DSN='postgresql://user:pass@127.0.0.1:5432/roehub'
export ROEHUB_ENV='prod'
export ROEHUB_BACKTEST_CONFIG='configs/prod/backtest.yaml'
uv run python -m apps.worker.backtest_job_runner.main.main --instance-index 0
```

Для fleet materialization каждый supervised worker process должен получать свой детерминированный
`--instance-index` в диапазоне `0..worker_processes-1`. Это значение входит в `locked_by` вместе
с `hostname` и `pid`, поэтому logs и lease owner остаются однозначными для каждого instance.

На `Mac Studio` launchd materialization рендерится из `backtest.jobs.worker_processes` через
`scripts/macos/render_backtest_job_runner_launchd.py`. Per-instance service shape остаётся purely
operational:

- prod labels: `com.roehub.backtest-job-runner.<instance_index>`
- test labels: `com.roehub.test.backtest-job-runner.<instance_index>`
- prod logs:
  `/Users/daniildegtyarev/Library/Logs/roehub/backtest-job-runner.<instance_index>.out.log`
- test logs:
  `/Users/daniildegtyarev/Library/Logs/roehub/test-backtest-job-runner.<instance_index>.out.log`

Метрики также биндуются per instance по детерминированному правилу:

- effective `metrics_port = base_metrics_port + instance_index`
- если `--metrics-port` не передан, `base_metrics_port` по умолчанию равен `9204`
- supervisor/service manager должен передавать уникальный `instance_index`, чтобы каждый worker
  instance имел distinct metrics endpoint
- launchd для prod передаёт общий `--metrics-port 9204`, для test `--metrics-port 19204`

Больше queue concurrency достигается только добавлением supervised worker processes. Один process
по-прежнему владеет single claim loop и обрабатывает one claimed job at a time.

## 4) Семантика toggle

Если `backtest.jobs.enabled=false` в runtime-конфиге:
- worker пишет лог `component=backtest-job-runner status=disabled`
- процесс завершается с кодом `0`
- claim loop не запускается

Такое поведение ожидаемо и безопасно для maintenance window.

Если `launchd` fleet materialization выполняется при `backtest.jobs.enabled=false`, helper script
должен оставить `0` worker services и удалить stale managed plists для этого profile.

## 5) Сигналы здоровья

Endpoint метрик:

```bash
curl -fsS http://127.0.0.1:$((9204 + 0))/metrics | head
```

Для worker fleet проверка должна использовать порт конкретного instance. Например, для
`instance_index=2` при базовом порте `9204` endpoint будет `http://127.0.0.1:9206/metrics`.

Проверка полного fleet на `Mac Studio`:

```bash
launchctl list | grep backtest-job-runner
```

Reload full supervised surface:

```bash
bash scripts/macos/reload_launchd_services.sh prod
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
- `execution_mode`
- `artifact_slot`
- `artifact_manifest_hash`

## 5.1 Closure smoke после rollout

Команда минимальной проверки после релиза worker/API/runtime:

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
```

Ожидаемый результат:

- background path продолжает читать pinned artifacts;
- `0 CH calls on hot path`;
- `0 IndicatorCompute.compute(...) calls on hot path`.

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
- у `running` job поле `cancel_requested_at` фиксируется один раз и остаётся видимым до terminal
  state
- такой `running` job продолжает удерживать publish guard для своего pinned slot identity до
  `succeeded|failed|cancelled`

Проверить статус:

```bash
curl -fsS -b cookies.txt http://127.0.0.1:8000/backtests/jobs/<job_id>
```

Проверить политику по top-строкам:

```bash
curl -fsS -b cookies.txt "http://127.0.0.1:8000/backtests/jobs/<job_id>/top?limit=10"
```

Для jobs, которые не в `succeeded`, `report_table_md` и `trades` не возвращаются.
Это одинаково для `background_auto` и `background_manual_legacy`.

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
- `background_auto` остаётся в `queued` без перехода к `running`: проверить worker toggle,
  claim loop и lease SQL.
- `background_manual_legacy` или `background_auto` падает с pin/manifest drift:
  проверить `artifact_slot`, `artifact_slot_generation`, `artifact_manifest_hash`,
  `artifact_asof_date` на row и соответствие published slot.
- В `/top` появились `report_table_md` или `trades_json`: это regression, потому что worker
  snapshots и terminal persisted rows должны оставаться summary-only.

## 11) Когда нужен rollback

Rollback запускать по `docs/runbooks/backtest-rollout-rollback.md`, если наблюдается хотя бы один
из признаков:

- repeated `event=job_failed` без локально исправимой request-ошибки;
- unexpected hot-path dependency on ClickHouse или `IndicatorCompute.compute(...)`;
- pin drift между claimed rows и published slot;
- массовое зависание `queued|running` runs после rollout.
