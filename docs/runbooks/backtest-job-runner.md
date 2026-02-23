# Backtest job runner runbook

Runbook for the `backtest-job-runner` worker process used by Backtest Jobs v1.

## 1) Scope and references

This runbook covers:
- startup and toggles
- required env vars
- metrics and logs
- stuck jobs diagnosis
- cancel and lease-lost behavior

Architecture references:
- `docs/architecture/backtest/backtest-job-runner-worker-v1.md`
- `docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md`
- `docs/architecture/backtest/backtest-jobs-api-v1.md`

## 2) Required environment

Minimum required variables for worker runtime:
- `STRATEGY_PG_DSN` (runtime Postgres DSN for jobs storage)
- `ROEHUB_ENV` (`dev`, `test`, or `prod`)

Optional config path override:
- `ROEHUB_BACKTEST_CONFIG` (path to `backtest.yaml`)

Migration variable (not used by worker runtime, used by migration runner):
- `POSTGRES_DSN`

ClickHouse vars (used by candle reader wiring):
- `CH_HOST`
- `CH_PORT`
- `CH_DATABASE`
- `CH_USER` (or `CLICKHOUSE_USER`)
- `CH_PASSWORD` (or `CLICKHOUSE_PASSWORD`)
- `CH_SECURE` (`0` or `1`)
- `CH_VERIFY` (`0` or `1`)

## 3) Start commands

Local run (dev config):

```bash
export STRATEGY_PG_DSN='postgresql://user:pass@127.0.0.1:5432/roehub'
export ROEHUB_ENV='dev'
uv run python -m apps.worker.backtest_job_runner.main.main --config configs/dev/backtest.yaml --metrics-port 9204
```

Local run with env-based config resolution:

```bash
export STRATEGY_PG_DSN='postgresql://user:pass@127.0.0.1:5432/roehub'
export ROEHUB_ENV='prod'
export ROEHUB_BACKTEST_CONFIG='configs/prod/backtest.yaml'
uv run python -m apps.worker.backtest_job_runner.main.main
```

## 4) Toggle semantics

If `backtest.jobs.enabled=false` in runtime config:
- worker logs `component=backtest-job-runner status=disabled`
- process exits with code `0`
- no claim loop starts

This behavior is expected and safe for maintenance windows.

## 5) Health signals

Metrics endpoint:

```bash
curl -fsS http://127.0.0.1:9204/metrics | head
```

Primary counters:
- `backtest_job_runner_claim_total`
- `backtest_job_runner_succeeded_total`
- `backtest_job_runner_failed_total`
- `backtest_job_runner_cancelled_total`
- `backtest_job_runner_lease_lost_total`

Primary histograms and gauges:
- `backtest_job_runner_job_duration_seconds`
- `backtest_job_runner_stage_duration_seconds`
- `backtest_job_runner_active_claimed_jobs`

Key log fields to monitor:
- `job_id`
- `attempt`
- `locked_by`
- `stage`
- `event`

## 6) Stuck jobs diagnosis

### 6.1 Find running jobs with expired lease

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

If `lease_expires_at < now()`, reclaim is expected. Claim SQL uses `FOR UPDATE SKIP LOCKED`.

### 6.2 Reclaim semantics in v1

On reclaim attempt:
- worker may restart from `stage_a`
- `processed_units` and `stage` can reset
- `attempt` increases

Observed `/top` behavior:
- previous persisted rows may remain visible until first overwrite in new attempt
- this temporary stale `/top` is expected in v1

### 6.3 Stage A shortlist and snapshot checks

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

## 7) Cancel runbook

Request cancel:

```bash
curl -fsS -X POST -b cookies.txt \
  http://127.0.0.1:8000/backtests/jobs/<job_id>/cancel
```

Expected behavior:
- `queued` job: immediate `cancelled`
- `running` job: best-effort, cancellation happens on batch boundaries

Check status:

```bash
curl -fsS -b cookies.txt http://127.0.0.1:8000/backtests/jobs/<job_id>
```

Check top rows policy:

```bash
curl -fsS -b cookies.txt "http://127.0.0.1:8000/backtests/jobs/<job_id>/top?limit=10"
```

For non-succeeded jobs, `report_table_md` and `trades` are not returned.

## 8) Lease-lost runbook

Symptoms:
- worker log contains `event=lease_lost`
- `backtest_job_runner_lease_lost_total` increases

Expected behavior:
- worker that lost lease stops writing for this job immediately
- no terminal finish write by that worker instance
- another worker can reclaim and continue

Useful check:

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

## 9) API list cursor smoke

`GET /backtests/jobs` returns opaque `next_cursor` in `base64url(json)` format.

Round trip smoke:
1. call `GET /backtests/jobs?limit=25`
2. copy `next_cursor` from response
3. call `GET /backtests/jobs?limit=25&cursor=<next_cursor>`
4. verify deterministic order by `created_at DESC, job_id DESC`

## 10) Common failures and actions

- Missing `STRATEGY_PG_DSN`: startup fails fast, set env var and restart.
- Invalid `CH_*` values: startup fails in ClickHouse settings loader.
- Disabled jobs toggle: check config for `backtest.jobs.enabled=false`.
- Rising failed counter: inspect `last_error` and `last_error_json` in `backtest_jobs`.
