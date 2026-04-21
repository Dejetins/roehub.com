# Runbook — Backtest Artifact Runtime Rollout / Rollback

Этот runbook фиксирует R10-03 operational closure для artifact-backed runtime в `dev`, `test` и
`prod`. Документ не вводит новый cutover path: он описывает, как безопасно выкатывать уже shipped
runtime и как откатываться при нарушении deterministic test/perf/runbook contract.

## Status

- Status: active R10-03 closure runbook.
- Canonical scope:
  - sync launch через `POST /backtests`;
  - canonical claimed background execution через `execution_mode=background_auto`;
  - already-persisted compatibility rows могут всё ещё использовать
    `execution_mode=background_manual_legacy`;
  - summary-only persisted rows (`report_table_md=NULL`, `trades_json=NULL`);
  - artifact publish / slot pinning / rollback coordination.

Связанные документы:

- `docs/architecture/backtest/README.md`
- `docs/architecture/backtest/README.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/runbooks/backtest-job-runner.md`

## 1) Preconditions

Перед rollout должны быть готовы:

- published slot с валидным `current.yaml`;
- worker/API build, который уже использует artifact-backed runtime;
- доступ к `STRATEGY_PG_DSN` и artifact root;
- план отката на предыдущий release image/config;
- понимание, есть ли активные pinned background runs.

## 2) Rollout sequence

### 2.1 Validate artifacts and runtime contracts

```bash
uv run pytest -q \
  tests/unit/contexts/backtest/application/services/v2 \
  tests/unit/contexts/backtest/application/use_cases/test_run_backtest_job_runner_v1.py \
  tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py \
  tests/unit/apps/api/test_backtests_routes.py
```

### 2.2 Validate perf closure

```bash
uv run pytest -q \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py \
  tests/perf_smoke/contexts/backtest/test_backtest_staged_runner_perf_smoke.py
```

Ожидаемый результат:

- `0 CH calls on hot path`
- `0 IndicatorCompute.compute(...) calls on hot path`
- legacy R0 reference остаётся executable для baseline comparison

### 2.3 Deploy API and worker

Порядок rollout:

1. выкатить API build;
2. выкатить `backtest-job-runner` worker fleet;
3. подтвердить `service-level smoke`: fleet materialized, live worker instances совпадают с
   `worker_processes`, и worker не пишет `status=disabled` при `backtest.jobs.enabled=true`;
4. не менять `current.yaml` вручную во время rollout.

### 2.4 Verify post-rollout behavior

Проверки после деплоя:

- sync launch возвращает persisted metadata с `execution_mode=sync_inline`;
- canonical background launches продолжают показывать `background_auto`;
- already-persisted compatibility rows при необходимости всё ещё показывают
  `background_manual_legacy`;
- `/top` и persisted rows не materialize'ят `report_table_md` и `trades_json`;
- новых ClickHouse/indicator-compute hot-path зависимостей не появилось.

Минимальная SQL/HTTP verification sequence:

```sql
SELECT
  job_id,
  state,
  execution_mode,
  artifact_slot,
  artifact_slot_generation,
  artifact_manifest_hash,
  report_table_md,
  trades_json
FROM backtest_job_top_variants tv
JOIN backtest_jobs j ON j.job_id = tv.job_id
ORDER BY j.created_at DESC, j.job_id DESC, tv.rank ASC
LIMIT 20;
```

Ожидаемо:

- `execution_mode in ('sync_inline', 'background_auto', 'background_manual_legacy')`
- `report_table_md IS NULL`
- `trades_json IS NULL`

## 3) Rollback triggers

Rollback обязателен, если возникает хотя бы один trigger:

- perf smoke перестал проходить;
- hot path снова обращается к ClickHouse;
- hot path снова вызывает `IndicatorCompute.compute(...)`;
- claimed background runs массово падают из-за pin drift;
- persisted summary rows перестали быть summary-only;
- API/worker не могут стабильно обслуживать canonical `background_auto` path или already-persisted
  compatibility rows с `background_manual_legacy`.

## 4) Rollback actions

Порядок rollback:

1. остановить rollout новых release units;
2. вернуть API и worker на предыдущий стабильный release;
3. не переписывать published slot contents вручную;
4. если проблема в artifacts:
   - сохранить текущий `current.yaml`;
   - восстановить предыдущий known-good pointer/release только штатной процедурой publish;
5. для активных background runs:
   - `queued` rows можно отменить сразу;
   - `running` rows с `cancel_requested_at` всё ещё держат pin guard до terminal state;
6. повторно прогнать минимальную verification sequence на предыдущем release.

## 5) Post-rollback verification

```bash
uv run pytest -q \
  tests/perf_smoke/contexts/backtest/test_backtest_r0_baseline_perf_smoke.py
```

```sql
SELECT
  job_id,
  state,
  execution_mode,
  cancel_requested_at,
  artifact_slot,
  artifact_manifest_hash
FROM backtest_jobs
ORDER BY created_at DESC, job_id DESC
LIMIT 20;
```

Rollback считается завершённым только если:

- новая failure signature исчезла;
- background claim loop снова стабилен;
- sync/background launches не нарушают summary-only contract;
- perf smoke снова подтверждает zero-call hot path.

## 6) Stop conditions

Остановить rollout и не продолжать вручную, если:

- slot publish blocked by active pin;
- manifests validation fails;
- perf smoke fails;
- owner-facing API contract начал отдавать unexpected payload changes.

В этих случаях не делать ручной cleanup slot contents и не редактировать `current.yaml`
вне publish procedure.
