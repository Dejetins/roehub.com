# Mac Studio Native Backend Operations

Актуальный операционный ранбук для production backend runtime на `Mac Studio` без Docker/Colima.

## Статус

- production backend runtime работает как native host-services и user `LaunchAgents`;
- Docker/Colima path считается legacy и не является target state;
- публичный edge/TLS остается на `VPS` (`roehub.com`), `Mac Studio` остается private backend host.

## Runtime owner

- runtime owner: `daniildegtyarev`;
- все кастомные backend процессы запускаются как user-level `launchd` services;
- self-hosted runner и `tailscale serve` остаются в той же user-session модели.

## Paths and files

- app checkout/runtime: `/opt/roehub/app`
- keycloak runtime: `/opt/roehub/keycloak/current`
- host config root: `/opt/roehub/config`
- backtest artifact root: `/opt/roehub/state/backtest_artifacts/v2`
- host binaries: `/opt/roehub/bin`, `/opt/clickhouse/clickhouse`
- prod env: `/Users/daniildegtyarev/.config/roehub/roehub.env`
- keycloak env: `/Users/daniildegtyarev/.config/roehub/keycloak.env`
- test env: `/Users/daniildegtyarev/.config/roehub/roehub.test.env`
- launch agents: `/Users/daniildegtyarev/Library/LaunchAgents`

Совместимый env-path для legacy loaders:

- `/etc/roehub/roehub.env` -> symlink на `/Users/daniildegtyarev/.config/roehub/roehub.env`

Ключевые файлы конфигурации:

- `/opt/roehub/config/prometheus.prod.yml`
- `/opt/roehub/config/prometheus.test.yml`
- `/opt/roehub/config/blackbox.yml`
- `/opt/roehub/config/blackbox.test.yml`
- `/opt/roehub/config/clickhouse.config.xml`
- `/opt/roehub/config/clickhouse.config.test.xml`

## Services map

Production:

- `postgresql@16` (`brew services`, `127.0.0.1:5432`)
- `redis` (`brew services`, `127.0.0.1:6379`)
- `grafana` (`brew services`, `127.0.0.1:3000`)
- `prometheus` (`brew services`, `127.0.0.1:9090`)
- `node_exporter` (`brew services`, `127.0.0.1:9100`)
- `com.roehub.clickhouse` (`launchd`, `127.0.0.1:8123/9000`)
- `com.roehub.blackbox-exporter` (`launchd`, `127.0.0.1:9115`)
- `com.roehub.clickhouse-exporter` (`launchd`, `127.0.0.1:9116`)
- `com.roehub.redis-exporter` (`launchd`, `127.0.0.1:9121`)
- `com.roehub.postgres-exporter` (`launchd`, `127.0.0.1:9187`)
- `com.roehub.tailscale-runtime` (`launchd`, periodic one-shot reconnection/check + serve sync)
- `com.roehub.keycloak` (`launchd`, auth provider, `127.0.0.1:18080`, ready-check `127.0.0.1:19000/health/ready`)
- `com.roehub.api` (`launchd`, `127.0.0.1:8000`)
- `com.roehub.market-data-ws-worker` (`launchd`, metrics `127.0.0.1:9201`)
- `com.roehub.market-data-scheduler` (`launchd`, metrics `127.0.0.1:9202`)
- `com.roehub.backtest-artifact-publisher` (`launchd`, metrics `127.0.0.1:9203`, daily `03:05 Europe/Moscow`)
- `com.roehub.backtest-job-runner.<instance_index>` (`launchd`, materialized from `backtest.jobs.worker_processes`, metrics `127.0.0.1:(9204 + instance_index)`)

Test:

- `com.roehub.test.postgres` (`127.0.0.1:15433`)
- `com.roehub.test.redis` (`127.0.0.1:16379`)
- `com.roehub.test.clickhouse` (`127.0.0.1:18124/19001`)
- `com.roehub.test.grafana` (`127.0.0.1:13000`)
- `com.roehub.test.prometheus` (`127.0.0.1:19090`)
- `com.roehub.test.blackbox-exporter` (`127.0.0.1:19115`)
- `com.roehub.test.api` (`127.0.0.1:18000`)
- `com.roehub.test.market-data-ws-worker` (metrics `127.0.0.1:19201`)
- `com.roehub.test.market-data-scheduler` (metrics `127.0.0.1:19202`)
- `com.roehub.test.backtest-artifact-publisher` (metrics `127.0.0.1:19203`, daily `03:05 Europe/Moscow`)
- `com.roehub.test.backtest-job-runner.<instance_index>` (materialized from `backtest.jobs.worker_processes`, metrics `127.0.0.1:(19204 + instance_index)`)

## Common commands

Подготовка native окружения:

```bash
bash scripts/macos/install_native_backend_prereqs.sh
bash scripts/macos/bootstrap_native_prod.sh
bash scripts/macos/bootstrap_native_test.sh
```

Reload сервисов:

```bash
bash scripts/macos/reload_launchd_services.sh prod
bash scripts/macos/reload_launchd_services.sh test
bash scripts/macos/reload_launchd_services.sh all
```

Monit (native service supervision):

```bash
/opt/homebrew/bin/brew services start monit
/opt/homebrew/bin/brew services restart monit
/opt/homebrew/bin/brew services stop monit
```

Monit проверки/управление:

```bash
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc status
/opt/homebrew/opt/monit/bin/monit -t -c /opt/homebrew/etc/monitrc
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc reload
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | grep roehub_keycloak
```

Keycloak quick checks:

```bash
launchctl list | grep com.roehub.keycloak
launchctl print gui/$(id -u)/com.roehub.keycloak | grep -E 'state =|pid =|last exit code ='
curl -fsS http://127.0.0.1:19000/health/ready
```

Keycloak auth operations (realm/client/OTP/local setup):

- `docs/runbooks/keycloak-local-setup-and-ops.md`
- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`

`bootstrap_native_prod.sh` и `bootstrap_native_test.sh` устанавливают static launchd templates и
рендерят per-instance `backtest-job-runner` plists из `backtest.jobs.worker_processes`.
`bootstrap_native_prod.sh` дополнительно синхронизирует Monit snippets из репозитория:
`infra/scripts/monit/*.monitrc` и `infra/scripts/monit/launchctl_service_control.sh`.
В production baseline сюда входит `infra/scripts/monit/roehub-keycloak.monitrc`.
`reload_launchd_services.sh` сначала выгружает текущие static/worker services, затем заново
рендерит worker fleet и bootstrap-ит ровно желаемую cardinality для profile.

GitHub Actions workflow `deploy-backend` должен использовать этот же install/reload path:
сначала `bash scripts/macos/bootstrap_native_prod.sh`, затем
`bash scripts/macos/reload_launchd_services.sh prod`. Когда `backtest.jobs.enabled=true`,
production rollout без live `backtest-job-runner` fleet и без совпадения с
`backtest.jobs.worker_processes` считается некорректным, даже если `com.roehub.api` уже поднялся.
Этот gate является `service-level smoke`, а не request-path smoke, и использует supervisor,
process, and `metrics endpoint` signals. Правило остаётся жестким: `no synthetic production job`.

Операционная vocabulary для claimed background path остаётся следующей:

- `background_auto` является canonical background mode для новых runs;
- `background_manual_legacy` остаётся compatibility-only literal для уже persisted rows;
- архитектурные правила и compatibility boundaries принадлежат
  [`docs/architecture/backtest/README.md`](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/backtest/README.md),
  а этот ранбук описывает только deploy/ops surface.

Отдельно проверить worker fleet:

```bash
launchctl list | grep backtest-job-runner
```

Ручная service-level smoke проверка после deploy:

```bash
cd /opt/roehub/app
base_metrics_port=9204
worker_runtime=()
while IFS= read -r runtime_value; do
  worker_runtime+=("${runtime_value}")
done < <(
  /opt/roehub/app/.venv/bin/python -c "from trading.contexts.backtest.adapters.outbound import load_backtest_runtime_config; config = load_backtest_runtime_config('/opt/roehub/app/configs/prod/backtest.yaml'); print('1' if config.jobs.enabled else '0'); print(config.jobs.worker_processes)"
)
jobs_enabled="${worker_runtime[0]}"
worker_processes="${worker_runtime[1]}"
echo "service-level smoke target=backtest-job-runner"
echo "jobs.enabled=${jobs_enabled}"
echo "worker_processes=${worker_processes}"
echo "metrics_endpoint_base_port=${base_metrics_port}"
echo "no synthetic production job"
if [ "${jobs_enabled}" != '1' ]; then
  echo "backtest-job-runner fleet is not required because jobs.enabled=false"
  exit 0
fi
registered_workers="$(
  launchctl list \
    | awk '$3 ~ /^com\.roehub\.backtest-job-runner\.[0-9]+$/ {count++} END {print count+0}'
)"
live_workers="$(
  launchctl list \
    | awk '$3 ~ /^com\.roehub\.backtest-job-runner\.[0-9]+$/ && $1 ~ /^[0-9]+$/ {count++} END {print count+0}'
)"
echo "registered_backtest_job_runner_workers=${registered_workers}"
echo "live_backtest_job_runner_workers=${live_workers}"
launchctl list | grep backtest-job-runner
test "${registered_workers}" = "${worker_processes}"
test "${live_workers}" = "${worker_processes}"
for ((instance_index = 0; instance_index < worker_processes; instance_index++)); do
  launchctl print "gui/$(id -u)/com.roehub.backtest-job-runner.${instance_index}" | grep -E 'state =|pid =|last exit code ='
  metrics_ok='0'
  for attempt in 1 2 3 4 5 6; do
    if curl -fsS "http://127.0.0.1:$((base_metrics_port + instance_index))/metrics" \
      | grep -q '^# HELP backtest_job_runner_claim_total '; then
      metrics_ok='1'
      break
    fi
    sleep 5
  done
  test "${metrics_ok}" = '1'
done
```

Failure interpretation:
- `jobs.enabled=false`: worker fleet intentionally не materialize-ится и deploy не должен падать
  только из-за отсутствия `backtest-job-runner` services;
- install/bootstrap failure: `bootstrap_native_prod.sh` не установил launchd/materialized plists;
- reload failure: `reload_launchd_services.sh prod` не зарегистрировал ожидаемый fleet size;
- fleet cardinality mismatch: registered/live services не совпали с `worker_processes`, значит
  rollout incorrect даже при живом `com.roehub.api`;
- immediate error exit: `launchctl print` показывает `last exit code != 0` для worker service;
- liveness/observability failure: service label есть, но нет live pid или не отвечает `metrics endpoint`;
- API `401` сам по себе не заменяет worker smoke и не делает deploy green.

Schema bootstrap (identity SQL + Alembic):

```bash
cd /opt/roehub/app
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
/opt/roehub/app/.venv/bin/python -m apps.migrations.bootstrap_main
```

Быстрые smoke-проверки:

```bash
bash scripts/macos/smoke_prod.sh
bash scripts/macos/smoke_test.sh
```

ClickHouse partition dedup (safe full-history run):

```bash
bash scripts/macos/clickhouse_partition_dedup.sh all
```

Подробный runbook:

- `docs/runbooks/clickhouse-partition-dedup.md`
- `docs/runbooks/clickhouse-memory-profiles.md`

## Hourly page cache purge

Опциональный root `LaunchDaemon` для периодического сброса macOS disk/page cache:

- label: `com.roehub.purge-hourly`
- plist path: `/Library/LaunchDaemons/com.roehub.purge-hourly.plist`
- command: `/usr/sbin/purge`
- cadence: каждые `3600` секунд

Важно:

- `purge` очищает disk/page cache, но не лечит `malloc`/heap leaks;
- после тяжёлых чтений `ClickHouse` cache снова прогреется;
- использовать только как ops workaround, а не как замену root-cause fix.

Включение:

```bash
sudo tee /Library/LaunchDaemons/com.roehub.purge-hourly.plist >/dev/null <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.roehub.purge-hourly</string>

    <key>ProgramArguments</key>
    <array>
      <string>/usr/sbin/purge</string>
    </array>

    <key>StartInterval</key>
    <integer>3600</integer>

    <key>RunAtLoad</key>
    <false/>

    <key>StandardOutPath</key>
    <string>/var/log/roehub-purge.log</string>

    <key>StandardErrorPath</key>
    <string>/var/log/roehub-purge.err.log</string>
  </dict>
</plist>
EOF

sudo chown root:wheel /Library/LaunchDaemons/com.roehub.purge-hourly.plist
sudo chmod 644 /Library/LaunchDaemons/com.roehub.purge-hourly.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/com.roehub.purge-hourly.plist
```

Проверка статуса:

```bash
sudo launchctl print system/com.roehub.purge-hourly
```

Ожидаемые признаки:

- `path = /Library/LaunchDaemons/com.roehub.purge-hourly.plist`
- `program = /usr/sbin/purge`
- `run interval = 3600 seconds`
- после первого запуска `runs > 0`

Ручной trigger для smoke-проверки:

```bash
sudo launchctl kickstart -k system/com.roehub.purge-hourly
sudo launchctl print system/com.roehub.purge-hourly | grep -E 'runs =|last exit code =|state ='
sudo tail -n 50 /var/log/roehub-purge.log /var/log/roehub-purge.err.log
```

Отключение:

```bash
sudo launchctl bootout system /Library/LaunchDaemons/com.roehub.purge-hourly.plist
sudo rm /Library/LaunchDaemons/com.roehub.purge-hourly.plist
```

Настройка `tailscale serve`:

```bash
bash scripts/macos/configure_tailscale_serve.sh
```

Проверка/восстановление env symlink:

```bash
sudo install -d -m 755 /etc/roehub
sudo ln -sfn /Users/daniildegtyarev/.config/roehub/roehub.env /etc/roehub/roehub.env
```

Ручной publish backtest artifacts:

```bash
cd /opt/roehub/app
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
/opt/roehub/app/.venv/bin/python -m apps.cli.main.main backtest-artifact-publish --config /opt/roehub/app/configs/prod/backtest_artifacts.yaml --exchange binance --market-type spot --symbol BTCUSDT
/opt/roehub/app/.venv/bin/python -m apps.cli.main.main backtest-artifact-publish --config /opt/roehub/app/configs/prod/backtest_artifacts.yaml --exchange binance --market-type spot --symbol BTCUSDT --full-rebuild
```

Рекомендуемый вариант для первого bootstrap/publish с отдельным логом:

```bash
cd /opt/roehub/app
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
/opt/roehub/app/.venv/bin/python -m apps.cli.main.main backtest-artifact-publish \
  --config /opt/roehub/app/configs/prod/backtest_artifacts.yaml \
  --exchange binance \
  --market-type spot \
  --symbol BTCUSDT \
  --full-rebuild \
  2>&1 | tee /tmp/backtest-artifact-publish-BTCUSDT.log
```

Dedicated scheduled publisher service:

- service label: `com.roehub.backtest-artifact-publisher`
- timezone-explicit cadence: daily at `03:05 Europe/Moscow`
- metrics endpoint: `http://127.0.0.1:9203/metrics`
- host lock file: `/opt/roehub/state/backtest_artifacts/v2/.backtest_artifact_publisher.lock`
- universe source-of-truth: `market_data.ref_instruments` with enabled+tradable rows
- key health metrics:
  - `backtest_artifact_publish_runs_total{status}`
  - `backtest_artifact_publish_symbols_total{status}`
  - `backtest_artifact_publish_blocked_total{reason}`
  - `backtest_artifact_publish_last_success_unixtime`
  - `backtest_artifact_tail_rebuild_bars_total{stage}`
- manual CLI is for one explicit symbol root; full enabled+tradable universe is executed by the
  scheduled service in the next `03:05 Europe/Moscow` window
- `validation_budgets.max_hit_times_cells_full_rebuild` covers bootstrap/explicit full rebuild,
  while steady-state scheduler runs keep `validation_budgets.max_hit_times_cells`
- current prod/test/dev configs use `signal_artifacts: all_supported_v1`, so one publish must
  materialize the full signal registry for every symbol root
- `configs/<env>/indicators.yaml -> compute.numba.max_compute_bytes_total` remains a public/runtime
  guard only; artifact publisher wiring bypasses this ceiling with dedicated offline compute
  config, so full-registry signal precompute must not fail with `ComputeBudgetExceeded`
- indicators-config resolution for artifact precompute is deterministic:
  `ROEHUB_INDICATORS_CONFIG` -> sibling of the explicit artifact config
  (for example `/opt/roehub/app/configs/prod/backtest_artifacts.yaml` ->
  `/opt/roehub/app/configs/prod/indicators.yaml`) -> `ROEHUB_ENV` -> final `dev` default
- this means manual CLI and the scheduled publisher do not require a separate
  `ROEHUB_ENV=prod` export when they already run against the explicit production artifact config;
  only `ROEHUB_INDICATORS_CONFIG` can intentionally override that match
- R13-01 narrows only the heaviest non-`ma.*` signal defaults in `configs/<env>/indicators.yaml`;
  operator expectation should be smaller signal matrices for `momentum.trix`, `momentum.stoch`,
  `trend.adx`, `volatility.hv`, and the other targeted variant-heavy families without changing
  `all_supported_v1` coverage or `ma.*` defaults
- R12 target execution model for Mac Studio is bounded and `timeframe-scoped`:
  - one open `current_timeframe` session at a time;
  - bounded `signal_worker_processes` inside that session;
  - per-worker budget via `signal_worker_memory_budget_bytes`;
  - eager disk writes via `np.memmap`, not delayed giant in-memory signal tensors.
- Practical operator rule:
  - if logs suggest multiple simultaneous `current_timeframe` sessions or no chunk progress while
    memory pressure grows, treat that as execution-policy drift.

## Manual health checks

Production:

```bash
brew services list
launchctl list | grep -E "roehub|clickhouse|blackbox|redis-exporter|postgres-exporter|actions.runner|tailscale"
curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9100
curl -I http://127.0.0.1:9115
curl -I http://127.0.0.1:9116
curl -I http://127.0.0.1:9121
curl -I http://127.0.0.1:9187
curl -i http://127.0.0.1:8000/auth/current-user
curl -fsS http://127.0.0.1:9201/metrics | head
curl -fsS http://127.0.0.1:9202/metrics | head
curl -fsS http://127.0.0.1:9203/metrics | head
for service_label in $(launchctl list | awk '$3 ~ /^com\\.roehub\\.backtest-job-runner\\.[0-9]+$/ {print $3}'); do
  instance_index="${service_label##*.}"
  curl -fsS "http://127.0.0.1:$((9204 + instance_index))/metrics" | grep -m1 '^# HELP backtest_job_runner_claim_total '
done
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT 1"
redis-cli -h 127.0.0.1 -p 6379 PING
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
PGPASSWORD="${POSTGRES_PASSWORD}" psql -h 127.0.0.1 -p 5432 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -Atqc "SELECT to_regclass('public.identity_users'), to_regclass('public.identity_exchange_keys'), to_regclass('public.alembic_version')"
tailscale serve status
```

Примечание по `com.roehub.tailscale-runtime`: это periodic one-shot job (`StartInterval`),
поэтому `state = not running` между запусками — нормально; важно, чтобы `last exit code = 0`.

Test:

```bash
pg_isready -h 127.0.0.1 -p 15433
redis-cli -h 127.0.0.1 -p 16379 PING
curl -I http://127.0.0.1:18124/ping
curl -i http://127.0.0.1:18000/auth/current-user
curl -I http://127.0.0.1:13000
curl -I http://127.0.0.1:19090
curl -fsS http://127.0.0.1:19201/metrics | head
curl -fsS http://127.0.0.1:19202/metrics | head
curl -fsS http://127.0.0.1:19203/metrics | head
for service_label in $(launchctl list | awk '$3 ~ /^com\\.roehub\\.test\\.backtest-job-runner\\.[0-9]+$/ {print $3}'); do
  instance_index="${service_label##*.}"
  curl -fsS "http://127.0.0.1:$((19204 + instance_index))/metrics" | grep -m1 '^# HELP backtest_job_runner_claim_total '
done
```

Проверка Prometheus targets после reboot:

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, scrapeUrl: .scrapeUrl}'
```

## Logs and diagnostics

Homebrew services logs (через launchctl):

```bash
brew services info postgresql@16
brew services info redis
brew services info grafana
brew services info prometheus
brew services info node_exporter
```

Custom service logs:

```bash
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/api.out.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/api.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/market-data-ws-worker.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/market-data-scheduler.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/backtest-artifact-publisher.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/blackbox-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/redis-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/postgres-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/tailscale-runtime.err.log
```

Manual CLI progress diagnostics:

```bash
tail -n 200 /tmp/backtest-artifact-publish-BTCUSDT.log
rg "event=artifact_precompute_(stage_started|stage_finished|completed|failed)" /tmp/backtest-artifact-publish-BTCUSDT.log
rg "current_timeframe|current_indicator_id|chunk_index|chunk_count|completed_chunks_total|completed_indicators_total|rewritten_tail_bars" /tmp/backtest-artifact-publish-BTCUSDT.log
find /opt/roehub/state/backtest_artifacts/v2/binance/spot/BTCUSDT \( -name current.yaml -o -name manifest.yaml -o -name '*.npy' \) | head -100
```

Важно:

- `http://127.0.0.1:9203/metrics` показывает scheduled publisher service, а не разовый manual CLI;
- если CLI идёт долго, проверяйте именно `/tmp/backtest-artifact-publish-BTCUSDT.log`, а не только
  `backtest-artifact-publisher.err.log`.

Проверка active launch agents:

```bash
launchctl list | grep -E "com.roehub\.(api|market-data|backtest-job-runner|clickhouse|blackbox|test\.)"
```

## Frequent failure modes

`com.roehub.clickhouse` падает при старте с `last exit code = 91`:

- проверьте `/opt/roehub/config/clickhouse.users.roehub.xml`;
- у пользователя `default` должен быть `no_password`;
- проверьте внутренний лог: `/opt/roehub/clickhouse/logs/clickhouse-server.err.log`.

`grafana` в `brew services` показывает `error 78`:

- проверьте права на `/opt/homebrew/var/lib` и `/opt/homebrew/var/lib/grafana`;
- `daniildegtyarev` должен иметь execute/read/write доступ к этим путям;
- проверьте логи: `/opt/homebrew/var/log/grafana-stderr.log`.

`market-data-*` падают с `PermissionError: /etc/roehub/roehub.env`:

- восстановите symlink `/etc/roehub/roehub.env` -> user env;
- перезагрузите services: `bash scripts/macos/reload_launchd_services.sh prod`.

`backtest-artifact-publisher` показывает `backtest_artifact_publish_blocked_total{reason="lock_held"}` или не публикует новые успехи:

- убедитесь, что нет второго ручного процесса `python -m apps.scheduler.backtest_artifact_publisher.main.main`;
- проверьте lock file `/opt/roehub/state/backtest_artifacts/v2/.backtest_artifact_publisher.lock`;
- проверьте `launchctl list | grep backtest-artifact-publisher` и лог
  `/Users/daniildegtyarev/Library/Logs/roehub/backtest-artifact-publisher.err.log`;
- если сервис был остановлен после `03:05 Europe/Moscow`, перезапустите его до следующего окна или
  выполните ручной publish нужного symbol root.

`backtest-job-runner` fleet не совпадает с `backtest.jobs.worker_processes` или один из instance быстро выходит:

- если `backtest.jobs.enabled=false`, это ожидаемое состояние и fleet не обязателен;
- проверьте `configs/prod/backtest.yaml` или `configs/test/backtest.yaml` и значение
  `jobs.enabled` и `worker_processes`;
- перерендирите и перезагрузите fleet: `bash scripts/macos/reload_launchd_services.sh prod`
  или `bash scripts/macos/reload_launchd_services.sh test`;
- проверьте `launchctl list | grep backtest-job-runner`;
- проверьте instance-specific logs в `/Users/daniildegtyarev/Library/Logs/roehub/` с suffix
  `.0`, `.1`, ...;
- проверьте доступность per-instance metrics endpoint на `9204 + instance_index` для prod или
  `19204 + instance_index` для test.

`backtest-artifact-publisher` растит `backtest_artifact_publish_blocked_total{reason="inactive_slot_pinned"}`:

- проверьте активные `background_auto` runs и already-persisted compatibility rows c
  `background_manual_legacy`, которые ещё pin'ят inactive slot;
- не удаляйте `current.yaml` и не переписывайте slot вручную;
- дождитесь terminal transition run'ов или отмените stuck job штатным способом.

`backtest-artifact-publisher` растит `backtest_artifact_publish_blocked_total{reason="validation_failed"}` или
`backtest_artifact_publish_runs_total{status="unexpected_error"}`:

- откройте `/Users/daniildegtyarev/Library/Logs/roehub/backtest-artifact-publisher.err.log`;
- проверьте доступность ClickHouse, `STRATEGY_PG_DSN`, и содержимое
  `/opt/roehub/config/prometheus.prod.yml`;
- проверьте structured progress fields `current_timeframe`, `current_indicator_id`,
  `chunk_index/chunk_count`, `completed_chunks_total/completed_indicators_total` и сравните их с
  memory/disk pressure on host;
- если сервис снова выглядит как giant tensor-first execution без bounded chunk progress,
  уменьшайте future `signal_worker_processes` / chunk sizing в `execution_policy`, а не пытайтесь
  лечить это ручной чисткой slot contents.
- при необходимости выполните один ручной `backtest-artifact-publish` для конкретного symbol root,
  чтобы локализовать проблему на одном инструменте.

`api` падает с `STRATEGY_PG_DSN is required`:

- добавьте `STRATEGY_PG_DSN`, `IDENTITY_PG_DSN`, `POSTGRES_DSN` в prod env;
- перезагрузите `com.roehub.api`.

`com.roehub.api` не стартует после включения `IDENTITY_FAIL_FAST=true` с ошибкой про `KEYCLOAK_*`:

- проверьте, что в env заданы:
  - `KEYCLOAK_BASE_URL`
  - `KEYCLOAK_REALM`
  - `KEYCLOAK_CLIENT_ID`
  - `KEYCLOAK_CLIENT_SECRET`
  - `KEYCLOAK_REDIRECT_URI`
  - `KEYCLOAK_LOGOUT_REDIRECT_URI`
  - `IDENTITY_SESSION_IDLE_TTL_SECONDS`
  - `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`
- проверьте формат URI и соответствие redirect URI настройкам Keycloak client;
- перезагрузите `com.roehub.api`.
- Telegram notifier secret не является частью API-auth fail-fast checks и не нужен для запуска
  `/auth/login`/`/auth/current-user` после Keycloak cutover.

`/auth/login` или `/auth/current-user` возвращают `5xx` после перехода на Keycloak:

- проверьте доступность Keycloak realm endpoint от API host;
- если задана явная endpoint-конфигурация, проверьте согласованность:
  - `KEYCLOAK_AUTH_URL`
  - `KEYCLOAK_TOKEN_URL`
  - `KEYCLOAK_END_SESSION_URL`
  - `KEYCLOAK_INTROSPECTION_URL`
- если явные endpoint не заданы, проверьте корректность derive из `KEYCLOAK_BASE_URL` + `KEYCLOAK_REALM`;
- проверьте session policy:
  - `IDENTITY_SESSION_COOKIE_NAME`
  - `IDENTITY_SESSION_IDLE_TTL_SECONDS`
  - `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`
- проверьте `api.err.log` и перезагрузите `com.roehub.api`.

Schema bootstrap падает с `Missing Alembic config file: /opt/roehub/app/alembic.ini`:

- доставьте `alembic.ini` в `/opt/roehub/app/alembic.ini`;
- проверьте, что deploy workflow копирует `alembic.ini` вместе с `apps/`, `src/`, `alembic/`, `migrations/`;
- повторите `/opt/roehub/app/.venv/bin/python -m apps.migrations.bootstrap_main`.

`postgres-exporter`/`redis-exporter` не стартуют:

- проверьте наличие бинарей: `/opt/roehub/bin/postgres_exporter`, `/opt/roehub/bin/redis_exporter`;
- переустановите prerequisites: `bash scripts/macos/install_native_backend_prereqs.sh`;
- проверьте exporter-коннекторы в env (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `ROEHUB_REDIS_PASSWORD`).

`clickhouse-exporter` в `down`/`connection refused` на `127.0.0.1:9116`:

- проверьте статус launchd: `launchctl print gui/$(id -u)/com.roehub.clickhouse-exporter`;
- проверьте ошибки: `tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse-exporter.err.log`;
- если есть `TypeError: unhashable type: 'ClickHouseExporterCollector'`, обновите код (`git pull`) и перезапустите сервис;
- проверьте модуль вручную: `/opt/roehub/app/.venv/bin/python -m apps.monitoring.clickhouse_exporter --host 127.0.0.1 --port 9116 --scrape-uri http://127.0.0.1:8123/ --database market_data --user ${CLICKHOUSE_USER:-roe} --password ${CLICKHOUSE_PASSWORD:-}`;
- после исправления перезапустите: `bash scripts/macos/reload_launchd_services.sh prod`.

`tailscale` после reboot не входит в `Running`, serve недоступен:

- проверьте launch agent: `launchctl print gui/$(id -u)/com.roehub.tailscale-runtime`;
- проверьте ошибку рантайма: `tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/tailscale-runtime.err.log`;
- проверьте backend state: `tailscale status --json | jq -r '.BackendState'`;
- вручную поднимите соединение: `tailscale up`;
- примените serve mapping: `bash scripts/macos/configure_tailscale_serve.sh`;
- перезапустите runtime agent: `bash scripts/macos/reload_launchd_services.sh prod`.

`com.roehub.tailscale-runtime` показывает `state = not running`, но backend уже `Running`:

- это штатно для one-shot periodic job;
- если при этом `tailscale status --json | jq -r '.BackendState'` = `Running` и `tailscale serve status` содержит все нужные mapping, действия не требуются;
- если `last exit code != 0`, посмотрите `tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/tailscale-runtime.err.log`.
