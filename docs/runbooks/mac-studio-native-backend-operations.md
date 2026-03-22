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
- host config root: `/opt/roehub/config`
- host binaries: `/opt/roehub/bin`, `/opt/clickhouse/clickhouse`
- prod env: `/Users/daniildegtyarev/.config/roehub/roehub.env`
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
- `com.roehub.api` (`launchd`, `127.0.0.1:8000`)
- `com.roehub.market-data-ws-worker` (`launchd`, metrics `127.0.0.1:9201`)
- `com.roehub.market-data-scheduler` (`launchd`, metrics `127.0.0.1:9202`)

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

Настройка `tailscale serve`:

```bash
bash scripts/macos/configure_tailscale_serve.sh
```

Проверка/восстановление env symlink:

```bash
sudo install -d -m 755 /etc/roehub
sudo ln -sfn /Users/daniildegtyarev/.config/roehub/roehub.env /etc/roehub/roehub.env
```

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
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/blackbox-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/redis-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/postgres-exporter.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/tailscale-runtime.err.log
```

Проверка active launch agents:

```bash
launchctl list | grep -E "com.roehub\.(api|market-data|clickhouse|blackbox|test\.)"
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

`api` падает с `STRATEGY_PG_DSN is required`:

- добавьте `STRATEGY_PG_DSN`, `IDENTITY_PG_DSN`, `POSTGRES_DSN` в prod env;
- перезагрузите `com.roehub.api`.

`/auth/telegram/login` падает с `500` и в логе есть `psycopg.errors.UndefinedTable: relation "identity_users" does not exist`:

- выполните schema bootstrap: `/opt/roehub/app/.venv/bin/python -m apps.migrations.bootstrap_main`;
- проверьте таблицы: `to_regclass('public.identity_users')`, `to_regclass('public.identity_exchange_keys')`, `to_regclass('public.alembic_version')`;
- перезагрузите `com.roehub.api`.

`/auth/telegram/login` отвечает `422` с `invalid_telegram_payload` и текстом `PostgresIdentityUserRepository cannot map user row`:

- обновите backend до версии с UTC-normalization в `identity` postgres repositories;
- перезапустите `com.roehub.api`;
- повторите login-проверку.

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
