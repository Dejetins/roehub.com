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
- `com.roehub.clickhouse` (`launchd`, `127.0.0.1:8123/9000`)
- `com.roehub.blackbox-exporter` (`launchd`, `127.0.0.1:9115`)
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

Быстрые smoke-проверки:

```bash
bash scripts/macos/smoke_prod.sh
bash scripts/macos/smoke_test.sh
```

Настройка `tailscale serve`:

```bash
bash scripts/macos/configure_tailscale_serve.sh
```

## Manual health checks

Production:

```bash
brew services list
launchctl list | grep -E "roehub|clickhouse|blackbox|actions.runner|tailscale"
curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9115
curl -i http://127.0.0.1:8000/auth/current-user
curl -fsS http://127.0.0.1:9201/metrics | head
curl -fsS http://127.0.0.1:9202/metrics | head
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT 1"
redis-cli -h 127.0.0.1 -p 6379 PING
tailscale serve status
```

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

## Logs and diagnostics

Homebrew services logs (через launchctl):

```bash
brew services info postgresql@16
brew services info redis
brew services info grafana
brew services info prometheus
```

Custom service logs:

```bash
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/api.out.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/api.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/market-data-ws-worker.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/market-data-scheduler.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse.err.log
tail -n 200 /Users/daniildegtyarev/Library/Logs/roehub/blackbox-exporter.err.log
```

Проверка active launch agents:

```bash
launchctl list | grep -E "com.roehub\.(api|market-data|clickhouse|blackbox|test\.)"
```
