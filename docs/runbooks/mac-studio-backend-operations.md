# Mac Studio Backend Operations

Операционный ранбук для backend runtime на `Mac Studio` в текущей production topology.

## Runtime owner

- backend runtime владеет и запускает текущий macOS user `daniildegtyarev`;
- `Colima`, backend compose stack, `tailscale serve`, локальные runtime scripts и monitoring
  больше не завязаны на отдельного `deploy` пользователя;
- production web ingress и public TLS по-прежнему живут на `VPS`.

## Paths and environment

- backend compose: `/opt/roehub/docker-compose.backend.yml`
- runtime env file: `/Users/daniildegtyarev/.config/roehub/roehub.env`
- market-data source bundle: `/opt/roehub/market-data-src`

Базовые переменные для ручных операций:

```bash
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export ROEHUB_APP_IMAGE=ghcr.io/dejetins/roehub-app:main
export ROEHUB_ENV_FILE=/Users/daniildegtyarev/.config/roehub/roehub.env
```

## Start / restart all backend services

Полный старт backend stack:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" up -d
```

Полный restart backend stack:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" restart
```

Полная остановка backend stack:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" stop
```

Статус всего stack:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" ps
```

## Start / restart individual services

Stateful services:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" up -d postgres clickhouse redis
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" restart postgres clickhouse redis
```

Monitoring:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" up -d grafana prometheus blackbox
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" restart grafana prometheus blackbox
```

App layer:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" up -d db-bootstrap api market-data-ws-worker market-data-scheduler
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" restart api market-data-ws-worker market-data-scheduler
```

Только market-data workers:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" restart market-data-ws-worker market-data-scheduler
```

## Logs

```bash
docker logs --tail=100 roehub-postgres-1
docker logs --tail=100 roehub-clickhouse-1
docker logs --tail=100 roehub-api-1
docker logs --tail=100 roehub-market-data-ws-worker-1
docker logs --tail=100 roehub-market-data-scheduler-1
docker logs --tail=100 grafana
docker logs --tail=100 prometheus
docker logs --tail=100 blackbox
```

Follow logs:

```bash
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" logs -f api
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" logs -f market-data-ws-worker
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" logs -f market-data-scheduler
```

## Health checks

Local host checks on `Mac Studio`:

```bash
curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9115
curl -i http://127.0.0.1:8000/auth/current-user
```

Expected:

- `Grafana` -> `200` or `302 /login`
- `Prometheus` -> `200` or `405` on `HEAD`
- `Blackbox` -> `200`
- `API /auth/current-user` -> `401`

## ClickHouse DDL

Apply DDL after fresh runtime bootstrap:

```bash
set -a
source "$ROEHUB_ENV_FILE"
set +a

docker exec -i roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --multiquery < /opt/roehub/market-data-src/migrations/clickhouse/market_data_ddl.sql
```

## Tailscale Serve endpoints

Persistent tailnet-only endpoints:

- `Grafana`: `https://macstudio-daniil.tail0ebbbc.ts.net:3443/`
- `API`: `https://macstudio-daniil.tail0ebbbc.ts.net/`
- `Postgres`: `macstudio-daniil.tail0ebbbc.ts.net:15432`
- `ClickHouse HTTP`: `macstudio-daniil.tail0ebbbc.ts.net:18123`
- `ClickHouse native`: `macstudio-daniil.tail0ebbbc.ts.net:19000`

Status:

```bash
tailscale serve status
```

## DBeaver

PostgreSQL:

- Host: `macstudio-daniil.tail0ebbbc.ts.net`
- Port: `15432`
- Database: `roehub` or `POSTGRES_DB`
- User: `roehub` or `POSTGRES_USER`
- Password: `POSTGRES_PASSWORD`
- SSL: `disable`
- SSH tunnel: `off`

ClickHouse:

- Host: `macstudio-daniil.tail0ebbbc.ts.net`
- Port: `18123`
- Database: `market_data` or `CH_DATABASE`
- User: `CLICKHOUSE_USER`
- Password: `CLICKHOUSE_PASSWORD`
- SSL: `disable`
- SSH tunnel: `off`

## Autostart model

Current recommended autostart on `Mac Studio`:

- `brew services start colima`
- user `LaunchAgent` for `Tailscale.app`
- user `LaunchAgent` for `tailscale serve`
- user `LaunchAgent` for backend compose startup script

Check startup components:

```bash
brew services list | grep colima
launchctl list | grep roehub
tail -n 100 "$HOME/Library/Logs/roehub/backend.out.log"
tail -n 100 "$HOME/Library/Logs/roehub/backend.err.log"
tail -n 100 "$HOME/Library/Logs/com.roehub.tailscale-serve.out.log"
tail -n 100 "$HOME/Library/Logs/com.roehub.tailscale-serve.err.log"
```
