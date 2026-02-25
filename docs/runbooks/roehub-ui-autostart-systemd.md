# Roehub UI Autostart (systemd) + Helper Commands

This runbook documents how to:

- Ensure Roehub UI containers (`api`, `web`, `gateway`) start automatically after server reboot.
- Provide two simple helper commands:
  - `roehub_up` (start/update containers)
  - `roehub_restart` (restart containers)

Applies to the server layout used in this repository:

- Compose file: `/opt/roehub/docker-compose.yml`
- Environment file: `/etc/roehub/roehub.env`
- UI profile: `--profile ui`
- Build context: `/opt/roehub/market-data-src`

Related:
- `infra/docker/docker-compose.yml` (service definitions)
- `docs/runbooks/web-ui-gateway-same-origin.md`

## 1) Create systemd unit

Create the unit file:

```bash
sudoedit /etc/systemd/system/roehub-ui.service
```

Paste the following content and save:

```ini
[Unit]
Description=Roehub UI stack (api/web/gateway) via Docker Compose
Wants=network-online.target
After=network-online.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/roehub
Environment=COMPOSE_PROJECT_NAME=roehub
Environment=MARKET_DATA_BUILD_CONTEXT=/opt/roehub/market-data-src
Environment=MARKET_DATA_DOCKERFILE=infra/docker/Dockerfile.market_data
ExecStart=/usr/bin/docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env --profile ui up -d --remove-orphans api web gateway
ExecStop=/usr/bin/docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env --profile ui stop api web gateway
TimeoutStartSec=300
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now roehub-ui.service
systemctl status roehub-ui.service --no-pager
```

Logs:

```bash
journalctl -u roehub-ui.service -b --no-pager
```

## 2) Create helper commands

### 2.1) roehub_up

```bash
sudoedit /usr/local/bin/roehub_up
```

Paste:

```bash
#!/usr/bin/env bash
set -euo pipefail

export COMPOSE_PROJECT_NAME=roehub
export MARKET_DATA_BUILD_CONTEXT=/opt/roehub/market-data-src
export MARKET_DATA_DOCKERFILE=infra/docker/Dockerfile.market_data

exec docker compose -f /opt/roehub/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui \
  up -d --remove-orphans api web gateway
```

### 2.2) roehub_restart

```bash
sudoedit /usr/local/bin/roehub_restart
```

Paste:

```bash
#!/usr/bin/env bash
set -euo pipefail

export COMPOSE_PROJECT_NAME=roehub

exec docker compose -f /opt/roehub/docker-compose.yml \
  --env-file /etc/roehub/roehub.env \
  --profile ui \
  restart api web gateway
```

Make both scripts executable:

```bash
sudo chmod +x /usr/local/bin/roehub_up /usr/local/bin/roehub_restart
```

Verify:

```bash
command -v roehub_up roehub_restart
roehub_up
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E 'roehub-(api|web|gateway)-' || true
```

## 3) Common operations

Check bound ports:

```bash
sudo ss -ltnp | egrep ':(8080|80|443)\s'
```

Rebuild after code update (manual):

```bash
cd /opt/roehub/market-data-src && git pull --ff-only

COMPOSE_PROJECT_NAME=roehub \
MARKET_DATA_BUILD_CONTEXT=/opt/roehub/market-data-src \
MARKET_DATA_DOCKERFILE=infra/docker/Dockerfile.market_data \
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env --profile ui up -d --build --remove-orphans api web gateway
```

Stop UI stack:

```bash
sudo systemctl stop roehub-ui.service
```

Start UI stack:

```bash
sudo systemctl start roehub-ui.service
```
