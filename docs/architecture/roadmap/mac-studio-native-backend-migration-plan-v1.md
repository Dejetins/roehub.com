# План полного переезда backend Roehub на native macOS (Mac Studio) v1

План фиксирует пошаговый переход backend Roehub с текущего Docker/Colima runtime на полностью native runtime на `Mac Studio`: Homebrew там, где это поддерживается, `uv + launchd` для Python-процессов, отдельный native `ClickHouse`, отдельный native `blackbox_exporter`, отдельная test-среда без Docker и полный вывод из эксплуатации текущего container stack.

Референсы:
- `docs/runbooks/mac-studio-backend-operations.md`
- `docs/runbooks/prod-migration-linux-to-mac-studio.md`
- `docs/runbooks/market-data-autonomous-docker.md`
- `docs/runbooks/strategy-live-worker.md`
- `docs/runbooks/backtest-job-runner.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `infra/docker/docker-compose.backend.yml`
- `.github/workflows/deploy-backend.yml`
- `.github/workflows/deploy-web.yml`
- `.github/workflows/publish-app-image.yml`
- `infra/monitoring/monitoring/prometheus/prometheus.yml`
- `infra/monitoring/monitoring/blackbox/blackbox.yml`
- `configs/prod/market_data.yaml`
- `configs/prod/strategy.yaml`
- `configs/prod/backtest.yaml`
- `scripts/ops/optimize_canonical_partitions.sh`

---

## Зафиксированные решения

- Целевой backend runtime на `Mac Studio` полностью уходит от `Colima`, `Docker` и Docker volumes.
- Публичный edge не меняется: `VPS` остается единственным публичным ingress/TLS хостом; `Mac Studio` остается private backend host.
- Python-сервисы backend переходят на `uv` и запускаются как user-level `launchd` services под пользователем `daniildegtyarev`.
- `Postgres`, `Redis`, `Grafana`, `Prometheus` ставятся из `Homebrew`; для production используются нативные host-services, для test-контура - те же нативные бинарники в отдельных `launchd` service definitions с отдельными портами и data dirs.
- `ClickHouse` не остается в Docker и не переводится на `brew services`; используется отдельный native binary/service с собственным layout в `/opt/roehub`.
- `blackbox_exporter` не остается в Docker и не переводится на `brew services`; используется отдельный native binary/service с собственным `launchd` label.
- Текущий Docker stack после подтвержденных backup/export и успешного native smoke выключается, удаляется, а Docker volumes удаляются.
- `self-hosted` GitHub Actions runner остается на `Mac Studio` под тем же пользователем `daniildegtyarev`.
- В рамках этого плана не поднимаем и не валидируем:
  - `strategy-live-worker`
  - `backtest-job-runner`
- Test environment создается на том же `Mac Studio`, но в отдельном native runtime-контуре с отдельными портами, отдельными data dirs, отдельным env-файлом и без импорта production данных.

---

## Подтвержденные факты по текущей машине

Эти факты уже подтверждены на текущем `Mac Studio` по SSH:

- host: `Mac Studio`, `Apple M2 Max`, `64 GB RAM`, macOS `15.7.4`
- свободное место на системном диске: примерно `753 GiB`
- runtime owner: `daniildegtyarev`
- `autoLoginUser = daniildegtyarev`
- `FileVault is Off`
- GUI session доступна, user `LaunchAgents` уже используются
- текущий runner зарегистрирован в GitHub как `mac-studio-prod`, статус `online`, labels:
  - `self-hosted`
  - `macOS`
  - `ARM64`
  - `roehub`
  - `prod`
  - `mac-studio`
- текущий backend сейчас еще containerized и идет через `Colima`:
  - `postgres`
  - `clickhouse`
  - `redis`
  - `api`
  - `market-data-ws-worker`
  - `market-data-scheduler`
  - `grafana`
  - `prometheus`
  - `blackbox`
- текущие Tailscale Serve endpoints уже настроены для prod API/Grafana/Postgres/ClickHouse.

Важно:

- утверждение про `LaunchAgent` и `LimitLoadToSessionType=Aqua` взято не из предположения, а из фактических файлов на `Mac Studio`:
  - `~/Library/LaunchAgents/com.roehub.backend.plist`
  - `~/Library/LaunchAgents/com.roehub.tailscale-serve.plist`

---

## Scope и границы

### Входит в этот план

- production native backend на `Mac Studio`
- production native monitoring на `Mac Studio`
- self-hosted runner на `Mac Studio`
- test native backend environment на том же `Mac Studio`
- полный cleanup текущего Docker/Colima runtime на `Mac Studio`
- обновление repo/workflows/docs под native runtime

### Не входит в этот план

- `strategy-live-worker`
- `backtest-job-runner`
- перенос `web` с `VPS`
- отказ от `GHCR` для `web`
- публичный ingress на `Mac Studio`

---

## Целевой runtime после переезда

### Production runtime

| Сервис | Target install/runtime | Управление | Локальные порты | Data/config path |
|---|---|---|---|---|
| `postgres` | `Homebrew postgresql@16` | `brew services` | `127.0.0.1:5432` | `/opt/homebrew/var/postgresql@16` |
| `redis` | `Homebrew redis` | `brew services` | `127.0.0.1:6379` | `/opt/homebrew/etc/redis.conf`, `/opt/homebrew/var/db/redis` |
| `clickhouse` | native binary | `launchd` | `127.0.0.1:8123`, `127.0.0.1:9000` | `/opt/roehub/clickhouse` |
| `grafana` | `Homebrew grafana` | `brew services` | `127.0.0.1:3000` | `/opt/homebrew/etc/grafana`, `/opt/homebrew/var/lib/grafana` |
| `prometheus` | `Homebrew prometheus` | `brew services` | `127.0.0.1:9090` | `/opt/homebrew/etc/prometheus.args`, `/opt/homebrew/var/prometheus`, `/opt/roehub/config/prometheus.prod.yml` |
| `blackbox_exporter` | native binary | `launchd` | `127.0.0.1:9115` | `/opt/roehub/bin`, `/opt/roehub/config/blackbox.yml` |
| `api` | repo checkout + `uv` | `launchd` | `127.0.0.1:8000` | `/opt/roehub/app`, `/Users/daniildegtyarev/.config/roehub/roehub.env` |
| `market-data-ws-worker` | repo checkout + `uv` | `launchd` | metrics `127.0.0.1:9201` | `/opt/roehub/app`, `/Users/daniildegtyarev/.config/roehub/roehub.env` |
| `market-data-scheduler` | repo checkout + `uv` | `launchd` | metrics `127.0.0.1:9202` | `/opt/roehub/app`, `/Users/daniildegtyarev/.config/roehub/roehub.env` |
| `tailscale serve` | existing Tailscale install | `launchd` | public tailnet-only mapping | `~/.local/bin/roehub_tailscale_serve_config` |
| `actions runner` | existing install | existing runner LaunchAgent | n/a | `/opt/actions-runner/roehub` |

### Test runtime

| Сервис | Target install/runtime | Управление | Локальные порты | Data/config path |
|---|---|---|---|---|
| `postgres-test` | `Homebrew postgresql@16` binary | `launchd` | `127.0.0.1:15433` | `/opt/roehub/test/postgresql` |
| `redis-test` | `Homebrew redis` binary | `launchd` | `127.0.0.1:16379` | `/opt/roehub/test/redis` |
| `clickhouse-test` | native binary | `launchd` | `127.0.0.1:18124`, `127.0.0.1:19001` | `/opt/roehub/test/clickhouse` |
| `grafana-test` | `Homebrew grafana` binary | `launchd` | `127.0.0.1:13000` | `/opt/roehub/test/grafana` |
| `prometheus-test` | `Homebrew prometheus` binary | `launchd` | `127.0.0.1:19090` | `/opt/roehub/test/prometheus`, `/opt/roehub/config/prometheus.test.yml` |
| `blackbox-exporter-test` | native binary | `launchd` | `127.0.0.1:19115` | `/opt/roehub/test/blackbox`, `/opt/roehub/config/blackbox.test.yml` |
| `api-test` | repo checkout + `uv` | `launchd` | `127.0.0.1:18000` | `/opt/roehub/app`, `/Users/daniildegtyarev/.config/roehub/roehub.test.env` |
| `market-data-ws-worker-test` | repo checkout + `uv` | `launchd` | metrics `127.0.0.1:19201` | `/opt/roehub/app`, `/Users/daniildegtyarev/.config/roehub/roehub.test.env` |
| `market-data-scheduler-test` | repo checkout + `uv` | `launchd` | metrics `127.0.0.1:19202` | `/opt/roehub/app`, `/Users/daniildegtyarev/.config/roehub/roehub.test.env` |

### Tailscale Serve после переезда

Production endpoints сохраняем:

- `https://macstudio-daniil.tail0ebbbc.ts.net/` -> prod API (`127.0.0.1:8000`)
- `https://macstudio-daniil.tail0ebbbc.ts.net:3443/` -> prod Grafana (`127.0.0.1:3000`)
- `macstudio-daniil.tail0ebbbc.ts.net:15432` -> prod Postgres (`127.0.0.1:5432`)
- `macstudio-daniil.tail0ebbbc.ts.net:18123` -> prod ClickHouse HTTP (`127.0.0.1:8123`)
- `macstudio-daniil.tail0ebbbc.ts.net:19000` -> prod ClickHouse native (`127.0.0.1:9000`)

Test endpoints добавляем:

- `https://macstudio-daniil.tail0ebbbc.ts.net:8443/` -> test API (`127.0.0.1:18000`)
- `https://macstudio-daniil.tail0ebbbc.ts.net:3444/` -> test Grafana (`127.0.0.1:13000`)
- `macstudio-daniil.tail0ebbbc.ts.net:25432` -> test Postgres (`127.0.0.1:15433`)
- `macstudio-daniil.tail0ebbbc.ts.net:28123` -> test ClickHouse HTTP (`127.0.0.1:18124`)
- `macstudio-daniil.tail0ebbbc.ts.net:29000` -> test ClickHouse native (`127.0.0.1:19001`)

---

## Данные и политика миграции

### Что переносим из текущего Docker runtime

- `Postgres` - переносим обязательно
- `ClickHouse` - переносим обязательно
- `Grafana` - переносим желательно, чтобы не потерять dashboards/state
- `roehub.env` - переносим обязательно

### Что не переносим как state

- `Redis` - поднимаем с нуля
- `Prometheus` TSDB - поднимаем с нуля
- Docker volumes - после успешного native cutover удаляем

### Почему так

- `Postgres` и `ClickHouse` содержат production данные и не могут быть просто отброшены.
- `Redis` в текущей topology используется как runtime bus/cache; чистый старт допустим.
- `Prometheus` history не критична для cutover; новый TSDB проще и безопаснее.
- `Grafana` dashboards и data source state удобнее сохранить, чем пересобирать вручную.

---

## Файлы в репозитории, которые должны быть добавлены или изменены

### Новые документы

- `docs/architecture/roadmap/mac-studio-native-backend-migration-plan-v1.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`

### Документы, которые нужно обновить

- `docs/runbooks/mac-studio-backend-operations.md`
  - перевести в архивный статус и сослаться на native runbook
- `docs/runbooks/prod-migration-linux-to-mac-studio.md`
  - убрать `Colima`/Docker backend как target state
- `docs/runbooks/market-data-autonomous-docker.md`
  - оставить Docker как legacy/local-only path
  - добавить ссылку на native production path
- `README.md`
  - убрать утверждение про единственный production `docker-compose.yml`
- `docs/architecture/README.md`
  - обновить индекс документов

### Workflow-файлы

- `.github/workflows/deploy-backend.yml`
  - полностью переписать с Docker deploy на native deploy
- `.github/workflows/deploy-web.yml`
  - проверить, что upstream API URL и smoke не требуют изменений
- `.github/workflows/publish-app-image.yml`
  - оставить для `web`, но явно убрать зависимость backend deploy от GHCR image

### Runtime/config файлы

- `configs/test/market_data.yaml`
  - добавить отдельный test runtime config для market-data
- `configs/test/whitelist.csv`
  - добавить отдельный test whitelist
- `infra/monitoring/monitoring/prometheus/prometheus.yml`
  - либо оставить как legacy Docker config, либо вынести production-native config в новые файлы
- `infra/monitoring/monitoring/blackbox/blackbox.yml`
  - переиспользовать или дублировать для native layout

### Новые infra/runtime файлы

- `infra/macos/launchd/com.roehub.api.plist`
- `infra/macos/launchd/com.roehub.market-data-ws-worker.plist`
- `infra/macos/launchd/com.roehub.market-data-scheduler.plist`
- `infra/macos/launchd/com.roehub.clickhouse.plist`
- `infra/macos/launchd/com.roehub.blackbox-exporter.plist`
- `infra/macos/launchd/com.roehub.test.postgres.plist`
- `infra/macos/launchd/com.roehub.test.redis.plist`
- `infra/macos/launchd/com.roehub.test.clickhouse.plist`
- `infra/macos/launchd/com.roehub.test.grafana.plist`
- `infra/macos/launchd/com.roehub.test.prometheus.plist`
- `infra/macos/launchd/com.roehub.test.blackbox-exporter.plist`
- `infra/macos/launchd/com.roehub.test.api.plist`
- `infra/macos/launchd/com.roehub.test.market-data-ws-worker.plist`
- `infra/macos/launchd/com.roehub.test.market-data-scheduler.plist`
- `infra/macos/prometheus/prometheus.prod.yml`
- `infra/macos/prometheus/prometheus.test.yml`
- `infra/macos/blackbox/blackbox.yml`
- `infra/macos/blackbox/blackbox.test.yml`
- `infra/macos/clickhouse/config.xml`
- `infra/macos/clickhouse/users.d/roehub.xml`
- `infra/macos/clickhouse/config.test.xml`

### Новые host bootstrap/deploy scripts

- `scripts/macos/install_native_backend_prereqs.sh`
- `scripts/macos/export_container_backups.sh`
- `scripts/macos/bootstrap_native_prod.sh`
- `scripts/macos/bootstrap_native_test.sh`
- `scripts/macos/reload_launchd_services.sh`
- `scripts/macos/smoke_prod.sh`
- `scripts/macos/smoke_test.sh`
- `scripts/macos/configure_tailscale_serve.sh`

### Существующие scripts, которые нужно обновить

- `scripts/ops/optimize_canonical_partitions.sh`
  - убрать Docker compose dependency
  - перейти на native `clickhouse-client`
  - сменить default env path на `/Users/daniildegtyarev/.config/roehub/roehub.env`

---

## Host-файлы и каталоги, которые должны появиться на Mac Studio

### Production

- `/opt/roehub/app`
- `/opt/roehub/bin`
- `/opt/roehub/config`
- `/opt/roehub/state/backups`
- `/opt/roehub/clickhouse`
- `/Users/daniildegtyarev/.config/roehub/roehub.env`
- `~/Library/LaunchAgents/com.roehub.api.plist`
- `~/Library/LaunchAgents/com.roehub.market-data-ws-worker.plist`
- `~/Library/LaunchAgents/com.roehub.market-data-scheduler.plist`
- `~/Library/LaunchAgents/com.roehub.clickhouse.plist`
- `~/Library/LaunchAgents/com.roehub.blackbox-exporter.plist`

### Test

- `/opt/roehub/test/postgresql`
- `/opt/roehub/test/redis`
- `/opt/roehub/test/clickhouse`
- `/opt/roehub/test/grafana`
- `/opt/roehub/test/prometheus`
- `/opt/roehub/test/blackbox`
- `/Users/daniildegtyarev/.config/roehub/roehub.test.env`
- `~/Library/LaunchAgents/com.roehub.test.postgres.plist`
- `~/Library/LaunchAgents/com.roehub.test.redis.plist`
- `~/Library/LaunchAgents/com.roehub.test.clickhouse.plist`
- `~/Library/LaunchAgents/com.roehub.test.grafana.plist`
- `~/Library/LaunchAgents/com.roehub.test.prometheus.plist`
- `~/Library/LaunchAgents/com.roehub.test.blackbox-exporter.plist`
- `~/Library/LaunchAgents/com.roehub.test.api.plist`
- `~/Library/LaunchAgents/com.roehub.test.market-data-ws-worker.plist`
- `~/Library/LaunchAgents/com.roehub.test.market-data-scheduler.plist`

---

## Пошаговый план работ

## Фаза 0 - stop line и предварительная фиксация состояния

### Цель

- зафиксировать текущий live state до начала разрушения Docker stack
- подтвердить autologin/user-session model
- подтвердить runner/Tailscale/backend status

### Где выполняем

- с любой админской машины, где работает SSH в `Mac Studio`

### Пользователь

- локальный админ на вашей рабочей машине
- на `Mac Studio` команды выполняются под `daniildegtyarev`

### Команды

```bash
ssh macstudio 'whoami && sw_vers'
ssh macstudio 'defaults read /Library/Preferences/com.apple.loginwindow autoLoginUser'
ssh macstudio 'fdesetup status'
ssh macstudio 'export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin; colima status || true; docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" || true'
ssh macstudio 'launchctl list | grep -E "roehub|actions.runner|tailscale|colima" || true'
gh api repos/Dejetins/roehub.com/actions/runners
curl -fsS https://roehub.com/__edge_id
curl -sS -o /tmp/current-user.json -w '%{http_code}' https://roehub.com/api/auth/current-user
```

### Ожидаемый результат

- `autoLoginUser` возвращает `daniildegtyarev`
- `FileVault is Off.`
- видим текущий container stack на `Mac Studio`
- runner `mac-studio-prod` в GitHub `online`
- `https://roehub.com/__edge_id` возвращает `vps-edge`
- `https://roehub.com/api/auth/current-user` возвращает `401`

---

## Фаза 1 - подготовить изменения в репозитории

### Цель

- подготовить repo к native runtime и native test environment
- убрать backend deploy зависимость от Docker compose

### Где выполняем

- локально в checkout репозитория

### Пользователь

- разработчик/оператор на рабочей машине

### Целевые изменения

- добавить файлы из секции `Файлы в репозитории, которые должны быть добавлены или изменены`
- добавить `configs/test/market_data.yaml`
- добавить `configs/test/whitelist.csv`
- переписать `.github/workflows/deploy-backend.yml` под native deploy
- добавить native runbook
- обновить docs index

### Команды проверки

```bash
uv run python -m tools.docs.generate_docs_index --check
uv run ruff check .
uv run pyright
uv run pytest -q -ra
git diff -- docs/ .github/workflows/ configs/ infra/ scripts/
```

### Ожидаемый результат

- docs index не дрейфует
- линтер/type-check/tests проходят
- diff содержит только целевые native migration changes

---

## Фаза 2 - подготовить каталоги и backup layout на Mac Studio

### Цель

- создать stable layout для native runtime
- подготовить каталог для backup/export перед разбором Docker stack

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`
- для `mkdir/chown` в `/opt` нужны `sudo`

### Команды

```bash
ssh macstudio '
  sudo mkdir -p /opt/roehub/app /opt/roehub/bin /opt/roehub/config /opt/roehub/state/backups /opt/roehub/clickhouse /opt/roehub/test &&
  sudo chown -R daniildegtyarev:staff /opt/roehub &&
  mkdir -p /Users/daniildegtyarev/.config/roehub /Users/daniildegtyarev/.local/bin /Users/daniildegtyarev/Library/Logs/roehub /Users/daniildegtyarev/Library/LaunchAgents
'
```

### Ожидаемый результат

- все каталоги созданы
- `/opt/roehub` принадлежит `daniildegtyarev`
- пользовательские каталоги конфигов/логов готовы

---

## Фаза 3 - экспортировать production данные из текущего Docker runtime

### Цель

- сохранить production данные до выключения Docker backend
- получить контрольные файлы и checksums

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Предусловия

- текущий Docker stack еще жив
- файл `/Users/daniildegtyarev/.config/roehub/roehub.env` доступен

### Команды

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  export ROEHUB_ENV_FILE=/Users/daniildegtyarev/.config/roehub/roehub.env
  mkdir -p /opt/roehub/state/backups/postgres /opt/roehub/state/backups/clickhouse /opt/roehub/state/backups/grafana /opt/roehub/state/backups/redis
  set -a
  source "$ROEHUB_ENV_FILE"
  set +a
  docker exec roehub-postgres-1 psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "select now();"
  docker exec roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT count() FROM market_data.canonical_candles_1m"
  docker exec roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT max(ts_open) FROM market_data.canonical_candles_1m"
  docker exec -t roehub-postgres-1 pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > /opt/roehub/state/backups/postgres/roehub_prod.dump
  shasum -a 256 /opt/roehub/state/backups/postgres/roehub_prod.dump > /opt/roehub/state/backups/postgres/roehub_prod.dump.sha256
  docker exec roehub-clickhouse-1 mkdir -p /var/lib/clickhouse/backup
  docker exec roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "BACKUP DATABASE market_data TO File('/var/lib/clickhouse/backup/market_data_prod.zip')"
  docker cp roehub-clickhouse-1:/var/lib/clickhouse/backup/market_data_prod.zip /opt/roehub/state/backups/clickhouse/market_data_prod.zip
  shasum -a 256 /opt/roehub/state/backups/clickhouse/market_data_prod.zip > /opt/roehub/state/backups/clickhouse/market_data_prod.zip.sha256
  docker run --rm -v grafana_data:/from -v /opt/roehub/state/backups/grafana:/to alpine sh -c "cp -a /from/. /to/"
  tar -C /opt/roehub/state/backups -czf /opt/roehub/state/backups/grafana_prod.tar.gz grafana
  shasum -a 256 /opt/roehub/state/backups/grafana_prod.tar.gz > /opt/roehub/state/backups/grafana_prod.tar.gz.sha256
  docker exec redis redis-cli SAVE || true
  docker run --rm -v roehub_redis_data:/from -v /opt/roehub/state/backups/redis:/to alpine sh -c "cp -a /from/. /to/" || true
  cp /Users/daniildegtyarev/.config/roehub/roehub.env /opt/roehub/state/backups/roehub.env.backup
  cp /opt/roehub/docker-compose.backend.yml /opt/roehub/state/backups/docker-compose.backend.yml.backup
'
```

### Ожидаемый результат

- есть `Postgres` dump и `sha256`
- есть `ClickHouse` backup archive и `sha256`
- есть `Grafana` archive и `sha256`
- есть копия текущего env файла
- есть копия текущего compose manifest

### Дополнительная проверка

```bash
ssh macstudio 'ls -lah /opt/roehub/state/backups/postgres /opt/roehub/state/backups/clickhouse /opt/roehub/state/backups | sed -n "1,120p"'
```

---

## Фаза 4 - остановить текущий Docker backend, но пока не удалять volumes

### Цель

- снять production runtime с Docker/Colima
- освободить порты под native services
- оставить возможность последней аварийной сверки через сохраненные volumes до финального cleanup

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Команды

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  export ROEHUB_ENV_FILE=/Users/daniildegtyarev/.config/roehub/roehub.env
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.backend.plist || true
  docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" down --remove-orphans
  docker ps --format "table {{.Names}}\t{{.Status}}"
'
```

### Ожидаемый результат

- контейнеров Roehub backend больше нет в `docker ps`
- порты `5432`, `6379`, `8000`, `8123`, `9000`, `3000`, `9090`, `9115` свободны

### Проверка портов

```bash
ssh macstudio 'lsof -nP -iTCP:5432,6379,8000,8123,9000,3000,9090,9115 -sTCP:LISTEN || true'
```

---

## Фаза 5 - установить native зависимости и бинарники

### Цель

- поставить все необходимые host binaries
- подготовить production native runtime без Docker

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Команды

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  brew update
  brew install uv postgresql@16 redis grafana prometheus
  mkdir -p /opt/clickhouse /opt/roehub/bin /opt/roehub/state/downloads
  curl -L https://github.com/ClickHouse/ClickHouse/releases/download/v26.2.4.23-stable/clickhouse-macos-aarch64 -o /opt/clickhouse/clickhouse
  echo "fa6589cd762fb4d77f839c84e78a87706a30a414506da3ae9ebcc8720fbed7a1  /opt/clickhouse/clickhouse" | shasum -a 256 -c -
  chmod +x /opt/clickhouse/clickhouse
  curl -L https://github.com/prometheus/blackbox_exporter/releases/download/v0.28.0/blackbox_exporter-0.28.0.darwin-arm64.tar.gz -o /opt/roehub/state/downloads/blackbox_exporter-0.28.0.darwin-arm64.tar.gz
  echo "ec6c70ccca92e209dd22be76a4fa244f4bd31afdae3ddb2bb082144100ec52bb  /opt/roehub/state/downloads/blackbox_exporter-0.28.0.darwin-arm64.tar.gz" | shasum -a 256 -c -
  rm -rf /opt/roehub/state/downloads/blackbox_exporter-0.28.0
  mkdir -p /opt/roehub/state/downloads/blackbox_exporter-0.28.0
  tar -xzf /opt/roehub/state/downloads/blackbox_exporter-0.28.0.darwin-arm64.tar.gz -C /opt/roehub/state/downloads/blackbox_exporter-0.28.0
  install -m 0755 /opt/roehub/state/downloads/blackbox_exporter-0.28.0/blackbox_exporter-0.28.0.darwin-arm64/blackbox_exporter /opt/roehub/bin/blackbox_exporter
  /opt/clickhouse/clickhouse local --query "SELECT version()"
  /opt/roehub/bin/blackbox_exporter --version | head -n 1
'
```

### Ожидаемый результат

- `uv`, `postgresql@16`, `redis`, `grafana`, `prometheus` установлены через `Homebrew`
- native `clickhouse` binary лежит в `/opt/clickhouse/clickhouse`
- native `blackbox_exporter` binary лежит в `/opt/roehub/bin/blackbox_exporter`

---

## Фаза 6 - поднять production native data services

### Цель

- поднять `Postgres`, `Redis`, `ClickHouse`, `Grafana`, `Prometheus`, `blackbox_exporter` без Docker
- восстановить необходимые production данные

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`
- где требуется системная конфигурация Homebrew services - тот же пользователь

### Шаг 6.1 - Postgres prod

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  brew services start postgresql@16
  sleep 5
  pg_isready -h 127.0.0.1 -p 5432
  set -a
  source /Users/daniildegtyarev/.config/roehub/roehub.env
  set +a
  psql postgres -c "DO \\$\\$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '\''${POSTGRES_USER}'\'') THEN CREATE ROLE ${POSTGRES_USER} LOGIN PASSWORD '\''${POSTGRES_PASSWORD}'\''; ELSE ALTER ROLE ${POSTGRES_USER} WITH LOGIN PASSWORD '\''${POSTGRES_PASSWORD}'\''; END IF; END \\$\\$;"
  psql postgres -c "SELECT 'DROP DATABASE IF EXISTS ${POSTGRES_DB}' WHERE FALSE;"
  psql postgres -c "SELECT 1;"
  createdb -O "$POSTGRES_USER" "$POSTGRES_DB" || true
  pg_restore --no-owner --role="$POSTGRES_USER" -d "$POSTGRES_DB" /opt/roehub/state/backups/postgres/roehub_prod.dump
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT now();"
'
```

Ожидаемый результат:

- `pg_isready` возвращает `accepting connections`
- dump восстановлен без ошибок
- целевая БД доступна локально на `127.0.0.1:5432`

### Шаг 6.2 - Redis prod

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  brew services start redis
  sleep 3
  redis-cli -h 127.0.0.1 -p 6379 PING
'
```

Ожидаемый результат:

- `PONG`

### Шаг 6.3 - ClickHouse prod

```bash
ssh macstudio '
  mkdir -p /opt/roehub/clickhouse/{data,tmp,logs,backups,config.d,users.d}
  cp /opt/roehub/state/backups/clickhouse/market_data_prod.zip /opt/roehub/clickhouse/backups/market_data_prod.zip
  ls -lah /opt/roehub/clickhouse/backups/market_data_prod.zip
'
```

Дальше после доставки repo templates на машину:

```bash
ssh macstudio '
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.clickhouse.plist || true
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.clickhouse.plist
  sleep 10
  /opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT version()"
  set -a
  source /Users/daniildegtyarev/.config/roehub/roehub.env
  set +a
  /opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "RESTORE DATABASE market_data FROM Disk('backups', 'market_data_prod.zip')"
  /opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT count() FROM market_data.canonical_candles_1m"
  /opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT max(ts_open) FROM market_data.canonical_candles_1m"
'
```

Ожидаемый результат:

- `clickhouse` отвечает на `8123/9000`
- `RESTORE DATABASE market_data ...` завершается без ошибки
- контрольные метрики `count()` и `max(ts_open)` совпадают с зафиксированными перед остановкой Docker stack

### Шаг 6.4 - Grafana prod

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  brew services start grafana
  sleep 5
  brew services stop grafana
  rm -rf /opt/homebrew/var/lib/grafana/*
  tar -xzf /opt/roehub/state/backups/grafana_prod.tar.gz -C /opt/roehub/state/backups
  cp -a /opt/roehub/state/backups/grafana/. /opt/homebrew/var/lib/grafana/
  brew services start grafana
  sleep 5
  curl -I http://127.0.0.1:3000
'
```

Ожидаемый результат:

- `Grafana` стартует
- `curl -I http://127.0.0.1:3000` возвращает `200` или `302`

### Шаг 6.5 - Prometheus prod

После доставки native config в `/opt/roehub/config/prometheus.prod.yml`:

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  cat > /opt/homebrew/etc/prometheus.args <<"EOF"
--config.file=/opt/roehub/config/prometheus.prod.yml
--storage.tsdb.path=/opt/homebrew/var/prometheus
--storage.tsdb.retention.time=90d
EOF
  brew services restart prometheus
  sleep 5
  curl -I http://127.0.0.1:9090
'
```

Ожидаемый результат:

- `Prometheus` доступен на `127.0.0.1:9090`
- `curl -I` возвращает `200` или `405`

### Шаг 6.6 - blackbox exporter prod

После доставки `com.roehub.blackbox-exporter.plist` и `/opt/roehub/config/blackbox.yml`:

```bash
ssh macstudio '
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.blackbox-exporter.plist || true
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.blackbox-exporter.plist
  sleep 3
  curl -I http://127.0.0.1:9115
'
```

Ожидаемый результат:

- `blackbox_exporter` отвечает `200`

---

## Фаза 7 - поднять production native Python services

### Цель

- развернуть native source-based runtime для `api` и market-data workers
- привязать launchd services к стабильному checkout

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Команды подготовки runtime

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin
  rm -rf /opt/roehub/app
  mkdir -p /opt/roehub/app
'
```

Реальный sync кода делает GitHub Actions workflow, но для первичного bootstrap допустим `rsync`/`scp` или checkout напрямую в `/opt/roehub/app`.

После доставки кода:

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin
  cd /opt/roehub/app
  uv python install 3.12
  uv sync --locked
  uv run python -V
'
```

Ожидаемый результат:

- в `/opt/roehub/app/.venv` собрана production venv
- `Python 3.12.x`

### Команды запуска services

После установки plists в `~/Library/LaunchAgents`:

```bash
ssh macstudio '
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.api.plist || true
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.api.plist
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.market-data-ws-worker.plist || true
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.market-data-ws-worker.plist
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.market-data-scheduler.plist || true
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.market-data-scheduler.plist
  sleep 10
  curl -i http://127.0.0.1:8000/auth/current-user
  curl -fsS http://127.0.0.1:9201/metrics | sed -n "1,5p"
  curl -fsS http://127.0.0.1:9202/metrics | sed -n "1,5p"
'
```

### Ожидаемый результат

- `API` отвечает `401` на `/auth/current-user`
- metrics market-data worker/scheduler доступны локально
- процессы стартуют без Docker dependency

---

## Фаза 8 - настроить production Tailscale Serve поверх native сервисов

### Цель

- сохранить текущий private access contract после ухода от Docker

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Команды

```bash
ssh macstudio '
  /usr/local/bin/tailscale serve reset || true
  /usr/local/bin/tailscale serve --bg http://127.0.0.1:8000
  /usr/local/bin/tailscale serve --bg --https=3443 3000
  /usr/local/bin/tailscale serve --bg --tcp=15432 tcp://127.0.0.1:5432
  /usr/local/bin/tailscale serve --bg --tcp=18123 tcp://127.0.0.1:8123
  /usr/local/bin/tailscale serve --bg --tcp=19000 tcp://127.0.0.1:9000
  /usr/local/bin/tailscale serve --bg --https=8443 18000
  /usr/local/bin/tailscale serve --bg --https=3444 13000
  /usr/local/bin/tailscale serve --bg --tcp=25432 tcp://127.0.0.1:15433
  /usr/local/bin/tailscale serve --bg --tcp=28123 tcp://127.0.0.1:18124
  /usr/local/bin/tailscale serve --bg --tcp=29000 tcp://127.0.0.1:19001
  /usr/local/bin/tailscale serve status
'
```

### Ожидаемый результат

- видим и production, и test endpoints в `tailscale serve status`
- публичный internet ingress по-прежнему отсутствует на `Mac Studio`

---

## Фаза 9 - поднять native test environment

### Цель

- создать второй backend contour на `Mac Studio` для безопасной проверки native runtime
- не смешивать production state и test state

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Подход

- test env не импортирует production данные
- test env использует:
  - `ROEHUB_ENV=test`
  - `configs/test/backtest.yaml`
  - `configs/test/strategy.yaml`
  - `configs/test/indicators.yaml`
  - новый `configs/test/market_data.yaml`
  - новый `configs/test/whitelist.csv`
- все порты и data dirs отделены от prod

### Команды

```bash
ssh macstudio '
  mkdir -p /opt/roehub/test/postgresql /opt/roehub/test/redis /opt/roehub/test/clickhouse/{data,tmp,logs,backups} /opt/roehub/test/grafana /opt/roehub/test/prometheus /opt/roehub/test/blackbox
  install -m 0600 /dev/null /Users/daniildegtyarev/.config/roehub/roehub.test.env
'
```

После доставки test env file и test plists:

```bash
ssh macstudio '
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.postgres.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.redis.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.clickhouse.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.grafana.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.prometheus.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.blackbox-exporter.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.api.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.market-data-ws-worker.plist
  launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.test.market-data-scheduler.plist
  sleep 15
  pg_isready -h 127.0.0.1 -p 15433
  redis-cli -h 127.0.0.1 -p 16379 PING
  curl -I http://127.0.0.1:18124/ping
  curl -i http://127.0.0.1:18000/auth/current-user
  curl -I http://127.0.0.1:13000
  curl -I http://127.0.0.1:19090
  curl -fsS http://127.0.0.1:19201/metrics | sed -n "1,5p"
  curl -fsS http://127.0.0.1:19202/metrics | sed -n "1,5p"
'
```

### Ожидаемый результат

- все test services поднимаются независимо от prod
- test API отвечает `401`
- test monitoring доступен локально
- test services не занимают production порты

---

## Фаза 10 - перевести backend deploy workflow на native source-based deploy

### Цель

- сделать backend deploy на `Mac Studio` полностью без Docker

### Где выполняем

- в репозитории и GitHub Actions

### Пользователь

- разработчик в repo
- workflow исполняется runner'ом `mac-studio-prod`

### Новый backend deploy workflow должен делать

1. `actions/checkout`
2. sync файлов в `/opt/roehub/app`:
   - `apps/`
   - `src/`
   - `configs/`
   - `alembic/`
   - `migrations/`
   - `.python-version`
   - `pyproject.toml`
   - `uv.lock`
   - `scripts/macos/`
   - `infra/macos/`
3. `uv python install 3.12`
4. `uv sync --locked`
5. install/update prod launchd plists
6. reload `api` и market-data services через `launchctl`
7. при изменении monitoring configs - reload `prometheus`/`blackbox`/`grafana`
8. smoke:
   - `curl -i http://127.0.0.1:8000/auth/current-user` -> `401`
   - `curl -fsS http://127.0.0.1:9201/metrics`
   - `curl -fsS http://127.0.0.1:9202/metrics`
   - `tailscale serve status`

### Команды проверки после workflow rewrite

```bash
gh workflow view "Deploy Backend"
gh run list --workflow "Deploy Backend" --limit 5
```

### Ожидаемый результат

- backend deploy больше не использует `docker compose`
- backend deploy не зависит от `GHCR` image для `Mac Studio`
- backend deploy работает на том же self-hosted runner

---

## Фаза 11 - cutover verification

### Цель

- доказать, что production backend уже работает native, а старый Docker runtime больше не нужен

### Где выполняем

- `Mac Studio`
- рабочая машина оператора
- optionally `VPS`

### Пользователь

- `daniildegtyarev`

### Команды production verification

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin
  brew services list
  launchctl list | grep -E "roehub|clickhouse|blackbox|actions.runner|tailscale"
  curl -I http://127.0.0.1:3000
  curl -I http://127.0.0.1:9090
  curl -I http://127.0.0.1:9115
  curl -i http://127.0.0.1:8000/auth/current-user
  curl -fsS http://127.0.0.1:9201/metrics | sed -n "1,10p"
  curl -fsS http://127.0.0.1:9202/metrics | sed -n "1,10p"
  /opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT 1"
  redis-cli -h 127.0.0.1 -p 6379 PING
  tailscale serve status
'
curl -sS -o /tmp/roehub-public.html -w '%{http_code}' https://roehub.com/
curl -sS -o /tmp/roehub-api.json -w '%{http_code}' https://roehub.com/api/auth/current-user
```

### Ожидаемый результат

- локальные native services healthy
- `API` локально дает `401`
- public web дает `200`
- public `/api/auth/current-user` дает `401`
- `tailscale serve` указывает на native local ports, а не на Docker runtime

---

## Фаза 12 - удалить текущий Docker/Colima runtime и volumes

### Цель

- полностью вывести из эксплуатации старую containerized backend topology

### Где выполняем

- `Mac Studio`

### Пользователь

- `daniildegtyarev`

### Предусловие

- все smoke-проверки из `Фаза 11` пройдены
- backups из `Фаза 3` сохранены и проверены

### Команды

```bash
ssh macstudio '
  export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
  docker ps -a --format "table {{.Names}}\t{{.Status}}" || true
  docker volume ls || true
  docker volume rm roehub_pg_data roehub_ch_data roehub_ch_logs roehub_redis_data roehub_prom_data grafana_data || true
  brew services stop colima || true
  colima delete -f || true
  brew uninstall -f colima docker docker-compose caddy || true
  launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.backend.plist || true
  rm -f /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.backend.plist
  rm -f /Users/daniildegtyarev/Library/LaunchAgents/homebrew.mxcl.colima.plist
  rm -f /Users/daniildegtyarev/.local/bin/roehub_backend_up
  docker ps || true
  docker volume ls || true
'
```

### Ожидаемый результат

- `Colima` больше нет
- Docker runtime и Docker volumes удалены
- старый `com.roehub.backend` LaunchAgent удален
- на `Mac Studio` остался только native backend runtime

---

## Фаза 13 - post-cutover acceptance checklist

- `Mac Studio` не зависит от `Colima` и Docker.
- production backend полностью native.
- production monitoring полностью native.
- runner на `Mac Studio` online и не зависит от Docker.
- prod API доступен локально и через tailnet.
- `VPS` по-прежнему отдает `web` и reverse proxy на `Mac Studio API`.
- test backend environment существует на том же `Mac Studio` и работает независимо от prod.
- Docker volumes удалены.
- старый `com.roehub.backend` launch agent удален.
- `strategy-live-worker` и `backtest-job-runner` не были подняты в рамках этого cutover.

---

## Набор smoke-проверок, которые должны войти в финальный native runbook

### Production

```bash
brew services list
launchctl list | grep roehub
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

Ожидание:

- `Grafana` -> `200` или `302`
- `Prometheus` -> `200` или `405`
- `blackbox_exporter` -> `200`
- `API /auth/current-user` -> `401`
- `Redis` -> `PONG`
- `ClickHouse` -> `1`

### Test

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

Ожидание:

- test `Postgres`, `Redis`, `ClickHouse`, `Grafana`, `Prometheus`, `API`, market-data workers healthy
- ни один test port не конфликтует с prod

---

## Основные риски и как закрываем

### 1. Риск потери production данных при удалении volumes

Митигатор:

- обязательный export `Postgres` и `ClickHouse`
- `sha256` на backup artifacts
- удаление volumes только после native restore verification

### 2. Риск несовместимости native ClickHouse

Митигатор:

- не копировать Docker volume raw-файлы напрямую в native layout
- использовать logical `BACKUP/RESTORE`
- проверять `count()` и `max(ts_open)` до/после restore

### 3. Риск расхождения production и test runtime

Митигатор:

- один и тот же source checkout `/opt/roehub/app`
- разница только в env, ports, data dirs и `configs/test/*`

### 4. Риск того, что deploy-backend workflow останется docker-oriented

Митигатор:

- переписать workflow до cutover
- тестовый ручной `workflow_dispatch`
- smoke внутри workflow против native local ports

### 5. Риск user-session зависимостей launchd

Митигатор:

- использовать уже подтвержденный auto-login под `daniildegtyarev`
- хранить все custom services в user `LaunchAgents`
- runner и `tailscale serve` оставить в той же user-session модели

---

## Критерии готовности к выполнению плана

- в repo подготовлены все native runtime файлы
- `deploy-backend.yml` переписан под native deploy
- `configs/test/market_data.yaml` и `configs/test/whitelist.csv` добавлены
- backup artifacts успешно созданы и проверены
- native prod stack поднят и проходит smoke
- native test stack поднят и проходит smoke
- `VPS` продолжает успешно проксировать `/api/*` на native prod API
- Docker runtime на `Mac Studio` выключен и удален

---

## Что считаем итогом проекта

Итог достигнут, когда одновременно верны все пункты ниже:

- backend Roehub на `Mac Studio` работает полностью нативно, без `Colima` и без Docker
- production backend services доступны локально и через tailnet
- self-hosted runner работает на том же `Mac Studio`
- существует отдельная native test environment на том же `Mac Studio`
- текущий Docker stack и его volumes удалены
- документация, workflows и scripts в репозитории соответствуют новой native topology


Выполни по порядку:
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export ROEHUB_ENV_FILE=/Users/daniildegtyarev/.config/roehub/roehub.env
# 1) CLICKHOUSE: запускаем вручную (обход launchd)
launchctl bootout "gui/$(id -u)" "/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.clickhouse.plist" || true
pkill -f "/opt/clickhouse/clickhouse server" || true
mkdir -p /opt/roehub/clickhouse/data /opt/roehub/clickhouse/tmp /opt/roehub/clickhouse/logs /opt/roehub/clickhouse/backups /opt/roehub/clickhouse/access /opt/roehub/clickhouse/user_files /opt/roehub/clickhouse/format_schemas
sudo chown -R daniildegtyarev:staff /opt/roehub/clickhouse /opt/roehub/config
nohup /bin/zsh -lc 'ulimit -n 262144; set -a; source /Users/daniildegtyarev/.config/roehub/roehub.env; set +a; exec /opt/clickhouse/clickhouse server --config-file /opt/roehub/config/clickhouse.config.xml' >/Users/daniildegtyarev/Library/Logs/roehub/clickhouse.manual.out.log 2>/Users/daniildegtyarev/Library/Logs/roehub/clickhouse.manual.err.log &
sleep 6
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT version()"
Если версия вывелась — продолжай:
set -a
source "$ROEHUB_ENV_FILE"
set +a
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user "${CLICKHOUSE_USER:-roe}" --password "$CLICKHOUSE_PASSWORD" --query "CREATE DATABASE IF NOT EXISTS market_data"
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user "${CLICKHOUSE_USER:-roe}" --password "$CLICKHOUSE_PASSWORD" --multiquery < /Users/daniildegtyarev/projects/roehub.com/migrations/clickhouse/market_data_ddl.sql
Теперь Grafana:
# 2) GRAFANA: чистый старт
brew services stop grafana || true
python3 -c "import os,shutil; p='/opt/homebrew/var/lib/grafana'; os.makedirs(p,exist_ok=True); [shutil.rmtree(os.path.join(p,n)) if os.path.isdir(os.path.join(p,n)) and not os.path.islink(os.path.join(p,n)) else os.unlink(os.path.join(p,n)) for n in os.listdir(p)]"
sudo chown -R daniildegtyarev:staff /opt/homebrew/var/lib/grafana /opt/homebrew/var/log/grafana
brew services start grafana
sleep 5
curl -I http://127.0.0.1:3000
И финальная проверка:
brew services restart postgresql@16
brew services restart redis
brew services restart prometheus
launchctl bootout "gui/$(id -u)" "/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.blackbox-exporter.plist" || true
launchctl bootstrap "gui/$(id -u)" "/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.blackbox-exporter.plist"
sleep 3
pg_isready -h 127.0.0.1 -p 5432
redis-cli -h 127.0.0.1 -p 6379 PING
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT 1"
curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9115
Если первый блок снова не поднимет ClickHouse — сразу пришли:
tail -n 120 /Users/daniildegtyarev/Library/Logs/roehub/clickhouse.manual.err.log