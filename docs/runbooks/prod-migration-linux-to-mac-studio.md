# Переезд продакшена с Linux на Mac Studio (macOS)

Пошаговый runbook: перенос прод-стека Roehub (Docker Compose) + данных Postgres/ClickHouse на Mac Studio, настройка публичного `roehub.com` и удаленного управления.

## Цели и принципы

- **Цель:** Mac Studio становится единственной production-машиной (Linux сервер будет выключен).
- **Данные:** мигрируем **все** данные `Postgres` + `ClickHouse` (и при желании `Grafana/Prometheus/Redis` volumes).
- **CI:** остается на GitHub-hosted runners (`ubuntu-latest`) — см. `b/.github/workflows/ci.yml`.
- **Deploy:** выполняется на **self-hosted runner на Mac Studio** — адаптация `b/.github/workflows/deploy.yml`.
- **Удаленное управление:** только через **Tailscale + SSH** (порт 22 наружу не открываем).
- **Публичный доступ:** `roehub.com` должен работать с Mac Studio (TLS обязателен).

## Что уже есть в репозитории (важно для миграции)

- Прод-лейаут деплоя:
  - compose: `/opt/roehub/docker-compose.yml`
  - env: `/etc/roehub/roehub.env`
  - build context: `/opt/roehub/market-data-src`
- Deploy workflow (сейчас Linux): `b/.github/workflows/deploy.yml`.
- Compose (prod): `infra/docker/docker-compose.yml`.

## Публичный ingress (статический IP: `185.155.18.21`)

У вас есть выделенный статический IPv4 `185.155.18.21`, поэтому базовый и самый прозрачный вариант для продакшена:

- публичный `HTTPS` на Mac Studio через **Caddy** (Let's Encrypt)
- Roehub gateway остается на `127.0.0.1:8080` и не торчит наружу

Ниже описаны два варианта (B — рекомендуемый при статическом IP).

### Вариант A: Cloudflare Tunnel (без проброса портов)

Подходит, если:

- нет статического IPv4,
- есть CGNAT,
- не хочется открывать 80/443 на роутере,
- нужен быстрый и предсказуемый TLS.

Минусы: зависимость от Cloudflare.

### Вариант B (рекомендуемый): Прямой вход на 80/443 + Caddy на Mac Studio

Подходит, если:

- есть публичный IPv4 `185.155.18.21`,
- можно принимать входящие 80/443 на Mac Studio (и на периметре, если он есть),
- хочется полностью self-hosted TLS.

Минусы: нужно аккуратно закрыть остальную поверхность (SSH/прочие сервисы не выставлять наружу).

Дальше в runbook шаги общие + развилка по ingress.

---

## Фаза 0 — Подготовка и стоп-линия (Linux прод)

Задача: подготовить перенос так, чтобы можно было повторить и проверить, и чтобы данные были консистентны.

На Linux прод-сервере:

1) Зафиксировать версию стека и состояние:

```bash
docker --version
docker compose version
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

2) Убедиться, что есть актуальный env-файл и он содержит нужные ключи (минимум):

- `POSTGRES_PASSWORD`
- `CLICKHOUSE_PASSWORD`
- `ROEHUB_ENV=prod`
- `IDENTITY_COOKIE_SECURE=true` (в проде обязательно)
- `GF_SERVER_ROOT_URL` (для grafana)

Проверка (без вывода секретов):

```bash
sudo test -s /etc/roehub/roehub.env
sudo rg -n '^(ROEHUB_ENV|POSTGRES_PASSWORD|CLICKHOUSE_PASSWORD|IDENTITY_COOKIE_SECURE|GF_SERVER_ROOT_URL)=' /etc/roehub/roehub.env
```

3) Подготовить окно даунтайма для финального cutover (когда будем останавливать Linux и снимать volumes).

---

## Фаза 1 — Сетевой доступ к Mac Studio (LAN + remote)

### 1.1 Thunderbolt Bridge (быстрый локальный канал)

На MacBook Pro и Mac Studio:

1) `System Settings → Network` включить **Thunderbolt Bridge**.
2) Задать статические IPv4 (пример):

- Mac Studio: `10.50.0.1/30`
- MacBook Pro: `10.50.0.2/30`

3) Проверить ping и SSH:

```bash
ping 10.50.0.1
ssh <user>@10.50.0.1
```

### 1.2 SSH на Mac Studio

На Mac Studio:

1) `System Settings → General → Sharing → Remote Login` включить.
2) Разрешить доступ только для выделенного пользователя (см. Фаза 2).

### 1.3 Tailscale (удаленное управление из другой сети)

На Mac Studio:

```bash
# (если Homebrew еще не установлен — см. Фаза 2)
brew install --cask tailscale
```

Дальше зайти в Tailscale и залогиниться в tailnet.

Проверка:

```bash
tailscale status
tailscale ip -4
```

На MacBook Pro:

```bash
brew install --cask tailscale
```

Рекомендация по SSH:

- Не открывать порт 22 наружу (даже при статическом IP).
- Ходить на Mac Studio по SSH через Tailscale IP/hostname.

Пример `~/.ssh/config` на MacBook:

```sshconfig
Host roehub-studio-lan
  HostName 10.50.0.1
  User deploy
  IdentityFile ~/.ssh/id_ed25519

Host roehub-studio-vpn
  HostName <mac-studio-tailnet-hostname>
  User deploy
  IdentityFile ~/.ssh/id_ed25519

# (опционально) Если все же нужен прямой SSH по статическому IP:
# Host roehub-studio-public
#   HostName 185.155.18.21
#   User deploy
#   IdentityFile ~/.ssh/id_ed25519
```

Опционально (если все же открываете SSH наружу):

- только key-based auth
- `PasswordAuthentication no`
- ограничить `AllowUsers deploy`
- по возможности ограничить вход по IP (pf)

---

## Фаза 2 — Подготовить Mac Studio как production-хост

### 2.1 Создать пользователя для прод-операций

Можно делать все из своего admin-аккаунта — это будет работать, но это хуже по безопасности и сопровождению.

Почему отдельный пользователь — "по-взрослому":

- **Least privilege:** GitHub Actions runner выполняет код из репозитория; если runner живет в твоем admin-аккаунте, компрометация job == компрометация твоего профиля/ключей/настроек.
- **Меньше blast radius:** ошибочный скрипт деплоя не должен иметь `sudo` по умолчанию.
- **Разделение секретов:** проще хранить production-артефакты (`/opt/roehub`, `/etc/roehub`) с понятными владельцами/правами.
- **Аудит и стабильность:** отдельный user для автоматики меньше завязан на твой Keychain/GUI/личные тулзы.

Рекомендуемая модель для твоего случая:

- твой пользователь (admin) — **только** администрирование macOS и интерактивный SSH
- `deploy` (standard user, без admin) — **только** GitHub Actions runner + docker/compose операции

Важно: `deploy` не обязан иметь SSH-доступ. Можно оставить **Remote Login только для твоего пользователя**, а runner запускать от `deploy` как сервис.

На Mac Studio (выполнить в Terminal под админом):

```bash
# создать пользователя deploy (пример)
sudo sysadminctl -addUser deploy -shell /bin/zsh -home /Users/deploy
sudo sysadminctl -resetPasswordFor deploy

# убедиться, что deploy НЕ в группе admin
dseditgroup -o checkmember -m deploy admin || true
```

Дальше в `System Settings → General → Sharing → Remote Login`:

- выбрать **Only these users**
- добавить **только** твой основной аккаунт
- не добавлять `deploy`

### 2.2 Установить базовый софт

На Mac Studio под пользователем `deploy`:

1) Homebrew (если еще нет):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2) Пакеты:

```bash
brew install git uv jq rsync zstd ripgrep docker-buildx
```

### 2.3 Docker engine (рекомендуем: Colima)

Цель: иметь headless Docker engine, который можно запускать без GUI и использовать в GitHub Actions runner.

Альтернатива: Docker Desktop (проще стартует/обслуживается, но GUI/лицензирование могут быть нежелательны для прод-сервера).

На Mac Studio под `deploy`:

```bash
brew install colima docker docker-compose
```

Если ты уже запустил `colima start` под своим основным пользователем (не `deploy`) — это не ошибка,
но важно помнить: у каждого пользователя свой `~/.colima` и свой Docker socket. Для прод-деплоя через
GitHub Actions runner лучше, чтобы **runner и Colima жили под одним и тем же пользователем** (обычно `deploy`).

Запуск Colima (подбери ресурсы под объем CH/PG; пример):

```bash
colima start --cpu 6 --memory 16 --disk 200
docker version
docker compose version

# убедиться, что выбран docker context colima
docker context ls
docker context use colima
```

Если `docker compose ...` не работает и пишет `unknown command: docker compose`:

Это означает, что Compose v2 plugin не найден. Исправление (под тем же пользователем, который будет запускать deploy):

```bash
brew install docker-compose

mkdir -p ~/.docker/cli-plugins
ln -sfn "$(brew --prefix)/opt/docker-compose/bin/docker-compose" ~/.docker/cli-plugins/docker-compose

docker compose version
```

Если сборка образов падает с ошибкой про BuildKit/buildx (например:
`BuildKit is enabled but the buildx component is missing or broken`):

```bash
brew install docker-buildx

# проверить под пользователем deploy
sudo -iu deploy /bin/zsh -lc "docker buildx version"
```

Альтернатива (не рекомендуется для этого репо): использовать команду `docker-compose` вместо `docker compose`.
В репозитории и workflow'ах ожидается именно `docker compose`.

Примечание:

- Если ClickHouse/Postgres данные большие, **сразу** выделяй достаточно `--disk` (переразмерить позже сложнее).

### 2.4 Подготовить серверный layout (/opt + /etc)

На Mac Studio:

```bash
sudo mkdir -p /opt/roehub /etc/roehub

# /opt/roehub должен быть writable для deploy (его использует deploy workflow)
sudo chown -R deploy:staff /opt/roehub
sudo chmod 755 /opt/roehub
```

Файл секретов:

```bash
sudo touch /etc/roehub/roehub.env
sudo chown deploy:staff /etc/roehub/roehub.env
sudo chmod 600 /etc/roehub/roehub.env
```

Дальше мы скопируем содержимое с Linux (Фаза 5).

### 2.5 Настройки macOS для режима "сервер"

На Mac Studio (под админом):

```bash
# не усыплять систему
sudo pmset -a sleep 0

# автоперезапуск при потере питания
sudo pmset -a autorestart 1

pmset -g
```

Рекомендация:

- отключить авто-установку апдейтов, которые могут ребутнуть машину в неожиданный момент
- оставить только уведомления/ручную установку

### 2.6 Автозапуск Colima после перезагрузки (launchd)

Без этого после ребута Docker может не подняться сам.

Важно для Roehub: деплой кладет bundle в `/opt/roehub`, а compose монтирует файлы из `/opt/roehub/monitoring/**`.
Colima VM по умолчанию шарит только `/Users`, поэтому нужно явно примонтировать `/opt/roehub` внутрь VM,
иначе контейнеры `prometheus`/`blackbox` могут падать на bind-mount конфигов.

Вариант (системный LaunchDaemon, стартует `colima` от имени пользователя `deploy`):

1) Создать скрипт:

```bash
sudo tee /usr/local/bin/roehub_colima_start >/dev/null <<'SH'
#!/usr/bin/env bash
set -euo pipefail

COLIMA_BIN="$(/opt/homebrew/bin/brew --prefix)/bin/colima"
if [ ! -x "${COLIMA_BIN}" ]; then
  echo "colima not found at ${COLIMA_BIN}" >&2
  exit 1
fi

exec sudo -u deploy "${COLIMA_BIN}" start --cpu 6 --memory 16 --disk 200 --mount-type virtiofs --mount /opt/roehub:w
SH

sudo chmod +x /usr/local/bin/roehub_colima_start
```

2) Создать plist `/Library/LaunchDaemons/com.roehub.colima.plist`:

```bash
sudo tee /Library/LaunchDaemons/com.roehub.colima.plist >/dev/null <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.roehub.colima</string>

    <key>ProgramArguments</key>
    <array>
      <string>/usr/local/bin/roehub_colima_start</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/var/log/roehub_colima.out.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/roehub_colima.err.log</string>
  </dict>
</plist>
PLIST

sudo chown root:wheel /Library/LaunchDaemons/com.roehub.colima.plist
sudo chmod 644 /Library/LaunchDaemons/com.roehub.colima.plist
```

3) Загрузить сервис:

```bash
sudo launchctl load -w /Library/LaunchDaemons/com.roehub.colima.plist
sudo launchctl list | rg com.roehub.colima || true
```

Проверка (под `deploy`):

```bash
docker version
docker compose version
```

---

## Фаза 3 — GitHub Actions runner на Mac Studio (deploy)

Цель: сохранить текущую модель деплоя (self-hosted runner), но перенести runner на Mac Studio.

### 3.1 Создать runner в репозитории

В GitHub:

`Repo → Settings → Actions → Runners → New self-hosted runner → macOS`

На Mac Studio под пользователем `deploy`:

```bash
mkdir -p /opt/actions-runner/roehub
cd /opt/actions-runner/roehub

# дальше команды будут такими, как выдаст GitHub (curl + tar + ./config.sh)
```

Рекомендуемые labels для runner:

- `roehub`
- `prod`
- `mac-studio`

Пример (GitHub выдаст актуальный токен):

```bash
./config.sh \
  --url https://github.com/<ORG>/<REPO> \
  --token <TOKEN> \
  --name mac-studio-prod \
  --labels roehub,prod,mac-studio \
  --unattended
```

### 3.2 Запустить runner как сервис (launchd)

Рекомендуемый путь для macOS: service через `svc.sh`.

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

Проверка:

```bash
./svc.sh status
```

Важно: runner должен иметь доступ к `docker`/`docker compose` (Colima должен быть запущен для пользователя `deploy`).

---

## Фаза 4 — Обновить deploy workflow под Mac Studio

Сейчас `b/.github/workflows/deploy.yml` привязан к Linux runner:

```yaml
runs-on: [self-hosted, Linux, X64, roehub, prod]
```

Нужно заменить на labels Mac Studio runner:

```yaml
runs-on: [self-hosted, mac-studio, roehub, prod]
```

Дополнительно (рекомендуется): включить manual approval через GitHub Environments:

1) `Repo → Settings → Environments → New environment: production`
2) включить `Required reviewers`.
3) в job добавить:

```yaml
environment: production
```

После изменения workflow:

- Сделай commit в `main`.
- Убедись, что deploy job стартует **на Mac Studio runner**.

---

## Фаза 5 — Перенос конфигурации и данных с Linux на Mac Studio

### 5.1 Скопировать env-файл

На Mac Studio (вытянуть с Linux по SSH):

```bash
scp <linux_user>@<linux_host>:/etc/roehub/roehub.env /tmp/roehub.env
sudo mv /tmp/roehub.env /etc/roehub/roehub.env
sudo chown deploy:staff /etc/roehub/roehub.env
sudo chmod 600 /etc/roehub/roehub.env
```

Проверить, что переменные загрузились (не печатай секреты в лог):

```bash
sudo rg -n '^(ROEHUB_ENV|POSTGRES_DB|POSTGRES_USER|IDENTITY_COOKIE_SECURE)=' /etc/roehub/roehub.env
```

### 5.2 Остановить writer-сервисы на Linux (для консистентного snapshot)

На Linux прод-сервере (минимум — остановить то, что пишет в базы):

```bash
# остановить весь стек (самый простой безопасный вариант)
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env down

docker ps --format 'table {{.Names}}\t{{.Status}}' | rg 'roehub|postgres|clickhouse|prometheus|grafana|redis' || true
```

Примечание:

- Если downtime критичен, вместо `down` можно сначала остановить только writer’ы, сделать backup, потом `down`.

### 5.3 Перенос данных: рекомендуемая стратегия

Для надежности и переносимости между Linux → macOS/ARM рекомендуемый минимум:

- **Postgres:** логический дамп (`pg_dump` / `pg_dumpall --globals-only`) и restore.
- **ClickHouse:** перенос volume (быстрее) или ClickHouse BACKUP (если хочется полностью "официально").

Ниже приведены команды для обоих.

### 5.4 Postgres: pg_dump (рекомендуется)

На Linux (можно временно поднять только postgres, если уже сделали `down`):

```bash
set -euo pipefail

# поднять только postgres, если он не запущен
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env up -d postgres

# дождаться готовности
set -a
source /etc/roehub/roehub.env
set +a
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker exec -i roehub-postgres-1 pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# загрузить env
set -a
source /etc/roehub/roehub.env
set +a

backup_dir="$HOME/roehub-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"
cd "$backup_dir"

# globals (roles/privileges) — полезно, если есть дополнительные роли
docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" roehub-postgres-1 \
  pg_dumpall --globals-only -U "$POSTGRES_USER" > pg_globals.sql

# основной дамп БД
docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" roehub-postgres-1 \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc --no-owner --no-acl > roehub_pg.dump

ls -lh
sha256sum pg_globals.sql roehub_pg.dump > SHA256SUMS

echo "PG backup dir: ${backup_dir}"
pwd
```

### 5.5 ClickHouse: export volumes (быстро, обычно ок)

Рекомендуем переносить volumes “как есть” (особенно ClickHouse).

На Linux (после остановки стека):

```bash
set -euo pipefail

backup_dir="$HOME/roehub-volume-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"
cd "$backup_dir"

# ВНИМАНИЕ: имена volumes зависят от COMPOSE_PROJECT_NAME.
# При COMPOSE_PROJECT_NAME=roehub обычно это:
vols=(
  roehub_ch_data
  roehub_ch_logs
  # (опционально)
  # roehub_redis_data
  # roehub_prom_data
  # grafana_data
)

for v in "${vols[@]}"; do
  echo "Export volume: ${v}"
  docker run --rm \
    -v "${v}:/from" \
    -v "$PWD:/to" \
    alpine:3.20 \
    tar -C /from -cf "/to/${v}.tar" .
done

sha256sum *.tar > SHA256SUMS
ls -lh

echo "CH volumes backup dir: ${backup_dir}"
pwd
```

Если volumes называются иначе — посмотри:

```bash
docker volume ls
```

### 5.6 Передать архивы на Mac Studio

Вариант 1 (простой): Mac Studio сам скачивает с Linux (outbound SSH).

На Mac Studio:

```bash
# пример: забрать и pg_dump, и volume tar’ы
mkdir -p ~/roehub-migrate

rsync -avP <linux_user>@<linux_host>:~/roehub-backup-YYYYMMDD-HHMMSS/ ~/roehub-migrate/pg/
rsync -avP <linux_user>@<linux_host>:~/roehub-volume-backup-YYYYMMDD-HHMMSS/ ~/roehub-migrate/volumes/

cd ~/roehub-migrate/pg && shasum -a 256 -c SHA256SUMS
cd ~/roehub-migrate/volumes && shasum -a 256 -c SHA256SUMS
```

Вариант 2: если SSH нестабилен — поставь `tailscale` на Linux и гоняй `rsync` по tailnet.

---

## Фаза 6 — Импорт данных на Mac Studio

Перед началом Фазы 6 убедись, что на Mac Studio подготовлен deploy bundle в `/opt/roehub`.

Почему это важно:

- команды ниже используют `/opt/roehub/docker-compose.yml`
- этот файл обычно создается шагом `Sync deploy bundle to /opt/roehub` из workflow `b/.github/workflows/deploy.yml`

Если workflow еще не запускался на Mac Studio runner, можно один раз синхронизировать bundle руками.

### 5.x (подготовка) Синхронизировать deploy bundle в `/opt/roehub` (если файлов еще нет)

На Mac Studio (под твоим основным пользователем, где лежит клон репозитория):

```bash
# ЗАЙДИ В КОРЕНЬ РЕПОЗИТОРИЯ (там где есть infra/, src/, apps/)
cd /path/to/roehub.com

REPO_ROOT="$(pwd)"

sudo install -d /opt/roehub
sudo install -d /opt/roehub/monitoring
sudo install -d /opt/roehub/market-data-src
sudo install -d /opt/roehub/market-data-src/infra

# 1) main compose
sudo install -m 0644 "${REPO_ROOT}/infra/docker/docker-compose.yml" /opt/roehub/docker-compose.yml

# 2) market-data build context
sudo rsync -a --delete "${REPO_ROOT}/src/" /opt/roehub/market-data-src/src/
sudo rsync -a --delete "${REPO_ROOT}/apps/" /opt/roehub/market-data-src/apps/
sudo rsync -a --delete "${REPO_ROOT}/configs/" /opt/roehub/market-data-src/configs/
sudo rsync -a --delete "${REPO_ROOT}/alembic/" /opt/roehub/market-data-src/alembic/
sudo rsync -a --delete "${REPO_ROOT}/migrations/" /opt/roehub/market-data-src/migrations/
sudo rsync -a --delete "${REPO_ROOT}/infra/docker/" /opt/roehub/market-data-src/infra/docker/
sudo install -m 0644 "${REPO_ROOT}/alembic.ini" /opt/roehub/market-data-src/alembic.ini
sudo install -m 0644 "${REPO_ROOT}/pyproject.toml" /opt/roehub/market-data-src/pyproject.toml

# 3) monitoring
sudo rsync -a --delete "${REPO_ROOT}/infra/monitoring/monitoring/" /opt/roehub/monitoring/

# owner for deploy user
sudo chown -R deploy:staff /opt/roehub

# checks
test -s /opt/roehub/docker-compose.yml
test -s /opt/roehub/monitoring/prometheus/prometheus.yml
test -s /opt/roehub/monitoring/blackbox/blackbox.yml
test -s /opt/roehub/market-data-src/infra/docker/Dockerfile.market_data
```

Перед импортом:

1) Убедиться, что Docker работает (`docker version`).
2) Убедиться, что `COMPOSE_PROJECT_NAME=roehub` используется везде (иначе имена volumes будут другими).

### 6.1 ClickHouse: import volumes

На Mac Studio:

```bash
set -euo pipefail

cd ~/roehub-migrate/volumes

vols=(
  roehub_ch_data
  roehub_ch_logs
  # (опционально)
  # roehub_redis_data
  # roehub_prom_data
  # grafana_data
)

for v in "${vols[@]}"; do
  echo "Create volume: ${v}"
  docker volume create "${v}" >/dev/null

  echo "Import volume: ${v}"
  docker run --rm \
    -v "${v}:/to" \
    -v "$PWD:/from" \
    alpine:3.20 \
    sh -c "tar -C /to -xf /from/${v}.tar"
done

docker volume ls | rg 'roehub_|grafana_data'
```

### 6.2 Postgres: restore из pg_dump

На Mac Studio:

```bash
set -euo pipefail

# поднять postgres (создаст empty volume, если его еще нет)
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env up -d postgres

set -a
source /etc/roehub/roehub.env
set +a

# дождаться готовности
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker exec -i roehub-postgres-1 pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# применить globals (опционально; нужно только если на старом проде были дополнительные роли)
# cat ~/roehub-migrate/pg/pg_globals.sql | docker exec -i roehub-postgres-1 psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d postgres

# restore основной БД (custom format)
cat ~/roehub-migrate/pg/roehub_pg.dump | docker exec -i -e PGPASSWORD="$POSTGRES_PASSWORD" roehub-postgres-1 \
  pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-acl
```

---

## Фаза 7 — Первый запуск стека на Mac Studio (smoke)

На Mac Studio:

1) Первый деплой лучше сделать руками, чтобы видеть ошибки (после этого можно отдать на Actions).

```bash
export COMPOSE_PROJECT_NAME=roehub
export MARKET_DATA_BUILD_CONTEXT=/opt/roehub/market-data-src
export MARKET_DATA_DOCKERFILE=infra/docker/Dockerfile.market_data

docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env --profile ui up -d --build --remove-orphans
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env --profile ui ps
```

2) Быстрые проверки:

```bash
curl -fsS http://127.0.0.1:8080/ | head
curl -i http://127.0.0.1:8080/api/auth/current-user

docker logs --tail=200 roehub-postgres-1
docker logs --tail=200 roehub-clickhouse-1
docker logs --tail=200 prometheus
```

3) Проверка scrape market-data из Prometheus:

```bash
docker exec -it prometheus wget -T 2 -qO- http://market-data-ws-worker:9201/metrics | head
docker exec -it prometheus wget -T 2 -qO- http://market-data-scheduler:9202/metrics | head
```

Если поднялось — можно включать deploy через GitHub Actions.

---

## Фаза 8 — Публичный доступ `roehub.com` (TLS)

### Вариант A: Cloudflare Tunnel

На Mac Studio:

```bash
brew install cloudflare/cloudflare/cloudflared
cloudflared --version
```

Дальше:

1) Залогиниться и выдать cloudflared доступ к Cloudflare аккаунту:

```bash
cloudflared tunnel login
```

2) Создать tunnel:

```bash
cloudflared tunnel create roehub-mac-studio
```

3) Создать конфиг `~/.cloudflared/config.yml`:

```yaml
tunnel: roehub-mac-studio
credentials-file: /Users/deploy/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: roehub.com
    service: http://127.0.0.1:8080
  - hostname: www.roehub.com
    service: http://127.0.0.1:8080
  - service: http_status:404
```

4) Привязать DNS записи к tunnel:

```bash
cloudflared tunnel route dns roehub-mac-studio roehub.com
cloudflared tunnel route dns roehub-mac-studio www.roehub.com
```

5) Запустить как сервис:

```bash
sudo cloudflared service install
sudo launchctl list | rg cloudflared || true
```

Проверка:

```bash
curl -I https://roehub.com/
```

### Вариант B: Статический IP + Caddy

1) DNS:

- `A roehub.com -> 185.155.18.21`
- `A www.roehub.com -> 185.155.18.21`

Проверка:

```bash
dig +short roehub.com A
dig +short www.roehub.com A
```

2) Убедиться, что входящие 80/443 реально приходят на Mac Studio (если есть внешний firewall/маршрутизатор — открыть там).

3) На Mac Studio:

```bash
brew install caddy
sudo mkdir -p /etc/caddy
```

Примечание (Homebrew service):

- `brew services` для `caddy` по умолчанию использует конфиг `/opt/homebrew/etc/Caddyfile`.
- Чтобы не путаться, можно хранить "канонический" конфиг в `/etc/caddy/Caddyfile` и синхронизировать в Homebrew path:

```bash
sudo cp /etc/caddy/Caddyfile /opt/homebrew/etc/Caddyfile
```

4) Убедиться, что gateway слушает только localhost (рекомендуется):

В `/etc/roehub/roehub.env`:

```bash
GATEWAY_HOST_BIND=127.0.0.1
GATEWAY_HOST_PORT=8080
```

5) Создать `/etc/caddy/Caddyfile`:

```caddyfile
roehub.com, www.roehub.com {
  encode zstd gzip
  reverse_proxy 127.0.0.1:8080
}
```

6) Запустить Caddy как сервис:

```bash
sudo brew services start caddy
sudo brew services list | rg caddy
```

Проверка:

```bash
curl -I https://roehub.com/
```

Если macOS Firewall включен и блокирует входящие — разреши `caddy` (опционально):

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$(brew --prefix)/bin/caddy" || true
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$(brew --prefix)/bin/caddy" || true
```

---

## Фаза 9 — Cutover и отключение Linux

1) Убедиться, что Mac Studio стабильно обслуживает `https://roehub.com`.
2) Убедиться, что market-data и метрики живы (Prometheus scrape ok).
3) Отключить/удалить Linux runner (в GitHub Settings → Actions → Runners).
4) Остановить Linux сервер.

---

## Что обновить в документации (runbooks)

- `docs/runbooks/roehub-ui-autostart-systemd.md`
  - пометить как Linux-only;
  - добавить ссылку на этот документ для macOS.
- `docs/runbooks/market-data-autonomous-docker.md`
  - убрать Linux-специфичные пути (`/home/...`, `/opt/actions-runner/...`) или сделать их OS-agnostic;
  - добавить пример для Mac Studio (layout `/opt/roehub`, env `/etc/roehub/roehub.env`).
- `docs/runbooks/web-ui-gateway-same-origin.md`
  - добавить секцию про production ingress (Cloudflare Tunnel или Caddy), чтобы было понятно, где TLS.

## Пост-миграция (рекомендуется, но не блокер)

- Настроить регулярные бэкапы:
  - Postgres: `pg_dump`/volume snapshot;
  - ClickHouse: volume snapshot/backup strategy.
- Включить GitHub Environment `production` с manual approve для deploy.
- Сохранить emergency-доступ: Screen Sharing (только как запасной канал) + физический доступ.
