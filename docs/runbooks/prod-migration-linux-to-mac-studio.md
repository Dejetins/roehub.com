# Переезд продакшена с Linux на Mac Studio + VPS edge

Актуальный production runbook для Roehub после смены решения по ingress:

- `Mac Studio` остается приватным backend/data/compute host.
- публичный web ingress и TLS живут на отдельном `VPS`.
- старый Linux сервер выключается после подтвержденного cutover.

Документ заменяет предыдущий план, в котором `Mac Studio` должен был сам принимать публичные `80/443`.

## Целевая архитектура

### Роли хостов

- `VPS` (`155.212.170.144`, Ubuntu 24.04) - единственный публичный edge:
  - `Caddy` как host-service;
  - `web` SSR процесс/контейнер;
  - `Let's Encrypt`;
  - reverse proxy `/api/*` на `Mac Studio` через `Tailscale`.
- `Mac Studio` (`tailscale: 100.74.213.43`) - приватный backend/data/compute host:
  - `api`;
  - `postgres`;
  - `clickhouse`;
  - `redis`;
  - market-data workers/scheduler;
  - `grafana`, `prometheus`, `blackbox`;
  - `Colima` под пользователем `daniildegtyarev`;
  - self-hosted GitHub Actions runner.
- старый Linux сервер - только источник миграции и временный tailnet node до полного вывода из эксплуатации.

### Production flow

```text
Browser
  -> https://roehub.com
  -> VPS Caddy
     -> /            -> web on VPS
     -> /api/*       -> API on Mac Studio over Tailscale
                          -> Postgres / ClickHouse / Redis / workers on Mac Studio
```

### Invariants

- публичный DNS (`roehub.com`, `www.roehub.com`) указывает только на `VPS`.
- `Mac Studio` не принимает публичные `80/443`.
- SSH наружу на `Mac Studio` не открывается.
- администрирование обоих хостов - через `Tailscale + SSH`.
- monitoring и admin endpoints не публикуются в интернет.
- production deploy разделен на два независимых контура:
  - backend deploy -> `Mac Studio`;
  - web/edge deploy -> `VPS`.

## Что больше не делаем

Следующие шаги из старой версии runbook больше не являются production target:

- не поднимаем публичный `Caddy` на `Mac Studio`;
- не держим `web` и `gateway` как production ingress на `Mac Studio`;
- не пробрасываем `80/443` с домашнего роутера на `Mac Studio`;
- не используем `Cloudflare Tunnel` как основной production ingress;
- не используем `ispmanager` для production deployment;
- не пытаемся завершить старый план "Mac Studio как единственный публичный origin".

Если что-то из этого уже было временно настроено в процессе диагностики, после cutover это нужно удалить.

## Что считаем правильным production design

### Edge

- `Caddy` работает на `VPS` как системный сервис.
- `www.roehub.com` делает `301` на `roehub.com`.
- `Caddy` сам получает и обновляет сертификаты `Let's Encrypt`.

### Web

- production web живет на `VPS`.
- current SSR `apps/web` сохраняется, переписывать UI в static site сейчас не нужно.
- отдельный `gateway` полностью удален из runtime path; same-origin реализуют `VPS Caddy` и
  встроенный `/api/*` proxy в `apps/web` для local/dev.

Почему gateway можно убрать из production:

- `apps/web` сам отдает `/assets/*` через `StaticFiles`;
- same-origin routing `/api/*` можно реализовать напрямую в `Caddy` на `VPS`;
- лишний hop и отдельный контейнер на публичном edge не нужны.

### Backend

- `Mac Studio` держит stateful и compute-нагрузку.
- API не публикуется напрямую в интернет.
- внешний трафик к API идет только через `VPS`.

### Deploy

Рекомендуемая production-модель:

- GitHub-hosted CI builds multi-arch image(s) и публикует их в `GHCR`.
- `Mac Studio` не собирает production source bundle на лету, а делает `pull` готовых image tags.
- `VPS` тоже делает `pull` из `GHCR`, а не build из git checkout.

Это профессиональнее, чем текущий host-build deploy, потому что:

- prod-хосты становятся thin runtime targets;
- deploy детерминированнее;
- одна и та же версия image разворачивается на `linux/amd64` (`VPS`) и `linux/arm64` (`Mac Studio`) через multi-arch manifest;
- rollback проще.

## Что нужно изменить в репозитории перед финальным cutover

Текущие production артефакты в репозитории еще соответствуют старому плану и должны быть переработаны.

### 1. Разделить production deployment на backend и web

Текущий `infra/docker/docker-compose.yml` смешивает:

- stateful backend services;
- `api`;
- `web`;
- local/dev `web` routing.

Для новой схемы нужен split по ответственности.

Рекомендуемый target:

- `infra/docker/docker-compose.backend.yml`
  - `postgres`
  - `clickhouse`
  - `redis`
  - `db-bootstrap`
  - `api`
  - `market-data-ws-worker`
  - `market-data-scheduler`
  - `grafana`
  - `prometheus`
  - `blackbox`
- `infra/docker/docker-compose.web.prod.yml`
  - `web`

Текущий dev/local `gateway` уже удален из репозитория и больше не участвует в deploy path.

### 2. Убрать production dependency от `--profile ui` на Mac Studio

Сейчас `api`, `web`, `db-bootstrap` привязаны к `profiles: ["ui"]`.
Это неудобно и ведет к путанице.

Для production target:

- `api` и `db-bootstrap` должны стать backend-сервисами;
- `web` должен переехать в отдельный web compose для `VPS`;
- local/dev same-origin должен работать без отдельного proxy-контейнера.

### 3. Перевести production deploy на GHCR images

Рекомендуемый target:

- один multi-arch app image для:
  - `api`
  - `web`
  - `db-bootstrap`
  - workers/scheduler
- third-party images продолжают тянуться из upstream registries.

Пример naming policy:

- `ghcr.io/<owner>/roehub-app:<git-sha>`
- `ghcr.io/<owner>/roehub-app:main`

### 4. Разделить workflow deploy

Текущий `.github/workflows/deploy.yml` больше не отражает production topology.

Нужен target из двух workflow:

- `.github/workflows/publish-app-image.yml`
  - GitHub-hosted build/push multi-arch app image в `GHCR`
- `.github/workflows/deploy-backend.yml`
  - runs-on: `[self-hosted, macOS, ARM64, roehub, prod, mac-studio]`
  - deploy backend stack на `Mac Studio`
- `.github/workflows/deploy-web.yml`
  - runs-on: `ubuntu-latest`
  - SSH deploy на `root@VPS`
  - deploy/reload `web` + `Caddy`

Опционально:

- GitHub Environment `production` с manual approval.

### 4.1 GitHub variables/secrets для web deploy

Минимум для `.github/workflows/deploy-web.yml`:

- repository variable `PROD_VPS_HOST`
- repository variable `PROD_VPS_USER`
- repository secret `PROD_VPS_SSH_KEY`

Важно:

- `PROD_VPS_SSH_KEY` должен быть отдельным deploy key без passphrase;
- текущий локальный ключ с passphrase удобен для ручной работы, но не подходит для GitHub Actions.

## Фаза 0 - Стоп-линия и инвентаризация старого Linux

Цель: зафиксировать, что старый Linux больше не источник production traffic и нужен только до завершения cutover.

На старом Linux:

```bash
docker --version
docker compose version
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

Проверить, что уже забраны:

- `/etc/roehub/roehub.env`
- Postgres dump / restore assets
- ClickHouse / Redis / Prometheus / Grafana volume exports
- контрольные `sha256`

Если данные уже успешно восстановлены на `Mac Studio`, старый Linux больше не должен использоваться как fallback runtime.

## Фаза 1 - Подготовить Mac Studio как private backend host

### 1.1 Пользователи и доступ

Рекомендуемая модель:

- текущий macOS admin user `daniildegtyarev` является runtime owner для `Colima`, backend stack,
  `tailscale serve`, и локальных ops-команд;
- self-hosted GitHub runner должен быть переоформлен на этого же runtime owner.

SSH на `Mac Studio`:

- только через `Tailscale`;
- наружу `22/tcp` не открывать.

### 1.2 Tailscale

На `Mac Studio` должен быть стабильный tailnet access.

Текущий node:

- `macstudio-daniil`
- `100.74.213.43`

Проверка:

```bash
tailscale status
tailscale ip -4
```

### 1.3 Colima и Docker

`Colima` остается под пользователем `daniildegtyarev`.

Обязательные требования:

- один runtime owner: `daniildegtyarev`;
- runtime restart после reboot;
- mount `/opt/roehub` в VM (`virtiofs`), иначе monitoring bind-mounts ломаются.

Проверка под current runtime owner:

```bash
colima status
docker version
docker compose version
docker context ls
```

### 1.4 Серверный layout

Оставляем canonical layout:

- `/opt/roehub` - deploy bundle/runtime manifests
- `/Users/daniildegtyarev/.config/roehub/roehub.env` - backend secrets for runtime owner

Минимум:

```bash
sudo mkdir -p /opt/roehub
sudo chown -R daniildegtyarev:staff /opt/roehub
sudo chmod 755 /opt/roehub

mkdir -p /Users/daniildegtyarev/.config/roehub
sudo cp /etc/roehub/roehub.env /Users/daniildegtyarev/.config/roehub/roehub.env
sudo chown daniildegtyarev:staff /Users/daniildegtyarev/.config/roehub/roehub.env
chmod 600 /Users/daniildegtyarev/.config/roehub/roehub.env
```

### 1.5 Backend only на Mac Studio

Production target на `Mac Studio`:

- оставить:
  - `api`
  - `postgres`
  - `clickhouse`
  - `redis`
  - `market-data-ws-worker`
  - `market-data-scheduler`
  - `grafana`
  - `prometheus`
  - `blackbox`
  - `Colima`
  - `Tailscale`
  - self-hosted runner
- удалить после cutover:
  - публичный `Caddy`
  - `web`
  - `gateway`
  - любые временные публичные ingress-костыли

### 1.6 Чего не должно быть на Mac Studio после cutover

- не должно быть production зависимости от домашнего роутера для `80/443`;
- не должно быть активного публичного `Caddy`;
- не должно быть production `web` контейнера;
- не должно быть production `gateway` контейнера;
- в `/etc/hosts` на рабочих машинах не должно оставаться временных записей для `roehub.com`.

## Фаза 2 - Подготовить VPS как public edge

### 2.1 Базовые факты

Текущий `VPS`:

- IP: `155.212.170.144`
- OS: `Ubuntu 24.04`
- size: `1 vCPU / 2 GB RAM / 40 GB SSD`
- deploy user: `root`
- `ispmanager` не используем

### 2.2 Базовая инициализация VPS

Под `root`:

```bash
apt-get update
apt-get install -y ca-certificates curl git jq rsync docker.io docker-compose-plugin
systemctl enable --now docker
docker version
docker compose version
```

### 2.3 Tailscale на VPS

Цель: `VPS` должен видеть `Mac Studio` по tailnet.

Установка:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
tailscale up
tailscale status
tailscale ip -4
```

После этого зафиксировать:

- Tailscale IPv4 `VPS`
- имя node `VPS` в tailnet

Проверка связи с `Mac Studio`:

```bash
ping 100.74.213.43
```

### 2.4 Caddy на VPS

`Caddy` на `VPS` - host-service, не контейнер.

Установка:

```bash
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt-get update
apt-get install -y caddy
systemctl enable caddy
```

### 2.5 Runtime layout на VPS

Рекомендуемый layout:

- `/opt/roehub-web` - docker compose + env для `web`
- `/etc/caddy/Caddyfile` - public ingress config

```bash
mkdir -p /opt/roehub-web
```

## Фаза 3 - DNS и публичный edge

### 3.1 DNS target

После cutover:

- `A roehub.com -> 155.212.170.144`
- `A www.roehub.com -> 155.212.170.144`

Проверка:

```bash
dig +short roehub.com A
dig +short www.roehub.com A
```

### 3.2 Production Caddy config на VPS

Production target для `Caddy`:

```caddyfile
http://roehub.com {
  route {
    handle /__edge_id {
      respond "vps-edge\n" 200
    }

    redir https://roehub.com{uri} 308
  }
}

http://www.roehub.com {
  redir https://roehub.com{uri} 301
}

roehub.com {
  encode zstd gzip

  handle /__edge_id {
    respond "vps-edge\n" 200
  }

  handle_path /api/* {
    reverse_proxy https://macstudio-daniil.tail0ebbbc.ts.net {
      header_up Host macstudio-daniil.tail0ebbbc.ts.net
      transport http {
        tls_server_name macstudio-daniil.tail0ebbbc.ts.net
      }
    }
  }

  reverse_proxy 127.0.0.1:8010
}

www.roehub.com {
  redir https://roehub.com{uri} 301
}
```

Смысл:

- `/api/*` strip'ается на edge и уходит в `Mac Studio API`;
- все остальные маршруты идут на `web` на `VPS`.

Важно:

- это заменяет старую `nginx gateway` production схему;
- `api` на `Mac Studio` по-прежнему не знает про `/api` prefix.

### 3.3 TLS

После перевода DNS на `VPS`:

```bash
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy
curl -I https://roehub.com/
```

## Фаза 4 - Backend runtime на Mac Studio

### 4.1 State services

На `Mac Studio` должны жить:

- `postgres`
- `clickhouse`
- `redis`
- metrics stack

Эти данные уже были перенесены с Linux и проверены по `sha256`.

### 4.2 API

Production API должен обслуживаться на `Mac Studio` и быть доступен:

- локально на `Mac Studio`;
- с `MacBook` по private network;
- с `VPS` по `Tailscale`.

Важно:

- API не должен зависеть от `web` или `gateway` на `Mac Studio`;
- production API должен считаться backend service, а не частью `ui` profile.

### 4.3 Monitoring

`Grafana`, `Prometheus`, `Blackbox` остаются на `Mac Studio` и не публикуются наружу.

Доступ к ним:

- через `Tailscale`;
- или через SSH port-forward.

## Фаза 5 - Production images и registry

### 5.1 Рекомендуемый target

Перед cutover нужно перевести production deploy на images из `GHCR`.

Target:

- CI на GitHub-hosted runner собирает multi-arch app image;
- пушит его в `GHCR`;
- backend и web deploy используют один и тот же tag.

### 5.2 Почему это обязательная часть новой схемы

Без `GHCR` получится два разных anti-pattern:

- `Mac Studio` строит production runtime из локального checkout;
- `VPS` строит web из отдельного checkout по SSH.

Это неудобно для:

- traceability;
- rollback;
- repeatability;
- гарантии одинаковой версии между web и backend.

## Фаза 6 - Разделить deploy workflows

### 6.1 Backend deploy workflow

Цель:

- deploy backend only на `Mac Studio` через self-hosted runner.

Target обязанности workflow:

- login в `GHCR`;
- pull app image и внешних service images;
- deploy backend compose;
- smoke test `api` локально на `Mac Studio`.

### 6.2 Web deploy workflow

Цель:

- deploy `web` на `VPS` через GitHub-hosted runner по SSH.

Target обязанности workflow:

- login в `GHCR`;
- sync `docker-compose.web.prod.yml`, `Caddyfile`, и env на `VPS`;
- `docker compose pull && docker compose up -d` для `web`;
- `Caddyfile` validate/reload;
- smoke test `https://roehub.com/` и `https://roehub.com/api/auth/current-user`.

### 6.3 Runner topology

Финальный target:

- self-hosted runner только на `Mac Studio`;
- на `VPS` self-hosted runner не нужен;
- старый Linux runner должен быть удален после завершения cutover.

## Фаза 7 - Поднять self-hosted runner на Mac Studio

Это обязательный шаг для backend deploy на `Mac Studio`.

На `Mac Studio` под текущим runtime owner:

```bash
mkdir -p /opt/actions-runner/roehub
cd /opt/actions-runner/roehub
```

Дальше использовать команды из GitHub:

`Repo -> Settings -> Actions -> Runners -> New self-hosted runner -> macOS`

Рекомендуемые labels:

- `roehub`
- `prod`
- `mac-studio`

Если runner был исторически зарегистрирован под другим пользователем, его нужно удалить из GitHub,
перерегистрировать под `daniildegtyarev`, и затем проверить, что labels и runtime окружение остались теми же.

Если на `Mac Studio` включен automatic login под `daniildegtyarev`, runner можно держать в той же
user-session модели, что и `Colima`/`tailscale serve`, и использовать штатный `svc.sh`.

Базовая процедура перерегистрации:

```bash
./config.sh --url https://github.com/Dejetins/roehub.com --token <fresh_token> --name mac-studio-prod --labels roehub,prod,mac-studio --work _work --unattended --replace
./svc.sh install
./svc.sh start
./svc.sh status
```

Важно:

- runner и `Colima` должны жить под одним и тем же пользователем `daniildegtyarev`.
- ожидаемый статус в GitHub UI: `Online` / `Idle`.

## Фаза 8 - Smoke tests для новой topology

### 8.1 Проверки на Mac Studio

```bash
export ROEHUB_ENV_FILE=/Users/daniildegtyarev/.config/roehub/roehub.env
docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" ps
curl -i http://127.0.0.1:8000/auth/current-user
```

Ожидаемо:

- compose stack healthy;
- `401` без cookie на `/auth/current-user`.

### 8.2 Проверки с VPS

```bash
curl -i https://macstudio-daniil.tail0ebbbc.ts.net/auth/current-user
curl -I https://roehub.com/
curl -i https://roehub.com/api/auth/current-user
```

Ожидаемо:

- private API через `Tailscale Serve` reachable;
- главная страница отдается с `VPS`;
- `/api/auth/current-user` возвращает `401` без cookie, но не `502`.

### 8.3 Проверки браузером

Проверить с внешней сети:

- открывается `https://roehub.com/`;
- login flow работает;
- защищенные страницы (`/strategies`, `/backtests`) открываются после login;
- cookies живут на `roehub.com` same-origin path.

## Фаза 9 - Cleanup на Mac Studio после cutover

Cleanup делаем не мгновенно, а staged:

1. Сначала перевести production traffic на `VPS`.
2. Подержать новый контур стабильно минимум один рабочий цикл.
3. Только потом удалить устаревший public ingress с `Mac Studio`.

### 9.1 Что удалить с Mac Studio

- `Caddy` как публичный сервис;
- `/opt/homebrew/etc/Caddyfile`, если использовался только для публичного ingress;
- production `web` container;
- production `gateway` container;
- любые `hosts` overrides для `roehub.com` на админских машинах;
- router port-forward `80/443 -> Mac Studio`.

### 9.2 Что оставить на Mac Studio

- `api`
- data services
- workers
- monitoring
- `Colima`
- `Tailscale`
- self-hosted runner

### 9.3 Cleanup commands

Примерно:

```bash
sudo brew services stop caddy || true
sudo brew services list | rg caddy || true
docker ps --format 'table {{.Names}}\t{{.Status}}' | rg 'web|gateway' || true
```

Финальный cleanup зависит от того, как именно были временно подняты `web/gateway/caddy` в процессе диагностики.

## Фаза 10 - Когда можно выключать старый Linux сервер

Старый Linux можно выключать только после выполнения всех условий ниже.

### Hard criteria

- данные уже восстановлены и проверены на `Mac Studio`;
- `VPS` подключен к tailnet и стабильно видит `Mac Studio`;
- DNS `roehub.com` и `www.roehub.com` переведены на `VPS`;
- `Caddy` на `VPS` выдал валидный `Let's Encrypt` сертификат;
- `https://roehub.com/` работает из внешней сети;
- `https://roehub.com/api/auth/current-user` доходит до API и не дает `502`;
- backend deploy через self-hosted runner на `Mac Studio` работает;
- web deploy через GitHub-hosted runner по SSH на `VPS` работает;
- старый Linux больше не обслуживает production traffic;
- старый Linux runner удален из GitHub.

### Recommended final actions

1. Сделать финальный backup/снимок старого Linux.
2. Удалить старый Linux runner из GitHub Settings.
3. Остановить приложения на старом Linux.
4. Выключить сервер.

Если политика проекта - "без fallback", это не мешает сделать финальный snapshot перед выключением. Snapshot не считается рабочим fallback, но снижает операционный риск.

## Что уже можно считать выполненным из старого runbook

Для текущего проекта фактически уже были выполнены или в основном выполнены:

- подготовка `Mac Studio` как runtime host;
- перенос и проверка данных `Postgres` и volumes;
- `Tailscale` доступ между вашими машинами;
- `Colima` под `deploy`;
- проверка, что stateful stack поднимается на `Mac Studio`.

Что больше не нужно завершать по старой версии runbook:

- публичный ingress через `Caddy` на `Mac Studio`;
- домашний роутер как production edge;
- попытки довести production сайт на домашнем IP.

Что реально осталось сделать теперь:

- включить `VPS` в production topology;
- разрезать production deploy на `backend` и `web`;
- перевести deploy на `GHCR`;
- поднять self-hosted runner на `Mac Studio`;
- перевести DNS на `VPS`;
- удалить старый Linux runner и выключить старый Linux.

## Связанные документы, которые тоже нужно обновить

- `docs/runbooks/web-ui-gateway-same-origin.md`
  - переписать под local/dev same-origin без отдельного gateway;
  - зафиксировать, что production same-origin делает `Caddy` на `VPS`.
- `docs/runbooks/roehub-ui-autostart-systemd.md`
  - пометить как устаревший для новой production topology;
  - либо переписать под `VPS web only`, либо архивировать.
- `.github/workflows/deploy.yml`
  - удалить и заменить на `publish-app-image.yml`, `deploy-backend.yml`, `deploy-web.yml`.
- `infra/docker/docker-compose.yml`
  - перестать использовать как единый production manifest для двух разных хостов.

## Runtime operations

Для текущих operational команд см.:

- `docs/runbooks/mac-studio-backend-operations.md`

## Post-cutover checklist

- `roehub.com` публично работает с `VPS`.
- web SSR работает на `VPS`.
- API отвечает с `Mac Studio` через `VPS Caddy`.
- monitoring доступен только приватно.
- `Mac Studio` не торчит публично в интернет.
- старый Linux выключен.
- GitHub Actions deploy для backend и web работают независимо.
