# Web UI local same-origin (WEB-EPIC-02)

Runbook для local/dev same-origin потока `apps/web -> /api/* -> api` c Keycloak OIDC и Roehub session cookies.

Статус:

- `gateway` удален из репозитория;
- production same-origin делает `Caddy` на `VPS`;
- local/dev same-origin обеспечивает `apps/web` через встроенный `/api/*` proxy.

## Обязательный файл окружения

Используйте env, эквивалентный production:

- runtime owner path: `/Users/daniildegtyarev/.config/roehub/roehub.env`
- локальный dev path (пример): `./infra/docker/.env.local`

Минимальные ключи для UI-профиля:

- `POSTGRES_PASSWORD`
- `WEB_API_BASE_URL`
- `WEB_API_UPSTREAM_URL`
- `KEYCLOAK_BASE_URL`
- `KEYCLOAK_REALM`
- `KEYCLOAK_CLIENT_ID`
- `KEYCLOAK_CLIENT_SECRET`
- `KEYCLOAK_REDIRECT_URI`
- `KEYCLOAK_LOGOUT_REDIRECT_URI`
- `IDENTITY_SESSION_COOKIE_NAME`
- `IDENTITY_SESSION_IDLE_TTL_SECONDS`
- `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`

DSN-ключи `IDENTITY_PG_DSN`, `POSTGRES_DSN`, `STRATEGY_PG_DSN` можно не задавать:
`docker-compose.yml` в UI-профиле собирает их как conninfo из
`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.

Важно: `TELEGRAM_BOT_TOKEN` не участвует в web/API auth path.

## Запуск dev одной командой

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
  --profile ui up -d --build api web db-bootstrap
```

Ожидаемый адрес:

- `http://127.0.0.1:8010`

Быстрые проверки:

```bash
docker compose -f infra/docker/docker-compose.yml \
  --env-file /Users/daniildegtyarev/.config/roehub/roehub.env \
  --profile ui ps

curl -i http://127.0.0.1:8010/api/auth/current-user
curl -i http://127.0.0.1:8010/assets/site.css
```

## Поведение bootstrap БД

`db-bootstrap` запускается перед `api` и выполняет:

1. `python -m apps.migrations.bootstrap_main`
   - `IDENTITY_PG_DSN`/`POSTGRES_DSN` по умолчанию передаются как conninfo:
     `host=postgres port=5432 dbname=<POSTGRES_DB> user=<POSTGRES_USER> password=<POSTGRES_PASSWORD>`
2. SQL baseline identity в `IDENTITY_PG_DSN`:
   - `0001_identity_v1.sql`
   - `0002_identity_2fa_totp_v1.sql` (историческая таблица, не используется как auth source)
   - `0003_identity_exchange_keys_v1.sql`
3. Migration `0004_identity_exchange_keys_v2.sql`.
4. Keycloak cutover migration `0005_identity_keycloak_cutover_v1.sql`:
   - добавляет `identity_users.keycloak_subject`
   - создаёт `identity_sessions`
5. Alembic head в `POSTGRES_DSN`:
   - `python -m apps.migrations.main --dsn "$POSTGRES_DSN"`

Сервис одноразовый (`restart: "no"`). Если bootstrap падает, `api` не стартует.

## Keycloak OIDC configuration

Для dev/prod same-origin web потока:

- `KEYCLOAK_REDIRECT_URI` должен совпадать с callback URL web (`/auth/callback`);
- `KEYCLOAK_LOGOUT_REDIRECT_URI` должен совпадать с post-logout URL web (`/login`).

Примеры:

- local/dev: `http://127.0.0.1:8010/auth/callback`, `http://127.0.0.1:8010/login`
- prod: `https://roehub.com/auth/callback`, `https://roehub.com/login`

Если заданы explicit OIDC URLs (`KEYCLOAK_AUTH_URL`, `KEYCLOAK_TOKEN_URL`,
`KEYCLOAK_END_SESSION_URL`, `KEYCLOAK_INTROSPECTION_URL`), они должны соответствовать тому же realm.

Browser auth-cookie хранит только opaque Roehub session id:

- `IDENTITY_SESSION_COOKIE_NAME` (обычно `roehub_session_id`)
- `IDENTITY_SESSION_IDLE_TTL_SECONDS`
- `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`

## Примечание по маршрутизации

`apps/web` принимает browser-side `/api/*` и проксирует их в upstream API без `/api` префикса:

- `/api/<path>` на web -> `/<path>` на API upstream.

В production эту же семантику на публичном edge реализует `VPS Caddy`.

## Production Caddy `/api/*` forwarded-origin contract

Production browser mutations depend on the backend seeing the public browser
origin, not only the internal Mac Studio upstream host. The active VPS Caddy
`/api/*` reverse proxy must forward:

```caddy
header_up X-Forwarded-Host {host}
header_up X-Forwarded-Proto {scheme}
header_up X-Roehub-Forwarded-Host {host}
header_up X-Roehub-Forwarded-Proto {scheme}
```

These headers are security-relevant. They let the backend accept a same-origin
browser request that has `Referer: https://roehub.com/settings` but no `Origin`
header, while true cross-origin requests remain rejected with
`csrf_origin_mismatch`.

`X-Forwarded-Host` and `X-Forwarded-Proto` are the standard proxy context.
`X-Roehub-Forwarded-Host` and `X-Roehub-Forwarded-Proto` are the edge-owned copy
used after the VPS -> Tailscale Serve hop, where standard forwarded headers can
be rewritten to the Mac Studio upstream host. Caddy must overwrite these values;
do not pass through client-provided values.

Verification commands:

```bash
curl -fsS https://roehub.com/__edge_id

grep -F 'header_up X-Forwarded-Host {host}' infra/caddy/Caddyfile.vps
grep -F 'header_up X-Forwarded-Proto {scheme}' infra/caddy/Caddyfile.vps
grep -F 'header_up X-Roehub-Forwarded-Host {host}' infra/caddy/Caddyfile.vps
grep -F 'header_up X-Roehub-Forwarded-Proto {scheme}' infra/caddy/Caddyfile.vps

ssh "$PROD_VPS_USER@$PROD_VPS_HOST" \
  "grep -F 'header_up X-Forwarded-Host {host}' /etc/caddy/Caddyfile"
ssh "$PROD_VPS_USER@$PROD_VPS_HOST" \
  "grep -F 'header_up X-Forwarded-Proto {scheme}' /etc/caddy/Caddyfile"
ssh "$PROD_VPS_USER@$PROD_VPS_HOST" \
  "grep -F 'header_up X-Roehub-Forwarded-Host {host}' /etc/caddy/Caddyfile"
ssh "$PROD_VPS_USER@$PROD_VPS_HOST" \
  "grep -F 'header_up X-Roehub-Forwarded-Proto {scheme}' /etc/caddy/Caddyfile"
ssh "$PROD_VPS_USER@$PROD_VPS_HOST" "caddy validate --config /etc/caddy/Caddyfile"
```

Do not repair this by weakening backend CSRF checks. If the active Caddy config
is missing these headers, sync `infra/caddy/Caddyfile.vps`, validate it, reload
Caddy, then prove the authenticated browser flow again.

## Связанные документы

- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- `docs/runbooks/keycloak-local-setup-and-ops.md`
