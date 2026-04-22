# Keycloak Local Setup and Ops

Runbook для локального/стендового Keycloak, который используется как единственный auth source для Roehub API/Web.

## Цель

- поднять Keycloak как OIDC provider;
- подключить Keycloak к Postgres;
- настроить realm/client/users/OTP policy;
- проверить Roehub flow `/auth/login -> /auth/callback -> /auth/current-user -> /auth/logout`.

## Предпосылки

- `Mac Studio` host;
- доступ к Postgres (`127.0.0.1:5432`);
- Roehub env (`/Users/daniildegtyarev/.config/roehub/roehub.env`).

## 1) Подготовить БД Keycloak (Postgres)

```bash
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a

PGPASSWORD="${POSTGRES_PASSWORD}" psql -h 127.0.0.1 -p 5432 -U "${POSTGRES_USER}" -d postgres -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'keycloak') THEN
    CREATE ROLE keycloak LOGIN PASSWORD 'change-me-keycloak';
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'keycloak') THEN
    CREATE DATABASE keycloak OWNER keycloak;
  END IF;
END
$$;
SQL
```

## 2) Установить Keycloak (native)

ASSUMPTION: используется официальный дистрибутив Keycloak из [keycloak.org](https://www.keycloak.org/downloads).

```bash
mkdir -p /opt/roehub/keycloak
cd /opt/roehub/keycloak

# пример: скачайте актуальный архив вручную и распакуйте его сюда
# итоговый путь: /opt/roehub/keycloak/keycloak-<version>/bin/kc.sh
```

## 3) Первый запуск (bootstrap admin)

```bash
cd /opt/roehub/keycloak/keycloak-<version>

export KC_DB=postgres
export KC_DB_URL_HOST=127.0.0.1
export KC_DB_URL_PORT=5432
export KC_DB_URL_DATABASE=keycloak
export KC_DB_USERNAME=keycloak
export KC_DB_PASSWORD=change-me-keycloak

export KEYCLOAK_ADMIN=admin
export KEYCLOAK_ADMIN_PASSWORD=change-me-admin

./bin/kc.sh start-dev --http-port 18080
```

Откройте [http://127.0.0.1:18080](http://127.0.0.1:18080) и войдите admin-учёткой.

## 4) Настроить realm/client для Roehub

Создайте realm: `roehub`.

Создайте OIDC client: `roehub-api` (confidential):

- `Client authentication`: ON;
- `Standard flow`: ON;
- `Valid redirect URIs`:
  - `http://127.0.0.1:8010/auth/callback`
  - `https://roehub.com/auth/callback`
- `Valid post logout redirect URIs`:
  - `http://127.0.0.1:8010/login`
  - `https://roehub.com/login`
- `Web origins`: `+` или конкретные origin по policy.

Сохраните client secret и задайте его в Roehub env (`KEYCLOAK_CLIENT_SECRET`).

## 5) Включить OTP policy в Keycloak

Realm settings -> Authentication:

- включите OTP policy (TOTP);
- в Browser flow добавьте `OTP Form` как Required/Conditional по вашей policy;
- для тестовых пользователей отметьте required action `Configure OTP`.

Важно: 2FA реализуется в Keycloak. Локальные `/2fa/*` endpoints в Roehub отсутствуют.

## 6) Настроить Roehub env

В `/Users/daniildegtyarev/.config/roehub/roehub.env`:

```bash
KEYCLOAK_BASE_URL=http://127.0.0.1:18080
KEYCLOAK_REALM=roehub
KEYCLOAK_CLIENT_ID=roehub-api
KEYCLOAK_CLIENT_SECRET=<client-secret>
KEYCLOAK_REDIRECT_URI=http://127.0.0.1:8010/auth/callback
KEYCLOAK_LOGOUT_REDIRECT_URI=http://127.0.0.1:8010/login
IDENTITY_SESSION_COOKIE_NAME=roehub_session_id
IDENTITY_SESSION_IDLE_TTL_SECONDS=1800
IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS=43200
```

## 7) Перезапустить Roehub API/Web

```bash
bash scripts/macos/reload_launchd_services.sh prod
```

## 8) Smoke checks

```bash
# login redirect
curl -i "http://127.0.0.1:8000/auth/login?next=%2Fstrategies"

# anonymous current-user
curl -i http://127.0.0.1:8000/auth/current-user

# metrics auth surface
curl -fsS http://127.0.0.1:8000/metrics | rg 'http_requests_total\{.*path="/auth/(login|callback|logout|current-user)"'
```

Ожидания:

- `/auth/login` -> `307` на Keycloak authorize URL;
- `/auth/current-user` без cookie -> `401`;
- после browser login/callback появляется cookie `roehub_session_id`;
- `/auth/logout` ревокает сессию и удаляет cookie.

## 9) Operation commands (Keycloak)

```bash
# local process (если запуск вручную в shell)
pkill -f 'kc.sh start-dev' || true

# health endpoint
curl -fsS http://127.0.0.1:18080/health/ready
```

Для production используйте non-dev mode (`kc.sh start`) и отдельный launchd unit с фиксированными env/paths.

## Troubleshooting

`/auth/login` или `/auth/callback` возвращают `5xx`:

- проверьте `KEYCLOAK_*` env в Roehub;
- проверьте redirect URI в Keycloak client;
- проверьте доступность `http://127.0.0.1:18080/realms/roehub/.well-known/openid-configuration`;
- проверьте `api.err.log`.

`/auth/current-user` всегда `401` после успешного login:

- проверьте cookie `roehub_session_id` в браузере;
- проверьте таблицу `identity_sessions`;
- проверьте TTL-параметры `IDENTITY_SESSION_*`.

## Связанные документы

- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
