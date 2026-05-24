# Keycloak Local Setup and Ops

Runbook для локального/стендового Keycloak, который используется как единственный auth source для Roehub API/Web.

## Цель

- поднять Keycloak как OIDC provider;
- подключить Keycloak к Postgres;
- настроить realm/client/users/OTP policy;
- проверить Roehub flow `/api/auth/login -> /api/auth/callback -> /api/auth/current-user -> /api/auth/logout` (public edge).

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
  - `https://roehub.com/api/auth/callback`
- `Valid post logout redirect URIs`:
  - `http://127.0.0.1:8010/login`
  - `https://roehub.com/login`
- `Web origins`: `https://roehub.com`, `https://www.roehub.com`, `http://127.0.0.1:8010` (или stricter policy по вашей схеме).

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

# browser-facing URLs (public edge)
KEYCLOAK_REDIRECT_URI=https://roehub.com/api/auth/callback
KEYCLOAK_LOGOUT_REDIRECT_URI=https://roehub.com/login
KEYCLOAK_AUTH_URL=https://roehub.com/realms/roehub/protocol/openid-connect/auth
KEYCLOAK_END_SESSION_URL=https://roehub.com/realms/roehub/protocol/openid-connect/logout

# backend-only URLs (API host -> local Keycloak)
KEYCLOAK_TOKEN_URL=http://127.0.0.1:18080/realms/roehub/protocol/openid-connect/token
KEYCLOAK_INTROSPECTION_URL=http://127.0.0.1:18080/realms/roehub/protocol/openid-connect/token/introspect

IDENTITY_SESSION_COOKIE_NAME=roehub_session_id
IDENTITY_SESSION_IDLE_TTL_SECONDS=1800
IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS=43200
```

## 6A) Настроить public routing для Keycloak UI

На `Mac Studio` должен быть поднят `tailscale serve` endpoint для Keycloak:

```bash
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --https=18443 --bg http://127.0.0.1:18080
/Applications/Tailscale.app/Contents/MacOS/Tailscale serve status
```

На публичном edge (`VPS Caddy`) должны быть роуты:

- `/realms/*` -> `https://macstudio-daniil.tail0ebbbc.ts.net:18443`
- `/resources/*` -> `https://macstudio-daniil.tail0ebbbc.ts.net:18443`

Source of truth: `infra/caddy/Caddyfile.vps`.

## 7) Перезапустить Roehub API/Web/Keycloak

```bash
bash scripts/macos/reload_launchd_services.sh prod

# если меняли env/plist Keycloak отдельно
launchctl kickstart -k gui/$(id -u)/com.roehub.keycloak
launchctl kickstart -k gui/$(id -u)/com.roehub.api
```

## 8) Smoke checks

```bash
# public login redirect
curl -i "https://roehub.com/api/auth/login?next=%2Fstrategies"

# anonymous current-user (public)
curl -i https://roehub.com/api/auth/current-user

# public realm discovery
curl -fsS https://roehub.com/realms/roehub/.well-known/openid-configuration | jq -r '.issuer,.authorization_endpoint'

# API auth surface metrics (local)
curl -fsS http://127.0.0.1:8000/metrics | rg 'http_requests_total\{.*path="/auth/(login|callback|logout|current-user)"'
```

Ожидания:

- `/api/auth/login` -> `307` на `https://roehub.com/realms/...`;
- `/api/auth/current-user` без cookie -> `401`;
- после browser login/callback появляется cookie `roehub_session_id`;
- `/api/auth/logout` ревокает сессию и удаляет cookie.

## 9) Operation commands (Keycloak)

```bash
# launchd state
launchctl list | grep com.roehub.keycloak
launchctl print gui/$(id -u)/com.roehub.keycloak | grep -E 'state =|pid =|last exit code ='

# restart keycloak/api pair
launchctl kickstart -k gui/$(id -u)/com.roehub.keycloak
launchctl kickstart -k gui/$(id -u)/com.roehub.api

# management readiness endpoint
curl -fsS http://127.0.0.1:19000/health/ready
```

## 9A) Agent E2E Auth Testing

Да, агент может выполнить e2e login flow через:

- `https://roehub.com/api/auth/login`
- `https://roehub.com/api/auth/callback`
- `https://roehub.com/api/auth/current-user`

Условия для non-interactive agent flow:

- выделенный test user;
- у пользователя нет `requiredActions`;
- для этого пользователя не требуется интерактивный OTP/setup шаг в Keycloak.

Если включен обязательный `CONFIGURE_TOTP`/`VERIFY_PROFILE` или другой required action, headless e2e через `curl` не завершится: нужен интерактивный браузерный шаг.

Пример agent smoke (public edge):

```bash
set -euo pipefail

export ROEHUB_E2E_USERNAME='<test-username>'
export ROEHUB_E2E_PASSWORD='<test-password>'

tmp_dir="$(mktemp -d)"
cookies="${tmp_dir}/cookies.txt"
login_headers="${tmp_dir}/login_headers.txt"
auth_page="${tmp_dir}/auth_page.html"
callback_body="${tmp_dir}/callback_body.txt"

curl -sS -c "${cookies}" -b "${cookies}" -D "${login_headers}" -o /dev/null \
  'https://roehub.com/api/auth/login?next=%2Fstrategies'

auth_url="$(awk 'BEGIN{IGNORECASE=1} /^location:/{print $2}' "${login_headers}" | tr -d '\r' | tail -n1)"
curl -sS -c "${cookies}" -b "${cookies}" -o "${auth_page}" "${auth_url}"

form_action="$(grep -o 'action=\"[^\"]*\"' "${auth_page}" | head -n1 | sed 's/action=\"//; s/\"$//')"
form_action="${form_action//&amp;/&}"

final_url="$(
  curl -sS -L -c "${cookies}" -b "${cookies}" -o "${callback_body}" \
    --data-urlencode "username=${ROEHUB_E2E_USERNAME}" \
    --data-urlencode "password=${ROEHUB_E2E_PASSWORD}" \
    --data-urlencode 'credentialId=' \
    "${form_action}" \
    -w '%{url_effective}'
)"

current_user_status="$(
  curl -sS -c "${cookies}" -b "${cookies}" -o "${tmp_dir}/current_user.json" \
    -w '%{http_code}' 'https://roehub.com/api/auth/current-user'
)"

logout_status="$(
  curl -sS -c "${cookies}" -b "${cookies}" -o /dev/null -w '%{http_code}' \
    -X POST 'https://roehub.com/api/auth/logout'
)"

after_logout_status="$(
  curl -sS -c "${cookies}" -b "${cookies}" -o "${tmp_dir}/after_logout.json" \
    -w '%{http_code}' 'https://roehub.com/api/auth/current-user'
)"

echo "final_url=${final_url}"
echo "current_user_status=${current_user_status}"
cat "${tmp_dir}/current_user.json"
echo
echo "logout_status=${logout_status}"
echo "after_logout_status=${after_logout_status}"
```

Ожидания для успешного agent e2e:

- `final_url=https://roehub.com/strategies` (или другой `next` path);
- `current_user_status=200`;
- `logout_status=204`;
- `after_logout_status=401`.

## 9B) Playwright acceptance for `/settings` exchange connections

Используйте этот чек только со smoke Keycloak account из безопасного operator
channel. Нельзя сохранять username/password, cookies, API keys, API secrets,
tokens, ciphertext или raw provider responses в repo/docs/logs.

Acceptance path:

- открыть `https://roehub.com/settings`;
- пройти login/callback через Keycloak;
- проверить, что browser-visible calls к `/api/ui/account/profile`,
  `/api/ui/account/integrations`, `/api/ui/account/limits` и
  `/api/ui/account/exchange-connections` не возвращают production 500;
- убедиться, что default permission на fresh add form равен `read`;
- отправить только dummy Binance/Bybit API key/secret на
  `/api/ui/account/exchange-connections`;
- подтвердить, что response не содержит `Mutation origin is not allowed` и
  `csrf_origin_mismatch`;
- проверить, что secret inputs очищены после success/failure;
- disable/delete dummy connection через поддержанный UI/API path;
- сохранить sanitized screenshot/network summary under `output/playwright/`;
- выполнить secret artifact grep перед финальным отчетом.

Минимальные локальные prerequisites:

```bash
command -v npx >/dev/null 2>&1
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"
test -f "$PWCLI"
npx playwright --version
```

## 10) Monit supervision (рекомендуется)

Keycloak имеет смысл добавить в Monit: это auth entrypoint для web login flow, и при падении `com.roehub.keycloak` вход в систему становится недоступен.

Конфиг в репозитории:

- `infra/scripts/monit/roehub-keycloak.monitrc`

Установка/применение:

```bash
install -m 0600 infra/scripts/monit/roehub-keycloak.monitrc /opt/homebrew/etc/monit.d/roehub-keycloak.monitrc
/opt/homebrew/opt/monit/bin/monit -t -c /opt/homebrew/etc/monitrc
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc reload
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | grep roehub_keycloak
```

Проверка health policy:

- process check: `matching "kc\\.sh start"`
- endpoint check: `127.0.0.1:19000/health/ready`
- action: restart через `launchctl_service_control.sh`.

## Troubleshooting

`/api/auth/login` или `/api/auth/callback` возвращают `5xx`:

- проверьте `KEYCLOAK_*` env в Roehub;
- проверьте redirect URI в Keycloak client;
- проверьте доступность `http://127.0.0.1:18080/realms/roehub/.well-known/openid-configuration`;
- проверьте `api.err.log`.

`/api/auth/login` редиректит на localhost/непубличный адрес:

- проверьте `KEYCLOAK_AUTH_URL`, `KEYCLOAK_REDIRECT_URI`, `KEYCLOAK_LOGOUT_REDIRECT_URI`;
- проверьте публичный маршрут `https://roehub.com/realms/roehub/.well-known/openid-configuration`;
- проверьте `tailscale serve status` на `Mac Studio` (должен быть `:18443 -> 127.0.0.1:18080`);
- проверьте `infra/caddy/Caddyfile.vps` и deploy web edge.

`/api/auth/current-user` всегда `401` после успешного login:

- проверьте cookie `roehub_session_id` в браузере;
- проверьте таблицу `identity_sessions`;
- проверьте TTL-параметры `IDENTITY_SESSION_*`.

## Связанные документы

- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
