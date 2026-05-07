# Identity Keycloak Auth Model v1

Документ фиксирует каноническую модель аутентификации Roehub после cutover на Keycloak: login через Keycloak + локальная server-side сессия Roehub.

## Статус

- active
- supersedes:
  - `docs/architecture/identity/identity-telegram-login-user-model-v1.md`
  - `docs/architecture/identity/identity-2fa-totp-policy-v1.md`

## Цель

- убрать Telegram/JWT/local-2FA как источники auth;
- сделать единый вход через Keycloak OIDC;
- оставить Roehub DB источником локальной identity/entitlement модели (`user_id`, `paid_level`);
- использовать только opaque cookie + server-side session lifecycle.

## Каноническая модель

### 1) Внешний IdP

- внешний identity provider: Keycloak;
- базовый внешний flow: OIDC Authorization Code;
- Web UI modal flow: same-origin credential form, где API обменивает username/password на Keycloak token через token endpoint и затем выпускает локальную Roehub session.
- внешний ключ пользователя: `keycloak_subject` (claim `sub`), хранится как opaque string.

### 2) Локальная identity Roehub

- `identity_users.user_id` — внутренний UUID Roehub;
- `identity_users.keycloak_subject` — внешний уникальный auth key;
- `identity_users.paid_level` — локальный entitlement источник (не Keycloak claim).

### 3) Browser auth модель

- браузер хранит только cookie `roehub_session_id` (имя настраивается через env);
- cookie значение — opaque UUID session id;
- raw Keycloak `access_token`/`id_token`/`refresh_token` в browser cookie не пишутся.
- username/password из Web UI login modal отправляются только в same-origin `POST /auth/password-login`, не пишутся в cookie и не сохраняются в Roehub DB.

### 4) Session lifecycle

- server-side сессии хранятся в `identity_sessions`;
- на `/auth/callback` или `/auth/password-login` создаётся/обновляется локальный user + создаётся session record;
- на `/auth/logout` session помечается revoked и cookie удаляется;
- `/auth/current-user` резолвит principal через session -> user snapshot.

## API surface

Identity endpoints:

- `GET /auth/login` — redirect в Keycloak authorize endpoint;
- `GET /auth/callback` — code exchange + introspection + upsert user + create local session;
- `POST /auth/password-login` — request DTO `{ "username": string, "password": string, "next": string | null }`; password grant через Keycloak token endpoint + introspection + upsert user + create local session; response DTO `{ "next": "/relative-path" }` + opaque session cookie; status `200/401/422/500`;
- `POST /auth/logout` — revoke local session + clear auth cookies;
- `GET /auth/current-user` — возвращает `{ "user_id": "...", "paid_level": "..." }`.

Protected routes в API используют только `RequireCurrentUserDependency` (cookie-only session resolution).

## Entitlements и 2FA

- entitlement source: Roehub DB (`identity_users.paid_level`);
- 2FA source: Keycloak OTP/realm policy;
- local `/2fa/*` endpoints и local 2FA gate в API отсутствуют.

## Инварианты

- `keycloak_subject` является внешним unique auth key;
- `user_id` никогда не подменяется внешним `sub`;
- без валидной Roehub session cookie protected endpoint возвращает `401`;
- сессия не активна, если `revoked_at` установлен или наступил `idle_expires_at`/`absolute_expires_at`;
- `paid_level` для `/auth/current-user` берётся из Roehub DB snapshot.

## Конфигурация runtime

Обязательные env-ключи для production (`IDENTITY_FAIL_FAST=true`):

- `KEYCLOAK_BASE_URL`
- `KEYCLOAK_REALM`
- `KEYCLOAK_CLIENT_ID`
- `KEYCLOAK_CLIENT_SECRET`
- `KEYCLOAK_REDIRECT_URI`
- `KEYCLOAK_LOGOUT_REDIRECT_URI`
- `IDENTITY_SESSION_COOKIE_NAME`
- `IDENTITY_SESSION_IDLE_TTL_SECONDS`
- `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`

Опциональные override endpoints:

- `KEYCLOAK_AUTH_URL`
- `KEYCLOAK_TOKEN_URL`
- `KEYCLOAK_END_SESSION_URL`
- `KEYCLOAK_INTROSPECTION_URL`

## Связанные файлы

- `apps/api/routes/identity.py`
- `apps/api/wiring/modules/identity.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/adapters/outbound/security/current_user/roehub_session_current_user.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py`
- `migrations/postgres/0005_identity_keycloak_cutover_v1.sql`

## Проверка

```bash
uv run pytest -q tests/unit/apps/api/test_identity_routes.py tests/unit/apps/api/test_identity_current_user_dependency.py
uv run pytest -q tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py
```
