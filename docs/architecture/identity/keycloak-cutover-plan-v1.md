# Keycloak Cutover Plan v1 (Legacy Auth Removal)

> Статус: `superseded` как будущий план с 2026-07-13.
> Его заменяет
> `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md`:
> чистая установка использует local auth, внешний OIDC необязателен, а
> существующая Keycloak-система не является migration source и не переносится.
> Stages `06`–`07` приняты в новой greenfield-модели; канонические контракты —
> `local-auth-sessions-recovery-v1.md` и
> `oidc-authentication-provider-v1.md`. Оставшийся текст ниже является только
> историческим описанием прежнего Keycloak-only направления.
> Real-boundary validation этого superseded-плана больше не выполняется; её
> заменяют browser/API/migration evidence, заданные новыми Stages `06`–`07`.

## 1) Scope And Fixed Decisions

Этот план фиксирует полный отказ от legacy-auth модели и переход на Keycloak-only authentication/authorization.

Принятые решения:

- Удаляем `POST /auth/telegram/login` и весь `auth_telegram` flow.
- Удаляем `Hs256JwtCodec` и `IDENTITY_JWT_SECRET` как источник auth.
- `TELEGRAM_BOT_TOKEN` больше не используется в API-auth, остаётся только для strategy notifier.
- Telegram больше не является primary identity key (`telegram_user_id` не используется как ключ авторизации).
- Web login через Telegram widget/script удаляется.
- Маппинг legacy users не делаем (только 2-3 тестовых пользователя).
- Источник 2FA: Keycloak OTP policy. Локальные `/2fa/*` и local 2FA-gate убираются.
- Telegram OIDC broker на этом этапе не делаем.
- Браузерная auth-cookie хранит только opaque session id (`roehub_session_id`); raw Keycloak access/id/refresh tokens в cookie не пишем.
- Session model для API: server-side persisted session store. В `prod` in-memory session storage не допускается.
- Канонический provider-facing validation path в cutover v1: OIDC Authorization Code Flow с confidential client на backend. Per-request JWKS verification не используем; introspection допускается только как backend revalidation/hardening, а не как browser-token auth path.
- `identity_users.user_id` остаётся внутренним Roehub UUID. Keycloak `sub` трактуется как opaque external subject и хранится отдельно в `identity_users.keycloak_subject`.
- Source of truth для `paid_level` — Roehub DB (`identity_users.paid_level`). Keycloak claims не являются canonical entitlement source.
- Успешный OIDC login всегда приводит к `find-or-create/upsert` локального пользователя и обновлению `last_login_at`.
- Browser/web protected path — cookie-only session model; Bearer fallback не входит в default cutover path.
- Logout обязательно инвалидирует локальную сессию Roehub; IdP end-session redirect остаётся опциональным hardening-слоем, а не базовым contract requirement.

Ранее открытое допущение закрыто:

- `sub` больше не предполагается UUID-совместимым и не маппится напрямую в `UserId`.

## 2) Target Architecture (After Cutover)

- Единственный auth provider: Keycloak (OIDC Authorization Code Flow).
- API хранит server-side session record в Roehub identity storage; в браузер выдаётся только opaque session id в HttpOnly cookie.
- Keycloak tokens обрабатываются только на backend во время callback/revalidation и никогда не сериализуются в browser cookie.
- `CurrentUserPrincipal` остаётся единым контрактом для strategy/backtest/market-data endpoints.
- `/auth/current-user` сохраняется как совместимый endpoint для web login gate.
- `/2fa/setup` и `/2fa/verify` удаляются.
- 2FA enforcement выполняется только в Keycloak realm/client policy.
- `user_id` в Roehub API остаётся внутренним UUID из Roehub DB; `keycloak_subject` — внешний unique auth key.
- `paid_level` в principal и `/auth/current-user` читается из Roehub DB, а не из Keycloak claims.

## 2A) Decision Locks Before Remaining Steps

- Любая реализация, где browser cookie содержит raw Keycloak access token, считается промежуточной и MUST быть заменена до начала Step 6A.
- Любая реализация, где `CurrentUser` напрямую доверяет browser-provided bearer/token path, не соответствует целевой browser/web модели cutover v1.
- Любая реализация, где `sub == UserId`, считается неверной для final target architecture.
- Любая реализация, где `paid_level` читается из Keycloak claim как canonical entitlement source, считается неверной для final target architecture.
- До завершения cutover в `prod` MUST существовать persisted session storage с bounded idle/absolute TTL.
- Step 6+ исполняются только в модели: `browser -> opaque session cookie -> local session lookup -> Roehub user snapshot`.

## 3) Step-By-Step Migration Plan

## Step 0. Freeze Legacy Auth Surface

Что делаем:

- Фиксируем изменение как hard cutover без dual-mode (`legacy|keycloak`).
- Определяем DoD: ни одного runtime call path на Telegram login/JWT codec/local 2FA.

Затрагиваемые файлы:

- `docs/architecture/identity/keycloak-cutover-plan-v1.md` (этот документ)
- `docs/architecture/operations/native-service-control-monitoring-admin-target-v1.md` (обновить статус/ссылку на plan)

Критерий готовности:

- Зафиксирован единый migration path и целевая модель Keycloak-only.

## Step 1. Keycloak Runtime Settings And Fail-Fast In API

Что делаем:

- В wiring identity удаляем env-политику legacy auth (`IDENTITY_JWT_SECRET`, `JWT_TTL_DAYS`, cookie-name legacy jwt).
- Добавляем Keycloak env-настройки и fail-fast в `prod`.
- Добавляем runtime policy для server-side session cookie и session TTL.
- Сохраняем `IDENTITY_PG_DSN` и `IDENTITY_EXCHANGE_KEYS_KEK_B64` для identity storage/exchange keys.

Новые/целевые env keys:

- `KEYCLOAK_BASE_URL`
- `KEYCLOAK_REALM`
- `KEYCLOAK_CLIENT_ID`
- `KEYCLOAK_CLIENT_SECRET`
- `KEYCLOAK_REDIRECT_URI`
- `KEYCLOAK_LOGOUT_REDIRECT_URI`
- `KEYCLOAK_AUTH_URL`
- `KEYCLOAK_TOKEN_URL`
- `KEYCLOAK_END_SESSION_URL` (derive из realm или явная настройка в `prod`)
- `KEYCLOAK_INTROSPECTION_URL` (optional hardening для backend revalidation; не mandatory для browser auth path)
- `IDENTITY_SESSION_COOKIE_NAME` (default target: `roehub_session_id`)
- `IDENTITY_SESSION_IDLE_TTL_SECONDS`
- `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`

Затрагиваемые файлы:

- `apps/api/wiring/modules/identity.py`
- `tests/unit/apps/api/test_identity_wiring_module.py`
- `infra/docker/.env.example`
- `infra/docker/docker-compose.yml`
- `infra/docker/docker-compose.backend.yml`
- `infra/macos/launchd/com.roehub.api.plist`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`

Критерий готовности:

- API стартует только при валидной Keycloak code-flow и session-конфигурации.
- `IDENTITY_JWT_SECRET`, legacy jwt-cookie settings и mandatory `KEYCLOAK_JWKS_URL` не участвуют в auth.

## Step 2. Replace CurrentUser Resolver: Legacy JWT Cookie -> Roehub Server-Side Session

Что делаем:

- Реализацию `CurrentUser` переводим на session-backed resolver.
- `RequireCurrentUserDependency` продолжает выдавать `CurrentUserPrincipal`, но источник principal — локальная Roehub session + Roehub user snapshot.
- Удаляем runtime path через `JwtCookieCurrentUser`.
- Удаляем default browser/runtime path, где principal строится из raw Keycloak token, переданного браузером.
- Убираем Bearer fallback из default browser/web cutover path.

Затрагиваемые файлы:

- `src/trading/contexts/identity/application/ports/current_user.py`
- `src/trading/contexts/identity/application/ports/session_repository.py` (новый)
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/adapters/outbound/security/current_user/jwt_cookie_current_user.py` (удаление/замена)
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py` (новый)
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/session_repository.py` (new for tests/dev only)
- `src/trading/contexts/identity/adapters/outbound/security/current_user/__init__.py`
- `src/trading/contexts/identity/adapters/outbound/security/__init__.py`
- `src/trading/contexts/identity/adapters/outbound/__init__.py`
- `src/trading/contexts/identity/adapters/__init__.py`
- `src/trading/contexts/identity/application/ports/__init__.py`
- `src/trading/contexts/identity/application/__init__.py`
- `src/trading/contexts/identity/__init__.py`

Критерий готовности:

- Все protected API роуты работают через session-backed current-user dependency.
- Browser auth cookie содержит только opaque session id.
- Реализация вида `CurrentUser.require(token=browser_cookie_access_token)` отсутствует в финальном runtime path.

## Step 3. Replace Auth API Endpoints (Telegram -> OIDC)

Что делаем:

- Удаляем `auth_telegram` router.
- Добавляем OIDC endpoints:
  - `GET /auth/login`
  - `GET /auth/callback`
  - `POST /auth/logout`
  - `GET /auth/current-user` (совместимый контракт: `user_id`, `paid_level`)
- `GET /auth/callback` после code exchange создаёт локального пользователя/сессию и выставляет opaque session cookie.
- `POST /auth/logout` инвалидирует локальную Roehub session и очищает opaque session cookie.

Затрагиваемые файлы:

- `apps/api/routes/identity.py`
- `apps/api/wiring/modules/identity.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py` (удаление/замена)
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/__init__.py`
- `src/trading/contexts/identity/adapters/inbound/api/__init__.py`
- `src/trading/contexts/identity/adapters/inbound/__init__.py`
- `apps/api/main/app.py` (убрать 2FA exception handler registration)

Критерий готовности:

- В API больше нет endpoint `POST /auth/telegram/login`.
- `/auth/current-user` продолжает быть точкой истины для web login gate.
- В browser cookie нет raw Keycloak token.
- Logout снимает локальную Roehub session, а не только удаляет legacy cookie key.

## Step 4. Remove Legacy Telegram/JWT Application Layer

Что делаем:

- Удаляем use-case/ports/value objects, относящиеся к Telegram login и локальному JWT.
- Удаляем `TelegramUserId` как auth key.
- Нормализуем user repository contract под `keycloak_subject` как внешний auth key, а не под `user_id=sub`.

Затрагиваемые файлы:

- `src/trading/contexts/identity/application/use_cases/telegram_login.py`
- `src/trading/contexts/identity/application/use_cases/__init__.py`
- `src/trading/contexts/identity/application/ports/telegram_auth_payload_validator.py`
- `src/trading/contexts/identity/application/ports/jwt_codec.py`
- `src/trading/contexts/identity/application/ports/user_repository.py` (заменить legacy methods на `find_by_keycloak_subject` + `upsert_keycloak_login`/эквивалент)
- `src/trading/contexts/identity/domain/entities/user.py` (убрать зависимость от `telegram_user_id`, сохранить `user_id` как внутренний Roehub UUID)
- `src/trading/contexts/identity/domain/value_objects/telegram_user_id.py` (удаление)
- `src/trading/contexts/identity/domain/value_objects/__init__.py`
- `src/trading/contexts/identity/adapters/outbound/security/jwt/hs256_jwt_codec.py` (удаление)
- `src/trading/contexts/identity/adapters/outbound/security/jwt/__init__.py`
- `src/trading/contexts/identity/adapters/outbound/security/telegram/telegram_login_widget_payload_validator.py` (удаление)
- `src/trading/contexts/identity/adapters/outbound/security/telegram/__init__.py`

Критерий готовности:

- В `src/trading/contexts/identity` нет исполняемого кода, связанного с Telegram Login Widget или локальным HS256 JWT.
- Локальный user model не предполагает `user_id=sub` и не использует `telegram_user_id` как auth key.

## Step 5. Remove Local 2FA (TOTP) And 2FA Gate

Что делаем:

- Удаляем local TOTP setup/verify endpoints и local 2FA policy gate.
- В exchange keys routes оставляем auth dependency без local 2FA dependency.
- 2FA control полностью выносится в Keycloak policy.

Затрагиваемые файлы:

- `src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py` (удаление)
- `src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py` (удаление)
- `src/trading/contexts/identity/adapters/inbound/api/deps/__init__.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py`
- `apps/api/routes/identity.py`
- `apps/api/main/app.py`
- `apps/api/wiring/modules/identity.py`
- `src/trading/contexts/identity/application/ports/two_factor_*` (удаление)
- `src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py` (удаление)
- `src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py` (удаление)
- `src/trading/contexts/identity/application/use_cases/two_factor_errors.py` (удаление)
- `src/trading/contexts/identity/domain/entities/two_factor_auth.py` (удаление)
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/two_factor_repository.py` (удаление)
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/two_factor_repository.py` (удаление)
- `src/trading/contexts/identity/adapters/outbound/security/two_factor/*` (удаление)
- `src/trading/contexts/identity/adapters/outbound/policy/two_factor_policy_gate.py` (удаление)

Критерий готовности:

- В API отсутствуют `/2fa/setup` и `/2fa/verify`.
- В коде отсутствует `RequireTwoFactorEnabledDependency`.

## Step 5A. Lock Secure Session/User Decisions Before DB Cutover

Что делаем:

- Явно фиксируем final target architecture для remaining work.
- Фиксируем, что browser auth path = opaque session cookie only.
- Фиксируем, что Roehub `user_id` не зависит от формата Keycloak `sub`.
- Фиксируем, что `paid_level` читается из Roehub DB.
- Фиксируем, что реализация с raw Keycloak token в browser cookie является промежуточной и подлежит замене до runtime completion.

Затрагиваемые файлы:

- `docs/architecture/identity/keycloak-cutover-plan-v1.md` (этот документ)
- `docs/architecture/identity/identity-keycloak-auth-model-v1.md` (новый canonical doc в Step 10)

Критерий готовности:

- После Step 5A у remaining implementation нет открытых решений по session model, subject mapping, entitlement source и logout semantics.

## Step 6. Identity DB Migration For Keycloak-Native User Model

Что делаем:

- Делаем схему `identity_users` независимой от `telegram_user_id`.
- Добавляем `keycloak_subject` как внешний unique auth key.
- Добавляем persisted session storage для opaque Roehub sessions.
- Удаляем зависимость auth-потока от `identity_2fa`.
- Готовим схему к Keycloak-native login (по `keycloak_subject` + локальному `user_id` + `last_login_at`).

Предпочтительный вариант миграции:

- Новый SQL migration (например `0005_identity_keycloak_cutover_v1.sql`):
  - `ALTER TABLE identity_users ADD COLUMN keycloak_subject TEXT NULL;`
  - create partial unique index on `identity_users(keycloak_subject)` where `keycloak_subject IS NOT NULL`.
  - `ALTER TABLE identity_users ALTER COLUMN telegram_user_id DROP NOT NULL;`
  - drop unique index `idx_identity_users_telegram_user_id` как legacy-auth artifact.
  - создать `identity_sessions` table для server-side session records:
    - `session_id UUID PRIMARY KEY`
    - `user_id UUID NOT NULL REFERENCES identity_users(user_id) ON DELETE CASCADE`
    - `created_at TIMESTAMPTZ NOT NULL`
    - `last_seen_at TIMESTAMPTZ NOT NULL`
    - `idle_expires_at TIMESTAMPTZ NOT NULL`
    - `absolute_expires_at TIMESTAMPTZ NOT NULL`
    - `revoked_at TIMESTAMPTZ NULL`
  - опционально drop `identity_2fa` table (если полностью удаляем local 2FA).
  - отдельным post-cutover migration после ручной привязки 2-3 retained test users поднять `keycloak_subject` до `NOT NULL`, если это останется актуальным.

Затрагиваемые файлы:

- `migrations/postgres/0001_identity_v1.sql` (актуализация baseline комментариев/shape)
- `migrations/postgres/0002_identity_2fa_totp_v1.sql` (deprecate note)
- `migrations/postgres/0005_identity_keycloak_cutover_v1.sql` (новый)
- `apps/migrations/bootstrap.py`
- `apps/migrations/bootstrap_main.py`
- `apps/migrations/__init__.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/user_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py` (новый)
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/session_repository.py` (new for tests/dev only)
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py`

Критерий готовности:

- DB bootstrap не ожидает legacy-auth инварианты.
- User upsert/find работает по `keycloak_subject` и локальному `user_id`.
- Persisted Roehub session lifecycle (`create/read/revoke`) поддерживается на уровне schema + repository contract.

## Step 6A. Reconcile Runtime Auth Flow With Persisted Session Storage

Что делаем:

- После Step 6 переводим callback/current-user/logout на реальную persisted session model.
- `GET /auth/callback`:
  - завершает code exchange,
  - извлекает/нормализует `keycloak_subject`,
  - делает `upsert_keycloak_login`/эквивалент,
  - создаёт Roehub session record,
  - выставляет opaque session cookie.
- `GET /auth/current-user` читает principal из session store + Roehub user repository.
- `POST /auth/logout` ревокает local session record и очищает opaque session cookie.
- Любой provisional path с raw provider token в browser cookie удаляется.

Затрагиваемые файлы:

- `apps/api/routes/identity.py`
- `apps/api/wiring/modules/identity.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/application/ports/current_user.py`
- `src/trading/contexts/identity/application/ports/session_repository.py` (новый)
- `src/trading/contexts/identity/application/ports/user_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py` (новый)
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/session_repository.py` (new for tests/dev only)
- `tests/unit/apps/api/test_identity_routes.py`
- `tests/unit/apps/api/test_identity_current_user_dependency.py`
- `tests/unit/apps/api/test_identity_wiring_module.py`

Критерий готовности:

- Browser получает только opaque session id cookie.
- `CurrentUserPrincipal` строится из local session + Roehub user snapshot.
- В runtime path отсутствует зависимость от browser-supplied access token/bearer path.

## Step 7. Web Auth UX Switch (Telegram Widget -> Keycloak Redirect)

Что делаем:

- Переводим `/login` страницу на кнопку/redirect в `/api/auth/login`.
- Удаляем Telegram widget JS/script callback.
- Сохраняем server-side login gate через `/api/auth/current-user`.
- `/logout` страница завершает local Roehub session и при необходимости обрабатывает post-logout redirect.

Затрагиваемые файлы:

- `apps/web/templates/login.html`
- `apps/web/templates/logout.html`
- `apps/web/main/app.py`
- `apps/web/main/api_client.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_api_client.py`

Критерий готовности:

- В web templates нет `telegram-widget.js`.
- `/login` корректно отправляет пользователя в Keycloak login flow.
- `/logout` не зависит от legacy cookie-key и завершает именно opaque Roehub session.

## Step 8. Keep Telegram Only In Strategy Notifier Scope

Что делаем:

- Проверяем, что `TELEGRAM_BOT_TOKEN` используется только в strategy notifier/runtime config.
- Удаляем любые ссылки на его auth-роль в API docs/runbooks.

Затрагиваемые файлы:

- `apps/api/wiring/modules/identity.py` (убрать зависимость от `TELEGRAM_BOT_TOKEN`)
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/architecture/strategy/strategy-runtime-config-v1.md` (уточнить, что token не связан с user-auth)
- `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md` (уточнить раздел identity dependency)

Не удаляем из strategy scope:

- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
- `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py`
- `src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py`
- `configs/dev/strategy.yaml`
- `configs/test/strategy.yaml`
- `configs/prod/strategy.yaml`

Критерий готовности:

- `TELEGRAM_BOT_TOKEN` отсутствует в API-auth path и присутствует только в strategy notifier path.

## Step 9. End-To-End Test Migration

Что делаем:

- Удаляем/переписываем тесты Telegram login/JWT/local 2FA.
- Добавляем тесты OIDC login/callback/current-user/logout.
- Добавляем тесты session repository, session TTL/revoke semantics и отсутствие raw-token cookie path.
- Сохраняем тесты всех protected routes на контракт `CurrentUserPrincipal`.

Затрагиваемые файлы:

- `tests/unit/apps/api/test_identity_routes.py`
- `tests/unit/apps/api/test_identity_current_user_dependency.py`
- `tests/unit/apps/api/test_identity_wiring_module.py`
- `tests/unit/apps/api/test_identity_two_factor_routes.py` (удаление/замена)
- `tests/unit/apps/api/test_identity_two_factor_gate_dependency.py` (удаление/замена)
- `tests/unit/apps/api/test_identity_exchange_keys_routes.py`
- `tests/unit/contexts/identity/application/test_telegram_login_use_case.py` (удаление/замена)
- `tests/unit/contexts/identity/adapters/outbound/security/test_telegram_login_widget_payload_validator.py` (удаление)
- `tests/unit/contexts/identity/application/test_two_factor_totp_use_cases.py` (удаление/замена)
- `tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_timezone_normalization.py`
- `tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py` (новый)
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_api_client.py`

Критерий готовности:

- Нет тестов на legacy endpoints/classes.
- Есть полный набор тестов для Keycloak flow, Roehub session lifecycle и текущего API protected surface.
- Нет тестов, где browser auth моделируется raw Keycloak token cookie.

## Step 10. Documentation Full Sweep

Что делаем:

- Полностью убираем legacy-auth описание из архитектурных и runbook документов.
- Добавляем новый canonical doc по Keycloak auth model.

Новые документы:

- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- `docs/runbooks/keycloak-local-setup-and-ops.md`

Обязательные документы к обновлению:

- `docs/architecture/README.md`
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md`
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` (архивировать/заменить ссылкой)
- `docs/architecture/identity/identity-2fa-totp-policy-v1.md` (архивировать, заменить на Keycloak OTP)
- `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md`
- `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md`
- `docs/architecture/roadmap/base_milestone_plan.md`
- `docs/architecture/roadmap/milestone-3-epics-v1.md`
- `docs/architecture/roadmap/milestone-6-epics-v1.md`
- `docs/architecture/shared-kernel-primitives.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/repository_three.md` (перегенерировать индекс/ссылки после удаления legacy файлов)

Критерий готовности:

- В docs нет инструкций использовать Telegram login/JWT cookie/local 2FA как auth source.
- В docs явно зафиксирована server-side session model, `keycloak_subject` как внешний auth key и Roehub DB как entitlement source.

## Step 11. Final Cleanup And Dead Code Removal

Что делаем:

- Удаляем неиспользуемые импорты/экспорты и stale references в `__init__`.
- Удаляем удалённые файлы из tree и корректируем индексы docs.

Затрагиваемые файлы:

- `src/trading/contexts/identity/__init__.py`
- `src/trading/contexts/identity/adapters/__init__.py`
- `src/trading/contexts/identity/adapters/inbound/__init__.py`
- `src/trading/contexts/identity/adapters/inbound/api/__init__.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/__init__.py`
- `src/trading/contexts/identity/adapters/outbound/__init__.py`
- `src/trading/contexts/identity/application/__init__.py`
- `src/trading/contexts/identity/application/ports/__init__.py`
- `src/trading/contexts/identity/application/use_cases/__init__.py`

Критерий готовности:

- `rg "telegram login|auth_telegram|Hs256JwtCodec|JwtCookieCurrentUser|/2fa/setup|/2fa/verify|IDENTITY_JWT_SECRET"` не находит runtime references вне архивных документов.

## Step 12. Production Validation Checklist

Проверки после cutover:

- API startup в `prod` не требует `IDENTITY_JWT_SECRET`/`TELEGRAM_BOT_TOKEN` для auth.
- `/auth/login -> Keycloak -> /auth/callback -> /auth/current-user` работает end-to-end.
- Browser auth cookie содержит только opaque session id; raw Keycloak token в browser cookie отсутствует.
- `strategy`, `backtest`, `market-data` protected endpoints работают с новым principal.
- `exchange-keys` операции работают без local 2FA dependency.
- Web protected pages корректно редиректят в новый login flow.
- Roehub session create/read/revoke работает через persisted storage и покрыт smoke-проверкой.
- `paid_level` в `/auth/current-user` и principal совпадает с Roehub DB snapshot, а не с Keycloak claims.
- Monit/launchd/docker runbooks обновлены и проверены smoke-командами.

---

## Appendix A. Legacy Auth Inventory (65 files)

```text
apps/api/main/app.py
apps/api/routes/identity.py
apps/api/wiring/modules/identity.py
apps/web/main/api_client.py
apps/web/main/app.py
apps/web/templates/login.html
docs/architecture/README.md
docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
docs/architecture/identity/identity-2fa-totp-policy-v1.md
docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
docs/architecture/identity/identity-telegram-login-user-model-v1.md
docs/architecture/roadmap/base_milestone_plan.md
docs/architecture/roadmap/milestone-3-epics-v1.md
docs/architecture/roadmap/milestone-6-epics-v1.md
docs/architecture/shared-kernel-primitives.md
docs/architecture/strategy/strategy-runtime-config-v1.md
docs/repository_three.md
docs/runbooks/mac-studio-monitoring-plan.md
docs/runbooks/mac-studio-native-backend-operations.md
docs/runbooks/web-ui-gateway-same-origin.md
migrations/postgres/0001_identity_v1.sql
src/trading/contexts/identity/__init__.py
src/trading/contexts/identity/adapters/__init__.py
src/trading/contexts/identity/adapters/inbound/__init__.py
src/trading/contexts/identity/adapters/inbound/api/__init__.py
src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
src/trading/contexts/identity/adapters/inbound/api/routes/__init__.py
src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py
src/trading/contexts/identity/adapters/outbound/__init__.py
src/trading/contexts/identity/adapters/outbound/persistence/in_memory/user_repository.py
src/trading/contexts/identity/adapters/outbound/persistence/postgres/gateway.py
src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py
src/trading/contexts/identity/adapters/outbound/security/__init__.py
src/trading/contexts/identity/adapters/outbound/security/current_user/__init__.py
src/trading/contexts/identity/adapters/outbound/security/current_user/jwt_cookie_current_user.py
src/trading/contexts/identity/adapters/outbound/security/jwt/__init__.py
src/trading/contexts/identity/adapters/outbound/security/jwt/hs256_jwt_codec.py
src/trading/contexts/identity/adapters/outbound/security/telegram/__init__.py
src/trading/contexts/identity/adapters/outbound/security/telegram/telegram_login_widget_payload_validator.py
src/trading/contexts/identity/adapters/outbound/time/system_identity_clock.py
src/trading/contexts/identity/application/__init__.py
src/trading/contexts/identity/application/ports/__init__.py
src/trading/contexts/identity/application/ports/clock.py
src/trading/contexts/identity/application/ports/current_user.py
src/trading/contexts/identity/application/ports/jwt_codec.py
src/trading/contexts/identity/application/ports/telegram_auth_payload_validator.py
src/trading/contexts/identity/application/ports/user_repository.py
src/trading/contexts/identity/application/use_cases/__init__.py
src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
src/trading/contexts/identity/application/use_cases/telegram_login.py
src/trading/contexts/identity/domain/entities/user.py
src/trading/contexts/identity/domain/value_objects/__init__.py
src/trading/contexts/identity/domain/value_objects/telegram_chat_id.py
src/trading/contexts/identity/domain/value_objects/telegram_user_id.py
src/trading/shared_kernel/primitives/paid_level.py
src/trading/shared_kernel/primitives/user_id.py
tests/unit/apps/api/test_identity_exchange_keys_routes.py
tests/unit/apps/api/test_identity_routes.py
tests/unit/apps/api/test_identity_two_factor_gate_dependency.py
tests/unit/apps/api/test_identity_two_factor_routes.py
tests/unit/apps/api/test_identity_wiring_module.py
tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_timezone_normalization.py
tests/unit/contexts/identity/adapters/outbound/security/test_telegram_login_widget_payload_validator.py
tests/unit/contexts/identity/application/test_telegram_login_use_case.py
```

## Appendix B. Local 2FA Inventory (51 files)

```text
apps/api/main/app.py
apps/api/routes/identity.py
apps/api/wiring/modules/identity.py
apps/migrations/bootstrap.py
docs/architecture/identity/identity-2fa-totp-policy-v1.md
docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
docs/architecture/roadmap/milestone-3-epics-v1.md
docs/repository_three.md
docs/runbooks/web-ui-gateway-same-origin.md
migrations/postgres/0002_identity_2fa_totp_v1.sql
src/trading/contexts/identity/__init__.py
src/trading/contexts/identity/adapters/__init__.py
src/trading/contexts/identity/adapters/inbound/__init__.py
src/trading/contexts/identity/adapters/inbound/api/__init__.py
src/trading/contexts/identity/adapters/inbound/api/deps/__init__.py
src/trading/contexts/identity/adapters/inbound/api/deps/two_factor_enabled.py
src/trading/contexts/identity/adapters/inbound/api/routes/__init__.py
src/trading/contexts/identity/adapters/inbound/api/routes/exchange_keys.py
src/trading/contexts/identity/adapters/inbound/api/routes/two_factor_totp.py
src/trading/contexts/identity/adapters/outbound/__init__.py
src/trading/contexts/identity/adapters/outbound/persistence/in_memory/__init__.py
src/trading/contexts/identity/adapters/outbound/persistence/in_memory/two_factor_repository.py
src/trading/contexts/identity/adapters/outbound/persistence/postgres/__init__.py
src/trading/contexts/identity/adapters/outbound/persistence/postgres/two_factor_repository.py
src/trading/contexts/identity/adapters/outbound/policy/__init__.py
src/trading/contexts/identity/adapters/outbound/policy/two_factor_policy_gate.py
src/trading/contexts/identity/adapters/outbound/security/__init__.py
src/trading/contexts/identity/adapters/outbound/security/two_factor/__init__.py
src/trading/contexts/identity/adapters/outbound/security/two_factor/aes_gcm_envelope_secret_cipher.py
src/trading/contexts/identity/adapters/outbound/security/two_factor/pyotp_totp_provider.py
src/trading/contexts/identity/application/__init__.py
src/trading/contexts/identity/application/ports/__init__.py
src/trading/contexts/identity/application/ports/two_factor_policy_gate.py
src/trading/contexts/identity/application/ports/two_factor_repository.py
src/trading/contexts/identity/application/ports/two_factor_secret_cipher.py
src/trading/contexts/identity/application/ports/two_factor_totp_provider.py
src/trading/contexts/identity/application/use_cases/__init__.py
src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
src/trading/contexts/identity/application/use_cases/two_factor_errors.py
src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
src/trading/contexts/identity/domain/entities/__init__.py
src/trading/contexts/identity/domain/entities/two_factor_auth.py
tests/unit/apps/api/test_identity_exchange_keys_routes.py
tests/unit/apps/api/test_identity_routes.py
tests/unit/apps/api/test_identity_two_factor_gate_dependency.py
tests/unit/apps/api/test_identity_two_factor_routes.py
tests/unit/apps/api/test_identity_wiring_module.py
tests/unit/apps/migrations/test_bootstrap_apply_flow.py
tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_timezone_normalization.py
tests/unit/contexts/identity/application/test_two_factor_totp_use_cases.py
```
