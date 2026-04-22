# Keycloak Cutover Restart Prompt-Pack v1

Статус: proposed executable restart-pack after target-architecture hardening  
Дата: 2026-04-22

Связанные документы:
- `docs/architecture/identity/keycloak-cutover-plan-v1.md`
- `apps/api/wiring/modules/identity.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/adapters/outbound/security/current_user/keycloak_introspection_current_user.py`
- `src/trading/contexts/identity/application/ports/user_repository.py`
- `tests/unit/apps/api/test_identity_wiring_module.py`
- `tests/unit/apps/api/test_identity_current_user_dependency.py`
- `tests/unit/apps/api/test_identity_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_api_client.py`

## 1. Цель restart-pack

Этот документ задаёт **правильную последовательность перезапуска** уже начатого Keycloak cutover.

Он нужен потому, что original steps `1-5` были выполнены только частично и в коде уже есть промежуточные решения, которые **не соответствуют final target architecture** из `keycloak-cutover-plan-v1.md`.

Restart-pack фиксирует:
- какие шаги считаются reopened;
- в каком порядке их исполнять;
- что именно сохранить из текущего дерева;
- что обязательно удалить/заменить;
- какие проверки запускать после каждого шага;
- какой prompt давать агенту на каждый шаг.

## 2. Текущее исходное состояние

Считать фактом:
- `Step 1-5` были начаты и частично реализованы.
- Эти шаги **нельзя** повторно исполнять как greenfield.
- Их нужно выполнять как **delta-reconciliation against current tree**.

Подтверждённые provisional conflicts, которые НЕ являются целевой архитектурой:
- `apps/api/wiring/modules/identity.py` всё ещё тянет legacy/provisional cookie semantics и auth wiring вокруг token-driven flow.
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py` в provisional виде естественно скатывается к записи raw provider token в cookie.
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py` имеет token-resolution policy, совместимую с bearer fallback, что не соответствует final browser/web path.
- `src/trading/contexts/identity/adapters/outbound/security/current_user/keycloak_introspection_current_user.py` строит principal из token claims, включая `paid_level`, что противоречит final source-of-truth policy.
- `src/trading/contexts/identity/application/ports/user_repository.py` и связанные adapters в provisional варианте тяготеют к модели `user_id=sub`, а не к `keycloak_subject -> local Roehub user`.

## 3. Необсуждаемые глобальные правила

Любой агент, исполняющий задачи из этого pack, MUST:
- считать `docs/architecture/identity/keycloak-cutover-plan-v1.md` основным source of truth;
- считать приоритетом секции `1`, `2`, `2A`, `5A`, `6`, `6A` этого плана;
- трактовать текущий код как **input for reconciliation**, а не как правильную целевую реализацию;
- сохранять всё, что уже соответствует final architecture;
- удалять/заменять только конфликтующие provisional решения;
- делать минимальный diff без побочных refactor wave;
- не добавлять новые зависимости;
- не трогать unrelated contexts;
- добавлять/обновлять тесты только в рамках текущего шага;
- запускать **только целевые checks шага**, а не весь repo на каждом этапе.

Дополнительные архитектурные lock-rules:
- Browser cookie хранит только opaque Roehub session id.
- Raw Keycloak access/id/refresh tokens не сериализуются в browser cookie.
- `user_id` остаётся внутренним Roehub UUID.
- `keycloak_subject` является внешним auth key.
- `paid_level` читается из Roehub DB, а не из Keycloak claim.
- Browser/web protected path = cookie-only.
- `prod` не использует in-memory session storage.

## 4. Обязательный execution order

Исполнять строго в таком порядке:

1. `P0` — Status Lock + Read Rules
2. `P1` — `Step 1R` Runtime Settings Reconciliation
3. `P2` — `Step 4R` User/Auth Contract Reconciliation
4. `P3` — `Step 6` DB Migration For `keycloak_subject` And `identity_sessions`
5. `P4` — `Step 2R` CurrentUser Session Resolver
6. `P5` — `Step 3R` Auth Endpoints Session Lifecycle
7. `P6` — `Step 6A` Runtime Reconciliation Sweep
8. `P7` — `Step 7` Web Auth UX Switch
9. `P8` — `Step 8` Telegram Scope Cleanup
10. `P9` — `Step 9` Test Migration
11. `P10` — `Step 10` Documentation Sweep
12. `P11` — `Step 11` Final Cleanup
13. `P12` — `Step 12` Production Validation

Почему порядок именно такой:
- сначала закрываем архитектурные и runtime contracts;
- потом фиксируем DB shape;
- затем переводим runtime на persisted session model;
- только после этого доделываем web/tests/docs.

## 5. Универсальный output contract для агента

Перед любыми правками агент обязан коротко перечислить:
1. `retain`
2. `conflicts to remove`
3. `files to edit`

После правок агент обязан вернуть:
1. что изменено;
2. какие checks запущены;
3. что осталось на следующий prompt.

## 6. Prompt Pack

## P0. Status Lock + Read Rules

Тип: no-code orientation step

Цель:
- перечитать `keycloak-cutover-plan-v1.md`;
- принять final architecture как mandatory target;
- не писать код в этом prompt;
- зафиксировать `retain/conflicts/files` для следующего шага.

Prompt:

```text
Прочитай docs/architecture/identity/keycloak-cutover-plan-v1.md и docs/architecture/identity/keycloak-cutover-restart-prompt-pack-v1.md.

Не пиши код.

Считай final architecture mandatory:
- browser cookie stores only opaque Roehub session id
- user_id is internal Roehub UUID
- keycloak_subject is external auth key
- paid_level comes from Roehub DB
- browser/web path is cookie-only

Сделай только короткий execution brief:
1. retain from current tree
2. conflicts to remove
3. files to edit for the next prompt

Не предлагай альтернативную архитектуру.
```

## P1. Step 1R — Runtime Settings Reconciliation

Тип: reopened-delta

Цель:
- привести runtime settings и fail-fast policy к final session model;
- не переписывать весь identity module;
- убрать только конфликтующие runtime assumptions.

Primary files:
- `apps/api/wiring/modules/identity.py`
- `tests/unit/apps/api/test_identity_wiring_module.py`
- `infra/docker/.env.example`
- `infra/docker/docker-compose.yml`
- `infra/docker/docker-compose.backend.yml`
- `infra/macos/launchd/com.roehub.api.plist`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`

Retain:
- Keycloak base realm/client settings;
- `IDENTITY_PG_DSN` и `IDENTITY_EXCHANGE_KEYS_KEK_B64`;
- existing fail-fast discipline in `prod`, where it still matches final model.

Remove / replace:
- legacy/provisional cookie name assumptions around jwt-style auth;
- mandatory reliance on `KEYCLOAK_JWKS_URL` for this cutover path;
- any runtime assumption that browser cookie stores provider token.

Add:
- `IDENTITY_SESSION_COOKIE_NAME`
- `IDENTITY_SESSION_IDLE_TTL_SECONDS`
- `IDENTITY_SESSION_ABSOLUTE_TTL_SECONDS`
- optional `KEYCLOAK_END_SESSION_URL`
- explicit prod fail-fast for persisted session config

DoD:
- wiring reflects server-side session architecture;
- no runtime setting suggests raw Keycloak token cookie path;
- tests cover new settings contract.

Checks:

```bash
uv run pytest -q tests/unit/apps/api/test_identity_wiring_module.py
```

Prompt:

```text
Исполни P1 / Step 1R reconciliation по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Работай against current tree, не как greenfield.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом внеси минимальный diff только для runtime settings и wiring policy:
- browser cookie is opaque session id, not provider token
- add session cookie/TTL settings
- keep Keycloak code-flow settings
- remove mandatory JWKS-driven assumptions for this cutover path

Не переходи к session repository или auth route logic.

После правок запусти:
uv run pytest -q tests/unit/apps/api/test_identity_wiring_module.py
```

## P2. Step 4R — User/Auth Contract Reconciliation

Тип: reopened-delta

Цель:
- перевести identity contracts на модель `keycloak_subject -> local Roehub user`;
- не реализовывать ещё persisted sessions;
- подготовить domain/persistence contracts к Step 6.

Primary files:
- `src/trading/contexts/identity/application/ports/user_repository.py`
- `src/trading/contexts/identity/domain/entities/user.py`
- `src/trading/contexts/identity/application/ports/__init__.py`
- `src/trading/contexts/identity/application/__init__.py`
- `src/trading/contexts/identity/__init__.py`
- related persistence adapter exports if needed

Retain:
- `User` как минимальный identity aggregate snapshot;
- `user_id` как Roehub UUID;
- `paid_level`, `created_at`, `last_login_at`, `is_deleted`.

Remove / replace:
- contracts that imply `user_id=sub`;
- contracts that model `telegram_user_id` as auth key;
- `upsert_oidc_login(user_id=sub)` semantics.

Add:
- `find_by_keycloak_subject(...)`
- `upsert_keycloak_login(keycloak_subject=..., login_at=...)`
- explicit contract wording that `paid_level` comes from Roehub DB

DoD:
- contracts no longer assume direct `sub -> UserId`;
- final DB migration can be implemented without another contract rewrite.

Checks:

```bash
uv run pyright
```

Prompt:

```text
Исполни P2 / Step 4R reconciliation по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Работай against current tree.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом переведи identity contracts на final auth model:
- user_id is internal Roehub UUID
- keycloak_subject is external auth key
- paid_level is sourced from Roehub DB
- no contract may imply user_id=sub

Не реализуй ещё DB migration и не переходи к auth routes.

После правок запусти:
uv run pyright
```

## P3. Step 6 — DB Migration For `keycloak_subject` And `identity_sessions`

Тип: standard-next step

Цель:
- подготовить schema и repositories под final session model.

Primary files:
- `migrations/postgres/0005_identity_keycloak_cutover_v1.sql`
- `migrations/postgres/0001_identity_v1.sql`
- `migrations/postgres/0002_identity_2fa_totp_v1.sql`
- `apps/migrations/bootstrap.py`
- `apps/migrations/bootstrap_main.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/user_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/user_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/session_repository.py`
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/session_repository.py`
- `src/trading/contexts/identity/application/ports/session_repository.py`

Retain:
- current `identity_users` table as Roehub local user store;
- existing timezone normalization discipline;
- current migration numbering sequence.

Remove / replace:
- `telegram_user_id` as required auth invariant;
- missing local session persistence;
- repository logic that only works for `user_id=sub`.

Add:
- `keycloak_subject` column + unique constraint strategy;
- `identity_sessions` table;
- repository contracts for create/read/revoke session lifecycle;
- repo support for `find_by_keycloak_subject` and `upsert_keycloak_login`.

DoD:
- schema supports final auth model;
- repositories support final lookup/upsert/session lifecycle;
- no remaining DB invariant forces legacy auth shape.

Checks:

```bash
uv run pytest -q \
  tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_timezone_normalization.py \
  tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py
```

Prompt:

```text
Исполни P3 / Step 6 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Работай against current tree и уже обновлённые contracts.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом реализуй DB cutover:
- add keycloak_subject as external auth key
- add identity_sessions for persisted Roehub sessions
- make telegram_user_id non-required
- align postgres/in-memory repositories with final contracts

Не переходи к auth route wiring.

После правок запусти:
uv run pytest -q \
  tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_timezone_normalization.py \
  tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py
```

## P4. Step 2R — CurrentUser Session Resolver

Тип: reopened-delta

Цель:
- перевести protected-route principal resolution на persisted Roehub session;
- убрать provisional token-driven browser path.

Primary files:
- `src/trading/contexts/identity/application/ports/current_user.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/adapters/outbound/security/current_user/keycloak_introspection_current_user.py`
- `src/trading/contexts/identity/adapters/outbound/security/current_user/__init__.py`
- `tests/unit/apps/api/test_identity_current_user_dependency.py`

Retain:
- `CurrentUserPrincipal` as stable protected-route contract;
- deterministic unauthorized errors at API boundary.

Remove / replace:
- browser cookie interpreted as provider access token;
- default bearer fallback semantics for web/browser path;
- principal resolution from token claims as final runtime path.

Add:
- session-id resolution from cookie;
- session lookup + Roehub user lookup;
- unauthorized semantics for missing, expired, revoked, malformed session.

DoD:
- `CurrentUser` reads local session and local user snapshot;
- no final browser path depends on provider token from request.

Checks:

```bash
uv run pytest -q tests/unit/apps/api/test_identity_current_user_dependency.py
```

Prompt:

```text
Исполни P4 / Step 2R reconciliation по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Работай against current tree и уже реализованные session/user repositories.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом переведи CurrentUser на final model:
- resolve opaque session id from cookie
- load Roehub session record
- load Roehub user snapshot
- build CurrentUserPrincipal from local data

Удали final browser dependency on provider token and bearer fallback.

После правок запусти:
uv run pytest -q tests/unit/apps/api/test_identity_current_user_dependency.py
```

## P5. Step 3R — Auth Endpoints Session Lifecycle

Тип: reopened-delta

Цель:
- привести `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/current-user` к final session lifecycle.

Primary files:
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `apps/api/routes/identity.py`
- `apps/api/wiring/modules/identity.py`
- `tests/unit/apps/api/test_identity_routes.py`

Retain:
- OIDC Authorization Code Flow structure;
- state/next guards;
- deterministic callback failure semantics.

Remove / replace:
- writing raw provider token into browser cookie;
- logout that only clears cookie without local session revoke;
- `paid_level` sourced from token claims.

Add:
- callback creates/updates Roehub user by `keycloak_subject`;
- callback creates Roehub session record and sets opaque session cookie;
- current-user returns Roehub DB-backed principal;
- logout revokes local session and clears opaque cookie.

DoD:
- browser only receives opaque session cookie;
- current-user is DB-backed;
- logout invalidates local session.

Checks:

```bash
uv run pytest -q tests/unit/apps/api/test_identity_routes.py
```

Prompt:

```text
Исполни P5 / Step 3R reconciliation по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Работай against current tree.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом переведи auth endpoints на final lifecycle:
- /auth/callback exchanges code, resolves keycloak_subject, upserts Roehub user, creates session, sets opaque cookie
- /auth/current-user returns Roehub DB-backed user snapshot
- /auth/logout revokes local session and clears opaque cookie

Не возвращай raw provider token в browser cookie ни в каком виде.

После правок запусти:
uv run pytest -q tests/unit/apps/api/test_identity_routes.py
```

## P6. Step 6A — Runtime Reconciliation Sweep

Тип: standard-next step

Цель:
- согласовать wiring, repositories, current-user и auth routes в единый runtime path;
- убрать любые промежуточные обходы.

Primary files:
- `apps/api/routes/identity.py`
- `apps/api/wiring/modules/identity.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `src/trading/contexts/identity/application/ports/current_user.py`
- `src/trading/contexts/identity/application/ports/session_repository.py`
- `src/trading/contexts/identity/application/ports/user_repository.py`

DoD:
- единый runtime path = `browser cookie -> session repo -> user repo -> principal`;
- no provisional token-path remains in runtime code.

Checks:

```bash
uv run pytest -q \
  tests/unit/apps/api/test_identity_wiring_module.py \
  tests/unit/apps/api/test_identity_current_user_dependency.py \
  tests/unit/apps/api/test_identity_routes.py
```

Prompt:

```text
Исполни P6 / Step 6A по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом сделай reconciliation sweep:
- wiring, routes, current-user and repositories must converge to one final runtime model
- remove any leftover provisional provider-token path
- ensure naming/settings/cookie semantics are consistent everywhere

После правок запусти:
uv run pytest -q \
  tests/unit/apps/api/test_identity_wiring_module.py \
  tests/unit/apps/api/test_identity_current_user_dependency.py \
  tests/unit/apps/api/test_identity_routes.py
```

## P7. Step 7 — Web Auth UX Switch

Тип: standard-next step

Цель:
- завершить web login/logout UX под final API auth model.

Primary files:
- `apps/web/templates/login.html`
- `apps/web/templates/logout.html`
- `apps/web/main/app.py`
- `apps/web/main/api_client.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_api_client.py`

DoD:
- `/login` redirects to `/api/auth/login`;
- Telegram widget removed;
- `/logout` completes Roehub session termination flow.

Checks:

```bash
uv run pytest -q \
  tests/unit/apps/web/test_app_routes.py \
  tests/unit/apps/web/test_api_client.py
```

Prompt:

```text
Исполни P7 / Step 7 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Сначала перечисли:
1. retain
2. conflicts to remove
3. files to edit

Потом приведи web UX к final auth model:
- remove Telegram widget/script callback
- /login should enter Keycloak flow through /api/auth/login
- /logout should terminate Roehub local session, not a legacy jwt-cookie flow

После правок запусти:
uv run pytest -q \
  tests/unit/apps/web/test_app_routes.py \
  tests/unit/apps/web/test_api_client.py
```

## P8. Step 8 — Telegram Scope Cleanup

Тип: standard-next step

Цель:
- убедиться, что `TELEGRAM_BOT_TOKEN` больше не участвует в API-auth path.

Primary files:
- `apps/api/wiring/modules/identity.py`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/architecture/strategy/strategy-runtime-config-v1.md`
- `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md`

DoD:
- Telegram остаётся только в strategy notifier scope.

Checks:

```bash
rg -n "TELEGRAM_BOT_TOKEN|telegram.*auth|auth.*telegram" apps docs src
```

Prompt:

```text
Исполни P8 / Step 8 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Удали только auth-related references.
Не трогай legitimate strategy notifier usage.

После правок проверь grep-ом, что Telegram больше не фигурирует в API-auth path.
```

## P9. Step 9 — Test Migration

Тип: standard-next step

Цель:
- убрать legacy tests и собрать полный target test surface для final auth model.

Primary files:
- `tests/unit/apps/api/test_identity_routes.py`
- `tests/unit/apps/api/test_identity_current_user_dependency.py`
- `tests/unit/apps/api/test_identity_wiring_module.py`
- `tests/unit/apps/api/test_identity_exchange_keys_routes.py`
- `tests/unit/apps/web/test_app_routes.py`
- `tests/unit/apps/web/test_api_client.py`
- `tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py`
- obsolete legacy identity tests scheduled for delete/replace

DoD:
- нет тестов на legacy endpoints/classes;
- есть тесты на Roehub session lifecycle;
- нет тестов, моделирующих browser auth raw token cookie.

Checks:

```bash
uv run pytest -q \
  tests/unit/apps/api/test_identity_routes.py \
  tests/unit/apps/api/test_identity_current_user_dependency.py \
  tests/unit/apps/api/test_identity_wiring_module.py \
  tests/unit/apps/api/test_identity_exchange_keys_routes.py \
  tests/unit/apps/web/test_app_routes.py \
  tests/unit/apps/web/test_api_client.py \
  tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py
```

Prompt:

```text
Исполни P9 / Step 9 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Удали/замени legacy identity tests и собери финальный target test surface под:
- opaque Roehub session cookie
- keycloak_subject -> local Roehub user
- paid_level from Roehub DB

После правок запусти только целевой identity/web test bundle из prompt-pack.
```

## P10. Step 10 — Documentation Sweep

Тип: standard-next step

Цель:
- синхронизировать docs с final architecture.

DoD:
- docs больше не описывают Telegram login, JWT cookie auth, local 2FA как auth source;
- docs явно фиксируют server-side session model.

Checks:

```bash
rg -n "auth/telegram/login|JWT cookie|local 2FA|telegram-widget.js|user_id=sub" docs
```

Prompt:

```text
Исполни P10 / Step 10 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Синхронизируй docs только под final auth model.
Не оставляй описаний provisional raw-token cookie path.
```

## P11. Step 11 — Final Cleanup

Тип: standard-next step

Цель:
- убрать stale exports/imports и dead references после cutover.

DoD:
- runtime tree не содержит legacy auth symbols вне archival docs.

Checks:

```bash
rg "telegram login|auth_telegram|Hs256JwtCodec|JwtCookieCurrentUser|/2fa/setup|/2fa/verify|IDENTITY_JWT_SECRET" .
```

Prompt:

```text
Исполни P11 / Step 11 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Сделай только cleanup после уже завершённого cutover.
Не меняй поведение, только удаляй stale references и exports.
```

## P12. Step 12 — Production Validation

Тип: standard-final step

Цель:
- пройти финальный validation checklist для cutover.

Validation targets:
- API startup in `prod` no longer requires legacy auth secrets;
- browser auth cookie contains only opaque session id;
- `/auth/login -> Keycloak -> /auth/callback -> /auth/current-user` works;
- protected routes read principal from local session + Roehub DB;
- exchange-keys work without local 2FA;
- web protected pages redirect correctly;
- docs/runbooks updated.

Checks:

```bash
uv run pytest -q \
  tests/unit/apps/api/test_identity_routes.py \
  tests/unit/apps/api/test_identity_current_user_dependency.py \
  tests/unit/apps/api/test_identity_wiring_module.py \
  tests/unit/apps/api/test_identity_exchange_keys_routes.py \
  tests/unit/apps/web/test_app_routes.py \
  tests/unit/apps/web/test_api_client.py \
  tests/unit/contexts/identity/adapters/outbound/persistence/postgres/test_identity_session_repository.py
```

Prompt:

```text
Исполни P12 / Step 12 по docs/architecture/identity/keycloak-cutover-plan-v1.md.

Ничего не реархитектурируй.
Проверь final checklist, добей только missing gaps, затем верни:
1. что validated
2. что не validated
3. какие manual prod checks ещё нужны
```

## 7. Stop-Line Rules

Агент MUST остановиться и сообщить о blocker, если:
- для final runtime path всё ещё требуется raw Keycloak token в browser cookie;
- `keycloak_subject` невозможно получить детерминированно из callback payload;
- local Roehub session persistence невозможно внедрить без изменения уже зафиксированных contracts;
- в worktree обнаружен конфликт с чужими незавершёнными правками в тех же файлах.

## 8. Closure Condition

Restart-pack считается закрытым только когда:
- `P1-P12` завершены по порядку;
- reopened steps больше не имеют provisional leftovers;
- final runtime path соответствует `browser -> opaque session cookie -> Roehub session store -> Roehub user snapshot`.
