# Web UI v1 -- SSR + HTMX skeleton + Auth UX (WEB-EPIC-01)

Документ фиксирует архитектуру WEB-EPIC-01: `apps/web` как отдельный web upstream (Python SSR + Jinja2 + HTMX), который рендерит HTML, использует JSON API через `/api/...` и обеспечивает обязательный login gate.

## Цель

- запустить минимальный web UI процесс отдельно от API;
- зафиксировать UX авторизации через Keycloak redirect flow;
- закрепить server-side login gate через `/api/auth/current-user`.

## Контекст

- JSON API identity surface:
  - `GET /auth/login`
  - `GET /auth/callback`
  - `POST /auth/logout`
  - `GET /auth/current-user`
- Milestone 6 фиксирует same-origin delivery: browser вызывает API только по `/api/...`.
- `apps/web` — HTML facade над API, без прямого wiring use-cases из `src/trading/**`.

См.:
- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`

## Scope

### 1) `apps/web` как отдельный процесс

- app factory + router;
- SSR templates (Jinja2) и базовый layout;
- страницы v1:
  - `/` (public landing)
  - `/login`
  - `/logout`
  - `/strategies` (protected)

### 2) Auth UX (Keycloak OIDC)

- `/login` рендерит кнопку/ссылку в `GET /api/auth/login`;
- `/api/auth/login` запускает OIDC flow и делает redirect в Keycloak;
- `GET /auth/callback` в API завершает code flow и ставит opaque cookie `roehub_session_id`;
- protected pages делают server-side check `GET /api/auth/current-user`;
- при `401` web редиректит на `/login`.

### 3) Logout UX

- `/logout` вызывает `POST /api/auth/logout`;
- API ревокает локальную Roehub session и очищает auth-cookie;
- web редиректит пользователя на `/login`.

### 4) Internal API client (web -> api)

- `WEB_API_BASE_URL` — base URL для server-side API calls;
- `WEB_API_UPSTREAM_URL` — upstream URL для встроенного `/api/*` proxy;
- authenticated server-side вызовы форвардят `Cookie` header из browser request.

## Non-goals

- SPA framework;
- local auth/JWT/Telegram widget;
- local `/2fa/*` UI;
- realtime streams UI (SSE/WebSocket).

## Ключевые решения

### 1) `apps/web` = HTML facade над JSON API

Web не wires domain/application use-cases напрямую, а работает через стабильный API-контракт.

### 2) Login gate централизован через `/api/auth/current-user`

Auth state определяется только ответом API current-user dependency:

- `200` -> страница доступна;
- `401` -> redirect на `/login`.

### 3) Browser cookie — только opaque Roehub session id

Web не читает и не парсит provider token; auth-cookie обрабатывается как opaque cookie, принадлежащая API.

### 4) Open redirect защита на `/login?next=`

Разрешаются только относительные пути, начинающиеся с `/`.

## Контракты и инварианты

- browser-side API calls только через `/api/...`;
- web server форвардит `Cookie` header без знания структуры auth-токенов;
- `/logout` всегда выполняет `POST /api/auth/logout`;
- Telegram widget script не используется в templates.

## Связанные файлы

Docs:
- `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`

API:
- `apps/api/routes/identity.py`
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_oidc.py`
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py`

Web:
- `apps/web/main/app.py`
- `apps/web/main/api_client.py`
- `apps/web/templates/login.html`
- `apps/web/templates/logout.html`

## Как проверить

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Manual smoke:

1. открыть `/login` и перейти по кнопке `Continue with Keycloak`;
2. пройти Keycloak login;
3. открыть `/strategies` (должно быть `200`);
4. открыть `/logout` и убедиться в redirect на `/login`.
