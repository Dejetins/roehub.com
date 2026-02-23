# Web UI v1 -- SSR + HTMX skeleton + Auth UX (WEB-EPIC-01)

Документ фиксирует архитектуру WEB-EPIC-01: `apps/web` как отдельный web upstream (Python SSR + Jinja2 + HTMX), который рендерит HTML, использует существующий JSON API через `/api/...` и обеспечивает обязательный login gate.

## Цель

- Запустить минимальный web UI процесс отдельно от API.
- Зафиксировать UX авторизации через Telegram Login Widget (Variant A) и cookie-based identity.
- Заложить каркас страниц и общий паттерн интеграции web -> api (server-side calls + browser-side `/api/*` calls).

## Контекст

- В проекте уже есть JSON API контракты:
  - identity: `POST /auth/telegram/login`, `POST /auth/logout`, `GET /auth/current-user`.
  - backtest: `POST /backtests`.
  - backtest jobs: `POST/GET /backtests/jobs*`.
  - strategy: `POST/GET/DELETE /strategies*`, `POST /strategies/clone`.
- Milestone 6 фиксирует same-origin delivery через gateway (Nginx): браузер вызывает API только по `/api/...`.
- `apps/web` в WEB-EPIC-01 является HTML facade над API (web не wires domain/application use-cases напрямую).

См.:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-01.
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — Telegram login + JWT cookie policy.
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py` — фактические identity endpoints.

## Scope

### 1) `apps/web` как отдельный процесс

- App factory + router.
- SSR templates (Jinja2) и базовый layout:
  - navigation (ссылки на `/`, `/strategies`, `/backtests`, `/backtests/jobs`),
  - user badge (paid level + user_id в сокращенном виде),
  - error banner для ошибок API/web.
- HTMX partials (минимально): завести паттерн partial endpoints для таблиц/форм.

Страницы v1 (каркас):

- `/` — публичный landing (без требования auth).
- `/login` — login page с Telegram widget.
- `/logout` — logout page (см. решение 4).
- `/strategies` — protected page (пока каркас).
- `/backtests` — protected page (пока каркас).
- `/backtests/jobs` — protected page (пока каркас).

### 2) Auth UX (Telegram Login Widget Variant A)

- Login widget использует `bot_username=RoehubAuth_bot`.
- JS отправляет payload в `POST /api/auth/telegram/login` (JSON) и после успеха делает redirect.
- Protected pages делают server-side check `GET /api/auth/current-user` и при 401 редиректят на `/login`.

### 3) Internal API client (web -> api)

- `WEB_API_BASE_URL` (env) задаёт базовый URL API для server-side запросов web приложения.
  Примеры:
  - docker: `http://api:8000`
  - локально: `http://127.0.0.1:8000`

- Для authenticated вызовов web форвардит `Cookie` header из browser request в API request.

## Non-goals

- SPA framework.
- UI для 2FA и exchange keys.
- Realtime streams UI (SSE/WebSocket).
- Реализация полноценных UI фич стратегии/backtest/jobs (это WEB-EPIC-04..06); в WEB-EPIC-01 только каркас страниц и паттерны.

## Ключевые решения

### 1) `apps/web` = HTML facade над JSON API (без прямого wiring use-cases)

Web приложение не импортирует и не wires `src/trading/**` use-cases. Вместо этого:

- browser-side взаимодействия используют `/api/...` (через gateway в Milestone 6);
- server-side (login gate, page loaders) используют internal HTTP client к `WEB_API_BASE_URL`.

Причины:
- минимизируем дублирование composition root (уже есть `apps/api/main/app.py`);
- сохраняем один source-of-truth контрактов: JSON API.

Последствия:
- web зависит от стабильности API контрактов (это ожидаемо);
- тесты web в основном проверяют routing/login gate и корректное поведение при 401/ошибках api.

### 2) Public landing на `/`, protected pages редиректят на `/login`

`/` не требует auth и не показывает защищённые данные.

Protected pages (`/strategies`, `/backtests`, `/backtests/jobs`) выполняют проверку current user:

- `GET /api/auth/current-user` (server-side вызов через `WEB_API_BASE_URL` + forwarded Cookie)
- при 200: рендер страницы
- при 401: redirect на `/login`

### 3) Статические ассеты по префиксу `/assets/*`

Все ссылки на CSS/JS в templates используют `/assets/...`.

Причина:
- этот префикс закреплён в WEB-EPIC-02 для Nginx gateway.

### 4) Logout реализован через JS call к `/api/auth/logout` (вариант 1)

`/logout` — web page с минимальным JS:

- делает `fetch('/api/auth/logout', { method: 'POST', credentials: 'include' })`
- затем `window.location = '/login'`

Почему:
- cookie очищается самим API (`Set-Cookie` deletion в ответе `POST /auth/logout`),
- web не обязан знать cookie name/secure/samesite параметры.

### 5) Login redirect и защита от open redirect

`/login` принимает `next` query param для возврата после логина.

Правило безопасности:
- разрешаем только относительные пути, начинающиеся с `/`.
  Например:
  - OK: `/strategies`
  - NOT OK: `https://evil.com/` (должно быть проигнорировано и заменено на `/`)

## Контракты и инварианты

- UI вызывает JSON API только по `/api/...` (browser-side контракт).
- Web server (SSR) использует `WEB_API_BASE_URL` для server-side вызовов к API.
- Web server форвардит `Cookie` header как есть (без разбора/парсинга cookie name).
- Для protected pages:
  - `401` от `GET /api/auth/current-user` -> redirect на `/login`.
- `/logout` всегда вызывает `POST /api/auth/logout` и всегда редиректит на `/login`.
- `/assets/*` — единый префикс статических файлов.
- Bot username для Telegram widget фиксирован: `RoehubAuth_bot`.

## Связанные файлы

Docs:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` — WEB-EPIC-01 scope.
- `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — Telegram login + cookie policy.

API identity:
- `apps/api/routes/identity.py` — identity router facade.
- `src/trading/contexts/identity/adapters/inbound/api/routes/auth_telegram.py` — `/auth/telegram/login`, `/auth/logout`, `/auth/current-user`.
- `src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py` — 401 policy.

Web (будущая реализация):
- `apps/web/main/app.py` — web app factory.
- `apps/web/main/main.py` — web process entrypoint.
- `apps/web/templates/**` — Jinja2 templates.
- `apps/web/dist/**` — статические ассеты (pin versions).

## Как проверить

```bash
# 1) линтер
uv run ruff check .

# 2) типы
uv run pyright

# 3) тесты
uv run pytest -q

# 4) docs index (если добавлялся/обновлялся этот документ)
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

Manual smoke (после появления gateway в WEB-EPIC-02):

1) Открыть `https://roehub.com/login`.
2) Нажать Telegram login, убедиться что `POST /api/auth/telegram/login` устанавливает cookie.
3) Открыть `/strategies` -> страница доступна.
4) Открыть `/logout` -> cookie очищается, происходит redirect на `/login`.

## Риски и открытые вопросы

- Риск: cookie-based auth без CSRF защиты для state-changing запросов. Митигация v1: same-origin + `SameSite=lax`. CSRF вводим в отдельном milestone/epic при необходимости.
- Риск: без gateway (до WEB-EPIC-02) browser-side `/api/*` не будет доступен. Для проверки end-to-end нужен Nginx.
