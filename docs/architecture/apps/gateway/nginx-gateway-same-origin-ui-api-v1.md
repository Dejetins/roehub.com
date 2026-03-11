# Nginx Gateway v1 -- Same-Origin UI + API routing (WEB-EPIC-02)

Статус: архивный документ.

`nginx gateway` удален из репозитория и больше не используется ни в prod, ни в local/dev.

Актуальная схема same-origin:

- production: `Caddy` на `VPS` проксирует `/api/*` на private API `Mac Studio`;
- local/dev: `apps/web` сам проксирует `/api/*` в `WEB_API_UPSTREAM_URL`.

Актуальные документы:

- `docs/runbooks/prod-migration-linux-to-mac-studio.md`
- `docs/runbooks/web-ui-gateway-same-origin.md`
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md`

Этот файл сохранен только для historical reference и не описывает текущий runtime path.
