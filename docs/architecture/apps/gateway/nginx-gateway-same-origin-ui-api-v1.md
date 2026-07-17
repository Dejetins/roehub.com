# Nginx Gateway v1 -- Same-Origin UI + API routing (WEB-EPIC-02)

Статус: архивный документ.

`nginx gateway` удален из репозитория и больше не используется ни в prod, ни в local/dev.

Текущий remote same-origin runtime не настроен. Локальный `apps/web` может
проксировать `/api/*` в `WEB_API_UPSTREAM_URL` только для явно выбранного
development ticket.

Этот файл сохранен только для historical reference и не описывает текущий runtime path.
