# Identity v1: Telegram-only login + user model + CurrentUser

Статус: архивный документ.

Этот документ описывал legacy auth-модель (Telegram Login Widget + локальный JWT cookie) и больше не является source of truth.

## Что заменило

- Каноническая модель auth: `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- Runtime/ops setup: `docs/runbooks/keycloak-local-setup-and-ops.md`

## Актуальная позиция

- Telegram login не используется как auth source.
- Локальный HS256 JWT cookie не используется как auth source.
- Browser auth-cookie в Roehub хранит только opaque session id (`roehub_session_id`).
- Внешний auth key: `keycloak_subject` (`sub` из Keycloak).
