# Identity 2FA TOTP policy v1

Статус: архивный документ.

Этот документ описывал legacy local-2FA модель (`/2fa/setup`, `/2fa/verify`, local gate в API) и больше не является source of truth.

## Что заменило

- Каноническая модель auth: `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- Операционный runbook: `docs/runbooks/keycloak-local-setup-and-ops.md`

## Актуальная позиция

- В API Roehub нет локальных `/2fa/*` endpoint-ов.
- `RequireTwoFactorEnabledDependency` не используется.
- 2FA policy реализуется в Keycloak (realm/client authentication flow, OTP policy).
