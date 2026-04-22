# Identity exchange keys storage + 2FA gate policy v1

Статус: архивный документ.

Документ фиксировал раннюю версию exchange-keys policy с local 2FA gate в API.

## Что заменило

- Актуальная auth-модель: `docs/architecture/identity/identity-keycloak-auth-model-v1.md`
- Актуальная storage/policy версия: `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md`

## Актуальная позиция

- Exchange keys endpoints защищаются стандартным `current_user` (Roehub session cookie).
- Local 2FA gate в API не используется.
- Если требуется OTP enforcement, он выполняется в Keycloak policy до выдачи Roehub session.
