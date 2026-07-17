# Roehub UI Autostart (systemd) + Helper Commands

Статус: архивный документ.

Этот ранбук описывает старую Linux/systemd схему для `api + web + gateway` в одном compose-стеке.
Текущий production Roehub больше так не работает:

- публичный web ingress и TLS живут на `VPS`;
- backend/data/compute живут на `Mac Studio`;
- production deploy выполняется через GitHub Actions workflows, а не через `systemd` helper-скрипты.

Что использовать вместо этого:

- основной runbook: `docs/runbooks/mac-studio-native-backend-operations.md`
- dev/local same-origin stack without separate gateway: `docs/runbooks/web-ui-gateway-same-origin.md`
- production backend manifest: `infra/docker/docker-compose.backend.yml`
- production web manifest: `infra/docker/docker-compose.web.prod.yml`

Этот документ оставлен только как historical reference и не должен использоваться для текущего production deploy.
