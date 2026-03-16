# Переезд продакшена с Linux на Mac Studio + VPS edge

Статус: завершенный этап migration на новую topology.

Документ сохранен как краткая фиксация итоговой схемы и точек входа в актуальные runbook'и.

## Итоговая production topology

- `VPS` остается единственным публичным edge (`roehub.com`, TLS, reverse proxy `/api/*`).
- `Mac Studio` остается private backend host.
- backend runtime на `Mac Studio` - полностью native (без Docker/Colima в target state).

## Что считать актуальным source of truth

- migration plan: `docs/architecture/roadmap/mac-studio-native-backend-migration-plan-v1.md`
- backend operations: `docs/runbooks/mac-studio-native-backend-operations.md`
- web same-origin/edge contract: `docs/runbooks/web-ui-gateway-same-origin.md`

## Что считать legacy

- любые Docker/Colima инструкции для production backend на `Mac Studio`;
- старые шаги deploy через `docker compose` на `Mac Studio`;
- старые команды, предполагающие production runtime в контейнерах.

## Базовые acceptance-checks

- `https://roehub.com/` -> `200`
- `https://roehub.com/api/auth/current-user` -> `401`
- локально на `Mac Studio`: `http://127.0.0.1:8000/auth/current-user` -> `401`
- `tailscale serve status` указывает на native local ports, а не на Docker runtime.
