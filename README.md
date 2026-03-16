# Roehub

## Operations Notes

- Production backend runtime on `Mac Studio` is native (no Docker/Colima target state).
- Public edge and TLS stay on `VPS`; backend on `Mac Studio` stays private and is reached via same-origin `/api/*` contract from the edge.
- Standalone `market_data` compose (`infra/docker/docker-compose.market_data.yml`) is local/dev legacy path only.

Runbooks:
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/market-data-autonomous-docker.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
