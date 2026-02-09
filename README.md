# Roehub

## Operations Notes

- Production docker deployment uses a single compose file: `infra/docker/docker-compose.yml`.
- `market-data-ws-worker` and `market-data-scheduler` are part of that stack and are scraped by Prometheus via service DNS (`market-data-ws-worker:9201`, `market-data-scheduler:9202`).
- Standalone `market_data` compose (`infra/docker/docker-compose.market_data.yml`) is for local/dev автономного запуска.

Runbooks:
- `docs/runbooks/market-data-autonomous-docker.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
