# Strategy live worker runbook

Статус: архивный документ для STR-EPIC-06 experimental path.

Текущий production native backend runtime на `Mac Studio` не включает отдельный
`strategy-live-worker` service в обязательном контуре миграции.

## Что использовать вместо этого

- базовые operations и smoke для production backend: `docs/runbooks/mac-studio-native-backend-operations.md`
- market data streams / metrics:
  - `docs/runbooks/market-data-redis-streams.md`
  - `docs/runbooks/market-data-metrics.md`
  - `docs/runbooks/market-data-metrics-reference-ru.md`

## Архитектурные документы STR-EPIC-06

- `docs/architecture/strategy/strategy-runtime-config-v1.md`
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md`

Если нужно вернуть production запуск `strategy-live-worker` как постоянный сервис,
это отдельное архитектурное решение и отдельный rollout-plan для native topology.
