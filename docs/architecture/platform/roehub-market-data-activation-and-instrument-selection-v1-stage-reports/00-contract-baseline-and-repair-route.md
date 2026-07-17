# Этап 00 — baseline контрактов и маршрут исправления

## Результат

Этап принят. Создан отдельный последовательный repair route, который не меняет исторический статус Stage `24` исходного self-hosted плана и не обходит его требования к ingress и native `linux/amd64`.

## Проверенные факты

- `roehub` является единственной `internal` Docker-сетью. `market-data-ws` и `market-data-scheduler` подключены только к ней, поэтому не могут резолвить публичные exchange endpoints.
- `market-data-ws` получает перечень включённых инструментов единожды при старте. Scheduler наполняет reference set позднее, что оставляет worker без subscription.
- Mounted `whitelist.csv` — текущий глобальный источник policy; он не отражает организационное намерение пользователя и не допускает безопасного независимого редактирования.
- `ref_instruments` — глобальный биржевой reference set. В нём нет выбора организации, состояния strategy pin, coverage или artifact inventory.
- Publisher container ограничен `1 GiB`, но internal budget/parallelism допускают больше доступной памяти. Первое построение должно быть ручным и single-symbol.
- Сохранённая интерактивная OpenBao instance безопасно неинициализирована и sealed. Docker health означает только доступность процесса, а не готовность secret service.

## Матрица контрактов

| Поверхность | Старый контракт | Новый контракт | Класс | Миграция и проверка |
|---|---|---|---|---|
| `public_api` | Только enabled reference/search и BTCUSDT readiness | Каталог, org selections, effective pins, coverage, artifact inventory | `compatible-change` | Новые versioned routes/DTO; API/RBAC tests |
| `port_contract` | Scheduler получает CSV, WS одноразовый reader | Effective-selection reader и refreshable subscription plan | `breaking-change` | New port/adapters, worker regression/runtime proof |
| `DTO_schema` | Reference market DTO | Additive catalogue/selection/coverage DTO | `compatible-change` | Schema/API consumer tests |
| `persisted_schema` | Global ClickHouse reference | PostgreSQL organization selections plus read models | `breaking-change` | Forward migration, constraints, empty install proof |
| `config_schema` | `MARKET_DATA_WHITELIST_PATH` and generated CSV | Bootstrap default plus runtime selection policy | `breaking-change` | Versioned generated config; no silent fallback |
| `service_call` | Internal-only services unable to reach exchanges | Egress for exactly two bounded workers | `breaking-change` | Compose exact-network assertion, adapter allowlist, timeouts/error metrics |
| `external_effect` | No external data requests from workload | Public market data only, no credentials/orders; bounded retries | `compatible-change` | One-symbol runtime proof and counter checks |
| `metrics_runbooks` | Liveness metrics only | Readiness: connection/messages/inserts/freshness/errors | `compatible-change` | Prometheus/operational-health/runbook checks |
| `browser_default` | File driven list invisible to user | Onboarding/settings choice and status | `compatible-change` | Browser, mobile and accessibility smoke |
| `secret_delivery` | Shared runtime secret volume and disposable verification | owner-operated 3/2 bootstrap + least privilege tokens | `breaking-change` | Disposable proof; durable custody owner action |

## Решения

1. `InstrumentSelection` хранит намерение организации; `OrganizationEffectiveSelection` объединяет selections и active strategy pins, а `GlobalEffectiveCollectorSet` объединяет organisation-effective sets только для workers. API не раскрывает межорганизационные selections/pins. Изменение выбора не блокируется стратегией.
2. Каталог обновляется отдельным bounded metadata-only refresh и имеет `fresh`/`stale`/`failed`; он не равен automatic selection/backfill. Начальный controlled rollout — только `binance:futures:BTCUSDT`.
3. Coverage возвращает `unknown`, когда нет доказанного expected interval; `100%` допускается только с рассчитанными expected closed minutes. Artifact size — фактические bytes current published slot.
4. Docker egress network ограничивает субъектов доступа, но не домены. В v1 exchange adapters являются запретительной границей allowed providers/endpoints; proxy-based FQDN enforcement вынесен из этого ремонта.
5. Stage `04` докажет disposable path с тремя PGP recipients и threshold `2-of-3`, затем передаст durable custody владельцу. Private keys, shares, recovery identity и tokens не создаются и не хранятся исполнительной средой.

## Доказательства и ограничения

Проверены manifest/schema/topology source, worker startup path, scheduler sequence, publisher runtime budget, UI/API/market-data entrypoints и состояние локальной Docker Desktop среды. Независимая cold-head проверка заблокировала первый вариант из-за отсутствия index evidence, tenant/catalog/publisher contract gaps и неверной формулировки OpenBao quorum. Все замечания исправлены; `uv run python -m tools.docs.generate_docs_index --check` и `git diff --check` прошли. Это design/config доказательство; внешние exchange/DB/browser действия начнутся только с Stage `01`.

## Файлы и handoff

Созданы plan doc, stage ledger, 6 independently executable prompts и этот report. Чужие изменения отсутствовали на старте ветки. Следующий разрешённый этап: `01`.
