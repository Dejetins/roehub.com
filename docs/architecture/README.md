# Архитектурная документация

`docs/architecture/README.md` — каноническая точка входа для навигации по всем документам в `docs/**`.

## Ключевые документы

- [Indicators Architecture](docs/architecture/indicators/README.md)
- [Market Data — Application Ports (Walking Skeleton v1)](docs/architecture/market_data/market-data-application-ports.md)
- [Strategy — Milestone 3 Epics](docs/architecture/strategy/strategy-milestone-3-epics-v1.md)
- [Shared Kernel — Primitives](docs/architecture/shared-kernel-primitives.md)
- [Runbook: Help commands](docs/runbooks/help_commands.md)

## Как обновлять индекс

- Обновить файл: `python -m tools.docs.generate_docs_index`
- Проверить актуальность: `python -m tools.docs.generate_docs_index --check`

## Содержание

<!-- BEGIN GENERATED DOCS INDEX -->
### Архитектура

- [Архитектурная документация](docs/architecture/README.md) — `docs/architecture/README.md` — `docs/architecture/README.md` — каноническая точка входа для навигации по всем документам в `docs/**`.
- [CLI — Backfill 1m Candles (Parquet -> ClickHouse) (Walking Skeleton v1)](docs/architecture/apps/cli/cli-backfill-1m.md) — `docs/architecture/apps/cli/cli-backfill-1m.md` — Этот документ фиксирует CLI entrypoint (composition root) для запуска walking skeleton v1:
- [Indicators Architecture](docs/architecture/indicators/README.md) — `docs/architecture/indicators/README.md` — Документы архитектуры bounded context `indicators`.
- [Indicators — Порты application + доменный walking skeleton v1](docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md) — `docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md` — Этот документ является source of truth для **IND-EPIC-01**.
- [Indicators — CandleFeed ACL: dense timeline + NaN holes (v1)](docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md) — `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md` — Этот документ фиксирует архитектуру и контракты для **IND-EPIC-04 — CandleFeed ACL: dense timeline + NaN holes** в bounded context `indicators`.
- [Indicators — Compute Engine Core (Numba) v1](docs/architecture/indicators/indicators-compute-engine-core.md) — `docs/architecture/indicators/indicators-compute-engine-core.md` — Документ фиксирует фактическую реализацию `IND-EPIC-05` в `indicators`: CPU/Numba skeleton, warmup, runtime config, total memory guard.
- [Indicators — Grid Builder + Batch Estimator + Guards (v1)](docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md) — `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md` — **Документ:** `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
- [Indicators — MA compute (Numba/Numpy) + sources v1](docs/architecture/indicators/indicators-ma-compute-numba-v1.md) — `docs/architecture/indicators/indicators-ma-compute-numba-v1.md` — Этот документ является source of truth для **IND-EPIC-06 — Реализация группы MA + базовые “строительные блоки”** в bounded context `indicators`.
- [Indicators — MA](docs/architecture/indicators/indicators-ma.md) — `docs/architecture/indicators/indicators-ma.md` — Документ фиксирует поддерживаемые `ma.*` идентификаторы, ссылки на реализацию и практические ограничения для сопровождения группы MA.
- [Indicators — Momentum](docs/architecture/indicators/indicators-momentum.md) — `docs/architecture/indicators/indicators-momentum.md` — Документ фиксирует поддерживаемые `momentum.*` идентификаторы, ключевые ссылки на реализацию и типовые ошибки сопровождения для группы Momentum.
- [Indicators Overview](docs/architecture/indicators/indicators-overview.md) — `docs/architecture/indicators/indicators-overview.md` — `indicators` в Milestone 2 отвечает за детерминированный расчёт индикаторных тензоров и за контракты, которые нужны для безопасного расширения библиотеки индикаторов без "магии".
- [Indicators — Registry + YAML Defaults (v1)](docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md) — `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md` — Этот документ фиксирует архитектуру и контракты для **IND-EPIC-02 — Registry (code defs + YAML defaults)** в bounded context `indicators`.
- [indicators-structure-normalization-compute-numba-v1.md](docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md) — `docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md` — Этот документ является **source of truth** для **IND-EPIC-09 — Structure/Normalization features (“признаки режима”)** в bounded context `indicators`.
- [Indicators — Structure](docs/architecture/indicators/indicators-structure.md) — `docs/architecture/indicators/indicators-structure.md` — Документ фиксирует поддерживаемые `structure.*` идентификаторы, различия между formula spec и prod defaults, а также gotchas группы Structure/Normalization.
- [Indicators — Trend + Volume compute (Numba/Numpy) + outputs v1](docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md) — `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md` — Этот документ является **source of truth** для **IND-EPIC-08 — Trend + Volume (каналы/пробои/денежный поток)** в bounded context `indicators`.
- [Indicators — Trend](docs/architecture/indicators/indicators-trend.md) — `docs/architecture/indicators/indicators-trend.md` — Документ фиксирует поддерживаемые `trend.*` идентификаторы, ссылки на реализацию и семантические ограничения группы Trend.
- [Indicators — Volatility + Momentum compute (Numba/Numpy) v1](docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md) — `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md` — Этот документ является source of truth для **IND-EPIC-07 — Volatility + Momentum (основа для будущих стратегий)** в bounded context `indicators`.
- [Indicators — Volatility](docs/architecture/indicators/indicators-volatility.md) — `docs/architecture/indicators/indicators-volatility.md` — Документ фиксирует поддерживаемые `volatility.*` идентификаторы, ссылки на реализацию и эксплуатационные нюансы группы Volatility.
- [Indicators — Volume](docs/architecture/indicators/indicators-volume.md) — `docs/architecture/indicators/indicators-volume.md` — Документ фиксирует поддерживаемые `volume.*` идентификаторы, ссылки на реализацию и критичные edge-cases группы Volume.
- [Market Data — Application Ports (Walking Skeleton v1)](docs/architecture/market_data/market-data-application-ports.md) — `docs/architecture/market_data/market-data-application-ports.md` — Этот документ фиксирует минимальные контракты (ports) application-слоя для bounded context `market_data`
- [Market Data — Live Feed to Strategies via Redis Streams (v1)](docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md) — `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md` — This document defines EPIC 4 live feed delivery for Market Data v2:
- [Market Data — Real Adapters: ClickHouse + Parquet Source (Walking Skeleton v1)](docs/architecture/market_data/market-data-real-adapters-clickhouse-parquet.md) — `docs/architecture/market_data/market-data-real-adapters-clickhouse-parquet.md` — Этот документ фиксирует “компромиссный” шаг walking skeleton v1:
- [Market Data — Reference Data Sync (Whitelist -> ClickHouse) (v2)](docs/architecture/market_data/market-data-reference-data-sync-v2.md) — `docs/architecture/market_data/market-data-reference-data-sync-v2.md` — Этот документ фиксирует правила и минимальные механизмы заполнения reference-таблиц ClickHouse
- [Market Data — REST Historical Catch-up + Gap Fill 1m (v2)](docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md) — `docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md` — Этот документ фиксирует правила и минимальные механизмы **REST-догонки** и **автоматического gap fill**
- [Market Data — Runtime Config & Invariants (v2)](docs/architecture/market_data/market-data-runtime-config-invariants-v2.md) — `docs/architecture/market_data/market-data-runtime-config-invariants-v2.md` — Этот документ фиксирует runtime-конфигурацию и инварианты исполнения для bounded context `market_data`
- [Market Data — Use-Case: Backfill 1m Candles (Walking Skeleton v1)](docs/architecture/market_data/market-data-use-case-backfill-1m.md) — `docs/architecture/market_data/market-data-use-case-backfill-1m.md` — Этот документ фиксирует первый application use-case bounded context `market_data` для walking skeleton v1.
- [Market Data — WS Live Ingestion Worker & Maintenance Scheduler (v1)](docs/architecture/market_data/market-data-ws-live-ingestion-worker-v1.md) — `docs/architecture/market_data/market-data-ws-live-ingestion-worker-v1.md` — Этот документ описывает **EPIC 3 — WS Live Ingestion Worker: close-only, <1s to raw** и сопутствующий **maintenance scheduler**.
- [Shared Kernel — Primitives (Trading)](docs/architecture/shared-kernel-primitives.md) — `docs/architecture/shared-kernel-primitives.md` — Этот документ фиксирует минимальные доменные примитивы Shared Kernel и правила, которые должны быть одинаковыми во всех bounded contexts.
- [Strategy — User Goal & Scope (Milestone 3 / EPIC 0)](docs/architecture/strategy/strategy-milestone-3-epics-v1.md) — `docs/architecture/strategy/strategy-milestone-3-epics-v1.md` — Документ фиксирует целевой пользовательский сценарий и границы `Strategy v1` (Milestone 3) перед реализацией EPIC’ов 1–13.

### Ранбуки

- [help commands](docs/runbooks/help_commands.md) — `docs/runbooks/help_commands.md`
- [Runbook — Numba cache dir + threads](docs/runbooks/indicators-numba-cache-and-threads.md) — `docs/runbooks/indicators-numba-cache-and-threads.md` — Документ описывает, как управлять `numba_cache_dir` и количеством потоков (`NUMBA_NUM_THREADS`) для `indicators` compute.
- [Runbook — Numba warmup / JIT](docs/runbooks/indicators-numba-warmup-jit.md) — `docs/runbooks/indicators-numba-warmup-jit.md` — Документ помогает диагностировать задержки первого расчёта индикаторов и отличать ожидаемый JIT warmup от неисправности.
- [Runbook — Troubleshooting: why NaN?](docs/runbooks/indicators-why-nan.md) — `docs/runbooks/indicators-why-nan.md` — Документ объясняет, откуда берутся `NaN` в `indicators` и как отличить корректную политику от дефекта.
- [Market Data Docker Runbook](docs/runbooks/market-data-autonomous-docker.md) — `docs/runbooks/market-data-autonomous-docker.md` — Runbook для `market-data-ws-worker` и `market-data-scheduler`.
- [Market Data Metrics Reference (RU)](docs/runbooks/market-data-metrics-reference-ru.md) — `docs/runbooks/market-data-metrics-reference-ru.md` — Подробный справочник метрик для:
- [Market Data Metrics](docs/runbooks/market-data-metrics.md) — `docs/runbooks/market-data-metrics.md` — Документ фиксирует основные Prometheus-метрики для:
- [Runbook — Market Data Redis Streams](docs/runbooks/market-data-redis-streams.md) — `docs/runbooks/market-data-redis-streams.md` — Операционные команды для live feed stream’ов, которые публикует `market-data-ws-worker`.

### API

- (пока нет документов)

### Решения

- (пока нет документов)

### Прочее

- [Шаблон архитектурного документа](docs/_templates/architecture-doc-template.md) — `docs/_templates/architecture-doc-template.md` — Используйте этот шаблон для новых документов в `docs/architecture/**`; сразу после `# H1` оставляйте краткое описание в одной строке.
- [repository three](docs/repository_three.md) — `docs/repository_three.md`
<!-- END GENERATED DOCS INDEX -->
