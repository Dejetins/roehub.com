# Архитектурная документация

`docs/architecture/README.md` — каноническая точка входа для навигации по всем документам в `docs/**`.

## Ключевые документы

- [Indicators Architecture](docs/architecture/indicators/README.md)
- [Backtest v1 — Grid Builder + Staged Runner + Sync Guards (BKT-EPIC-04)](docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md)
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
- [STR-EPIC-02 — Strategy API v1 (updated): immutable CRUD + clone + run control + identity + standard errors](docs/architecture/api/api-errors-and-422-payload-v1.md) — `docs/architecture/api/api-errors-and-422-payload-v1.md` — Цель: пользователь через API может создать стратегию, клонировать как шаблон/из существующей, запускать/останавливать run’ы. Все правила владения/видимости — доменно-очевидны (use-case), identity подключается через порт `current_user`, ошибки унифицированы через `RoehubError` и общий 422 payload.
- [CLI — Backfill 1m Candles (Parquet -> ClickHouse) (Walking Skeleton v1)](docs/architecture/apps/cli/cli-backfill-1m.md) — `docs/architecture/apps/cli/cli-backfill-1m.md` — Этот документ фиксирует CLI entrypoint (composition root) для запуска walking skeleton v1:
- [Backtest v1 — Bounded Context + Domain + Use-Case Skeleton (Milestone 4 / BKT-EPIC-01)](docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md) — `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` — Документ фиксирует архитектуру BKT-EPIC-01: как вводим bounded context `backtest` (domain/application/ports), какие минимальные контракты (DTO/ошибки) и как он интегрируется с `market_data`/`indicators`/`strategy`, не фиксируя преждевременно детали исполнения/метрик.
- [Backtest v1 — Candle Timeline + Rollup + Warmup Policy (BKT-EPIC-02)](docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md) — `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md` — Фиксирует контракт BKT-EPIC-02: как backtest v1 строит свечной таймлайн на выбранном timeframe из canonical 1m, как работает best-effort rollup и warmup lookback.
- [Backtest v1 — Grid Builder + Staged Runner + Sync Guards (BKT-EPIC-04)](docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md) — `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md` — Фиксирует контракт BKT-EPIC-04: как детерминированно строится grid вариантов для sync backtest v1 (Stage A/Stage B), какие guards применяются, и почему API возвращает только top-K результатов.
- [Backtest v1 — Signals-from-indicators (v1) + AND aggregation (BKT-EPIC-03)](docs/architecture/backtest/backtest-signals-from-indicators-v1.md) — `docs/architecture/backtest/backtest-signals-from-indicators-v1.md` — Фиксирует контракт BKT-EPIC-03: как backtest v1 превращает значения индикаторов (primary output) и данные свечей в дискретные сигналы `LONG|SHORT|NEUTRAL`, и как эти сигналы агрегируются AND-политикой для стратегии.
- [Identity 2FA TOTP policy v1](docs/architecture/identity/identity-2fa-totp-policy-v1.md) — `docs/architecture/identity/identity-2fa-totp-policy-v1.md` — Документ фиксирует минимальный TOTP-флоу (setup/verify) и политику “exchange keys require 2FA” для Roehub Identity (milestone 3, ID-EPIC-02).
- [Identity exchange keys storage + 2FA gate policy v1](docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md) — `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md` — Документ фиксирует контракт ID-EPIC-03: безопасное хранение exchange API keys (только storage) с обязательным 2FA gate для всех операций CRUD v1.
- [Identity exchange keys storage + 2FA gate policy v2](docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md) — `docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md` — Документ фиксирует контракт ID-EPIC-03 для хранения exchange API keys (storage-only) c обязательным 2FA gate и шифрованием всех чувствительных полей.
- [Identity v1: Telegram-only login + user model + CurrentUser](docs/architecture/identity/identity-telegram-login-user-model-v1.md) — `docs/architecture/identity/identity-telegram-login-user-model-v1.md` — Документ фиксирует архитектуру identity-контекста: вход только через Telegram Login Widget (вариант A), выпуск JWT cookie и наличие `user_id` в контексте всех API запросов.
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
- [Общие принципы (на весь проект)](docs/architecture/roadmap/base_milestone_plan.md) — `docs/architecture/roadmap/base_milestone_plan.md` — - Доменные и прикладные части **не знают**, Binance это или Bybit.
- [milestone 2 epics v1](docs/architecture/roadmap/milestone-2-epics-v1.md) — `docs/architecture/roadmap/milestone-2-epics-v1.md`
- [Milestone 3 — EPIC map (v1)](docs/architecture/roadmap/milestone-3-epics-v1.md) — `docs/architecture/roadmap/milestone-3-epics-v1.md` — Карта EPIC’ов для Milestone 3: Telegram-only регистрация + 2FA-гейтинг ключей + Strategy v1 (immutable) + live runner + realtime streams + telegram notifications.
- [Milestone 4 — EPIC map (v1)](docs/architecture/roadmap/milestone-4-epics-v1.md) — `docs/architecture/roadmap/milestone-4-epics-v1.md` — Карта EPIC'ов для Milestone 4: Backtest v1 (close-fill) по одному инструменту с multi-variant grid (комбинации индикаторов + диапазоны параметров), direction modes, 4 режима position sizing, close-based SL/TP, комиссии/slippage и расширенный отчёт метрик.
- [Shared Kernel — Primitives (Trading)](docs/architecture/shared-kernel-primitives.md) — `docs/architecture/shared-kernel-primitives.md` — Этот документ фиксирует минимальные доменные примитивы Shared Kernel и правила, которые должны быть одинаковыми во всех bounded contexts.
- [Strategy API v1: immutable CRUD + clone + run control + identity integration (v1)](docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md) — `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md` — Фиксирует архитектурные решения для STR-EPIC-02: доменно-очевидные правила владения стратегией, неизменяемые стратегии, клонирование, управление запусками и интеграция с identity через порт `current_user`.
- [Strategy v1 — Immutable Spec + Storage + Runs/Events + Migrations Automation](docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md) — `docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md` — Документ фиксирует доменную модель Strategy v1 (immutable spec), хранение в Postgres (strategies/runs/events) и правила автоприменения миграций через Alembic.
- [Strategy — Live Runner via Redis Streams (v1)](docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md) — `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md` — Фиксирует контракт STR-EPIC-03: как один live-runner обслуживает множество пользователей/стратегий, читая 1m свечи из Redis Streams, делая rollup, warmup, checkpointing и repair(read).
- [Strategy — User Goal & Scope (Milestone 3 / EPIC 0)](docs/architecture/strategy/strategy-milestone-3-epics-v1.md) — `docs/architecture/strategy/strategy-milestone-3-epics-v1.md` — Документ фиксирует целевой пользовательский сценарий и границы `Strategy v1` (Milestone 3) перед реализацией EPIC’ов 1–13.
- [Strategy realtime output via Redis Streams v1](docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md) — `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md` — Архитектурный контракт v1 для публикации realtime метрик и событий стратегии в Redis Streams (per-user), чтобы UI мог подписаться и видеть состояние run’ов.
- [Strategy runtime config v1: `configs/*/strategy.yaml` + toggles + metrics port + env overrides](docs/architecture/strategy/strategy-runtime-config-v1.md) — `docs/architecture/strategy/strategy-runtime-config-v1.md` — Документ фиксирует STR-EPIC-06: единый runtime-конфиг для Strategy (API + live worker) с fail-fast валидацией, enable-тумблерами, метриками и поддержкой env overrides.
- [Strategy Telegram notifier v1: best-effort adapter + notification policy](docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md) — `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md` — Документ фиксирует контракт STR-EPIC-05: как Strategy live-runner отправляет Telegram-уведомления по ключевым событиям без влияния на устойчивость pipeline.

### Ранбуки

- [help commands](docs/runbooks/help_commands.md) — `docs/runbooks/help_commands.md`
- [Runbook — Numba cache dir + threads](docs/runbooks/indicators-numba-cache-and-threads.md) — `docs/runbooks/indicators-numba-cache-and-threads.md` — Документ описывает, как управлять `numba_cache_dir` и количеством потоков (`NUMBA_NUM_THREADS`) для `indicators` compute.
- [Runbook — Numba warmup / JIT](docs/runbooks/indicators-numba-warmup-jit.md) — `docs/runbooks/indicators-numba-warmup-jit.md` — Документ помогает диагностировать задержки первого расчёта индикаторов и отличать ожидаемый JIT warmup от неисправности.
- [Runbook — Troubleshooting: why NaN?](docs/runbooks/indicators-why-nan.md) — `docs/runbooks/indicators-why-nan.md` — Документ объясняет, откуда берутся `NaN` в `indicators` и как отличить корректную политику от дефекта.
- [Market Data Docker Runbook](docs/runbooks/market-data-autonomous-docker.md) — `docs/runbooks/market-data-autonomous-docker.md` — Runbook для `market-data-ws-worker` и `market-data-scheduler`.
- [Market Data Metrics Reference (RU)](docs/runbooks/market-data-metrics-reference-ru.md) — `docs/runbooks/market-data-metrics-reference-ru.md` — Подробный справочник метрик для:
- [Market Data Metrics](docs/runbooks/market-data-metrics.md) — `docs/runbooks/market-data-metrics.md` — Документ фиксирует основные Prometheus-метрики для:
- [Runbook — Market Data Redis Streams](docs/runbooks/market-data-redis-streams.md) — `docs/runbooks/market-data-redis-streams.md` — Операционные команды для live feed stream’ов, которые публикует `market-data-ws-worker`.
- [Strategy live worker runbook](docs/runbooks/strategy-live-worker.md) — `docs/runbooks/strategy-live-worker.md` — Runbook для `apps/worker/strategy_live_runner`: как поднять worker, проверить метрики, проверить Redis Streams realtime output и (опционально) Telegram notify.

### API

- (пока нет документов)

### Решения

- (пока нет документов)

### Прочее

- [Шаблон архитектурного документа](docs/_templates/architecture-doc-template.md) — `docs/_templates/architecture-doc-template.md` — Используйте этот шаблон для новых документов в `docs/architecture/**`; сразу после `# H1` оставляйте краткое описание в одной строке.
- [repository three](docs/repository_three.md) — `docs/repository_three.md`
<!-- END GENERATED DOCS INDEX -->
