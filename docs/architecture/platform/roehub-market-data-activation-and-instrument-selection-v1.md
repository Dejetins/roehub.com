# Roehub — активация рыночных данных и выбор инструментов v1

Этот план устраняет разрыв между автономной установкой и контролируемым получением публичных рыночных данных, а также заменяет файловый whitelist выбором инструментов пользователем.

## Цель и границы

После выполнения владелец организации в onboarding и настройках видит полный каталог доступных на поддерживаемых биржах инструментов, выбирает нужные, меняет свой выбор без остановки работающей стратегии и видит для каждого инструмента размер артефактов и полноту данных. Сбор данных начинается с `binance:futures:BTCUSDT`, выполняется последовательно и не должен вызывать неконтролируемое потребление памяти или диска.

Внутренняя сеть остаётся `internal: true` для базы данных, Redis, OpenBao, API и Web. Лишь `market-data-scheduler` и `market-data-ws` получают вторую сеть `market-data-egress` для исходящих запросов к публичному рынку. Отдельная `web-ingress` подключается только к `web`, чтобы Docker Desktop мог безопасно опубликовать локальный HTTP-порт без ручного подключения к `bridge`; она не используется для биржевых клиентов и не получает секретов провайдеров. Сеть Docker ограничивает круг контейнеров, но не умеет надёжно ограничивать домены; в v1 доступ к биржам контролируется allowlist-адаптерами, тайм-аутами, лимитами и наблюдаемостью. FQDN-enforcement прокси — отдельное последующее усиление, не создаваемое неявно.

Не входят: реальные торговые операции, автоматическое включение всех инструментов, передача секретов, ослабление recovery-контракта OpenBao, production deployment и доказательство `linux/amd64`. Эти границы остаются за исходным [планом self-hosted platform](/Users/daniildegtyarev/Projects/roehub.com/docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md).

## Проверенные исходные факты

- `market-data-ws` один раз читает список включённых инструментов при запуске; scheduler наполняет список позднее, поэтому worker остаётся без подписок.
- Оба worker подключены только к `roehub` с `internal: true`; поэтому DNS и публичные REST/WebSocket обращения недоступны.
- whitelist — глобальный файл, а не пользовательская настройка; текущий `ref_instruments` не хранит принадлежность организации, выбор и покрытие.
- publisher имеет контейнерный лимит `1 GiB`, но конфигурация допускает четыре процесса и бюджет `2 GiB`, что неприемлемо для первого реального построения.
- OpenBao интерактивной установки намеренно `initialized=false`, `sealed=true`. Полная безопасная инициализация требует трёх внешних PGP-получателей долей, порога `2-of-3` и отдельного хранения recovery identity.

## Целевая модель

```text
Организация ──выбор──> InstrumentSelection ──┐
Активная стратегия ──pin──> EffectiveSelection ├──> scheduler / WS / REST
Каталог биржи ────────────────────────────────┘          │
                                                         v
                                              ClickHouse canonical candles
                                                         │
                                                         v
                                       coverage + artifact inventory ──> API/Web
```

`InstrumentSelection` принадлежит организации и является желанием пользователя. `OrganizationEffectiveSelection` — объединение её выбора и временных pin её активных стратегий; удаление выбора не останавливает существующую стратегию и не удаляет её данные. После завершения стратегии pin исчезает при следующей сверке. Удаление не имеет внешнего побочного эффекта, идемпотентно и не должно быть заблокировано состоянием стратегии.

`GlobalEffectiveCollectorSet` — объединение `OrganizationEffectiveSelection` всех организаций. Только worker использует этот технический набор для общего публичного ingestion. API никогда не раскрывает другой организации её selections, pins или источник включения инструмента; он возвращает только состояние текущей organisation и публичные агрегаты покрытия/артефактов.

Каталог остаётся глобальной справочной информацией биржи. Отдельный bounded metadata-only refresh последовательно читает поддерживаемые exchange metadata endpoints, upsert-ит catalog и сохраняет `refreshed_at`/последнюю ошибку; он никогда не создаёт selection, subscription или historical backfill. API возвращает поддерживаемые рынки с явными состояниями `fresh`, `stale` или `failed`; при stale/failed пользователь может видеть ранее подтверждённый каталог, но не получает ложного обещания его актуальности. Покрытие для 1m-данных рассчитывается как `distinct closed canonical candles / expected closed minutes` в определённом источником допустимом интервале; отсутствие начала истории выражается как `unknown`, а не как ложные `100%`. Размер артефакта — суммарный размер текущего опубликованного slot, `0` при отсутствии, а не прогноз диска.

## Контрактное влияние

| Поверхность | Изменение | Класс | Совместимость и откат |
|---|---|---|---|
| API / DTO | Добавляются каталог, selections, effective state, coverage и artifact inventory | `compatible-change` | Новые versioned endpoints/поля; старый readiness остаётся до migration docs |
| Хранение | PostgreSQL selections и migrations; ClickHouse read-model для coverage | `breaking-change` | Только greenfield/новая migration; downgrade сохраняет rows, но не использует их |
| Конфигурация | whitelist перестаёт быть источником runtime policy; появляется bootstrap default `BTCUSDT` | `breaking-change` | Явная migration/versioned generated config; без silent fallback к CSV |
| Вызовы сервисов | worker читают effective selections, scheduler/WS получают egress | `breaking-change` | fail-closed при пустом/устаревшем каталоге; retries bounded, error metrics |
| Браузер | onboarding и settings показывают выбор/покрытие/размер/pin | `compatible-change` | Empty/loading/error states и accessibility coverage обязательны |
| OpenBao | owner-init и per-service credential delivery | `breaking-change` | Uninitialized/sealed остаётся safe degraded; rollback не раскрывает root/unseal material |

## Этапы

| Этап | Результат | Предшественник | Реальная граница |
|---|---|---|---|
| `00` | Зафиксированы контракты, migration path и baseline | `N/A` | статический audit + focused tests |
| `01` | Egress только для двух workers, динамические subscriptions и readiness | `00` | Docker runtime с публичным `BTCUSDT` |
| `02` | Организационный выбор, каталог, coverage и UI onboarding/settings | `01` | PostgreSQL/API/Browser |
| `03` | Один ручной publish, согласованная память и controlled expansion policy | `02` | Docker/ClickHouse/artifact runtime |
| `04` | Owner-operated OpenBao init flow, network/credential hardening и custody handoff | `03` | disposable runtime; production-like durable custody остаётся owner action |
| `05` | Новый локальный release candidate, lifecycle и original-plan handoff | `04` | Docker Desktop/browser; `linux/amd64` остаётся отдельным blocker |

## Общие правила

- Источники исполнения: этот plan doc, `.codex/agents/generated/roehub-market-data-activation-and-instrument-selection-v1/` и связанный stage ledger.
- Использовать только ветку `codex/market-data-egress-instrument-onboarding`; чужие изменения не включать.
- Не писать secrets, PGP fingerprints, unseal shares, AppRole SecretID, tokens или provider responses в Git, evidence, логи или чат.
- Память: последовательные runtime операции, `COMPOSE_PARALLEL_LIMIT=1`; publisher ограничить одним worker и бюджетом не выше фактического container limit с запасом. Не запускать полный каталог или все исторические backfill параллельно.
- Publisher читает только `GlobalEffectiveCollectorSet`; он не имеет legacy default для всех enabled reference rows и остаётся выключенным по расписанию до отдельного capacity decision после успешного ручного `BTCUSDT` publish.
- Stage `04` принимается после подтверждённого disposable 3-recipient, `2-of-3` owner-init path, per-service credential isolation и документированного owner-custody handoff. Реальная durable initialization локальной/удалённой установки — последующее действие владельца, а не ложный blocker локального candidate lifecycle.
- После изменения release-config/OCI пересобрать кандидат Stage `22`, повторить нужные части `23` и локальный `24`; исходный `24` остаётся `blocked` до native `linux/amd64`.
- Удалённая синхронизация означает scoped commit/push только этой ветки после доказательств; production/Mac Studio не трогать.

## Критерий завершения

Этапы `00`–`05` приняты или содержат явный owner-custody handoff для OpenBao; локальный Docker Desktop демонстрирует рабочий `BTCUSDT` path и UI. Финальный production OpenBao bootstrap и native `linux/amd64` остаются отдельными строго контролируемыми действиями.
