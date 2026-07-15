# Этап 11 — экземпляры провайдеров уведомлений и Telegram

## Статус

- Этап: `11`.
- Статус: `accepted`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; одноразовые PostgreSQL, OpenBao и
  контролируемый Telegram-compatible HTTP stub с искусственными организациями,
  экземплярами, credential references, маршрутами и доставками.
- Broad real Telegram send, production recipients, production credentials,
  текущие базы и production configuration не читались и не изменялись.
- Следующий разрешённый этап: `12`.

## Результат

Создан стабильный `NotificationProvider/v1`. Пакет провайдера содержит
descriptor, SemVer, JSON Schema конфигурации, каналы, шаблоны и ограниченный
набор error codes. Пакет и экземпляр являются разными ресурсами: пакет
устанавливается отдельно, а экземпляр создаётся для всей установки или одной
организации. PostgreSQL связывает `package_id` и `provider_key` составным
внешним ключом.

Экземпляр содержит только безопасную product configuration и typed OpenBao
reference. Use case загружает установленный descriptor и применяет его JSON
Schema с format checking до записи экземпляра. Domain и база запрещают raw
secret-shaped keys, credentials в URL, невалидный scope и Telegram instance без
secret reference. Credential reference канонически включает organization либо
`installation` и конкретный `provider_instance_id`; ошибочная ссылка на другой
scope отклоняется также PostgreSQL CHECK. Built-in пакеты
`log_only`, controlled `fake` и `telegram_bot_api` используют тот же контракт;
custom HTTP client отправляет versioned payload
`notifications.roehub.io/v1` и заголовок `X-Roehub-Delivery-Id`.

Notification event, route, delivery, attempt, Telegram update и report-run
стали organization-scoped. Route и delivery дополнительно содержат
`provider_instance_id`; составные ограничения и repository checks запрещают
cross-organization и cross-instance связи. API/UI выводит организацию из
server-side current-user context и закрывает операцию при нуле или нескольких
подходящих Telegram instances.

## Telegram provider и секреты

`TelegramBotProvider/v1` разрешает credential перед каждым provider call через
OpenBao. Recipient binding хранит в PostgreSQL только masked reference и typed
OpenBao reference; значение адреса находится в OpenBao. Dispatcher использует
read-only identity, а отдельный Telegram worker имеет ровно дополнительное
право записи recipient secret. Реальный OpenBao proof подтвердил восемь
service identities, отсутствие shared broad token и запрет записи dispatcher-у.

Installation-wide bot определяет организацию подтверждённой binding либо по
хешу ещё не использованного `/start` binding code. Organization bot жёстко
фиксирует свою организацию. Неоднозначный scope закрывает обработку. Raw chat
ID существует только как `SecretValue` внутри входной доверенной границы и не
попадает в repr, PostgreSQL, CLI output, метрики, отчёт или журнал.

В PostgreSQL добавлены durable update cursor, command registry, одноразовые
binding codes и recipient bindings. Update, response route и response delivery
фиксируются одним data-modifying CTE в одной PostgreSQL-транзакции. Binding code
consume и recipient binding также атомарны; внешний recipient secret получает
уникальный binding-specific path, поэтому проигравшая конкурентная попытка не
может перезаписать секрет победителя. Cursor продвигается compare-and-set только
после durable handling; повторный `update_id` восстанавливает результат без
нового действия.
`roehubctl telegram connect` создаёт одноразовый код только в новом файле mode
`0600` и выводит лишь безопасные identifiers, срок и статус.

## Доставка, повторы и деградация

Provider call принадлежит `NotificationProvider/v1`. По умолчанию бюджет
соединения равен `3` секундам, общий бюджет — `10` секундам. `delivery_id`
является provider idempotency identity. Повтор разрешён только для
классифицированной ошибки до возможного принятия с capped exponential backoff,
deterministic bounded jitter и учётом `Retry-After`.

Timeout после возможного принятия становится `unknown`. Истёкшая lease
`claimed` после shutdown также восстанавливается как durable `unknown` с
`provider_shutdown`, а не отправляется повторно. `unknown` нельзя автоматически
вернуть в `pending/retry`; явный replay сохраняет исходную доставку как
доказательство и создаёт новую identity с
`replayed_from_delivery_id`. Cancellation активного provider call сначала
фиксируется как durable `unknown/provider_cancelled`, после чего сигнал
отмены передаётся worker lifecycle.

Health probe изменяет состояние только затронутого provider instance.
Prometheus публикует redacted counters `sent/retry/unknown/dead_letter`, latency,
pending/unknown gauges и health по `provider_instance`. Alert
`NotificationsProviderInstanceDegraded` связан с operator runbook. Alert
`NotificationsCriticalUnknownDelivery` охватывает критические и торговые
категории. Organization bot никогда автоматически не заменяется installation
bot для критического или торгового сообщения.

Прямой Strategy mode, читавший глобальный `TELEGRAM_BOT_TOKEN`, исключён из
runtime composition и свежих конфигов. Strategy создаёт organization-scoped
notification через Notifications. Runtime inventory содержит только отдельные
OpenBao token-file references dispatcher и Telegram worker; raw Telegram token
keys отсутствуют.

## Миграция и отсутствие переноса данных

`0016_notification_provider_instances_v1.sql` является greenfield-миграцией.
Guard прекращает работу при существующих notification events, routes,
deliveries, attempts, Telegram updates или report runs. Миграция не выполняет
backfill, импорт legacy token/recipient, repair текущих rows, dual-read или
alias.

Миграция создаёт пакеты и экземпляры, cursor, command registry, binding codes и
recipient bindings; добавляет organization/provider-instance scope в
существующие таблицы; связывает route→instance, delivery→route/instance,
attempt/update→instance и user-owned rows→membership. Scope trigger проверяет
installation/org ownership и одинаковый `provider_key`.

Текущие checksums:

- файл `0016_notification_provider_instances_v1.sql`:
  `9908a7ac66aa175f4448ecb17fcd9c8911637ed8ed8b0f4f5bb74983d3f4e70c`;
- фаза `notification-providers-0016`:
  `d4080671d481e25b7a1c6266249c2fddb4cc59255d361ce35a8a9b7ba6173d1a`.

## Реальная граница проверки

Это `real-boundary evidence`, а не tests-only acceptance: проверка использует
реальные контейнеры, production repositories и ограничения чистой PostgreSQL,
реальный OpenBao API и сетевой HTTP stub с контролируемым поведением.

`uv run python -m apps.migrations.verify_storage_runtime` на Docker CLI
`29.6.1`, Engine `29.5.2` и Compose `5.3.1` подняла чистые PostgreSQL `16.14`,
ClickHouse `24.8.14.39` и Redis `7.2.14`. Прошли fresh bootstrap,
interruption/recovery, idempotent rerun, persistent-volume restart, external
readiness и cleanup.

Notification probe через production repositories, реальные ограничения новой
PostgreSQL и контролируемый HTTP stub доказал:

1. две организации используют два независимых provider instances;
2. одинаковый `delivery_id` идемпотентен;
3. production `TelegramProviderWorker` восстанавливает уже атомарно сохранённый
   `update_id`, обрабатывает следующий update и продвигает durable cursor;
4. secret references изолированы по организациям и экземплярам, а неверная
   cross-scope credential reference отклоняется базой;
5. отказ до возможного принятия получает `retry`;
6. timeout после возможного принятия получает `unknown`;
7. `Retry-After`, bounded backoff и jitter сохраняются;
8. cancellation активного provider call сохраняет `unknown` до передачи
   `CancelledError`, а shutdown recovery даёт `unknown` без повторной отправки;
9. explicit replay создаёт новую pending delivery с durable lineage и не
   изменяет исходный `unknown`;
10. health переходит между `ready` и `degraded` только для экземпляра;
11. cross-organization write отклоняется базой;
12. critical fallback не используется;
13. command registry содержит `18` записей.

`uv run python infra/openbao/verify_runtime.py` повторно прошла на реальном
OpenBao container: `service_identities=8`, `shared_broad_tokens=false`,
forbidden-output scan и cleanup завершились, dispatcher не может записывать
recipient secrets.

Реальная отправка в Telegram не запускалась: prompt делает её необязательной и
требует отдельного разрешения на конкретного test recipient. Отсутствие такого
canary не ограничивает доказанную локальную границу Stage `11`.

## Проверки качества

- Финальная focused regression suite — `26 passed`; дополнительный CLI/infra
  asset suite — `28 passed`.
- Полный `uv run pytest -q` — `1809 passed`, `4` существующих предупреждения
  `httpx` о будущем изменении per-request cookies.
- Целевой `pyright` — `0 errors, 0 warnings`.
- Полный `uv run ruff check .` — `passed`.
- Полный `uv run pyright` дополнительно запускался, но не является gate этого
  этапа: `153` ошибки и `2` предупреждения находятся в сохранённых чужих
  `local_artifacts` и exchange cleanup tools; Stage `11` scope чист.
- Runtime input inventory `--check` — `passed`, `140` имён без значений;
  глобальные raw Telegram token keys отсутствуют.
- Docker storage/provider runtime proof — `passed` с cleanup.
- Docker/OpenBao runtime/recovery proof — `passed` с cleanup.
- CLI help для `providers install`, `providers add` и `telegram connect` —
  `passed`.
- Docs index и project map generation/`--check` — `passed`.
- `git diff --check` выполняется после независимой проверки и финализации
  отчёта/журнала.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| `NotificationProvider/v1` descriptor/port | `compatible-change` | Добавлен стабильный расширяемый контракт и встроенные реализации; внешнего v1 контракта раньше не было. |
| Notification domain/use cases | `breaking-change` | Event, route, delivery, attempt, update и report требуют organization/provider-instance scope. |
| Persistence | `breaking-change` | Clean-only `0016`, новые NOT NULL/composite FK/uniqueness/triggers и отдельные package/instance resources; backfill отсутствует. |
| Config/defaults | `breaking-change` | Прямой runtime mode с глобальным Telegram token запрещён; свежая конфигурация использует provider instances и OpenBao references. |
| Secret boundary | `breaking-change` | Bot credential и recipient перенесены из raw env/DB chat binding в typed OpenBao refs и отдельные service identities. |
| Idempotency/retry identity | `breaking-change` | `delivery_id` и `provider_instance_id` становятся обязательной versioned delivery identity; `unknown` не переотправляется автоматически. |
| API/CLI | `compatible-change` | Добавлены provider install/add и Telegram connect operations; API перестал неявно выбирать Telegram provider при неоднозначности. |
| Межсервисные вызовы | `breaking-change` | Strategy отправляет notification через organization-scoped Notifications вместо прямого Telegram adapter. |
| Метрики/алерты/runbook | `compatible-change` | Добавлены instance labels, health и critical/trading unknown alert без секретных значений. |
| Compute/trading formulas | `none` | Торговые вычисления и решения не менялись. |
| Browser-visible defaults | `compatible-change` | UI использует server-derived organization/provider resolution и fail-closed ambiguity; новый экран не добавлялся. |
| Внешние production effects | `none` | Только disposable containers, artificial rows и локальный stub; реальный Telegram, deploy и production mutation отсутствовали. |

Основная классификация Stage `11` — `breaking-change`, допустимая для
greenfield v1 без backfill, dual-read и legacy alias.

## Файлы этапа

Созданы provider domain/ports/use cases, PostgreSQL repositories, Telegram и
custom HTTP adapters, `0016`, notification runtime probe, CLI commands,
отдельная OpenBao policy Telegram worker, focused tests и этот отчёт. Изменены
notification/strategy/API/worker composition, storage manifest/verifier,
OpenBao identities/verifier, runtime inventory, Prometheus alert, runbook и
архитектурный план.

Удалённых tracked-файлов нет. `.codex/PLANS.md`, supersession docs,
`local_artifacts`, exchange cleanup tools и остальные unrelated dirty changes
сохранены. Staging, commit, push, deploy, production data read и production
mutation не выполнялись.

## Независимая проверка

Единственная проверка `independent subagent` завершилась исходным вердиктом
`Block` и обнаружила пять обязательных замечаний: раздельную фиксацию Telegram
update/route/delivery, отсутствие исполнения package JSON Schema, устаревшие
checksums, отсутствие cancellation proof и недостаточную привязку credential
reference к scope/instance. Все пять исправлены. Дополнительно runtime proof
переведён с ручной записи update/cursor на production `TelegramProviderWorker`
и добавлен explicit replay proof. Второй независимый review не запускался;
актуальный snapshot прошёл локальную холодную перепроверку с итогом
`Release after fixes`. Дополнительно исправлена route identity для перехода
от unbound admin response к bound user response; регрессионный тест и повторные
full/runtime gates прошли.

## Передача этапу 12

Stage `11` принят. Stage `12` получает
package/instance separation, typed secret refs, scoped delivery identity,
durable Telegram state, bounded provider call semantics и no-critical-fallback
как первый concrete plugin-style contract.
