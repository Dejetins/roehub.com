# Этап 10 — организационная изоляция торгового контура

## Статус

- Этап: `10`.
- Статус: `accepted`.
- Дата: `2026-07-13`.
- Режим: `goal_driven`.
- Граница доказательств: `N/A`; одноразовый PostgreSQL и контролируемые
  адаптеры с искусственными организациями, пользователями, подключениями,
  стратегиями и заявками.
- Production-базы, пользователи, секреты, конфигурация и артефакты не
  читались и не изменялись.
- Следующий разрешённый этап: `11`.

## Результат

Организационный контекст проходит через стратегию, запуск, сигнал, источник
исполнения, намерение, риск, биржевое подключение, заявку, исполнение, позицию,
paper accounting, RL-политику риска и RL-лимит активных тикеров. API и UI
получают `organization_id` только из серверного контекста текущего
пользователя.

Публичный `POST /ui/execution/intents` больше не принимает `risk_context`.
Модель запроса запрещает лишние поля, а маршрут вызывает
`ExecutionRiskContextResolver`. PostgreSQL-реализация подтверждает источник,
организацию, владельца, подключение, активную версию credential, среду и
соответствие инструмента. Не подключённые пока account-state, kill-switch и
лимиты остаются `False`, поэтому серверная граница отказывает по умолчанию и
клиент не может превратить собственные утверждения в принятую заявку.

Standalone RL CLI и ручной `testnet` также не подставляют положительные
признаки риска. Без доверенных сервисов состояния они сохраняют источник и
получают детерминированный отказ до dispatch. Paper-пути остаются отдельными и
заканчиваются `paper_no_exchange_submit` без биржевой отправки.

Execution source и intent используют namespace
`io.roehub.execution-idempotency/v1`, включающий `organization_id` и account
namespace. `client_order_id` имеет префикс `rh1_` и строится из уже
organization/account-scoped hash. При неизвестном результате существующая
заявка сначала сверяется по `client_order_id`; повторная отправка до сверки
запрещена.

## Владение и полномочия

| Ресурс | Класс владения | Источник полномочий и побочный эффект |
|---|---|---|
| Стратегии, запуски, profiles, bindings и signals | `organization-owned`, пользователь внутри организации | Server-derived scope и составные внешние ключи связывают одну organization/user/strategy/run цепочку. |
| Compatibility readiness, market requirements и scenario matrix | `organization-owned` | Запись и чтение требуют organization/owner/source-job/strategy одной организации. |
| Биржевые подключения, credential versions и account projections | `organization-owned`, account=`exchange_connection_id` | Credential и snapshots ссылаются на connection той же организации; расшифрованное значение существует только в доверенной exchange-control/execution границе. |
| RL risk policies, audit, ticker overrides и activations | `organization-owned` | Policy, strategy, live profile, quota и active ticker используют organization/owner scope; один пользователь в разных организациях не делит торговую квоту. |
| Source events, intents, risk audit и notification outbox | `organization-owned` | Сервер разрешает риск; org/account namespace участвует в idempotency; source/signal/intent/connection связи защищены составными внешними ключами. |
| Orders, events, fills, funding и reconciliation | `organization-owned` | Каждая запись хранит `organization_id` и ссылается на order/intent той же организации. |
| Private-stream sessions | `organization-owned` | Явный `organization_id`, составной внешний ключ на connection и org-scoped uniqueness. |
| Request observations | `organization-owned` для распознанного intent; `installation-operational` для неразрешимого входа | У распознанного intent организация ограничена составным внешним ключом; у malformed/quarantined сообщения она может быть `NULL`. |
| Process heartbeat | `installation-shared` | Готовность одного service instance; пользовательского торгового ресурса нет. |
| Paper orders/fills/accounting, scenario coverage и position ownership | `organization-owned` | Stable identity включает organization namespace; strategy/run/source/connection связи остаются same-org. |

## Миграция и отсутствие переноса данных

`0015_trading_organization_isolation_v1.sql` является greenfield-миграцией.
Перед изменением она проверяет пустоту перечисленных в guard таблиц, которые
получают новый обязательный organization scope: strategy runtime state,
compatibility/scenario/paper coverage, RL risk/ticker state, account snapshots,
capital/paper state и execution ledger. При любой строке выполнение
останавливается. В миграции нет `UPDATE`, backfill, импорта legacy owner или
dual-read.

Новые ограничения включают:

- `execution_intents_org_connection_fk` для same-org account identity;
- `execution_source_events_org_signal_fk`, `execution_intents_org_signal_fk` и
  `execution_notification_org_signal_fk` для strategy signal lineage;
- составные связи RL risk policy/audit и ticker activation с membership,
  strategy и live profile;
- составные связи paper coverage с scenario, job, strategy, profile, run,
  signal, source, intent, order, fill и accounting;
- organization-scoped uniqueness и read indexes.

Текущие checksums:

- файл `0015_trading_organization_isolation_v1.sql`:
  `e04daad19bbb500e7cdb17ef59570b6ee05896f98f8020e25c841a91f6ab7053`;
- фаза `trading-tenancy-0015`:
  `f5af81ccb02877ea51da45dc189518c2bc118d817174570eb84db606db93277d`.

## Реальная граница проверки

`uv run python -m apps.migrations.verify_storage_runtime` подняла чистые
PostgreSQL `16.14`, ClickHouse `24.8.14.39` и Redis `7.2.14` через Docker CLI
`29.6.1` и Engine `29.5.2`. Прошли fresh bootstrap, intentional
interruption/recovery, idempotent rerun, persisted-volume restart, external
readiness и cleanup.

Trading probe через production repositories и реальные ограничения новой БД
подтвердил:

1. независимый paper flow двух организаций;
2. пустой cross-organization repository read;
3. отказ cross-owner strategy write по membership foreign key;
4. отказ связи intent с чужим connection по
   `execution_intents_org_connection_fk`;
5. отказ server-side resolver для connection другой организации;
6. реальный HTTP-запрос с клиентским `risk_context` получает `422`, а число
   intents не меняется;
7. серверный fail-closed risk denial имеет причину `kill_switch_closed`;
8. одинаковый intent внутри organization/account namespace дедуплицируется;
9. private-stream session и request observation реально сохраняются;
10. неизвестный статус сверяется без повторного submit;
11. контролируемая попытка с connection environment=`mainnet` проходит через
    execution process и останавливается причиной `mainnet_hard_block` до
    адаптера; `mainnet_submits=0`.

Approved host-local testnet credentials не предоставлялись. Внешний
provider smoke не запускался и не требуется prompt-ом при их отсутствии.
Контролируемый testnet adapter вызван один раз только внутри одноразового
контейнера; сеть провайдера, production credentials и реальные ордера не
использовались.

## Проверки качества

- `uv run python -m apps.migrations.verify_storage_runtime` — `passed`;
  чистая миграция, HTTP/БД/исполнение и cleanup подтверждены.
- Полный `uv run pytest -q` — `1795 passed`, `4` существующих предупреждения
  `httpx` о будущем изменении per-request cookies.
- `uv run ruff check .` — `passed`.
- Целевой `pyright` по API, worker и изменённым trading contexts —
  `0 errors, 0 warnings`.
- Runtime input inventory `--check` — `passed`, `139` имён без значений.
- Docs index и project map generation/`--check` — `passed`.
- `git diff --check` — `passed` после документальной финализации.

## Независимая проверка и исправления

- Режим: единственная проверка `independent subagent`.
- Первоначальный вердикт: `Block`.
- Критические замечания: клиентский `risk_context`, отсутствие same-org связи
  intent→connection и неполная signal lineage.
- Существенные замечания: фиктивные положительные risk facts в testnet CLI и
  ручном маршруте; доказательство account mismatch и `mainnet_submits=0` было
  декларативным; не проверялись persisted private session/observation.
- Исправлено: риск разрешается сервером и отказывает по умолчанию; добавлены
  ограничения БД; RL policy/ticker state включён в organization scope; рабочие
  testnet producers больше не создают положительную risk authority; Docker
  probe выполняет HTTP-подмену, прямой внешний ключ, private session,
  observation и реальный service-level mainnet guard.
- Локальная холодная перепроверка после исправлений: `Release after fixes`.
- Повторная независимая проверка не запускалась, чтобы сохранить требование
  ровно одной независимой проверки.

Остаточный риск: production resolver намеренно не принимает внешнее
исполнение, пока trusted account freshness, kill-switch, reservation и limit
services не подключены. Это функциональное ограничение с безопасным отказом,
а не обход. Реальный testnet provider boundary остаётся непроверенным без
разрешённых host-local credentials.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| Public execution API/DTO | `breaking-change` | `risk_context` удалён из запроса; серверный resolver обязателен, лишнее поле даёт `422`. |
| Strategy/API/UI | `breaking-change` | Organization scope обязателен и выводится сервером; старые signatures без него несовместимы. |
| Domain entities и ports | `breaking-change` | Strategy, execution, account, paper, RL policy/quota и position contracts требуют `organization_id`. |
| Persistence | `breaking-change` | Greenfield-only NOT NULL ownership, составные внешние ключи, uniqueness и индексы; legacy state не переносится. |
| Config/defaults | `breaking-change` | Testnet CLI/ручной путь без trusted risk authority теперь отказывают; execution adapters допускают только `disabled/testnet`. |
| Idempotency/client-order identity | `breaking-change` | Новый versioned organization/account namespace применяется с первой v1 записи, alias отсутствует. |
| Secret boundary | `compatible-change` | Секреты остаются в exchange-control; resolution дополнительно проверяет organization/user ownership. |
| Compute/trading formulas | `none` | Численные kernels и decision semantics не менялись. |
| Внешние production effects | `none` | Только disposable containers и artificial rows; provider network, mainnet, deploy и production mutation отсутствовали. |

Основная классификация этапа — `breaking-change`, допустимая для greenfield v1
без backfill, alias и dual-read.

## Файлы этапа

Созданы migration `0015`, PostgreSQL risk resolver, organization-scoped
runtime probe, focused tests и этот отчёт. Изменены strategy/live-execution/
exchange-control/RL domain, ports, use cases и repositories; API/worker/CLI/
wiring; storage manifest/verifier; platform plan, ledger и generated
architecture indexes.

Удалённых tracked-файлов нет. `.codex/PLANS.md`, принятые артефакты прошлых
этапов и остальные чужие dirty changes сохранены. Staging, commit, push,
deploy, production data read и production mutation не выполнялись.

## Передача этапу 11

Этап `11` получает server-derived organization scope, organization-owned
account/intent/order identity, versioned idempotency, fail-closed risk resolver,
unknown-state reconciliation и жёсткий запрет mainnet. Notification provider
не должен принимать exchange credentials, client risk authority или ослаблять
same-org ownership.
