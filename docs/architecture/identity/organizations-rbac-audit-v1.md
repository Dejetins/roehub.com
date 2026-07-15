# Организации, RBAC и административный аудит v1

## Цель и пользовательский результат

Новая самостоятельно разворачиваемая установка Roehub хранит пользователей,
организации, членство и полномочия в своём PostgreSQL. Один пользователь может
состоять в нескольких организациях, но каждый запрос получает область доступа
из серверного хранилища. Переданный клиентом `organization_id` является только
указателем на ресурс и никогда сам по себе не даёт доступа.

Первый локальный пользователь, созданный одноразовой bootstrap-процедурой,
один раз инициализирует пустую установку и становится `installation_owner` и
владельцем первой организации. Stage `06` добавит эту локальную процедуру и
полную аутентификацию поверх принятого здесь application boundary. Импорт
текущих пользователей, организаций, `paid_level` или владения ресурсами не
выполняется.

`installation_owner` не получает неявную роль `owner` во всех организациях:
для обычных организационных операций ему также нужно активное членство. Это
сохраняет серверную область организации и не превращает инстанционное
восстановление в скрытое повседневное повышение полномочий.

## Границы этапа

Stage `05` включает:

- singleton-установку, `installation_owner`, организации и членство;
- роли `owner`, `admin`, `operator`, `trader`, `viewer`;
- приглашения, права на экземпляры плагинов и временный support-доступ;
- versioned API `/api/v1/installations/*` и `/api/v1/organizations/*`;
- неизменяемый административный аудит без чувствительных payload;
- организационные внешние ключи для provenance, exchange connections,
  strategies и positions.

Не входят browser login/recovery и точный `recent-auth` reauthentication flow
(Stage `06`), OIDC linking (Stage `07`), OpenBao (Stage `08`) и перевод всех
research/trading use cases на новый `organization_id` (Stages `09`,`10`). До
этого старые resource-write adapters могут завершаться fail-closed на новом
обязательном столбце вместо создания ресурса без организации.

## Текущее состояние и решение

До Stage `05` `identity_users.user_id` был глобальной identity, а большинство
ресурсов принадлежали `user_id` или `owner_user_id`. Такая форма не умеет
однозначно представить пользователя в двух организациях и допускает
семантические ссылки между ресурсами разных владельцев.

Выбрана модель modular monolith с одним identity aggregate и PostgreSQL как
источником истины. Identity domain/application владеют ролевой матрицей и
решениями доступа; inbound FastAPI adapter переводит versioned DTO; PostgreSQL
adapter атомарно сохраняет членство и аудит. Другие bounded contexts используют
общий `OrganizationId`, но не импортируют ORM/SQL identity.

Альтернатива «организация определяется единственным владельцем-пользователем»
отклонена: пользователь может состоять в нескольких организациях. Отдельный
authorization microservice также отклонён: он добавил бы сетевую согласованность
к транзакционным инвариантам без самостоятельной потребности в масштабировании.

## Модель полномочий

| Возможность | `owner` | `admin` | `operator` | `trader` | `viewer` |
|---|---:|---:|---:|---:|---:|
| Просмотр организации/участников/плагинов | да | да | да | да | да |
| Изменение организации | да | да | нет | нет | нет |
| Управление участниками и ролями | да | да | нет | нет | нет |
| Управление правами плагинов | да | да | нет | нет | нет |
| Эксплуатационные действия | да | да | да | нет | нет |
| Торговые действия | да | нет | нет | да | нет |
| `mainnet.approve` | да | нет | нет | нет | нет |
| Просмотр административного аудита | да | да | да | нет | нет |

`admin` не может выдать роль `owner`, изменить или удалить владельца, обойти
`mainnet.approve` и получить recovery/support authority. Последний активный
`owner` защищён и application-инвариантом, и PostgreSQL trigger. Последний
`installation_owner` также защищён trigger; advisory transaction locks
сериализуют конкурирующие удаления и понижения.

## Поток доверия и API

1. Cookie dependency разрешает стабильный внутренний `user_id`.
2. Organization use case читает членство/installation ownership из server-side
   repository.
3. Роль преобразуется в фиксированный набор permissions; provider claims и
   `paid_level` в решении RBAC не участвуют.
4. Привилегированные мутации требуют session marker не старше 10 минут. Stage
   `06` заменит marker времени создания сессии на отдельное подтверждение.
5. Успешная или отклонённая API-операция записывается без исходного payload.

Основные маршруты:

- `POST /api/v1/installations/bootstrap`;
- `GET|POST /api/v1/organizations`;
- `POST|PATCH|DELETE /api/v1/organizations/{organization_id}/members...`;
- `POST /api/v1/organizations/{organization_id}/invitations`;
- `PUT /api/v1/organizations/{organization_id}/plugins/{plugin_id}/permissions/{user_id}`;
- `POST /api/v1/installations/support-access`;
- `GET /api/v1/organizations/{organization_id}/audit`.

## Persisted invariants

Миграция `0011_identity_organizations_rbac_audit_v1.sql` применяется только
после Alembic и `0010`. Она fail-closed, если целевые product resource tables
уже содержат строки: это предотвращает неявный backfill текущих данных.

`identity_memberships(organization_id,user_id)` является составной точкой
целостности. `exchange_connections`, `strategy_strategies`, `backtest_jobs`,
`strategy_backtest_variant_provenance`, `strategy_position_ownership`,
`exchange_account_snapshots`, `exchange_position_snapshots` и
`strategy_exchange_bindings` получают обязательный `organization_id`.
Составные внешние ключи гарантируют одновременно:

- владелец является участником той же организации;
- provenance ссылается на strategy и backtest job той же организации;
- position ownership и binding ссылаются на strategy и exchange connection той
  же организации;
- exchange position ссылается на account snapshot и connection той же
  организации;
- orphan-ссылки отклоняются PostgreSQL.

## Аудит, support и чувствительные данные

`identity_administrative_audit_events` является append-only: `UPDATE` и
`DELETE` блокируются trigger. Записываются actor, область, действие, тип/ID
цели, `succeeded|rejected`, безопасный reason code и время. Email приглашения
хранится только как SHA-256; API его не возвращает. Database constraint
запрещает ключи `password`, `token`, `secret`, `credential`, `cookie`,
`authorization`, `dsn`, `api_key` и `private_key` на любой глубине JSON text.

Support-доступ отсутствует по умолчанию, создаётся только
`installation_owner`, имеет причину и срок не более 24 часов и оставляет audit
event. Истёкший grant закрывается и может быть безопасно заменён новым с
отдельным событием `support_access.expired`. Он не создаёт постоянной
специальной роли.

## Совместимость и поэтапный переход

| Поверхность | Классификация | Последствие |
|---|---|---|
| Новые `/api/v1` DTO и маршруты | `compatible-change` | Добавочная поверхность; существующие endpoints сохранены. |
| Identity/RBAC semantics | `breaking-change` | Authorization больше не выводится из `paid_level` или client payload. |
| PostgreSQL ownership schema | `breaking-change` | Только пустая greenfield база; backfill и dual-read отсутствуют. |
| Общие `InstallationId`/`OrganizationId` | `compatible-change` | Новые value objects для последующих contexts. |
| Resource ownership/request namespace | `breaking-change` | Stages `09`,`10` обязаны включить `organization_id` в write/read/hash boundaries. |
| Audit/support operations | `compatible-change` | Новые versioned события и fail-closed операции. |
| Browser UI | `none` | Административный UI относится к Stage `19`. |

Поиск `paid_level` нашёл 48 файлов-потребителей в `src`, `apps`, migrations и
tests. Поле намеренно сохранено; удаление до миграции admission/UI/RL consumers
было бы недоказанным breaking change.

Rollback текущих production-данных не нужен и не разрешён. Внутри нового
self-hosted lifecycle откат выполняется восстановлением пустого/резервного
тома до применения `organization-0011`; Stage `21` докажет backup/restore.

## Доказательства и stop gates

- focused identity/API/migration tests охватывают две организации, все роли,
  admin member/plugin management, `recent-auth`, last-owner и аудит;
- `uv run python -m apps.migrations.verify_storage_runtime` поднимает реальные
  PostgreSQL/ClickHouse/Redis, применяет `organization-0011` и выполняет
  отрицательные SQL-проверки same-org/orphan/audit;
- production users, базы, identifiers и credentials не читаются;
- любой cross-org доступ, чувствительный audit payload или попытка применить
  схему поверх заполненных resource tables блокирует этап.

## Остаточные риски и передача

- Stages `09`,`10` должны обновить resource adapters/DTO/hash identity; до этого
  обязательный `organization_id` намеренно предотвращает старые unscoped writes.
- Stage `06` должен заменить session-created marker настоящим recent-auth и
  добавить браузерный bootstrap/recovery proof.
- Stage `08` должен перенести support/OpenBao secrets в отдельную доверенную
  границу; Stage `19` добавит административный интерфейс.
