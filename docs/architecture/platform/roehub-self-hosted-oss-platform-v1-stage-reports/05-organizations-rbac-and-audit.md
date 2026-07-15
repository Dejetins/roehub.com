# Этап 05 — организации, RBAC и административный аудит

## Статус

- `stage`: `05`.
- `status`: `accepted`.
- `proof_boundary`: `N/A`, новая одноразовая установка и пустой PostgreSQL.
- `real-boundary evidence`: `passed`; настоящий PostgreSQL проверил схему,
  владение, межорганизационные запреты и неизменяемый аудит.
- `production users/databases`: не читались, не импортировались и не
  изменялись.
- `next_allowed_stage`: `06`.

## Что реализовано

1. Добавлены singleton-установка, `installation_owner`, организации, членство,
   приглашения и роли `owner`, `admin`, `operator`, `trader`, `viewer`.
2. Organization scope разрешается сервером по стабильному `user_id` и
   членству. Переданный клиентом `organization_id` не является полномочием.
3. Ролевая матрица разделяет member/role/plugin administration,
   эксплуатацию, торговлю, аудит и owner-only `mainnet.approve`.
   `admin` не может выдать роль `owner`, изменить или удалить владельца.
4. Привилегированные мутации требуют маркер недавней аутентификации не старше
   10 минут. Stage `06` заменит временную семантику времени создания сессии
   полноценным подтверждением `recent-auth`.
5. Добавлены versioned endpoints `/api/v1/installations/*` и
   `/api/v1/organizations/*` для bootstrap, организаций, участников,
   приглашений, прав плагинов, support-доступа и аудита.
6. Support-доступ отсутствует по умолчанию, выдаётся только
   `installation_owner`, ограничен 24 часами и всегда аудируется.
7. Успешные и отклонённые административные операции создают append-only audit
   event без исходного request payload. Email приглашения сохраняется только
   как SHA-256; чувствительные ключи запрещены PostgreSQL constraint.
8. Миграция `0011_identity_organizations_rbac_audit_v1.sql` входит в единый
   Stage `04` lifecycle как версия `organization-0011` и fail-closed при
   наличии legacy resource rows. Backfill и dual-read не добавлялись.
9. Обязательный `organization_id` и составные внешние ключи добавлены для
   exchange connections, strategies, backtest jobs/provenance, position
   ownership и exchange snapshots/bindings. Ссылки должны принадлежать одной
   организации, а потерянные ссылки отклоняются базой.
10. Поиск нашёл 48 файлов-потребителей `paid_level`; поле сохранено. Удаление
    отложено до доказанной миграции admission, rate-limit и UI-потребителей.

Архитектурное решение и матрица ролей зафиксированы в
`docs/architecture/identity/organizations-rbac-audit-v1.md`.

## Реальная граница

Команда:

```bash
uv run python -m apps.migrations.verify_storage_runtime
```

Результат:

```text
{"cleanup":"passed","compose":"passed","docker":"29.6.1|29.5.2","external_readiness":"passed","fresh_bootstrap":"passed","idempotent_rerun":"passed","interrupted_recovery":"passed","organization_audit_events":3,"organization_constraints":["audit_immutable","audit_sensitive_key","exchange_position_snapshot","last_owner","strategy_exchange_binding","strategy_position_ownership","strategy_provenance"],"organization_isolation":"passed","persistent_volume_restart":"passed","schema":"io.roehub.storage-runtime-proof/v1alpha1"}
```

Проверка на пустых автоматически созданных volumes доказала:

- применение `organization-0011` через canonical migration lifecycle;
- две организации и все пять ролей;
- наличие только явно выданного временного support-доступа;
- запрет cross-org strategy/exchange binding и position ownership;
- запрет cross-org или orphan provenance;
- запрет cross-org exchange position snapshot;
- запрет понижения последнего владельца;
- запрет `UPDATE` административного аудита;
- запрет чувствительного ключа `token` в audit metadata;
- сохранение всех Stage `04` bootstrap/recovery/restart/readiness гарантий;
- удаление proof containers, сети и volumes после проверки.

## Локальные проверки

- сфокусированный `ruff` для identity/API/migrations/tests — `passed`;
- сфокусированный `pyright` для identity/API/migrations/tests — `passed`;
- identity/API/migration acceptance — `120 passed`;
- расширенный regression для `tests/unit/apps/api/`,
  `tests/unit/contexts/identity/` и `tests/unit/apps/migrations/` —
  `306 passed`;
- installation/storage inventory regression — `29 passed`;
- runtime input inventory generation/check — `passed`, 123 имени без значений;
- disposable Docker/PostgreSQL/ClickHouse/Redis proof — `passed`;
- docs index generation/check — `passed`;
- project map generation/check — `passed`, 5 generated artifacts;
- `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| API / DTO `/api/v1` | `compatible-change` | Добавлены versioned routes и DTO; существующие endpoints сохранены. |
| Identity / RBAC semantics | `breaking-change` | Полномочия выводятся из серверного членства, а не из `paid_level` или payload. |
| Application ports | `compatible-change` | Добавлен отдельный organization repository port без изменения старых ports. |
| PostgreSQL persistence | `breaking-change` | Greenfield-only org ownership, обязательные столбцы, составные FK и append-only audit. |
| Installation config | `none` | Пользовательская схема `roehub.yaml` не менялась. |
| Principal / organization scope | `breaking-change` | Серверная область организации становится обязательной частью authorization. |
| Request hash / cache / resource identity | `breaking-change` | Stages `09`,`10` обязаны включить `organization_id` во все read/write/hash boundaries. |
| Межсервисные вызовы | `none` | Внешние service/provider вызовы не добавлены. |
| Внешние эффекты | `none` | Email, OIDC, exchange и production операции не выполнялись. |
| Audit / support | `compatible-change` | Добавлены versioned безопасные операции и события; support отсутствует по умолчанию. |
| Browser defaults | `none` | Browser login/admin UI принадлежат Stages `06`,`19`. |

До Stages `09`,`10` старые resource-write adapters не передают обязательный
`organization_id` и потому намеренно завершаются fail-closed на новой схеме.
Это принятый промежуточный breaking boundary, а не совместимость со старой
заполненной базой.

## Холодная проверка артефактов

- Режим: `cold self-review fallback`; независимое делегирование в этом запуске
  не разрешено.
- Первоначальный вердикт: `Release after fixes`.
- Исправлено: `installation_owner` больше не получает неявную роль `owner` в
  организации без активного членства; отклонённая операция для
  несуществующего `organization_id` аудируется в installation scope и не
  заменяет исходную API-ошибку нарушением внешнего ключа; last-owner triggers
  сериализованы advisory lock; истёкший support grant можно безопасно заменить
  новым с отдельным событием аудита.
- Повторная проверка: `ruff`, `pyright`, `19 passed`, `306 passed` и реальный
  одноразовый PostgreSQL proof завершились успешно.
- Итоговый вердикт: `Release` для Stage `05`.
- Остаточные риски: настоящий browser `recent-auth`, invitation acceptance,
  OIDC, secrets boundary и полный перевод research/trading adapters ещё не
  реализованы и принадлежат последующим явно указанным этапам.

## Файлы этапа

Созданы:

- `src/trading/shared_kernel/primitives/{installation_id.py,organization_id.py}`;
- `src/trading/contexts/identity/domain/entities/organization.py`;
- `src/trading/contexts/identity/application/ports/organization_repository.py`;
- `src/trading/contexts/identity/application/use_cases/organizations.py`;
- `src/trading/contexts/identity/adapters/outbound/persistence/in_memory/organization_repository.py`;
- `src/trading/contexts/identity/adapters/outbound/persistence/postgres/organization_repository.py`;
- `src/trading/contexts/identity/adapters/inbound/api/routes/organizations.py`;
- `migrations/postgres/0011_identity_organizations_rbac_audit_v1.sql`;
- `apps/migrations/organization_runtime_probe.py`;
- `tests/unit/contexts/identity/application/test_organization_access.py`;
- `tests/unit/apps/api/test_organizations_routes.py`;
- `tests/unit/apps/migrations/test_identity_organizations_sql.py`;
- `docs/architecture/identity/organizations-rbac-audit-v1.md`;
- этот отчёт.

Изменены:

- identity/shared-kernel package exports;
- `apps/api/{common/errors.py,routes/identity.py,wiring/modules/identity.py}`;
- `apps/migrations/{__init__.py,bootstrap.py,storage.py,verify_storage_runtime.py}`;
- `migrations/postgres/manifest.json`;
- `tests/unit/apps/migrations/test_storage_lifecycle.py`;
- `configs/installation/runtime-input-inventory.json`;
- журнал этапов и generated docs/project-map outputs после их проверки.

Удалённых файлов нет. Foreign изменения `.codex/PLANS.md`, supersession docs и
смешанные generated files сохранены. Staging, commit, push, deploy и production
mutation не выполнялись.

## Остаточные риски и передача Stage 06

- Stage `06` должен реализовать локальные credentials, настоящую ротацию
  сессий, `recent-auth`, восстановление владельца и browser/API proof.
- Stage `07` добавит provider-neutral OIDC linking при сохранении local owner
  fallback.
- Stage `08` перенесёт секреты и recovery boundary в OpenBao.
- Stages `09`,`10` обязаны завершить org-scoped resource DTO, repository,
  request-hash и cache-key migration; до этого старая запись ресурсов
  блокируется новой схемой.
- Stage `19` реализует administrative Web UI поверх принятого API/RBAC.
