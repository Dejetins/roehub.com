# Этап 04 — инициализация хранилищ и миграции

## Статус

- `stage`: `04`.
- `status`: `accepted`.
- `proof_boundary`: `N/A`, пустые одноразовые PostgreSQL/ClickHouse/Redis.
- `real-boundary evidence`: `passed`; настоящий Docker Engine выполнил полный
  жизненный цикл на одноразовых PostgreSQL, ClickHouse и Redis.
- `production databases`: не читались и не изменялись.
- `next_allowed_stage`: `05`.

## Что реализовано

1. Добавлена единая команда `apps.migrations.storage_main`, которая проверяет
   capabilities, мигрирует PostgreSQL и ClickHouse в фиксированном порядке,
   проверяет Redis и возвращает `io.roehub.storage-status/v1alpha1`.
2. PostgreSQL получил durable migration state с SHA-256 для двух raw-SQL фаз и
   каждого Alembic revision. Alembic остаётся под advisory lock.
3. Исправлен clean-install порядок: `0010_strategy_exchange_bindings_v1.sql`
   теперь применяется после Alembic, когда уже существует
   `strategy_strategies`. Старый порядок был совместим только с ранее
   подготовленной базой и ломал бы пустую установку.
4. ClickHouse получил ordered manifest, rendered checksums для выбранного имени
   базы и durable state в самой целевой базе. Destructive DDL блокируется.
5. Embedded Compose использует digest-pinned PostgreSQL `16`, ClickHouse
   `24.8`, Redis `7.2` и Python base; данные находятся в автоматически
   создаваемых named volumes, сеть не публикует localhost ports.
6. External профиль имеет обязательный readiness gate: точные движки, версии,
   capabilities, checksum history и обязательные таблицы должны быть
   подтверждены до допуска.
7. Redis проверяется как AOF/no-eviction transport, но не используется как
   единственный или долговечный источник истины.
8. Зафиксированы prerequisites backup и машиночитаемое schema-version
   reporting для будущего `roehubctl`.

## Реальная граница

Команда:

```bash
uv run python -m apps.migrations.verify_storage_runtime
```

Результат:

```text
{"cleanup":"passed","compose":"passed","docker":"29.6.1|29.5.2","external_readiness":"passed","fresh_bootstrap":"passed","idempotent_rerun":"passed","interrupted_recovery":"passed","persistent_volume_restart":"passed","schema":"io.roehub.storage-runtime-proof/v1alpha1"}
```

Проверка использовала пустые автоматически созданные volumes и доказала:

- реальный PostgreSQL `16.14`, ClickHouse `24.8.14.39`, Redis `7.2.14`;
- намеренное прерывание после полного PostgreSQL head и частичного ClickHouse
  DDL без ClickHouse marker;
- успешное восстановление повторным canonical bootstrap;
- второй byte-equivalent/idempotent запуск без повторного применения версий;
- сохранение версий после остановки и запуска новых container processes на тех
  же volumes;
- внешний режим тех же трёх stores через strict readiness;
- удаление proof containers, network и volumes после проверки.

## Локальные проверки

- `uv run ruff check apps/migrations/ tests/unit/apps/migrations/` — `passed`.
- `uv run pyright apps/migrations/ tests/unit/apps/migrations/test_storage_lifecycle.py` — `passed`.
- `uv run pytest -q tests/unit/apps/migrations/` — `60 passed`.
- Stage `03` config/schema/golden regression — `21 passed`; после добавления
  storage-входов value-free inventory штатно обновлён до 123 имён.
- `docker compose ... config --quiet` для embedded/external — `passed`.
- JSON Schema для installation/status и оба migration manifests — `passed`.
- docs index generation/check — `passed`.
- project map check — `passed`, 5 generated artifacts актуальны.
- `git diff --check` — `passed`.

## Контракты и совместимость

| Поверхность | Классификация | Обоснование |
|---|---|---|
| API / DTO / application ports | `none` | Публичные HTTP DTO и доменные ports не менялись. |
| CLI / operations | `compatible-change` | Добавлена versioned command/status surface; старые entrypoints сохранены. |
| PostgreSQL persistence | `breaking-change` | Greenfield lifecycle вводит новый порядок, markers и checksum history; legacy import отсутствует. |
| ClickHouse persistence | `breaking-change` | DDL получает ordered versions, целевое имя базы и durable checksum state. |
| Redis semantics | `compatible-change` | Явно закреплены transport-only, AOF и no-eviction invariants. |
| Installation config | `breaking-change` | Localhost store hosts теперь запрещены; certified external profile fail-closed. |
| Identity / ownership | `none` на этом этапе | Новые org/RBAC constraints принадлежат Stage `05`; legacy current rows не импортировались. |
| Request hash / cache key | `none` | Request identity и content hashes приложения не менялись. |
| Migration identity | `breaking-change` | История фиксируется per-version SHA-256; переписывание applied history запрещено. |
| External effects | `none` | Provider/exchange/notification вызовы не выполнялись. |
| Audit / runbooks | `compatible-change` | Добавлены status contract, backup prerequisites и инструкция восстановления. |
| Browser defaults | `none` | Web UI не менялся. |

## Файлы этапа

Созданы:

- `apps/migrations/storage.py`;
- `apps/migrations/storage_main.py`;
- `apps/migrations/verify_storage_runtime.py`;
- `migrations/postgres/manifest.json`;
- `migrations/clickhouse/manifest.json`;
- `infra/docker/Dockerfile.storage-migrations`;
- `infra/docker/storage-embedded.compose.yml`;
- `infra/docker/storage-external.compose.yml`;
- `schemas/config/storage-status.schema.json`;
- `tests/unit/apps/migrations/test_storage_lifecycle.py`;
- `tests/fixtures/installation/roehub-external.yaml`;
- `tests/fixtures/storage/clickhouse-interrupted/*`;
- `docs/runbooks/storage-bootstrap-and-migrations.md`;
- этот отчёт.

Изменены:

- `apps/migrations/bootstrap.py`;
- `apps/migrations/__init__.py`;
- `schemas/config/roehub.schema.json`;
- журнал этапов;
- generated docs index после проверки.

Удалённых файлов нет. Старые migration/Compose history не переписывались.
Foreign изменения `.codex/PLANS.md`, supersession docs и generated project-map
сохранены. Staging, commit, push, deploy и production mutation не выполнялись.

## Остаточные риски и передача Stage 05

- Сертифицированная матрица намеренно узкая: PostgreSQL `16.x`, ClickHouse
  `24.8.x`, Redis `7.2.x`. Расширение требует отдельного compatibility proof.
- Stage `04` определил prerequisites backup, но не заменяет реальный
  backup/restore и RPO/RTO Stage `21`.
- Миграционный OCI image и storage layers должны пройти полный license/SBOM,
  signing и offline-bundle gate на Stages `22`–`24`.
- Stage `05` обязан создать fresh organization/RBAC/ownership schema поверх
  принятого PostgreSQL lifecycle без backfill текущих production rows.
