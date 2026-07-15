# Инициализация и миграции хранилищ

## Назначение

Единая команда жизненного цикла подготавливает пустые PostgreSQL и ClickHouse,
проверяет Redis и возвращает машиночитаемый статус для будущего `roehubctl`.
Она относится только к новой self-hosted-установке и никогда не читает, не
импортирует и не исправляет текущие production-базы Roehub.

Пользовательским установочным входом остаётся `roehub.yaml`. Команда получает
только сгенерированный `service-config.json`, а реальные DSN и URL — из
локальной среды процесса:

- `ROEHUB_STORAGE_POSTGRES_DSN`;
- `ROEHUB_STORAGE_CLICKHOUSE_DSN` для профилей `trading` и `ml`;
- `ROEHUB_STORAGE_REDIS_URL`.

Значения этих переменных не передаются аргументами, не записываются в статус и
не должны попадать в журнал. В штатном контуре их формирует доверенная граница
из ссылок OpenBao; ручное редактирование generated Compose не является
поддерживаемым способом настройки.

## Сертифицированные профили

| Хранилище | Сертифицированная версия | Обязательные возможности |
|---|---|---|
| PostgreSQL | `16.x` | transactional DDL, создание объектов в `public`, временные таблицы, advisory lock |
| ClickHouse | `24.8.x` | создание базы и таблиц, `MergeTree`, системный каталог |
| Redis | `7.2.x` | `PING`, read/write/delete, AOF, `maxmemory-policy=noeviction` |

Другой движок или неподтверждённая версия не считаются совместимыми только
потому, что принимают часть протокола. Проверка завершается ошибкой до допуска
приложения.

## Порядок bootstrap

```bash
python -m apps.migrations.storage_main bootstrap \
  --service-config /etc/roehub/service-config.json
```

Порядок фиксирован:

1. Проверяются endpoint host и возможности PostgreSQL, ClickHouse и Redis.
2. PostgreSQL применяет `identity-0001-0009` и сохраняет контрольную сумму.
3. Alembic доходит до единственной вершины и сохраняет контрольную сумму
   каждого revision.
4. После появления `strategy_strategies` применяется `strategy-0010`.
5. ClickHouse применяет `0001` и `0002`; состояние и rendered checksum хранятся
   в самой целевой базе.
6. Итоговый статус повторно доказывает версии и обязательные таблицы.

Redis не хранит миграционную истину. Его состояние можно восстановить из
PostgreSQL/ClickHouse и доменных процессов.

## Readiness и статус

Внешний профиль допускается только после успешной команды:

```bash
python -m apps.migrations.storage_main readiness \
  --service-config /etc/roehub/service-config.json
```

Для диагностики используется тот же строгий schema gate:

```bash
python -m apps.migrations.storage_main status \
  --service-config /etc/roehub/service-config.json \
  --output-json /run/roehub/storage-status.json
```

Выход имеет схему `io.roehub.storage-status/v1alpha1`, не содержит endpoint,
credentials или secret references и включает:

- сертифицированный движок и версию;
- фактические версии схем PostgreSQL/ClickHouse;
- подтверждённые возможности;
- prerequisites резервного копирования;
- итоговый `ready`.

## Восстановление после прерывания

Не удаляйте таблицы `roehub_storage_migrations`, `alembic_version` или
`roehub_schema_migrations` и не меняйте уже применённые файлы. После устранения
причины повторите ту же команду `bootstrap` с тем же комплектом release:

- завершённые версии сверяются по SHA-256 и пропускаются;
- незавершённый ClickHouse DDL повторяется, поэтому разрешён только
  идемпотентный `CREATE ... IF NOT EXISTS`;
- checksum drift уже применённой версии блокирует запуск и требует нового
  versioned migration, а не переписывания истории;
- PostgreSQL Alembic защищён advisory lock.

## Резервное копирование перед будущим обновлением

До применения новой версии обязательны:

1. Успешный `status` на исходной версии.
2. Согласованный `pg_dump` или snapshot PostgreSQL persistent volume.
3. Согласованный backup ClickHouse parts либо `FREEZE`-процедура.
4. Отдельно восстанавливаемые credentials/OpenBao material.
5. Документированный rebuild Redis; Redis backup не заменяет durable stores.

Stage `04` определяет эти prerequisites и версии. Реальный backup/restore,
измерение RPO/RTO и release-to-release rollback доказывает Stage `21`.

## Проверка на пустых одноразовых хранилищах

```bash
uv run python -m apps.migrations.verify_storage_runtime
```

Проверка сама создаёт Compose volumes, выполняет намеренно прерванную миграцию,
восстанавливает её, повторяет bootstrap, перезапускает контейнеры, проверяет
внешний профиль и в конце удаляет контейнеры и volumes. Предварительно вручную
создавать volumes не требуется.
