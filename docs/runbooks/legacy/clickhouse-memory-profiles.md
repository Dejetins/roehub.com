# ClickHouse Memory Profiles (Mac Studio)

Runbook фиксирует low-memory baseline для обычного режима и отдельный агрессивный профиль для редких массовых выгрузок.

## Профили

### 1) Default (low-memory)

Назначение: постоянный runtime с минимальным потреблением памяти.

Серверный baseline:

- `mark_cache_size = 268435456` (256 MB)
- `uncompressed_cache_size = 0`
- `background_schedule_pool_size = 64`
- `background_pool_size = 8`
- `background_merges_mutations_concurrency_ratio = 4`
- `max_server_memory_usage_to_ram_ratio = 0.65`

`query cache` выключен на уровне профилей (`use_query_cache = 0`), без top-level `query_cache_*` параметров.

Профиль `default` (user-level):

- `max_memory_usage = 4000000000`
- `max_threads = 6`
- `max_bytes_before_external_group_by = 536870912`
- `max_bytes_before_external_sort = 536870912`
- `use_uncompressed_cache = 0`
- `use_query_cache = 0`

### 2) Export profile (`export_high_throughput`)

Назначение: редкие массовые выгрузки, когда можно временно использовать больше ресурсов.

Профиль `export_high_throughput`:

- `max_memory_usage = 24000000000`
- `max_threads = 12`
- `max_bytes_before_external_group_by = 2147483648`
- `max_bytes_before_external_sort = 2147483648`
- `use_uncompressed_cache = 0`
- `use_query_cache = 0`

Пользователь для запуска в этом профиле:

- `roehub_export` (пароль из `CLICKHOUSE_PASSWORD`).

## Как использовать

### Обычный режим (default)

```bash
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user roehub
```

### Массовая выгрузка (export profile)

```bash
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --user roehub_export --password "$CLICKHOUSE_PASSWORD"
```

### Проверка активных лимитов в текущей сессии

```sql
SELECT
    currentUser() AS user,
    getSetting('max_memory_usage') AS max_memory_usage,
    getSetting('max_threads') AS max_threads,
    getSetting('max_bytes_before_external_group_by') AS max_bytes_before_external_group_by,
    getSetting('max_bytes_before_external_sort') AS max_bytes_before_external_sort,
    getSetting('use_query_cache') AS use_query_cache,
    getSetting('use_uncompressed_cache') AS use_uncompressed_cache;
```

### Точечный override для одной сессии/запроса

Если не нужен отдельный пользователь, допустимы `SET`/`SETTINGS`:

```sql
SET max_memory_usage = 24000000000;
SET max_threads = 12;
```

или

```sql
SELECT *
FROM market_data.canonical_candles_1m
SETTINGS
    max_memory_usage = 24000000000,
    max_threads = 12,
    max_bytes_before_external_group_by = 2147483648,
    max_bytes_before_external_sort = 2147483648;
```

## Проверка серверного baseline

```bash
/opt/clickhouse/clickhouse client --query "
SELECT name, value
FROM system.server_settings
WHERE name IN (
  'mark_cache_size',
  'uncompressed_cache_size',
  'background_schedule_pool_size',
  'background_pool_size',
  'background_merges_mutations_concurrency_ratio',
  'max_server_memory_usage_to_ram_ratio'
)
ORDER BY name
"
```

## Файлы конфигурации

- `infra/macos/clickhouse/config.xml`
- `infra/macos/clickhouse/config.test.xml`
- `infra/macos/clickhouse/users.d/roehub.xml`

## Перезапуск после изменений

```bash
bash scripts/macos/reload_launchd_services.sh prod
```

или точечно:

```bash
uid="$(id -u)"
launchctl bootout "gui/${uid}" /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.clickhouse.plist || true
launchctl bootstrap "gui/${uid}" /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.clickhouse.plist
```
