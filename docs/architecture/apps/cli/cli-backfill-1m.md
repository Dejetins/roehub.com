# CLI — Backfill 1m Candles (Parquet -> ClickHouse) (Walking Skeleton v1)

Этот документ фиксирует CLI entrypoint (composition root) для запуска walking skeleton v1:
- чтение 1m свечей из `.parquet`,
- запуск use-case `Backfill1mCandlesUseCase`,
- запись в ClickHouse `raw_*` таблицы через port `RawKlineWriter`,
- логирование отчёта выполнения.

Цель:
- иметь воспроизводимый “реальный” запуск без биржевых клиентов (`rest/ws`);
- не нарушать границы DDD/ports: CLI только собирает зависимости и вызывает use-case.


## Связанные документы
- `docs/architecture/shared-kernel-primitives.md`
- `docs/architecture/market_data/market-data-application-ports.md`
- `docs/architecture/market_data/market-data-use-case-backfill-1m.md`
- `docs/architecture/market_data/market-data-real-adapters-clickhouse-parquet.md`


## Важные решения

### 1) CLI не выбирает raw-таблицу напрямую
CLI передаёт `InstrumentId(market_id, symbol)` в use-case.

Выбор raw-таблицы — ответственность адаптера `ClickHouseRawKlineWriter`:
- `market_id ∈ {1,2}` -> `market_data.raw_binance_klines_1m`
- `market_id ∈ {3,4}` -> `market_data.raw_bybit_klines_1m`

CLI не содержит знания о DDL и не встраивает детали маршрутизации.

### 2) `--batch-size` по умолчанию включает batching (рекомендуемое v1)
Семантика batching в walking skeleton v1:

- `--batch-size 10000` (default):
  use-case буферизует до `N` строк и делает запись батчами (несколько insert).
  Это снижает потребление памяти и делает большие диапазоны эксплуатационно пригодными.

- `--batch-size N`:
  тот же механизм batching с указанным размером.

(В v1 не закрепляем “выключение батчинга” как часть CLI-контракта.
Если нужно “одним батчем” — это отдельное решение/флаг в будущем.)

### 3) Отчёт выполнения пишется в лог, не в stdout print
CLI не печатает `report` напрямую.
Отчёт сериализуется в `text` или `json` и отправляется через `logging`.

### 4) ClickHouse конфигурация берётся только из ENV
ENV является источником правды (truth source).
CLI не содержит fallback или перекрытия параметров базы через флаги.

Причина:
- единое управление конфигурацией и секретами;
- предсказуемость запусков в dev/test/prod окружениях.


## Команда

### backfill-1m

**Purpose**  
Запустить backfill 1m свечей из parquet в ClickHouse raw таблицы (canonical появится через MV автоматически).

**Arguments (required)**
- `--market-id <int>`
  Допустимые значения: `1..4` согласно `market_data.ref_market`.
- `--symbol <str>`
  Например `BTCUSDT`.
- `--start <ISO-UTC>`
  Начало диапазона (UTC, timezone-aware). Формат: ISO с `Z`.
- `--end <ISO-UTC>`
  Конец диапазона (UTC, timezone-aware). Формат: ISO с `Z`.
- `--parquet <path>`
  Путь к `.parquet` файлу или директории. Аргумент может повторяться.

**Arguments (optional)**
- `--config <path>`
  Default: `configs/dev/market_data.yaml`.
  Runtime config market mapping для генерации канонического `meta.instrument_key`.
- `--batch-size <int>`
  Default: `10000`.
  Если задано число — use-case пишет батчами по `N` строк.
- `--report-format <text|json>`
  Default: `text`.
  Формат логируемого отчёта.

**TimeRange semantics**
- `TimeRange` трактуется как полуинтервал `[start, end)`.

**Behavior**
- CLI собирает зависимости (wiring) и вызывает use-case.
- Пишет raw-строки в ClickHouse, canonical формируется через MV.
- Не выполняет read-back или проверку качества.
- Не делает skip-existing и не пытается “умно” определять, что уже загружено.


## ENV configuration (ClickHouse)

CLI использует следующие переменные окружения:

- `CH_HOST`
- `CH_PORT`
- `CH_USER`
- `CH_PASSWORD`
- `CH_DATABASE`
- `CH_SECURE`
- `CH_VERIFY`

Замечание:
- пароль всегда рекомендуется задавать через env/секреты окружения.


## Wiring (composition root)

CLI wiring собирает следующий граф зависимостей:

1) `clock = SystemClock()`

2) Parquet source:
- `cfg = load_market_data_runtime_config(--config)`
- `scanner = PyArrowParquetScanner(paths=[--parquet...])`
- `source = ParquetCandleIngestSource(scanner=scanner, cfg=cfg, clock=clock, batch_size=scanner_batch_size)`

Примечание:
- batching чтения parquet (scanner batch_size) — internal detail источника.
- `--batch-size` управляет batching записи use-case.

3) ClickHouse:
- `client = clickhouse_connect.get_client(host=..., port=..., username=..., password=..., secure=..., verify=...)`
- `gateway = ClickHouseConnectGateway(client)`
- `writer = ClickHouseRawKlineWriter(gateway=gateway, database=CH_DATABASE)`

4) Use-case:
- `use_case = Backfill1mCandlesUseCase(source=source, writer=writer, clock=clock, batch_size=<batch-size>)`

5) Execution:
- `command = Backfill1mCommand(instrument_id, time_range)`
- `report = use_case.run(command)`
- `logger.info(report)` в формате `text|json`


## Logging

Отчёт `Backfill1mReport` логируется одной записью уровня `INFO`.

Рекомендуемое содержимое (v1):
- `instrument_id`
- `time_range`
- `started_at`, `finished_at`
- `candles_read`, `rows_written`, `batches_written`

Формат `json` предназначен для последующего парсинга (например, лог-агентом/ELK).


## Non-goals (walking skeleton v1)

CLI и use-case намеренно не делают:
- чтение `canonical_candles_1m` для проверки;
- селективное “skip-existing” и дедуп на уровне raw/canonical;
- quality report (пропуски, полнота, outliers);
- сетевые источники `rest/ws`.


## File placement

- `apps/cli/main/main.py` — entrypoint
- `apps/cli/commands/backfill_1m.py` — реализация команды
- `apps/cli/wiring/modules/market_data.py` — сборка use-case и адаптеров
- `apps/cli/wiring/db/clickhouse.py` — создание CH client/gateway
- `apps/cli/wiring/clients/parquet.py` — создание parquet scanner/source
