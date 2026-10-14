# CLI — Backfill 1m Candles (Parquet -> PostgreSQL) (Walking Skeleton v1)

## Consumer wiring — S05, 2026-10-14

Запись raw/canonical/coverage выполняется транзакционно в PostgreSQL. REST/Parquet batches ограничены 8192 строками, metadata/funding — 1000. Месячные партиции заранее создаёт maintenance; application DDL не разрешён.

Текущий source-контракт: [PostgreSQL consumer composition](../../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/CONSUMERS.md).
Используется `STRATEGY_PG_DSN`, общий с control-plane fencing. Эта поправка
заменяет storage-specific wiring/команды прежнего baseline ниже; остальные
контракты сохраняются. Проверен disposable PG-контур; установленная среда
не переключалась. Установка/backup покрыты S06; полный runtime regression и installed cutover остаются S08/S10.

## PostgreSQL storage contract

Текущий source-контракт: [PostgreSQL storage](../../market_data/market-data-real-adapters-postgresql-parquet-v1.md). Реализация использует PostgreSQL; установленный runtime переключается отдельно в S10.

Потребители сохраняют domain/array/API и security границы. S05 переносит wiring и прямые SQL/script callers; S08 доказывает numerical/runtime parity. Упоминание прежнего engine ниже не задаёт целевую зависимость.


Этот документ фиксирует CLI entrypoint (composition root) для запуска walking skeleton v1:
- чтение 1m свечей из `.parquet`,
- запуск use-case `Backfill1mCandlesUseCase`,
- запись в PostgreSQL `raw_*` таблицы через port `RawKlineWriter`,
- логирование отчёта выполнения.

Цель:
- иметь воспроизводимый “реальный” запуск без биржевых клиентов (`rest/ws`);
- не нарушать границы DDD/ports: CLI только собирает зависимости и вызывает use-case.


## Связанные документы
- `docs/architecture/shared-kernel-primitives.md`
- `docs/architecture/market_data/market-data-application-ports.md`
- `docs/architecture/market_data/market-data-use-case-backfill-1m.md`
- `docs/architecture/market_data/market-data-real-adapters-postgresql-parquet-v1.md`


## Важные решения

### 1) CLI не выбирает raw-таблицу напрямую
CLI передаёт `InstrumentId(market_id, symbol)` в use-case.

Выбор raw-таблицы — ответственность адаптера `PostgresRawKlineWriter`:
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

### 4) PostgreSQL конфигурация берётся только из ENV
ENV является источником правды (truth source).
CLI не содержит fallback или перекрытия параметров базы через флаги.

Причина:
- единое управление конфигурацией и секретами;
- предсказуемость запусков в dev/test/prod окружениях.


## Команда

### backfill-1m

**Purpose**  
Запустить backfill 1m свечей из parquet в PostgreSQL raw таблицы (canonical/coverage/revision обновятся в той же транзакции).

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
- Пишет raw-строки в PostgreSQL, canonical/coverage/revision обновляются в той же транзакции.
- Не выполняет read-back или проверку качества.
- Не делает skip-existing и не пытается “умно” определять, что уже загружено.


## ENV configuration (PostgreSQL)

CLI требует явный `STRATEGY_PG_DSN` текущей мигрированной PostgreSQL базы.
DSN передаётся через приватное окружение и не попадает в отчёты.

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

3) PostgreSQL:
- `gateway = market_gateway(environ, workload="bulk")`
- `writer = PostgresRawKlineWriter(gateway)`
- Gateway ограничивает row/byte batches, deadline и число соединений до handshake.

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
- `apps/cli/wiring/db/postgres.py` — создание bounded PostgreSQL gateway
- `apps/cli/wiring/clients/parquet.py` — создание parquet scanner/source
