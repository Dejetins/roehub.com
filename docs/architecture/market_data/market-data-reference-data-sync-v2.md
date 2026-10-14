# Market Data — Reference Data Sync (Whitelist -> PostgreSQL) (v2)

## Consumer wiring — S05, 2026-10-14

Чтение справочника, поиска и coverage переведено на PostgreSQL; public DTO, организация, авторизация и loading/snapshot semantics сохранены.

Текущий source-контракт: [PostgreSQL consumer composition](../../../src/trading/contexts/market_data/adapters/outbound/persistence/postgres/CONSUMERS.md).
Используется `STRATEGY_PG_DSN`, общий с control-plane fencing. Эта поправка
заменяет storage-specific wiring/команды прежнего baseline ниже; остальные
контракты сохраняются. Проверен disposable PG-контур; установленная среда
не переключалась. Установка/backup покрыты S06; полный runtime regression и installed cutover остаются S08/S10.

## PostgreSQL storage contract

Текущий source-контракт: [PostgreSQL storage](market-data-real-adapters-postgresql-parquet-v1.md). Реализация использует PostgreSQL; установленный runtime переключается отдельно в S10.

Reference PK и updated_at precedence переносятся в PG upsert; IDs 1–4, enabled/tradable и nullable enrichment сохраняются. Равные версии разрешаются стабильным tie-break, не порядком прихода.


> Исторический документ: `whitelist.csv` больше не является источником runtime
> policy. Текущий metadata-only refresh заполняет глобальный каталог биржи, но
> не создаёт selection или historical backfill; выбор организации хранится в
> PostgreSQL. См. `roehub-market-data-activation-and-instrument-selection-v1`.

Этот документ фиксирует правила и минимальные механизмы заполнения reference-таблиц PostgreSQL
для bounded context `market_data` на этапе v2.

Цель:
- гарантировать наличие 4 рынков в `market_data.ref_market` (binance/bybit × spot/futures);
- синхронизировать whitelist CSV в `market_data.ref_instruments` без “дубликатов” при повторных запусках;
- обеспечить воспроизводимый операторский запуск через CLI и Jupyter notebook.

Источник правды по хранению reference data: PostgreSQL DDL:
- `market_data.ref_market`
- `market_data.ref_instruments`

Связанные документы:
- `Market Data — Runtime Config & Invariants (v2)` (configs + whitelist + slicing)


## Ключевые решения

### PostgreSQL reference state

`ref_market` имеет primary key `market_id`; seed добавляет только отсутствующие
рынки 1–4 (binance/bybit × spot/futures) через `ON CONFLICT DO NOTHING`.
`ref_instruments` имеет primary key `(market_id, symbol)`.

Текущий `reference_writers.py` делает version-aware upsert. Enable и enrichment
имеют независимые clocks; обновление enable не стирает метаданные. Равные версии
разрешаются стабильным payload tie-break. Nullable enrichment может очищать поле
в новой целой версии строки. Повторный неизменный sync идемпотентен.

Application Clock задаёт UTC `updated_at`. Порты `MarketRefWriter` и
`InstrumentRefWriter`, DTO `WhitelistInstrumentRow` и `InstrumentRefUpsert`
сохраняют свои сигнатуры. `existing_latest` читает текущую уникальную строку;
`upsert` атомарно применяет только новую/победившую версию.

`SeedRefMarketUseCase` и `SyncWhitelistToRefInstrumentsUseCase` остаются
совместимыми CLI/import utilities. `sync-instruments` и
`scripts/data/market_data/01_sync_instruments.ipynb` используют
`STRATEGY_PG_DSN` и `apps/cli/wiring/db/postgres.py`.
Их whitelist не заменяет организационный выбор из текущего activation contract.
Metadata-only catalog refresh не включает collection и не запускает backfill.

Схема: `migrations/postgres/0036_market_data_storage_v1.sql` и последующие
append-only миграции из manifest. Тесты: `test_transactional_ingestion_postgres.py`
(reference clocks/nulls/ties), `test_postgres_readers_frontier.py` (filters/order).
