# Market Data — Reference Data Sync (Whitelist -> ClickHouse) (v2)

Этот документ фиксирует правила и минимальные механизмы заполнения reference-таблиц ClickHouse
для bounded context `market_data` на этапе v2.

Цель:
- гарантировать наличие 4 рынков в `market_data.ref_market` (binance/bybit × spot/futures);
- синхронизировать whitelist CSV в `market_data.ref_instruments`;
- обеспечить воспроизводимый операторский запуск через CLI и Jupyter notebook.

Источник правды по хранению reference data: ClickHouse DDL:
- `market_data.ref_market`
- `market_data.ref_instruments`

Связанные документы:
- `Market Data — Runtime Config & Invariants (v2)` (configs + whitelist + slicing)


## Ключевые решения

### 1) Seed `ref_market` — insert-only-missing (строгая идемпотентность)
Таблица `ref_market` использует `ReplacingMergeTree(updated_at)`.
Чтобы повторный запуск не создавал дублей даже до merge,
seed выполняется как:
- читаем существующие `market_id` для набора {1,2,3,4}
- вставляем только отсутствующие.

### 2) Whitelist CSV — источник правды для набора инструментов
Whitelist определяет, какие тикеры считаются активными/неактивными.
Правила CSV и last-win описаны в документе runtime config.

### 3) Upsert `ref_instruments` делаем через INSERT новой версии (ReplacingMergeTree)
Таблица `ref_instruments` использует `ReplacingMergeTree(updated_at)` с ключом `(market_id, symbol)`.
Поэтому upsert реализуется как вставка новой версии строки с `updated_at = now64(3)` (UTC).

Политика полей:
- `is_tradable = 1` если `is_enabled = 1`, иначе `0`
- `status = 'ENABLED'` если `is_enabled = 1`, иначе `'DISABLED'`
- остальные поля (base/quote/steps/min_notional) остаются NULL до этапа enrich

### 4) Clock — источник `updated_at`
`updated_at` задаётся от application Clock (UTC), а не полагается на server default,
чтобы обеспечить детерминизм тестов и одинаковое поведение в разных entrypoints.


## DTO (application)

### WhitelistInstrumentRow
**Purpose**
Нормализованная строка whitelist как вход в application use-case (без привязки к CSV).

**Fields**
- `instrument_id: InstrumentId`
- `is_enabled: bool`

### InstrumentRefUpsert
**Purpose**
Команда на upsert одной строки `ref_instruments`.

**Fields**
- `market_id: MarketId`
- `symbol: Symbol`
- `status: str` (`ENABLED`|`DISABLED`)
- `is_tradable: int` (0|1)
- `updated_at: UtcTimestamp`


## Ports

### MarketRefWriter
**Purpose**
Порт записи reference-данных рынков в `ref_market`.

**Contract**
- `existing_market_ids(ids: Iterable[MarketId]) -> set[int]`
- `insert(rows: Iterable[RefMarketRow]) -> None`

### InstrumentRefWriter
**Purpose**
Порт upsert reference-данных инструментов в `ref_instruments`.

**Contract**
- `upsert(rows: Iterable[InstrumentRefUpsert]) -> None`


## Use-cases

### SeedRefMarketUseCase
**Purpose**
Гарантирует наличие 4 рынков (1..4) в `ref_market` без создания дублей.

**Semantics**
- читает существующие market_id
- вставляет отсутствующие строки

### SyncWhitelistToRefInstrumentsUseCase
**Purpose**
Синхронизирует whitelist (включая disabled) в `ref_instruments`.

**Semantics**
- валидирует, что market_id из whitelist присутствуют в runtime config
- формирует `InstrumentRefUpsert` для каждой строки whitelist
- делает upsert (INSERT новой версии) с `updated_at` от Clock


## CLI / Notebook

### CLI command: sync-instruments
Команда:
- читает `configs/.../market_data.yaml`
- читает `configs/.../whitelist.csv`
- выполняет:
  1) `SeedRefMarketUseCase`
  2) `SyncWhitelistToRefInstrumentsUseCase`

### Notebook: scripts/data/market_data/01_sync_instruments.ipynb
Повторяет тот же сценарий, но в ручном режиме для ops/аналитики.


## Out of scope
- enrich инструментов из биржевых REST (base/quote/steps/min_notional) — отдельный EPIC.
