# Market Data — Reference Data Sync (Whitelist -> ClickHouse) (v2)

Этот документ фиксирует правила и минимальные механизмы заполнения reference-таблиц ClickHouse
для bounded context `market_data` на этапе v2.

Цель:
- гарантировать наличие 4 рынков в `market_data.ref_market` (binance/bybit × spot/futures);
- синхронизировать whitelist CSV в `market_data.ref_instruments` без “дубликатов” при повторных запусках;
- обеспечить воспроизводимый операторский запуск через CLI и Jupyter notebook.

Источник правды по хранению reference data: ClickHouse DDL:
- `market_data.ref_market`
- `market_data.ref_instruments`

Связанные документы:
- `Market Data — Runtime Config & Invariants (v2)` (configs + whitelist + slicing)


## Ключевые решения

### 1) Seed `ref_market` — insert-only-missing (строгая идемпотентность)
Таблица `ref_market` использует `ReplacingMergeTree(updated_at)`.
Чтобы повторный запуск не создавал “дубликатов” даже до merge,
seed выполняется как:
- читаем существующие `market_id` для набора {1,2,3,4}
- вставляем только отсутствующие.

Список рынков (фиксированный для v2):
1) binance spot   `binance:spot`
2) binance futures `binance:futures`
3) bybit spot     `bybit:spot`
4) bybit futures  `bybit:futures`


### 2) Whitelist CSV — источник правды для набора инструментов
Whitelist определяет, какие тикеры считаются активными/неактивными.
Правила CSV и last-win описаны в документе runtime config.

Важно:
- строки с `is_enabled=0` НЕ участвуют в подписках/догонке, но должны быть отражены в `ref_instruments`
  (как `DISABLED`/`is_tradable=0`), чтобы фиксировать “операторское” намерение.


### 3) Sync `ref_instruments` — insert only new/changed (строгая идемпотентность)
Таблица `ref_instruments` использует `ReplacingMergeTree(updated_at)` с ключом `(market_id, symbol)`.

Мы НЕ делаем “слепой upsert” (INSERT на каждую строку whitelist), потому что это создаёт новые версии строк
даже без изменений и выглядит как “дубликаты” (до merge).

Правило sync:
- читаем текущее состояние `ref_instruments` для ключей `(market_id, symbol)` из whitelist
  (берём “последнюю” версию по `updated_at`, без FINAL);
- выполняем INSERT (новую версию строки) только если:
  - ключ отсутствует (новый инструмент), ИЛИ
  - изменилось целевое состояние от whitelist:
    - `status` (`ENABLED`/`DISABLED`)
    - `is_tradable` (1/0)

Следствие:
- повторный sync с неизменившимся whitelist делает `0` вставок (строгая идемпотентность);
- если `is_enabled` изменился — вставляется новая версия строки с новым `updated_at`.

Политика полей `ref_instruments` для v2:
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
Команда на запись “новой версии” строки `ref_instruments` (ReplacingMergeTree).

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
Порт синхронизации reference-данных инструментов в `ref_instruments`.

**Contract**
- `existing_latest(market_id: MarketId, symbols: Sequence[Symbol]) -> Mapping[str, tuple[str, int]]`
  - возвращает текущее состояние для символов (symbol -> (status, is_tradable))
  - берётся “последняя” версия строки (по updated_at)
- `upsert(rows: Iterable[InstrumentRefUpsert]) -> None`
  - вставляет новые версии строк для new/changed инструментов


## Use-cases

### SeedRefMarketUseCase
**Purpose**  
Гарантирует наличие 4 рынков (1..4) в `ref_market` без создания “дубликатов”.

**Semantics**
- читает существующие market_id
- вставляет отсутствующие строки


### SyncWhitelistToRefInstrumentsUseCase
**Purpose**  
Синхронизирует whitelist (включая disabled) в `ref_instruments` строго идемпотентно.

**Semantics**
- валидирует, что market_id из whitelist присутствуют в runtime config
- читает текущее состояние `ref_instruments` по ключам whitelist
- формирует `InstrumentRefUpsert` только для new/changed строк
- делает INSERT новых версий строк с `updated_at` от Clock
- возвращает отчёт с количеством:
  - total
  - upserted
  - skipped_unchanged


## CLI / Notebook

### CLI command: sync-instruments
Команда:
- читает `configs/.../market_data.yaml`
- читает `configs/.../whitelist.csv`
- выполняет:
  1) `SeedRefMarketUseCase`
  2) `SyncWhitelistToRefInstrumentsUseCase`

В отчёте отображаются:
- сколько рынков добавлено
- сколько строк в whitelist
- сколько реально вставлено в ref_instruments
- сколько пропущено как неизменившиеся

### Notebook: scripts/data/market_data/01_sync_instruments.ipynb
Повторяет тот же сценарий, но в ручном режиме для ops/аналитики.


## Out of scope
- enrich инструментов из биржевых REST (base/quote/steps/min_notional) — отдельный EPIC.
