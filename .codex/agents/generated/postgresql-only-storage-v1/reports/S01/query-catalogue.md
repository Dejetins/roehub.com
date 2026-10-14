# S01 query catalogue — proposed PG access, observed source signatures

Baseline: current dirty tree; exact method/signature/body hashes and SQL fragments
are in `baseline-signatures.json`; importer graph is in `caller-edges.json`.
Source-only audit, no query execution or EXPLAIN result. Target query IDs below are
stable catalogue labels; S04 binds each to actual SQL and evidence, S09 measures
production implementations. Zero observed benchmark parity is claimed here.

## Families, callers and bounds

Timeout pairs are statement/lock; pool wait <=1 s is inside caller deadline.
Existing stricter deadlines win. A returned multi-year array is necessarily sized
by output, but transport chunks and duplicate materializations must stay bounded.

| ID | Ports/callers | Predicate/order and access path | Cardinality / class / timeout | Fixed parity cases |
|---|---|---|---|---|
| Q01 | RawKlineWriter; insert_buffer/persistent_writer; backfill/rest_fill/rest_catchup | COPY bounded staging, stable ordered day locks, version UPSERT raw/canonical/masks/summary; monthly partitions | existing finite batch/byte budget; ingestion 4 / 5s/250ms | same-key duplicates, stale/equal version permutation, metadata-only replay, failure and unknown commit |
| Q02 | CanonicalCandleReader; artifact publisher/precompute, indicator MarketDataCandleFeed, org-scoped backtest ACL, RL dataset | market/symbol + ts_open >=start and <end, PK ordered ts_open ASC, stream binary batches, monthly pruning | returned range; chunks <=8192 rows or8MiB; bulk2 /30s/250ms | all17 fields, NaN/null, UTC boundary, sorted unique, fixed NPY/backtest inputs |
| Q03 | CanonicalCandleIndexReader bounds/bounds_1m/max_ts_open_lt; tail/provider/bootstrap | global bounds from summary; bounded range/strict-before via clipped actual day masks, seek nearest occupied day index | <=one bound pair; interactive6 /1s/250ms | empty=NULL, epoch zero real, before inside minute, all-before and future-only |
| Q04 | daily_counts/distinct_ts_opens; gap verification/backfill callers | market/symbol/day range; clip boundary masks, popcount; enumerate only requested actual minutes when port explicitly returns timestamps | days in requested range; timestamps proportional to required output, never global series; interactive6 /1s/250ms (bounded repair10s) | partial days, no absences counted, duplicate replay exact |
| Q05 | CatalogCoverageReader counts/latest; API workspace/catalog and scheduler snapshot | <=4 markets, <=7day range, batched summary/daily masks, latest via summary+clipped mask, stable keys | <=80000 keys, no N+1 history scan; interactive6 /1s/250ms | missing keys/days, zero actual count, latest ignores source absences |
| Q06 | InstrumentCoverageReader.read; API instrument projection | exact actual popcount over expected range, summary for full interior; partial boundaries clipped | one state/percent result; interactive6 /1s/250ms | nonpositive expected window unknown; no false100% on absences |
| Q07 | HistoryCoverageReader counts/source_absent_counts/first_gap; work_requests/history_readiness/history_stream_gate | use generation/policy frontier, daily actual/absence masks with actual precedence; first incomplete day then minute | <=7day verification work slices, missing rows are gaps; reconcile2 /10s/250ms | disjoint complete jobs, missing day, stale negative evidence, actual overrides absence |
| Q08 | HistoryRangeVerifier.confirm; REST consumed-response verification | lock current lease/generation and bounded requested range; reconcile actual bits and accepted absence, advance only contiguous frontier | <=7days per verification; reconcile2 /10s/250ms | 429/451/timeout/partial page/stale lease never absence; crash/idempotent repair |
| Q09 | StoredCoordinates page/observation_page/observation; BaseSourceReconciler + stored_candle_signal hints | summary keyset >(market,symbol), <=50; exact actual clipped before cutoff + generation/revision consistent snapshot | <=50 observations, one for hint; reconcile2 /10s/250ms | empty keys still advance, old correction without hint, deleted/reset generation, unchanged bounds/count |
| Q10 | RefMarketWriter; seed_ref_market CLI/wiring | market PK IN bounded IDs; version upsert, stable1–4 | <=4 rows; reference2 /5s/250ms | idempotent seed, stale/equal versions, enabled state |
| Q11 | InstrumentRefWriter + enrichment; sync_whitelist/refresh_catalog/enrich | (market,symbol) bounded key batch, updated_at precedence, no per-key lookup loop | input chunks<=1000; reference2 /5s/250ms | nullable assets/steps/min_notional; enrichment does not erase unrelated columns |
| Q12 | enabled markets/instruments/search + BTCUSDT readiness; API/list/search/live workers | small reference tables, enabled/tradable join, prefix/filter and stable symbol order; paging/index appropriate to API | <=4 markets; bounded search limit from port; enumerate enabled universe by chunks internally; interactive6 /1s/250ms | market disabled, tradable false, empty query, market2/4 funding identity |
| Q13 | FundingInstrumentUniverseStore; sync_futures_funding_universe CLI | ref PK version upsert/list/get, preserve interval provenance/caps | bounded1000 writes/chunk; reference2 /5s/250ms | nullable universe metadata, non8h interval, stale/equal versions |
| Q14 | FundingRateWriter; backfill_funding_rates | monthly funding-time partitions, raw provider fields + canonical version UPSERT + durable revision atomic | bounded1000/chunk; ingestion4 /5s/250ms | nullable Binance mark, Bybit category default, correction causes artifact invalidation |
| Q15 | FundingRateCoverageReader; artifact funding input and funding catchup latest | market/symbol/funding_time >=start <end PK ordered ASC; reverse LIMIT1 latest; seek relevant occupied partitions | output funding events in range, stream chunks; bulk2 /30s/250ms | exact timestamps, interval changes, output mark_price NULL/data_quality1 baseline, empty coverage |
| Q16 | S02 partition/repair lifecycle only | serialized precreation for explicit import months/current+next; bounded repair checkpoint | maintenance reserve /60s/2s | month/year boundary, interrupted DDL, missing privileges, no default-partition hidden growth |

Q03/Q04/Q07/Q09 must not fall back to unbounded canonical scans on unavailable index.
Full history observation of sparse keys should use summary/occupied-day seeks;
verification is explicit bounded work. Sequential scans of genuinely small reference
or summary tables are permitted when measured; an index-only node is not required.

## Direct script and notebook queries

- `scripts/rl_trading/stage04b_binance_futures_history_backfill.py`: import/catchup,
  raw/index receipts Q01/Q03/Q07; CLI/store config must change together (S05).
- `stage05_roehub_dataset_builder.py`: direct `_read_candle_batch`, ordered range and
  `_bounded_source_window` map to Q02/Q03. Preserve slab/resume source signatures,
  float32 feature conversion after the existing float64 input boundary and feature hashes.
- `stage06_dataset_qa_session_extractor.py` and
  `stage08j_article_session_extractor_dataset.py`: direct source selection/export
  and downstream dataset QA, Q02/Q03; migrate executable SQL and resume manifests.
- `tests/notebook_tests/02_rest_catchup_1m.ipynb`: executable examples remain tooling,
  port to PG or remove with semantic test replacement; not immutable evidence merely
  because the filename contains a stage number (S07).
- `scripts/ops/optimize_canonical_partitions.sh`: retire; no PG OPTIMIZE analogue.
- `tools/qa/postgres_storage_prototype.py`: isolated comparison tool, not a production
  adapter; S07 ports/removes old-engine executable route, preserves results separately.

## Method-to-query mapping

Every public concrete storage adapter method is listed below. Private helper and
port signatures are retained in JSON. Gateway select/insert/command are infrastructure
transport with all calling families, not permission to execute arbitrary SQL.

| Baseline source/class/method | Target query | Body SHA-256 prefix |
|---|---|---|
| `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py:36` — `ClickHouseStoredArtifactCoordinates.page` | Q09 | `385ba7a9b3db1f17` |
| `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py:51` — `ClickHouseStoredArtifactCoordinates.observation_page` | Q09 | `4b2d89a508a16e25` |
| `src/trading/contexts/backtest_artifacts/adapters/outbound/persistence/clickhouse/stored_coordinates.py:61` — `ClickHouseStoredArtifactCoordinates.observation` | Q09 | `a2516c8cd789f535` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/btcusdt_market_readiness_reader.py:31` — `ClickHouseBTCUSDTMarketReadinessReader.list_btcusdt_reference_rows` | Q12 | `67edfc0ea10f3e7c` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py:28` — `ClickHouseCanonicalCandleIndexReader.bounds` | Q03/Q04 | `255a70d6049f1649` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py:68` — `ClickHouseCanonicalCandleIndexReader.bounds_1m` | Q03/Q04 | `da4c61ed4d4f19c8` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py:119` — `ClickHouseCanonicalCandleIndexReader.max_ts_open_lt` | Q03/Q04 | `ed92696a479386cc` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py:179` — `ClickHouseCanonicalCandleIndexReader.daily_counts` | Q03/Q04 | `822c578c5075d7ad` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py:237` — `ClickHouseCanonicalCandleIndexReader.distinct_ts_opens` | Q03/Q04 | `410132d77a3eedc0` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py:50` — `ClickHouseCanonicalCandleReader.read_1m` | Q02 | `b331a06debd8319f` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py:78` — `ClickHouseCanonicalCandleReader.read_1m_arrays` | Q02 | `e4e200bcea7ae7e4` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/catalog_coverage_reader.py:16` — `ClickHouseCatalogCoverageReader.counts` | Q05 | `191917f306b61707` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/catalog_coverage_reader.py:41` — `ClickHouseCatalogCoverageReader.latest` | Q05 | `ae4db63aedf5b131` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_instrument_reader.py:52` — `ClickHouseEnabledInstrumentReader.list_enabled_tradable` | Q12 | `68bd5840a44ca784` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_market_reader.py:57` — `ClickHouseEnabledMarketReader.list_enabled_markets` | Q12 | `9e737ef68b19d48d` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/enabled_tradable_instrument_search_reader.py:59` — `ClickHouseEnabledTradableInstrumentSearchReader.search_enabled_tradable_by_market` | Q12 | `a4a1f4e82bac1507` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:37` — `ClickHouseFundingRateStore.upsert_funding_instruments` | Q13/Q14/Q15 | `41c75146a2b0a789` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:41` — `ClickHouseFundingRateStore.list_tradable_funding_instruments` | Q13/Q14/Q15 | `a16bf6d1856503d5` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:73` — `ClickHouseFundingRateStore.get_funding_instrument` | Q13/Q14/Q15 | `a2d8e20ffde5623d` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:81` — `ClickHouseFundingRateStore.write_funding_rates` | Q13/Q14/Q15 | `1441b5d8eb3d5e53` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:101` — `ClickHouseFundingRateStore.latest_funding_time` | Q13/Q14/Q15 | `a658444c5d05489d` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py:122` — `ClickHouseFundingRateStore.read_funding_coverage` | Q13/Q14/Q15 | `beb0dbbefc1b1619` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:17` — `ClickHouseGateway.insert_rows` | transport: Q01–Q16 | `c1c77410d59f2221` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:20` — `ClickHouseGateway.select` | transport: Q01–Q16 | `0f3d5db70faeda70` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:41` — `ClickHouseConnectGateway.insert_rows` | transport: Q01–Q16 | `b9107d71e7082fdf` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:49` — `ClickHouseConnectGateway.select` | transport: Q01–Q16 | `e6018904379ffaf4` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:57` — `ClickHouseConnectGateway.command` | transport: Q01–Q16 | `a69c6bf0df5f2335` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:126` — `ThreadLocalClickHouseConnectGateway.insert_rows` | transport: Q01–Q16 | `440c128e53aa136d` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:153` — `ThreadLocalClickHouseConnectGateway.select` | transport: Q01–Q16 | `3cc0c97b98486756` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:209` — `ThreadLocalClickHouseConnectGateway.command` | transport: Q01–Q16 | `a446658f718220b8` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py:215` — `ClickHouseCommandGateway.command` | transport: Q01–Q16 | `06766379e34065fc` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py:18` — `ClickHouseHistoryCoverageReader.counts` | Q07 | `9e6f875bc56b8855` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py:46` — `ClickHouseHistoryCoverageReader.source_absent_counts` | Q07 | `5f429ef1bd949882` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_coverage_reader.py:77` — `ClickHouseHistoryCoverageReader.first_gap` | Q07 | `54e4f417af7a1b3c` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/history_range_verifier.py:19` — `ClickHouseHistoryRangeVerifier.confirm` | Q08 | `f036cbc3762c1842` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/instrument_coverage_reader.py:31` — `ClickHouseInstrumentCoverageReader.read` | Q06 | `57412bf7edc83413` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/raw_kline_writer.py:32` — `ClickHouseRawKlineWriter.write_1m` | Q01 | `e7ce845130558c4e` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_instruments_writer.py:48` — `ClickHouseInstrumentRefWriter.existing_latest` | Q11 | `d967dd17fd8f4bec` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_instruments_writer.py:108` — `ClickHouseInstrumentRefWriter.upsert` | Q11 | `36aa29bf97a2f971` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_instruments_writer.py:140` — `ClickHouseInstrumentRefWriter.existing_latest_enrichment` | Q11 | `af6fd607c05cd0f0` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_instruments_writer.py:217` — `ClickHouseInstrumentRefWriter.upsert_enrichment` | Q11 | `ba508e50ad0cf4b9` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_market_writer.py:22` — `ClickHouseMarketRefWriter.existing_market_ids` | Q10 | `75592f2570915dd9` |
| `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/ref_market_writer.py:41` — `ClickHouseMarketRefWriter.insert` | Q10 | `d4c24d97359e9574` |
