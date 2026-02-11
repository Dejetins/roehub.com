.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── apps/
│   ├── api/
│   │   ├── dto/
│   │   ├── main/
│   │   ├── routes/
│   │   └── wiring/
│   │       ├── clients/
│   │       ├── container/
│   │       ├── db/
│   │       └── modules/
│   ├── cli/
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── backfill_1m.py
│   │   │   ├── rest_catchup_1m.py
│   │   │   └── sync_instruments.py
│   │   ├── main/
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   ├── wiring/
│   │   │   ├── clients/
│   │   │   │   ├── __init__.py
│   │   │   │   └── parquet.py
│   │   │   ├── container/
│   │   │   ├── db/
│   │   │   │   ├── __init__.py
│   │   │   │   └── clickhouse.py
│   │   │   ├── modules/
│   │   │   │   ├── __init__.py
│   │   │   │   └── market_data.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   └── test_backfill_1m_parsing.py
│   ├── scheduler/
│   │   ├── main/
│   │   ├── market_data_scheduler/
│   │   │   ├── main/
│   │   │   │   ├── __init__.py
│   │   │   │   └── main.py
│   │   │   ├── wiring/
│   │   │   │   ├── modules/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── market_data_scheduler.py
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   └── wiring/
│   │       ├── clients/
│   │       ├── container/
│   │       ├── db/
│   │       └── modules/
│   ├── worker/
│   │   ├── handlers/
│   │   ├── main/
│   │   ├── market_data_ws/
│   │   │   ├── main/
│   │   │   │   ├── __init__.py
│   │   │   │   └── main.py
│   │   │   ├── wiring/
│   │   │   │   ├── modules/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── market_data_ws.py
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   └── wiring/
│   │       ├── clients/
│   │       ├── container/
│   │       ├── db/
│   │       └── modules/
│   └── __init__.py
├── configs/
│   ├── dev/
│   │   ├── market_data.yaml
│   │   └── whitelist.csv
│   ├── prod/
│   │   ├── market_data.yaml
│   │   └── whitelist.csv
│   └── test/
├── deploy/
├── docs/
│   ├── api/
│   ├── architecture/
│   │   ├── apps/
│   │   │   └── cli/
│   │   │       └── cli-backfill-1m.md
│   │   ├── market_data/
│   │   │   ├── market-data-application-ports.md
│   │   │   ├── market-data-live-feed-redis-streams-v1.md
│   │   │   ├── market-data-real-adapters-clickhouse-parquet.md
│   │   │   ├── market-data-reference-data-sync-v2.md
│   │   │   ├── market-data-rest-historical-catchup-1m-v2.md
│   │   │   ├── market-data-runtime-config-invariants-v2.md
│   │   │   ├── market-data-use-case-backfill-1m.md
│   │   │   ├── market-data-ws-live-ingestion-worker-v1.md
│   │   │   └── market_data_ddl.sql
│   │   ├── .DS_Store
│   │   └── shared-kernel-primitives.md
│   ├── decisions/
│   ├── runbooks/
│   │   ├── market-data-autonomous-docker.md
│   │   ├── market-data-metrics-reference-ru.md
│   │   ├── market-data-metrics.md
│   │   └── market-data-redis-streams.md
│   └── .DS_Store
├── infra/
│   ├── docker/
│   │   ├── .env.example
│   │   ├── Dockerfile.market_data
│   │   ├── docker-compose.market_data.yml
│   │   └── docker-compose.yml
│   ├── k8s/
│   └── monitoring/
│       └── monitoring/
│           ├── blackbox/
│           │   └── blackbox.yml
│           └── prometheus/
│               └── prometheus.yml
├── migrations/
│   ├── clickhouse/
│   └── postgres/
├── scripts/
│   ├── data/
│   ├── local/
│   └── ops/
├── src/
│   └── trading/
│       ├── contexts/
│       │   ├── backtest/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── feeds/
│       │   │   │       │   └── market_data_acl/
│       │   │   │       ├── persistence/
│       │   │   │       │   ├── filesystem/
│       │   │   │       │   └── postgres/
│       │   │   │       └── progress/
│       │   │   │           ├── logs/
│       │   │   │           └── messaging/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   ├── errors/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── feeds/
│       │   │   │   │   ├── progress/
│       │   │   │   │   └── stores/
│       │   │   │   └── use_cases/
│       │   │   ├── domain/
│       │   │   │   ├── entities/
│       │   │   │   ├── errors/
│       │   │   │   ├── events/
│       │   │   │   ├── specifications/
│       │   │   │   └── value_objects/
│       │   │   └── __init__.py
│       │   ├── indicators/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── caching/
│       │   │   │       ├── compute_numba/
│       │   │   │       └── compute_numpy/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   ├── errors/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── cache/
│       │   │   │   │   ├── compute/
│       │   │   │   │   └── registry/
│       │   │   │   └── use_cases/
│       │   │   ├── domain/
│       │   │   │   ├── entities/
│       │   │   │   ├── errors/
│       │   │   │   ├── specifications/
│       │   │   │   └── value_objects/
│       │   │   └── __init__.py
│       │   ├── market_data/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── clients/
│       │   │   │       │   ├── binance/
│       │   │   │       │   │   ├── __init__.py
│       │   │   │       │   │   └── ws_client.py
│       │   │   │       │   ├── bybit/
│       │   │   │       │   │   ├── __init__.py
│       │   │   │       │   │   └── ws_client.py
│       │   │   │       │   ├── common_http/
│       │   │   │       │   │   ├── __init__.py
│       │   │   │       │   │   └── http_client.py
│       │   │   │       │   ├── files/
│       │   │   │       │   │   ├── __init__.py
│       │   │   │       │   │   └── parquet_candle_ingest_source.py
│       │   │   │       │   ├── __init__.py
│       │   │   │       │   ├── rest_candle_ingest_source.py
│       │   │   │       │   └── rest_instrument_metadata_source.py
│       │   │   │       ├── config/
│       │   │   │       │   ├── __init__.py
│       │   │   │       │   ├── instrument_key.py
│       │   │   │       │   ├── runtime_config.py
│       │   │   │       │   └── whitelist.py
│       │   │   │       ├── messaging/
│       │   │   │       │   ├── kafka/
│       │   │   │       │   ├── redis/
│       │   │   │       │   │   ├── __init__.py
│       │   │   │       │   │   ├── noop_live_candle_publisher.py
│       │   │   │       │   │   └── redis_streams_live_candle_publisher.py
│       │   │   │       │   └── __init__.py
│       │   │   │       └── persistence/
│       │   │   │           ├── cache/
│       │   │   │           ├── clickhouse/
│       │   │   │           │   ├── __init__.py
│       │   │   │           │   ├── canonical_candle_index_reader.py
│       │   │   │           │   ├── canonical_candle_reader.py
│       │   │   │           │   ├── enabled_instrument_reader.py
│       │   │   │           │   ├── gateway.py
│       │   │   │           │   ├── raw_kline_writer.py
│       │   │   │           │   ├── ref_instruments_writer.py
│       │   │   │           │   └── ref_market_writer.py
│       │   │   │           └── filesystem/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   │   ├── __init__.py
│       │   │   │   │   ├── backfill_1m_command.py
│       │   │   │   │   ├── backfill_1m_report.py
│       │   │   │   │   ├── candle_with_meta.py
│       │   │   │   │   ├── reference_data.py
│       │   │   │   │   └── rest_fill_task.py
│       │   │   │   ├── errors/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── clock/
│       │   │   │   │   │   ├── __init__.py
│       │   │   │   │   │   └── clock.py
│       │   │   │   │   ├── feeds/
│       │   │   │   │   │   ├── __init__.py
│       │   │   │   │   │   └── live_candle_publisher.py
│       │   │   │   │   ├── sources/
│       │   │   │   │   │   ├── __init__.py
│       │   │   │   │   │   ├── candle_ingest_source.py
│       │   │   │   │   │   └── instrument_metadata_source.py
│       │   │   │   │   ├── stores/
│       │   │   │   │   │   ├── __init__.py
│       │   │   │   │   │   ├── canonical_candle_index_reader.py
│       │   │   │   │   │   ├── canonical_candle_reader.py
│       │   │   │   │   │   ├── enabled_instrument_reader.py
│       │   │   │   │   │   ├── instrument_ref_writer.py
│       │   │   │   │   │   ├── market_ref_writer.py
│       │   │   │   │   │   └── raw_kline_writer.py
│       │   │   │   │   ├── tx/
│       │   │   │   │   └── __init__.py
│       │   │   │   ├── services/
│       │   │   │   │   ├── __init__.py
│       │   │   │   │   ├── gap_tracker.py
│       │   │   │   │   ├── insert_buffer.py
│       │   │   │   │   ├── minute_utils.py
│       │   │   │   │   ├── reconnect_tail_fill.py
│       │   │   │   │   ├── rest_fill_queue.py
│       │   │   │   │   └── scheduler_backfill_planner.py
│       │   │   │   └── use_cases/
│       │   │   │       ├── __init__.py
│       │   │   │       ├── backfill_1m_candles.py
│       │   │   │       ├── enrich_ref_instruments_from_exchange.py
│       │   │   │       ├── rest_catchup_1m.py
│       │   │   │       ├── rest_fill_range_1m.py
│       │   │   │       ├── seed_ref_market.py
│       │   │   │       ├── sync_whitelist_to_ref_instruments.py
│       │   │   │       └── time_slicing.py
│       │   │   ├── domain/
│       │   │   │   ├── contracts/
│       │   │   │   ├── entities/
│       │   │   │   ├── errors/
│       │   │   │   ├── events/
│       │   │   │   ├── specifications/
│       │   │   │   └── value_objects/
│       │   │   └── __init__.py
│       │   ├── ml/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── inference/
│       │   │   │       ├── persistence/
│       │   │   │       └── training/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── engines/
│       │   │   │   │   ├── registries/
│       │   │   │   │   └── stores/
│       │   │   │   └── use_cases/
│       │   │   ├── domain/
│       │   │   └── __init__.py
│       │   ├── optimize/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── persistence/
│       │   │   │       │   └── postgres/
│       │   │   │       └── queues/
│       │   │   │           ├── in_memory/
│       │   │   │           └── redis/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── queues/
│       │   │   │   │   ├── repositories/
│       │   │   │   │   └── stores/
│       │   │   │   └── use_cases/
│       │   │   ├── domain/
│       │   │   └── __init__.py
│       │   ├── risk/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── persistence/
│       │   │   │       │   └── postgres/
│       │   │   │       └── readers/
│       │   │   │           └── backtest_acl/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── readers/
│       │   │   │   │   └── stores/
│       │   │   │   └── use_cases/
│       │   │   ├── domain/
│       │   │   └── __init__.py
│       │   ├── strategy/
│       │   │   ├── adapters/
│       │   │   │   ├── inbound/
│       │   │   │   └── outbound/
│       │   │   │       ├── feeds/
│       │   │   │       │   └── market_data_acl/
│       │   │   │       ├── persistence/
│       │   │   │       │   └── postgres/
│       │   │   │       └── sinks/
│       │   │   │           ├── memory/
│       │   │   │           └── messaging/
│       │   │   ├── application/
│       │   │   │   ├── dto/
│       │   │   │   ├── errors/
│       │   │   │   ├── ports/
│       │   │   │   │   ├── feeds/
│       │   │   │   │   ├── repositories/
│       │   │   │   │   └── sinks/
│       │   │   │   └── use_cases/
│       │   │   ├── domain/
│       │   │   │   ├── entities/
│       │   │   │   ├── errors/
│       │   │   │   ├── events/
│       │   │   │   ├── specifications/
│       │   │   │   └── value_objects/
│       │   │   └── __init__.py
│       │   └── __init__.py
│       ├── fastpath/
│       │   ├── backtest/
│       │   ├── features/
│       │   ├── indicators/
│       │   └── __init__.py
│       ├── integration/
│       │   ├── acl/
│       │   │   ├── backtest_to_optimize/
│       │   │   │   ├── dto/
│       │   │   │   ├── mapping/
│       │   │   │   └── ranking_inputs_impl/
│       │   │   ├── market_data_to_backtest/
│       │   │   │   ├── candle_feed_impl/
│       │   │   │   ├── dto/
│       │   │   │   └── mapping/
│       │   │   ├── market_data_to_strategy/
│       │   │   │   ├── dto/
│       │   │   │   ├── feed_impl/
│       │   │   │   └── mapping/
│       │   │   └── risk_to_backtest/
│       │   │       ├── constraints_impl/
│       │   │       ├── dto/
│       │   │       └── mapping/
│       │   ├── orchestration/
│       │   └── __init__.py
│       ├── platform/
│       │   ├── config/
│       │   ├── observability/
│       │   ├── serialization/
│       │   ├── time/
│       │   │   └── system_clock.py
│       │   └── __init__.py
│       ├── shared_kernel/
│       │   ├── errors/
│       │   ├── primitives/
│       │   │   ├── __init__.py
│       │   │   ├── candle.py
│       │   │   ├── candle_meta.py
│       │   │   ├── instrument_id.py
│       │   │   ├── market_id.py
│       │   │   ├── symbol.py
│       │   │   ├── time_range.py
│       │   │   ├── timeframe.py
│       │   │   └── utc_timestamp.py
│       │   └── __init__.py
│       └── __init__.py
├── tests/
│   ├── integration/
│   ├── notebook_tests/
│   │   ├── 01_sync_instruments.ipynb
│   │   └── 02_rest_catchup_1m.ipynb
│   ├── perf_smoke/
│   ├── unit/
│   │   ├── contexts/
│   │   │   └── market_data/
│   │   │       ├── adapters/
│   │   │       │   ├── test_clickhouse_canonical_candle_index_reader.py
│   │   │       │   ├── test_clickhouse_canonical_candle_reader.py
│   │   │       │   ├── test_clickhouse_enabled_instrument_reader.py
│   │   │       │   ├── test_clickhouse_raw_kline_writer.py
│   │   │       │   ├── test_clickhouse_ref_instruments_writer.py
│   │   │       │   ├── test_clickhouse_thread_local_gateway.py
│   │   │       │   ├── test_market_data_runtime_config.py
│   │   │       │   ├── test_parquet_candle_ingest_source.py
│   │   │       │   ├── test_redis_streams_live_candle_publisher.py
│   │   │       │   ├── test_rest_candle_ingest_source.py
│   │   │       │   ├── test_rest_instrument_metadata_source.py
│   │   │       │   ├── test_whitelist_csv_loader.py
│   │   │       │   ├── test_ws_binance_client.py
│   │   │       │   └── test_ws_bybit_client.py
│   │   │       └── application/
│   │   │           ├── services/
│   │   │           │   ├── test_gap_tracker.py
│   │   │           │   ├── test_insert_buffer.py
│   │   │           │   ├── test_reconnect_tail_fill.py
│   │   │           │   ├── test_rest_fill_queue.py
│   │   │           │   ├── test_scheduler_backfill_planner.py
│   │   │           │   ├── test_scheduler_startup_scan.py
│   │   │           │   └── test_ws_worker_publishes_redis.py
│   │   │           └── use_cases/
│   │   │               ├── test_backfill_1m_candles.py
│   │   │               ├── test_enrich_ref_instruments_from_exchange.py
│   │   │               ├── test_reference_data_sync.py
│   │   │               ├── test_rest_catchup_1m.py
│   │   │               ├── test_rest_fill_range_1m.py
│   │   │               └── test_time_slicing.py
│   │   └── shared_kernel/
│   │       └── primitives/
│   │           ├── test_candle.py
│   │           ├── test_candle_meta.py
│   │           ├── test_instrument_id.py
│   │           ├── test_market_id.py
│   │           └── test_timeframe.py
│   └── test_smoke.py
├── tools/
│   ├── ci/
│   ├── format/
│   └── lint/
├── .DS_Store
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile.api
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock