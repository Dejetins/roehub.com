./
|-- .cache/
|   `-- numba/
|       `-- dev/
|-- .dockerignore
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       `-- deploy.yml
|-- .gitignore
|-- .python-version
|-- Dockerfile.api
|-- LICENSE
|-- README.md
|-- apps/
|   |-- __init__.py
|   |-- api/
|   |   |-- __init__.py
|   |   |-- dto/
|   |   |   |-- __init__.py
|   |   |   `-- indicators.py
|   |   |-- main/
|   |   |   |-- __init__.py
|   |   |   |-- app.py
|   |   |   `-- main.py
|   |   |-- routes/
|   |   |   |-- __init__.py
|   |   |   |-- identity.py
|   |   |   `-- indicators.py
|   |   `-- wiring/
|   |       |-- __init__.py
|   |       |-- clients/
|   |       |-- container/
|   |       |-- db/
|   |       `-- modules/
|   |           |-- __init__.py
|   |           |-- identity.py
|   |           `-- indicators.py
|   |-- cli/
|   |   |-- __init__.py
|   |   |-- commands/
|   |   |   |-- __init__.py
|   |   |   |-- backfill_1m.py
|   |   |   |-- rest_catchup_1m.py
|   |   |   `-- sync_instruments.py
|   |   |-- main/
|   |   |   |-- __init__.py
|   |   |   `-- main.py
|   |   |-- test_backfill_1m_parsing.py
|   |   `-- wiring/
|   |       |-- __init__.py
|   |       |-- clients/
|   |       |   |-- __init__.py
|   |       |   `-- parquet.py
|   |       |-- container/
|   |       |-- db/
|   |       |   |-- __init__.py
|   |       |   `-- clickhouse.py
|   |       `-- modules/
|   |           |-- __init__.py
|   |           `-- market_data.py
|   |-- scheduler/
|   |   |-- main/
|   |   |-- market_data_scheduler/
|   |   |   |-- __init__.py
|   |   |   |-- main/
|   |   |   |   |-- __init__.py
|   |   |   |   `-- main.py
|   |   |   `-- wiring/
|   |   |       |-- __init__.py
|   |   |       `-- modules/
|   |   |           |-- __init__.py
|   |   |           `-- market_data_scheduler.py
|   |   `-- wiring/
|   |       |-- clients/
|   |       |-- container/
|   |       |-- db/
|   |       `-- modules/
|   `-- worker/
|       |-- handlers/
|       |-- main/
|       |-- market_data_ws/
|       |   |-- __init__.py
|       |   |-- main/
|       |   |   |-- __init__.py
|       |   |   `-- main.py
|       |   `-- wiring/
|       |       |-- __init__.py
|       |       `-- modules/
|       |           |-- __init__.py
|       |           `-- market_data_ws.py
|       `-- wiring/
|           |-- clients/
|           |-- container/
|           |-- db/
|           `-- modules/
|-- configs/
|   |-- dev/
|   |   |-- indicators.yaml
|   |   |-- market_data.yaml
|   |   `-- whitelist.csv
|   |-- prod/
|   |   |-- indicators.yaml
|   |   |-- market_data.yaml
|   |   `-- whitelist.csv
|   `-- test/
|       `-- indicators.yaml
|-- deploy/
|-- docs/
|   |-- _templates/
|   |   `-- architecture-doc-template.md
|   |-- api/
|   |-- architecture/
|   |   |-- README.md
|   |   |-- apps/
|   |   |   `-- cli/
|   |   |       `-- cli-backfill-1m.md
|   |   |-- identity/
|   |   |   |-- identity-2fa-totp-policy-v1.md
|   |   |   `-- identity-telegram-login-user-model-v1.md
|   |   |-- indicators/
|   |   |   |-- README.md
|   |   |   |-- indicators-application-ports-walking-skeleton-v1.md
|   |   |   |-- indicators-candlefeed-acl-dense-timeline-v1.md
|   |   |   |-- indicators-compute-engine-core.md
|   |   |   |-- indicators-grid-builder-estimate-guards-v1.md
|   |   |   |-- indicators-ma-compute-numba-v1.md
|   |   |   |-- indicators-ma.md
|   |   |   |-- indicators-momentum.md
|   |   |   |-- indicators-overview.md
|   |   |   |-- indicators-registry-yaml-defaults-v1.md
|   |   |   |-- indicators-structure-normalization-compute-numba-v1.md
|   |   |   |-- indicators-structure.md
|   |   |   |-- indicators-trend-volume-compute-numba-v1.md
|   |   |   |-- indicators-trend.md
|   |   |   |-- indicators-volatility-momentum-compute-numba-v1.md
|   |   |   |-- indicators-volatility.md
|   |   |   |-- indicators-volume.md
|   |   |   `-- indicators_formula.yaml
|   |   |-- market_data/
|   |   |   |-- market-data-application-ports.md
|   |   |   |-- market-data-live-feed-redis-streams-v1.md
|   |   |   |-- market-data-real-adapters-clickhouse-parquet.md
|   |   |   |-- market-data-reference-data-sync-v2.md
|   |   |   |-- market-data-rest-historical-catchup-1m-v2.md
|   |   |   |-- market-data-runtime-config-invariants-v2.md
|   |   |   |-- market-data-use-case-backfill-1m.md
|   |   |   `-- market-data-ws-live-ingestion-worker-v1.md
|   |   |-- roadmap/
|   |   |   |-- base_milestone_plan.md
|   |   |   |-- milestone-2-epics-v1.md
|   |   |   `-- milestone-3-epics-v1.md
|   |   |-- shared-kernel-primitives.md
|   |   `-- strategy/
|   |       `-- strategy-milestone-3-epics-v1.md
|   |-- decisions/
|   |-- repository_three.md
|   `-- runbooks/
|       |-- help_commands.md
|       |-- indicators-numba-cache-and-threads.md
|       |-- indicators-numba-warmup-jit.md
|       |-- indicators-why-nan.md
|       |-- market-data-autonomous-docker.md
|       |-- market-data-metrics-reference-ru.md
|       |-- market-data-metrics.md
|       `-- market-data-redis-streams.md
|-- infra/
|   |-- docker/
|   |   |-- .env.example
|   |   |-- Dockerfile.market_data
|   |   |-- docker-compose.market_data.yml
|   |   `-- docker-compose.yml
|   |-- k8s/
|   `-- monitoring/
|       `-- monitoring/
|           |-- blackbox/
|           |   `-- blackbox.yml
|           `-- prometheus/
|               `-- prometheus.yml
|-- migrations/
|   |-- clickhouse/
|   |   `-- market_data_ddl.sql
|   `-- postgres/
|       |-- 0001_identity_v1.sql
|       `-- 0002_identity_2fa_totp_v1.sql
|-- pyproject.toml
|-- repo_tree.md
|-- scripts/
|   |-- data/
|   |-- local/
|   `-- ops/
|       `-- optimize_canonical_partitions.sh*
|-- src/
|   `-- trading/
|       |-- __init__.py
|       |-- contexts/
|       |   |-- __init__.py
|       |   |-- backtest/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- feeds/
|       |   |   |       |   `-- market_data_acl/
|       |   |   |       |-- persistence/
|       |   |   |       |   |-- filesystem/
|       |   |   |       |   `-- postgres/
|       |   |   |       `-- progress/
|       |   |   |           |-- logs/
|       |   |   |           `-- messaging/
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |-- errors/
|       |   |   |   |-- ports/
|       |   |   |   |   |-- feeds/
|       |   |   |   |   |-- progress/
|       |   |   |   |   `-- stores/
|       |   |   |   `-- use_cases/
|       |   |   `-- domain/
|       |   |       |-- entities/
|       |   |       |-- errors/
|       |   |       |-- events/
|       |   |       |-- specifications/
|       |   |       `-- value_objects/
|       |   |-- identity/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- __init__.py
|       |   |   |   |-- inbound/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   `-- api/
|       |   |   |   |       |-- __init__.py
|       |   |   |   |       |-- deps/
|       |   |   |   |       |   |-- __init__.py
|       |   |   |   |       |   |-- current_user.py
|       |   |   |   |       |   `-- two_factor_enabled.py
|       |   |   |   |       `-- routes/
|       |   |   |   |           |-- __init__.py
|       |   |   |   |           |-- auth_telegram.py
|       |   |   |   |           `-- two_factor_totp.py
|       |   |   |   `-- outbound/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- persistence/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- in_memory/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   |-- two_factor_repository.py
|       |   |   |       |   |   `-- user_repository.py
|       |   |   |       |   `-- postgres/
|       |   |   |       |       |-- __init__.py
|       |   |   |       |       |-- gateway.py
|       |   |   |       |       |-- two_factor_repository.py
|       |   |   |       |       `-- user_repository.py
|       |   |   |       |-- policy/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   `-- two_factor_policy_gate.py
|       |   |   |       |-- security/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- current_user/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- jwt_cookie_current_user.py
|       |   |   |       |   |-- jwt/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- hs256_jwt_codec.py
|       |   |   |       |   |-- telegram/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- telegram_login_widget_payload_validator.py
|       |   |   |       |   `-- two_factor/
|       |   |   |       |       |-- __init__.py
|       |   |   |       |       |-- aes_gcm_envelope_secret_cipher.py
|       |   |   |       |       `-- pyotp_totp_provider.py
|       |   |   |       `-- time/
|       |   |   |           |-- __init__.py
|       |   |   |           `-- system_identity_clock.py
|       |   |   |-- application/
|       |   |   |   |-- __init__.py
|       |   |   |   |-- ports/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- clock.py
|       |   |   |   |   |-- current_user.py
|       |   |   |   |   |-- jwt_codec.py
|       |   |   |   |   |-- telegram_auth_payload_validator.py
|       |   |   |   |   |-- two_factor_policy_gate.py
|       |   |   |   |   |-- two_factor_repository.py
|       |   |   |   |   |-- two_factor_secret_cipher.py
|       |   |   |   |   |-- two_factor_totp_provider.py
|       |   |   |   |   `-- user_repository.py
|       |   |   |   `-- use_cases/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- setup_two_factor_totp.py
|       |   |   |       |-- telegram_login.py
|       |   |   |       |-- two_factor_errors.py
|       |   |   |       `-- verify_two_factor_totp.py
|       |   |   `-- domain/
|       |   |       |-- __init__.py
|       |   |       |-- entities/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- two_factor_auth.py
|       |   |       |   `-- user.py
|       |   |       `-- value_objects/
|       |   |           |-- __init__.py
|       |   |           |-- telegram_chat_id.py
|       |   |           `-- telegram_user_id.py
|       |   |-- indicators/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- __init__.py
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- caching/
|       |   |   |       |-- compute_numba/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- engine.py
|       |   |   |       |   |-- kernels/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   |-- _common.py
|       |   |   |       |   |   |-- ma.py
|       |   |   |       |   |   |-- momentum.py
|       |   |   |       |   |   |-- structure.py
|       |   |   |       |   |   |-- trend.py
|       |   |   |       |   |   |-- volatility.py
|       |   |   |       |   |   `-- volume.py
|       |   |   |       |   `-- warmup.py
|       |   |   |       |-- compute_numpy/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- ma.py
|       |   |   |       |   |-- momentum.py
|       |   |   |       |   |-- structure.py
|       |   |   |       |   |-- trend.py
|       |   |   |       |   |-- volatility.py
|       |   |   |       |   `-- volume.py
|       |   |   |       |-- config/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- yaml_defaults_loader.py
|       |   |   |       |   `-- yaml_defaults_validator.py
|       |   |   |       |-- feeds/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   `-- market_data_acl/
|       |   |   |       |       |-- __init__.py
|       |   |   |       |       `-- market_data_candle_feed.py
|       |   |   |       `-- registry/
|       |   |   |           |-- __init__.py
|       |   |   |           `-- yaml_indicator_registry.py
|       |   |   |-- application/
|       |   |   |   |-- __init__.py
|       |   |   |   |-- dto/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- candle_arrays.py
|       |   |   |   |   |-- compute_request.py
|       |   |   |   |   |-- estimate_result.py
|       |   |   |   |   |-- grid.py
|       |   |   |   |   |-- indicator_tensor.py
|       |   |   |   |   |-- registry_view.py
|       |   |   |   |   `-- variant_key.py
|       |   |   |   |-- errors/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- memory_guard_exceeded.py
|       |   |   |   |   `-- variants_guard_exceeded.py
|       |   |   |   |-- ports/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- cache/
|       |   |   |   |   |-- compute/
|       |   |   |   |   |   |-- __init__.py
|       |   |   |   |   |   `-- indicator_compute.py
|       |   |   |   |   |-- feeds/
|       |   |   |   |   |   |-- __init__.py
|       |   |   |   |   |   `-- candle_feed.py
|       |   |   |   |   `-- registry/
|       |   |   |   |       |-- __init__.py
|       |   |   |   |       `-- indicator_registry.py
|       |   |   |   |-- services/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   `-- grid_builder.py
|       |   |   |   `-- use_cases/
|       |   |   `-- domain/
|       |   |       |-- __init__.py
|       |   |       |-- definitions/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- ma.py
|       |   |       |   |-- momentum.py
|       |   |       |   |-- structure.py
|       |   |       |   |-- trend.py
|       |   |       |   |-- volatility.py
|       |   |       |   `-- volume.py
|       |   |       |-- entities/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- axis_def.py
|       |   |       |   |-- indicator_def.py
|       |   |       |   |-- indicator_id.py
|       |   |       |   |-- input_series.py
|       |   |       |   |-- layout.py
|       |   |       |   |-- output_spec.py
|       |   |       |   |-- param_def.py
|       |   |       |   `-- param_kind.py
|       |   |       |-- errors/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- compute_budget_exceeded.py
|       |   |       |   |-- grid_validation_error.py
|       |   |       |   |-- missing_input_series_error.py
|       |   |       |   |-- missing_required_series.py
|       |   |       |   `-- unknown_indicator_error.py
|       |   |       |-- specifications/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- grid_param_spec.py
|       |   |       |   `-- grid_spec.py
|       |   |       `-- value_objects/
|       |   |-- market_data/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- clients/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- binance/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- ws_client.py
|       |   |   |       |   |-- bybit/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- ws_client.py
|       |   |   |       |   |-- common_http/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- http_client.py
|       |   |   |       |   |-- files/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- parquet_candle_ingest_source.py
|       |   |   |       |   |-- rest_candle_ingest_source.py
|       |   |   |       |   `-- rest_instrument_metadata_source.py
|       |   |   |       |-- config/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- instrument_key.py
|       |   |   |       |   |-- runtime_config.py
|       |   |   |       |   `-- whitelist.py
|       |   |   |       |-- messaging/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- kafka/
|       |   |   |       |   `-- redis/
|       |   |   |       |       |-- __init__.py
|       |   |   |       |       |-- noop_live_candle_publisher.py
|       |   |   |       |       `-- redis_streams_live_candle_publisher.py
|       |   |   |       `-- persistence/
|       |   |   |           |-- cache/
|       |   |   |           |-- clickhouse/
|       |   |   |           |   |-- __init__.py
|       |   |   |           |   |-- canonical_candle_index_reader.py
|       |   |   |           |   |-- canonical_candle_reader.py
|       |   |   |           |   |-- enabled_instrument_reader.py
|       |   |   |           |   |-- gateway.py
|       |   |   |           |   |-- raw_kline_writer.py
|       |   |   |           |   |-- ref_instruments_writer.py
|       |   |   |           |   `-- ref_market_writer.py
|       |   |   |           `-- filesystem/
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- backfill_1m_command.py
|       |   |   |   |   |-- backfill_1m_report.py
|       |   |   |   |   |-- candle_with_meta.py
|       |   |   |   |   |-- reference_data.py
|       |   |   |   |   `-- rest_fill_task.py
|       |   |   |   |-- errors/
|       |   |   |   |-- ports/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- clock/
|       |   |   |   |   |   |-- __init__.py
|       |   |   |   |   |   `-- clock.py
|       |   |   |   |   |-- feeds/
|       |   |   |   |   |   |-- __init__.py
|       |   |   |   |   |   `-- live_candle_publisher.py
|       |   |   |   |   |-- sources/
|       |   |   |   |   |   |-- __init__.py
|       |   |   |   |   |   |-- candle_ingest_source.py
|       |   |   |   |   |   `-- instrument_metadata_source.py
|       |   |   |   |   |-- stores/
|       |   |   |   |   |   |-- __init__.py
|       |   |   |   |   |   |-- canonical_candle_index_reader.py
|       |   |   |   |   |   |-- canonical_candle_reader.py
|       |   |   |   |   |   |-- enabled_instrument_reader.py
|       |   |   |   |   |   |-- instrument_ref_writer.py
|       |   |   |   |   |   |-- market_ref_writer.py
|       |   |   |   |   |   `-- raw_kline_writer.py
|       |   |   |   |   `-- tx/
|       |   |   |   |-- services/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- gap_tracker.py
|       |   |   |   |   |-- insert_buffer.py
|       |   |   |   |   |-- minute_utils.py
|       |   |   |   |   |-- reconnect_tail_fill.py
|       |   |   |   |   |-- rest_fill_queue.py
|       |   |   |   |   `-- scheduler_backfill_planner.py
|       |   |   |   `-- use_cases/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- backfill_1m_candles.py
|       |   |   |       |-- enrich_ref_instruments_from_exchange.py
|       |   |   |       |-- rest_catchup_1m.py
|       |   |   |       |-- rest_fill_range_1m.py
|       |   |   |       |-- seed_ref_market.py
|       |   |   |       |-- sync_whitelist_to_ref_instruments.py
|       |   |   |       `-- time_slicing.py
|       |   |   `-- domain/
|       |   |       |-- contracts/
|       |   |       |-- entities/
|       |   |       |-- errors/
|       |   |       |-- events/
|       |   |       |-- specifications/
|       |   |       `-- value_objects/
|       |   |-- ml/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- inference/
|       |   |   |       |-- persistence/
|       |   |   |       `-- training/
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |-- ports/
|       |   |   |   |   |-- engines/
|       |   |   |   |   |-- registries/
|       |   |   |   |   `-- stores/
|       |   |   |   `-- use_cases/
|       |   |   `-- domain/
|       |   |-- optimize/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- persistence/
|       |   |   |       |   `-- postgres/
|       |   |   |       `-- queues/
|       |   |   |           |-- in_memory/
|       |   |   |           `-- redis/
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |-- ports/
|       |   |   |   |   |-- queues/
|       |   |   |   |   |-- repositories/
|       |   |   |   |   `-- stores/
|       |   |   |   `-- use_cases/
|       |   |   `-- domain/
|       |   |-- risk/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- persistence/
|       |   |   |       |   `-- postgres/
|       |   |   |       `-- readers/
|       |   |   |           `-- backtest_acl/
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |-- ports/
|       |   |   |   |   |-- readers/
|       |   |   |   |   `-- stores/
|       |   |   |   `-- use_cases/
|       |   |   `-- domain/
|       |   `-- strategy/
|       |       |-- __init__.py
|       |       |-- adapters/
|       |       |   |-- inbound/
|       |       |   `-- outbound/
|       |       |       |-- feeds/
|       |       |       |   `-- market_data_acl/
|       |       |       |-- persistence/
|       |       |       |   `-- postgres/
|       |       |       `-- sinks/
|       |       |           |-- memory/
|       |       |           `-- messaging/
|       |       |-- application/
|       |       |   |-- dto/
|       |       |   |-- errors/
|       |       |   |-- ports/
|       |       |   |   |-- feeds/
|       |       |   |   |-- repositories/
|       |       |   |   `-- sinks/
|       |       |   `-- use_cases/
|       |       `-- domain/
|       |           |-- entities/
|       |           |-- errors/
|       |           |-- events/
|       |           |-- specifications/
|       |           `-- value_objects/
|       |-- fastpath/
|       |   |-- __init__.py
|       |   |-- backtest/
|       |   |-- features/
|       |   `-- indicators/
|       |-- integration/
|       |   |-- __init__.py
|       |   |-- acl/
|       |   |   |-- backtest_to_optimize/
|       |   |   |   |-- dto/
|       |   |   |   |-- mapping/
|       |   |   |   `-- ranking_inputs_impl/
|       |   |   |-- market_data_to_backtest/
|       |   |   |   |-- candle_feed_impl/
|       |   |   |   |-- dto/
|       |   |   |   `-- mapping/
|       |   |   |-- market_data_to_strategy/
|       |   |   |   |-- dto/
|       |   |   |   |-- feed_impl/
|       |   |   |   `-- mapping/
|       |   |   `-- risk_to_backtest/
|       |   |       |-- constraints_impl/
|       |   |       |-- dto/
|       |   |       `-- mapping/
|       |   `-- orchestration/
|       |-- platform/
|       |   |-- __init__.py
|       |   |-- config/
|       |   |   |-- __init__.py
|       |   |   `-- indicators_compute_numba.py
|       |   |-- observability/
|       |   |-- serialization/
|       |   `-- time/
|       |       `-- system_clock.py
|       `-- shared_kernel/
|           |-- __init__.py
|           |-- errors/
|           `-- primitives/
|               |-- __init__.py
|               |-- candle.py
|               |-- candle_meta.py
|               |-- instrument_id.py
|               |-- market_id.py
|               |-- paid_level.py
|               |-- symbol.py
|               |-- time_range.py
|               |-- timeframe.py
|               |-- user_id.py
|               `-- utc_timestamp.py
|-- tests/
|   |-- integration/
|   |-- notebook_tests/
|   |   |-- 01_sync_instruments.ipynb
|   |   `-- 02_rest_catchup_1m.ipynb
|   |-- perf_smoke/
|   |   `-- contexts/
|   |       `-- indicators/
|   |           |-- test_compute_numba_perf_smoke.py
|   |           |-- test_indicators_ma.py
|   |           |-- test_indicators_structure.py
|   |           |-- test_indicators_trend_volume.py
|   |           `-- test_indicators_vol_mom.py
|   |-- test_smoke.py
|   `-- unit/
|       |-- apps/
|       |   |-- api/
|       |   |   |-- test_identity_routes.py
|       |   |   |-- test_identity_two_factor_gate_dependency.py
|       |   |   |-- test_identity_two_factor_routes.py
|       |   |   |-- test_identity_wiring_module.py
|       |   |   `-- wiring/
|       |   |       `-- modules/
|       |   `-- cli/
|       |       `-- commands/
|       |           `-- test_rest_catchup_1m_cli.py
|       |-- contexts/
|       |   |-- identity/
|       |   |   |-- adapters/
|       |   |   |   `-- outbound/
|       |   |   |       `-- security/
|       |   |   |           `-- test_telegram_login_widget_payload_validator.py
|       |   |   `-- application/
|       |   |       |-- test_telegram_login_use_case.py
|       |   |       `-- test_two_factor_totp_use_cases.py
|       |   |-- indicators/
|       |   |   |-- adapters/
|       |   |   |   `-- outbound/
|       |   |   |       |-- compute_numba/
|       |   |   |       |   |-- test_common_kernels.py
|       |   |   |       |   |-- test_engine.py
|       |   |   |       |   |-- test_ma_kernels.py
|       |   |   |       |   |-- test_momentum_kernels.py
|       |   |   |       |   |-- test_runtime_wiring.py
|       |   |   |       |   |-- test_structure_kernels.py
|       |   |   |       |   |-- test_trend_kernels.py
|       |   |   |       |   |-- test_volatility_kernels.py
|       |   |   |       |   `-- test_volume_kernels.py
|       |   |   |       |-- compute_numpy/
|       |   |   |       |   |-- test_ma_oracle.py
|       |   |   |       |   |-- test_momentum_oracle.py
|       |   |   |       |   |-- test_structure_oracle.py
|       |   |   |       |   |-- test_trend_oracle.py
|       |   |   |       |   |-- test_volatility_oracle.py
|       |   |   |       |   `-- test_volume_oracle.py
|       |   |   |       |-- config/
|       |   |   |       |   `-- test_yaml_defaults_validator.py
|       |   |   |       |-- feeds/
|       |   |   |       |   `-- test_market_data_acl_candle_feed.py
|       |   |   |       `-- registry/
|       |   |   |           `-- test_yaml_indicator_registry.py
|       |   |   |-- api/
|       |   |   |   |-- test_indicators_compute.py
|       |   |   |   `-- test_indicators_estimate.py
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |   |-- test_candle_arrays_invariants.py
|       |   |   |   |   `-- test_variant_key.py
|       |   |   |   `-- services/
|       |   |   |       `-- test_grid_builder.py
|       |   |   `-- domain/
|       |   |       |-- test_axis_def_oneof_values.py
|       |   |       |-- test_definitions_baseline.py
|       |   |       |-- test_grid_param_spec_shapes.py
|       |   |       |-- test_indicator_def_consistency.py
|       |   |       `-- test_param_def_invariants.py
|       |   `-- market_data/
|       |       |-- adapters/
|       |       |   |-- test_clickhouse_canonical_candle_index_reader.py
|       |       |   |-- test_clickhouse_canonical_candle_reader.py
|       |       |   |-- test_clickhouse_enabled_instrument_reader.py
|       |       |   |-- test_clickhouse_raw_kline_writer.py
|       |       |   |-- test_clickhouse_ref_instruments_writer.py
|       |       |   |-- test_clickhouse_thread_local_gateway.py
|       |       |   |-- test_market_data_runtime_config.py
|       |       |   |-- test_parquet_candle_ingest_source.py
|       |       |   |-- test_redis_streams_live_candle_publisher.py
|       |       |   |-- test_rest_candle_ingest_source.py
|       |       |   |-- test_rest_instrument_metadata_source.py
|       |       |   |-- test_whitelist_csv_loader.py
|       |       |   |-- test_ws_binance_client.py
|       |       |   `-- test_ws_bybit_client.py
|       |       `-- application/
|       |           |-- services/
|       |           |   |-- test_gap_tracker.py
|       |           |   |-- test_insert_buffer.py
|       |           |   |-- test_reconnect_tail_fill.py
|       |           |   |-- test_rest_fill_queue.py
|       |           |   |-- test_scheduler_backfill_planner.py
|       |           |   |-- test_scheduler_startup_scan.py
|       |           |   `-- test_ws_worker_publishes_redis.py
|       |           `-- use_cases/
|       |               |-- test_backfill_1m_candles.py
|       |               |-- test_enrich_ref_instruments_from_exchange.py
|       |               |-- test_reference_data_sync.py
|       |               |-- test_rest_catchup_1m.py
|       |               |-- test_rest_fill_range_1m.py
|       |               `-- test_time_slicing.py
|       |-- platform/
|       |   `-- config/
|       |       `-- test_indicators_compute_numba_config.py
|       |-- shared_kernel/
|       |   `-- primitives/
|       |       |-- test_candle.py
|       |       |-- test_candle_meta.py
|       |       |-- test_instrument_id.py
|       |       |-- test_market_id.py
|       |       |-- test_paid_level.py
|       |       |-- test_timeframe.py
|       |       `-- test_user_id.py
|       `-- tools/
|           `-- test_generate_docs_index.py
|-- tools/
|   |-- ci/
|   |-- docs/
|   |   `-- generate_docs_index.py
|   |-- format/
|   `-- lint/
`-- uv.lock

344 directories, 416 files
