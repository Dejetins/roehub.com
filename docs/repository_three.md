./
|-- .cache/
|   |-- numba/
|   |   `-- dev/
|   `-- uv/
|-- .dockerignore
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       `-- publish-app-image.yml
|-- .gitignore
|-- .opencode/
|   `-- openai-codex-auth-config.json
|-- .python-version
|-- .uv-cache/
|-- Dockerfile.api
|-- LICENSE
|-- README.md
|-- alembic/
|   |-- env.py
|   |-- script.py.mako
|   `-- versions/
|       |-- 20260215_0001_strategy_storage_v1.py
|       |-- 20260216_0002_strategy_run_metadata_v1.py
|       |-- 20260222_0003_backtest_jobs_v1.py
|       |-- 20260326_0004_backtest_job_artifact_pin_v1.py
|       |-- 20260329_0005_backtest_persisted_run_storage_v1.py
|       |-- 20260411_0006_backtest_execution_profile_metadata_v1.py
|       |-- 20260414_0007_backtest_stage_a_no_risk_exact_rows_v1.py
|       |-- 20260418_0008_backtest_stage_a_parity_runtime_state_v1.py
|       `-- 20260418_0009_backtest_execution_profile_metadata_parity_v1.py
|-- alembic.ini
|-- apps/
|   |-- __init__.py
|   |-- api/
|   |   |-- __init__.py
|   |   |-- common/
|   |   |   |-- __init__.py
|   |   |   `-- errors.py
|   |   |-- dto/
|   |   |   |-- __init__.py
|   |   |   |-- backtest_jobs.py
|   |   |   |-- backtest_runs.py
|   |   |   |-- backtest_runtime_defaults.py
|   |   |   |-- backtests.py
|   |   |   |-- indicators.py
|   |   |   `-- market_data_reference.py
|   |   |-- main/
|   |   |   |-- __init__.py
|   |   |   |-- app.py
|   |   |   `-- main.py
|   |   |-- monitoring.py
|   |   |-- routes/
|   |   |   |-- __init__.py
|   |   |   |-- backtest_jobs.py
|   |   |   |-- backtest_runs.py
|   |   |   |-- backtests.py
|   |   |   |-- identity.py
|   |   |   |-- indicators.py
|   |   |   |-- market_data_reference.py
|   |   |   |-- operations.py
|   |   |   `-- strategies.py
|   |   `-- wiring/
|   |       |-- __init__.py
|   |       |-- clients/
|   |       |-- container/
|   |       |-- db/
|   |       `-- modules/
|   |           |-- __init__.py
|   |           |-- backtest.py
|   |           |-- identity.py
|   |           |-- indicators.py
|   |           |-- market_data_reference.py
|   |           `-- strategy.py
|   |-- cli/
|   |   |-- __init__.py
|   |   |-- commands/
|   |   |   |-- __init__.py
|   |   |   |-- backfill_1m.py
|   |   |   |-- backtest_artifact_publish.py
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
|   |-- migrations/
|   |   |-- __init__.py
|   |   |-- bootstrap.py
|   |   |-- bootstrap_main.py
|   |   `-- main.py
|   |-- monitoring/
|   |   |-- __init__.py
|   |   `-- clickhouse_exporter.py
|   |-- scheduler/
|   |   |-- backtest_artifact_publisher/
|   |   |   |-- __init__.py
|   |   |   |-- main/
|   |   |   |   |-- __init__.py
|   |   |   |   `-- main.py
|   |   |   `-- wiring/
|   |   |       |-- __init__.py
|   |   |       `-- modules/
|   |   |           |-- __init__.py
|   |   |           `-- backtest_artifact_publisher.py
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
|   |-- web/
|   |   |-- __init__.py
|   |   |-- main/
|   |   |   |-- __init__.py
|   |   |   |-- api_client.py
|   |   |   |-- app.py
|   |   |   |-- main.py
|   |   |   |-- security.py
|   |   |   `-- settings.py
|   |   `-- templates/
|   |       |-- backtest_history.html
|   |       |-- backtest_job_details.html
|   |       |-- backtest_jobs_list.html
|   |       |-- backtest_run_summary.html
|   |       |-- backtest_variant_detail.html
|   |       |-- backtests.html
|   |       |-- base.html
|   |       |-- landing.html
|   |       |-- login.html
|   |       |-- logout.html
|   |       |-- partials/
|   |       |   `-- user_badge.html
|   |       |-- protected_page.html
|   |       |-- strategies_list.html
|   |       |-- strategy_builder.html
|   |       `-- strategy_details.html
|   `-- worker/
|       |-- backtest_job_runner/
|       |   |-- __init__.py
|       |   |-- main/
|       |   |   |-- __init__.py
|       |   |   `-- main.py
|       |   `-- wiring/
|       |       |-- __init__.py
|       |       `-- modules/
|       |           |-- __init__.py
|       |           `-- backtest_job_runner.py
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
|       |-- strategy_live_runner/
|       |   |-- __init__.py
|       |   |-- main/
|       |   |   |-- __init__.py
|       |   |   `-- main.py
|       |   `-- wiring/
|       |       |-- __init__.py
|       |       `-- modules/
|       |           |-- __init__.py
|       |           `-- strategy_live_runner.py
|       `-- wiring/
|           |-- clients/
|           |-- container/
|           |-- db/
|           `-- modules/
|-- configs/
|   |-- dev/
|   |   |-- backtest.yaml
|   |   |-- backtest_artifacts.yaml
|   |   |-- indicators.yaml
|   |   |-- market_data.yaml
|   |   |-- strategy.yaml
|   |   |-- strategy_live_runner.yaml
|   |   `-- whitelist.csv
|   |-- prod/
|   |   |-- backtest.yaml
|   |   |-- backtest_artifacts.yaml
|   |   |-- indicators.yaml
|   |   |-- market_data.yaml
|   |   |-- strategy.yaml
|   |   |-- strategy_live_runner.yaml
|   |   `-- whitelist.csv
|   `-- test/
|       |-- backtest.yaml
|       |-- backtest_artifacts.yaml
|       |-- indicators.yaml
|       |-- market_data.yaml
|       |-- strategy.yaml
|       |-- strategy_live_runner.yaml
|       `-- whitelist.csv
|-- deploy/
|-- docs/
|   |-- INDEX.md
|   |-- _templates/
|   |   `-- architecture-doc-template.md
|   |-- api/
|   |-- architecture/
|   |   |-- README.md
|   |   |-- api/
|   |   |   `-- api-errors-and-422-payload-v1.md
|   |   |-- apps/
|   |   |   |-- cli/
|   |   |   |   `-- cli-backfill-1m.md
|   |   |   |-- gateway/
|   |   |   |   `-- nginx-gateway-same-origin-ui-api-v1.md
|   |   |   `-- web/
|   |   |       |-- web-backtest-history-and-variant-detail-v2.md
|   |   |       |-- web-backtest-jobs-ui-async-v1.md
|   |   |       |-- web-backtest-runtime-defaults-endpoint-v1.md
|   |   |       |-- web-backtest-sync-ui-preflight-save-variant-v1.md
|   |   |       |-- web-strategy-ui-crud-builder-delete-v1.md
|   |   |       |-- web-ui-skeleton-ssr-htmx-auth-v1.md
|   |   |       `-- web-ui-tests-docs-index-v1.md
|   |   |-- backtest/
|   |   |   |-- README.md
|   |   |   |-- backtest-core-refactor-prompt-pack-v1.md
|   |   |   `-- deep-research-report.md
|   |   |-- identity/
|   |   |   |-- identity-2fa-totp-policy-v1.md
|   |   |   |-- identity-exchange-keys-storage-2fa-gate-policy-v1.md
|   |   |   |-- identity-exchange-keys-storage-2fa-gate-policy-v2.md
|   |   |   |-- identity-keycloak-auth-model-v1.md
|   |   |   |-- identity-telegram-login-user-model-v1.md
|   |   |   |-- keycloak-cutover-plan-v1.md
|   |   |   `-- keycloak-cutover-restart-prompt-pack-v1.md
|   |   |-- indicators/
|   |   |   |-- README.md
|   |   |   |-- indicators-application-ports-walking-skeleton-v1.md
|   |   |   |-- indicators-candlefeed-acl-dense-timeline-v1.md
|   |   |   |-- indicators-compute-engine-core.md
|   |   |   |-- indicators-grid-builder-estimate-guards-v1.md
|   |   |   |-- indicators-grid-compute-perf-optimization-plan-v1.md
|   |   |   |-- indicators-kernels-f32-migration-plan-v1.md
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
|   |   |   |-- market-data-reference-api-v1.md
|   |   |   |-- market-data-reference-data-sync-v2.md
|   |   |   |-- market-data-rest-historical-catchup-1m-v2.md
|   |   |   |-- market-data-runtime-config-invariants-v2.md
|   |   |   |-- market-data-use-case-backfill-1m.md
|   |   |   `-- market-data-ws-live-ingestion-worker-v1.md
|   |   |-- operations/
|   |   |   `-- native-service-control-monitoring-admin-target-v1.md
|   |   |-- roadmap/
|   |   |   |-- backtest-engine-vnext-implementation-plan-v1.md
|   |   |   |-- backtest-engine-vnext-notebook-parity-plan-v1.md
|   |   |   |-- backtest-engine-vnext-parity-corrective-plan-v1.md
|   |   |   |-- backtest-engine-vnext-parity-corrective-plan-v2.md
|   |   |   |-- backtest-job-runner-v2-implementation-plan.md
|   |   |   |-- backtest-refactor-final-plan-v2.md
|   |   |   |-- backtest-runtime-acceleration-plan-v1.md
|   |   |   |-- base_milestone_plan.md
|   |   |   |-- milestone-2-epics-v1.md
|   |   |   |-- milestone-3-epics-v1.md
|   |   |   |-- milestone-4-epics-v1.md
|   |   |   |-- milestone-5-epics-v1.md
|   |   |   `-- milestone-6-epics-v1.md
|   |   |-- shared-kernel-primitives.md
|   |   `-- strategy/
|   |       |-- strategy-api-immutable-crud-clone-run-control-v1.md
|   |       |-- strategy-domain-spec-immutable-storage-runs-events-v1.md
|   |       |-- strategy-live-runner-redis-streams-v1.md
|   |       |-- strategy-milestone-3-epics-v1.md
|   |       |-- strategy-realtime-output-redis-streams-v1.md
|   |       |-- strategy-runtime-config-v1.md
|   |       `-- strategy-telegram-notifier-best-effort-policy-v1.md
|   |-- decisions/
|   |-- repository_three.md
|   `-- runbooks/
|       |-- backtest-artifacts-rebuild.md
|       |-- backtest-job-runner.md
|       |-- backtest-rollout-rollback.md
|       |-- clickhouse-memory-profiles.md
|       |-- clickhouse-partition-dedup.md
|       |-- help_commands.md
|       |-- indicators-numba-cache-and-threads.md
|       |-- indicators-numba-warmup-jit.md
|       |-- indicators-why-nan.md
|       |-- keycloak-local-setup-and-ops.md
|       |-- market-data-autonomous-docker.md
|       |-- market-data-metrics-reference-ru.md
|       |-- market-data-metrics.md
|       |-- market-data-redis-streams.md
|       |-- prod-dashboard-metrics-reference-ru.md
|       |-- roehub-ui-autostart-systemd.md
|       |-- strategy-live-worker.md
|       `-- web-ui-gateway-same-origin.md
|-- infra/
|   |-- caddy/
|   |   `-- Caddyfile.vps
|   |-- docker/
|   |   |-- .env.example
|   |   |-- Dockerfile.market_data
|   |   |-- docker-compose.backend.yml
|   |   |-- docker-compose.market_data.yml
|   |   |-- docker-compose.web.prod.yml
|   |   `-- docker-compose.yml
|   |-- k8s/
|   |-- macos/
|   |   |-- blackbox/
|   |   |   |-- blackbox.test.yml
|   |   |   `-- blackbox.yml
|   |   |-- clickhouse/
|   |   |   |-- config.test.xml
|   |   |   |-- config.xml
|   |   |   `-- users.d/
|   |   |       `-- roehub.xml
|   |   |-- launchd/
|   |   |   |-- com.roehub.api.plist
|   |   |   |-- com.roehub.backtest-artifact-publisher.plist
|   |   |   |-- com.roehub.backtest-job-runner@.plist.template
|   |   |   |-- com.roehub.blackbox-exporter.plist
|   |   |   |-- com.roehub.clickhouse-exporter.plist
|   |   |   |-- com.roehub.clickhouse.plist
|   |   |   |-- com.roehub.market-data-scheduler.plist
|   |   |   |-- com.roehub.market-data-ws-worker.plist
|   |   |   |-- com.roehub.postgres-exporter.plist
|   |   |   |-- com.roehub.redis-exporter.plist
|   |   |   |-- com.roehub.tailscale-runtime.plist
|   |   |   |-- com.roehub.test.api.plist
|   |   |   |-- com.roehub.test.backtest-artifact-publisher.plist
|   |   |   |-- com.roehub.test.blackbox-exporter.plist
|   |   |   |-- com.roehub.test.clickhouse.plist
|   |   |   |-- com.roehub.test.grafana.plist
|   |   |   |-- com.roehub.test.market-data-scheduler.plist
|   |   |   |-- com.roehub.test.market-data-ws-worker.plist
|   |   |   |-- com.roehub.test.postgres.plist
|   |   |   |-- com.roehub.test.prometheus.plist
|   |   |   `-- com.roehub.test.redis.plist
|   |   `-- prometheus/
|   |       |-- prometheus.prod.yml
|   |       `-- prometheus.test.yml
|   |-- monitoring/
|   |   |-- host-macos/
|   |   |   `-- install-node-exporter.sh*
|   |   `-- monitoring/
|   |       |-- blackbox/
|   |       |   `-- blackbox.yml
|   |       |-- grafana/
|   |       |   |-- dashboards/
|   |       |   |   `-- roehub/
|   |       |   |       |-- 01-platform-overview.json
|   |       |   |       |-- 02-mac-studio-host.json
|   |       |   |       |-- 03-containers.json
|   |       |   |       |-- 04-datastores.json
|   |       |   |       `-- 05-api-market-data.json
|   |       |   `-- provisioning/
|   |       |       |-- dashboards/
|   |       |       |   `-- roehub-dashboards.yml
|   |       |       `-- datasources/
|   |       |           `-- roehub-prometheus.yml
|   |       `-- prometheus/
|   |           |-- prometheus.yml
|   |           `-- rules/
|   |               `-- mac-studio-monitoring.rules.yml
|   `-- scripts/
|       `-- monit/
|           |-- launchctl_service_control.sh*
|           |-- roehub-backtest-artifact-publisher.monitrc
|           |-- roehub-keycloak.monitrc
|           `-- roehub-market-data.monitrc
|-- migrations/
|   |-- clickhouse/
|   |   `-- market_data_ddl.sql
|   `-- postgres/
|       |-- 0001_identity_v1.sql
|       |-- 0002_identity_2fa_totp_v1.sql
|       |-- 0003_identity_exchange_keys_v1.sql
|       |-- 0004_identity_exchange_keys_v2.sql
|       `-- 0005_identity_keycloak_cutover_v1.sql
|-- notebooks/
|   |-- .cache/
|   |   `-- numba/
|   |       `-- notebooks/
|   |-- 01_backtest_regression_smoke.ipynb
|   |-- 02_backtest_perf_memory_smoke.ipynb
|   |-- 03_sync_backtest_pickle_rollup_1h_ma_grid.ipynb
|   |-- 03_sync_backtest_pickle_rollup_1h_ma_grid.py
|   `-- README.md
|-- pyproject.toml
|-- pyrightconfig.json
|-- repo_tree.md
|-- scripts/
|   |-- data/
|   |-- local/
|   |-- macos/
|   |   |-- bootstrap_native_prod.sh*
|   |   |-- bootstrap_native_test.sh*
|   |   |-- clickhouse_partition_dedup.sh*
|   |   |-- configure_tailscale_serve.sh*
|   |   |-- ensure_tailscale_runtime.sh*
|   |   |-- export_container_backups.sh*
|   |   |-- install_native_backend_prereqs.sh*
|   |   |-- reload_launchd_services.sh*
|   |   |-- render_backtest_job_runner_launchd.py
|   |   |-- smoke_prod.sh*
|   |   `-- smoke_test.sh*
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
|       |   |   |   |-- __init__.py
|       |   |   |   |-- inbound/
|       |   |   |   `-- outbound/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- acl/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   `-- strategy_repository_reader.py
|       |   |   |       |-- artifacts_fs/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- current_pointer_writer.py
|       |   |   |       |   `-- path_builder.py
|       |   |   |       |-- config/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- backtest_artifacts_runtime_config.py
|       |   |   |       |   `-- backtest_runtime_config.py
|       |   |   |       |-- defaults/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   `-- indicators_yaml_defaults_provider.py
|       |   |   |       |-- feeds/
|       |   |   |       |   `-- market_data_acl/
|       |   |   |       |-- persistence/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- filesystem/
|       |   |   |       |   `-- postgres/
|       |   |   |       |       |-- __init__.py
|       |   |   |       |       |-- backtest_job_lease_repository.py
|       |   |   |       |       |-- backtest_job_repository.py
|       |   |   |       |       |-- backtest_job_results_repository.py
|       |   |   |       |       `-- gateway.py
|       |   |   |       `-- progress/
|       |   |   |           |-- logs/
|       |   |   |           `-- messaging/
|       |   |   |-- application/
|       |   |   |   |-- __init__.py
|       |   |   |   |-- dto/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   `-- run_backtest.py
|       |   |   |   |-- errors/
|       |   |   |   |-- ports/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- backtest_job_repositories.py
|       |   |   |   |   |-- backtest_job_request_decoder.py
|       |   |   |   |   |-- current_user.py
|       |   |   |   |   |-- feeds/
|       |   |   |   |   |-- progress/
|       |   |   |   |   |-- staged_runner.py
|       |   |   |   |   |-- stores/
|       |   |   |   |   `-- strategy_reader.py
|       |   |   |   |-- services/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- grid_builder_v1.py
|       |   |   |   |   |-- job_runner_streaming_v1.py
|       |   |   |   |   |-- run_control_v1.py
|       |   |   |   |   |-- signals_from_indicators_v1.py
|       |   |   |   |   `-- warmup_estimator.py
|       |   |   |   `-- use_cases/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- backtest_jobs_api_v1.py
|       |   |   |       |-- backtest_runs_api_v1.py
|       |   |   |       |-- backtest_runs_history_api_v1.py
|       |   |   |       |-- errors.py
|       |   |   |       `-- request_runtime_contract_v1.py
|       |   |   `-- domain/
|       |   |       |-- __init__.py
|       |   |       |-- entities/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- backtest_job.py
|       |   |       |   |-- backtest_job_results.py
|       |   |       |   |-- backtest_placeholders.py
|       |   |       |   `-- execution_v1.py
|       |   |       |-- errors/
|       |   |       |   |-- __init__.py
|       |   |       |   `-- backtest_errors.py
|       |   |       |-- events/
|       |   |       |-- specifications/
|       |   |       `-- value_objects/
|       |   |           |-- __init__.py
|       |   |           |-- backtest_job_cursor.py
|       |   |           |-- execution_v1.py
|       |   |           |-- signal_v1.py
|       |   |           `-- variant_identity.py
|       |   |-- backtest_artifacts/
|       |   |   |-- __init__.py
|       |   |   |-- adapters/
|       |   |   |   |-- __init__.py
|       |   |   |   `-- outbound/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- artifacts_fs/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- current_pointer_writer.py
|       |   |   |       |   `-- path_builder.py
|       |   |   |       |-- config/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   `-- backtest_artifacts_runtime_config.py
|       |   |   |       |-- defaults/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   `-- indicators_yaml_defaults_provider.py
|       |   |   |       `-- persistence/
|       |   |   |           |-- __init__.py
|       |   |   |           `-- postgres/
|       |   |   |               |-- __init__.py
|       |   |   |               |-- backtest_job_repository.py
|       |   |   |               `-- gateway.py
|       |   |   `-- application/
|       |   |       |-- __init__.py
|       |   |       |-- services/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- numba_runtime_v1.py
|       |   |       |   `-- v2/
|       |   |       |       |-- __init__.py
|       |   |       |       |-- adaptive_selector_v2.py
|       |   |       |       |-- artifact_backed_stage_b_scorer_v2.py
|       |   |       |       |-- artifact_manifest_loader.py
|       |   |       |       |-- artifact_manifest_validator.py
|       |   |       |       |-- artifact_precompute_coordinator.py
|       |   |       |       |-- artifact_precompute_runner.py
|       |   |       |       |-- artifact_runtime_core_v2.py
|       |   |       |       |-- artifact_runtime_plan_v2.py
|       |   |       |       |-- artifact_runtime_timeline_v2.py
|       |   |       |       |-- artifact_slot_publisher.py
|       |   |       |       |-- artifact_slot_resolver.py
|       |   |       |       |-- benchmark_corpus_v2.py
|       |   |       |       |-- contracts.py
|       |   |       |       |-- diversified_retention_v2.py
|       |   |       |       |-- execution_profile_v2.py
|       |   |       |       |-- family_plugins/
|       |   |       |       |   |-- __init__.py
|       |   |       |       |   |-- circuit_breaker_v2.py
|       |   |       |       |   |-- contracts_v2.py
|       |   |       |       |   |-- ma_family_plugin_v2.py
|       |   |       |       |   `-- registry_v2.py
|       |   |       |       |-- generic_row_scorer_v2.py
|       |   |       |       |-- hierarchical_shortlist_builder_v2.py
|       |   |       |       |-- hit_times_compute_v2.py
|       |   |       |       |-- metrics_kernel.py
|       |   |       |       |-- notebook_parity_benchmark_corpus_v2.py
|       |   |       |       |-- price_arrays_loader.py
|       |   |       |       |-- risk_exit_kernel_1m.py
|       |   |       |       |-- signal_aggregator_kernel.py
|       |   |       |       |-- signal_chunk_planner_v2.py
|       |   |       |       |-- signal_features_loader_v2.py
|       |   |       |       |-- signal_matrix_loader.py
|       |   |       |       |-- signal_rules_engine_v2.py
|       |   |       |       |-- stage_a_shortlist_builder_v2.py
|       |   |       |       |-- stage_b_golden_fixtures_v2.py
|       |   |       |       `-- trade_compactor_kernel.py
|       |   |       `-- use_cases/
|       |   |           |-- __init__.py
|       |   |           |-- publish_backtest_artifacts_v2.py
|       |   |           `-- run_backtest_job_runner_v1.py
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
|       |   |   |   |       |   `-- current_user.py
|       |   |   |   |       `-- routes/
|       |   |   |   |           |-- __init__.py
|       |   |   |   |           |-- auth_oidc.py
|       |   |   |   |           `-- exchange_keys.py
|       |   |   |   `-- outbound/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- persistence/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- in_memory/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   |-- exchange_keys_repository.py
|       |   |   |       |   |   |-- session_repository.py
|       |   |   |       |   |   `-- user_repository.py
|       |   |   |       |   `-- postgres/
|       |   |   |       |       |-- __init__.py
|       |   |   |       |       |-- exchange_keys_repository.py
|       |   |   |       |       |-- gateway.py
|       |   |   |       |       |-- session_repository.py
|       |   |   |       |       `-- user_repository.py
|       |   |   |       |-- policy/
|       |   |   |       |   `-- __init__.py
|       |   |   |       |-- security/
|       |   |   |       |   |-- __init__.py
|       |   |   |       |   |-- current_user/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- roehub_session_current_user.py
|       |   |   |       |   |-- exchange_keys/
|       |   |   |       |   |   |-- __init__.py
|       |   |   |       |   |   `-- aes_gcm_envelope_secret_cipher.py
|       |   |   |       |   |-- jwt/
|       |   |   |       |   |-- telegram/
|       |   |   |       |   `-- two_factor/
|       |   |   |       `-- time/
|       |   |   |           |-- __init__.py
|       |   |   |           `-- system_identity_clock.py
|       |   |   |-- application/
|       |   |   |   |-- __init__.py
|       |   |   |   |-- ports/
|       |   |   |   |   |-- __init__.py
|       |   |   |   |   |-- clock.py
|       |   |   |   |   |-- current_user.py
|       |   |   |   |   |-- exchange_keys_repository.py
|       |   |   |   |   |-- exchange_keys_secret_cipher.py
|       |   |   |   |   |-- session_repository.py
|       |   |   |   |   `-- user_repository.py
|       |   |   |   `-- use_cases/
|       |   |   |       |-- __init__.py
|       |   |   |       |-- create_exchange_key.py
|       |   |   |       |-- delete_exchange_key.py
|       |   |   |       |-- exchange_keys_errors.py
|       |   |   |       |-- exchange_keys_models.py
|       |   |   |       `-- list_exchange_keys.py
|       |   |   `-- domain/
|       |   |       |-- __init__.py
|       |   |       |-- entities/
|       |   |       |   |-- __init__.py
|       |   |       |   |-- exchange_key.py
|       |   |       |   `-- user.py
|       |   |       `-- value_objects/
|       |   |           |-- __init__.py
|       |   |           `-- telegram_chat_id.py
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
|       |   |   |           |   |-- enabled_market_reader.py
|       |   |   |           |   |-- enabled_tradable_instrument_search_reader.py
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
|       |   |   |   |   |-- canonical_candle_batch_1m.py
|       |   |   |   |   |-- reference_api.py
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
|       |   |   |   |   |   |-- enabled_market_reader.py
|       |   |   |   |   |   |-- enabled_tradable_instrument_search_reader.py
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
|       |   |   |       |-- list_enabled_markets.py
|       |   |   |       |-- rest_catchup_1m.py
|       |   |   |       |-- rest_fill_range_1m.py
|       |   |   |       |-- search_enabled_tradable_instruments.py
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
|       |       |   |-- __init__.py
|       |       |   |-- inbound/
|       |       |   `-- outbound/
|       |       |       |-- __init__.py
|       |       |       |-- acl/
|       |       |       |   |-- __init__.py
|       |       |       |   `-- identity/
|       |       |       |       |-- __init__.py
|       |       |       |       `-- confirmed_telegram_chat_binding_resolver.py
|       |       |       |-- config/
|       |       |       |   |-- __init__.py
|       |       |       |   |-- live_runner_runtime_config.py
|       |       |       |   |-- scalar_env_overrides.py
|       |       |       |   `-- strategy_runtime_config.py
|       |       |       |-- feeds/
|       |       |       |   `-- market_data_acl/
|       |       |       |-- messaging/
|       |       |       |   |-- __init__.py
|       |       |       |   |-- redis/
|       |       |       |   |   |-- __init__.py
|       |       |       |   |   |-- redis_streams_live_candle_stream.py
|       |       |       |   |   `-- redis_streams_realtime_output_publisher.py
|       |       |       |   `-- telegram/
|       |       |       |       |-- __init__.py
|       |       |       |       |-- log_only_telegram_notifier.py
|       |       |       |       |-- telegram_bot_api_notifier.py
|       |       |       |       `-- telegram_notifier_hooks.py
|       |       |       |-- persistence/
|       |       |       |   |-- __init__.py
|       |       |       |   |-- in_memory/
|       |       |       |   |   |-- __init__.py
|       |       |       |   |   |-- strategy_event_repository.py
|       |       |       |   |   |-- strategy_repository.py
|       |       |       |   |   `-- strategy_run_repository.py
|       |       |       |   `-- postgres/
|       |       |       |       |-- __init__.py
|       |       |       |       |-- gateway.py
|       |       |       |       |-- strategy_event_repository.py
|       |       |       |       |-- strategy_repository.py
|       |       |       |       `-- strategy_run_repository.py
|       |       |       |-- sinks/
|       |       |       |   |-- memory/
|       |       |       |   `-- messaging/
|       |       |       `-- time/
|       |       |           |-- __init__.py
|       |       |           |-- system_runner_sleeper.py
|       |       |           `-- system_strategy_clock.py
|       |       |-- application/
|       |       |   |-- __init__.py
|       |       |   |-- dto/
|       |       |   |-- errors/
|       |       |   |-- ports/
|       |       |   |   |-- __init__.py
|       |       |   |   |-- clock.py
|       |       |   |   |-- current_user.py
|       |       |   |   |-- feeds/
|       |       |   |   |-- live_candle_stream.py
|       |       |   |   |-- realtime_output_publisher.py
|       |       |   |   |-- repositories/
|       |       |   |   |   |-- __init__.py
|       |       |   |   |   |-- strategy_event_repository.py
|       |       |   |   |   |-- strategy_repository.py
|       |       |   |   |   `-- strategy_run_repository.py
|       |       |   |   |-- sinks/
|       |       |   |   |-- sleeper.py
|       |       |   |   `-- telegram_notifier.py
|       |       |   |-- services/
|       |       |   |   |-- __init__.py
|       |       |   |   |-- live_runner.py
|       |       |   |   |-- telegram_notification_policy.py
|       |       |   |   |-- timeframe_rollup.py
|       |       |   |   `-- warmup_estimator.py
|       |       |   `-- use_cases/
|       |       |       |-- __init__.py
|       |       |       |-- _shared.py
|       |       |       |-- clone_strategy.py
|       |       |       |-- create_strategy.py
|       |       |       |-- delete_strategy.py
|       |       |       |-- errors.py
|       |       |       |-- get_my_strategy.py
|       |       |       |-- list_my_strategies.py
|       |       |       |-- run_strategy.py
|       |       |       `-- stop_strategy.py
|       |       `-- domain/
|       |           |-- __init__.py
|       |           |-- entities/
|       |           |   |-- __init__.py
|       |           |   |-- strategy.py
|       |           |   |-- strategy_event.py
|       |           |   |-- strategy_run.py
|       |           |   `-- strategy_spec_v1.py
|       |           |-- errors/
|       |           |   |-- __init__.py
|       |           |   `-- strategy_errors.py
|       |           |-- events/
|       |           |-- services/
|       |           |   |-- __init__.py
|       |           |   |-- run_invariants.py
|       |           |   `-- strategy_name.py
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
|       |   |-- errors/
|       |   |   |-- __init__.py
|       |   |   `-- roehub_error.py
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
|   |   |-- 02_rest_catchup_1m.ipynb
|   |   |-- 04_indicator_grid_export.ipynb
|   |   |-- 05_hit_time_grid.ipynb
|   |   |-- 06_backtest_compute.ipynb
|   |   |-- new_engine/
|   |   |   |-- 01_run_322_btcusdt_1h_artifact_probe.ipynb
|   |   |   `-- 02_run_f7d2_btcusdt_15m_no_risk_probe.ipynb
|   |   `-- precompute/
|   |       `-- btcusdt_5m/
|   |-- perf_smoke/
|   |   `-- contexts/
|   |       |-- backtest/
|   |       |   |-- fixtures/
|   |       |   |   |-- backtest_notebook_parity_benchmark_corpus_v1.json
|   |       |   |   |-- backtest_runtime_acceleration_benchmark_corpus_v1.json
|   |       |   |   |-- r0_benchmark_scenarios.json
|   |       |   |   |-- r0_parity_scope.json
|   |       |   |   `-- r5_stage_b_golden_cases.json
|   |       |   |-- test_backtest_adaptive_selector_rollout_v2.py
|   |       |   |-- test_backtest_family_plugin_rollout_v2.py
|   |       |   |-- test_backtest_hybrid_shortlist_rollout_v2.py
|   |       |   |-- test_backtest_notebook_parity_perf_smoke_v1.py
|   |       |   `-- test_backtest_staged_runner_perf_smoke.py
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
|       |   |   |-- test_api_error_handlers.py
|       |   |   |-- test_app_strategy_router_toggle.py
|       |   |   |-- test_backtest_jobs_dto.py
|       |   |   |-- test_backtest_jobs_routes.py
|       |   |   |-- test_backtest_runs_dto.py
|       |   |   |-- test_backtest_runs_routes.py
|       |   |   |-- test_backtest_wiring_module.py
|       |   |   |-- test_backtests_dto.py
|       |   |   |-- test_backtests_routes.py
|       |   |   |-- test_identity_current_user_dependency.py
|       |   |   |-- test_identity_exchange_keys_routes.py
|       |   |   |-- test_identity_routes.py
|       |   |   |-- test_identity_wiring_module.py
|       |   |   |-- test_indicators_wiring_module.py
|       |   |   |-- test_market_data_reference_routes.py
|       |   |   |-- test_market_data_reference_wiring_module.py
|       |   |   |-- test_operations_routes.py
|       |   |   |-- test_strategies_routes.py
|       |   |   |-- test_strategy_wiring_module.py
|       |   |   `-- wiring/
|       |   |       `-- modules/
|       |   |-- cli/
|       |   |   |-- commands/
|       |   |   |   `-- test_rest_catchup_1m_cli.py
|       |   |   `-- test_backtest_artifact_publish_cli.py
|       |   |-- migrations/
|       |   |   |-- test_bootstrap_apply_flow.py
|       |   |   |-- test_bootstrap_conninfo_dsn.py
|       |   |   |-- test_bootstrap_decisions.py
|       |   |   `-- test_main_dsn_formats.py
|       |   |-- monitoring/
|       |   |   `-- test_clickhouse_exporter.py
|       |   |-- scheduler/
|       |   |   |-- test_backtest_artifact_publisher_app.py
|       |   |   `-- test_backtest_artifact_publisher_metrics.py
|       |   |-- test_backtest_job_runner_main.py
|       |   |-- test_strategy_live_runner_main.py
|       |   |-- web/
|       |   |   |-- test_api_client.py
|       |   |   |-- test_app_routes.py
|       |   |   |-- test_backtest_runs_ui_asset.py
|       |   |   |-- test_backtest_ui_asset.py
|       |   |   `-- test_security.py
|       |   `-- worker/
|       |       `-- backtest_job_runner/
|       |           `-- wiring/
|       |               `-- modules/
|       |                   `-- test_backtest_job_runner.py
|       |-- contexts/
|       |   |-- backtest/
|       |   |   |-- adapters/
|       |   |   |   |-- outbound/
|       |   |   |   |   `-- artifacts_fs/
|       |   |   |   |       |-- test_backtest_artifact_path_builder_v2.py
|       |   |   |   |       `-- test_current_pointer_writer_v2.py
|       |   |   |   |-- test_backtest_artifacts_runtime_config.py
|       |   |   |   |-- test_backtest_runtime_config.py
|       |   |   |   |-- test_indicators_yaml_defaults_provider.py
|       |   |   |   |-- test_postgres_backtest_job_repositories.py
|       |   |   |   `-- test_strategy_repository_reader.py
|       |   |   |-- application/
|       |   |   |   |-- dto/
|       |   |   |   |   `-- test_run_backtest_request.py
|       |   |   |   |-- services/
|       |   |   |   |   |-- test_grid_builder_v1.py
|       |   |   |   |   |-- test_job_runner_streaming_v1.py
|       |   |   |   |   |-- test_signals_from_indicators_v1.py
|       |   |   |   |   `-- v2/
|       |   |   |   |       |-- artifact_testkit_v2.py
|       |   |   |   |       |-- fixtures/
|       |   |   |   |       |   `-- stage_b_golden_fixtures_v2.json
|       |   |   |   |       |-- test_adaptive_selector_v2.py
|       |   |   |   |       |-- test_artifact_manifest_validator_v2.py
|       |   |   |   |       |-- test_artifact_precompute_runner_v2.py
|       |   |   |   |       |-- test_artifact_slot_publisher_v2.py
|       |   |   |   |       |-- test_artifact_slot_resolver_v2.py
|       |   |   |   |       |-- test_diversified_retention_v2.py
|       |   |   |   |       |-- test_family_plugin_circuit_breaker_v2.py
|       |   |   |   |       |-- test_family_plugin_contracts_v2.py
|       |   |   |   |       |-- test_family_plugin_registry_v2.py
|       |   |   |   |       |-- test_generic_row_scorer_v2.py
|       |   |   |   |       |-- test_hierarchical_shortlist_builder_v2.py
|       |   |   |   |       |-- test_hit_times_compute_v2.py
|       |   |   |   |       |-- test_ma_family_plugin_v2.py
|       |   |   |   |       |-- test_metrics_kernel_v2.py
|       |   |   |   |       |-- test_price_arrays_loader_v2.py
|       |   |   |   |       |-- test_risk_exit_kernel_1m_v2.py
|       |   |   |   |       |-- test_signal_aggregator_kernel_v2.py
|       |   |   |   |       |-- test_signal_features_loader_v2.py
|       |   |   |   |       |-- test_signal_matrix_loader_v2.py
|       |   |   |   |       |-- test_signal_rules_engine_v2.py
|       |   |   |   |       |-- test_stage_a_shortlist_builder_v2.py
|       |   |   |   |       |-- test_stage_b_golden_fixtures_v2.py
|       |   |   |   |       |-- test_trade_compactor_kernel_v2.py
|       |   |   |   |       `-- test_yaml_backtest_artifact_loader_v2.py
|       |   |   |   |-- test_backtest_errors.py
|       |   |   |   `-- use_cases/
|       |   |   |       |-- test_backtest_jobs_api_v1.py
|       |   |   |       |-- test_backtest_runs_api_v1.py
|       |   |   |       |-- test_backtest_runs_history_api_v1.py
|       |   |   |       |-- test_publish_backtest_artifacts_v2.py
|       |   |   |       |-- test_request_runtime_contract_v1.py
|       |   |   |       `-- test_run_backtest_job_runner_v1.py
|       |   |   |-- domain/
|       |   |   |   |-- entities/
|       |   |   |   |   |-- test_backtest_job_entities.py
|       |   |   |   |   `-- test_execution_v1_entities.py
|       |   |   |   `-- value_objects/
|       |   |   |       `-- test_variant_identity.py
|       |   |   `-- golden/
|       |   |       |-- multi-trade.md
|       |   |       `-- no-trades.md
|       |   |-- identity/
|       |   |   |-- adapters/
|       |   |   |   `-- outbound/
|       |   |   |       |-- persistence/
|       |   |   |       |   `-- postgres/
|       |   |   |       |       |-- test_identity_session_repository.py
|       |   |   |       |       `-- test_timezone_normalization.py
|       |   |   |       `-- security/
|       |   |   |           `-- test_exchange_keys_aes_gcm_envelope_secret_cipher.py
|       |   |   `-- application/
|       |   |       `-- test_exchange_keys_use_cases.py
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
|       |   |-- market_data/
|       |   |   |-- adapters/
|       |   |   |   |-- test_clickhouse_canonical_candle_index_reader.py
|       |   |   |   |-- test_clickhouse_canonical_candle_reader.py
|       |   |   |   |-- test_clickhouse_enabled_instrument_reader.py
|       |   |   |   |-- test_clickhouse_enabled_market_reader.py
|       |   |   |   |-- test_clickhouse_enabled_tradable_instrument_search_reader.py
|       |   |   |   |-- test_clickhouse_raw_kline_writer.py
|       |   |   |   |-- test_clickhouse_ref_instruments_writer.py
|       |   |   |   |-- test_clickhouse_thread_local_gateway.py
|       |   |   |   |-- test_market_data_runtime_config.py
|       |   |   |   |-- test_parquet_candle_ingest_source.py
|       |   |   |   |-- test_redis_streams_live_candle_publisher.py
|       |   |   |   |-- test_rest_candle_ingest_source.py
|       |   |   |   |-- test_rest_instrument_metadata_source.py
|       |   |   |   |-- test_whitelist_csv_loader.py
|       |   |   |   |-- test_ws_binance_client.py
|       |   |   |   `-- test_ws_bybit_client.py
|       |   |   `-- application/
|       |   |       |-- services/
|       |   |       |   |-- test_gap_tracker.py
|       |   |       |   |-- test_insert_buffer.py
|       |   |       |   |-- test_reconnect_tail_fill.py
|       |   |       |   |-- test_rest_fill_queue.py
|       |   |       |   |-- test_scheduler_backfill_planner.py
|       |   |       |   |-- test_scheduler_startup_scan.py
|       |   |       |   `-- test_ws_worker_publishes_redis.py
|       |   |       `-- use_cases/
|       |   |           |-- test_backfill_1m_candles.py
|       |   |           |-- test_enrich_ref_instruments_from_exchange.py
|       |   |           |-- test_reference_api_use_cases.py
|       |   |           |-- test_reference_data_sync.py
|       |   |           |-- test_rest_catchup_1m.py
|       |   |           |-- test_rest_fill_range_1m.py
|       |   |           `-- test_time_slicing.py
|       |   `-- strategy/
|       |       |-- adapters/
|       |       |   |-- test_log_only_telegram_notifier.py
|       |       |   |-- test_postgres_confirmed_telegram_chat_binding_resolver.py
|       |       |   |-- test_postgres_strategy_repositories.py
|       |       |   |-- test_redis_strategy_live_candle_stream.py
|       |       |   |-- test_redis_strategy_realtime_output_publisher.py
|       |       |   |-- test_strategy_live_runner_runtime_config.py
|       |       |   |-- test_strategy_live_runner_wiring_module.py
|       |       |   |-- test_strategy_runtime_config.py
|       |       |   `-- test_telegram_bot_api_notifier.py
|       |       |-- application/
|       |       |   |-- test_strategy_live_runner.py
|       |       |   |-- test_strategy_use_cases.py
|       |       |   `-- test_telegram_notification_policy.py
|       |       `-- domain/
|       |           `-- test_strategy_domain.py
|       |-- infra/
|       |   |-- test_monitoring_assets.py
|       |   |-- test_prod_compose_files.py
|       |   `-- test_ui_compose_profile.py
|       |-- platform/
|       |   `-- config/
|       |       `-- test_indicators_compute_numba_config.py
|       |-- scripts/
|       |   `-- macos/
|       |       `-- test_render_backtest_job_runner_launchd.py
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
|-- typings/
|   `-- numba/
|       `-- __init__.pyi
`-- uv.lock

463 directories, 940 files
