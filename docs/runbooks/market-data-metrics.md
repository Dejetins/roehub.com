# Market Data Metrics

Короткий операционный каталог метрик для production (`Mac Studio`, native runtime).

Полное подробное описание каждой метрики (на русском):
- `docs/runbooks/market-data-metrics-reference-ru.md`

Референс по прод-дашборду (панели, PromQL, типы визуализации):
- `docs/runbooks/prod-dashboard-metrics-reference-ru.md`

Source of truth scrape-конфига:
- `infra/macos/prometheus/prometheus.prod.yml`

## Jobs и endpoint'ы

- `market-data-ws-worker` -> `127.0.0.1:9201/metrics`
- `market-data-scheduler` -> `127.0.0.1:9202/metrics`
- `backtest-artifact-publisher` -> `127.0.0.1:9203/metrics`
- `clickhouse-exporter` -> `127.0.0.1:9116/metrics`
- `postgres-exporter` -> `127.0.0.1:9187/metrics`
- `redis-exporter` -> `127.0.0.1:9121/metrics`
- `node-exporter` -> `127.0.0.1:9100/metrics`
- `blackbox-http` / `blackbox-tcp` -> `127.0.0.1:9115/probe`
- `prometheus` -> `127.0.0.1:9090/metrics`

## Самые важные метрики

### Worker (`market-data-ws-worker`)

- `ws_connected`
- `ws_messages_total`
- `insert_errors_total`
- `redis_publish_errors_total`
- `ws_closed_to_insert_done_seconds`

### Scheduler (`market-data-scheduler`)

- `scheduler_job_errors_total`
- `scheduler_tasks_enqueued_total`
- `scheduler_rest_catchup_gap_rows_written_total`

### Backtest Artifact Publisher (`backtest-artifact-publisher`)

- `backtest_artifact_publish_runs_total`
- `backtest_artifact_publish_symbols_total`
- `backtest_artifact_publish_blocked_total`
- `backtest_artifact_publish_last_success_unixtime`
- `backtest_artifact_tail_rebuild_bars_total`

Intra-run progress is intentionally split out of Prometheus and lives in structured logs:

- `artifact_precompute_stage_started`
- `artifact_precompute_stage_finished`
- `current_timeframe`
- `current_indicator_id`
- `chunk_index`
- `chunk_count`
- `completed_chunks_total`
- `completed_indicators_total`

### Monitoring/infra

- `up{job=...}`
- `probe_success`
- `clickhouse_exporter_scrape_success`
- `pg_up`
- `redis_up`
- `node_load1`
- `node_memory_free_bytes`
- `node_filesystem_avail_bytes`

## Полный список метрик по сервисам (с кратким смыслом)

### 1) `market-data-ws-worker`

| Метрика | Краткий смысл |
|---|---|
| `ws_connected` | Текущее число активных WS-соединений |
| `ws_reconnects_total` | Счетчик переподключений WS |
| `ws_messages_total` | Счетчик полученных WS-сообщений |
| `ws_errors_total` | Ошибки обработки WS |
| `ignored_non_closed_total` | Отброшенные non-closed обновления |
| `insert_rows_total` | Записанные строки в raw |
| `insert_batches_total` | Число insert-батчей |
| `insert_duration_seconds` | Гистограмма длительности insert |
| `insert_errors_total` | Ошибки insert |
| `ws_closed_to_insert_start_seconds` | Латентность closed-candle -> start insert |
| `ws_closed_to_insert_done_seconds` | Латентность closed-candle -> insert done |
| `ws_out_of_order_total` | Out-of-order свечи |
| `ws_duplicates_total` | Дубли свечей |
| `redis_publish_total` | Успешные публикации в Redis Streams |
| `redis_publish_errors_total` | Ошибки публикации в Redis |
| `redis_publish_duplicates_total` | Дубли/нарушение порядка stream id |
| `redis_publish_duration_seconds` | Гистограмма длительности publish |
| `rest_fill_tasks_total` | Принятые fill-задачи |
| `rest_fill_active` | Активные fill-задачи |
| `rest_fill_errors_total` | Ошибки fill-задач |
| `rest_fill_duration_seconds` | Гистограмма длительности fill-задач |

### 2) `market-data-scheduler`

| Метрика | Краткий смысл |
|---|---|
| `scheduler_job_runs_total` | Запуски scheduler jobs |
| `scheduler_job_errors_total` | Ошибки scheduler jobs |
| `scheduler_job_duration_seconds` | Гистограмма длительности jobs |
| `scheduler_tasks_planned_total` | Запланированные задачи |
| `scheduler_tasks_enqueued_total` | Enqueued задачи |
| `scheduler_startup_scan_instruments_total` | Инструменты, обработанные startup scan |
| `scheduler_rest_catchup_instruments_total` | Инструменты periodic catchup |
| `scheduler_rest_catchup_tail_minutes_total` | Сумма tail-минут periodic catchup |
| `scheduler_rest_catchup_tail_rows_written_total` | Записанные tail-строки |
| `scheduler_rest_catchup_gap_days_scanned_total` | Просканированные дни на gaps |
| `scheduler_rest_catchup_gap_days_with_gaps_total` | Дни, где найдены gaps |
| `scheduler_rest_catchup_gap_ranges_filled_total` | Закрытые gap-диапазоны |
| `scheduler_rest_catchup_gap_rows_written_total` | Записанные строки по gaps |

### 3) `backtest-artifact-publisher`

| Метрика | Краткий смысл |
|---|---|
| `backtest_artifact_publish_runs_total` | Итоговые daily publish-cycle runs по статусу |
| `backtest_artifact_publish_duration_seconds` | Гистограмма длительности полного publish-цикла |
| `backtest_artifact_publish_symbols_total` | Обработанные symbol roots по итоговому статусу |
| `backtest_artifact_publish_blocked_total` | Блокировки publish-run по причинам |
| `backtest_artifact_publish_last_success_unixtime` | Unix time последнего цикла с хотя бы одним успешным publish |
| `backtest_artifact_tail_rebuild_bars_total` | Сколько баров реально переписано в bounded tail по stage |

Structured progress fields for this service are not separate Prometheus series. Operators should
read them from logs:

- `artifact_precompute_stage_started` / `artifact_precompute_stage_finished` tell which pipeline
  stage is active;
- `current_timeframe` and `current_indicator_id` show the currently open timeframe session and
  indicator target;
- `chunk_index` / `chunk_count` distinguish a healthy long bootstrap from a stuck worker pool;
- `completed_chunks_total` / `completed_indicators_total` show real end-to-end progress inside the
  current timeframe-local session.

### 4) `clickhouse-exporter`

| Метрика | Краткий смысл |
|---|---|
| `clickhouse_exporter_scrape_duration_seconds` | Длительность последнего scrape |
| `clickhouse_exporter_scrape_success` | Успешность последнего scrape (`1`/`0`) |
| `clickhouse_uptime_seconds` | Uptime процесса ClickHouse |
| `clickhouse_system_metric_value` | Текущие значения выбранных system.metrics |
| `clickhouse_system_event_total` | Кумулятивные значения выбранных system.events |

### 5) `postgres-exporter`

| Метрика | Краткий смысл |
|---|---|
| `pg_up` | Доступность PostgreSQL |
| `pg_exporter_last_scrape_error` | Ошибка последнего scrape |
| `pg_exporter_last_scrape_duration_seconds` | Длительность последнего scrape |
| `pg_exporter_scrapes_total` | Число scrape экспортера |
| `pg_scrape_collector_success` | Успешность коллекторов экспортера |
| `pg_scrape_collector_duration_seconds` | Длительность коллекторов |
| `pg_stat_database_xact_commit` | Коммиты транзакций по БД |
| `pg_stat_database_xact_rollback` | Rollback транзакций по БД |
| `pg_stat_database_numbackends` | Число backend-соединений |
| `pg_stat_activity_count` | Активность сессий по состояниям |
| `pg_locks_count` | Число locks по режимам |
| `pg_database_size_bytes` | Размер БД |
| `pg_settings_max_connections` | Лимит max_connections |
| `pg_replication_is_replica` | Признак replica/primary |
| `pg_replication_lag_seconds` | Lag репликации |

### 6) `redis-exporter`

| Метрика | Краткий смысл |
|---|---|
| `redis_up` | Доступность Redis |
| `redis_exporter_last_scrape_error` | Ошибка последнего scrape |
| `redis_exporter_last_scrape_duration_seconds` | Длительность scrape |
| `redis_exporter_scrapes_total` | Число scrape экспортера |
| `redis_connected_clients` | Число подключенных клиентов |
| `redis_blocked_clients` | Число заблокированных клиентов |
| `redis_commands_processed_total` | Обработанные команды Redis |
| `redis_commands_total` | Команды по типам (`cmd`) |
| `redis_commands_failed_calls_total` | Ошибки выполнения команд |
| `redis_commands_rejected_calls_total` | Отклоненные команды |
| `redis_memory_used_bytes` | Используемая память |
| `redis_mem_fragmentation_ratio` | Фрагментация памяти |
| `redis_db_keys` | Число ключей по DB |
| `redis_evicted_keys_total` | Вытесненные ключи |
| `redis_expired_keys_total` | Истекшие ключи |
| `redis_keyspace_hits_total` | Cache hits |
| `redis_keyspace_misses_total` | Cache misses |
| `redis_total_reads_processed` | Read-операции |
| `redis_total_writes_processed` | Write-операции |
| `redis_instance_info` | Тех.информация об инстансе |

### 7) `node-exporter`

| Метрика | Краткий смысл |
|---|---|
| `node_exporter_build_info` | Версия node_exporter |
| `node_boot_time_seconds` | Время старта хоста |
| `node_cpu_seconds_total` | CPU-время по ядрам/режимам |
| `node_load1` | Load average за 1 минуту |
| `node_load5` | Load average за 5 минут |
| `node_load15` | Load average за 15 минут |
| `node_memory_total_bytes` | Общая память |
| `node_memory_free_bytes` | Свободная память |
| `node_memory_active_bytes` | Активная память |
| `node_filesystem_size_bytes` | Размер файловых систем |
| `node_filesystem_avail_bytes` | Доступное место файловых систем |
| `node_disk_read_bytes_total` | Прочитанные байты диска |
| `node_disk_written_bytes_total` | Записанные байты диска |
| `node_network_receive_bytes_total` | Полученные сетевые байты |
| `node_network_transmit_bytes_total` | Отправленные сетевые байты |
| `node_time_seconds` | Текущее время хоста |
| `node_uname_info` | Информация об ОС/ядре |

### 8) `blackbox-exporter`

| Метрика | Краткий смысл |
|---|---|
| `probe_success` | Успех пробы (`1`/`0`) |
| `probe_duration_seconds` | Полная длительность пробы |
| `probe_http_status_code` | HTTP-код цели |
| `probe_http_duration_seconds` | Фазовая HTTP-латентность |
| `probe_tcp_connect_duration_seconds` | Время TCP connect |

### 9) `prometheus` self metrics

| Метрика | Краткий смысл |
|---|---|
| `up{job="prometheus"}` | Доступность Prometheus |
| `prometheus_config_last_reload_successful` | Успех последнего reload конфигурации |
| `prometheus_tsdb_head_series` | Число активных series |
| `prometheus_target_scrape_pool_targets` | Число targets по jobs |
| `prometheus_rule_group_last_duration_seconds` | Длительность оценки rule group |
| `prometheus_rule_group_last_evaluation_timestamp_seconds` | Время последней оценки rule group |

## Автоматические серии

Для Counter/Histogram дополнительно публикуются:
- `*_created`;
- для Histogram: `*_bucket`, `*_sum`, `*_count`.

## Быстрые проверки

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job)\t\(.health)\t\(.scrapeUrl)"' | sort
curl -fsS http://127.0.0.1:9201/metrics | rg '^(ws_|insert_|rest_fill_|redis_publish_)'
curl -fsS http://127.0.0.1:9202/metrics | rg '^scheduler_'
curl -fsS http://127.0.0.1:9203/metrics | rg '^backtest_artifact_'
curl -fsS http://127.0.0.1:9116/metrics | rg '^clickhouse_'
curl -fsS http://127.0.0.1:9187/metrics | rg '^pg_'
curl -fsS http://127.0.0.1:9121/metrics | rg '^redis_'
curl -fsS http://127.0.0.1:9100/metrics | rg '^node_'
```
