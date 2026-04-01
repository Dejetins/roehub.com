# Market Data Metrics Reference (RU)

Статус: production monitoring baseline для native runtime на `Mac Studio`.

Документ описывает:
- полный каталог метрик, которые используются в текущем операционном контуре market-data и monitoring;
- какие метрики критичны в первую очередь;
- как интерпретировать отклонения.

Source of truth scrape-конфига:
- `infra/macos/prometheus/prometheus.prod.yml`

## Контур сервисов и jobs

В production в этом документе покрываются jobs:
- `market-data-ws-worker` (`127.0.0.1:9201`)
- `market-data-scheduler` (`127.0.0.1:9202`)
- `backtest-artifact-publisher` (`127.0.0.1:9203`)
- `clickhouse-exporter` (`127.0.0.1:9116`)
- `postgres-exporter` (`127.0.0.1:9187`)
- `redis-exporter` (`127.0.0.1:9121`)
- `node-exporter` (`127.0.0.1:9100`)
- `blackbox-http` / `blackbox-tcp` (`127.0.0.1:9115`)
- `prometheus` (`127.0.0.1:9090`)

## Самые важные метрики (операционный минимум)

| Сервис | Критичные метрики | Что значит норма |
|---|---|---|
| `market-data-ws-worker` | `ws_connected`, `ws_messages_total`, `insert_errors_total`, `redis_publish_errors_total`, `ws_closed_to_insert_done_seconds` | Соединения есть, сообщения/вставки растут, ошибок за окно нет, p95 latency стабильна |
| `market-data-scheduler` | `scheduler_job_errors_total`, `scheduler_tasks_enqueued_total`, `scheduler_rest_catchup_gap_rows_written_total` | Ошибки не растут, enqueue идет по плану, gap-progress не стоит при необходимости catchup |
| `backtest-artifact-publisher` | `backtest_artifact_publish_runs_total`, `backtest_artifact_publish_symbols_total`, `backtest_artifact_publish_blocked_total`, `backtest_artifact_publish_last_success_unixtime`, `backtest_artifact_tail_rebuild_bars_total` | Daily publish-cycle завершается success, blocked/error серии не растут, freshness обновляется после окна `03:05 Europe/Moscow`, tail bars остаются bounded, а stage/timeframe/chunk progress читается из structured logs |
| `clickhouse-exporter` | `clickhouse_exporter_scrape_success`, `clickhouse_uptime_seconds`, `clickhouse_system_event_total{event="InsertedRows"}` | `scrape_success=1`, uptime растет, вставки есть при живом потоке |
| `postgres-exporter` | `pg_up`, `pg_exporter_last_scrape_error`, `pg_stat_database_numbackends` | `pg_up=1`, scrape без ошибок, число коннектов в разумном диапазоне |
| `redis-exporter` | `redis_up`, `redis_exporter_last_scrape_error`, `redis_commands_processed_total`, `redis_memory_used_bytes` | `redis_up=1`, команды обрабатываются, память без резких аномалий |
| `node-exporter` | `node_load1`, `node_memory_free_bytes`, `node_filesystem_avail_bytes` | Нет длительной перегрузки CPU, памяти и диска достаточно |
| monitoring (`blackbox`, `prometheus`) | `probe_success`, `probe_http_status_code`, `prometheus_tsdb_head_series`, `prometheus_rule_group_last_duration_seconds` | Пробы успешны, Prometheus стабильно скрапит и считает правила |

## Полный каталог метрик

Важно:
- ниже перечислены все метрики, которые считаются обязательными в текущем monitoring baseline Roehub;
- для Counter/Histogram дополнительно автоматически публикуются серии `*_created`, `*_bucket`, `*_sum`, `*_count` (см. раздел про автоматические серии).

### 1) `market-data-ws-worker`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `ws_connected` | Gauge | - | Текущее число активных WS-соединений | Обычно > 0 |
| `ws_reconnects_total` | Counter | - | Счетчик переподключений WS | Медленный рост; всплески = сеть/endpoint |
| `ws_messages_total` | Counter | - | Счетчик полученных WS-сообщений | Стабильный рост на живом рынке |
| `ws_errors_total` | Counter | - | Ошибки обработки WS | Рост за окно = инцидент |
| `ignored_non_closed_total` | Counter | - | Отброшенные non-closed kline | Рост ожидаем |
| `insert_rows_total` | Counter | - | Записанные строки в raw | Стабильный рост |
| `insert_batches_total` | Counter | - | Число insert-батчей | Растет вместе с `insert_rows_total` |
| `insert_duration_seconds` | Histogram | - | Длительность insert-батча | p95/p99 без резких скачков |
| `insert_errors_total` | Counter | - | Ошибки вставки | Рост за окно = инцидент |
| `ws_closed_to_insert_start_seconds` | Histogram | - | Latency: closed-candle -> start insert | Контроль pre-insert части |
| `ws_closed_to_insert_done_seconds` | Histogram | - | Latency: closed-candle -> insert done | Основной end-to-end SLO |
| `ws_out_of_order_total` | Counter | - | Out-of-order свечи | Редкий рост допустим |
| `ws_duplicates_total` | Counter | - | Дубли свечей | Редкий рост допустим |
| `redis_publish_total` | Counter | - | Успешные публикации в Redis Streams | Стабильный рост |
| `redis_publish_errors_total` | Counter | - | Ошибки публикации в Redis | Рост за окно = проблема live feed |
| `redis_publish_duplicates_total` | Counter | - | Дубли/нарушение монотонности stream id | Редкий рост допустим |
| `redis_publish_duration_seconds` | Histogram | - | Длительность вызова publish | Контроль латентности live feed |
| `rest_fill_tasks_total` | Counter | - | Принятые rest fill задачи | Растет при bootstrap/gap/tail |
| `rest_fill_active` | Gauge | - | Текущее число активных fill задач | Колеблется в пределах concurrency |
| `rest_fill_errors_total` | Counter | - | Ошибки rest fill задач | Рост за окно = деградация fill |
| `rest_fill_duration_seconds` | Histogram | - | Длительность rest fill задачи | p95/p99 контролируют скорость catchup |

### 2) `market-data-scheduler`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `scheduler_job_runs_total` | Counter | `job` | Число запусков scheduler jobs | Растет по расписанию |
| `scheduler_job_errors_total` | Counter | `job` | Ошибки scheduler jobs | Рост за окно = инцидент |
| `scheduler_job_duration_seconds` | Histogram | `job` | Длительность job | Резкие скачки = деградация |
| `scheduler_tasks_planned_total` | Counter | `reason` | Запланированные задачи | Рост ожидаем при bootstrap/catchup |
| `scheduler_tasks_enqueued_total` | Counter | `reason` | Реально enqueued задачи | Обычно <= planned |
| `scheduler_startup_scan_instruments_total` | Counter | - | Сколько инструментов обработал startup scan | Рост при startup scan |
| `scheduler_rest_catchup_instruments_total` | Counter | `status` | Обработанные инструменты periodic catchup | Норма: рост по `status="ok"` |
| `scheduler_rest_catchup_tail_minutes_total` | Counter | - | Сумма tail-минут periodic catchup | Рост при tail-repair |
| `scheduler_rest_catchup_tail_rows_written_total` | Counter | - | Записанные tail-строки | Рост при хвостовом отставании |
| `scheduler_rest_catchup_gap_days_scanned_total` | Counter | - | Просканированные UTC-дни на gaps | Стабильный рост при scan |
| `scheduler_rest_catchup_gap_days_with_gaps_total` | Counter | - | Дни, где найдены gaps | После стабилизации темп падает |
| `scheduler_rest_catchup_gap_ranges_filled_total` | Counter | - | Закрытые gap-диапазоны | Рост в фазе восстановления |
| `scheduler_rest_catchup_gap_rows_written_total` | Counter | - | Записанные строки по gaps | Ключевой индикатор реального прогресса |

### 3) `backtest-artifact-publisher`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `backtest_artifact_publish_runs_total` | Counter | `status` | Итоговые daily publish-cycle runs по финальному статусу | Основной рост у `status="succeeded"` |
| `backtest_artifact_publish_duration_seconds` | Histogram | - | Длительность полного scheduler publish-cycle | Стабильный профиль в nightly окне |
| `backtest_artifact_publish_symbols_total` | Counter | `status` | Обработанные symbol roots по итоговому статусу | Основной рост у `status="planned"` и `status="succeeded"` |
| `backtest_artifact_publish_blocked_total` | Counter | `reason` | Блокировки publish-run по конечной причине | Обычно `0`, допустим редкий `lock_held` |
| `backtest_artifact_publish_last_success_unixtime` | Gauge | - | Unix timestamp последнего scheduler cycle с хотя бы одним успешным publish | После daily run обновляется и остаётся внутри freshness window |
| `backtest_artifact_tail_rebuild_bars_total` | Counter | `stage` | Сколько баров реально переписано в bounded tail по stage | Рост должен быть bounded; резкий скачок = массовый full rebuild fallback |

#### Structured progress fields for `backtest-artifact-publisher`

Эти поля не являются отдельными Prometheus-метриками. Они обязательны в structured logs и нужны
для интерпретации длинного bootstrap/full rebuild:

| Field | Где смотреть | Что означает |
|---|---|---|
| `artifact_precompute_stage_started` | structured log event | Старт нового pipeline stage |
| `artifact_precompute_stage_finished` | structured log event | Успешное завершение stage |
| `current_timeframe` | log field | Какой timeframe session сейчас открыт |
| `current_indicator` | log field | Какой signal target materialize'ится внутри session |
| `chunk_index` | log field | Номер текущего chunk job |
| `chunk_jobs_total` | log field | Сколько chunk jobs всего у текущего `(indicator_id, timeframe)` |
| `reused_prefix_bars` | log field / diagnostics | Сколько баров stage переиспользовал без переписывания |
| `rewritten_tail_bars` | log field / diagnostics | Сколько баров stage реально переписал |

Интерпретация:

- один `current_timeframe` одновременно = норма для `timeframe-scoped execution`;
- рост `chunk_index` при стабильном `current_timeframe` = healthy progress;
- отсутствие chunk progress при росте memory pressure = сигнал, что executor drift'нул к giant
  in-memory behavior;
- для daily rebuild ожидаемо `reused_prefix_bars >> rewritten_tail_bars`;
- для bootstrap допустим `reused_prefix_bars = 0` и большой `rewritten_tail_bars`.

### 4) `clickhouse-exporter`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `clickhouse_exporter_scrape_duration_seconds` | Gauge | - | Длительность последнего scrape ClickHouse | Без резких пиков |
| `clickhouse_exporter_scrape_success` | Gauge | - | Успех последнего scrape (`1`/`0`) | Должно быть `1` |
| `clickhouse_uptime_seconds` | Gauge | - | Uptime процесса ClickHouse | Должен расти |
| `clickhouse_system_metric_value` | Gauge | `metric` | Текущее значение из `system.metrics` | Зависит от `metric` |
| `clickhouse_system_event_total` | Counter | `event` | Кумулятивные счетчики из `system.events` | Ключевые события должны расти |

`clickhouse_system_metric_value{metric=...}` в baseline:
- `BackgroundMergesAndMutationsPoolTask`
- `HTTPConnection`
- `Query`
- `TCPConnection`

`clickhouse_system_event_total{event=...}` в baseline:
- `InsertedBytes`
- `InsertedRows`
- `Query`
- `SelectQuery`
- `SelectedBytes`
- `SelectedRows`

### 5) `postgres-exporter`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `pg_up` | Gauge | - | Доступность PostgreSQL для экспортера | Должно быть `1` |
| `pg_exporter_last_scrape_error` | Gauge | - | Ошибка последнего scrape (`1`/`0`) | Должно быть `0` |
| `pg_exporter_last_scrape_duration_seconds` | Gauge | - | Длительность последнего scrape | Без резких пиков |
| `pg_exporter_scrapes_total` | Counter | - | Число scrape экспортера | Растет |
| `pg_scrape_collector_success` | Gauge | `collector` | Успешность коллектора | Должно быть `1` |
| `pg_scrape_collector_duration_seconds` | Gauge | `collector` | Длительность коллектора | Контроль деградации |
| `pg_stat_database_xact_commit` | Counter | `datid`,`datname` | Число commit транзакций | Растет при нагрузке |
| `pg_stat_database_xact_rollback` | Counter | `datid`,`datname` | Число rollback транзакций | Рост анализируется в контексте ошибок |
| `pg_stat_database_numbackends` | Gauge | `datid`,`datname` | Текущее число backend-соединений | Не должно упираться в лимит |
| `pg_stat_activity_count` | Gauge | `datname`,`state`,... | Активность сессий по состояниям | Нет аномального роста `active/idle in transaction` |
| `pg_locks_count` | Gauge | `datname`,`mode` | Число locks по режимам | Пики + latency = расследование |
| `pg_database_size_bytes` | Gauge | `datname` | Размер БД | Рост контролируется capacity-планом |
| `pg_settings_max_connections` | Gauge | `server` | Лимит max_connections | Сопоставлять с `numbackends` |
| `pg_replication_is_replica` | Gauge | - | Признак replica (`1`) или primary (`0`) | Для текущего контура ожидается `0` |
| `pg_replication_lag_seconds` | Gauge | - | Lag репликации в секундах | Для single-node/primary обычно `0` |

### 6) `redis-exporter`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `redis_up` | Gauge | - | Доступность Redis для экспортера | Должно быть `1` |
| `redis_exporter_last_scrape_error` | Gauge | `err` | Ошибка последнего scrape (`1`/`0`) | Должно быть `0` |
| `redis_exporter_last_scrape_duration_seconds` | Gauge | - | Длительность последнего scrape | Без резких пиков |
| `redis_exporter_scrapes_total` | Counter | - | Число scrape экспортера | Растет |
| `redis_connected_clients` | Gauge | - | Подключенные клиенты | Контроль пиков |
| `redis_blocked_clients` | Gauge | - | Заблокированные клиенты | Длительный рост = деградация |
| `redis_commands_processed_total` | Counter | - | Обработанные команды | Растет при живой нагрузке |
| `redis_commands_total` | Counter | `cmd` | Число команд по типу | Нагрузка по командам |
| `redis_commands_failed_calls_total` | Counter | `cmd` | Ошибки выполнения команд | Рост = ошибки/таймауты |
| `redis_commands_rejected_calls_total` | Counter | `cmd` | Отклоненные команды | Рост = лимиты/проблемы сервера |
| `redis_memory_used_bytes` | Gauge | - | Используемая память Redis | Контроль capacity |
| `redis_mem_fragmentation_ratio` | Gauge | - | Фрагментация памяти | Стабильно высокий уровень = tuning |
| `redis_db_keys` | Gauge | `db` | Число ключей в DB | Контроль роста данных |
| `redis_evicted_keys_total` | Counter | - | Вытесненные ключи | Рост = pressure по памяти |
| `redis_expired_keys_total` | Counter | - | Истекшие ключи | Рост ожидаем для TTL-нагрузки |
| `redis_keyspace_hits_total` | Counter | - | Cache hits | Используется вместе с misses |
| `redis_keyspace_misses_total` | Counter | - | Cache misses | Рост без hits = низкая эффективность |
| `redis_total_reads_processed` | Counter | - | Количество read-операций | Нагрузка чтения |
| `redis_total_writes_processed` | Counter | - | Количество write-операций | Нагрузка записи |
| `redis_instance_info` | Gauge | `redis_version`,`role`,... | Техническая информация об инстансе | Для валидации роли/версии |

### 7) `node-exporter`

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `node_exporter_build_info` | Gauge | `version`,... | Версия node_exporter | Контроль версии |
| `node_boot_time_seconds` | Gauge | - | Время старта хоста | Используется для детекта reboot |
| `node_cpu_seconds_total` | Counter | `cpu`,`mode` | CPU-время по ядрам/режимам | Высокий non-idle = нагрузка |
| `node_load1` | Gauge | - | Load average за 1 минуту | Рост > доступных CPU = перегрузка |
| `node_load5` | Gauge | - | Load average за 5 минут | Тренд средней нагрузки |
| `node_load15` | Gauge | - | Load average за 15 минут | Долгий тренд нагрузки |
| `node_memory_total_bytes` | Gauge | - | Общая память | База для расчетов использования |
| `node_memory_free_bytes` | Gauge | - | Свободная память | Длительное падение = pressure |
| `node_memory_active_bytes` | Gauge | - | Активно используемая память | Контекст pressure |
| `node_filesystem_size_bytes` | Gauge | `mountpoint`,... | Размер ФС | Capacity |
| `node_filesystem_avail_bytes` | Gauge | `mountpoint`,... | Доступное место ФС | Критично для data/log storage |
| `node_disk_read_bytes_total` | Counter | `device` | Прочитанные байты диска | IO-профиль чтения |
| `node_disk_written_bytes_total` | Counter | `device` | Записанные байты диска | IO-профиль записи |
| `node_network_receive_bytes_total` | Counter | `device` | Полученные байты сети | Сетевой трафик ingress |
| `node_network_transmit_bytes_total` | Counter | `device` | Отправленные байты сети | Сетевой трафик egress |
| `node_time_seconds` | Gauge | - | Текущее время хоста | Тех.проверка времени |
| `node_uname_info` | Gauge | `sysname`,`release`,... | Информация об ОС/ядре | Диагностика окружения |

### 8) `blackbox-exporter` (`blackbox-http`, `blackbox-tcp`)

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `probe_success` | Gauge | `instance`,`job` | Результат пробы (`1`/`0`) | Должно быть `1` |
| `probe_duration_seconds` | Gauge | `instance`,`job` | Полная длительность пробы | Без резких пиков |
| `probe_http_status_code` | Gauge | `instance`,`job` | HTTP-код целевого endpoint | Должен соответствовать ожиданию |
| `probe_http_duration_seconds` | Gauge | `phase`,`instance`,`job` | HTTP latency по фазам | Рост фаз = деградация сети/цели |
| `probe_tcp_connect_duration_seconds` | Gauge | `instance`,`job` | Время TCP connect | Рост = сеть/порт/нагрузка |

### 9) `prometheus` (self metrics)

| Метрика | Тип | Labels | Описание | Норма/сигнал |
|---|---|---|---|---|
| `up{job="prometheus"}` | Gauge | `job`,`instance` | Доступность самого Prometheus | Должно быть `1` |
| `prometheus_config_last_reload_successful` | Gauge | - | Успешность последнего reload конфига | Должно быть `1` |
| `prometheus_tsdb_head_series` | Gauge | - | Число активных series в head | Резкие скачки = cardinality-риск |
| `prometheus_target_scrape_pool_targets` | Gauge | `scrape_job` | Число targets по job | Соответствует конфигу |
| `prometheus_rule_group_last_duration_seconds` | Gauge | `rule_group` | Длительность последнего rule eval | Не должно приближаться к interval |
| `prometheus_rule_group_last_evaluation_timestamp_seconds` | Gauge | `rule_group` | Время последней оценки rules | Должно обновляться регулярно |

## Автоматические серии Prometheus

Для полноты интерпретации:
- у Counter обычно есть основная серия и `*_created`;
- у Histogram есть `*_bucket`, `*_sum`, `*_count`, `*_created`;
- поэтому при проверке "всех" серий на endpoint метрик число строк всегда больше числа логических метрик в таблицах выше.

## Справочник labels

`scheduler job`:
- `sync_whitelist`
- `enrich`
- `startup_scan`
- `rest_insurance_catchup`

`scheduler reason`:
- `scheduler_bootstrap`
- `historical_backfill`
- `scheduler_tail`

`scheduler status`:
- `ok`
- `failed`
- `skipped_no_seed`

`backtest_artifact_publish_runs_total{status=...}`:
- `succeeded`
- `inactive_slot_pinned`
- `validation_failed`
- `lock_held`
- `unexpected_error`

`backtest_artifact_publish_symbols_total{status=...}`:
- `planned`
- `succeeded`
- `inactive_slot_pinned`
- `validation_failed`
- `unexpected_error`

`backtest_artifact_publish_blocked_total{reason=...}`:
- `lock_held`
- `inactive_slot_pinned`
- `validation_failed`

`backtest_artifact_tail_rebuild_bars_total{stage=...}`:
- `prices`
- `mappings`
- `signals`
- `hit_times`

`clickhouse_system_metric_value{metric=...}`:
- `BackgroundMergesAndMutationsPoolTask`
- `HTTPConnection`
- `Query`
- `TCPConnection`

`clickhouse_system_event_total{event=...}`:
- `InsertedBytes`
- `InsertedRows`
- `Query`
- `SelectQuery`
- `SelectedBytes`
- `SelectedRows`

## Быстрые проверки

Проверка health всех jobs:

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job)\t\(.health)\t\(.scrapeUrl)"' | sort
```

Проверка presence всех baseline метрик по сервисам:

```bash
curl -fsS http://127.0.0.1:9201/metrics | rg '^(ws_|insert_|rest_fill_|redis_publish_)'
curl -fsS http://127.0.0.1:9202/metrics | rg '^scheduler_'
curl -fsS http://127.0.0.1:9203/metrics | rg '^backtest_artifact_'
curl -fsS http://127.0.0.1:9116/metrics | rg '^clickhouse_'
curl -fsS http://127.0.0.1:9187/metrics | rg '^pg_'
curl -fsS http://127.0.0.1:9121/metrics | rg '^redis_'
curl -fsS http://127.0.0.1:9100/metrics | rg '^node_'
```

## PromQL для дежурного мониторинга

Состояние jobs:

```promql
sum by (job) (
  up{job=~"prometheus|node-exporter|postgres-exporter|redis-exporter|clickhouse-exporter|market-data-ws-worker|market-data-scheduler"}
)
```

Состояние blackbox probes:

```promql
sum by (job, instance) (probe_success{job=~"blackbox-http|blackbox-tcp"})
```

Worker p95 closed->insert done:

```promql
histogram_quantile(
  0.95,
  sum(rate(ws_closed_to_insert_done_seconds_bucket{job="market-data-ws-worker"}[5m])) by (le)
)
```

Ошибки worker за 15 минут:

```promql
increase(ws_errors_total{job="market-data-ws-worker"}[15m])
+ increase(insert_errors_total{job="market-data-ws-worker"}[15m])
+ increase(redis_publish_errors_total{job="market-data-ws-worker"}[15m])
```

Ошибки scheduler за 15 минут:

```promql
increase(scheduler_job_errors_total{job="market-data-scheduler"}[15m])
```

Состояние ClickHouse exporter:

```promql
clickhouse_exporter_scrape_success{job="clickhouse-exporter"}
```

## Связанные документы

- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/runbooks/mac-studio-native-backend-operations.md`
- `infra/macos/prometheus/prometheus.prod.yml`
